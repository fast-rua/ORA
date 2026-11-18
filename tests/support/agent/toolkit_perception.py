"""One public publication served by the toolkit, on each backend's default surface.

``toolkit_host`` publishes the pixels and returns a toolkit host over them, so a test calls a tool and checks what
the model would receive (``Calls``). The plain constructor needs a verified robot geometry binding, so ``Stand-in``
below says exactly what is lifted onto the host. The host runs the synchronous dispatch: the perception tools are
read-only there, and it installs every tool through ``core.invoke`` with its closed result-schema check.
"""

from __future__ import annotations

import dataclasses
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from ora.agent.context.observation.contact_points import project_policy_public_facts
from ora.agent.evidence.policy_state import REQUIRED_ABSENT, PolicyStateStore
from ora.agent.model_services.clients import SAM3_MODEL_IDENTITY
from ora.agent.toolkit.interface.types import canonical, digest
from ora.agent.toolkit.registry import load_tools
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.agent.toolkit.host.runtime.dispatch import install
from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA
from tests.support.agent.surface_catalog import surface_of
from tests.support.agent.toolkit_recording import proposal

ROOT = Path(__file__).resolve().parents[3]
MANUAL = ROOT / "tests/data/prompts/tool-schema-delivery-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}

#: Backend -> default surface and its public cameras, first camera the tilted fixed one.
SURFACES = {
    "libero": ("libero_placement_direct_overhead_v1", ("agentview", "wrist")),
    "robocasa": ("robocasa_mobile_direct_v1", ("agentview", "wrist")),
    "robolab": ("robolab_direct_v1", ("agentview", "wrist")),
    "robotwin": ("robotwin_toolchain_b2_closed_noservo_v2",
                 ("head_camera", "front_camera", "left_camera", "right_camera")),
    "yam": ("yam_direct_v1", ("overhead", "front", "wrist_left", "wrist_right")),
}
WIDTH, HEIGHT = 64, 48
#: A box on the table, in pixels, as a mask service would answer it.
BOX = (24, 20, 44, 34)
TABLE_Z_M, BOX_Z_M = 0.95, 1.05
TILTS_DEG = (40.0, 0.0, 12.0, -12.0)


class Grounder:
    """The perception service: one mask for a text query, a mask for each point query."""

    backend_id = "sam3"
    backend_version = "test"
    expected_binding = {"fixture": "sam"}

    def configuration(self):
        return dict(endpoint="http://fake.invalid", test=True)

    @staticmethod
    def _mask(shape):
        mask = np.zeros(shape[:2], bool)
        mask[BOX[1]:BOX[3], BOX[0]:BOX[2]] = True
        return mask

    def ground(self, *, rgb_u8, text, max_candidates):
        return ([dict(rank=0, source_index=0, backend_score=.9, bbox_xyxy=list(BOX), mask=self._mask(rgb_u8.shape))],
                dict(request_id="fixture", returned_candidate_count=1, _service_identity={"test": True},
                     model=SAM3_MODEL_IDENTITY,
                     latency_ms=dict(roundtrip=1., service_queue=0., service_inference=1., service_total=1.)))

    def segment_points(self, *, rgb_u8, depth_m, points_uv):
        rows = [dict(query_index=index, source_index=0, point_uv=list(point), backend_score=.9,
                     bbox_xyxy=list(BOX), mask=self._mask(rgb_u8.shape)) for index, point in enumerate(points_uv)]
        queries = [dict(query_index=index, point_uv=list(point), raw_candidate_count=1, error_code=None)
                   for index, point in enumerate(points_uv)]
        return rows, dict(query_results=queries, image_encodings=1)


def camera_pose(tilt_deg, *, offset_m=(0.0, 0.0, 2.0)):
    """cam2world of an OpenCV camera looking down, tilted about the world x axis."""
    looking_down = Rotation.from_euler("x", 180, degrees=True)
    pose = np.eye(4)
    pose[:3, :3] = (Rotation.from_euler("x", tilt_deg, degrees=True) * looking_down).as_matrix()
    pose[:3, 3] = offset_m
    return pose


def rendered_camera(cam2world, *, focal=60.0):
    """RGB and depth of a table with a box on it, seen by a pinhole at ``cam2world``."""
    columns, rows = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    rays = np.stack([(columns - WIDTH / 2) / focal, (rows - HEIGHT / 2) / focal, np.ones_like(columns)], axis=-1)
    rise = (rays @ cam2world[:3, :3].T)[..., 2]
    heights = np.full(rays.shape[:2], TABLE_Z_M)
    heights[BOX[1]:BOX[3], BOX[0]:BOX[2]] = BOX_Z_M
    depth = ((heights - cam2world[2, 3]) / rise).astype(np.float32)
    rgb = np.stack([(columns * 4) % 256, (rows * 5) % 256, heights * 100], axis=-1).astype(np.uint8)
    intrinsic = [[focal, 0.0, WIDTH / 2], [0.0, focal, HEIGHT / 2], [0.0, 0.0, 1.0]]
    return rgb, depth, intrinsic


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return dict(path=path.parent.name + "/" + path.name, sha256=digest(data), bytes=len(data))


def _camera(root, name, tilt_deg, *, step, calibrated=True, depth=True):
    from io import BytesIO

    cam2world = camera_pose(tilt_deg, offset_m=(0.02 * step, 0.0, 2.0))
    rgb, metres, intrinsic = rendered_camera(cam2world)
    png, npy = BytesIO(), BytesIO()
    Image.fromarray(rgb, mode="RGB").save(png, format="PNG")
    np.save(npy, metres, allow_pickle=False)
    calibration = dict(intrinsic_cv=intrinsic, valid_depth_range_m=dict(near=.01, far=50.))
    if calibrated:
        calibration.update(extrinsic_cv=np.linalg.inv(cam2world).tolist(), cam2world_cv=cam2world.tolist())
    files = dict(rgb_lossless_png=_write(root / name / "rgb.png", png.getvalue()),
                 calibration=_write(root / name / "calibration.json", canonical(calibration)))
    if depth:
        files["depth_numpy_m_f32"] = _write(root / name / "depth-m.npy", npy.getvalue())
    return dict(width=WIDTH, height=HEIGHT, files=files)


def _proprioception(backend, step, *, both_arms):
    """Each backend's published end effectors, with the facts its projection reads."""
    from tests.support.agent.public_robot_facts import fixture

    if backend == "yam":
        return {"end_effectors": {arm: {"pose_xyz_xyzw": [x, 0.1 + .01 * step, 1.1, 0., 0., 0., 1.],
                                        "pose_frame": "yam_linear_4310_tcp", "gripper": .5, "jaw_separation_m": .04,
                                        "gripper_finger_qpos": [.02], "gripper_command_target": "open",
                                        "arm_joint_qpos": [0.] * 6}
                                  for arm, x in (("left", -.2), ("right", .2))}, "single_arm": False}
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        _, observation, _ = fixture(Path(scratch), backend)
    proprioception = json.loads(json.dumps(observation["proprioception"]))
    effectors = proprioception["end_effectors"]
    for index, effector in enumerate(effectors.values()):
        effector["pose_xyz_xyzw"] = [0.1 - .2 * index, 0.1 + .01 * step, 1.1, 1., 0., 0., 0.]
    if both_arms and "left" not in effectors:
        effectors["left"] = json.loads(json.dumps(effectors["right"]))
    return proprioception


def _publish_step(run, backend, step, *, uncalibrated, without_depth, absent, both_arms):
    profile, cameras = SURFACES[backend]
    surface = surface_of(profile)
    observation_id = f"{backend}:observation-{step:04d}"
    root = run / "policy-observations" / observation_id
    records = {name: _camera(root, name, TILTS_DEG[index % len(TILTS_DEG)], step=step,
                             calibrated=name not in uncalibrated, depth=name not in without_depth)
               for index, name in enumerate(cameras) if name not in absent}
    proprioception = _proprioception(backend, step, both_arms=both_arms)
    (root / "proprioception.json").write_bytes(canonical(proprioception))
    manifest = dict(schema="ora-observation.v1", observation_id=observation_id, sequence=step,
                    captured_at_unix_s=123., round_index=0, main_camera=cameras[0], cameras=records,
                    proprioception=proprioception, forbidden_modalities_absent=sorted(REQUIRED_ABSENT))
    descriptor = (surface.view_config or {}).get("public_geometry_descriptor")
    if descriptor is not None:
        manifest["robot_geometry_verification"] = dict(schema=GEOMETRY_VERIFICATION_SCHEMA, passed=True,
            descriptor_sha256=descriptor["sha256"], scope="loaded_robot_geometry_only")
    (root / "observation.json").write_bytes(canonical(manifest))
    return manifest, root


def _publish_run(run, backend, **publication):
    run.mkdir(parents=True)
    store = PolicyStateStore(run, "run", "a" * 64)
    state = None
    for step in (0, 1):
        manifest, root = _publish_step(run, backend, step, **publication)
        state = store.publish(observation=manifest, observation_root=root, cause="after_action" if step else "reset",
                              task_instruction="fixture task", previous_public_action=None,
                              remaining_budget={"robot_decisions": 9, "model_calls": 19})
    pointer = dict(open=True, observation_id=manifest["observation_id"],
                   observation_sha256=digest((root / "observation.json").read_bytes()),
                   source_root=str(root), generation=0, policy_state=dict(step=1, state_sha256=state["state_sha256"]))
    control = run / "control"
    control.mkdir()
    (control / "current-observation.json").write_bytes(canonical(pointer))
    event = dict(type="fresh_observation_published", seq=1, run_id="run",
                 payload=dict(observation_id=manifest["observation_id"]))
    (run / "agent-events.jsonl").write_bytes(canonical(event) + b"\n")
    return control, store.root


def _host(run, backend, **publication):
    control, state_root = _publish_run(run, backend, **publication)
    surface = surface_of(SURFACES[backend][0])
    host = ManipulationToolRuntime(
        control_root=control, manual_path=MANUAL, event_path=run / "events.jsonl", workspace_bounds_m=BOUNDS,
        state_root=None, tool_surface_profile="native_atomic_causal_v1", controlled_end_effector="right",
        segmentation_backend=Grounder(), grounding_backend=Grounder())
    host.state_root = state_root
    host.surface, host.view_config = surface, dict(surface.view_config)
    host.allowed_tools = tuple(item["name"] for item in surface.definitions)
    host.public_facts_projector = project_policy_public_facts
    return host


@contextmanager
def _read_only(name, context, requested):
    assert requested == 0
    yield SimpleNamespace(recovered=False)


def _stand_in(monkeypatch):
    """What the host lane's units give a backend-neutral dispatch; this file names each one.

    Rows 1 to 3 of the plan (grasp compilers and motion configuration need an X5 descriptor)
    and row 14 (grounding cameras are RoboDojo's) are replaced here, so these tests see only
    perception.
    """
    from ora.agent.toolkit.host.runtime import inputs as toolkit_inputs, tool_data as toolkit_tool_data
    from ora.agent.toolkit.host import motion as toolkit_motion

    configuration = toolkit_inputs.configuration
    monkeypatch.setattr(toolkit_tool_data, "grasp_configuration", lambda host: {})
    monkeypatch.setattr(toolkit_motion, "configuration",
                        lambda host: dict(workspace_bounds_m=dict(host.workspace_bounds_m)))
    monkeypatch.setattr(toolkit_inputs, "configuration", lambda host: {
        **configuration(host), "grounding_cameras": sorted(host._physical_policy_cameras())})


def toolkit_on(host, monkeypatch):
    """Move ``host`` to the toolkit tools on its surface; a tool the toolset lacks is dropped."""
    _stand_in(monkeypatch)
    tools = load_tools()
    surface = host.surface
    definitions = tuple(item for item in surface.definitions if item["name"] in tools)
    host.surface = dataclasses.replace(
        surface, definitions=definitions,
        capabilities=(*surface.capabilities, "toolkit_only_v1", "toolkit_synchronous_execution_v1"))
    host.allowed_tools = tuple(item["name"] for item in definitions)
    host.toolkit_executor = SimpleNamespace(invocation=_read_only)
    install(host)
    return host


def toolkit_host(tmp_path, backend, monkeypatch, *, uncalibrated=(), without_depth=(), absent=(), both_arms=False):
    """The toolkit host of one backend over one publication.

    ``uncalibrated`` cameras publish no extrinsic (a YAM camera with no calibration on file),
    ``without_depth`` no depth file, ``absent`` nothing. ``both_arms`` gives a single-arm backend
    a left arm too: toolkit ``observe`` needs one until rows 7 and 19 of the plan land, and the
    public-facts projection refuses one, so such a call goes through ``without_public_facts``.
    """
    publication = dict(uncalibrated=uncalibrated, without_depth=without_depth, absent=absent, both_arms=both_arms)
    return toolkit_on(_host(tmp_path / "new", backend, **publication), monkeypatch)




def without_public_facts(*hosts):
    """Drop the public facts and the observe schema that requires them, for a publication their projection refuses."""
    for host in hosts:
        host.surface = dataclasses.replace(
            host.surface, result_schemas={k: v for k, v in host.surface.result_schemas.items() if k != "observe"},
            capabilities=tuple(name for name in host.surface.capabilities if name != "current_public_robot_facts_v1"))


def without_input_schema(*hosts):
    """Stop ``core.invoke`` refusing by input schema, so a call reaches each stack's own refusals."""
    from ora.agent.toolkit.host.runtime.core import ADVISORY_CAPABILITY

    for host in hosts:
        host.surface = dataclasses.replace(host.surface, capabilities=tuple(
            name for name in host.surface.capabilities if name != ADVISORY_CAPABILITY))


def _outcome(call):
    from ora.agent.pi.tool_rpc import _tool_error

    try:
        return dict(result=call())
    except Exception as error:  # the model's view of a refusal, whichever stack raised it
        return dict(error=_tool_error(error))


class Calls:
    """Call a tool on the toolkit host, each call a proposal the dispatch can index, and check what the model gets
    against ``check`` (``toolkit_goldens.Goldens.checker``)."""

    def __init__(self, new, check):
        self.new, self.check, self.index = new, check, 0

    def call(self, name, arguments, *, checked=True):
        parent = proposal(self.new, name, arguments, index=self.index)
        self.index += 1
        actual = _outcome(lambda: self.new.invoke_with_context(name, arguments, call_context=parent))
        if checked:
            self.check(f"call-{self.index - 1}", actual, self.new.control_root.parent)
        return actual

    def overlays(self, host):
        """The digest of each overlay file ``host`` wrote, by its path under the artifact root."""
        return {path.relative_to(host.artifact_root).as_posix(): digest(path.read_bytes())
                for path in host.artifact_root.glob("*/overlays/*.png")}
