"""One tool call on a stand-in host on the toolkit, and what it leaves: what the model gets back and the action.

A stand-in host is the real ``ManipulationToolRuntime`` on a registered ToolSurface, over one fabricated public RGB-D
observation, with fake perception services and the toy environment bridge. Its proprioception has the keys its backend's
session publishes. ``tests/ora/agent/toolkit/test_stack_parity.py`` says what this can and cannot see.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import httpx
import numpy as np
from PIL import Image

from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.agent.context.observation.contact_points import project_policy_public_facts
from ora.agent.evidence.policy_state import REQUIRED_ABSENT, PolicyStateStore
from ora.agent.session_core.tool_arguments import _tool_error
from ora.agent.toolkit.interface.types import canonical, digest
from ora.agent.toolkit.host.grasp.contact_graspnet_client import ContactGraspNetApiProposer
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.agent.toolkit.host.runtime.base import RobotToolRuntime, manual_sections
from ora.agent.error_classes import ToolError
from ora.contracts.action_plan import PLAN_SCHEMA, is_request, segment
from ora.contracts.gripper_feedback import gripper_feedback
from ora.contracts.reach_check import REQUEST_SCHEMA as REACH_CHECK
from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA
from ora.environment.controllers.robodojo_arx import public_feedback_contract
from tests.support.agent.kinematics_bridge import Bridge
from tests.support.agent.surface_catalog import catalog_grants, handlers_only
from tests.support.agent.toolkit_recording import proposal
from tests.support.agent.toolkit_tools import Grounder, grasp_service, install

ROOT = Path(__file__).resolve().parents[3]
MANUAL = ROOT / "tests/data/prompts/tool-schema-delivery-v1.md"
BOUNDS = {"x": [-0.75, 0.75], "y": [-0.65, 0.55], "z": [0.72, 1.4]}
OBSERVATION = "stand-in-observation-0"
SIZE = 128
#: One block on a table, the same in every camera: its rows and columns, and the depth of each.
BLOCK = (50, 78, 50, 78)
TABLE_DEPTH_M, BLOCK_DEPTH_M = 0.85, 0.80
CAMERA_TO_WORLD = [[1.0, 0.0, 0.0, 0.1], [0.0, -1.0, 0.0, -0.1], [0.0, 0.0, -1.0, 1.6], [0.0, 0.0, 0.0, 1.0]]
POSE = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
#: Keys whose values are the clock's, not the tool's.
VOLATILE = frozenset({"latency_ms"})
#: Keys holding image bytes, compared by digest.
IMAGES = frozenset({"image_base64", "images_base64"})

@dataclass(frozen=True)
class Surface:
    """What a stand-in needs of one surface: its cameras, the proprioception its backend publishes, and how its moving
    tools spell a target."""

    profile: str
    cameras: tuple[str, ...]
    proprioception: Mapping[str, Any]
    flat: bool = False
    #: The argument naming a target position on an atomic surface, whose moves take no ``position_mode``.
    atomic_position: str | None = None
    #: RoboDojo's synchronous surface without synchronous execution, so its calls propose their actions (a path no
    #: run takes), with the controller contract its observations publish.
    handlers_only: bool = False
    #: The five-tool surface, which runs on ``RobotToolRuntime``.
    robot: bool = False
    #: What the worker's read-only bridge answers: None is the toy, which answers every request; otherwise a dry run's
    #: plan and the requests of the schemas named here, and nothing else.
    bridge: tuple[str, ...] | None = None
    #: Whether the worker answers a motion probe.
    probe: bool = True
    #: The probe is RoboTwin's own (``RoboTwinWorkerProbe``) instead of the toy's.
    robotwin_probe: bool = False


def _single_arm(**effector: Any) -> dict[str, Any]:
    return {"end_effectors": {"right": {"pose_xyz_xyzw": list(POSE), "gripper": 1.0, "gripper_command": 1.0,
                                        **effector}}, "single_arm": True}


def _two_arms(**effector: Any) -> dict[str, Any]:
    return {"end_effectors": {arm: {"pose_xyz_xyzw": list(POSE), **effector} for arm in ("left", "right")}}


_GRIP_SITE = _single_arm(pose_frame="robosuite_grip_site", gripper_finger_qpos=[0.04, -0.04],
                         gripper_command_target="unknown")
_ROBOTWIN = {**_two_arms(controlled_point_semantics="parallel_jaw_grip_center_tcp", gripper_command_target="open",
                         gripper_opening_m=0.09, gripper_state={
                             "commanded_opening_normalized": 1.0, "drive_target_opening_normalized": 1.0,
                             "gripper_opening_m": 0.09, "measured_opening_normalized": 1.0,
                             "measured_primary_joint_position": 0.045,
                             "measurement_source": "articulation_qpos_primary_gripper_joint",
                             "opening_m_semantics": "jaw_separation_m_sum_of_measured_prismatic_finger_travel",
                             "opening_semantics": "normalized_closed_0_open_1_not_aperture_m"}),
             "joints": {"left_arm": [0.0] * 6, "right_arm": [0.0] * 6},
             "gripper_evidence_semantics": "command_target_and_measured_jaw_separation_m"}
_ROBODOJO = {**_two_arms(pose_frame="robodojo_episode_local_ee_link", gripper=1.0, gripper_finger_qpos=[0.044, 0.044],
                         gripper_command=1.0, gripper_command_target="open", arm_joint_qpos=[0.0] * 6,
                         gripper_feedback=gripper_feedback(
                             requested_opening_normalized=1.0, drive_target_opening_normalized=1.0,
                             measured_opening_normalized=1.0, effective_target_opening_normalized=1.0,
                             tolerance_normalized=0.03, drive_target_source="resolved_sdk_joint_position_target",
                             measurement_source="primary_finger_encoder",
                             pose_hold_semantics="preserve_last_applied_drive_target")), "single_arm": False}
_YAM = {**_two_arms(pose_frame="yam_linear_4310_tcp", gripper=1.0, jaw_separation_m=0.08, gripper_finger_qpos=[0.04],
                    gripper_command_target="open", arm_joint_qpos=[0.0] * 6),
        "single_arm": False, "position_frame": "yam_station_world"}
_AGENTVIEW = ("agentview", "wrist")
SURFACES = {
    "libero": Surface("libero_placement_direct_overhead_v1", _AGENTVIEW, _GRIP_SITE, flat=True),
    "robotwin": Surface("robotwin_toolchain_b2_closed_noservo_v2",
                        ("head_camera", "front_camera", "left_camera", "right_camera"), _ROBOTWIN, robotwin_probe=True),
    "robocasa": Surface("robocasa_mobile_direct_v1", _AGENTVIEW, {
        **_GRIP_SITE, "base": {"pose_xy_yaw": [0.0, 0.0, 0.0], "frame": "robocasa_world", "torso_height_m": 0.3}},
        flat=True, bridge=()),
    "robolab": Surface("robolab_direct_v1", _AGENTVIEW, _single_arm(
        pose_frame="robotiq_2f85_tcp", jaw_separation_m=0.085, gripper_finger_qpos=[0.0],
        gripper_command_target="open"), flat=True, bridge=(REACH_CHECK,), probe=False),
    "yam": Surface("yam_direct_v1", ("overhead", "front", "wrist_left", "wrist_right"), _YAM, bridge=(),
                   probe=False),
    "synthetic": Surface("native_atomic_causal_v1", _AGENTVIEW, _GRIP_SITE, flat=True, atomic_position="xyz_m"),
    "local": Surface("local_atomic_cpu_v1", _AGENTVIEW, _GRIP_SITE, flat=True, atomic_position="target_xyz_world_m"),
    "robodojo": Surface("robodojo_placement_toolkit_v5", ("head", "left_wrist", "right_wrist"), _ROBODOJO,
                        handlers_only=True),
    "robot": Surface("robot_runtime_v2", _AGENTVIEW, _two_arms(), robot=True),
}
#: The manual a Synthetic run binds, and its first section, which a row asks for by name.
ROBOT_MANUAL = ROOT / "ora/policy/embodiments/aloha-agilex/manual-v1.md"
MANUAL_SECTION = manual_sections(ROBOT_MANUAL)[0]
#: A later argument naming what the row's earlier call returned.
LAST_PREVIEW = "$preview_id"


@dataclass(frozen=True)
class Call:
    """One tool call. ``needs`` is what the host prepares first: ``target``, a grounded block whose ``target_ref`` joins
    the arguments, or ``candidate``, that and a grasp plan whose first ``candidate_id`` does.

    ``first`` are earlier calls of one episode and ``task_check`` is the record the Supervisor wrote at this boundary.
    The row checks only this last call."""

    tool: str
    arguments: Mapping[str, Any]
    needs: str | None = None
    first: tuple["Call", ...] = ()
    task_check: Mapping[str, Any] | None = None


class Environment(Bridge):
    """The toy bridge, its answers within what the surfaces' closed result schemas admit, and a plan for a dry run: a
    base move is the line its servo is commanded along, unplanned, as RoboCasa answers it."""

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if is_request(request) and request["action"]["action"] == "move_base":
            base = request["action"]["arguments"]["target"]
            return {"schema": PLAN_SCHEMA, "status": "planned", "route": "base_servo", "segments": [
                segment("base", None, "stand-in", poses=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                                         [base["x_m"], base["y_m"], 0.0, 0.0, 0.0, 0.0, 1.0]],
                        planned=False, yaw_rad=base["yaw_rad"])]}
        if is_request(request):
            arguments = request["action"]["arguments"]
            target = arguments.get("targets", {}).get("right", arguments)
            pose = [*target["position_m"], *target["orientation_xyzw"]]
            return {"schema": PLAN_SCHEMA, "status": "planned", "route": "stand-in", "segments": [
                segment("arm", "right", "stand-in", poses=[list(POSE), pose], planned=True, native_actions=4)]}
        answer = super().__call__(request)
        for candidate in answer.get("candidates", ()):
            candidate["iterations"] = 1
        return answer


class Worker(Environment):
    """A worker's bridge: a dry run's plan and the requests of ``schemas``, and nothing else. The YAM station's and
    RoboCasa's answer plans only (``YamSession.align_plan``, ``RoboCasaSession.align_plan``), RoboLab's the internal
    reach check too (``RobolabReachCheckMixin``)."""

    def __init__(self, schemas: tuple[str, ...]) -> None:
        super().__init__()
        self.schemas = schemas

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not is_request(request) and request.get("schema") not in self.schemas:
            raise ValueError("this worker serves only action plans on the read-only bridge")
        return super().__call__(request)


class RoboTwinWorkerProbe:
    """RoboTwin's motion probe as its worker answers it: ``RoboTwinLiveSession.probe_motion`` over the vendored planner
    with a fake solver (``tests/support/robotwin/ik_feedback.py``). Each query's exact variant plans, and each of its
    parallel-jaw equivalents finds no legal IK endpoint, so its planner record carries the public planner error object,
    as a real probe's rejected variants do. The toy probe never sends one."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.probed: list[dict[str, Any]] = []

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        from ora.environment.backends.robotwin.planning import RoboTwinPlanRejected, select_topk_validated_plan
        from tests.support.robotwin.ik_feedback import _no_valid_endpoint, _session, _valid_and_planned, _validated

        def admitted(candidate: Mapping[str, Any]) -> Any:
            if candidate.get("status") != "Success":
                raise RoboTwinPlanRejected("planner_failed", "planner did not produce a successful trajectory")
            return _validated()

        self.probed.append(json.loads(json.dumps(request)))
        # Every variant's IK finds no legal endpoint, but each query's exact variant (its first) is planned.
        session = _session(self.root / f"probe-{len(self.probed):02d}", _no_valid_endpoint)
        session._select_motion_probe_candidate = lambda probe, arm, variant, raw: (*select_topk_validated_plan(
            _valid_and_planned() if variant["variant_id"].startswith("variant-00-") else raw,
            validator=admitted)[:2], {})
        session._register_motion_probe_plan = lambda probe, arm, index, *rest: f"plan-{arm}-{index:02d}"
        return session.probe_motion(dict(request))


class Perception(Grounder):
    """The fake SAM3: the block's mask, for a text or for each point."""

    def _mask(self) -> Any:
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        mask[BLOCK[0]:BLOCK[1], BLOCK[2]:BLOCK[3]] = True
        return mask

    def ground(self, *, rgb_u8: Any, text: str, max_candidates: int) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        candidates, metadata = super().ground(rgb_u8=rgb_u8, text=text, max_candidates=max_candidates)
        candidates[0].update(mask=self._mask(), bbox_xyxy=[BLOCK[2], BLOCK[0], BLOCK[3], BLOCK[1]])
        return candidates, metadata

    def segment_points(self, *, rgb_u8: Any, depth_m: Any, points_uv: list[list[int]]) -> tuple[list, Mapping[str, Any]]:
        candidates = [{"query_index": index, "source_index": 0, "point_uv": point, "mask": self._mask(),
                       "backend_score": 0.9} for index, point in enumerate(points_uv)]
        queries = [{"query_index": index, "point_uv": point, "raw_candidate_count": 1, "error_code": None}
                   for index, point in enumerate(points_uv)]
        return candidates, {"query_results": queries, "image_encodings": 1}


def _camera(root: Path, name: str) -> dict[str, Any]:
    rgb = np.full((SIZE, SIZE, 3), 90, dtype=np.uint8)
    depth = np.full((SIZE, SIZE), TABLE_DEPTH_M, dtype=np.float32)
    rows, columns = slice(BLOCK[0], BLOCK[1]), slice(BLOCK[2], BLOCK[3])
    rgb[rows, columns] = (200, 40, 30)
    depth[rows, columns] = BLOCK_DEPTH_M
    directory = root / name
    directory.mkdir(parents=True)
    Image.fromarray(rgb).save(directory / "rgb.png")
    np.save(directory / "depth-m.npy", depth, allow_pickle=False)
    (directory / "calibration.json").write_text(json.dumps({
        "intrinsic_cv": [[100.0, 0.0, SIZE / 2], [0.0, 100.0, SIZE / 2], [0.0, 0.0, 1.0]],
        "extrinsic_cv": np.linalg.inv(np.array(CAMERA_TO_WORLD)).tolist(), "cam2world_cv": CAMERA_TO_WORLD,
        "depth_unit": "metres", "depth_kind": "distance_to_image_plane",
        "depth_array_order": "opencv_top_left_row_major"}, sort_keys=True))
    files = {}
    for role, filename in (("rgb_lossless_png", "rgb.png"), ("depth_numpy_m_f32", "depth-m.npy"),
                           ("calibration", "calibration.json")):
        path = directory / filename
        files[role] = {"path": f"{name}/{filename}", "sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}
    return {"width": SIZE, "height": SIZE, "modalities": ["rgb", "depth", "calibration"], "files": files}


@functools.lru_cache(maxsize=None)
def resolved(profile: str) -> Any:
    """One resolution per surface: a host resolves its surface on construction, and a row builds two hosts."""
    return resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile))


def _publish(root: Path, surface: Surface) -> tuple[Path, Path]:
    """Write the observation, its policy state and its control pointer, as a Supervisor publishes them."""

    source = root / "policy-observations" / OBSERVATION
    source.mkdir(parents=True)
    proprioception = dict(surface.proprioception)
    manifest = {"schema": "ora-observation.v1", "observation_id": OBSERVATION, "sequence": 0, "round_index": 0,
                "main_camera": surface.cameras[0], "encoding": "metric_rgbd_v1", "captured_unix_s": 1.0,
                "cameras": {name: _camera(source, name) for name in surface.cameras},
                "proprioception": proprioception, "forbidden_modalities_absent": sorted(REQUIRED_ABSENT),
                "previous_execution_receipt": None}
    descriptor = (resolved(surface.profile).view_config or {}).get("public_geometry_descriptor")
    if descriptor:
        manifest["robot_geometry_verification"] = {
            "schema": GEOMETRY_VERIFICATION_SCHEMA, "descriptor_sha256": descriptor["sha256"],
            "scope": "loaded_robot_geometry_only", "passed": True}
    if surface.handlers_only:
        manifest["controller_feedback"] = public_feedback_contract()
    (source / "proprioception.json").write_bytes(canonical(proprioception))
    (source / "observation.json").write_bytes(canonical(manifest))
    store = PolicyStateStore(root, "stand-in", "a" * 64)
    state = store.publish(observation=manifest, observation_root=source, cause="reset", task_instruction="stand-in",
                          previous_public_action=None, remaining_budget={"robot_decisions": 10, "model_calls": 20})
    control = root / "control"
    control.mkdir()
    (control / "current-observation.json").write_bytes(canonical({
        "open": True, "observation_id": OBSERVATION, "generation": 0, "source_root": str(source),
        "observation_sha256": digest((source / "observation.json").read_bytes()),
        "policy_state": {"step": 0, "state_sha256": state["state_sha256"]}}))
    (root / "agent-events.jsonl").write_bytes(canonical({
        "type": "fresh_observation_published", "seq": 1, "run_id": "run",
        "payload": {"observation_id": OBSERVATION}}) + b"\n")
    return control, store.root


def stand_in(root: Path, key: str, monkeypatch: Any) -> RobotToolRuntime:
    """The real runtime on ``key``'s surface over the stand-in observation, for ``on_toolkit`` to install the toolkit
    on. RoboDojo's surface leaves synchronous execution out (``Surface.handlers_only``)."""

    surface = SURFACES[key]
    root.mkdir(parents=True, exist_ok=True)
    control, state_root = _publish(root, surface)
    monkeypatch.setattr(ContactGraspNetApiProposer, "require_ready", lambda self: {"fixture": True})
    monkeypatch.setattr("ora.agent.toolkit.host.runtime.base.resolve_tool_surface", lambda profile, **_: resolved(profile))
    if surface.robot:
        return handlers_only(RobotToolRuntime(control_root=control, manual_path=ROBOT_MANUAL,
                                              event_path=root / "events.jsonl", workspace_bounds_m=BOUNDS,
                                              state_root=state_root, tool_surface_profile=surface.profile))
    bridge = Environment() if surface.bridge is None else Worker(surface.bridge)
    probe = RoboTwinWorkerProbe(root / "worker") if surface.robotwin_probe else bridge.probe
    host = ManipulationToolRuntime(
        control_root=control, manual_path=MANUAL, event_path=root / "events.jsonl", workspace_bounds_m=BOUNDS,
        state_root=state_root, tool_surface_profile=surface.profile, toolkit_grants=catalog_grants(surface.profile),
        controlled_end_effector="right", segmentation_backend=Perception(), grounding_backend=Perception(),
        public_facts_projector=project_policy_public_facts, align_plan_callback=bridge,
        motion_probe_callback=probe if surface.probe else None)
    contact = getattr(host.grasp_backend, "contact_backend", host.grasp_backend)
    if isinstance(contact, ContactGraspNetApiProposer):  # built for this surface; only its transport is fake
        contact._client.close()
        object.__setattr__(contact, "_client", httpx.Client(transport=httpx.MockTransport(grasp_service([]))))
    # The surface without the toolkit's execution capabilities (``surface_catalog.handlers_only``); ``on_toolkit``
    # puts ``toolkit_only_v1`` back.
    return handlers_only(host) if host.surface.supports("toolkit_only_v1") else host




def _answer(host: ManipulationToolRuntime, call: Call, index: int) -> Any:
    """``call``'s result, the ``index``-th tool call of the episode."""

    invoke = getattr(host, "invoke_with_context", None)
    if invoke is None:
        return host.invoke(call.tool, call.arguments)
    return invoke(call.tool, call.arguments, call_context=proposal(host, call.tool, dict(call.arguments), index))


def prepare(host: ManipulationToolRuntime, key: str, call: Call) -> tuple[Call, int]:
    """``call`` with what it ``needs``, prepared on ``host`` by the calls a policy makes first, and the index of the
    row's own first call."""

    if call.needs is None:
        return call, 0
    grounded = _answer(host, Call("vision_ground", {"camera": SURFACES[key].cameras[0], "text": "block"}), 0)
    reference = grounded["candidates"][0]["target_ref"]
    if call.needs != "candidate":
        return replace(call, arguments={**call.arguments, "target_ref": reference}), 1
    arm = {} if SURFACES[key].flat else {"arm": "right"}
    plan = _answer(host, Call("grasp_plan", {"target_ref": reference, **arm}), 1)
    return replace(call, arguments={**call.arguments, "candidate_id": plan["candidates"][0]["candidate_id"]}), 2


def on_toolkit(host: ManipulationToolRuntime) -> ManipulationToolRuntime:
    """Move ``host`` to the toolkit, which has every tool of every surface here."""

    install(host)
    return host


def _visible(value: Any) -> Any:
    """``value`` without the clock, with each image as its digest."""

    if isinstance(value, Mapping):
        return {str(key): (hashlib.sha256(json.dumps(item).encode()).hexdigest() if key in IMAGES else _visible(item))
                for key, item in value.items() if key not in VOLATILE}
    if isinstance(value, (list, tuple)):
        return [_visible(item) for item in value]
    return value


def _called(host: ManipulationToolRuntime, call: Call, index: int) -> dict[str, Any]:
    """The result or refusal of one call of a row."""

    try:
        return {"result": _visible(_answer(host, call, index))}
    except Exception as error:  # noqa: BLE001  the refusal or the fault is what is compared
        seen = {"refusal": _tool_error(error)["error"]}
        if not isinstance(error, ToolError):
            seen["raised"] = type(error).__name__
        return seen


def outcome(host: ManipulationToolRuntime, call: Call, start: int = 0) -> dict[str, Any]:
    """What ``call`` left: its result or its refusal, as the model gets it, and the action it submitted. Its earlier
    calls are the episode's from the ``start``-th."""

    if call.task_check is not None:
        (host.control_root / "policy-task-check.json").write_text(json.dumps(call.task_check))
    earlier: dict[str, Any] = {}
    for index, step in enumerate(call.first):
        earlier = _called(host, step, start + index)
    arguments = {name: earlier.get("result", {}).get("preview_id") if value == LAST_PREVIEW else value
                 for name, value in call.arguments.items()}
    seen = _called(host, replace(call, arguments=arguments), start + len(call.first))
    action = host.control_root / "submitted-action.json"
    if action.is_file():
        seen["action"] = json.loads(action.read_bytes())
    # Each host has its own root; what a result says of it is not the tool's answer.
    return json.loads(json.dumps(seen).replace(str(host.control_root.parent), "<root>"))


def answered(root: Path, key: str, call: Call, monkeypatch: Any) -> dict[str, Any]:
    """``call`` on a fresh stand-in on the toolkit, after what it ``needs``."""

    host = on_toolkit(stand_in(root, key, monkeypatch))
    prepared, start = prepare(host, key, call)
    return outcome(host, prepared, start)
