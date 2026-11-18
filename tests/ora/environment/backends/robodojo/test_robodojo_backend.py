"""RoboDojo policy boundary and lifecycle checks without an Isaac process."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from ora.environment.backends.robodojo.bridge import EvaluationLifecycle, renderer_table_binding
from ora.environment.backends.robodojo import bridge
from ora.environment.backends.robodojo import graphics_compat
from ora.environment.backends.robodojo import session as session_module
from ora.environment.backends.robodojo.common import (
    CLOSED_COMMAND_FLOOR, gripper_command_target,
)
from ora.environment.backends.robodojo.session import RobodojoError, RobodojoSession
from tests.support.robodojo.backend import (
    Robot, Manager, Plant, _fake_source_manifest, pose, v3,
    probe_request, trajectory_to,
)

pytestmark = pytest.mark.usefixtures("fake_sdk_candidate_boundary")


def test_resolved_sdk_control_holds_targets_under_feedback_and_repeated_reads():
    class FeedbackControl:
        def get_action(self, manager, env_idx):
            return {"left_arm": {"position": [0.2]}, "left_ee": {"position": [manager.measured + .01]},
                    "right_ee": {"position": [manager.measured + .01]}}

    manager = SimpleNamespace(measured=0.0)
    previous = {"left_arm": {"position": [0.1]}, "left_ee": {"position": [.044]},
                "right_ee": {"position": [.044]}}
    original = FeedbackControl()
    command, targets = bridge._fixed_meta_control(original, manager, previous, {"left_arm": {}})
    assert isinstance(command, FeedbackControl)
    assert targets == {**previous, "left_arm": {"position": [.2]}}
    manager.measured = -.02
    assert command.get_action(manager, env_idx=0) == targets
    command.get_action(manager, env_idx=0)["left_ee"]["position"][0] = 100
    assert command.get_action(manager, env_idx=0) == targets
    # An explicit gripper action retains the SDK's feedback-limited command.
    changed, targets = bridge._fixed_meta_control(FeedbackControl(), manager, previous, {"left_ee": {}})
    assert targets["left_ee"]["position"] == [-.01]
    assert targets["left_arm"] == previous["left_arm"]
    assert targets["right_ee"] == previous["right_ee"]
    assert FeedbackControl().get_action(manager, env_idx=0)["right_ee"]["position"] == [-.01]


@pytest.mark.parametrize("method", ["get_functional_points", "get_support_points"])
def test_private_nested_evaluator_geometry_reads_use_cpu_adapter(method):
    root = np.array([3.25, -4.5, 2.75, 1, 0, 0, 0], dtype=np.float32)
    obj = SimpleNamespace(get_state=lambda is_relative: {"root_pose": root},
                          env_origin=np.array([3, -5, 2], dtype=np.float32))

    class Layout:
        instance_type_by_env = [{"rack": "geometry"}]

        def get_scene_object(self, env_idx, inst_name):
            return obj

        def get_instance_pose(self, **kwargs):
            raise RuntimeError("unadapted GPU geometry arithmetic")

        def get_functional_points(self, **kwargs):
            return self.get_instance_pose(**kwargs)

        def get_support_points(self, **kwargs):
            return self.get_instance_pose(**kwargs)

    delegate = Layout()
    with pytest.raises(RuntimeError, match="unadapted"):
        getattr(delegate, method)(env_idx=0, inst_name="rack")
    result = getattr(bridge.EvaluatorLayoutInput(delegate), method)(env_idx=0, inst_name="rack")
    np.testing.assert_array_equal(result[0], [.25, .5, .75])
    np.testing.assert_array_equal(root, [3.25, -4.5, 2.75, 1, 0, 0, 0])


@pytest.mark.parametrize("fault", [None, "resolution", "lens", "offset"])
def test_intrinsics_follow_effective_render_projection_not_raw_vertical_aperture(fault):
    from ora.environment.backends.robodojo.camera_capture import renderer_intrinsics

    parameters = {
        "cameraModel": "pinhole", "cameraApertureOffset": [0, 0],
        "cameraAperture": [20, 12], "cameraFocalLength": 9,
        "renderProductResolution": [640, 480],
        "cameraProjection": [.9, 0, 0, 0, 0, 1.2, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
    }
    if fault == "resolution":
        parameters["renderProductResolution"] = [480, 640]
    elif fault == "lens":
        parameters["cameraModel"] = "fisheye"
    elif fault == "offset":
        parameters["cameraApertureOffset"] = [.1, 0]
    if fault:
        with pytest.raises(ValueError):
            renderer_intrinsics(parameters, (640, 480))
    else:
        np.testing.assert_allclose(renderer_intrinsics(parameters, (640, 480)),
                                   [[288, 0, 320], [0, 288, 240], [0, 0, 1]])


def test_private_evaluator_pose_copy_preserves_values_and_leaves_physics_manager_untouched():
    from ora.environment.backends.robodojo.bridge import EvaluatorLayoutInput

    class DevicePose:
        def __init__(self, value):
            self.value = value

        def __array__(self):
            raise TypeError("a GPU pose cannot be converted directly")

        def detach(self):
            return self

        def cpu(self):
            return SimpleNamespace(numpy=lambda: self.value)

    position = np.array([.125, -.25, .875], dtype=np.float32)
    rotation = np.array([0, 1, 0, 0], dtype=np.float64)
    poses = (DevicePose(position), DevicePose(rotation))
    calls = []

    def get_pose(*args, **kwargs):
        calls.append((args, kwargs))
        return poses

    delegate = SimpleNamespace(get_instance_pose=get_pose, private_marker=object())
    adapter = EvaluatorLayoutInput(delegate)
    result = adapter.get_instance_pose(env_idx=0, inst_name="fixture", relative=True)
    assert calls == [((), {"env_idx": 0, "label": None, "inst_name": "fixture", "relative": True})]
    for actual, expected in zip(result, (position, rotation)):
        assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)
    assert delegate.get_instance_pose()[0] is poses[0]
    assert adapter.private_marker is delegate.private_marker


def test_private_geometry_pose_uses_same_origin_subtraction_on_cpu():
    from ora.environment.backends.robodojo.bridge import EvaluatorLayoutInput

    root = np.array([3.25, -4.5, 2.75, 1, 0, 0, 0], dtype=np.float32)
    origin = np.array([3, -5, 2], dtype=np.float32)
    calls = []

    def get_state(*, is_relative):
        calls.append(is_relative)
        if is_relative:
            raise RuntimeError("mixed CPU/GPU origin")
        return {"root_pose": root}

    obj = SimpleNamespace(get_state=get_state, env_origin=origin)
    delegate = SimpleNamespace(instance_type_by_env=[{"pad": "geometry"}],
                               get_scene_object=lambda env_idx, inst_name: obj)
    position, rotation = EvaluatorLayoutInput(delegate).get_instance_pose(0, inst_name="pad")
    np.testing.assert_array_equal(position, np.array([.25, .5, .75], dtype=np.float32))
    np.testing.assert_array_equal(rotation, root[3:])
    assert position.dtype == root.dtype and calls == [False]
    assert root[0] == 3.25 and origin[0] == 3


@pytest.mark.parametrize("fault", [None, "timeout", "physics"])
def test_texture_wait_requires_real_readiness_without_physics_advance(tmp_path, monkeypatch, fault):
    from ora.environment.backends.robodojo import camera_capture

    clock = {"frames": 0, "physics": 17, "time": 0.0}
    monkeypatch.setattr(camera_capture.time, "monotonic", lambda: clock["time"])
    manager = SimpleNamespace(num_envs=1, tiled_cameras=[
        SimpleNamespace(_render_product_path=f"/camera/{index}") for index in range(3)])

    def render():
        clock["frames"] += 1
        clock["time"] += 1
        if fault == "physics":
            clock["physics"] += 1

    def run():
        camera_capture.wait_for_render_assets(
            manager=manager, render=render,
            assets_loading=lambda: fault == "timeout" or clock["frames"] < 2,
            stage_status=lambda: {"loading": ["local textures", 2, 3]},
            physics_clock=lambda: clock["physics"], output_root=tmp_path, timeout_s=3)

    if fault is None:
        run()
        assert clock["frames"] == 2 and clock["physics"] == 17
        evidence = [json.loads(line) for line in (tmp_path / "render-readiness.jsonl").read_text().splitlines()]
        assert evidence[0]["assets_loading"] is True
        assert evidence[-1]["assets_loading"] is False
    else:
        with pytest.raises(TimeoutError if fault == "timeout" else ValueError):
            run()
        assert clock["frames"] <= 3


def test_crash_report_guard_precedes_sdk_startup_and_preserves_local_dumps(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge.os, "environ", dict(bridge.os.environ))
    monkeypatch.setattr(sys, "path", list(sys.path))
    for variable in bridge._CRASH_REPORT_ENV:
        monkeypatch.setenv(variable, "unsafe-inherited-value")
    root = tmp_path / "source"
    experience = root / "third_party/IsaacLab/apps/isaaclab.python.kit"
    experience.parent.mkdir(parents=True)
    experience.touch()
    output = tmp_path / "episode"
    monkeypatch.setattr(bridge, "source_manifest", lambda config: {"loaded_files": {}})
    monkeypatch.setattr(graphics_compat, "activate", lambda root: {"files": {}})
    monkeypatch.setitem(sys.modules, "env.global_configs", SimpleNamespace())

    class StopBeforeSDK(RuntimeError):
        pass

    def launcher(**kwargs):
        assert kwargs["headless"] is True and kwargs["enable_cameras"] is True
        assert kwargs["fast_shutdown"] is False
        assert kwargs["device"] == "cuda:0" and kwargs["experience"] == str(experience)
        assert "--/renderer/activeGpu=2" in kwargs["kit_args"]
        arguments = kwargs["kit_args"].split()
        assert "--/crashreporter/url=" in arguments
        assert "--/app/uploadDumpsOnStartup=false" in arguments
        assert "--/crashreporter/skipOldDumpUpload=true" in arguments
        assert "--/crashreporter/devOnlyOverridePrivacyAndForceUpload=false" in arguments
        assert "--/crashreporter/enabled=true" in arguments
        assert "--/crashreporter/preserveDump=true" in arguments
        assert "--/privacy/performance=false" in arguments
        # Kit launches the telemetry transmitter during extension startup, before
        # the post-start privacy load; it must already see every category off.
        assert "--/privacy/usage=false" in arguments
        assert "--/privacy/personalization=false" in arguments
        assert "--/telemetry/enableAnonymousData=false" in arguments
        assert "--/telemetry/enableAnonymousAppName=false" in arguments
        privacy = output / "crash-reporting-privacy.toml"
        assert f"--/structuredLog/privacySettingsFile={privacy}" in arguments
        assert bridge.tomllib.loads(privacy.read_text()) == {
            "privacy": {"performance": False, "personalization": False, "usage": False}}
        assert {key: bridge.os.environ[key] for key in bridge._CRASH_REPORT_ENV} == bridge._CRASH_REPORT_ENV
        raise StopBeforeSDK

    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=launcher))
    with pytest.raises(StopBeforeSDK):
        bridge.build_environment({"runtime_directory": str(tmp_path / "runtime"), "robodojo_root": str(root),
                                  "asset_root": str(tmp_path / "assets"), "output_root": str(output),
                                  "renderer_gpu": 2, "physics_device": "cuda:0"})


@pytest.mark.parametrize("mutation", [None, "layer_bytes", "extra_property", "wrong_limit", "manifest", "identity", "readback_limit", "patch"])
def test_vulkan_compatibility_requires_exact_installation_and_preserves_device_identity(tmp_path, monkeypatch, mutation):
    import hashlib

    monkeypatch.setattr(graphics_compat.os, "environ", {})
    root = tmp_path / "graphics-compat"
    root.mkdir()
    layer, profile, manifest = (root / name for name in ("layer.so", "profile.json", "manifest.json"))
    layer.write_bytes(b"fixture layer")
    limit = 123 if mutation == "wrong_limit" else graphics_compat.CAP
    properties = {"VkPhysicalDeviceMaintenance3Properties": {"maxMemoryAllocationSize": limit}}
    if mutation == "extra_property":
        properties["VkPhysicalDeviceVulkan11Properties"] = {"deviceUUID": "invented"}
    profile.write_text(json.dumps({
        "capabilities": {"allocation": {"properties": properties}},
        "profiles": {graphics_compat.PROFILE: {"capabilities": ["allocation"]}},
    }))
    manifest.write_text(json.dumps({"layer": {
        "name": "VK_LAYER_KHRONOS_profiles", "type": "GLOBAL", "library_path": str(layer),
        "enable_environment": {"GLOBAL_ENABLE" if mutation == "manifest" else "ROBODOJO_VULKAN_COMPAT_ENABLE": "1"},
        "disable_environment": {"ROBODOJO_VULKAN_COMPAT_DISABLE": ""},
    }}))
    patch = root / "core-id.patch"
    patch.write_bytes((Path(__file__).resolve().parents[5] / "ora/environment/backends/robodojo/installation/vulkan-profiles-core-id.patch").read_bytes())
    receipt = {
        "patches": [{"path": patch.name, "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}],
        "schema": "ora.robodojo-graphics-compat-installation.v1",
        "source_url": "https://github.com/KhronosGroup/Vulkan-Profiles.git",
        "source_revision": graphics_compat.SOURCE_REVISION,
        **{role: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
           for role, path in (("layer", layer), ("profile", profile), ("manifest", manifest))},
    }
    (root / "installation.json").write_text(json.dumps(receipt))
    if mutation == "patch":
        patch.write_bytes(b"unreviewed patch")
    if mutation == "layer_bytes":
        layer.write_bytes(b"altered layer")
    before = [{"device_uuid": "a" * 32, "physical_properties_sha256": "b" * 64,
               "max_memory_allocation_size": 2**64 - 1}]
    after = deepcopy(before)
    after[0]["max_memory_allocation_size"] = 1 if mutation == "readback_limit" else graphics_compat.CAP
    if mutation == "identity":
        after[0]["device_uuid"] = "0" * 32

    def readback(*, expected_layer=None):
        if expected_layer is not None:
            assert expected_layer == layer
            assert graphics_compat.os.environ["VK_KHRONOS_PROFILES_EMULATE_PORTABILITY"] == "false"
            assert graphics_compat.os.environ["VK_KHRONOS_PROFILES_SIMULATE_CAPABILITIES"] == "SIMULATE_PROPERTIES_BIT"
            return after
        return before

    monkeypatch.setattr(graphics_compat, "properties", readback)
    if mutation is not None:
        with pytest.raises(graphics_compat.GraphicsCompatibilityError):
            graphics_compat.readback(tmp_path)
    else:
        value = graphics_compat.readback(tmp_path)
        assert value["before"] == before and value["after"] == after
        assert value["files"] == {**{key: receipt[key] for key in ("layer", "profile", "manifest")}, "patch": receipt["patches"][0]}


@pytest.mark.parametrize("kind,mutation", [("rgb", None), ("depth", None),
    ("depth", "dtype"), ("rgb", "shape"), ("rgb", "empty"), ("rgb", "multi_env")])
def test_camera_readback_preserves_original_pixels_and_rejects_malformed_buffers(tmp_path, kind, mutation):
    from ora.environment.backends.robodojo.camera_capture import install_single_environment_readback

    channels = 4 if kind == "rgb" else 1
    values = np.arange(3 * 5 * channels, dtype=np.uint8 if kind == "rgb" else np.float32).reshape(3, 5, channels)
    if kind == "depth":
        values[0, 0, 0] = np.inf
    expected = values.copy()
    if mutation == "dtype":
        values = values.astype(np.float64)
    if mutation == "shape":
        values = values.transpose(1, 0, 2)
    if mutation == "empty":
        values = values[:0]

    class Annotator:
        def get_data(self, *, device):
            assert device == "cpu"
            return {"data": values, "info": {"fixture": True}}

    view = SimpleNamespace(prims=[object()], camera_resolution=(5, 3), tiled_resolution=(5, 3),
                           _render_product_path="/Render/OriginalCamera", _annotators={
                               "rgba" if kind == "rgb" else "distance_to_image_plane": Annotator()})
    manager = SimpleNamespace(num_envs=2 if mutation == "multi_env" else 1, tiled_cameras=[view] * 3)
    if mutation == "multi_env":
        with pytest.raises(ValueError, match="one environment"):
            install_single_environment_readback(manager, tmp_path)
        return
    install_single_environment_readback(manager, tmp_path)
    if mutation:
        with pytest.raises(ValueError, match="unexpected shape/dtype"):
            view.get_data(kind)
    else:
        actual, info = view.get_data(kind)
        np.testing.assert_array_equal(actual, expected[None])
        values.fill(0)
        np.testing.assert_array_equal(actual, expected[None])
        assert info == {"fixture": True}
    evidence = json.loads((tmp_path / "camera-buffer-readback.json").read_text())
    assert evidence["buffers"][0]["pixels_resampled"] is False


@pytest.mark.parametrize("changed", [None, *bridge._CRASH_REPORT_SETTINGS,
                                    "/structuredLog/privacySettingsFile", "environment", "consent_file"])
def test_crash_report_readback_rejects_upload_or_diagnostic_setting_drift(tmp_path, monkeypatch, changed):
    for key in bridge._CRASH_REPORT_ENV:
        monkeypatch.setenv(key, "inherited")
    _, privacy = bridge._configure_crash_reporting(tmp_path)
    actual = {**bridge._CRASH_REPORT_SETTINGS, "/structuredLog/privacySettingsFile": str(privacy)}
    if changed == "environment":
        monkeypatch.setenv("OMNI_CRASHREPORTER_URL", "https://example.invalid/upload")
    elif changed == "consent_file":
        privacy.chmod(0o644)
        privacy.write_text("[privacy]\nperformance = true\n")
    elif changed is not None:
        actual[changed] = not actual[changed] if isinstance(actual[changed], bool) else "unexpected"
    if changed is not None:
        with pytest.raises(RobodojoError, match="crash-report privacy"):
            bridge._crash_reporting_readback(actual, privacy)
    else:
        evidence = bridge._crash_reporting_readback(actual, privacy)
        assert evidence["uploads_disabled"] is True and evidence["local_dumps_preserved"] is True
        assert evidence["privacy_file"]["path"] == privacy.name
        assert len(evidence["privacy_file"]["sha256"]) == 64


@pytest.mark.parametrize("corruption", [None, "wrong_cuda", "wrong_index", "ambiguous_prefix", "two_active", "no_active"])
def test_renderer_readback_requires_actual_active_uuid_and_cuda_agreement(corruption):
    gpu_uuid = "GPU-a0161c6f-1234-5678-1234-123456789abc"
    table = "\n".join([
        "| 2 | NVIDIA RTX 5880 | Yes: 0 | | 49386 MB | 10de | 0 |",
        "|   |                 |        | |          | 26b3 | a0161c6f.. |",
        "|   |                 |        | |          | 38   |             |",
    ])
    inventory = [{"index": 2, "uuid": gpu_uuid, "pci_bus_id": "00000000:38:00.0"}]
    selected_index, cuda_uuid = 2, gpu_uuid
    if corruption == "wrong_cuda":
        cuda_uuid = "GPU-00000000-1234-5678-1234-123456789abc"
    elif corruption == "wrong_index":
        selected_index = 0
    elif corruption == "ambiguous_prefix":
        inventory.append({**inventory[0], "index": 3, "uuid": gpu_uuid[:-1] + "d"})
    elif corruption == "two_active":
        table += "\n" + table
    elif corruption == "no_active":
        table = table.replace("Yes: 0", "      ")
    if corruption is not None:
        with pytest.raises(RobodojoError):
            renderer_table_binding(table, selected_index=selected_index, cuda_uuid=cuda_uuid, inventory=inventory)
        return
    binding = renderer_table_binding(table, selected_index=selected_index, cuda_uuid=cuda_uuid, inventory=inventory)
    assert binding["selected_gpu_uuid"] == gpu_uuid
    assert binding["renderer_vulkan_index"] == 2 and binding["physics_cuda_logical_index"] == 0


@pytest.fixture
def session(tmp_path):
    plant = Plant()
    instance = RobodojoSession(
        config={"workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}},
        output_root=tmp_path / "native",
        environment_factory=lambda config: (
            plant, SimpleNamespace(close=lambda: None), _fake_source_manifest(tmp_path),
        ),
    )
    instance._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    instance.start()
    try:
        yield instance
    finally:
        instance.close()


def action(session, tmp_path, *, kind="move_pose", arm="left", value=None):
    observation, _ = session.observe(tmp_path / "observations")
    arguments = ({"arm": arm, "position_m": [0.02, 0, 0.8], "orientation_xyzw": [0, 0, 0, 1]}
                 if kind == "move_pose" else {"arm": arm, "gripper": 0.0 if value is None else value})
    return {"schema": "ora-action.v3", "observation_id": observation["observation_id"],
            "action_id": f"action-{session.action_sequence}", "action": kind, "arguments": arguments,
            "bound_observation_sha256": "a" * 64}


def test_observation_is_measured_immutable_rgbd_and_omits_private_state(session, tmp_path):
    manager = session.environment.robot_manager
    manager.fingers["left_arm"] = [0.017, 0.017]
    payload, paths = session.observe(tmp_path / "observations")
    assert payload["proprioception"]["end_effectors"]["left"]["gripper"] == pytest.approx(0.5)
    assert payload["proprioception"]["end_effectors"]["left"]["gripper_command"] == 1.0
    assert "object_gt" not in payload
    assert len(paths) == 3
    assert session.environment.sim._sim_step_counter == 500
    assert all(path.stat().st_mode & 0o222 == 0 for path in paths)
    assert not session.is_terminal
    assert session.persist_official_verdict()["official_success"] is None


def test_pose_action_keeps_gripper_and_other_arm_commands(session, tmp_path):
    before = deepcopy(session.environment.robot_manager.control_manager.prev_control)
    receipt = session.execute(action(session, tmp_path))["receipt"]
    assert receipt["motion_status"] == "reached"
    assert receipt["native_actions"] == 1 and receipt["physics_steps"] == 10
    assert set(session.environment.commands[0]) == {"left_arm_joint_state"}
    after = session.environment.robot_manager.control_manager.prev_control
    assert before[0]["right_arm_joint_state"] == after[0]["right_arm_joint_state"]
    assert before[0]["left_ee_joint_state"] == after[0]["left_ee_joint_state"]
    next_observation, _ = session.observe(tmp_path / "observations")
    reference = next_observation["previous_execution_receipt"]
    assert reference["receipt"] == receipt
    assert json.loads((session.output_root / reference["path"]).read_text()) == receipt


def test_sdk_target_mismatch_aborts_even_when_retained_intent_is_unchanged(session, tmp_path):
    original = session._applied_position_targets

    def corrupted(*readings):
        values = original(*readings)
        values["left_ee_joint_state"] = [.03, .03]
        return values

    session._applied_position_targets = corrupted
    with pytest.raises(RobodojoError, match="SDK applied"):
        session.execute(action(session, tmp_path))
    assert session.abort_reason == "robodojo_applied_command_mismatch"
    assert session.gripper_directives["left"] == 1.0
    journal = session.output_root / "controller/action-0000.jsonl"
    evidence = json.loads(journal.read_text().splitlines()[0])
    assert evidence["applied_position_targets"]["left_ee_joint_state"] == [.03, .03]
    assert session.persist_official_verdict()["official_success"] is None


@pytest.mark.parametrize('mode', ['session', 'sparse'])
@pytest.mark.parametrize('corruption', [None, 'auxiliary', 'held_target'])
def test_interactive_arm_executes_and_is_journaled_without_becoming_a_policy_target(
    request, tmp_path, mode, corruption,
):
    session = request.getfixturevalue(mode)
    plant = session.environment
    plant.interact = True
    manager = plant.robot_manager
    robot = Robot('support')
    robot.type = 'support'
    manager.robot_list.append(robot)
    manager.robot_key.append(deepcopy(manager.robot_key[0]))
    manager.joints[robot.arm_name] = list(manager.home)
    manager.fingers[robot.arm_name] = [.044, .044]
    support_key = manager.process_name(robot.arm_name)
    manager.control_manager.prev_control[0].update({
        support_key: {'position': list(manager.home)},
        manager.process_name(robot.gripper_name): {'position': [.044, .044]},
    })
    native_tick = plant.native_tick

    def autonomous_tick(control):
        auxiliary = list(manager.home)
        auxiliary[0] = .1
        return native_tick({**control, support_key: {'position': auxiliary}})
    plant.native_tick = autonomous_tick
    if corruption:
        readback = session._applied_position_targets
        def damaged(*readings):
            result = readback(*readings)
            result[support_key if corruption == 'auxiliary' else 'right_arm_joint_state'][0] += .01
            return result
        session._applied_position_targets = damaged
    command = (action(session, tmp_path, kind='set_gripper', value=.5) if mode == 'session'
               else v3(session, tmp_path, kind='set_gripper', targets={'left': 'close'}))
    if corruption:
        with pytest.raises(RobodojoError, match='SDK applied'):
            session.execute(command)
        assert session.abort_reason == 'robodojo_applied_command_mismatch'
    else:
        receipt = session.execute(command)['receipt']
        assert receipt['command_invariant']['verified'] is True
        assert all('support' not in key for field in ('command_targets_before', 'command_targets_after')
                   for key in receipt[field])
        observation, _ = session.observe(tmp_path/'post-observation')
        assert set(observation['proprioception']['end_effectors']) == {'left', 'right'}
    samples = [json.loads(line) for line in (session.output_root/'controller/action-0000.jsonl').read_text().splitlines()]
    assert samples[0]['command'][support_key]['position'][0] == .1
    assert support_key in samples[0]['applied_position_targets']


def test_native_two_tool_path_projects_actual_robodojo_state_and_receipt(session, tmp_path):
    """The legacy two-tool surface over the RoboDojo session: the policy's first
    observation carries the image, its submitted action executes, and the next
    dump_state shows the receipt and the reached pose (the policy turn itself
    runs in the Pi core, covered by tests/agent_core)."""
    from ora.agent.context import ContextProjector
    from ora.agent.evidence.policy_state import PolicyStateStore, verify_policy_state
    from ora.agent.toolkit.host.runtime.factory import build_tool_runtime
    from tests.support.agent.observation_delivery import Delivery, host

    root = Path(__file__).resolve().parents[5]
    # Sample policy text copied from the robodojo-compat package (relocation U6).
    rules = root / "tests/data/robodojo/planner-v1.md"
    prompt = root / "tests/data/robodojo/core-v1.md"
    manual = root / "ora/policy/embodiments/arx-x5/compiled-v1/manual.md"
    store = PolicyStateStore(tmp_path, "robodojo-native-boundary", "a" * 64)
    projector = ContextProjector(
        tmp_path / "workspace", tmp_path / "control", prompt, manual, rules,
        main_camera="head", expose_manual=False,
    )
    tools = build_tool_runtime(
        control_root=tmp_path / "control", manual_path=manual, event_path=tmp_path / "events.jsonl",
        workspace_bounds_m=session.config["workspace_bounds_m"],
        allowed_tools=["observation.dump_state", "robot.submit_action"], state_root=store.root,
        profile_id="robot_runtime_v2", component_overrides=None,
    )

    def publish(previous=None):
        payload, _ = session.observe(tmp_path / "policy-observations")
        source = tmp_path / "policy-observations" / payload["observation_id"]
        shared = dict(observation=payload, observation_root=source,
                      task_instruction="Pick up the displayed object by 10 cm.",
                      remaining_budget={"robot_decisions": 199, "model_calls": 398},
                      previous_execution_receipt=previous)
        snapshot = store.publish(**shared, cause="reset" if previous is None else "after_action",
                                 previous_public_action=None)
        record = store.steps[-1]
        state_ref = {"step": snapshot["step"], "path": record["path"],
                     "sha256": record["sha256"], "state_sha256": snapshot["state_sha256"]}
        return projector.publish(**shared, policy_state=state_ref)

    pointer = publish()
    # The observation the policy session drafts for this legacy composition.
    planner = host(tmp_path, tools=tools, context_composition=None, context_retention="rewrite_compact_v1")
    initial, _ = Delivery(planner).observation(
        pointer, task_instruction="Pick up the displayed object by 10 cm.",
        image_path=projector.workspace / pointer["main_image"])
    assert any(part["type"] == "image_url" for part in initial["content"])
    assert tools.invoke("observation.dump_state", {})["image_camera"] == "head"
    # RoboDojo takes no target-state v4, the only form robot.submit_action compiles.
    receipt = session.execute(action(session, tmp_path, arm="right"))
    publish(receipt)
    state = tools.invoke("observation.dump_state", {})
    assert state["image_camera"] == "head" and state["image_base64"]
    assert state["camera_inventory"]["left_wrist"]["calibration_dynamic"] is True
    assert state["camera_inventory"]["right_wrist"]["modalities"] == ["rgb", "depth", "calibration"]
    assert state["state"]["previous_execution_receipt"] == receipt
    assert state["state"]["proprioception"]["end_effectors"]["right"]["pose_xyz_xyzw"][:3] == [0.02, 0, 0.8]
    assert verify_policy_state(tmp_path)["status"] == "valid"


def test_gripper_action_holds_pose_target_and_other_arm(session, tmp_path):
    before = deepcopy(session.pose_targets)
    receipt = session.execute(action(session, tmp_path, kind="set_gripper"))["receipt"]
    assert receipt["reached_gripper"] == 0
    assert set(session.environment.commands[0]) == {"left_ee_joint_state"}
    assert session.pose_targets == before


@pytest.mark.parametrize("corruption", [None, "pinned_assets", "missing_collector", "missing_link",
                                        "wrong_camera_mount", "source_hash", "unreviewed_assets"])
def test_embodiment_activation_requires_bound_compiled_runtime_evidence(session, tmp_path, corruption):
    import hashlib
    from ora.orchestrator.backends.robodojo.embodiment import CARD_ROOT, create_activation
    from ora.agent.composition.embodiment import descriptor_activation_ready
    from ora.environment.controllers.robodojo_arx import controller_binding

    descriptor = json.loads((CARD_ROOT / "descriptor-v1.json").read_text())
    provenance = descriptor["provenance"]["sources"][1]
    qualification_root = tmp_path / "activation-probe"
    reports = []
    for index in range(2):
        environment_root = qualification_root / f"reset-{index}" / "environment"
        environment_root.mkdir(parents=True)
        source = {"source_revisions": {"assets": provenance["revision"]},
                  "loaded_files": {"Assets/Robots/x5/robot_config.yml": provenance["sha256"]}}
        # The pinned Assets hold the descriptor's robot files byte for byte; any other revision is drift.
        if corruption == "pinned_assets":
            from ora.environment.benchmarks.robodojo import REVISIONS
            assert REVISIONS["assets"] != provenance["revision"]
            source["source_revisions"]["assets"] = REVISIONS["assets"]
        elif corruption == "unreviewed_assets":
            source["source_revisions"]["assets"] = "d" * 40
        if corruption == "source_hash":
            source["loaded_files"]["Assets/Robots/x5/robot_config.yml"] = "0" * 64
        source_path = environment_root / "source-manifest.json"
        source_path.write_text(json.dumps(source))
        runtime = session._runtime_embodiment()
        if corruption == "missing_collector":
            runtime.pop("collector")
        elif corruption == "missing_link":
            runtime["arms"]["left"]["compiled_body_names"].remove("link6")
        elif corruption == "wrong_camera_mount":
            runtime["cameras"]["left_wrist"]["mount_link"] = "robot1/camera"
        reset = {"source_manifest": {"sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()},
                 "control_binding": controller_binding(), "runtime_embodiment": runtime}
        (environment_root / "reset.json").write_text(json.dumps(reset))
        sensors = {name: {"finite_positive_fraction": 1.0, "shape": [8, 10]}
                   for name in ("head", "left_wrist", "right_wrist")}
        report = {"passed": True, "reset": reset, "initial_sensors": sensors, "repeated_sensors": sensors}
        (environment_root.parent / "probe-report.json").write_text(json.dumps(report))
        reports.append(report)
    if corruption not in (None, "pinned_assets"):
        with pytest.raises(RuntimeError, match="activation rejected"):
            create_activation(qualification_root, reports)
        assert not (qualification_root / "embodiment/activation.json").exists()
        return
    binding = create_activation(qualification_root, reports)
    activation_path = qualification_root / binding["activation_record"]["path"]
    activation = json.loads(activation_path.read_text())
    assert descriptor_activation_ready(descriptor, activation_record=activation) is True
    assert hashlib.sha256(activation_path.read_bytes()).hexdigest() == binding["activation_record"]["sha256"]
    assert activation_path.stat().st_mode & 0o222 == 0


def test_failed_ik_does_not_step_or_change_gripper(session, tmp_path):
    session.environment.robot_manager.fail_ik = True
    before = deepcopy(session.environment.robot_manager.control_manager.prev_control)
    receipt = session.execute(action(session, tmp_path))["receipt"]
    assert receipt["motion_status"] == "ik_failed"
    assert receipt["native_actions"] == receipt["physics_steps"] == 0
    assert before == session.environment.robot_manager.control_manager.prev_control
    assert session.environment.commands == []


def test_official_terminal_is_snapshot_and_rejects_postterminal_actions(session, tmp_path):
    session.environment.terminal_at = 1
    proposal = action(session, tmp_path)
    session.execute(proposal)
    assert session.persist_official_verdict()["official_success"] is True
    session.environment.success = [False]  # Mutating an upstream eligibility flag cannot revise the snapshot.
    for _ in range(4):
        assert session.status()["reason"] == "official_success"
    session.observe(tmp_path / "observations")
    with pytest.raises(RobodojoError, match="cannot accept"):
        session.execute(proposal)
    session.abort("operator_stop")
    assert session.persist_official_verdict()["official_success"] is True
    assert session.environment.take_action_cnt == [1]


def test_official_exhaustion_is_false_but_outer_budget_is_incomplete(session, tmp_path):
    session.environment.take_action_cnt = [1049]
    session.sim_steps = 10490
    session.execute(action(session, tmp_path, kind="set_gripper"))
    verdict = session.persist_official_verdict()
    assert verdict["terminal"] and verdict["official_success"] is False
    assert verdict["native_actions"] == 1050


def test_policy_budget_reports_the_standing_official_predicate(session, tmp_path):
    """A budget stop is a behavioral boundary, so it carries a Boolean.

    RoboDojo latches success inside ``RewardManager.step``, which runs after
    every native action, and nothing advances physics outside a native tick.
    So an episode with no official terminal transition has a standing official
    predicate that was already evaluated after the last physics step and was
    not a success. That is also why the Boolean on this path is always false:
    a success would have produced the terminal transition instead.
    """

    session.execute(action(session, tmp_path))
    assert session.persist_official_verdict()["verdict_basis"] == "episode_incomplete"
    session.abort("robot_decision_budget")
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is False and verdict["terminal"] is False
    assert verdict["verdict_basis"] == "standing_official_predicate_at_policy_budget"
    assert verdict["fault_reason"] is None
    assert verdict["abort_reason"] == "robot_decision_budget"


def test_exhausted_episode_physics_budget_reports_the_same_predicate(session, tmp_path):
    session.max_sim_steps = 10
    session.execute(action(session, tmp_path))
    assert session.is_terminal
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is False and verdict["terminal"] is False
    assert verdict["verdict_basis"] == "standing_official_predicate_at_policy_budget"
    assert verdict["fault_reason"] is None


def test_official_success_on_the_last_decision_outranks_the_budget_stop(session, tmp_path):
    """The only reachable true verdict at a budget edge is a real transition."""

    session.environment.terminal_at = 1
    session.execute(action(session, tmp_path))
    session.abort("robot_decision_budget")
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is True and verdict["terminal"] is True
    assert verdict["verdict_basis"] == "official_terminal_transition"
    assert verdict["fault_reason"] is None and verdict["abort_reason"] is None


def test_infrastructure_fault_never_becomes_an_official_task_failure(session, tmp_path):
    session.environment.leak_clock = True
    with pytest.raises(RobodojoError, match="advanced physics"):
        session.observe(tmp_path / "observations")
    session.abort("robot_decision_budget")  # A later budget cannot relabel a fault.
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is None and verdict["terminal"] is False
    assert verdict["verdict_basis"] == "infrastructure_abort"
    assert verdict["fault_reason"] == "observation_advanced_physics"
    assert verdict["abort_reason"] == "observation_advanced_physics"


def test_observation_clock_violation_and_stale_action_fail_closed(session, tmp_path):
    proposal = action(session, tmp_path)
    proposal["observation_id"] = "stale"
    with pytest.raises(RobodojoError, match="latest unconsumed"):
        session.execute(proposal)
    session.environment.leak_clock = True
    with pytest.raises(RobodojoError, match="advanced physics"):
        session.observe(tmp_path / "observations")
    assert session.is_terminal


def test_official_transition_requires_registration_and_never_repolls_terminal():
    lifecycle = object.__new__(EvaluationLifecycle)
    lifecycle.reward_registered = False
    lifecycle._official_terminal = None
    with pytest.raises(RobodojoError, match="not been initialized"):
        lifecycle.advance_official_terminal()
    lifecycle.reward_registered = True
    lifecycle.take_action_cnt, lifecycle.success, lifecycle.end_flag = [200], [True], [False]
    lifecycle.step_lim = 200
    calls = []
    lifecycle.reward_manager = SimpleNamespace(get_reward=lambda final_check: calls.append(final_check) or [1.0])
    assert lifecycle.advance_official_terminal()["official_success"] is True
    assert lifecycle.advance_official_terminal()["official_success"] is True
    assert calls == [True]  # Success on the last native action takes precedence over exhaustion.


def test_interpolation_preserves_upstream_eight_ramp_two_hold_contract():
    lifecycle = object.__new__(EvaluationLifecycle)
    lifecycle.obs_manager = SimpleNamespace(collect_interval=10)
    lifecycle.robot_manager = Manager()
    sequence = lifecycle.interpolate({"left_arm_joint_state": {"position": [0.09, 0, 0.8, 0, 0, 0]}})
    assert len(sequence) == 10
    assert sequence[0]["left_arm_joint_state"]["position"][0] == pytest.approx(0.01)
    assert sequence[7]["left_arm_joint_state"]["position"][0] == pytest.approx(0.08)
    assert sequence[8]["left_arm_joint_state"]["position"][0] == pytest.approx(0.09)
    assert sequence[9] == sequence[8]


def test_every_reset_reinitializes_reward_after_task_reset_and_before_support():
    events = []

    class ParentTask:
        def reset(self, seed=None, options=None):
            events.append("task_reset_clears_checks")
            self.reward_manager.check_list = [[]]

    class Lifecycle(EvaluationLifecycle, ParentTask):
        def setup_scene(self):
            events.append("settle")

        def run_reward(self):
            events.append("register_reward")
            self.reward_manager.check_list[0].append("official_predicate")

        def get_score(self):
            events.append("register_score")

        def query_support_arm_traj(self, env_idx):
            events.append("support_init")

    lifecycle = object.__new__(Lifecycle)
    lifecycle.num_envs = 1
    lifecycle.interact = True
    lifecycle.step_lim = 1050
    lifecycle.scene_manager = SimpleNamespace(layout_manager=SimpleNamespace(
        set_saved_layout=lambda env_idx, layout: events.append("bind_layout"),
        instance_type_by_env=[{"mug_0": "rigid"}],
        get_scene_object=lambda env_idx, inst_name: object()))
    lifecycle.seed_manager = SimpleNamespace(get_seed_scene_info=lambda seed: {})
    lifecycle.obs_manager = SimpleNamespace(reset=lambda: events.append("observation_reset"))
    lifecycle.robot_manager = SimpleNamespace(set_origin_endpose=lambda: events.append("origin"),
                                             set_robot_init_state=lambda: events.append("controller_initial_state"))
    lifecycle.reward_manager = SimpleNamespace(check_list=[[]], final_check_list=[[]], trigger_check_list=[[]],
                                               init_state=lambda: events.append("predicate_initial_state"))
    lifecycle.capture_manager = SimpleNamespace(tiled_cameras=[])
    lifecycle._release_capture_products = lambda: events.append("release_previous_capture_products")
    for index in range(2):
        events.clear()
        lifecycle.reset(seed=[0])
        assert events == (["release_previous_capture_products"] if index else []) + [
            "bind_layout", "task_reset_clears_checks", "observation_reset", "settle", "origin",
            "controller_initial_state", "predicate_initial_state", "register_reward", "register_score", "support_init"]
        # The upstream reset appends new views; the next reset must release them first.
        lifecycle.capture_manager.tiled_cameras = [object()]
        assert len(lifecycle.reward_manager.check_list[0]) == 1
        assert lifecycle.success == [True] and lifecycle.end_flag == [False]


def _layout_reloading_lifecycle(events):
    """Model SceneManager.reload_env_scene: a reload of the saved layout renames every
    record from a monotone id counter while the CUDA pipeline keeps the round-0
    rigid objects registered under their original names."""
    registry = {}
    counter = {"value": 0}

    class Layout:
        def __init__(self):
            self.instance_type_by_env = [{}]

        def set_saved_layout(self, env_idx, layout):
            return None

        def load_saved_layout(self):
            self.instance_type_by_env[0] = {}
            for category, kind in (("mug_8", "rigid"), ("mug_5", "rigid"), ("cup_holder_0", "geometry")):
                counter["value"] += 1
                self.instance_type_by_env[0][f"{category}_{counter['value']}"] = kind

        def get_scene_object(self, env_idx, inst_name):
            return registry.get(inst_name)

    class ParentTask:
        def reset(self, seed=None, options=None):
            scene = self.scene_manager
            if self.sim is None:
                self.sim = object()
                scene.layout_manager.load_saved_layout()
                registry.update({name: object() for name, kind in scene.layout_manager.instance_type_by_env[0].items()
                                 if kind == "rigid"})
                scene.setup_scene = True
            events.append(("upstream_reset", scene.setup_scene))
            if not scene.setup_scene:
                scene.layout_manager.load_saved_layout()
            for name, kind in scene.layout_manager.instance_type_by_env[0].items():
                if kind == "geometry":
                    registry[name] = object()
            scene.setup_scene = False
            self.reward_manager.check_list = [[]]

    class Lifecycle(EvaluationLifecycle, ParentTask):
        def setup_scene(self):
            events.append("settle")

        def run_reward(self):
            self.reward_manager.check_list[0].append("official_predicate")

    lifecycle = object.__new__(Lifecycle)
    lifecycle.num_envs, lifecycle.sim = 1, None
    lifecycle.step_lim = 1050
    lifecycle.scene_manager = SimpleNamespace(layout_manager=Layout(), setup_scene=False)
    lifecycle.seed_manager = SimpleNamespace(get_seed_scene_info=lambda seed: {})
    lifecycle.obs_manager = SimpleNamespace(reset=lambda: None)
    lifecycle.robot_manager = SimpleNamespace(set_origin_endpose=lambda: None, set_robot_init_state=lambda: None)
    lifecycle.reward_manager = SimpleNamespace(check_list=[[]], final_check_list=[[]], trigger_check_list=[[]],
                                               init_state=lambda: None)
    lifecycle.capture_manager = SimpleNamespace(tiled_cameras=[])
    return lifecycle, registry


def test_later_resets_keep_the_layout_identities_the_upstream_reload_would_rename():
    events = []
    lifecycle, registry = _layout_reloading_lifecycle(events)
    lifecycle.reset(seed=[0])
    round0 = dict(lifecycle.scene_manager.layout_manager.instance_type_by_env[0])
    assert round0 == {"mug_8_1": "rigid", "mug_5_2": "rigid", "cup_holder_0_3": "geometry"}
    lifecycle.reset(seed=[0])
    assert lifecycle.scene_manager.layout_manager.instance_type_by_env[0] == round0
    assert set(round0) <= set(registry)
    # The first reset runs upstream's own post-initialization path; the bridge
    # restores that flag before every later reset of the same plant.
    assert [event for event in events if isinstance(event, tuple)] == [("upstream_reset", True)] * 2
    assert lifecycle.scene_manager.setup_scene is False and lifecycle.reward_registered is True


def test_reset_fails_closed_when_layout_records_no_longer_name_scene_objects():
    events = []
    lifecycle, _registry = _layout_reloading_lifecycle(events)
    lifecycle._keep_layout_identities = lambda: None
    lifecycle.reset(seed=[0])
    with pytest.raises(RobodojoError, match="no scene object after reset: mug_5_5, mug_8_4"):
        lifecycle.reset(seed=[0])
    assert "settle" not in events[events.index(("upstream_reset", False)):]
    assert lifecycle.reward_registered is False


def test_native_tick_rejects_terminal_before_any_reward_update():
    lifecycle = object.__new__(EvaluationLifecycle)
    lifecycle._official_terminal = {"official_success": True}
    lifecycle.end_flag = [True]
    with pytest.raises(RobodojoError, match="cannot accept"):
        lifecycle.native_tick({})


def test_close_is_idempotent(session):
    calls = []
    session.app.close = lambda: calls.append("app_close")
    session.close()
    session.close()
    assert calls == ["app_close"]
    assert session.environment.closed


@pytest.mark.parametrize("fault", [None, "delayed", "split_frames", "pose_lag", "timeout", "physics_leak"])
def test_render_barrier_joins_all_products_to_current_physics_without_stepping(tmp_path, fault):
    from ora.environment.backends.robodojo.camera_capture import synchronize_render_products

    state = {"renders": 0, "physics": 250}

    def render():
        state["renders"] += 1
        if fault == "physics_leak":
            state["physics"] += 1

    def stamp(index):
        late = index == 2 and (fault == "timeout" or (fault == "delayed" and state["renders"] == 1))
        return {"simulationTime": 0.8 if late else 1.0}

    def reference(index):
        split = fault == "split_frames" and index == 2 and state["renders"] == 1
        return {"referenceTimeNumerator": state["renders"] - int(split), "referenceTimeDenominator": 60}

    manager = SimpleNamespace(num_envs=1, tiled_cameras=[
        SimpleNamespace(_render_product_path=f"/camera/{index}",
                        _harness_camera_params=SimpleNamespace(get_data=lambda: {
                            "cameraViewTransform": np.eye(4).reshape(-1).tolist()}),
                        _harness_simulation_time=SimpleNamespace(get_data=lambda i=index: stamp(i)),
                        _harness_reference_time=SimpleNamespace(get_data=lambda i=index: reference(i)))
        for index in range(3)])
    def physical_poses():
        pose = np.diag([1., -1., -1., 1.])
        if fault == "pose_lag" and state["renders"] < 3:
            pose[0, 3] = .5
        return [pose] * 3

    kwargs = dict(manager=manager, render=render, physics_clock=lambda: state["physics"],
                  physical_poses=physical_poses,
                  simulation_time=lambda: 1.0, output_root=tmp_path, max_renders=3)
    if fault in {"timeout", "physics_leak"}:
        with pytest.raises((TimeoutError, ValueError)):
            synchronize_render_products(**kwargs)
    else:
        assert synchronize_render_products(**kwargs)["matched"] is True
        assert state["renders"] == (3 if fault == "pose_lag" else 2)
        assert state["physics"] == 250
    record = json.loads((tmp_path / "render-synchronization.jsonl").read_text())
    assert record["matched"] is (fault not in {"timeout", "physics_leak"})
    if fault == "timeout":
        assert len(record["samples"]) == 3


@pytest.mark.parametrize("stale", [False, True])
def test_calibration_rejects_rendered_pose_from_before_robot_motion(tmp_path, stale):
    from ora.environment.backends.robodojo.camera_capture import renderer_intrinsics

    parameters = {"cameraModel": "pinhole", "renderProductResolution": [640, 480],
                  "cameraProjection": np.diag([.9, 1.2, 1., 1.]).reshape(-1).tolist(),
                  "cameraAperture": [2., 2.], "cameraApertureOffset": [0., 0.],
                  "cameraViewTransform": np.eye(4).reshape(-1).tolist()}
    intrinsic = renderer_intrinsics(parameters, (640, 480))
    # The physical wrist has moved half a metre, while its render product
    # still reports the old optical transform. A static camera would miss this.
    physical = np.array([[.5 if stale else 0., 0., 0.]])
    camera = SimpleNamespace(get_intrinsics_matrix=lambda: intrinsic,
                             _prim_view=SimpleNamespace(get_world_poses=lambda **kw: (physical, [[1., 0., 0., 0.]])))
    view = SimpleNamespace(_render_product_path="/camera/wrist", camera_resolution=(640, 480),
                           _harness_camera_params=SimpleNamespace(get_data=lambda: parameters),
                           _harness_simulation_time=SimpleNamespace(get_data=lambda: {"simulationTime": 1.0}),
                           _harness_reference_time=SimpleNamespace(get_data=lambda: {"referenceTimeNumerator": 1,
                                                                                  "referenceTimeDenominator": 60}))
    session = RobodojoSession(config={}, output_root=tmp_path)
    session.environment = SimpleNamespace(
        camera_manager=SimpleNamespace(camera_names=[["wrist"]], cameras=[[camera]],
                                       get_camera_intrinsics=lambda *a, **kw: intrinsic),
        capture_manager=SimpleNamespace(tiled_cameras=[view]), use_fabric=True,
        sim=SimpleNamespace(scene=SimpleNamespace(env_origins=np.zeros((1, 3)))))
    if stale:
        with pytest.raises(RobodojoError, match="rendered camera pose differs"):
            session._calibration("wrist")
    else:
        session._calibration("wrist")
        assert session.calibration_readbacks["wrist"]["current_physical_pose_verified"] is True


# --- sparse motion v3 over the same fake SDK boundary ---------------------


SPARSE_SCHEMA = "ora.robodojo-sparse-motion-receipt.v1"


def test_sparse_v3_requires_the_motion_codec(session, tmp_path):
    with pytest.raises(RobodojoError, match="requires the motion_action_v3 codec"):
        session.execute(v3(session, tmp_path, targets={"left": pose([0.02, 0, 0.8])}))
    with pytest.raises(RobodojoError, match="requires the motion_action_v3 codec"):
        session.probe_motion(probe_request(session, tmp_path, [0.02, 0, 0.8])[1])
    with pytest.raises(RobodojoError, match="action_codec"):
        RobodojoSession(config={"action_codec": "target_state_v4"}, output_root=tmp_path / "x")


def test_sparse_gripper_targets_hold_pose_targets(sparse, tmp_path):
    before_targets = deepcopy(sparse.pose_targets)
    receipt = sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "close"}))["receipt"]
    assert receipt["execution_status"] == "reached" and receipt["executor"] == ["robodojo_sdk_gripper_actuator"]
    assert receipt["motion_strategy"] is None and receipt["arms"]["left"]["gripper_error"] == pytest.approx(0.0)
    assert receipt["arms"]["left"]["pose_error"] is None
    assert receipt["after"]["left"]["gripper"] == pytest.approx(0.0)
    assert receipt["command_invariant"] == {
        "schema": "ora.command-invariant.v2", "controlled_dimension": "gripper",
        "held_dimension": "eef_pose_target", "verified": True,
        "applied_targets": "per_physics_step_sdk_joint_position_buffer",
    }
    assert set(sparse.environment.commands[0]) == {"left_ee_joint_state"}
    assert sparse.pose_targets == before_targets and sparse.gripper_directives["left"] == 0.0
    assert sparse.world_mutation_actions == ["set_gripper"]
    both = sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "open", "right": "close"}))["receipt"]
    assert both["execution_status"] == "reached" and both["target_arms"] == ["left", "right"]
    assert set(sparse.environment.commands[-1]) == {"left_ee_joint_state", "right_ee_joint_state"}


def test_sparse_probe_answers_without_writing_and_refuses_a_bad_request(sparse, tmp_path, monkeypatch):
    from tests.support.agent.reach_check import planner_ik

    clock = sparse.environment.sim.sim.current_time_step_index
    # No planner bound: no pose is answered, and no IK runs.
    _observation, request = probe_request(sparse, tmp_path, [0.06, 0, 0.8])
    row = sparse.probe_motion(request)["results"][0]
    assert row["reachable"] is False and row["variants"][0]["failure"]["code"] == "no_planner"
    assert row["plan_ref"] is None and sparse.environment.commands == []
    assert sparse.environment.sim.sim.current_time_step_index == clock
    # A plan reference is refused unexecuted, as every move_pose is.
    from ora.contracts.refusal import EnvironmentRefusal
    with pytest.raises(EnvironmentRefusal, match="no move_pose"):
        sparse.execute(v3(sparse, tmp_path, plan_ref="plan-" + "0" * 32))
    requests = planner_ik(sparse, monkeypatch)
    _observation, request = probe_request(sparse, tmp_path, [0.05, 0, 3.0])
    with pytest.raises(RobodojoError, match="outside the action workspace"):
        sparse.probe_motion(request)
    with pytest.raises(RobodojoError, match="drifted"):
        sparse.probe_motion({**request, "extra": True})
    assert requests == []
    _observation, request = probe_request(sparse, tmp_path, [0.05, 0, 0.8])
    solve = sparse.motion_planner.solve_ik

    def leaking(ik_request):
        sparse.environment.sim.sim.current_time_step_index += 1
        return solve(ik_request)

    sparse.motion_planner.solve_ik = leaking
    with pytest.raises(RuntimeError, match="advanced physics"):
        sparse.probe_motion(request)
    assert sparse.is_terminal and sparse.abort_reason == "reach_check_advanced_physics"


def test_sparse_official_terminal_during_execution_snapshots_the_verdict(sparse, tmp_path):
    sparse.environment.terminal_at = 1
    receipt = trajectory_to(sparse, tmp_path, {"left": [0.10, 0.05, 0.80]})["receipt"]
    assert receipt["native_actions"] == 1 and receipt["terminal_after_action"] is True
    assert receipt["execution_status"] == "not_reached"
    assert sparse.persist_official_verdict()["official_success"] is True
    with pytest.raises(RobodojoError, match="cannot accept"):
        trajectory_to(sparse, tmp_path, {"left": [0.1, 0, 0.8]})
    assert sparse.persist_official_verdict()["official_success"] is True


def test_restart_round_isolates_targets_and_lifecycle(sparse, tmp_path):
    trajectory_to(sparse, tmp_path, {"left": [0.05, 0.05, 0.85]})
    sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "close"}))
    sparse.abort("operator_stop")
    assert sparse.is_terminal
    with pytest.raises(RobodojoError, match="advance by exactly one"):
        sparse.restart_round(2)
    with pytest.raises(RobodojoError, match="seed 0"):
        sparse.restart_round(1, seed=5)
    sparse.persist_official_verdict()
    with pytest.raises(RobodojoError, match="archived"):
        sparse.restart_round(1)
    (sparse.output_root / "official-verdict.json").unlink()
    reset = sparse.restart_round(1, seed=0)
    assert reset["schema"] == "ora.robodojo-round-reset.v1"
    assert reset["round_index"] == 1 and reset["seed"] == 0 and reset["reward_registered"] is True
    assert reset["artifact_sequences_preserved"] == {
        "observation_sequence": 2, "action_sequence": 2, "motion_probe_sequence": 0,
    }
    assert reset["public_environment_context"]["environment_id"] == "robodojo"
    assert sparse.environment.harness_round_index == 1 and sparse.environment.resets == 2
    assert sparse.round_index == 1 and not sparse.is_terminal and sparse.abort_reason is None
    assert sparse.world_mutation_sequence == 0
    assert sparse.pose_targets["left"][:3] == [0, 0, 0.8] and sparse.gripper_directives["left"] == 1.0
    assert sparse.last_safe_joints["left"] == sparse.home_joints["left"] == [0, 0, 0.8, 0, 0, 0]
    assert sparse.sim_steps == 0 and sparse.last_receipt is None
    assert json.loads((sparse.output_root / "round-resets/round-0001.json").read_text()) == reset
    observation, _ = sparse.observe(tmp_path / "observations")
    assert observation["round_index"] == 1 and observation["sequence"] == 2
    assert observation["previous_execution_receipt"] is None
    receipt = trajectory_to(sparse, tmp_path, {"left": [0.02, 0, 0.8]})["receipt"]
    assert receipt["round_index"] == 1 and receipt["action_id"] == "action-2"
    assert (sparse.output_root / "receipts/action-0002.json").exists()
    assert sparse.persist_official_verdict()["round_index"] == 1
    with pytest.raises(RobodojoError, match="integer"):
        sparse.restart_round(True)


def _second_round(sparse, tmp_path):
    """Drive one round to a verdict and reset into round 1."""

    trajectory_to(sparse, tmp_path, {"left": [0.05, 0.05, 0.85]})
    sparse.abort("operator_stop")
    sparse.persist_official_verdict()
    (sparse.output_root / "official-verdict.json").unlink()
    return sparse.restart_round(1, seed=0)


def test_a_second_round_starts_from_a_measured_restored_world(sparse, tmp_path):
    """The round-reset record proves the restore instead of asserting it.

    Every field below is read back from the plant or from this session's own
    carried state after the reset: the official evaluator was rebuilt, its
    counters and failure latch are back at their reset values, the layout the
    session was started against still hashes the same, both arms returned to
    the pose the session's first reset measured, and no per-episode channel
    kept a value.
    """

    reset = _second_round(sparse, tmp_path)
    isolation = reset["round_isolation"]

    assert isolation["schema"] == "ora.robodojo-round-isolation.v1"
    assert isolation["passed"] is True and isolation["failures"] == []
    assert isolation["checks"] == {
        "arm_home_restored": True,
        "carried_channels_cleared": True,
        "episode_counters_zeroed": True,
        "failure_latch_cleared": True,
        "layout_identity_unchanged": True,
        "official_lifecycle_rebound": True,
        "official_terminal_cleared": True,
    }
    measured = isolation["measured"]
    # reward_registered is measured on the plant, not written as a literal:
    # the round-reset record republishes exactly what the audit read.
    assert measured["reward_registered"] is True
    assert reset["reward_registered"] is measured["reward_registered"]
    assert measured["step_lim"] == measured["native_action_limit"] == 1050
    assert measured["take_action_cnt"] == 0 and measured["end_flag"] is False
    assert measured["failure_latch_clear"] is True
    assert measured["stale_channels"] == []
    assert measured["layout_sha256"] == measured["layout_sha256_at_start"]
    assert measured["max_home_position_error_m"] == 0.0
    assert measured["physics_steps_before_policy"] == reset[
        "physics_steps_before_policy"
    ]
    assert sorted(measured["cleared_channels"]) == sorted(
        session_module.PER_ROUND_CLEARED_CHANNELS
    )
    # And the world really is back: the same record is on disk for the join.
    assert json.loads(
        (sparse.output_root / "round-resets/round-0001.json").read_text()
    ) == reset


@pytest.mark.parametrize(
    "channel", sorted(session_module.PER_ROUND_CLEARED_CHANNELS)
)
def test_a_stale_value_in_any_carried_channel_refuses_the_round(
    sparse, tmp_path, monkeypatch, channel,
):
    """Every declared per-round channel is proved empty, one by one.

    The failure is an infrastructure fault -- the session records one and the
    round keeps a null official verdict -- not a task outcome.
    """

    stale = {
        "action_sequences": [{"parent": "stale"}],
        "abort_reason": "carried_stop",
        "boundary_reason": "carried_boundary",
        "fault_reason": "carried_fault",
        "last_observation_id": "robodojo:observation-0000",
        "last_receipt": {"action_id": "action-0"},
        "official_terminal": {"official_success": True},
        "sim_steps": 40,
        "world_mutation_actions": ["move_free_space"],
        "world_mutation_sequence": 3,
    }[channel]
    original = type(sparse)._bind_reset_state

    def leaking(self):
        state = original(self)
        setattr(self, channel, stale)
        return state

    monkeypatch.setattr(type(sparse), "_bind_reset_state", leaking)

    with pytest.raises(RobodojoError, match="not restore an isolated world"):
        _second_round(sparse, tmp_path)

    record = json.loads(
        (sparse.output_root / "round-resets/round-0001.json").read_text()
    )
    isolation = record["round_isolation"]
    assert isolation["passed"] is False
    assert isolation["measured"]["stale_channels"] == [channel]
    assert any(channel in failure for failure in isolation["failures"])
    # An infrastructure fault is recorded, so the round keeps a null official
    # verdict. When the leaked channel is fault_reason itself, the session
    # keeps the first reason rather than overwriting it.
    assert sparse.fault_reason == (
        stale if channel == "fault_reason"
        else "robodojo_round_reset_isolation_failed"
    )


@pytest.mark.parametrize(
    "field, value, failed_check",
    [
        ("take_action_cnt", [7], "episode_counters_zeroed"),
        ("end_flag", [True], "episode_counters_zeroed"),
        ("success", [False], "failure_latch_cleared"),
        ("_official_terminal", {"official_success": False},
         "official_terminal_cleared"),
    ],
)
def test_an_unrestored_official_lifecycle_refuses_the_round(
    sparse, tmp_path, monkeypatch, field, value, failed_check,
):
    """A contradictory round result never reaches the new round's policy.

    This is the defect class the guarantee exists for: an evaluator latch or
    episode counter that survives the reset would make the freshly reset world
    answer with the previous round's outcome.
    """

    plant = sparse.environment
    original = type(plant).reset

    def leaking(self, seed):
        original(self, seed)
        setattr(self, field, value)

    monkeypatch.setattr(type(plant), "reset", leaking)

    with pytest.raises(RobodojoError, match="not restore an isolated world"):
        _second_round(sparse, tmp_path)

    record = json.loads(
        (sparse.output_root / "round-resets/round-0001.json").read_text()
    )
    assert record["round_isolation"]["checks"][failed_check] is False


def test_a_changed_layout_file_refuses_the_round(sparse, tmp_path):
    """The world is only restored if it is restored from the *same* layout."""

    layout = Path(sparse.source["layout_file"]["path"])
    layout.write_text('{"objects": ["moved"]}', encoding="utf-8")

    with pytest.raises(RobodojoError, match="layout_identity_unchanged"):
        _second_round(sparse, tmp_path)

    record = json.loads(
        (sparse.output_root / "round-resets/round-0001.json").read_text()
    )
    measured = record["round_isolation"]["measured"]
    assert measured["layout_sha256"] != measured["layout_sha256_at_start"]


def test_an_arm_that_does_not_return_home_refuses_the_round(
    sparse, tmp_path, monkeypatch,
):
    """A plant that resets its evaluator but not its robot is not restored.

    The Harness writes the session-home joints before the upstream reset; a
    plant on which neither that write nor the reset moves the arm still fails
    round isolation, so the guard is unchanged.
    """

    manager = sparse.environment.robot_manager
    monkeypatch.setattr(
        type(manager), "reset_home", lambda self: None, raising=True,
    )
    monkeypatch.setattr(
        type(manager), "write_joint_state", lambda self, *a, **k: None, raising=True,
    )
    trajectory_to(sparse, tmp_path, {"left": [0.30, 0.20, 1.10]})
    sparse.abort("operator_stop")
    sparse.persist_official_verdict()
    (sparse.output_root / "official-verdict.json").unlink()

    with pytest.raises(RobodojoError, match="arm_home_restored"):
        sparse.restart_round(1, seed=0)

    record = json.loads(
        (sparse.output_root / "round-resets/round-0001.json").read_text()
    )
    measured = record["round_isolation"]["measured"]
    assert measured["max_home_position_error_m"] > record["round_isolation"][
        "tolerances"
    ]["home_position_m"]


def test_sparse_receipts_project_into_the_unified_envelope(sparse, tmp_path):
    from ora.agent.session_core.receipt_projection import (
        ReceiptProjectionContext, resolve_receipt_effect_projector,
    )

    receipt = trajectory_to(sparse, tmp_path, {"left": [0.02, 0, 0.8]})["receipt"]
    projector = resolve_receipt_effect_projector(receipt["schema"])
    assert projector is not None
    projected = projector(receipt, ReceiptProjectionContext(last_action_id=receipt["action_id"], last_action_public_name="move_pose"))
    assert projected["status"] == "reached" and projected["executor"] == ["robodojo_joint_trajectory_knots_linear_ff"]
    assert projected["motion_strategy"] == "joint_trajectory" and projected["detail"]["target_arms"] == ["left"]
    assert "control_trace" not in json.dumps(projected)


def test_joint_jump_bounds_outside_their_ranges_are_refused(tmp_path):
    with pytest.raises(RobodojoError, match="supported ranges"):
        RobodojoSession(config={"action_codec": "motion_action_v3", "global_joint_jump_limit_rad": 0.5},
                        output_root=tmp_path / "bad")


def test_sparse_gripper_close_converges_to_the_measured_jaw_floor(sparse, tmp_path):
    manager = sparse.environment.robot_manager
    for articulation in manager.robot_key:
        articulation.data.joint_pos_limits[0][6] = [0.0, 0.044]
        articulation.data.joint_pos_limits[0][7] = [0.0, 0.044]
    receipt = sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "close"}))["receipt"]
    assert receipt["execution_status"] == "reached" and receipt["native_actions"] == 1
    assert receipt["gripper_floors"]["left"] == pytest.approx(0.01 / 0.054)
    assert receipt["gripper_targets_effective"]["left"] == pytest.approx(0.01 / 0.054)
    assert receipt["gripper_directives"]["left"] == 0.0
    assert receipt["after"]["left"]["gripper"] == pytest.approx(0.01 / 0.054)
    assert receipt["arms"]["left"]["gripper_error"] == pytest.approx(0.0)
    assert manager.control_manager.prev_control[0]["left_ee_joint_state"]["position"] == [-0.01, -0.01]


def test_x5_command_target_reads_the_command_scale_not_a_finger_measurement():
    # Source identities: the command scale is robot_config.yml's
    # gripper_scale [-0.01, 0.044]; X5A.urdf limits joint7/joint8 to
    # [0, 0.044], 10 mm above the command origin, so a fully closed jaw
    # floors near 0.19 on the command scale rather than at 0.0.
    assert CLOSED_COMMAND_FLOOR == pytest.approx(0.01 / 0.054, abs=1e-6)
    assert gripper_command_target(1.0) == "open"
    assert gripper_command_target(0.0) == "close"
    # A future path publishing the executed floored command still reads close.
    assert gripper_command_target(CLOSED_COMMAND_FLOOR) == "close"
    assert gripper_command_target(0.5) == "unknown"
    assert gripper_command_target(None) == "unknown"
    assert gripper_command_target(True) == "unknown"
    assert gripper_command_target(float("nan")) == "unknown"


def test_proprioception_publishes_the_commanded_gripper_category_per_arm(sparse, tmp_path):
    """The session states each arm's directive; no consumer re-derives one."""
    manager = sparse.environment.robot_manager
    for articulation in manager.robot_key:
        articulation.data.joint_pos_limits[0][6] = [0.0, 0.044]
        articulation.data.joint_pos_limits[0][7] = [0.0, 0.044]
    payload, _paths = sparse.observe(tmp_path / "observations")
    effectors = payload["proprioception"]["end_effectors"]
    assert set(effectors) == {"left", "right"}
    for arm in ("left", "right"):
        assert effectors[arm]["gripper_command_target"] == "open"

    sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "close"}))
    effectors = sparse.observe(tmp_path / "observations")[0][
        "proprioception"]["end_effectors"]
    assert effectors["left"]["gripper_command_target"] == "close"
    assert effectors["right"]["gripper_command_target"] == "open"
    # The jaw is empty and fully closed, so the measurement rests on the
    # articulation's own floor while the written command stays at 0.0. The
    # published category follows the command, and a floor reading is published
    # as a measurement only: the observation still asserts no contact evidence.
    assert effectors["left"]["gripper"] == pytest.approx(CLOSED_COMMAND_FLOOR)
    assert effectors["left"]["gripper_command"] == 0.0
    assert "contact" in payload["forbidden_modalities_absent"]
    assert not [
        key for key in effectors["left"]
        if "contact" in key or "grasp" in key or "retention" in key
    ]

    sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "open"}))
    # Pin the fingers at the closed floor an obstructed open would leave them
    # at: the category is the directive, so it must read open regardless.
    manager.fingers["left_arm"] = [0.0, 0.0]
    effectors = sparse.observe(tmp_path / "observations")[0][
        "proprioception"]["end_effectors"]
    assert effectors["left"]["gripper"] == pytest.approx(CLOSED_COMMAND_FLOOR)
    assert effectors["left"]["gripper_command_target"] == "open"


def test_runtime_action_v3_set_gripper_publishes_the_same_category(session, tmp_path):
    receipt = session.execute(
        action(session, tmp_path, kind="set_gripper", value=0.0),
    )["receipt"]
    assert receipt["after"]["left"]["gripper_command_target"] == "close"
    assert receipt["after"]["right"]["gripper_command_target"] == "open"
    assert receipt["before"]["left"]["gripper_command_target"] == "open"
    effectors = session.observe(tmp_path / "observations")[0][
        "proprioception"]["end_effectors"]
    assert effectors["left"]["gripper_command_target"] == "close"
    assert effectors["right"]["gripper_command_target"] == "open"


def test_round_reset_writes_the_session_home_joints_before_the_upstream_reset(
    sparse, tmp_path, monkeypatch,
):
    """g004 classify_objects/build_tower: the upstream reset drives the arm home
    and cannot pass an obstruction on the home side (100 N m saturated, joints
    1-3 motionless), so round isolation failed. The Harness now writes the
    session-home joint state first and records it in the reset evidence."""

    manager = sparse.environment.robot_manager
    monkeypatch.setattr(type(manager), "reset_home", lambda self: None, raising=True)
    trajectory_to(sparse, tmp_path, {"left": [0.30, 0.20, 1.10]})
    moved = list(manager.joints["left_arm"])
    sparse.abort("operator_stop")
    sparse.persist_official_verdict()
    (sparse.output_root / "official-verdict.json").unlink()

    reset = sparse.restart_round(1, seed=0)

    assert reset["round_isolation"]["passed"] is True
    write = reset["pre_reset_home_joint_write"]
    assert write["method"] == "articulation_joint_state_write"
    left = write["arms"]["left"]
    assert left["joints_before"] == pytest.approx(moved)
    assert left["joint_distance_before_rad"] > 0.1
    assert left["joints_after_write"] == pytest.approx(left["home_joints"])
    assert manager.joints["left_arm"] == pytest.approx(left["home_joints"])
    record = json.loads((sparse.output_root / "round-resets/round-0001.json").read_text())
    assert record["pre_reset_home_joint_write"] == write


def test_round_reset_releases_the_previous_capture_products_before_upstream_recreates_them():
    events = []

    class Annotator:
        def __init__(self, name):
            self.name = name

        def detach(self, paths):
            events.append(("detach", self.name, tuple(paths)))

    class Product:
        def __init__(self, index):
            self.index = index

        def destroy(self):
            events.append(("destroy", self.index))

    views = [SimpleNamespace(
        _render_product_path=f"/rp/{index}", _annotators={"rgb": Annotator(f"rgb{index}")},
        _harness_camera_params=Annotator(f"params{index}"), _harness_simulation_time=Annotator(f"time{index}"),
        _harness_reference_time=Annotator(f"ref{index}"),
    ) for index in range(3)]
    manager = SimpleNamespace(
        tiled_cameras=list(views), tiled_render_products=[Product(index) for index in range(3)],
        annotator=[[], [], []], annotator_type=[["rgb"], ["rgb"], ["rgb"]], annotator_device=[[], [], []],
        _output_buffers={0: {}, 1: {}, 2: {}},
    )
    lifecycle = object.__new__(EvaluationLifecycle)
    lifecycle.capture_manager = manager
    lifecycle._release_capture_products()
    assert [event for event in events if event[0] == "destroy"] == [("destroy", 0), ("destroy", 1), ("destroy", 2)]
    assert ("detach", "params1", ("/rp/1",)) in events and ("detach", "rgb2", ("/rp/2",)) in events
    assert events.index(("detach", "rgb0", ("/rp/0",))) < events.index(("destroy", 0))
    assert manager.tiled_cameras == [] and manager.tiled_render_products == []
    assert manager.annotator == [] and manager.annotator_type == [] and manager.annotator_device == []
    assert manager._output_buffers == {}


def test_a_plant_that_cannot_report_its_lifecycle_refuses_the_round(
    sparse, tmp_path, monkeypatch,
):
    """Unreadable is refused, not assumed clean.

    A plant that cannot report its own episode lifecycle cannot prove it was
    restored, so the lifecycle checks fail rather than being skipped.
    """

    plant = sparse.environment
    original = type(plant).reset

    def leaking(self, seed):
        original(self, seed)
        del self.end_flag

    monkeypatch.setattr(type(plant), "reset", leaking)

    with pytest.raises(RobodojoError, match="not restore an isolated world"):
        _second_round(sparse, tmp_path)

    isolation = json.loads(
        (sparse.output_root / "round-resets/round-0001.json").read_text()
    )["round_isolation"]
    assert isolation["measured"]["reward_registered"] is False
    assert isolation["checks"]["official_lifecycle_rebound"] is False
    assert isolation["checks"]["episode_counters_zeroed"] is False
    assert isolation["checks"]["failure_latch_cleared"] is False


def test_source_manifest_attests_the_codec_the_worker_was_bound_to():
    """The manifest reports the resolved binding, never a compat-era literal.

    A RoboDojo cell that runs the motion codec used to attest
    ``runtime_action_v3`` because the worker returned a literal. The
    evaluator compares the manifest against the RunPlan's own codec, so the
    lie invalidated every landed episode at seal, long after the run.
    """

    bound = bridge.manifest_bindings({"action_codec": "motion_action_v3"}, "hang_mugs")
    assert bound["action_codec"] == "motion_action_v3"
    assert bound["official_native_action_limit"] == 800
    assert bound["sensor_profile"] == "rgbd_metric_v1"
    compat = bridge.manifest_bindings({"action_codec": "runtime_action_v3"}, "hang_mugs")
    assert compat["action_codec"] == "runtime_action_v3"


def test_source_manifest_refuses_an_unresolved_or_drifting_binding():
    """Every divergence is refused at worker startup, not discovered at seal."""

    with pytest.raises(bridge.RobodojoError):
        bridge.manifest_bindings({}, "hang_mugs")
    with pytest.raises(bridge.RobodojoError):
        bridge.manifest_bindings({"action_codec": "target_state_v4"}, "hang_mugs")
    for field, value in (
        ("sensor_profile", "rgb_only_v1"), ("physics_dt", 0.002), ("control_hz", 30),
        ("eval_seed", 3), ("layout_id", -1), ("layout_id", True), ("num_envs", 2),
        ("native_action_limit", 600),
    ):
        with pytest.raises(bridge.RobodojoError):
            bridge.manifest_bindings(
                {"action_codec": "motion_action_v3", field: value}, "hang_mugs",
            )
    # An unscheduled worker binds the RunPlan's one official layout;
    # its file and bytes are checked against the pinned Assets at start.
    assert bridge.manifest_bindings({"action_codec": "motion_action_v3", "layout_id": 2}, "hang_mugs")[
        "layout_id"] == 2
@pytest.mark.parametrize("upstream_renames", [False, True])
def test_repeated_reset_identity_is_measured_on_one_live_plant(
    tmp_path, upstream_renames,
):
    """The one scene fact two fresh workers can never establish.

    ``bridge._keep_layout_identities`` -- the workaround for upstream
    ``SceneManager.reload_env_scene`` regenerating every instance name from
    LayoutManager's monotone id counter -- runs only on a second reset of a
    *live* plant, i.e. only from ``restart_round``. The long infrastructure
    probe's two episodes are two fresh worker processes, so it never reaches
    this path; the per-task qualification layer does, and publishes both
    registrations so the comparison is evidence rather than an assumption.
    """

    plant = Plant()
    plant.rename_on_reset = upstream_renames
    instance = RobodojoSession(
        config={"workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}},
        output_root=tmp_path / "native",
        environment_factory=lambda config: (
            plant, SimpleNamespace(close=lambda: None), _fake_source_manifest(tmp_path),
        ),
    )
    instance._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    first = instance.start()["scene_registration"]
    assert first["instances"] == {"mug_0": "rigid", "rack_0": "geometry"}
    assert first["registered_instance_count"] == 2
    # The task's own registered limit, read from the environment rather than
    # restated: this is what the task layer proves in place of the official
    # terminal transition it cannot afford to witness.
    assert first["native_action_limit"] == plant.step_lim
    assert first["evaluator_registration"] == {
        "check_list": True, "final_check_list": False, "trigger_check_list": False,
    }

    repeated = instance.restart_round(1)["scene_registration"]
    if upstream_renames:
        assert repeated["instances"] == {"mug_1": "rigid", "rack_1": "geometry"}
        assert repeated != first
        assert (repeated["instance_identities_sha256"]
                != first["instance_identities_sha256"])
    else:
        assert repeated == first


def test_target_state_two_phases_share_one_consumed_observation(session, tmp_path):
    from ora.orchestrator.episode.action_executor import execute_runtime_action
    parent = action(session, tmp_path)
    parent.update(schema="ora-action.v4", action="target_state", arguments={
        "arm": "left", "pose": {"position_m": [.02, 0, .8], "orientation_xyzw": [0, 0, 0, 1]},
        "gripper": 0.,
    })
    receipt = execute_runtime_action(session, parent)
    assert [r["status"] for r in receipt["phase_receipts"]] == ["reached", "executed"]
    assert session.action_sequences == []
    assert session.last_observation_id is None
    with pytest.raises(RobodojoError, match="unconsumed"):
        execute_runtime_action(session, parent)


def test_sequence_rejects_changed_phase_replay_and_uncommitted_child(session, tmp_path):
    from ora.contracts.action import target_state_phases
    parent = action(session, tmp_path)
    parent.update(schema="ora-action.v4", action="target_state", arguments={"arm": "left", "gripper": 0.})
    session.begin_action_sequence(parent)
    phase = target_state_phases(parent)[0]
    changed = deepcopy(phase)
    changed["arguments"]["gripper"] = 1.
    with pytest.raises(RobodojoError, match="changed its committed"):
        session.execute(changed)
    session.execute(phase)
    with pytest.raises(RobodojoError, match="duplicated"):
        session.execute(phase)
    session.end_action_sequence(parent["action_id"])
    assert not session.action_sequences


def test_closed_skill_consumes_bound_children_and_cleans_up_on_fault(session, tmp_path):
    from ora.orchestrator.episode.action_executor import execute_skill_sequence
    parent = action(session, tmp_path)
    parent.update(schema="ora-skill-sequence-action.v1", action="execute_skill_sequence", arguments={
        "skill": "grasp_execute", "candidate_id": "grasp-test", "rationale": "source-bound test",
        "stop_before_commit": False, "sequence": ["approach", "close", "verify"], "resume_from": 0,
    })
    def drive(action, *, stage, stage_index, **kwargs):
        if stage == "close":
            raise RuntimeError("injected driver fault")
        return {"kind": "mutation", "action": {
            **action, "schema": "ora-action.v4", "action": "target_state",
            "action_id": f"{action['action_id']}:{stage_index:02d}-{stage}",
            "arguments": {"arm": "left", "pose": {"position_m": [.02, 0, .8], "orientation_xyzw": [0, 0, 0, 1]}},
        }}
    reports = []
    with pytest.raises(RuntimeError, match="injected"):
        execute_skill_sequence(session, parent, initial_observation={}, initial_observation_root=tmp_path,
                               drive_stage=drive, build_receipt=lambda **kw: reports.append(kw) or {},
                               capture_observation=lambda: ({"observation_id": "fresh"}, tmp_path))
    assert not session.action_sequences
    assert len(reports) == 1
    assert len(session.environment.commands) == 1


@pytest.mark.parametrize("corruption", ["observation_id", "bound_observation_sha256", "action_id", "schema", "codec"])
def test_sparse_skill_admission_rejects_drift_without_consuming_child(sparse, tmp_path, corruption):
    parent = v3(sparse, tmp_path, kind="set_gripper", targets={"left": "close"})
    child = deepcopy(parent)
    parent.update(schema="ora-skill-sequence-action.v1", action="execute_skill_sequence", arguments={
        "skill": "grasp_execute", "candidate_id": "grasp-test", "rationale": "source-bound test",
        "stop_before_commit": False, "sequence": ["approach", "close", "verify"], "resume_from": 0,
    })
    child["action_id"] = parent["action_id"] + ":01-close"
    sparse.begin_action_sequence(parent)
    changed = deepcopy(child)
    if corruption == "codec":
        sparse.action_codec = "runtime_action_v3"
    elif corruption == "schema":
        changed.update(schema="ora-action.v3", arguments={"arm": "left", "gripper": 0.0})
    else:
        changed[corruption] = "b" * 64
    with pytest.raises(RobodojoError):
        sparse.execute(changed)
    assert sparse.action_sequences[-1]["next"] == 0
    assert sparse.environment.commands == []
    sparse.action_codec = "motion_action_v3"
    sparse.execute(child)
    with pytest.raises(RobodojoError, match="duplicated"):
        sparse.execute(child)
    sparse.end_action_sequence(parent["action_id"])


def test_cuda_garment_prim_is_created_before_physics_and_retained_until_close():
    class Manager:
        spawnable_object_types = ['rigid', 'articulation', 'garment']
        def __init__(self):
            self.physics_started = False
            self.garments = {}
            self._garment_objects = [self.garments]
            self.created = 0
        def spawn_scene_objects(self, env_id, exclude_types=None):
            if 'garment' not in (exclude_types or []):
                assert not self.physics_started, 'cloth must exist before PhysX starts'
                self.garments[env_id] = SimpleNamespace(
                    sample_mesh_vertices=lambda: ('world-particles', 'mesh-particles', None, None),
                    get_world_pose=lambda: ('prim-position', 'prim-quaternion'))
                self.created += 1
        def clear_scene_objects(self, env_id, exclude_types=None):
            if 'garment' not in (exclude_types or []):
                self.garments.pop(env_id, None)
    manager = Manager()
    bridge._prepare_cuda_garments(manager, 1)
    original = manager.garments[0]
    assert original.sample_mesh_vertices() == ('world-particles', 'mesh-particles', 'prim-position', 'prim-quaternion')
    manager.physics_started = True
    for _ in range(2):
        manager.clear_scene_objects(0, exclude_types=['rigid', 'articulation'])
        manager.spawn_scene_objects(0, exclude_types=['rigid', 'articulation'])
        assert manager.garments[0] is original
    assert manager.created == 1
    manager.clear_scene_objects(0)
    assert manager.garments == {}


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_sdk_limited_close_keeps_intent_through_observation_and_receipt(sparse, tmp_path, monkeypatch, arm):
    """Recorded SDK witness: first q=.044, requested q=-.01 -> drive q=.0332.

    An obstructed second finger differs; it must not become the limiter's
    reference or turn the close directive into an unknown/open command.
    """
    from ora.agent.context.observation.public_facts import public_end_effectors
    from ora.agent.session_core.receipt_projection import (
        ReceiptProjectionContext, resolve_receipt_effect_projector,
    )
    from ora.agent.context.causal_value import causal_provider_value
    manager = sparse.environment.robot_manager
    robot = sparse.robots[arm]
    key = manager.process_name(robot.gripper_name)
    for articulation in manager.robot_key:
        articulation.data.joint_pos_limits[0][6:] = [[0., .044], [0., .044]]
    native_tick = sparse.environment.native_tick
    measured = [.044, .0345]
    manager.fingers[robot.arm_name] = measured[:]

    def obstructed_tick(control):
        # Model the SDK postprocessor, not the pre-SDK command buffer.
        resolved = deepcopy(control)
        if key in resolved:
            requested = resolved[key]['position'][0]
            resolved[key]['position'] = [max(measured[0] - .2 * .054,
                min(measured[0] + .2 * .054, requested))] * 2
        result = native_tick(resolved)
        manager.fingers[robot.arm_name] = measured[:]
        return result

    monkeypatch.setattr(sparse.environment, 'native_tick', obstructed_tick)
    record = sparse.execute(v3(sparse, tmp_path, kind='set_gripper', targets={arm: 'close'}))['receipt']
    assert record['execution_status'] == 'not_reached'
    # The blocked jaws never move, so the shared stall rule ends the command
    # after three still native actions instead of the 20-action budget.
    assert record['failure_code'] == 'gripper_stalled_before_target'
    assert record['native_actions'] == 3
    assert record['arms'][arm]['gripper_outcome']['sdk_limited_drive_target_normalized'] == pytest.approx(.8)
    # The second finger stands at the limited target (.0345 m ~ .82), so the
    # outcome names the primary-relative hold beside both finger readings.
    assert record['arms'][arm]['gripper_outcome']['primary_finger_limit_holds_close'] is True
    assert record['arms'][arm]['gripper_outcome']['measured_finger_displacements_m'] == pytest.approx(measured)
    assert record['after'][arm]['gripper_command_target'] == 'close'
    feedback = record['after'][arm]['gripper_feedback']
    assert feedback['requested_opening_normalized'] == 0.
    assert feedback['drive_target_opening_normalized'] == pytest.approx(.8)
    # E11G12-001: both fingers' mean, not the pinned first finger's 1.0.
    assert feedback['measured_opening_normalized'] == pytest.approx((sum(measured) / 2 + .01) / .054)
    assert feedback['finger_lateral_offset_m'] == pytest.approx((.044 - .0345) / 2)
    assert feedback['effective_target_opening_normalized'] == pytest.approx(.01/.054)
    assert feedback['arrival_check']['tolerance_normalized'] == .03
    assert feedback['arrival_check']['within_tolerance'] is False
    # Every capability except the 0059 delivery rules: under
    # within_payload_deduplication_v1 this feedback is read from the current
    # facts instead (tests/ora/agent/context/delivery/test_payload_dedup.py).
    from ora.agent.session_core.effect_delivery import DELIVERY_CAPABILITIES
    surface = ReceiptProjectionContext(None, 'set_gripper', lambda name: name not in DELIVERY_CAPABILITIES)
    public = causal_provider_value(resolve_receipt_effect_projector(record['schema'])(record, surface))['detail']
    assert public['gripper_feedback'][arm]['requested_command'] == 'close'
    assert public['gripper_feedback'][arm]['drive_target_opening_normalized'] == pytest.approx(.8)
    assert public['acceptance_checks'][0]['tolerance']['opening_normalized'] == .03
    assert public['execution_window'] == {'native_actions_used': 3, 'native_action_budget': 20}
    payload, _ = sparse.observe(tmp_path / 'observations')
    state = public_end_effectors(payload['proprioception'], 'arx_x5_link6')[arm]['gripper']
    assert state['commanded_opening_normalized'] == 0.
    assert state['feedback'] == feedback
    assert payload['controller_feedback']['gripper_drive_limit']['maximum_normalized_offset'] == .2
    damaged = deepcopy(record)
    del damaged['after'][arm]['gripper_feedback']['effective_target_opening_normalized']
    with pytest.raises(ValueError, match='fields differ'):
        resolve_receipt_effect_projector(record['schema'])(damaged, surface)
    damaged = deepcopy(record)
    del damaged['after'][arm]['gripper_feedback']
    with pytest.raises(ValueError, match='both arm gripper snapshots'):
        resolve_receipt_effect_projector(record['schema'])(damaged, surface)
    # Next pose holds the already applied target, and the original intent.
    held = deepcopy(manager.control_manager.prev_control[0][key])
    next_record = trajectory_to(sparse, tmp_path, {arm: [.01, 0, .8]})['receipt']
    assert manager.control_manager.prev_control[0][key] == held
    assert next_record['after'][arm]['gripper_feedback']['requested_command'] == 'close'
    assert next_record['acceptance_checks'][0]['tolerance'] == {'position_m': .01, 'orientation_rad': .1}


def test_legacy_dojo_close_uses_physical_floor_and_pose_checks(session, tmp_path):
    manager = session.environment.robot_manager
    for articulation in manager.robot_key:
        articulation.data.joint_pos_limits[0][6:] = [[0., .044], [0., .044]]
    record = session.execute(action(session, tmp_path, kind='set_gripper'))['receipt']
    assert record['motion_status'] == 'reached'
    assert record['native_actions'] == 1
    assert record['acceptance_checks'][0]['arms']['left']['within_tolerance']['opening_normalized'] is True
    record = session.execute(action(session, tmp_path))['receipt']
    assert record['acceptance_checks'][0]['stage'] == 'final_endpoint'


def test_one_copy_of_the_arm_buffers_gives_what_the_sdk_readers_give(sparse, tmp_path):
    """``arm_readings``: the joints, link pose and applied targets the separate SDK reads return, in one copy."""
    from ora.environment.backends.robodojo.common import arm_readings, measured_pose
    from ora.environment.backends.robodojo.native_sequence import applied_velocity_targets

    sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "close"}))
    manager = sparse.environment.robot_manager
    manager.joints["right_arm"] = [0.31, -0.12, 0.74, 0.1, -0.2, 0.3]
    manager.fingers["right_arm"] = [0.021, 0.019]
    readings = arm_readings(manager, sparse.robots)
    assert list(readings) == ["left", "right"] == list(sparse.robots)
    for arm, robot in sparse.robots.items():
        articulation = manager.robot_key[manager.robot_list.index(robot)]
        reading = readings[arm]
        assert all(value.dtype == np.float64 for value in reading.values())
        assert reading["joints"][robot.arm_joint_indices].tolist() == list(manager.get_joint(robot)[0])
        assert reading["joints"][robot.gripper_joint_indices].tolist() == list(
            manager.get_end_effector_real_val(robot)[0])
        assert reading["position_targets"].tolist() == np.asarray(articulation.data.joint_pos_target[0]).tolist()
        assert reading["velocity_targets"].tolist() == np.asarray(articulation.data.joint_vel_target[0]).tolist()
        assert [*reading["link_wxyz"][:3], *reading["link_wxyz"][4:], reading["link_wxyz"][3]] == measured_pose(
            manager, robot)
    # The readers take the step's copy or make their own, with the same result.
    assert sparse._proprioception(readings) == sparse._proprioception()
    assert sparse._applied_position_targets(readings) == sparse._applied_position_targets()
    assert applied_velocity_targets(sparse, readings) == applied_velocity_targets(sparse)
    # The link pose is made relative to the environment origin as get_link_pose makes it.
    manager.scene.env_origins = np.array([[2.0, -1.0, 0.5]])
    shifted = arm_readings(manager, sparse.robots)
    expected = readings["left"]["link_wxyz"] - [2.0, -1.0, 0.5, 0, 0, 0, 0]
    assert shifted["left"]["link_wxyz"].tolist() == expected.tolist()
    manager.joints["left_arm"][0] = float("nan")
    with pytest.raises(RobodojoError, match="joint measurement is not finite"):
        sparse._proprioception()
