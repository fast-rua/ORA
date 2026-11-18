"""The toolkit host runs on any embodiment; what it hands a RoboDojo tool does not change.

A surface without ``toolkit_synchronous_execution_v1`` has the environment plan its moves. Everything the host learned
for that is a key only such a surface gets: a context without one is RoboDojo's, as every sealed run recorded it.
"""
import json

import pytest

from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.agent.model_services.clients import Sam3ApiSegmenter
from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.state.deltas import read_rows
from ora.agent.toolkit.host import motion as toolkit_motion
from ora.agent.toolkit.host.runtime.factory import build_tool_runtime
from ora.agent.toolkit.host.runtime.inputs import freeze
from ora.agent.toolkit.host.observe import configuration as observe_configuration
from ora.agent.toolkit.host.runtime.tool_data import configuration, grasp_configuration, grounding_cameras
from tests.support.agent.stack_parity import SURFACES as STAND_INS, on_toolkit, stand_in
from tests.support.agent.toolkit_atomic import complete_cameras
from tests.support.agent.toolkit_recording import proposal
from tests.support.agent.toolkit_tools import V5_GRANTS, install, make_host, use_toolkit_v5

SURFACES = ("libero", "robotwin", "robocasa", "robolab", "yam")
SURFACE_CAMERAS = {STAND_INS[key].profile: STAND_INS[key].cameras[0] for key in SURFACES}
#: What RoboDojo's tool context holds, key by key: a new key here is a new read for a sealed run to miss.
ROBODOJO_MOTION_KEYS = {
    "capabilities", "examples", "global_planner", "held", "jaw_offsets", "jaws", "joint_trajectory", "local_axes",
    "mutation_refusals", "preview_hands", "reach_available", "view_config", "workspace_bounds_m"}
ROBODOJO_OBSERVE_KEYS = {
    "cameras", "current_step", "default_views", "history", "measured_jaw", "perception_feedback", "points",
    "public_facts", "retention", "rgb_resolution"}


@pytest.fixture
def host(request, tmp_path, monkeypatch):
    return on_toolkit(stand_in(tmp_path / "run", request.param, monkeypatch))


@pytest.fixture
def robodojo(tmp_path, monkeypatch):
    host, client, _ = make_host(tmp_path / "run", monkeypatch)
    complete_cameras(host)
    use_toolkit_v5(host)
    install(host)
    try:
        yield host
    finally:
        client.close()


def frozen(host):
    return freeze(host, BlobStore(host.control_root.parent, writable=True))[0]


@pytest.mark.parametrize("host", SURFACES, indirect=True)
def test_every_surface_freezes_its_tool_context(host):
    config = frozen(host)
    assert config["motion"]["environment_plans"] is True
    assert config["motion"]["joint_trajectory"] is False


@pytest.mark.parametrize("host", SURFACES, indirect=True)
def test_every_advertised_tool_runs_on_the_toolkit(host):
    advertised = {item["name"] for item in host.surface.definitions}
    assert advertised <= host._toolkit_installed_tools
    assert advertised == set(host._toolkit_handlers)


@pytest.mark.parametrize("host", SURFACES, indirect=True)
def test_tool_state_is_seeded_journaled_and_reset(host):
    run, access = host.control_root.parent, host.toolkit_state_access
    assert access.read("tool.marks") == {} and access.read("tool.held.right") is None
    arguments = {"view": SURFACE_CAMERAS[host.surface.profile_id], "point_ids": ["block"], "points_uv": [[64, 64]]}
    host.invoke_with_context("mark_point", arguments, call_context=proposal(host, "mark_point", arguments))
    assert set(access.read("tool.marks")) == {"block"}
    host.reset_round(1)
    assert access.read("tool.marks") == {}
    rows = read_rows(run)
    assert [row["context"].get("operation", row["context"]["kind"]) for row in rows if row["kind"] == "begin"] == [
        "initialize", "parent", "reset"]
    written = [row for row in rows if row["kind"] == "write" and row["context"]["kind"] == "parent"]
    assert len(written) == 1 and written[0]["context"]["tool_call_id"] == "call-0"
    assert written[0]["context"]["model_call_id"] == "model-0"
    assert not (run / "state/tool-deltas.gaps.jsonl").exists()


@pytest.mark.parametrize("host", ["libero"], indirect=True)
def test_a_tool_that_writes_no_state_still_has_its_parent_in_the_journal(host):
    """A later parent's state is the rows before its ``begin``, so each call needs one."""
    host.invoke_with_context("observe", {}, call_context=proposal(host, "observe", {}))
    assert [row["kind"] for row in read_rows(host.control_root.parent)][-2:] == ["begin", "end"]


@pytest.mark.parametrize("host", ["libero"], indirect=True)
def test_a_surface_the_environment_plans_for_has_no_execution_ledger(host):
    assert host.toolkit_state_store.exec_ledger is None
    assert not (host.control_root.parent / "state/exec-ledger.jsonl").exists()


@pytest.mark.parametrize("host", ["libero"], indirect=True)
def test_a_call_with_no_recorded_parent_freezes_the_journal_and_keeps_its_result(host):
    """Evidence fails, the tool does not: nothing says whose write this was, so nothing after it is claimed."""
    arguments = {"view": "agentview", "point_ids": ["block"], "points_uv": [[64, 64]]}
    assert host.invoke("mark_point", arguments)["status"] == "solved"
    assert set(host.toolkit_state_access.read("tool.marks")) == {"block"}
    gaps = (host.control_root.parent / "state/tool-deltas.gaps.jsonl").read_text()
    assert "no trusted parent" in gaps
    assert host.toolkit_state_store.tool_journal.watermark()["complete"] is False


@pytest.mark.parametrize("host,offsets", [("libero", {"right": [0.0, 0.0, 0.0]}), ("robolab", {"right": [0.0, 0.0, 0.0]}),
                                          ("robotwin", {"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0]}),
                                          ("yam", {})], indirect=["host"])
def test_a_grip_centre_frame_has_zero_offset_and_a_frame_without_a_rule_has_none(host, offsets):
    motion = toolkit_motion.configuration(host)
    assert motion["jaw_offsets"] == offsets
    assert bool(motion["local_axes"]) is (host.surface.view_config.get("controlled_frame_convention") is not None)


@pytest.mark.parametrize("host", ["yam"], indirect=True)
def test_a_surface_without_a_controlled_frame_has_no_jaws_and_plans_no_grasp(host):
    assert host.grasp_backend is not None  # the runtime's default, on every surface
    assert toolkit_motion.configuration(host)["jaws"] is None
    assert "grasp" not in configuration(host)


@pytest.mark.parametrize("host,cameras", [("libero", ["agentview", "wrist"]), ("robocasa", ["agentview", "wrist"]),
                                          ("robolab", ["agentview", "wrist"]),
                                          ("robotwin", ["head_camera", "front_camera", "left_camera", "right_camera"])],
                         indirect=["host"])
def test_grounding_takes_the_cameras_of_the_surface(host, cameras):
    assert configuration(host)["grounding_cameras"] == cameras


def test_grounding_cameras_are_robodojo_s_where_a_sealed_surface_names_none():
    class Sealed:
        view_config = None

        @staticmethod
        def supports(name):
            return False

    assert grounding_cameras(Sealed) == ["head", "left_wrist", "right_wrist"]


@pytest.mark.parametrize("host,compiler", [("libero", "grip_site"), ("robocasa", "grip_site"), ("robolab", "robotiq")],
                         indirect=["host"])
def test_a_single_arm_surface_binds_its_one_compiler_and_camera_pair(host, compiler):
    grasp = grasp_configuration(host)
    assert set(grasp["compilers"]) == set(grasp["scene_cameras"]) == {"right"}
    assert compiler in grasp["compilers"]["right"]["id"]
    assert grasp["scene_cameras"]["right"] == ["agentview", "wrist"]


@pytest.mark.parametrize("host", ["robotwin"], indirect=True)
def test_a_dual_arm_surface_keeps_both_compilers(host):
    grasp = grasp_configuration(host)
    assert grasp["compilers"]["left"]["id"] == "contact_graspnet_to_robotwin_aloha_left_grip_center_v3"
    assert grasp["scene_cameras"]["right"] == ["head_camera", "right_camera"]


@pytest.mark.parametrize("host,arms", [("libero", ["right"]), ("robolab", ["right"]), ("robotwin", None), ("yam", None)],
                         indirect=["host"])
def test_observe_says_which_arm_a_one_armed_surface_has(host, arms):
    assert observe_configuration(host)[0].get("arms") == arms


@pytest.mark.parametrize("host,named", [("robocasa", True), ("robolab", False)], indirect=["host"])
def test_only_a_surface_with_a_base_has_move_base_in_its_tool_context(host, named):
    (host.control_root / "policy-task-check.json").write_text(json.dumps({"terminal": True}))
    refusals = toolkit_motion.configuration(host)["mutation_refusals"]
    assert ("move_base" in refusals) is named and refusals["wait_ticks"]["code"] == "episode_terminal"


#: The registered surfaces that moved to the toolkit, where the environment still plans each move.
MOVED = ("yam", "robocasa", "robolab", "libero", "robotwin")


@pytest.mark.parametrize("key", MOVED)
def test_the_factory_runs_a_moved_surface_on_the_toolkit_with_the_callbacks_the_coordinator_gives_it(
        tmp_path, monkeypatch, key):
    published = stand_in(tmp_path / "run", key, monkeypatch)
    monkeypatch.setattr(Sam3ApiSegmenter, "require_ready", lambda self: {"fixture": True})  # built for the surface
    host = build_tool_runtime(
        control_root=published.control_root, manual_path=published.manual_path, event_path=published.event_path,
        workspace_bounds_m=published.workspace_bounds_m, allowed_tools=None, state_root=published.state_root,
        profile_id=STAND_INS[key].profile, component_overrides={}, align_plan_callback=published.align_plan_callback,
        motion_probe_callback=published.motion_probe_callback)
    assert host.surface.supports("toolkit_only_v1") and toolkit_motion.environment_plans(host.surface)
    assert host.invoke("wait_ticks", {"ticks": 2})["status"] == "submitted"
    assert (tmp_path / "run/native/toolkit-io/tool-inputs.jsonl").is_file()


def test_robodojo_gets_no_new_key(robodojo):
    config = frozen(robodojo)
    assert set(config["motion"]) == ROBODOJO_MOTION_KEYS
    assert config["grounding_cameras"] == ["head", "left_wrist", "right_wrist"]
    assert set(config["grasp"]["compilers"]) == {"left", "right"}
    assert set(observe_configuration(robodojo)[0]) - {"step_range", "history_error", "parent_boundaries_only"} == (
        ROBODOJO_OBSERVE_KEYS)


def test_robodojo_keeps_its_jaw_centre_from_the_descriptor(robodojo):
    offsets = toolkit_motion.configuration(robodojo)["jaw_offsets"]
    assert set(offsets) == {"left", "right"} and offsets["left"] == offsets["right"]
    assert offsets["left"][0] == pytest.approx(0.1501)


def test_robodojo_tool_state_stays_journaled(robodojo):
    assert robodojo.toolkit_state_store.tool_journal is not None
    assert (robodojo.control_root.parent / "state/tool-deltas.jsonl").is_file()


@pytest.mark.parametrize("host", ["libero"], indirect=True)
def test_each_tool_may_hand_over_only_its_own_action(host):
    control, _, _ = host._current()
    action = {"schema": "ora-motion-action.v3", "action": "wait", "action_id": f"action-{control['generation']}",
              "observation_id": control["observation_id"], "bound_observation_sha256": control["observation_sha256"],
              "arguments": {"ticks": 3}}
    with pytest.raises(Exception, match="motion action boundary mismatch"):
        toolkit_motion.submit_adapter(host, "set_gripper")({"action": action}, {}, None)
    assert not (host.control_root / "submitted-action.json").exists()
    toolkit_motion.submit_adapter(host, "wait_ticks")({"action": action}, {}, None)
    assert json.loads((host.control_root / "submitted-action.json").read_text())["action"] == "wait"
    assert host._submitted_mutations == ["wait"]


def test_a_shipped_toolkit_surface_is_not_environment_planned():
    for profile in ("robodojo_placement_toolkit_v5", "robodojo_placement_toolkit_v6"):
        surface = resolve_tool_surface(profile, toolkit_grants=V5_GRANTS if profile.endswith("v5") else None)
        assert toolkit_motion.environment_plans(surface) is False
