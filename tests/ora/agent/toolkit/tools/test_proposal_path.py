"""Where the environment plans a move, a toolkit tool proposes the action and answers as it did beside its handler.

Each call runs on a stand-in host on the toolkit over one published observation; the ``submitted-action.json`` it
writes and the model-visible result are checked against the digests in ``tests/data/toolkit-goldens``, recorded while
the runtime handler answered each call alike. What differed by design until the grasp and reach work lands is named in
``ADVISORY`` and left out. A context without ``environment_plans``, every sealed RoboDojo one, keeps the tool
executing, and reads nothing the proposal path added.
"""
import json
from types import SimpleNamespace

import pytest

from ora.agent.toolkit.interface import dry_run
from ora.agent.toolkit.interface.types import Response, canonical
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.tools.move_pose import _context, _smooth
from ora.agent.toolkit.tools.move_pose.tool import measured_states, surface_targets
from ora.agent.toolkit.tools.recover_pose import tool as recover_pose
from ora.agent.toolkit.tools.set_gripper import tool as set_gripper
from ora.agent.toolkit.tools.wait_ticks import tool as wait_ticks
from ora.agent.toolkit.host.motion import admitted
from ora.agent.error_classes import ToolError
from ora.contracts.action_plan import is_request
from tests.support.agent.stack_parity import on_toolkit, stand_in
from tests.support.agent.toolkit_goldens import Goldens
from tests.support.agent.toolkit_recording import proposal

GOLDEN = Goldens(__file__)

LIBERO, ROBOTWIN, ROBOCASA, ROBOLAB, YAM = SURFACES = ("libero", "robotwin", "robocasa", "robolab", "yam")
SINGLE_ARM = (LIBERO, ROBOCASA, ROBOLAB)
#: The published annotations of the IK reference, left out: legacy's v2 generation is not ported (plan row 11).
ADVISORY = frozenset({"reach_along_request", "nearest_reachable_orientation", "jaw_symmetry"})
DELTA = {"position_mode": "delta", "translation_m": [0.0, 0.0, 0.01], "position_frame": "world",
         "orientation_mode": "hold_current", "interaction": "precision"}
HOLD = {"position_mode": "hold_current", "orientation_mode": "axis_angle_delta", "rotation_axis": "z",
        "rotation_angle_deg": 10.0, "orientation_frame": "end_effector"}
SPEEDS = ({"speed": "normal"}, {"speed": "fast", "slow_approaching": True}, {"speed_m_s": 0.05}, {"speed": None})


def flat(position):
    return [("move_pose", DELTA), ("move_pose", HOLD), ("move_to", {"position_m": position}),
            ("rotate_wrist", {"angle_deg": 15.0}), ("rotate_pitch", {"angle_deg": -10.0}),
            ("set_gripper", {"action": "open"}), ("set_gripper", {"action": "close"}),
            ("set_gripper", {"action": "close", "target_ref": "the cup"}), ("wait_ticks", {"ticks": 4}),
            ("recover_pose", {"strategy": "previous"}), ("recover_pose", {"strategy": "home"})]


def sparse(position):
    return [("move_pose", {"targets": {"right": DELTA}}), ("move_pose", {"targets": {"left": HOLD, "right": DELTA}}),
            ("move_to", {"targets": {"left": {"position_m": position}}}),
            ("rotate_wrist", {"targets": {"right": 15.0}}), ("rotate_pitch", {"targets": {"left": -10.0}}),
            ("set_gripper", {"targets": {"left": "open"}}), ("set_gripper", {"targets": {"left": "close", "right": "open"}}),
            ("set_gripper", {"targets": {"right": "close"}, "target_ref": "the cup"}), ("wait_ticks", {"ticks": 4}),
            ("recover_pose", {"targets": {"left": "previous", "right": "home"}})]


def cases():
    for profile in SURFACES:
        for name, arguments in (flat if profile in SINGLE_ARM else sparse)([-0.1, 0.05, 0.95]):
            if profile == YAM and "target_ref" in arguments:
                continue  # refused on both, below
            yield pytest.param(profile, name, arguments, id=f"{profile}-{name}-{json.dumps(arguments, sort_keys=True)[:40]}")


def call(host, name, arguments):
    return host.invoke_with_context(name, arguments, call_context=proposal(host, name, arguments))


def visible(result):
    def drop(value):
        if isinstance(value, dict):
            return {key: drop(item) for key, item in value.items() if key not in ADVISORY}
        return value

    return drop({key: value for key, value in result.items() if key != "_runtime"})


def submitted(host):
    return json.loads((host.control_root / "submitted-action.json").read_text())


@pytest.fixture
def host(request, tmp_path, monkeypatch):
    return on_toolkit(stand_in(tmp_path / "run", request.param, monkeypatch))




@pytest.fixture
def golden(request):
    return GOLDEN.checker(request)


def answer(host, result):
    """What the model gets back from a call on ``host``, the action it proposed and the mutations it counted."""
    return {"result": visible(result), "runtime": result["_runtime"], "mutations": host._submitted_mutations,
            "action": submitted(host)}




@pytest.mark.parametrize("host,name,arguments", list(cases()), indirect=["host"])
def test_a_call_proposes_the_action_and_answers_as_beside_the_handler(host, golden, name, arguments):
    if name not in host.allowed_tools:
        pytest.skip(f"{name} is not on this surface")
    result = call(host, name, arguments)
    golden("call", answer(host, result), host.control_root.parent)


@pytest.mark.parametrize("speed", SPEEDS)
@pytest.mark.parametrize("host", [LIBERO, ROBOCASA, ROBOLAB], indirect=True)
@pytest.mark.parametrize("name,arguments", [("move_pose", DELTA), ("move_to", {"position_m": [-0.1, 0.05, 0.95]}),
                                            ("rotate_wrist", {"angle_deg": 15.0}), ("rotate_pitch", {"angle_deg": 5.0})])
def test_a_flat_surface_carries_the_speed_in_the_action(host, golden, speed, name, arguments):
    if name not in host.allowed_tools:
        pytest.skip(f"{name} is not on this surface")
    call(host, name, {**arguments, **speed})
    golden("action", submitted(host), host.control_root.parent)


@pytest.mark.parametrize("speed", SPEEDS)
@pytest.mark.parametrize("host", [ROBOTWIN], indirect=True)
def test_a_sparse_surface_carries_the_speed_in_the_action(host, golden, speed):
    arguments = {"targets": {"right": DELTA}, **speed}
    call(host, "move_pose", arguments)
    golden("action", submitted(host), host.control_root.parent)


@pytest.mark.parametrize("arguments,field", [
    ({"speed": "reckless"}, "speed"), ({"speed": "fast", "speed_m_s": 0.1}, "speed_m_s"),
    ({"speed": "careful", "slow_approaching": True}, "slow_approaching")])
@pytest.mark.parametrize("host", [LIBERO], indirect=True)
def test_a_refused_speed_is_refused_with_its_field(host, golden, arguments, field):
    host._validate_public_arguments = lambda *a: None  # past the schema
    with pytest.raises(ToolError) as refused:
        call(host, "move_pose", {**DELTA, **arguments})
    golden("refusal", refused.value.as_dict(), host.control_root.parent)
    assert refused.value.field == field


@pytest.mark.parametrize("name,arguments", [("move_pose", DELTA), ("set_gripper", {"action": "close"}),
                                            ("recover_pose", {"strategy": "home"})])
@pytest.mark.parametrize("host", SINGLE_ARM, indirect=True)
def test_a_flat_surface_refuses_the_targets_wrapper(host, name, arguments):
    host._validate_public_arguments = lambda *a: None  # past the schema
    with pytest.raises(ToolError, match="takes flat arguments without a targets wrapper") as refused:
        call(host, name, {"targets": {"right": arguments}})
    assert refused.value.code == "invalid_motion_arguments"
    assert not (host.control_root / "submitted-action.json").exists()


@pytest.mark.parametrize("host", [LIBERO], indirect=True)
def test_a_target_outside_the_workspace_names_the_flat_field(host, golden):
    arguments = {**DELTA, "translation_m": [0.0, 0.0, 5.0]}
    with pytest.raises(ToolError) as refused:
        call(host, "move_pose", arguments)
    assert refused.value.code == "workspace_bounds" and refused.value.field == "translation_m"
    golden("refusal", refused.value.as_dict(), host.control_root.parent)


@pytest.mark.parametrize("host", [LIBERO, ROBOTWIN], indirect=True)
def test_a_named_close_feeds_move_clearance(host, golden):
    arguments = ({"targets": {"right": "close"}, "target_ref": "the cup"} if "left" in host._current()[2][
        "proprioception"]["end_effectors"] else {"action": "close", "target_ref": "the cup"})
    call(host, "set_gripper", arguments)
    golden("held", host._observed_clearance_holds, host.control_root.parent)
    assert host._observed_clearance_holds["right"]["target_ref"] == "the cup"


@pytest.mark.parametrize("name,arguments,message", [
    ("move_pose", {"targets": {"right": DELTA}, "speed": "fast"}, "unexpected fields"),
    ("move_pose", {"targets": {"right": DELTA}, "expect_contact": True}, "unexpected fields"),
    ("move_pose", {"targets": {"right": DELTA}, "exact_orientation": True}, "unexpected fields"),
    ("set_gripper", {"targets": {"right": "close"}, "target_ref": "the cup"}, "unexpected fields")])
@pytest.mark.parametrize("host", [YAM], indirect=True)
def test_a_field_the_surface_does_not_publish_stays_unknown(host, name, arguments, message):
    with pytest.raises(ToolError, match=message) as refused:
        call(host, name, arguments)
    assert refused.value.code == "invalid_motion_arguments"
    assert not (host.control_root / "submitted-action.json").exists()


@pytest.mark.parametrize("host", [YAM], indirect=True)
def test_a_surface_without_observed_clearance_or_intent_has_neither(host):
    result = call(host, "move_pose", {"targets": {"right": DELTA}})
    assert "observed_clearance" not in result["compiled_pose"]["targets"]["right"]


@pytest.mark.parametrize("host", [YAM], indirect=True)
def test_a_close_on_a_surface_without_the_annotation_carries_no_intent(host):
    assert "intent" not in call(host, "set_gripper", {"targets": {"right": "close"}})


@pytest.mark.parametrize("host", [LIBERO, ROBOTWIN], indirect=True)
def test_the_observation_closes_on_the_one_proposal(host):
    name, arguments = (flat if host.surface.supports("single_arm_flat_targets_v1") else sparse)([0, 0, 1])[8]
    call(host, name, arguments)
    with pytest.raises(ToolError) as refused:
        host.invoke_with_context(name, arguments, call_context=proposal(host, name, arguments, 1))
    assert refused.value.code == "action_pending_fresh_observation"


@pytest.mark.parametrize("arguments", [{"forward_m": 0.4, "left_m": 0.1}, {"x_m": -0.5, "y_m": 0.75, "yaw_deg": -120.0},
                                       {"turn_deg": 400.0}])
@pytest.mark.parametrize("host", [ROBOCASA], indirect=True)
def test_a_base_move_proposes_its_action_and_closes_the_observation(host, golden, arguments):
    result = call(host, "move_base", arguments)
    golden("call", answer(host, result), host.control_root.parent)
    assert host._submitted_mutations == ["move_base"]
    with pytest.raises(ToolError) as refused:
        host.invoke_with_context("move_base", arguments, call_context=proposal(host, "move_base", arguments, 1))
    assert refused.value.code == "action_pending_fresh_observation"


@pytest.mark.parametrize("host", [ROBOCASA], indirect=True)
def test_after_the_terminal_boundary_a_base_move_is_refused(host, golden):
    (host.control_root / "policy-task-check.json").write_text(json.dumps({"terminal": True}))
    with pytest.raises(ToolError) as refused:
        call(host, "move_base", {"forward_m": 0.1})
    assert refused.value.code == "episode_terminal"
    golden("refusal", refused.value.as_dict(), host.control_root.parent)


# ---- a context without the key: every sealed one ----------------------------------------------------------------


class Recorder:
    """A data interface that answers one observation and takes actions; ``state`` holds what a tool wrote."""

    def __init__(self, motion, **config):
        self.config = {"execution": {"generation": 3}, "motion": {"mutation_refusals": {}, "capabilities": [], **motion},
                       **config}
        self.requests, self.state_reads = [], []
        identity = {"observation_id": "obs", "observation_sha256": "a" * 64}
        effector = {"pose_frame": "world", "pose_xyz_xyzw": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "gripper_command_target": "open"}
        manifest = {"observation_id": "obs", "proprioception": {"end_effectors": {"left": dict(effector),
                                                                                  "right": dict(effector)}}}
        ref = {"path": "m", "sha256": "b" * 64, "bytes": 1, "media_type": "application/json"}
        self.artifacts = SimpleNamespace(tool_context=lambda: self.config, read_blob=lambda ref: canonical(manifest))
        self.state = SimpleNamespace(
            observation=lambda observation=None: self._ok({"observation": identity, "manifest": ref, "files": []}),
            read=self._read, write=lambda key, value: None)
        self.act = SimpleNamespace(execute_action=self._execute, submit_action=self._submit)

    @staticmethod
    def _ok(data):
        return Response(canonical(dict(schema="ora.toolkit.response.v1", method="x", status="ok", data=data,
                                       provenance=dict(source="simulator"), error=None)))

    def _read(self, key):
        self.state_reads.append(key)
        return {"arm_joint_qpos": [0.0] * 6, "pose_xyz_xyzw": [0.0] * 6 + [1.0]} if key.startswith("exec.") else None

    def _execute(self, action, budget):
        self.requests.append(("execute_action", action))
        return self._ok({"receipt": {"action_id": action["action_id"], "native_actions": 1}, "remaining_budget": 19})

    def _submit(self, *, action, preparation=None):
        self.requests.append(("submit_action", action))
        return self._ok({"accepted": True, "action_id": action["action_id"]})


CALLS = [(set_gripper, {"targets": {"right": "close"}}, {"set_gripper"}),
         (wait_ticks, {"ticks": 2}, {"wait"}), (recover_pose, {"targets": {"right": "home"}}, {"recover_motion"})]
WAIT = ["shared_hold_wait_action_v1", "motion_intent_router_v1"]


@pytest.mark.parametrize("tool,arguments,actions", CALLS)
def test_a_context_without_the_key_executes(tool, arguments, actions):
    api = Recorder({"capabilities": WAIT})
    result = tool.run(api, arguments)
    assert [method for method, _ in api.requests] == ["execute_action"]
    assert {action["action"] for _, action in api.requests} == actions
    assert result["status"] == "executed" and "_runtime" not in result


@pytest.mark.parametrize("tool,arguments,actions", CALLS)
def test_the_key_hands_the_action_over_and_reads_no_execution_state(tool, arguments, actions):
    api = Recorder({"capabilities": WAIT, "environment_plans": True})
    result = tool.run(api, arguments)
    assert [method for method, _ in api.requests] == ["submit_action"]
    assert {action["action"] for _, action in api.requests} == actions
    assert result["status"] == "submitted" and result["_runtime"]["directive"] == "action_submitted"
    assert not [key for key in api.state_reads if key.startswith("exec.")]


def test_a_recorded_recovery_still_reads_its_anchors_and_sends_them():
    api = Recorder({})
    recover_pose.run(api, {"targets": {"right": "previous"}})
    assert api.state_reads == ["exec.recovery.right.previous"]
    assert set(api.requests[0][1]["arguments"]) == {"targets", "recovery_targets"}


def test_a_context_without_the_key_has_both_arms_and_no_proposal():
    context = {"motion": {"capabilities": []}}
    assert _context.arms(context) == ("left", "right") and not _context.environment_plans(context)
    assert _context.arms({}) == ("left", "right") and not _context.environment_plans({})


def test_a_context_without_the_key_keeps_the_sparse_refusal_for_a_flat_call():
    with pytest.raises(ToolRefusal, match="unexpected fields"):
        surface_targets({"action": "close"}, {"motion": {"capabilities": []}}, scalar_key="action")


def test_a_context_without_the_key_still_needs_both_arms():
    with pytest.raises(ToolRefusal, match="requires left and right"):
        measured_states({"proprioception": {"end_effectors": {"right": {"pose_xyz_xyzw": [0.0] * 7}}}})


def test_speed_options_keep_refusing_without_a_trajectory_on_a_context_without_the_key():
    assert _smooth.proposed_speed({"speed": "fast"}, {"motion": {"capabilities": []}}) == (None, {"speed": "fast"})


# ---- what follows a proposal, and a dry run of one ----------------------------------------------------------------


def publish_again(host):
    """The Supervisor's part: the action is consumed and the same observation is current again."""
    (host.control_root / "submitted-action.json").unlink()
    pointer = host.control_root / "current-observation.json"
    pointer.write_bytes(canonical({**json.loads(pointer.read_bytes()), "open": True}))


@pytest.mark.parametrize("host", [LIBERO, ROBOTWIN], indirect=True)
def test_a_named_close_reaches_the_next_moves_clearance(host, golden):
    single = host.surface.supports("single_arm_flat_targets_v1")
    close = {"action": "close", "target_ref": "the cup"} if single else {
        "targets": {"right": "close"}, "target_ref": "the cup"}
    move = DELTA if single else {"targets": {"right": DELTA}}
    call(host, "set_gripper", close)
    publish_again(host)
    result = host.invoke_with_context("move_pose", move, call_context=proposal(host, "move_pose", move, 1))
    golden("result", visible(result), host.control_root.parent)
    assert result["compiled_pose"]["targets"]["right"]["observed_clearance"]["status"] != "unknown"


def asked(host):
    """The bridge a dry run asks, which here cannot answer; what it was asked is what the real call would submit."""
    questions = []

    def answer(payload):
        if not is_request(payload):
            raise ValueError("this bridge answers no reach check")
        questions.append(payload)
        return {"status": "unavailable", "reason": "fixture"}

    host.align_plan_callback = answer
    return questions


@pytest.mark.parametrize("name,flat_call,sparse_call", [
    ("move_pose", DELTA, {"targets": {"right": DELTA}}),
    ("move_pose", {**DELTA, "speed": "careful"}, {"targets": {"right": DELTA}, "speed": "careful"}),
    ("move_to", {"position_m": [-0.1, 0.05, 0.95]}, {"targets": {"left": {"position_m": [-0.1, 0.05, 0.95]}}}),
    ("rotate_wrist", {"angle_deg": 15.0}, {"targets": {"right": 15.0}}),
    ("recover_pose", {"strategy": "home"}, {"targets": {"right": "home"}})])
@pytest.mark.parametrize("host", [LIBERO, ROBOTWIN], indirect=True)
def test_a_dry_run_asks_the_environment_what_the_call_would_submit_and_submits_nothing(
        host, golden, name, flat_call, sparse_call):
    arguments = {**(flat_call if host.surface.supports("single_arm_flat_targets_v1") else sparse_call), "dry_run": True}
    questions = asked(host)
    with pytest.raises(ToolError, match="unavailable") as refused:
        call(host, name, arguments)
    assert refused.value.code == "dry_run_unavailable"
    assert questions
    golden("questions", questions, host.control_root.parent)
    assert not (host.control_root / "submitted-action.json").exists()


def recorded(host):
    """The tool events ``host`` logged."""
    rows = (json.loads(line)["type"] for line in host.event_path.read_text().splitlines())
    return [row for row in rows if row.startswith("tool_")]


@pytest.mark.parametrize("host", [LIBERO, ROBOTWIN], indirect=True)
def test_a_dry_run_past_the_workspace_is_refused_and_a_dry_run_is_recorded(host, golden):
    single = host.surface.supports("single_arm_flat_targets_v1")
    outside = {**DELTA, "translation_m": [0.0, 0.0, 5.0]}
    arguments = {**(outside if single else {"targets": {"right": outside}}), "dry_run": True}
    with pytest.raises(ToolError) as refused:
        call(host, "move_pose", arguments)
    golden("refusal", refused.value.as_dict(), host.control_root.parent)
    arguments = {**(DELTA if single else {"targets": {"right": DELTA}}), "dry_run": True}
    result = call(host, "move_pose", arguments)
    assert result["feasible"] is True
    assert recorded(host) == ["tool_error", "tool_result"]


def test_an_action_the_real_call_would_refuse_is_refused_by_its_dry_run():
    with pytest.raises(ToolError, match="runtime action fields differ") as refused:
        dry_run.NoActions(admit=admitted).submit_action(action={"schema": "ora-motion-action.v3"})
    assert refused.value.code == "invalid_arguments"


@pytest.mark.parametrize("host", [LIBERO], indirect=True)
def test_a_dry_run_of_a_jaw_stops_at_the_proposal(host, golden):
    arguments = {"action": "close", "dry_run": True}
    result = call(host, "set_gripper", arguments)
    assert result["route"] == "jaw_command"
    golden("arms", result["arms"], host.control_root.parent)
    assert not (host.control_root / "submitted-action.json").exists()
