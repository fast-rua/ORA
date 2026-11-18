"""``robot.query``'s host side (``ora/agent/code_mode/query.py``, ``toolkit/host/query.py``).

Each query is a call of the toolkit lib a motion tool plans with, over the episode's data interface, with typed values
and degrees in and degrees out. On the recorded RoboDojo observation (``tests/data/toolkit-general-pickup``) each query
equals the lib function it wraps, called through an ordinary journaled interface with the radian arguments the lib takes
and its answer put through the policy-degree table: angles to 1e-6 degrees (the table rounds there), metres exactly. The
environment's read-only bridge is a deterministic toy, as the lib's own tests use: there is no recorded kinematics
answer in the repository, and the equivalence is the wrapper's, not the solver's. A query records nothing, and the
refusals (arguments, no toolkit, no bridge, a closed observation) are named.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

import pytest

from ora.agent.code_mode import query, query_spec
from ora.agent.composition.surface.units import policy_degrees
from ora.agent.toolkit.interface.projections import motion_probe
from ora.agent.toolkit.lib.motion_plan import checks
from ora.agent.toolkit.tools.move_pose import _smooth
from ora.agent.toolkit.lib import motion, observation
from ora.agent.toolkit.lib.motion_plan import kinematics
from ora.agent.toolkit.lib.values import JsonValue
from ora.agent.toolkit.host.failures import service_adapter
from ora.agent.toolkit.host.motion import joint_trajectory_surface, plans_moves
from ora.agent.toolkit.host.query import query_api
from ora.agent.error_classes import ToolError
from tests.support.agent.kinematics_bridge import Bridge
from tests.support.agent.stack_parity import on_toolkit, stand_in
from tests.support.agent.toolkit_goldens import Goldens
from tests.support.agent.toolkit_tools import api_for

POSE = [0.12, -0.2, 0.9, 0.0, 0.0, 0.0, 1.0]
NEXT = [0.14, -0.2, 0.9, 0.0, 0.0, 0.0, 1.0]
JOINTS = [1.0, -2.0, 3.0, 0.5, -0.5, 20.0]
RAD = "_rad"
GOLDEN = Goldens(__file__)


@pytest.fixture
def served(published):
    """The recorded observation's tool runtime as a toolkit surface would be: the data interface bound, a toy bridge."""

    bridge = Bridge()
    published._toolkit_tools_enabled = True  # what dispatch.install says
    published.align_plan_callback, published.motion_probe_callback = bridge, bridge.probe
    return published, bridge


def close(left: Any, right: Any, path: str = "") -> None:
    """Equal, floats to 1e-6 (a degree's rounding) and everything else exactly."""

    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            close(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right), path
        for index, (item, other) in enumerate(zip(left, right)):
            close(item, other, f"{path}[{index}]")
    elif isinstance(left, float) or isinstance(right, float):
        assert left == pytest.approx(right, abs=1e-6), path
    else:
        assert left == right, path


def radians(values: list[float]) -> list[float]:
    return [math.radians(value) for value in values]


def ask(host, name: str, **arguments: Any) -> dict[str, Any]:
    return query.prepare(name, arguments)(query_api(host))


def test_ik_equals_the_lib_chain_in_degrees(served) -> None:
    host, bridge = served
    answer = ask(host, "ik", arm="right", poses=[POSE, NEXT, [0.7, 0.0, 0.9, 0, 0, 0, 1]], seed_joints_deg=JOINTS,
                 max_joint_step_deg=10, ranking="canonical_first", screen="self")
    reference = policy_degrees(kinematics.ik_chain(
        api_for(host), arm="right", poses=[POSE, NEXT, [0.7, 0.0, 0.9, 0, 0, 0, 1]], seed=radians(JOINTS),
        max_joint_step_rad=math.radians(10), ranking="canonical_first", screen="self"))
    close(answer, reference)
    sent = bridge.requests[0]
    assert sent["seed_joints_rad"] == pytest.approx(radians(JOINTS)) and sent["screen"] == "self"
    assert sent["max_joint_step_rad"] == pytest.approx(math.radians(10)) and sent["ranking"] == "canonical_first"
    assert [row["status"] for row in answer["rows"]] == ["ok", "ok", "ik_failed"] and answer["first_failure_index"] == 2
    assert answer["rows"][0]["joints_deg"][0] == pytest.approx(1.2 * 180 / math.pi, abs=1e-6)
    assert "orientation_error_deg" in answer["rows"][0] and not any(RAD in key for key in answer["rows"][0])


def test_ik_defaults_are_the_measured_seed_and_a_line_step(served) -> None:
    host, bridge = served
    ask(host, "ik", arm="left", poses=POSE)  # one pose alone
    sent = bridge.requests[0]
    assert sent["seed_joints_rad"] is None and sent["poses_xyz_xyzw"] == [POSE]
    assert sent["max_joint_step_rad"] == pytest.approx(math.radians(query_spec.DEFAULT_STEP_DEG))
    assert (sent["ranking"], sent["screen"]) == ("nearest", "self_and_measured_other_arm")


def test_fk_equals_the_lib_poses_from_degrees(served) -> None:
    host, bridge = served
    answer = ask(host, "fk", arm="left", joints_deg=[JOINTS, [0.0] * 6])
    assert answer == {"poses": kinematics.fk(api_for(host), arm="left", joints=[radians(JOINTS), [0.0] * 6])}
    close(bridge.requests[0]["joints_rad"], [radians(JOINTS), [0.0] * 6])
    assert ask(host, "fk", arm="left", joints_deg=JOINTS) == {"poses": [answer["poses"][0]]}, "one vector alone"


def test_screen_path_equals_the_lib_screen_of_degree_knots(served) -> None:
    host, bridge = served
    knots = {"left": [JOINTS, [0.0] * 6, JOINTS], "right": [[0.0] * 6, JOINTS, [0.0] * 6]}
    answer = ask(host, "screen_path", knots_deg=knots)
    reference = policy_degrees(kinematics.screen_path(api_for(host), {arm: [radians(row) for row in rows]
                                                                      for arm, rows in knots.items()}))
    close(answer, reference)
    assert answer["status"] == "clear" and answer["segments_screened"] == 2
    close(bridge.requests[0]["knots_rad"]["left"][0], radians(JOINTS))


def test_reach_check_equals_the_lib_check_in_degrees(served) -> None:
    host, bridge = served
    answer = ask(host, "reach_check", arm="right", poses=[POSE, NEXT], interaction=["free_space", "contact_approach"],
                 from_index=[None, 0])
    request = {"schema": "ora.reach-check-request.v1", "arm": "right", "queries": [
        {"pose_xyzw": POSE, "interaction": "free_space", "from_index": None},
        {"pose_xyzw": NEXT, "interaction": "contact_approach", "from_index": 0}]}
    reference = policy_degrees(motion.reach_check(api_for(host), JsonValue.of(request)).data.json())
    close(answer, reference)
    assert bridge.requests[0] == request and answer["results"][1]["joint_delta_deg"] == pytest.approx(
        math.degrees(0.4), abs=1e-6)
    assert not any(RAD in key for row in answer["results"] for key in row)
    one = ask(host, "reach_check", arm="right", poses=[POSE, NEXT], interaction="precision")
    assert [item["interaction"] for item in bridge.requests[-1]["queries"]] == ["precision", "precision"]
    assert [item["from_index"] for item in bridge.requests[-1]["queries"]] == [None, None] and len(one["results"]) == 2


def _planned(host, bridge, monkeypatch) -> dict[str, Any]:
    """The host as one whose motion tools plan on the policy side, with the shared planner answering."""

    bridge.planner = True
    motion = {"capabilities": [], "global_planner": True, "local_axes": {"approach": [1.0, 0.0, 0.0]},
              "workspace_bounds_m": {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]}}
    monkeypatch.setattr("ora.agent.toolkit.host.query.planned_motion", lambda host: motion)
    return {"motion": motion}


def test_a_probe_gets_the_verdict_the_real_move_gets(served, monkeypatch) -> None:
    host, bridge = served
    config = _planned(host, bridge, monkeypatch)
    bridge.refusals[NEXT[0]] = "goal_unreachable"
    answer = ask(host, "probe_motion", arm="right", poses=[POSE, NEXT], variant_policy="parallel_jaw_equivalents")
    first, second = answer["results"]
    assert not bridge.probed and answer["world_state_changed"] is False  # the environment's IK probe is not asked
    assert first["reachable"] is True and first["selected_variant_id"] == first["variants"][0]["variant_id"]
    assert [row["route"] for row in first["variants"]] == ["global", "global"]
    assert first["variants"][1]["target_pose_xyzw"][:3] == POSE[:3]
    assert first["variants"][1]["target_pose_xyzw"][3:] == [1.0, 0.0, 0.0, 0.0]
    assert second["reachable"] is False and second["selected_variant_id"] is None
    assert [row["failure"]["code"] for row in second["variants"]] == ["goal_unreachable", "goal_unreachable"]
    action = {"arguments": {"targets": {"right": {"position_m": NEXT[:3], "orientation_xyzw": NEXT[3:]}},
                            "interactions": {"right": "free_space"}}}
    with pytest.raises(checks.PlanRefusal) as refused:
        _smooth.plan_action(query_api(host), action, config, budget=50)
    assert refused.value.code == second["variants"][0]["failure"]["code"]
    planned = _smooth.plan_action(query_api(host), {"arguments": {
        "targets": {"right": {"position_m": POSE[:3], "orientation_xyzw": POSE[3:]}},
        "interactions": {"right": "free_space"}}}, config, budget=500)[1]
    assert first["variants"][0]["trajectory_steps"] == planned.knots["native_actions"]
    assert first["variants"][0]["trajectory_duration_s"] == pytest.approx(planned.knots["duration_s"])
    assert not any(RAD in key for row in first["variants"] for key in row)


def test_a_probe_outside_the_workspace_is_the_real_call_s_refusal(served, monkeypatch) -> None:
    host, bridge = served
    current = ask(host, "robot_state")["arms"]["right"]["pose"]["xyz"]
    _planned(host, bridge, monkeypatch)["motion"]["workspace_bounds_m"].update(
        {axis: [value - 0.001, value + 0.001] for axis, value in zip("xyz", current)})
    row = ask(host, "probe_motion", arm="right", poses=[POSE])["results"][0]["variants"][0]
    assert row["reachable"] is False and row["failure"]["code"] == "workspace_bounds"


def test_a_probe_where_no_planner_answers_is_the_real_move_s_joint_line(served, monkeypatch) -> None:
    host, bridge = served
    _planned(host, bridge, monkeypatch)
    bridge.planner = False
    row = ask(host, "probe_motion", arm="right", poses=[POSE])["results"][0]["variants"][0]
    assert row["reachable"] is True and row["route"] == "joint"


def test_what_a_contract_declares_and_a_host_serves_are_one_judgement_that_names_no_backend() -> None:
    surface = SimpleNamespace(supports=lambda name: name == "toolkit_synchronous_execution_v1", profile_id="any")
    asynchronous = SimpleNamespace(supports=lambda name: False, profile_id="any")
    assert joint_trajectory_surface(surface) and not joint_trajectory_surface(asynchronous)
    assert plans_moves(SimpleNamespace(align_plan_callback=print, surface=surface))
    assert not plans_moves(SimpleNamespace(align_plan_callback=None, surface=surface))
    assert not plans_moves(SimpleNamespace(align_plan_callback=print, surface=asynchronous))


def test_probe_motion_equals_the_lib_probe_in_degrees(served) -> None:
    host, bridge = served
    answer = ask(host, "probe_motion", arm="right", poses=[POSE, NEXT], variant_policy="parallel_jaw_equivalents")
    request = {"queries": [{"query_id": f"query-{index}", "arm": "right",
                            "pose": {"position_m": pose[:3], "orientation_xyzw": pose[3:]}}
                           for index, pose in enumerate([POSE, NEXT])], "variant_policy": "parallel_jaw_equivalents"}
    # a tool's probe goes through the host's own, which adds the observation's identity and registers the plan handles
    api = api_for(host)
    api.adapters["probe_motion"] = service_adapter(lambda parameters, inputs, store: motion_probe(host._motion_probe(
        {key: parameters["request"][key] for key in ("queries", "variant_policy")})), "grasp_proposal_failed")
    reference = policy_degrees(motion.probe_motion(api, JsonValue.of(request)).data.json())
    close(answer, reference)
    assert [item["query_id"] for item in bridge.probed[0]["queries"]] == ["query-0", "query-1"]
    assert answer["results"][0]["variants"][0]["joint_delta_deg"] == pytest.approx(
        [math.degrees(value) for value in (0.1, 0.2, 0.3, 0.0, 0.0, 0.5)], abs=1e-6)
    assert answer["world_state_changed"] is False


def test_align_plan_equals_the_lib_ranking_in_degrees(served) -> None:
    host, bridge = served
    answer = ask(host, "align_plan", arm="both", poses=[POSE, NEXT], approach_side="front")
    request = {"schema": "ora.align-plan-request.v1", "target_ref": None, "approach_side": "front", "arm": "both",
               "target_poses_xyzw": [POSE, NEXT]}
    close(answer, policy_degrees(motion.align_plan(api_for(host), JsonValue.of(request)).data.json()))
    assert bridge.requests[0] == request and answer["candidates"][0]["orientation_error_deg"] == pytest.approx(
        math.degrees(0.02), abs=1e-6)


def test_robot_state_equals_the_lib_state_of_the_recorded_observation(served) -> None:
    host, _ = served
    answer = ask(host, "robot_state")
    _control, _root, manifest = host._current()
    reference = observation.robot_state(JsonValue.of(manifest["proprioception"]))
    assert list(answer["arms"]) == [arm.name for arm in reference.arms] == ["left", "right"]
    for arm in reference.arms:
        item = answer["arms"][arm.name]
        assert item["pose"] == {"frame": arm.pose.frame, "xyz": list(arm.pose.position_m),
                                "xyzw": list(arm.pose.orientation_xyzw)}
        assert item["pose"]["frame"] == "robodojo_episode_local_ee_link"
        assert item["joint_positions_deg"] == pytest.approx([math.degrees(q) for q in arm.arm_joint_qpos], abs=1e-6)
        assert item["gripper_normalized"] == arm.gripper_normalized
    published_arm = manifest["proprioception"]["end_effectors"]["right"]
    public = answer["public"]["end_effectors"]["right"]
    assert "arm_joint_qpos" not in public, "radians without a unit in the name are the joint_positions_deg above"
    assert public["pose_xyz_xyzw"] == published_arm["pose_xyz_xyzw"]
    assert public["gripper_feedback"] == published_arm["gripper_feedback"]


def test_a_query_records_nothing(served) -> None:
    host, _ = served
    run = host.control_root.parent
    before = sorted(str(path.relative_to(run)) for path in run.rglob("*") if path.is_file())
    for name, arguments in (("ik", {"arm": "left", "poses": POSE}), ("fk", {"arm": "left", "joints_deg": JOINTS}),
                            ("reach_check", {"arm": "left", "poses": POSE}),
                            ("probe_motion", {"arm": "left", "poses": POSE}), ("robot_state", {})):
        ask(host, name, **arguments)
    after = sorted(str(path.relative_to(run)) for path in run.rglob("*") if path.is_file())
    assert after == before, "no I/O journal event, blob or gap file"


# --- what is refused, and how -----------------------------------------------------------------------------------

@pytest.mark.parametrize("name, arguments, field", [
    ("nothing", {}, "name"),
    ("ik", {"arm": "left"}, "poses"),
    ("ik", {"poses": POSE}, "arm"),
    ("ik", {"arm": "left", "poses": POSE, "joint_limits": 1}, "joint_limits"),
    ("ik", {"arm": "left", "poses": [POSE[:6]]}, "poses[0]"),
    ("ik", {"arm": "left", "poses": [[*POSE[:3], 0, 0, 0, 3]]}, "poses[0]"),
    ("ik", {"arm": "left", "poses": [[math.nan, *POSE[1:]]]}, "poses[0][0]"),
    ("ik", {"arm": "left", "poses": [POSE] * 257}, "poses"),
    ("ik", {"arm": "left", "poses": POSE, "seed_joints_deg": [0, 0]}, "seed_joints_deg"),
    ("ik", {"arm": "left", "poses": POSE, "max_joint_step_deg": 0}, "max_joint_step_deg"),
    ("ik", {"arm": "left", "poses": POSE, "max_joint_step_deg": 361}, "max_joint_step_deg"),
    ("ik", {"arm": "left", "poses": POSE, "ranking": "best"}, "ranking"),
    ("ik", {"arm": "", "poses": POSE}, "arm"),
    ("fk", {"arm": "left", "joints_deg": [0, 0, 0]}, "joints_deg[0]"),
    ("fk", {"arm": "left", "joints_deg": [[0.0] * 6] * 257}, "joints_deg"),
    ("screen_path", {"knots_deg": {}}, "knots_deg"),
    ("screen_path", {"knots_deg": {"left": [JOINTS, JOINTS], "right": [JOINTS]}}, "knots_deg"),
    ("reach_check", {"arm": "left", "poses": [POSE] * 17}, "poses"),
    ("reach_check", {"arm": "left", "poses": [POSE, POSE], "from_index": [None, 1]}, "from_index"),
    ("reach_check", {"arm": "left", "poses": POSE, "interaction": "teleport"}, "interaction"),
    ("probe_motion", {"arm": "left", "poses": [POSE] * 13}, "poses"),
    ("probe_motion", {"arm": "left", "poses": POSE, "variant_policy": "all"}, "variant_policy"),
    ("align_plan", {"arm": "left", "poses": POSE, "approach_side": "under"}, "approach_side"),
    ("robot_state", {"arm": "left"}, "arm"),
])
def test_arguments_are_checked_before_any_environment_is_asked(name, arguments, field) -> None:
    with pytest.raises(ToolError) as error:
        query.prepare(name, arguments)
    assert error.value.field == field and error.value.code in {"invalid_arguments", "unknown_query"}
    assert error.value.retryable is True


def test_an_unknown_query_names_the_ones_there_are() -> None:
    with pytest.raises(ToolError) as error:
        query.prepare("dry_run", {})
    assert error.value.code == "unknown_query" and error.value.constraints["allowed"] == list(query_spec.NAMES)
    with pytest.raises(ToolError) as error:
        query.prepare("ik", [1, 2])
    assert error.value.field == "arguments"


def test_a_surface_without_toolkit_tools_has_no_interface_to_query_through(published) -> None:
    published.align_plan_callback = Bridge()
    with pytest.raises(ToolError) as error:
        query_api(published)
    assert error.value.code == "query_unavailable" and "toolkit" in str(error.value) and not error.value.retryable


@pytest.mark.parametrize("key", ["libero", "robotwin", "robocasa", "robolab", "yam"])
def test_a_surface_the_environment_plans_for_stays_unqueried_when_it_runs_the_toolkit(
        request, tmp_path, monkeypatch, key) -> None:
    """V10: what a script's card does not offer, the runtime refuses, in the words the handlers' surface refused it."""
    with pytest.raises(ToolError) as toolkit:
        query_api(on_toolkit(stand_in(tmp_path / "toolkit", key, monkeypatch)))
    GOLDEN.checker(request)("refusal", toolkit.value.as_dict())


def test_no_bridge_is_unavailable_never_faked(served) -> None:
    host, _ = served
    host.align_plan_callback = None
    for name, arguments in (("ik", {"arm": "left", "poses": POSE}), ("fk", {"arm": "left", "joints_deg": JOINTS}),
                            ("screen_path", {"knots_deg": {"left": [JOINTS]}}),
                            ("reach_check", {"arm": "left", "poses": POSE}),
                            ("align_plan", {"arm": "left", "poses": POSE})):
        with pytest.raises(ToolError) as error:
            ask(host, name, **arguments)
        assert error.value.code == "query_unavailable", name
        assert "bridge" in str(error.value), name
    host.motion_probe_callback = None
    with pytest.raises(ToolError) as error:
        ask(host, "probe_motion", arm="left", poses=POSE)
    assert error.value.code == "query_unavailable" and "bridge" in str(error.value)
    assert ask(host, "robot_state")["arms"], "the measured state needs no bridge"


def test_a_backend_that_answers_no_kinematics_query_says_so(served) -> None:
    from ora.contracts.refusal import EnvironmentRefusal

    host, _ = served

    def unsupported(request):
        raise EnvironmentRefusal("LiberoSession answers no kinematics query (unsupported_operation)", field="schema",
                                 received=request["schema"], allowed=[])

    host.align_plan_callback = unsupported
    with pytest.raises(ToolError) as error:
        ask(host, "ik", arm="left", poses=POSE)
    assert error.value.code == "query_unavailable" and "unsupported_operation" in str(error.value)


def test_the_environments_own_refusal_keeps_its_field_and_allowed_values(served) -> None:
    from ora.contracts.refusal import EnvironmentRefusal

    host, _ = served

    def refuses(request):
        raise EnvironmentRefusal("arm must be left or right", field="arm", received="middle", allowed=["left", "right"])

    host.align_plan_callback = refuses
    for name, arguments in (("ik", {"arm": "middle", "poses": POSE}), ("reach_check", {"arm": "middle", "poses": POSE}),
                            ("align_plan", {"arm": "middle", "poses": POSE})):
        with pytest.raises(ToolError) as error:
            ask(host, name, **arguments)
        assert error.value.code == "invalid_arguments" and error.value.field == "arm", name
        assert error.value.received == "middle" and error.value.constraints == {"allowed": ["left", "right"]}, name


def test_a_closed_observation_is_the_hosts_own_refusal(served) -> None:
    host, _ = served
    pointer = host.control_root / "current-observation.json"
    control = json.loads(pointer.read_text(encoding="utf-8"))
    pointer.write_text(json.dumps({**control, "open": False}), encoding="utf-8")
    with pytest.raises(ToolError) as error:
        query_api(host)
    assert error.value.code == "action_pending_fresh_observation"


def test_every_query_is_documented_by_the_arguments_it_takes() -> None:
    from ora.agent.code_mode import docs

    entries = docs.query_entries()
    assert list(entries) == [f"query.{name}" for name in query_spec.NAMES]
    for name in query_spec.NAMES:
        definition = query_spec.definition(name)
        assert set(definition["inputSchema"]["properties"]) == set(query_spec.argument_names(name))
        parts = entries[f"query.{name}"]
        assert list(parts) == ["description", "arguments", "result", "examples"], name
        assert parts["description"].startswith(f'robot.query("{name}"'), name
        for argument in query_spec.argument_names(name):
            assert f"- {argument}:" in parts["arguments"], (name, argument)
        assert "Result" in parts["result"] and f'robot.query("{name}"' in parts["examples"]
        line = docs.index_line(definition)
        assert line.startswith(f'- robot.query("{name}"') and line.endswith(f'(robot.docs("query_{name}"))')
        assert "\n" not in line
    for name in ("ik", "fk", "reach_check", "probe_motion", "align_plan", "screen_path"):
        text = "\n".join(entries[f"query.{name}"].values())
        assert "degrees" in text or "_deg" in text, name
    assert not any(RAD in text for parts in entries.values() for text in parts.values()), "no radian name is taught"
