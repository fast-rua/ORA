"""move_base: absolute and relative targets become one world-frame motion-v3 action that the environment plans.

The data interface here answers one observation and takes the proposal; what the runtime handlers propose for the same
calls is compared in ``tests/ora/agent/toolkit/test_stack_parity.py`` and ``../test_proposal_path.py``.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from ora.agent.toolkit.interface.types import Response, canonical
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.tools.move_base import tool
from ora.environment.controllers.robocasa_omron import BASE_MAX_TRAVEL_M

ROOT = Path(__file__).resolve().parents[6]
PLANS = ["mobile_base_navigation_v1"]


class Api:
    """One observation with the measured base pose ``pose``; ``submitted`` holds what the tool handed over."""

    def __init__(self, *, pose=(1.0, 2.0, math.pi / 2), capabilities=PLANS, environment_plans=True, refusals=None):
        self.submitted = []
        motion = {"capabilities": list(capabilities), "mutation_refusals": refusals or {},
                  **({"environment_plans": True} if environment_plans else {})}
        self.config = {"execution": {"generation": 3}, "motion": motion}
        proprioception = {"end_effectors": {"right": {"pose_xyz_xyzw": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}}}
        if pose is not None:
            proprioception["base"] = {"pose_xy_yaw": list(pose), "frame": "robocasa_world"}
        manifest = {"observation_id": "obs", "proprioception": proprioception}
        identity = {"observation_id": "obs", "observation_sha256": "a" * 64}
        ref = {"path": "m", "sha256": "b" * 64, "bytes": 1, "media_type": "application/json"}
        self.artifacts = SimpleNamespace(tool_context=lambda: self.config, read_blob=lambda ref: canonical(manifest))
        self.state = SimpleNamespace(
            observation=lambda observation=None: self._ok({"observation": identity, "manifest": ref, "files": []}))
        self.act = SimpleNamespace(submit_action=self._submit)

    @staticmethod
    def _ok(data):
        return Response(canonical(dict(schema="ora.toolkit.response.v1", method="x", status="ok", data=data,
                                       provenance=dict(source="simulator"), error=None)))

    def _submit(self, *, action, preparation=None):
        self.submitted.append(action)
        return self._ok({"accepted": True, "action_id": action["action_id"]})


def target(api):
    (action,) = api.submitted
    assert (action["schema"], action["action"], action["action_id"]) == ("ora-motion-action.v3", "move_base", "action-3")
    assert (action["observation_id"], action["bound_observation_sha256"]) == ("obs", "a" * 64)
    return action["arguments"]["target"]


def test_an_absolute_target_is_the_world_pose_with_yaw_in_radians():
    api = Api()
    assert tool.run(api, {"x_m": 2.5, "y_m": -1.0, "yaw_deg": 270.0}) == {
        "status": "submitted", "action": "move_base",
        "_runtime": {"directive": "action_submitted", "action_id": "action-3"}}
    moved = target(api)
    assert (moved["x_m"], moved["y_m"]) == (2.5, -1.0)
    assert math.isclose(moved["yaw_rad"], -math.pi / 2)


def test_a_relative_target_is_resolved_against_the_measured_base_heading():
    api = Api(pose=(1.0, 2.0, math.pi / 2))  # facing +y
    tool.run(api, {"forward_m": 0.5, "left_m": 0.2, "turn_deg": 90.0, "x_m": None, "y_m": None, "yaw_deg": None})
    moved = target(api)
    assert math.isclose(moved["x_m"], 1.0 - 0.2) and math.isclose(moved["y_m"], 2.5)
    assert math.isclose(abs(moved["yaw_rad"]), math.pi)


def test_a_turn_alone_keeps_the_base_position():
    api = Api(pose=(0.0, 0.0, 0.0))
    tool.run(api, {"turn_deg": -30.0})
    moved = target(api)
    assert (moved["x_m"], moved["y_m"]) == (0.0, 0.0)
    assert math.isclose(moved["yaw_rad"], math.radians(-30.0))


@pytest.mark.parametrize("arguments", [
    {"x_m": 1.0, "y_m": 1.0, "yaw_deg": 0.0, "forward_m": 0.1},
    {"x_m": 1.0, "y_m": 1.0},
    {},
])
def test_mixed_or_partial_forms_are_refused_naming_both_forms_and_no_values(arguments):
    api = Api()
    with pytest.raises(ToolRefusal) as refused:
        tool.run(api, arguments)
    assert refused.value.code == "invalid_arguments"
    assert refused.value.constraints == {"absolute_world": ["x_m", "y_m", "yaw_deg"],
                                         "relative_to_base": ["forward_m", "left_m", "turn_deg"]}
    assert refused.value.fields["received"] == sorted(arguments)
    assert api.submitted == []


@pytest.mark.parametrize("arguments,missing", [({"yaw_deg": 90.0}, "x_m and y_m"), ({"x_m": 1.0, "y_m": 2.0}, "yaw_deg"),
                                               ({"y_m": 2.0, "yaw_deg": None}, "x_m and yaw_deg")])
def test_a_partial_world_target_is_refused_naming_what_it_lacks(arguments, missing):
    """26 of 27 RoboCasa365 refusals were a part of the world target, which "never both" did not describe."""
    with pytest.raises(ToolRefusal) as refused:
        tool.run(Api(), arguments)
    assert str(refused.value).startswith(f"the absolute target is missing {missing}: move_base takes either all of ")
    mixed = pytest.raises(ToolRefusal, tool.run, Api(), {"x_m": 1.0, "y_m": 1.0, "yaw_deg": 0.0, "forward_m": 0.1})
    assert str(mixed.value).startswith("move_base takes either all of x_m, y_m and yaw_deg"), "a mix says so as before"


@pytest.mark.parametrize("arguments,field", [
    ({"forward_m": BASE_MAX_TRAVEL_M + 0.1}, "forward_m"),
    ({"x_m": 1.0 + BASE_MAX_TRAVEL_M + 0.1, "y_m": 2.0, "yaw_deg": 0.0}, "x_m"),
])
def test_a_target_beyond_the_travel_bound_is_refused_before_submission(arguments, field):
    api = Api()
    with pytest.raises(ToolRefusal) as refused:
        tool.run(api, arguments)
    assert refused.value.constraints == {"max_travel_m": BASE_MAX_TRAVEL_M}
    assert refused.value.fields["field"] == field
    assert api.submitted == []


def test_non_finite_numbers_are_refused_by_name():
    with pytest.raises(ToolRefusal) as refused:
        tool.run(Api(), {"forward_m": float("nan")})
    assert refused.value.fields["field"] == "forward_m" and refused.value.code == "invalid_arguments"


@pytest.mark.parametrize("api", [Api(capabilities=()), Api(environment_plans=False)], ids=["no_base", "executing"])
def test_only_a_surface_with_a_base_whose_environment_plans_has_the_tool(api):
    """A context without ``environment_plans`` (RoboDojo's, every sealed one) has no base to drive."""
    with pytest.raises(ToolRefusal) as refused:
        tool.run(api, {"forward_m": 0.1})
    assert refused.value.code == "tool_unavailable" and refused.value.fields["retryable"] is False
    assert api.submitted == []


def test_the_host_s_mutation_refusal_comes_first():
    terminal = {"code": "episode_terminal", "message": "move_base is unavailable after the trusted environment "
                "terminal boundary", "retryable": False, "field": None, "received": None, "constraints": {},
                "infrastructure": False}
    api = Api(refusals={"move_base": terminal})
    with pytest.raises(ToolRefusal) as refused:
        tool.run(api, {})
    assert refused.value.code == "episode_terminal" and str(refused.value) == terminal["message"]


def test_a_call_without_a_measured_base_pose_is_refused():
    with pytest.raises(ToolRefusal, match="the current base pose is unavailable") as refused:
        tool.run(Api(pose=None), {"forward_m": 0.1})
    assert refused.value.code == "pose_state_unavailable"


def test_the_tool_refuses_the_bound_the_base_servo_enforces():
    assert tool.MAX_TRAVEL_M == BASE_MAX_TRAVEL_M


def test_its_description_is_the_robocasa_surface_s_definition_and_result():
    """RoboCasa's surface keeps its own ``move_base.json`` only for the inputs its null-as-omitted capability rewrote
    (every optional input also takes null); the description and the result it delivers are this tool's own."""
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface

    description = json.loads((ROOT / "ora/agent/toolkit/tools/move_base/description.json").read_text())
    surface = resolve_tool_surface("robocasa_mobile_direct_v1")
    definition = next(item for item in surface.definitions if item["name"] == "move_base")
    refs = {item["slot"]: item["ref"] for item in surface.components}
    assert refs["result:move_base"] == "ora/agent/toolkit/tools/move_base/description.json"
    assert surface.result_schemas["move_base"] == description["resultSchema"]
    assert definition["description"] == description["description"]
    delivered = definition["inputSchema"]["properties"]
    for name, field in description["inputSchema"]["properties"].items():
        assert delivered[name] == {"type": [field["type"], "null"], "description": f"{field['description']} Null means omitted."}
