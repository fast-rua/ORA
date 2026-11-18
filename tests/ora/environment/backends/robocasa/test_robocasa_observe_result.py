"""The live RoboCasa365 ``observe`` result admits the mobile base it always carries.

Every live ``observe`` on ``robocasa_mobile_direct_v1`` failed with
``internal_tool_error``: ``$result.public_robot_facts has unknown result fields
['mobile_base']``. The public-facts projection publishes the measured base pose
as ``public_robot_facts.mobile_base`` whenever the observation carries
``proprioception.base``, as every RoboCasa observation does, and the
``move_base`` help tells the policy to read it there; the surface bound the
closed fixed-base LIBERO ``observe`` result, which does not declare it.

These tests drive the real dispatcher (``build_tool_runtime`` with the real
public-facts projector) on a RoboCasa-shaped observation, and check that the
fixed-base surfaces sharing that LIBERO result keep it unchanged.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.context.delivery.tool_results import deliver_tool_result
from ora.agent.context.observation.contact_points import project_policy_public_facts
from ora.agent.toolkit.host.runtime.factory import build_tool_runtime
from tests.support.agent.atomic_causal_runtime import MANUAL
from tests.support.agent.public_robot_facts import published_fixture


ROOT = Path(__file__).resolve().parents[5]
SURFACE = "robocasa_mobile_direct_v1"
#: RoboLab declares its own observe result too (its Robotiq end effector,
#: tests/ora/environment/backends/robolab/test_robolab_observe_result.py).
FIXED_BASE = ("libero_placement_direct_overhead_v1",)
#: The fixed-base observe result LIBERO resolves.
LIBERO_OBSERVE = ROOT / "ora/policy/tool_surfaces/explicit_motion_v3/observe.json"
#: The fixture's ``proprioception.base`` (pose_xy_yaw [1.5, -2.0, 0.5 rad],
#: torso 0.1 m) as the projection publishes it.
MOBILE_BASE = {"x_m": 1.5, "y_m": -2.0, "yaw_deg": 28.65, "frame": "world", "torso_height_m": 0.1}
#: Tools that need a live model service; leaving them out does not touch the result schemas.
MODEL_SERVICE_TOOLS = {"vision_segment", "vision_ground", "grasp_plan"}


def _runtime(tmp_path: Path):
    surface, _, projector, store = published_fixture(tmp_path, "robocasa")
    assert surface.profile_id == SURFACE
    return surface, build_tool_runtime(
        control_root=projector.control_root, manual_path=MANUAL, event_path=tmp_path / "events.jsonl",
        workspace_bounds_m={"x": [-3, 3], "y": [-3, 3], "z": [0, 3]},
        allowed_tools=[item["name"] for item in surface.definitions if item["name"] not in MODEL_SERVICE_TOOLS],
        state_root=store.root, profile_id=SURFACE, component_overrides={},
        controlled_end_effector="right", public_facts_projector=project_policy_public_facts,
    )


@pytest.mark.parametrize("arguments", [{}, {"views": ["agentview"]}, {"views": ["agentview", "wrist"]}])
def test_observe_validates_and_carries_the_mobile_base(tmp_path, arguments):
    surface, runtime = _runtime(tmp_path)
    result = runtime.invoke("observe", arguments)
    assert result["success"] is True
    assert result["public_robot_facts"]["mobile_base"] == MOBILE_BASE
    delivered = deliver_tool_result(surface, "observe", result)
    assert delivered["public_robot_facts"]["mobile_base"] == MOBILE_BASE


def test_the_robocasa_observe_result_is_the_fixed_base_one_plus_the_mobile_base():
    robocasa = copy.deepcopy(resolve_tool_surface(SURFACE).result_schemas["observe"])
    declared = robocasa["properties"]["public_robot_facts"]
    base = declared["properties"].pop("mobile_base")
    assert base["additionalProperties"] is False
    assert set(base["properties"]) == set(MOBILE_BASE)
    assert "mobile_base" not in declared["required"]
    assert robocasa == json.loads(LIBERO_OBSERVE.read_text())["resultSchema"]


@pytest.mark.parametrize("profile", FIXED_BASE)
def test_the_fixed_base_surfaces_keep_their_observe_result(profile):
    observe = resolve_tool_surface(profile).result_schemas["observe"]
    assert observe == json.loads(LIBERO_OBSERVE.read_text())["resultSchema"]
    assert "mobile_base" not in observe["properties"]["public_robot_facts"]["properties"]
