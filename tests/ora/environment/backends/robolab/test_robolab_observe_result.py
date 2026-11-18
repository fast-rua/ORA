"""The live RoboLab ``observe`` result admits the Robotiq end effector it always carries.

Every live ``observe`` on ``robolab_direct_v1`` failed with
``internal_tool_error``: ``$result.public_robot_facts.end_effectors.right.controlled_frame
is outside the result enum ['panda_grip_site', 'aloha_grip_center_tcp']``. The
public-facts projection publishes RoboLab's end effector on the Robotiq TCP
(``robotiq_2f85_tcp``), with the measured finger joint, the pad-face aperture
and the command state; the surface bound the fixed-base LIBERO ``observe``
result, whose end effector is the Panda grip site or the ALOHA TCP.

These tests drive the real dispatcher (``build_tool_runtime`` with the real
public-facts projector) on a RoboLab-shaped observation, and check that the
LIBERO surface keeps its ``observe`` result unchanged.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.context.delivery.tool_results import deliver_tool_result
from ora.agent.context.observation.contact_points import project_policy_public_facts
from ora.agent.toolkit.host.runtime.factory import build_tool_runtime
from tests.support.agent.atomic_causal_runtime import MANUAL
from tests.support.agent.public_robot_facts import fixture, published_fixture


ROOT = Path(__file__).resolve().parents[5]
SURFACE = "robolab_direct_v1"
#: The fixed-base observe result LIBERO resolves (and RoboLab resolved until revision 2).
LIBERO_OBSERVE = ROOT / "ora/policy/tool_surfaces/explicit_motion_v3/observe.json"
#: The fixture's Robotiq jaw (finger joint 0.39 rad, jaw separation 0.0425 m,
#: opening 0.5, commanded open) as the projection publishes it.
ROBOTIQ_GRIPPER = {
    "measured_finger_joint_rad": .39, "measured_aperture_mm": 42.5, "measured_opening_normalized": .5,
    "measurement_semantics": "fingertip_pad_inner_face_separation_from_measured_link_poses",
    "grasp_evidence": "not_established_by_joint_measurement",
    "commanded_opening_normalized": 1., "command_target": "open",
}
#: Tools that need a live model service; leaving them out does not touch the result schemas.
MODEL_SERVICE_TOOLS = {"vision_segment", "vision_ground", "grasp_plan"}


def _runtime(tmp_path: Path):
    surface, _, projector, store = published_fixture(tmp_path, "robolab")
    assert surface.profile_id == SURFACE
    return surface, projector, store, build_tool_runtime(
        control_root=projector.control_root, manual_path=MANUAL, event_path=tmp_path / "events.jsonl",
        workspace_bounds_m={"x": [-3, 3], "y": [-3, 3], "z": [0, 3]},
        allowed_tools=[item["name"] for item in surface.definitions if item["name"] not in MODEL_SERVICE_TOOLS],
        state_root=store.root, profile_id=SURFACE, component_overrides={},
        controlled_end_effector="right", public_facts_projector=project_policy_public_facts,
    )


@pytest.mark.parametrize("arguments", [{}, {"views": ["agentview"]}, {"views": ["agentview", "wrist"]}])
def test_observe_validates_and_carries_the_robotiq_end_effector(tmp_path, arguments):
    surface, _, _, runtime = _runtime(tmp_path)
    result = runtime.invoke("observe", arguments)
    assert result["success"] is True
    effector = result["public_robot_facts"]["end_effectors"]["right"]
    assert effector["controlled_frame"] == "robotiq_2f85_tcp"
    assert effector["gripper"] == ROBOTIQ_GRIPPER
    # The result validates in radians; delivery restates the joint in degrees.
    delivered = dict(deliver_tool_result(surface, "observe", result)["public_robot_facts"]
                     ["end_effectors"]["right"]["gripper"])
    assert delivered.pop("measured_finger_joint_deg") == pytest.approx(math.degrees(.39), abs=1e-6)
    assert delivered == {key: value for key, value in ROBOTIQ_GRIPPER.items() if key != "measured_finger_joint_rad"}


def test_a_historical_observe_validates_too(tmp_path):
    _, projector, store, runtime = _runtime(tmp_path)
    _, observation, source = fixture(tmp_path, "robolab", 2, published=True)
    store.publish(observation=observation, observation_root=source, cause="after_action",
        task_instruction="Inspect", remaining_budget={}, previous_public_action=None)
    projector.publish(observation=observation, observation_root=source,
        task_instruction="Inspect", remaining_budget={}, policy_state=store.steps[-1])
    result = runtime.invoke("observe", {"step": 0, "views": ["agentview", "wrist"]})
    assert result["status"] == "actual_historical_observation"
    facts = result["public_robot_facts"]
    assert facts["snapshot_role"] == "historical"
    assert facts["end_effectors"]["right"]["gripper"] == ROBOTIQ_GRIPPER


def test_the_robolab_observe_result_is_the_libero_one_with_the_robotiq_end_effector():
    robolab = copy.deepcopy(resolve_tool_surface(SURFACE).result_schemas["observe"])
    libero = json.loads(LIBERO_OBSERVE.read_text())["resultSchema"]
    effector = robolab["$defs"]["end_effector"]["properties"]
    assert effector["controlled_frame"] == {"type": "string", "const": "robotiq_2f85_tcp"}
    gripper = effector["gripper"]
    assert gripper["additionalProperties"] is False
    assert set(gripper["properties"]) == set(ROBOTIQ_GRIPPER)
    assert not {"commanded_opening_normalized", "command_target"} & set(gripper["required"])
    for key in ("controlled_frame", "gripper"):
        effector[key] = libero["$defs"]["end_effector"]["properties"][key]
    assert robolab == libero


def test_libero_keeps_its_observe_result():
    observe = resolve_tool_surface("libero_placement_direct_overhead_v1").result_schemas["observe"]
    assert observe == json.loads(LIBERO_OBSERVE.read_text())["resultSchema"]
    assert observe["$defs"]["end_effector"]["properties"]["controlled_frame"]["enum"] == [
        "panda_grip_site", "aloha_grip_center_tcp"]
