"""The 0059 ports on the RoboTwin and LIBERO default surfaces.

Every unit is capability-gated and the current defaults
(``robotwin_toolchain_b2_closed_noservo_v2``,
``libero_placement_direct_overhead_v1``) declare them. This module checks:

* M5 ``world_axes``: the help names each embodiment's own declared axes (ALOHA
  TCP approach +X, jaw line +Y; Panda grip site approach +Z, jaw line +X), and
  the dispatcher compiles a world-axes request onto them;
* V1: the LIBERO help names the 1024x800 depth-profile card size.

On LIBERO, M2 (the CuRobo reach backend) also replaces grasp_plan's help with
the released text plus the IK-selection paragraph and adds the pooled
``ik_reachability`` fields; ``test_libero_reach_ports.py`` drives them. The
real dispatcher delivers every port through result-schema admission.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.embodiment.geometry import public_local_axes
from ora.agent.composition.schemas.world_axes import TARGET_AXES_PROPERTY
from ora.agent.toolkit.host.perception import grounding_preview
from ora.contracts.pose import quaternion_matrix

#: The default surface of each backend, which declares every 0059 unit.
DEFAULTS = {
    "robotwin": "robotwin_toolchain_b2_closed_noservo_v2",
    "libero_pro": "libero_placement_direct_overhead_v1",
}
POSE_TOOLS = ("move_pose", "move_to", "rotate_wrist", "rotate_pitch")
#: Result fields the ports add under compiled_pose.targets.<arm>.
ADDED_TARGET_FIELDS = {
    "robotwin": {"reach_along_request", "observed_clearance", "target_axes_world"},
    "libero_pro": {"reach_along_request", "observed_clearance", "target_axes_world"},
}


@pytest.fixture(params=["robotwin", "libero_pro"])
def backend(request):
    return request.param


def _descriptions(surface):
    return {item["name"]: item["description"] for item in surface.definitions}


def _target_properties(schema):
    return schema["properties"]["compiled_pose"]["properties"]["targets"]["additionalProperties"]["properties"]


# --------------------------------------------------------------------------
# M5: world_axes on each embodiment's declared axes
# --------------------------------------------------------------------------

AXIS_NAMES = {(1, 0, 0): "+X", (0, 1, 0): "+Y", (0, 0, 1): "+Z"}


def test_world_axes_names_the_embodiment_s_own_axes(backend):
    surface = resolve_tool_surface(DEFAULTS[backend])
    convention = surface.view_config["controlled_frame_convention"]
    axes = public_local_axes(convention)
    for tool in POSE_TOOLS:
        assert _target_properties(surface.result_schemas[tool])["target_axes_world"] == TARGET_AXES_PROPERTY
    move_pose = _descriptions(surface)["move_pose"]
    stated = re.search(r"approach_world \(.*?local (\+[XYZ])\) and jaw_line_world \(its local (\+[XYZ])\)", move_pose)
    assert stated, move_pose[-400:]
    assert (stated[1], stated[2]) == (AXIS_NAMES[tuple(int(v) for v in axes["approach"])],
                                      AXIS_NAMES[tuple(int(v) for v in axes["closing"])])
    assert "link6" not in move_pose
    assert "world_axes takes approach_world and jaw_line_world;" in move_pose


# --------------------------------------------------------------------------
# V1: the depth-profile card size the LIBERO help names
# --------------------------------------------------------------------------

def test_the_libero_help_names_the_profiled_card_size():
    delivered = _descriptions(resolve_tool_surface(DEFAULTS["libero_pro"]))["vision_ground"]
    legacy = f"{grounding_preview.CARD_WIDTH}x{grounding_preview.LEGACY_CARD_HEIGHT}"
    profiled = f"{grounding_preview.CARD_WIDTH}x{grounding_preview.PROFILE_CARD_HEIGHT}"
    assert delivered.count(profiled) == 1 and legacy not in delivered


# --------------------------------------------------------------------------
# The real dispatcher: every port through result-schema admission
# --------------------------------------------------------------------------

def _dispatcher(backend, tmp_path):
    from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
    from tests.support.agent.observed_clearance_v2 import ALOHA_DOWN, LIBERO_CAMERAS, PANDA_DOWN, ROBOTWIN_CAMERAS, _aloha, \
        _panda, _write_scene
    from tests.support.agent.surface_catalog import catalog_grants
    from tests.support.agent.toolkit_run import served
    from tests.support.robodojo.closed_skills_default import BothArmsProposer
    from tests.support.robodojo.perception import FakeHeadGrounding

    source = tmp_path / "observation"
    if backend == "robotwin":
        observation = _write_scene(source, {**ROBOTWIN_CAMERAS, "front_camera": ((0.0, -0.8, 1.2), (0.0, 1.0, -0.5))},
                                   {"left": _aloha([-0.3, -0.3, 1.1, *ALOHA_DOWN]),
                                    "right": _aloha([0.3, -0.3, 1.1, *ALOHA_DOWN])})
    else:
        observation = _write_scene(source, LIBERO_CAMERAS, {"right": _panda([0.3, 0.3, 1.1, *PANDA_DOWN])})
    control = tmp_path / "control"
    control.mkdir()
    runtime = ManipulationToolRuntime(
        control_root=control, manual_path=Path(__file__).resolve().parents[5]
        / "ora/policy/prompt_bundles/templates/tool-schema-delivery-v1.md",
        event_path=tmp_path / "events.jsonl", workspace_bounds_m={"x": [-0.6, 0.6], "y": [-0.4, 0.4],
                                                                  "z": [0.7, 1.45]},
        state_root=None, tool_surface_profile=DEFAULTS[backend], toolkit_grants=catalog_grants(DEFAULTS[backend]),
        controlled_end_effector="right",
        segmentation_backend=FakeHeadGrounding(), grounding_backend=FakeHeadGrounding(),
        grasp_backend=BothArmsProposer())
    pointer = {"open": True, "observation_id": observation["observation_id"],
               "observation_path": str(source / "observation.json"), "generation": 3}
    runtime._current = lambda: (pointer, source, observation)
    return served(runtime)  # in the run's layout, as a run serves it (``toolkit_run.served``)


def _robotwin_answer(request):
    from ora.contracts.reach_check import RESULT_SCHEMA

    return {"schema": RESULT_SCHEMA, "arm": request["arm"], "ik_model": "robotwin_single_seed_damped_least_squares_v1",
            "matches_move_pose_admission": False, "motion_planner_called": False,
            "claim_boundary": "One damped-least-squares descent; not move_pose's admission.",
            "results": [{"reachable": True, "reason": None, "route": None, "position_error_m": 0.001,
                         "orientation_error_rad": 0.01, "joint_limit_margin_rad": 0.5, "joint_delta_rad": 0.2}
                        for _query in request["queries"]]}


def _libero_answer(request):
    from ora.environment.backends.libero_pro.reach_check import CLAIM_BOUNDARY_ROBOT_ONLY, IK_MODEL_ROBOT_ONLY

    return {**_robotwin_answer(request), "ik_model": IK_MODEL_ROBOT_ONLY, "claim_boundary": CLAIM_BOUNDARY_ROBOT_ONLY}


def test_the_dispatcher_delivers_every_port_through_result_admission(backend, tmp_path):
    runtime = _dispatcher(backend, tmp_path)
    requests = []
    answer = _robotwin_answer if backend == "robotwin" else _libero_answer
    runtime.align_plan_callback = lambda request: requests.append(request) or answer(request)
    arm, jaw_line = ("left", [0, 1, 0]) if backend == "robotwin" else ("right", [1, 0, 0])
    request = {"interaction": "free_space", "position_mode": "absolute", "position_m": [0.0, 0.0, 0.95],
               "orientation_mode": "world_axes", "approach_world": [0, 0, -1], "jaw_line_world": jaw_line}
    result = runtime.invoke("move_pose", {"targets": {arm: request}} if backend == "robotwin" else request)
    target = result["compiled_pose"]["targets"][arm]
    assert set(target) >= ADDED_TARGET_FIELDS[backend]
    assert target["target_axes_world"] == {"approach": [0.0, 0.0, -1.0], "jaw_line": [float(v) for v in jaw_line]}
    clearance = target["observed_clearance"]
    tip = 0.03557 if backend == "robotwin" else 0.00925  # fingertips beyond the grip centre along the approach
    assert clearance["status"] == "measured"
    assert clearance["footprint_lowest_m"][2] == pytest.approx(0.95 - tip, abs=2e-4)
    assert clearance["clearance_m"] == pytest.approx(0.95 - tip - 0.842, abs=0.005)
    reach = target["reach_along_request"]
    assert len(requests) == 1
    assert requests[0]["arm"] == arm
    assert "measured_disagreement" not in reach
    # No new policy-visible key collides with the agent-trace privileged-field denylist.
    from ora.agent.evidence.trace_privacy import _forbidden_path

    assert _forbidden_path(target) is None
    for tool in POSE_TOOLS:
        assert _forbidden_path(_target_properties(runtime.surface.result_schemas[tool])) is None, tool
    submitted = json.loads((runtime.control_root / "submitted-action.json").read_text())["arguments"]["targets"]
    rotation = np.asarray(quaternion_matrix(submitted[arm]["orientation_xyzw"]))
    approach = public_local_axes(runtime.surface.view_config["controlled_frame_convention"])["approach"]
    assert rotation @ np.asarray(approach, float) == pytest.approx([0, 0, -1], abs=1e-9)
    assert submitted[arm]["position_m"] == pytest.approx([0.0, 0.0, 0.95])
