"""LIBERO answers the internal read-only reach check with CuRobo IK (integration 0059 M2).

The reach check (``contracts/reach_check.py``) asks one arm whether it can hold
exact world-frame poses; RoboTwin and RoboDojo answer it with their own IK, and
until M2 LIBERO had no ``reach_check_batch`` at all, so ``grasp_plan`` on LIBERO
could not screen its candidates (H200 BOOT-01). LIBERO now answers with one
CuRobo v0.7.8 IK solve per pose in the isolated planner worker -- the IK stage
of the MotionGen planner its global ``move_pose`` segment plans through --
with joint limits and Panda self-collision and no world model.

Offline and layered:

* the worker's ``ik`` command over a CPU stand-in for CuRobo (no Torch);
* the host-side client contract for that command;
* the session's ``reach_check_batch`` over a fake IK planner, through the real
  frame conversion ``move_pose`` uses (world grip site -> CuRobo base-frame
  panda_hand) and the real shared controller;
* the closed LIBERO reason set and the surface binder that selects it.

The agent edge with a LIBERO bridge is in ``test_libero_reach_ports.py``.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from ora.contracts.reach_check import (
    BACKEND_REASONS, LIBERO_REASONS, REQUEST_SCHEMA, RESULT_SCHEMA, ROBODOJO_REASONS,
    SHARED_REASONS, START_UNREACHABLE,
)
from ora.contracts.refusal import EnvironmentRefusal
from ora.environment.backends.libero_pro import reach_check as libero_reach
from ora.environment.backends.libero_pro.common import LiberoProError
from ora.environment.backends.libero_pro.planned_motion import LiberoPlannedMotionMixin
from ora.environment.backends.libero_pro.planner_port import IK_RECORD_SCHEMA
from ora.environment.backends.libero_pro.planner_request import PlannerRequestError, validate_libero_ik_request
from tests.support.libero.reach_check import (
    NAMES, MEASURED, LIMITS, DOWN, _matrix, _pose, WORLD_FROM_BASE, HAND_FROM_GRIP, BASE_FROM_HAND,
    FakeIk, _Session,
)


TOLERANCES = {"position_tolerance_m": 0.005, "orientation_tolerance_rad": 0.05}


def _base_hand_target(grip_world):
    """Independent of the session code: base <- world <- grip, then grip -> hand."""
    return _pose(np.linalg.inv(_matrix(WORLD_FROM_BASE)) @ _matrix(grip_world)
                 @ np.linalg.inv(_matrix(HAND_FROM_GRIP)))


def _query(pose, interaction="free_space", from_index=None):
    return {"pose_xyzw": list(pose), "interaction": interaction, "from_index": from_index}


def _ask(session, *queries, arm="right"):
    return session.align_plan({"schema": REQUEST_SCHEMA, "arm": arm, "queries": [dict(q) for q in queries]})


def _grip(position, orientation=DOWN):
    return [*position, *orientation]


# --------------------------------------------------------------------------
# Session: the frame move_pose commands, the real controller, a fake IK
# --------------------------------------------------------------------------

def test_libero_solves_the_grip_site_pose_move_pose_commands_in_curobo_s_hand_frame():
    planner = FakeIk()
    session = _Session(planner)
    target = _grip([-0.1, 0.1, 1.05])
    answer = _ask(session, _query(target))
    assert answer["schema"] == RESULT_SCHEMA and answer["arm"] == "right"
    assert answer["ik_model"] == libero_reach.IK_MODEL_ROBOT_ONLY == "libero_reach_ik_robot_only_v1"
    assert answer["matches_move_pose_admission"] is False and answer["motion_planner_called"] is False
    [sent] = planner.requests[0]["queries"]
    # The planner is asked for panda_hand in the base frame: the grip site
    # target composed with the measured hand-to-grip transform, exactly as the
    # global move_pose segment compiles it.
    assert sent["target_pose_base_xyzw"][:3] == pytest.approx(_base_hand_target(target)[:3], abs=1e-9)
    assert abs(float(np.dot(sent["target_pose_base_xyzw"][3:], _base_hand_target(target)[3:]))) == \
        pytest.approx(1.0, abs=1e-9)
    assert planner.requests[0]["joint_positions"] == MEASURED
    [row] = answer["results"]
    assert row["reachable"] is True and row["reason"] is None and row["route"] is None
    assert row["position_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert row["orientation_error_rad"] == pytest.approx(0.0, abs=1e-6)
    assert row["joint_delta_rad"] == pytest.approx(0.1)
    # joint 4 sits 0.8 rad below its upper stop -0.0698 at -2.2 ... the smallest margin is joint 7's
    assert row["joint_limit_margin_rad"] == pytest.approx(min(
        min(value - low, high - value) for value, low, high in zip(
            [MEASURED[0] + 0.1, *MEASURED[1:]], LIMITS[0], LIMITS[1]))
    )
    for phrase in ("One IK solve", "joint limits", "self-collision", "no table, object or scene",
                   "no trajectory", "OSC"):
        assert phrase in answer["claim_boundary"], phrase


def test_a_chained_approach_starts_from_its_pregrasp_solution_and_an_orphan_is_start_unreachable():
    planner = FakeIk()
    session = _Session(planner)
    pregrasp, approach = _grip([-0.1, 0.1, 1.05]), _grip([-0.1, 0.1, 0.98])
    far = _grip([0.6, 0.6, 1.05])
    answer = _ask(session, _query(pregrasp), _query(approach, "contact_approach", 0),
                  _query(far), _query(approach, "contact_approach", 2))
    sent = planner.requests[0]["queries"]
    assert [item["from_index"] for item in sent] == [None, 0, None, 2]
    first, chained, failed, orphan = answer["results"]
    assert first["reachable"] and chained["reachable"]
    # 0.1 rad from the chained seed, not 0.2 from the measured joints.
    assert chained["joint_delta_rad"] == pytest.approx(0.1)
    assert failed == {"reachable": False, "reason": "ik_failed", "route": None, "position_error_m": None,
                      "orientation_error_rad": None, "joint_limit_margin_rad": None, "joint_delta_rad": None}
    assert orphan["reachable"] is False and orphan["reason"] == START_UNREACHABLE


def test_a_solution_whose_grip_site_misses_the_arrival_bound_is_named():
    # CuRobo judges the hand; a turn of 0.06 rad about the hand also swings the
    # grip site 0.1034 m away by about 6 mm.
    answer = _ask(_Session(FakeIk(turn_rad=0.06)), _query(_grip([-0.1, 0.1, 1.05])),
                  _query(_grip([-0.1, 0.1, 1.0]), "contact_approach", 0))
    row, chained = answer["results"]
    assert row["reachable"] is False and row["reason"] == "fk_residual_exceeded"
    assert row["orientation_error_rad"] == pytest.approx(0.06, abs=1e-6)
    assert row["position_error_m"] == pytest.approx(0.1034 * math.sin(0.06), abs=2e-4)
    assert row["joint_delta_rad"] == pytest.approx(0.1)
    assert chained["reason"] == START_UNREACHABLE


def test_a_turn_inside_the_arrival_bound_is_reachable():
    [row] = _ask(_Session(FakeIk(turn_rad=0.02)), _query(_grip([-0.1, 0.1, 1.05])))["results"]
    assert row["reachable"] is True and row["orientation_error_rad"] == pytest.approx(0.02, abs=1e-6)


def test_curobo_fk_disagreeing_with_the_measured_hand_is_a_typed_answer():
    shifted = [BASE_FROM_HAND[0] + 0.01, *BASE_FROM_HAND[1:]]
    answer = _ask(_Session(FakeIk(start=shifted)), _query(_grip([-0.1, 0.1, 1.05])),
                  _query(_grip([0.6, 0.6, 1.05])))
    assert [row["reason"] for row in answer["results"]] == ["fk_calibration_mismatch"] * 2
    assert not any(row["reachable"] for row in answer["results"])


def test_the_check_refuses_to_answer_if_the_simulation_moved():
    session = _Session(None)
    session.motion_planner = FakeIk(step=lambda: setattr(session.environment, "_elapsed_steps", 8))
    with pytest.raises(LiberoProError, match="read-only"):
        _ask(session, _query(_grip([-0.1, 0.1, 1.05])))


def test_the_bridge_serves_only_the_reach_check_and_only_the_right_arm():
    session = _Session(FakeIk())
    with pytest.raises(LiberoProError, match="only the internal reach check"):
        session.align_plan({"schema": "ora.align-plan-request.v1", "target_ref": "t",
                            "approach_side": "top", "arm": "right", "target_poses_xyzw": []})
    with pytest.raises(EnvironmentRefusal) as refused:
        _ask(session, _query(_grip([-0.1, 0.1, 1.05])), arm="left")
    assert refused.value.field == "arm" and refused.value.allowed == ["right"]
    with pytest.raises(LiberoProError, match="drifted"):
        session.align_plan({"schema": REQUEST_SCHEMA, "arm": "right", "queries": [], "extra": 1})


def test_no_planner_or_a_planner_error_is_a_fault_not_an_answer():
    with pytest.raises(LiberoProError, match="CuRobo planner is not active"):
        _ask(_Session(None), _query(_grip([-0.1, 0.1, 1.05])))

    class Broken(FakeIk):
        def solve_ik(self, request):
            return {"schema": IK_RECORD_SCHEMA, "status": "planner_error", "world_model": "none",
                    "solve_time_s": 0.0, "error": {"type": "RuntimeError", "message": "cuda"}}

    with pytest.raises(LiberoProError, match="CuRobo IK failed: RuntimeError"):
        _ask(_Session(Broken()), _query(_grip([-0.1, 0.1, 1.05])))


def test_the_session_class_serves_the_bridge_through_its_reach_mixin():
    from ora.environment.backends.libero_pro.session import LiberoProSession

    assert issubclass(LiberoProSession, libero_reach.LiberoReachCheckMixin)
    assert LiberoProSession.align_plan is libero_reach.LiberoReachCheckMixin.align_plan


def test_the_real_session_answers_the_bridge_without_stepping_or_commanding(tmp_path, monkeypatch):
    """The real LiberoProSession over the CPU plant of ``test_libero_correctness_repair``."""
    from tests.support.libero.correctness_repair import TARGET, _session

    session, plant = _session(tmp_path, monkeypatch)
    # That fixture's base is the world origin and its hand sits at (0.2, 0, 0.9).
    planner = FakeIk(reach_m=1.5, start=[0.2, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0])
    session.motion_planner = planner
    steps, mutations = session.environment._elapsed_steps, session.world_mutation_sequence
    answer = session.align_plan({"schema": REQUEST_SCHEMA, "arm": "right", "queries": [_query(TARGET)]})
    [row] = answer["results"]
    assert row["reachable"] is True and answer["ik_model"] == libero_reach.IK_MODEL_ROBOT_ONLY
    # grip site (0.2, 0, 0.8) under hand (0.2, 0, 0.9): the hand target is 0.1 m above the grip target
    assert planner.requests[0]["queries"][0]["target_pose_base_xyzw"][:3] == pytest.approx(
        [TARGET[0], TARGET[1], TARGET[2] + 0.1])
    assert plant.commands == [] and session.environment._elapsed_steps == steps
    assert session.world_mutation_sequence == mutations and session.action_sequence == 0


# --------------------------------------------------------------------------
# The IK request contract
# --------------------------------------------------------------------------

def _ik_request(*queries):
    return {"joint_names": list(NAMES), "joint_positions": list(MEASURED),
            "queries": [{"target_pose_base_xyzw": [0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], "from_index": None},
                        *queries]}


@pytest.mark.parametrize("change", [
    lambda r: r.update(joint_names=list(reversed(NAMES))),
    lambda r: r["joint_positions"].__setitem__(2, float("nan")),
    lambda r: r.update(queries=[]),
    lambda r: r.update(queries=r["queries"] * 17),
    lambda r: r["queries"].append({"target_pose_base_xyzw": [0.4, 0, 0.5, 0, 0, 0, 0.5], "from_index": None}),
    lambda r: r["queries"].append({"target_pose_base_xyzw": [0.4, 0, 0.5, 0, 0, 0, 1], "from_index": 1}),
    lambda r: r["queries"].append({"target_pose_base_xyzw": [0.4, 0, 0.5, 0, 0, 0, 1], "from_index": True}),
    lambda r: r["queries"].append({"target_pose_base_xyzw": [0.4, 0, 0.5, 0, 0, 0, 1]}),
    lambda r: r.update(scene={}),
])
def test_the_ik_request_contract_is_closed(change):
    request = _ik_request()
    change(request)
    with pytest.raises(PlannerRequestError):
        validate_libero_ik_request(request)


# --------------------------------------------------------------------------
# The closed LIBERO reason set and its surface selection
# --------------------------------------------------------------------------

def test_the_libero_reason_set_is_closed_and_admitted_by_the_controller():
    assert LIBERO_REASONS == ("ik_failed", "fk_residual_exceeded", "fk_calibration_mismatch")
    assert set(LIBERO_REASONS) <= BACKEND_REASONS
    assert not set(LIBERO_REASONS) & set(SHARED_REASONS)
    source = (libero_reach.__file__)
    text = open(source, encoding="utf-8").read()
    for reason in LIBERO_REASONS:
        assert f'"{reason}"' in text, reason
    # The released sets are unchanged.
    assert "fk_calibration_mismatch" in ROBODOJO_REASONS


def test_the_grasp_annotation_selects_the_libero_set_from_its_identity_capability():
    """Same field shapes as RoboTwin's annotation; only the stage reason enum differs.

    Both surfaces are file-backed: their grasp_plan files hold what the released binder and
    LIBERO's sibling module wrote.
    """
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from ora.contracts.robotwin_reach import REASONS as ROBOTWIN_REASONS

    def annotation(profile):
        return copy.deepcopy(resolve_tool_surface(profile).result_schemas["grasp_plan"]["$defs"]["candidate"][
            "properties"]["ik_reachability"])

    libero = annotation("libero_placement_direct_overhead_v1")
    twin = annotation("robotwin_toolchain_b2_closed_noservo_v2")
    for stage in ("pregrasp", "approach"):
        assert libero["properties"][stage]["properties"]["reason"]["enum"] == [*SHARED_REASONS, *LIBERO_REASONS, None]
        reason = twin["properties"][stage]["properties"]["reason"]
        assert reason["enum"] == [*SHARED_REASONS, *ROBOTWIN_REASONS, None]
        reason["enum"] = [*SHARED_REASONS, *LIBERO_REASONS, None]
    assert libero == twin  # every other field is the released shape


def test_the_answer_names_the_planner_s_world_model():
    from ora.environment.controllers.depth_world import PUBLIC_DEPTH_VOXEL_V1, WORLD_MODEL

    assert libero_reach.ik_model_and_claim("none") == (libero_reach.IK_MODEL_ROBOT_ONLY,
                                                       libero_reach.CLAIM_BOUNDARY_ROBOT_ONLY)
    model, claim = libero_reach.ik_model_and_claim(WORLD_MODEL)
    assert model == libero_reach.IK_MODEL_DEPTH_WORLD == "libero_reach_ik_depth_world_v1"
    assert "no table, object or scene" not in claim and "CuRobo" not in claim
    for phrase in ("collision world", "both cameras", "robot removed", "held object", "counted as free",
                   f"more than {PUBLIC_DEPTH_VOXEL_V1.support_tolerance_m * 100:g} cm below", "no trajectory", "OSC"):
        assert phrase in claim, phrase
    with pytest.raises(LiberoProError, match="world model"):
        libero_reach.ik_model_and_claim("scene")
