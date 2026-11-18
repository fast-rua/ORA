"""The station's clearance gates refuse approaching a limit, never leaving one, and say what to change.

``Departure`` alone runs anywhere; the planner cases build the station model (MuJoCo and i2rt).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from ora.environment.backends.yam import motion


def test_a_motion_starting_inside_a_clearance_may_leave_it_but_not_approach():
    gate = motion.Departure(0.02, 0.012)
    assert all(gate.admits(d) for d in (0.012, 0.013, 0.019, 0.025, 0.021))
    assert not gate.admits(0.0199)                     # past the gate, the gate holds again
    sinking = motion.Departure(0.02, 0.012)
    assert sinking.admits(0.0125) and not sinking.admits(0.0121)
    assert not motion.Departure(0.02, 0.05).admits(0.019)   # starting clear changes nothing
    assert motion.Departure(0.02, None).admits(0.02) and not motion.Departure(0.02, None).admits(0.019)


def _station():
    pytest.importorskip("mujoco")
    pytest.importorskip("i2rt")
    from ora.environment.backends.yam.kinematics import StationKinematics, pose_matrix
    from ora.environment.backends.yam.scene import DEFAULT_TABLE, build_cell_model
    from ora.environment.controllers import yam_station as Y

    kin = StationKinematics(build_cell_model(table=DEFAULT_TABLE))
    q = {arm: np.zeros(6) for arm in Y.ARMS}
    rig = types.SimpleNamespace(arm_q=lambda arm: np.asarray(q[arm], float), table_top_z_m=float(DEFAULT_TABLE["top_z_m"]))
    session = types.SimpleNamespace(kinematics=kin, rig=rig, home_q={}, last_safe_q=None,
                                    finger_state=lambda: {arm: Y.FINGER_OPEN_M for arm in Y.ARMS})
    down = [1.0, 0.0, 0.0, 0.0]

    def place(arm, xyz, quat=down):
        result = kin.ik(arm, pose_matrix(xyz, quat), q[arm], max_iterations=400)
        assert result.ok, xyz
        q[arm] = result.q
        return pose_matrix(xyz, quat)

    return session, place, pose_matrix, down


def test_a_free_space_lift_from_a_precision_grasp_height_is_planned():
    session, place, pose_matrix, down = _station()
    place("left", [0.28, 0.22, 0.16])
    place("right", [0.42, -0.14, 0.012])               # where a precision descent left it
    assert session.kinematics.table_distance("right") < 0.02
    schedule = motion.plan_motion(session, {"right": ([pose_matrix([0.42, -0.14, 0.112], down)], "free_space")})
    assert schedule.summary["ticks"] > 0
    with pytest.raises(motion.Refusal) as sinking:     # going lower from there is still refused
        motion.plan_motion(session, {"right": ([pose_matrix([0.42, -0.14, 0.009], down)], "free_space")})
    assert sinking.value.code == "table_clearance"


def test_an_unreachable_target_reports_the_highest_height_that_solves():
    session, place, pose_matrix, down = _station()
    place("left", [0.38, 0.12, 0.14])
    with pytest.raises(motion.Refusal) as refused:     # the table centre, straight down, too high for the left arm
        motion.admit_target(session, "left", pose_matrix([0.383, 0.007, 0.14], down), "free_space")
    assert refused.value.code == "target_ik_unreachable"
    reachable = refused.value.detail["highest_reachable_z_m"]
    assert reachable is not None and 0.02 <= reachable < 0.14
    motion.admit_target(session, "left", pose_matrix([0.383, 0.007, reachable], down), "free_space")


def test_an_inter_arm_refusal_names_the_parts_that_meet_and_the_moving_arm():
    session, place, pose_matrix, down = _station()
    place("left", [0.383, 0.06, 0.08])
    place("right", [0.42, -0.14, 0.112])
    with pytest.raises(motion.Refusal) as refused:
        motion.plan_motion(session, {"left": ([pose_matrix([0.383, 0.007, 0.06], down)], "precision")})
    assert refused.value.code == "inter_arm_clearance"
    assert refused.value.detail["moving_arms"] == ["left"]
    assert set(refused.value.detail["closest_parts"]) == {"left", "right"}


def test_an_unreachable_target_names_the_nearest_position_that_solves_at_its_height():
    """E's instruction 5 (11-14): the right arm, straight down, at the table centre; the model searched it with
    30-120 dry runs per script. One refusal now says where the nearest reachable spot is."""
    session, place, pose_matrix, down = _station()
    place("right", [0.42, -0.14, 0.11])
    target = pose_matrix([0.45, 0.0, 0.03], down)
    # Off by default: the refusal names the highest height only.
    with pytest.raises(motion.Refusal) as refused:
        motion.admit_target(session, "right", target, "precision")
    assert refused.value.code == "target_ik_unreachable" and "highest_reachable_z_m" in refused.value.detail
    assert "nearest_reachable_position_m" not in refused.value.detail
    session.config = {"reach_query": True}  # runtime.reach_query, ora play --reach-query
    with pytest.raises(motion.Refusal) as refused:
        motion.admit_target(session, "right", target, "precision")
    assert refused.value.code == "target_ik_unreachable"
    nearest = refused.value.detail["nearest_reachable_position_m"]
    assert nearest is not None and nearest[2] == 0.03
    offset = refused.value.detail["nearest_reachable_offset_m"]
    assert 0.0 < offset <= motion.NEAREST_SEARCH_RADIUS_M
    assert nearest[1] < 0.0, "the right arm reaches on its own side"
    motion.admit_target(session, "right", pose_matrix(nearest, down), "precision")


def test_a_motion_from_rest_plans_when_the_measured_zero_puts_a_joint_just_past_its_limit():
    """11-14 on the real cell: at rest joints 2 and 3 sit on their hard stops, and the cell's measured encoder zeros
    (-0.009 to -0.018 rad) put those readings just outside the model's range. Every motion from rest was refused at
    its first waypoint (ik_unreachable, 2-4 mm) or tick (joint_limit). A joint past its limit at the start may stay
    there or move inward; it may not go further out."""
    session, _place, pose_matrix, _down = _station()
    kin = session.kinematics
    rest = {"left": np.array([-0.018, 0.002, 0.001, -0.053, -0.016, -0.071]),
            "right": np.array([-0.014, 0.001, 0.001, -0.052, -0.008, 0.001])}
    kin.joint_offsets = {"left": np.array([0.0, -0.0154, 0.0001, -0.0547, 0.0354, 0.0]),
                         "right": np.array([0.0, -0.0094, -0.0177, -0.0311, 0.0181, 0.0])}
    session.rig.arm_q = lambda arm: rest[arm].copy()
    kin.set_state(rest, session.finger_state())
    assert not kin.within_limits("right", rest["right"]) and kin.within_limits("right", rest["right"], start=rest["right"])
    start = kin.fk("right", rest["right"])
    lifted = start.copy()
    lifted[2, 3] += 0.03
    schedule = motion.plan_motion(session, {"right": ([lifted], "free_space")})
    assert schedule.summary["ticks"] > 0
    limits = kin.widened_limits("right", rest["right"])
    assert all(np.all(q["right"] >= limits[:, 0] - 1e-6) for q in schedule.ticks), "no joint went further out"
    motion.admit_target(session, "right", lifted, "free_space")
    # Without the widening the same IK fails, as it did on the cell.
    assert not kin.ik("right", start, rest["right"], max_iterations=400).ok
    assert kin.ik("right", start, rest["right"], max_iterations=400, start=rest["right"]).ok
