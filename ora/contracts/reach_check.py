"""Versioned read-only reach-check envelope shared by Agent and Environment.

The reach check asks one arm whether it can hold each of a batch of exact
world-frame poses, answered by the backend's own read-only inverse kinematics.
It is an internal Harness check -- ``grasp_plan``'s pooled IK selection is its
consumer -- and no ToolSurface declares it as a policy tool.
Each query names the move_pose ``interaction`` it models and may start from the
configuration an earlier query in the same batch reached (``from_index``), so a
pregrasp-then-approach chain is answered the way it would execute. It travels
over the same read-only IK bridge as ``align_plan`` (``plan_alignment`` routes
on the request ``schema``), so no new worker method, RPC verb or coordinator
callback exists for it. RoboTwin and RoboDojo route it through their existing
``align_plan`` session methods; LIBERO, which serves no ``align_plan``, answers
only this schema on the bridge (``backends/libero_pro/reach_check.py``). The
shared half lives here because ``agent`` may not import
``ora.environment``.

Each backend owns its closed rejection vocabulary; the published result schema
of the surface that serves it names the exact list. The union below is what the
controller accepts from any backend; anything else is contract drift, not a
policy error.
"""

from __future__ import annotations

from .alignment import AlignmentRequestDrift

REQUEST_SCHEMA = "ora.reach-check-request.v1"
RESULT_SCHEMA = "ora.reach-check-result.v1"

#: One call answers at most this many poses. It is a cost bound (one IK solve
#: per pose, two for a staged contact approach), and it is what a grasp_plan
#: annotation needs: three candidates x two jaw-equivalent members x pregrasp
#: and approach = 12.
MAXIMUM_POSES = 16

REQUEST_FIELDS = frozenset({"schema", "arm", "queries"})
QUERY_FIELDS = frozenset({"pose_xyzw", "interaction", "from_index"})
#: The move_pose interactions, whose route a backend may model.
INTERACTIONS = ("free_space", "precision", "contact_approach")

#: Assigned by the policy-side edge before the backend is asked: a pose the
#: action workspace refuses is refused by every motion tool too.
OUTSIDE_WORKSPACE = "outside_workspace"
#: A query chained (``from_index``) onto a pose that was not reachable has no
#: start configuration to solve from.
START_UNREACHABLE = "start_unreachable"
SHARED_REASONS = (OUTSIDE_WORKSPACE, START_UNREACHABLE)

#: The other arm in RoboTwin's reach check (``contracts/robotwin_reach.py``).
INTER_ARM_COLLISION_ENDPOINT = "inter_arm_collision_endpoint"
#: A path that meets the other arm held at its measured joints (RoboDojo below).
INTER_ARM_COLLISION_PATH = "inter_arm_collision_path"
#: RoboDojo: the shared planner's IK in the world a plan to the pose would see
#: (``backends/robodojo/reach_check.py``). ``fk_calibration_mismatch`` means the measured
#: link6 pose disagrees with FK of the measured joints (the arm is still settling or the
#: model is off); that and ``stale_observation`` are worth asking again after the next
#: observation. ``no_planner`` and ``planner_error`` leave the pose unanswered.
ROBODOJO_REASONS = (
    "ik_failed", "goal_below_support", "fk_residual_exceeded", "fk_calibration_mismatch",
    "stale_observation", "no_planner", "planner_error",
)
#: What the SDK route answered before; recorded results of older runs still hold these.
ROBODOJO_SDK_REASONS = (
    "joint_limit", "ik_branch_jump", "local_ik_branch_jump", "self_collision_path",
    INTER_ARM_COLLISION_PATH, "route_contact_approach_exceeds_local_bound",
)
IK_MODEL = "robodojo_reach_ik_depth_world_v1"
CLAIM_BOUNDARY = (
    "One IK solve per pose by the IK solver of the planner move_pose uses, seeded from the measured "
    "joints or a chained query's solution, with joint limits, self-collision and a collision world "
    "built from the current depth images. It plans no path, so a path can still be blocked: "
    "no payload, arrival, grasp or task-success claim."
)
#: LIBERO IK (``backends/libero_pro/reach_check.py``): no seed met
#: the solver's thresholds within its joint-limit and self-collision constraints;
#: a solution whose grip site lands outside the arrival tolerance; and, as on
#: RoboDojo, CuRobo's forward kinematics of the measured joints disagreeing with
#: the measured panda_hand pose, which voids the frame conversion for every query.
LIBERO_REASONS = ("ik_failed", "fk_residual_exceeded", "fk_calibration_mismatch")
BACKEND_REASONS = (
    frozenset(ROBODOJO_REASONS) | frozenset(ROBODOJO_SDK_REASONS) | frozenset(LIBERO_REASONS)
    | {START_UNREACHABLE, INTER_ARM_COLLISION_ENDPOINT}
)
#: RoboDojo names the move_pose route it modelled; the RoboTwin and LIBERO checks model none.
ROBODOJO_ROUTES = ("global", "local", "global_precontact_then_local")

ROW_FIELDS = frozenset({
    "reachable", "reason", "route", "position_error_m", "orientation_error_rad",
    "joint_limit_margin_rad", "joint_delta_rad",
})
BATCH_FIELDS = frozenset({
    "ik_model", "matches_move_pose_admission", "claim_boundary", "results",
})
RESULT_FIELDS = frozenset({
    "schema", "arm", "ik_model", "matches_move_pose_admission",
    "motion_planner_called", "claim_boundary", "results",
})


class ReachCheckRequestDrift(AlignmentRequestDrift):
    """The request or backend row shape is not the one both sides agreed on.

    A correctable input is an ``EnvironmentRefusal``; drift means the two sides
    were built from different revisions of this file, which no policy action
    repairs. It subclasses ``AlignmentRequestDrift`` because both travel over
    the one read-only IK bridge, so a backend that already maps alignment drift
    to its live error maps this one the same way.
    """
