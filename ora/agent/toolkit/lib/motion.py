"""Read-only worker queries. No online fallback or claim beyond the backend's public diagnostics."""
from ..interface.api import DataInterface
from .values import JsonValue, PublicResult


def probe_motion(api: DataInterface, request: JsonValue) -> PublicResult:
    """Replay: from record; changed/unrecorded IK target fails NotRecorded, never calls worker.

    Example: probe_motion(api, JsonValue.of(existing_probe_request)). Keep the complete
    versioned request (world position_m/xyzw, query order, arm, variant policy, observation
    and mutation bindings) and all public variants/diagnostics/handles. Joint deltas rad,
    durations s, endpoint errors m/rad. Does not select a winning candidate.
    """
    return PublicResult.of(api.planner.probe_motion(request=request.json()))


def align_plan(api: DataInterface, request: JsonValue) -> PublicResult:
    """Replay: from record; new pose batch fails NotRecorded. No implicit replan.

    Example: align_plan(api, JsonValue.of(existing_align_request)). Expose all 1–12
    arbitrary world poses allowed by the worker schema, all per-arm/source-index rows,
    position error m, orientation error rad, iterations and claim boundary. No generated rolls.
    """
    return PublicResult.of(api.planner.align_plan(request=request.json()))


def reach_check(api: DataInterface, request: JsonValue) -> PublicResult:
    """Replay: from record; new query fails NotRecorded, not an online reach computation.

    Example: reach_check(api, JsonValue.of(existing_reach_request)). Preserve complete
    public diagnostics, original from_index ordering, world metre/xyzw targets and
    radian joint margins. IK feasibility does not assert path/collision parity.
    """
    return PublicResult.of(api.planner.reach_check(request=request.json()))


def kinematics(api: DataInterface, request: JsonValue) -> PublicResult:
    """Replay: from record; a changed request fails NotRecorded, never an online IK solve.

    Example: kinematics(api, JsonValue.of(existing_kinematics_request)). The read-only
    ``ora.kinematics-request.v1`` query: IK chains seeded from the previous solution, FK and
    joint-path screens, joints in radians and poses in world metres/xyzw. lib.motion_plan.kinematics
    builds these requests for planned motion.
    """
    return PublicResult.of(api.planner.kinematics(request=request.json()))
