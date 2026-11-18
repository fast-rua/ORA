"""One planned move as one ``follow_joint_trajectory`` action, and its compact outcome.

Replay: the action is rebuilt from recorded inputs and must equal the recorded
``execute_action`` request; it never executes in replay (``control.execute_action``).
The environment interpolates every physics step between the knots with a
velocity feed-forward, guards tracking (default: a quarter of the knot
step, at least 0.012 rad, or a lag that keeps growing) and progress per
physics step, holds the measured joints if a guard trips, and settles only as
asked.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from ...interface.api import DataInterface
from ...interface.call_note import noted
from ..values import PublicResult

ACTION = "follow_joint_trajectory"


def trajectory_action(*, action_id: str, observation: Mapping, knots: Mapping[str, Sequence[Sequence[float]]],
                      expected: Mapping[str, Sequence[Sequence[float]]] | None = None,
                      settle: Mapping | None = None, guards: Mapping | None = None,
                      gripper_knots: Mapping[str, Sequence[float]] | None = None,
                      max_lead_m: float | None = None, part_of_move: Mapping | None = None,
                      speed_segments: Mapping[str, Sequence[Mapping]] | None = None,
                      expect_contact: bool = False, jaw_symmetry: Mapping[str, str] | None = None) -> dict:
    """Pure computation: the bound action for the observation identity ``observation``.

    ``expect_contact`` marks a deliberate push or press: the environment turns off its growing-error tracking
    rule (the tracking bound and the progress guard still stop a real block). ``jaw_symmetry`` (per arm, a
    ``orientation.JAW_SYMMETRY_LABELS`` label) is echoed by the receipt.

    ``part_of_move`` (``{"fraction": f, "goals_xyz_xyzw": {arm: pose}}``) says the knots end short of the
    move's goal (cut to fit a budget): the environment then reports the action not_reached
    (``native_action_budget_exhausted``) with the measured distance left to each goal, even when every knot ran.
    ``speed_segments`` (per arm, ``[{"speed": label, "first_knot": k, ...}]``) names the knots of each speed;
    the receipt reports the native actions each spent.
    Example: trajectory_action(action_id='action-3', observation=obs.identity.json(), knots={'left': rows})."""
    arms = {}
    for arm, rows in knots.items():
        entry = {"knots_rad": [[float(v) for v in row] for row in rows]}
        if expected and arm in expected:
            entry["expected_link6_xyz_xyzw"] = [[float(v) for v in pose] for pose in expected[arm]]
        if gripper_knots and arm in gripper_knots:
            entry["gripper_knots"] = [float(v) for v in gripper_knots[arm]]
        arms[arm] = entry
    arguments = {"arms": arms, "interpolation": "knots_linear_ff"}
    if guards is not None:
        arguments["guards"] = dict(guards)
    if settle is not None:
        arguments["settle"] = dict(settle)
    if max_lead_m is not None:
        arguments["gripper"] = {"max_lead_m": float(max_lead_m)}
    if speed_segments is not None:
        arguments["speed_segments"] = {arm: [dict(segment) for segment in segments]
                                       for arm, segments in speed_segments.items()}
    if expect_contact:
        arguments["expect_contact"] = True
    if jaw_symmetry:
        arguments["jaw_symmetry"] = {arm: str(label) for arm, label in jaw_symmetry.items()}
    if part_of_move is not None:
        arguments["part_of_move"] = {"fraction": float(part_of_move["fraction"]),
                                     "goals_xyz_xyzw": {arm: [float(v) for v in pose] for arm, pose
                                                        in part_of_move["goals_xyz_xyzw"].items()}}
    return {"schema": "ora-motion-action.v3", "action": ACTION, "action_id": action_id,
            "observation_id": observation["observation_id"],
            "bound_observation_sha256": observation["observation_sha256"], "arguments": arguments}


def follow(api: DataInterface, action: Mapping, *, settle_native_actions: int = 0, route: str | None = None) -> dict:
    """Replay: recorded primitive. Execute the trajectory as one action; returns ``receipt``, ``remaining_budget``.

    The budget asked for is the knots plus the settle. ``route`` (``planner.PathPlan.route``) goes on the toolkit
    call's evidence, not in the action. Example: follow(api, action, settle_native_actions=3, route='line')."""
    count = len(next(iter(action["arguments"]["arms"].values()))["knots_rad"])
    with noted(route=route):
        return PublicResult.of(api.act.execute_action(dict(action), count + int(settle_native_actions))).data.json()


def gripper_action(*, action_id: str, observation: Mapping, targets: Mapping[str, Mapping]) -> dict:
    """Pure computation: one rate-limited ``set_gripper`` (the object target form) for ``observation``.

    ``targets`` maps an arm to ``{"opening": "open" | "close" | 0..1, "rate_per_s": openings per second or
    None (full rate), "max_lead_m": the most the drive target may lead the measured finger (<= 0.0108) or None
    (the SDK's 10.8 mm), and optionally "arm_settle": {"joint_speed_rad_s", "max_native_actions"} and "end":
    {"tolerance", "stall_window_native_actions", "stall_still_step"}}``. The ramp starts from the drive target
    the plant still applies, so a held squeeze is kept. A plain "open"/"close" target is the released
    full-drive command (what every backend runs). Run it with ``api.act.execute_action(action, budget)``.
    Example: gripper_action(action_id='action-5', observation=obs.identity.json(),
    targets={'left': {'opening': 'close', 'rate_per_s': 1.0, 'max_lead_m': 0.004}})."""
    return {"schema": "ora-motion-action.v3", "action": "set_gripper", "action_id": action_id,
            "observation_id": observation["observation_id"],
            "bound_observation_sha256": observation["observation_sha256"],
            "arguments": {"targets": {arm: dict(target) if isinstance(target, Mapping) else target
                                      for arm, target in targets.items()}}}


def summary(receipt: Mapping) -> dict:
    """Pure computation: the compact outcome a tool returns (status, stop, residuals). Example: summary(receipt)."""
    evidence = (receipt.get("stage_evidence") or {}).get("trajectory") or {}
    return {key: evidence[key] for key in ("status", "knots_planned", "knots_executed", "stop", "settle", "arms",
                                           "speed_segments") if key in evidence}


__all__ = ["ACTION", "follow", "gripper_action", "summary", "trajectory_action"]
