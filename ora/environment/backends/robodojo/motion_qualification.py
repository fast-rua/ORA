"""Trusted scripted qualification for the RoboDojo motion v3 dispatch.

This is the RoboDojo counterpart of the LIBERO sparse-motion gate: a closed
case table exercised against the real dual ARX X5 plant in the isolated Isaac
worker through the exact public v3 contract (never a policy model). A passing
device-keyed record is the admission evidence for adding ``robodojo`` to the
sparse-motion qualification table -- the table row and the record land in the
same change.

Every arm move is the one an episode runs: a ``follow_joint_trajectory`` the
host plans with the toolkit lib (``plan``, injected; the environment never
imports the agent), a recovery trajectory (home, or a retrace of the last
move), the rate-limited ``set_gripper`` and ``wait``. The judgement layer
consumes only public receipt fields and the host's refusals, and is unit tested
with a fake boundary; the CLI assembles the live worker from the explicit
runtime configuration through the same bootstrap formal runs use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ora.contracts.joint_trajectory import TRAJECTORY_ACTION
from ora.contracts.wait import WAIT_MAX_TICKS

from .motion_checks import (
    ALL_CASE_IDS,
    ARM,
    FREE_ROUTES,
    JAW_APERTURE_CASE_ID,
    JAW_APERTURE_PROBE_ID,
    JAW_CALIBRATION_SCHEMA,
    JAW_CONTACT_LEG_REASON,
    JAW_CONTACT_LEG_STATUS,
    JAW_SWEEPS,
    MOTION_ACTION_V3_SCHEMA,
    MOTION_CASES,
    OTHER_ARM,
    SCHEMA,
    SEQUENCE_CASES,
    UNREACHABLE_POSITION,
    UNREACHABLE_REFUSAL_CODES,
    X5_BASE_MOUNTS_M,
    X5_LINK6_REACH_BOUND_M,
    _pose,
    _pose_error,
    build_case_knots,
    build_case_request,
    derive_jaw_step,
    jaw_aperture_calibration,
    judge_gripper_case,
    judge_jaw_aperture_sweeps,
    judge_motion_case,
    judge_recovery_case,
    judge_refusal,
    knot_arguments,
    measured_joints,
    measured_pad_face_separation_m,
    nominal_jaw_aperture_m,
    summarize_receipt,
)


def build_qualification_record(
    *, source_commit: str, host: Mapping[str, Any], session_identity: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    executed = {str(result.get("case_id")) for result in case_results}
    missing = [case_id for case_id in ALL_CASE_IDS if case_id not in executed]
    record = {
        "schema": SCHEMA, "source_commit": source_commit, "host": dict(host),
        "session": dict(session_identity), "evidence": dict(evidence or {}),
        "cases": [dict(result) for result in case_results], "missing_cases": missing,
        "passed": not missing and all(bool(result.get("passed")) for result in case_results),
        "created_unix_s": time.time(),
    }
    body = {key: value for key, value in record.items() if key != "created_unix_s"}
    record["record_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return record


def _action_id(step: int) -> str:
    return f"qualification-{step:04d}"


def _v3_action(step: int, observation: Mapping[str, Any], arguments: Mapping[str, Any],
               name: str) -> dict[str, Any]:
    return {
        "schema": MOTION_ACTION_V3_SCHEMA, "observation_id": str(observation["observation_id"]),
        "action_id": _action_id(step),
        "bound_observation_sha256": str(observation["observation_sha256"]),
        "action": name, "arguments": dict(arguments),
    }


def _probe_request(observation: Mapping[str, Any], pose: Sequence[float], policy: str,
                   offset: Sequence[float] = (-0.06, 0.0, 0.0)) -> dict[str, Any]:
    return {
        "schema": "ora.motion-probe-request.v1",
        "observation_id": str(observation["observation_id"]),
        "observation_sha256": str(observation["observation_sha256"]),
        "variant_policy": policy,
        "queries": [{"query_id": "q0", "arm": ARM, "pose": _pose(
            [float(pose[i]) + float(offset[i]) for i in range(3)], pose[3:],
        )}],
    }


def _lift(pose: Sequence[float], height_m: float) -> dict[str, Any]:
    return {"kind": "move", "targets": {ARM: [pose[0], pose[1], pose[2] + height_m, *pose[3:]]},
            "interactions": {ARM: "free_space"}}


class _Runner:
    """Drive every case through the injected live boundary, recording each result."""

    def __init__(self, *, observe: Callable[[], Mapping[str, Any]],
                 execute: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                 plan: Callable[[Mapping[str, Any], Mapping[str, Any], str], Mapping[str, Any]],
                 probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                 restart_round: Callable[[int], Mapping[str, Any]],
                 status: Callable[[], Mapping[str, Any]],
                 persist_verdict: Callable[[], Mapping[str, Any]],
                 current_poses: Callable[[Mapping[str, Any]], Mapping[str, Sequence[float]]],
                 native_action_limit: int, free_route: str,
                 measure_jaw_state: Callable[[], Mapping[str, Any]] | None = None) -> None:
        if free_route not in FREE_ROUTES:
            raise ValueError(f"free_route must be one of {FREE_ROUTES}")
        self.observe, self.execute, self.plan, self.probe = observe, execute, plan, probe
        self.restart_round, self.status, self.persist_verdict = restart_round, status, persist_verdict
        self.current_poses = current_poses
        self.measure_jaw_state = measure_jaw_state
        self.native_action_limit = int(native_action_limit)
        self.free_route = free_route
        self.results: list[dict[str, Any]] = []
        self.step = 0
        self.home_poses: dict[str, list[float]] = {}

    def record(self, case_id: str, judgement: Mapping[str, Any], receipt: Mapping[str, Any] | None,
               **extra: Any) -> None:
        self.results.append({
            "case_id": case_id, "passed": bool(judgement.get("passed")),
            "notes": list(judgement.get("notes", [])),
            "receipt": summarize_receipt(receipt) if receipt is not None else None, **extra,
        })

    def fail(self, case_id: str, error: BaseException, stage: str = "execution") -> None:
        self.results.append({"case_id": case_id, "passed": False,
                             "notes": [f"{stage} error: {error}"], "receipt": None})

    def act(self, observation: Mapping[str, Any], arguments: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        self.step += 1
        result = self.execute(_v3_action(self.step, observation, arguments, name))
        return result["receipt"] if isinstance(result, Mapping) and "receipt" in result else result

    def move(self, observation: Mapping[str, Any], request: Mapping[str, Any]
             ) -> tuple[Mapping[str, Any] | None, Mapping[str, Any]]:
        """Have the host plan ``request`` and run its trajectory: ``(receipt, planned)``, no receipt if refused.

        ``planned`` is ``{"action", "route"}`` or ``{"refusal": {"code", "message", ...}}``."""
        self.step += 1
        planned = self.plan(observation, request, _action_id(self.step))
        if "refusal" in planned:
            return None, planned
        result = self.execute(planned["action"])
        return (result["receipt"] if isinstance(result, Mapping) and "receipt" in result else result), planned

    def run_motion_matrix(self) -> None:
        first = self.observe()
        self.home_poses = {arm: [float(v) for v in pose] for arm, pose in self.current_poses(first).items()}
        observation = first
        for case in MOTION_CASES:
            case_id = str(case["case_id"])
            try:
                if case.get("knots"):
                    knots = build_case_knots(case_id, measured_joints(observation)[ARM])
                    receipt = self.act(observation, knot_arguments(knots), TRAJECTORY_ACTION)
                    self.record(case_id, judge_motion_case(case, receipt, free_route=self.free_route), receipt)
                else:
                    request = build_case_request(case_id, self.current_poses(observation))
                    receipt, planned = self.move(observation, request)
                    if receipt is None:
                        self.record(case_id, judge_refusal(case, planned["refusal"], free_route=self.free_route),
                                    None, refusal=dict(planned["refusal"]))
                    else:
                        self.record(case_id, judge_motion_case(
                            case, receipt, targets=request["targets"], route=planned.get("route"),
                            free_route=self.free_route), receipt, route=planned.get("route"))
            except Exception as error:  # recorded, never silently dropped
                self.fail(case_id, error)
            observation = self.observe()

    def run_sequence(self) -> None:
        try:
            observation = self.observe()
            receipt, planned = self.move(observation, {"kind": "home", "arms": [ARM, OTHER_ARM]})
            if receipt is None:
                self.record("recover_home_after_matrix", {"passed": False, "notes": [
                    f"home recovery refused: {planned['refusal']}"]}, None)
            else:
                judgement = judge_recovery_case(receipt, self.home_poses[ARM], strategy="home")
                other = judge_recovery_case(receipt, self.home_poses[OTHER_ARM], strategy="home", arm=OTHER_ARM)
                self.record("recover_home_after_matrix",
                            {"passed": judgement["passed"] and other["passed"],
                             "notes": judgement["notes"] + other["notes"]}, receipt, route=planned.get("route"))
        except Exception as error:
            self.fail("recover_home_after_matrix", error)
        for case_id, directive in (("gripper_close_holds_pose", "close"), ("gripper_open_holds_pose", "open")):
            try:
                observation = self.observe()
                receipt = self.act(observation, {"targets": {ARM: {"opening": directive}}}, "set_gripper")
                self.record(case_id, judge_gripper_case(directive, receipt), receipt)
            except Exception as error:
                self.fail(case_id, error)
        # Held out of the matrix with its case id; see SEQUENCE_CASES.
        # self.run_jaw_aperture_probe()
        try:
            observation = self.observe()
            anchor = [float(v) for v in self.current_poses(observation)[ARM]]
            lifted, _planned = self.move(observation, _lift(anchor, 0.05))
            observation = self.observe()
            receipt, planned = self.move(observation, {"kind": "retrace", "arms": [ARM]})
            if receipt is None:
                judgement = {"passed": False, "notes": [f"retrace refused: {planned['refusal']}"]}
            else:
                judgement = judge_recovery_case(receipt, anchor, strategy="retrace")
            if lifted is None or lifted.get("execution_status") != "reached":
                judgement = {"passed": False, "notes": judgement["notes"] + ["lift before the retrace did not reach"]}
            self.record("recover_retrace_after_move", judgement, receipt, route=planned.get("route"))
        except Exception as error:
            self.fail("recover_retrace_after_move", error)
        self.run_probe_cases()
        try:
            verdict = self.persist_verdict()
            terminal = self.status()
            passed = (verdict.get("terminal") is False and verdict.get("official_success") is None
                      and terminal.get("terminal") is False)
            self.record("official_preterminal_boundary", {"passed": passed, "notes": [] if passed else [
                f"preterminal verdict {verdict.get('official_success')!r} terminal {verdict.get('terminal')!r}",
            ]}, None, verdict={key: verdict.get(key) for key in ("terminal", "official_success", "native_actions")})
        except Exception as error:
            self.fail("official_preterminal_boundary", error)

    def run_jaw_aperture_probe(self) -> None:
        """Drive ``robodojo_x5_jaw_aperture_probe_v1``'s free-air legs.

        Both arms, both commands the public ``set_gripper`` contract admits,
        twice, in free air. Every command goes through the exact public v3
        boundary a policy uses, in the rate-limited form every surface sends;
        only the *reading* is private, and it comes from the worker's
        qualification-only jaw-state method.
        """

        if self.measure_jaw_state is None:
            self.results.append({
                "case_id": JAW_APERTURE_CASE_ID, "passed": False,
                "notes": ["no jaw-state measurement boundary was injected"],
                "receipt": None,
            })
            return
        try:
            arms = (ARM, OTHER_ARM)
            sweeps: dict[str, list[list[dict[str, Any]]]] = {
                arm: [] for arm in arms
            }
            for _sweep in range(JAW_SWEEPS):
                for arm in arms:
                    steps: list[dict[str, Any]] = []
                    for directive, command in (("open", 1.0), ("close", 0.0)):
                        observation = self.observe()
                        self.act(
                            observation, {"targets": {arm: {"opening": directive}}},
                            "set_gripper",
                        )
                        reading = self.measure_jaw_state()
                        steps.append(derive_jaw_step(
                            command=command, reading=reading["arms"][arm],
                        ))
                    sweeps[arm].append(steps)
            judgements = {
                arm: judge_jaw_aperture_sweeps(sweeps[arm]) for arm in arms
            }
            notes = [
                f"{arm}: {note}"
                for arm in arms for note in judgements[arm]["notes"]
            ]
            self.results.append({
                "case_id": JAW_APERTURE_CASE_ID,
                "passed": not notes,
                "notes": notes,
                "receipt": None,
                "probe_id": JAW_APERTURE_PROBE_ID,
                "arms": list(arms),
                "sweeps": [sweep for arm in arms for sweep in sweeps[arm]],
                "contact_leg": {
                    "status": JAW_CONTACT_LEG_STATUS,
                    "reason": JAW_CONTACT_LEG_REASON,
                },
                "command_boundary": (
                    "public_set_gripper_open_close_only_no_intermediate_"
                    "normalized_command_is_expressible"
                ),
            })
        except Exception as error:
            self.fail(JAW_APERTURE_CASE_ID, error)

    def run_probe_cases(self) -> None:
        try:
            observation = self.observe()
            pose = self.current_poses(observation)[ARM]
            variants = self.probe(_probe_request(observation, pose, "parallel_jaw_equivalents"))["results"][0].get("variants", [])
            passed = len(variants) == 2
            self.record("jaw_flip_variants_probed", {"passed": passed, "notes": [] if passed else [f"{len(variants)} variants"]},
                        None, variants=[{"variant_id": v.get("variant_id"), "reachable": v.get("reachable")} for v in variants])
        except Exception as error:
            self.fail("jaw_flip_variants_probed", error)
        try:
            observation = self.observe()
            pose = self.current_poses(observation)[ARM]
            probed = self.probe({
                **_probe_request(observation, pose, "exact"),
                "queries": [{"query_id": "q0", "arm": ARM, "pose": _pose(UNREACHABLE_POSITION, pose[3:])}],
            })["results"][0]
            receipt, planned = self.move(observation, {"kind": "move", "targets": {ARM: [*UNREACHABLE_POSITION, *pose[3:]]},
                                                       "interactions": {ARM: "free_space"}})
            notes: list[str] = []
            if probed.get("reachable") is not False:
                notes.append("probe admitted the unreachable pose")
            if receipt is not None:
                notes.append(f"the unreachable move ran: {receipt.get('execution_status')} with "
                             f"{receipt.get('native_actions')} native actions")
            elif planned["refusal"].get("code") not in UNREACHABLE_REFUSAL_CODES[self.free_route]:
                # The probe and the planner must agree on *why*: any other refusal proves nothing about reach.
                notes.append(f"the move was refused with {planned['refusal'].get('code')!r}, not as unreachable")
            self.record("probe_consistency_unreachable", {"passed": not notes, "notes": notes}, receipt,
                        refusal=None if receipt is not None else dict(planned["refusal"]))
        except Exception as error:
            self.fail("probe_consistency_unreachable", error)

    def run_round_isolation(self) -> None:
        try:
            terminal_before = self.status().get("terminal")
            reset = self.restart_round(1)
            notes: list[str] = []
            if reset.get("schema") != "ora.robodojo-round-reset.v1" or reset.get("round_index") != 1:
                notes.append(f"round reset record {reset.get('schema')} / {reset.get('round_index')}")
            if reset.get("seed") != 0 or reset.get("reward_registered") is not True:
                notes.append("round reset did not rebind seed 0 with a registered evaluator")
            if terminal_before is not True or self.status().get("terminal") is not False:
                notes.append("round reset did not revoke the previous round's terminal state")
            fresh = self.observe()
            if fresh.get("round_index") != 1 or fresh.get("previous_execution_receipt") is not None:
                notes.append("first observation of the new round carries previous-round state")
            distance, _angle = _pose_error(self.home_poses[ARM], self.current_poses(fresh)[ARM])
            if distance > 0.02:
                notes.append(f"round reset left the arm {distance:.3f} m from its round-0 home")
            pose = self.current_poses(fresh)[ARM]
            receipt, planned = self.move(fresh, _lift(pose, 0.04))
            if receipt is None:
                notes.append(f"round-1 move refused: {planned['refusal']}")
            else:
                if receipt.get("round_index") != 1 or receipt.get("execution_status") != "reached":
                    notes.append(f"round-1 move ended {receipt.get('execution_status')} in round {receipt.get('round_index')}")
                verdict = self.persist_verdict()
                if verdict.get("round_index") != 1 or verdict.get("native_actions") != receipt.get("native_actions"):
                    notes.append("round-1 verdict counters did not restart with the round")
            self.record("round_reset_isolation", {"passed": not notes, "notes": notes}, receipt,
                        reset={key: reset.get(key) for key in ("schema", "round_index", "seed", "artifact_sequences_preserved")})
        except Exception as error:
            self.fail("round_reset_isolation", error)

    def run_terminal_boundary(self) -> None:
        try:
            holds = 0
            last_receipt: Mapping[str, Any] | None = None
            while not self.status().get("terminal") and holds < self.native_action_limit + 5:
                observation = self.observe()
                last_receipt = self.act(observation, {"ticks": WAIT_MAX_TICKS}, "wait")
                holds += 1
            terminal = self.status()
            verdict = self.persist_verdict()
            notes: list[str] = []
            if terminal.get("terminal") is not True or not str(terminal.get("reason", "")).startswith("official_"):
                notes.append(f"episode did not end on the official boundary: {terminal}")
            if verdict.get("terminal") is not True or not isinstance(verdict.get("official_success"), bool):
                notes.append(f"official verdict is not a terminal Boolean: {verdict.get('official_success')!r}")
            if verdict.get("native_actions") != self.native_action_limit and verdict.get("official_success") is not True:
                notes.append(f"native actions {verdict.get('native_actions')} differ from the task limit")
            if last_receipt is not None and last_receipt.get("terminal_after_action") is not True:
                notes.append("final receipt did not observe the terminal transition")
            try:
                observation = self.observe()
                self.act(observation, {"ticks": 1}, "wait")
                notes.append("terminated environment accepted another action")
            except Exception:
                pass
            for _ in range(3):
                if self.status().get("terminal") is not True:
                    notes.append("terminal status changed on a read")
            after = self.persist_verdict()
            for key in ("official_success", "terminal", "verdict_basis", "fault_reason",
                        "native_actions", "physics_steps", "abort_reason"):
                if after.get(key) != verdict.get(key):
                    notes.append(f"post-terminal verdict changed: {key}")
            self.record("official_terminal_boundary_intact", {"passed": not notes, "notes": notes}, last_receipt,
                        holds=holds, verdict={key: verdict.get(key) for key in (
                            "terminal", "official_success", "native_actions", "physics_steps", "round_index")})
        except Exception as error:
            self.fail("official_terminal_boundary_intact", error)


def run_cases(
    *, observe: Callable[[], Mapping[str, Any]],
    execute: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    plan: Callable[[Mapping[str, Any], Mapping[str, Any], str], Mapping[str, Any]],
    probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    restart_round: Callable[[int], Mapping[str, Any]],
    status: Callable[[], Mapping[str, Any]],
    persist_verdict: Callable[[], Mapping[str, Any]],
    current_poses: Callable[[Mapping[str, Any]], Mapping[str, Sequence[float]]],
    native_action_limit: int,
    free_route: str,
    measure_jaw_state: Callable[[], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Drive every case through the injected live boundary.

    ``observe`` returns the fresh public observation with an added
    ``observation_sha256``; ``current_poses`` extracts both measured link6
    poses from it. ``plan(observation, request, action_id)`` is the host's
    planner: ``request`` is a move (``kind`` ``move``: link6 ``targets`` and
    their ``interactions``) or a recovery (``kind`` ``home`` or ``retrace``:
    ``arms``); it returns the bound ``follow_joint_trajectory`` (``action``,
    ``route``) or the toolkit's ``refusal`` before any action. ``free_route``
    names the route a free-space move must take (``FREE_ROUTES``). ``execute``
    lends each action its native-action budget. Round 0 runs the motion matrix
    and the sequence cases and then drives the official native-action boundary
    to its terminal transition with waits; the round reset that a persistent
    session performs after that boundary is checked last, in round 1.
    """
    runner = _Runner(observe=observe, execute=execute, plan=plan, probe=probe, restart_round=restart_round,
                     status=status, persist_verdict=persist_verdict, current_poses=current_poses,
                     native_action_limit=native_action_limit, free_route=free_route,
                     measure_jaw_state=measure_jaw_state)
    runner.run_motion_matrix()
    runner.run_sequence()
    runner.run_terminal_boundary()
    runner.run_round_isolation()
    return runner.results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="robodojo-motion-qualification",
        description=(
            "Run the scripted RoboDojo motion v3 qualification on the qualified host. "
            "Requires the explicit runtime configuration JSON (the compatibility package's "
            "runtime_configuration) and writes one device-keyed record."
        ),
    )
    parser.add_argument("--runtime-config", type=Path, required=True, help="RoboDojo runtime configuration JSON")
    parser.add_argument("--output", type=Path, required=True, help="evidence directory for the worker and record")
    parser.add_argument("--gate", type=Path, default=None,
                        help="optional device-keyed gate path receiving the identical record bytes")
    arguments = parser.parse_args(argv)
    raise SystemExit(
        "Use `ora qualify robodojo motion` with the same --runtime-config, --output and --gate arguments. "
        "The host launcher and execution-state owner now live in the orchestrator. "
        "No worker was started and no gate record was written."
    )


__all__ = [
    "ALL_CASE_IDS", "JAW_APERTURE_CASE_ID", "JAW_APERTURE_PROBE_ID",
    "JAW_CALIBRATION_SCHEMA", "MOTION_CASES", "SCHEMA", "SEQUENCE_CASES",
    "build_case_knots", "build_case_request", "build_qualification_record", "derive_jaw_step",
    "jaw_aperture_calibration", "judge_gripper_case", "judge_jaw_aperture_sweeps",
    "judge_motion_case", "judge_recovery_case", "judge_refusal", "measured_pad_face_separation_m",
    "nominal_jaw_aperture_m", "run_cases", "summarize_receipt", "main",
    # The refusal target and the reach bound it is proven against, for the card probe's tests.
    "UNREACHABLE_POSITION", "X5_BASE_MOUNTS_M", "X5_LINK6_REACH_BOUND_M",
]


if __name__ == "__main__":  # pragma: no cover - exercised on the GPU host
    import sys as _sys

    _sys.exit(main())
