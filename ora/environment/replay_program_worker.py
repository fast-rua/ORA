"""Replay-program execution inside the isolated simulator worker (``environment/replay_program.py``).

``replay_worker.run`` hands a ``ora.replay-program-job.v1`` here. Each program step
runs through the worker's own per-action code (``_replay_robotwin_entry``,
``_replay_libero_entry``) with every physics step captured, so the official predicate
is read at the step the run reached it; after each step the end pose is measured as
the run's receipts recorded it. RoboCasa365 has no per-action replay code: its steps
feed the recorded action vectors through the session's own control step and success
latch. A job with ``scene_trace`` also writes the scene trace of the same re-execution
(``environment/scene_trace.py``). RoboLab's job runs in the environment worker a run
uses, on its session, from the session's ``start``: Kit lives in that process and closes
as a run's does (``robolab_replay.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import replay_worker as worker

PROGRAM_JOB_SCHEMA = worker.PROGRAM_JOB_SCHEMA
PROGRAM_RESULT_SCHEMA = "ora.replay-program-result.v1"
EVERY_STEP = {"mode": "physics_stride_v1", "capture_stride": 1}


def _pose_comparison(replayed: list[float], expected: Any) -> dict[str, Any]:
    import math

    expected = [float(value) for value in expected]
    return {
        "bit_identical": replayed == expected,
        "position_mm": math.dist(replayed[:3], expected[:3]) * 1000.0,
        "replayed": replayed,
    }


def _program_entries(manifest: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    entries: dict[int, Mapping[str, Any]] = {}
    for entry in manifest["entries"]:
        sequence = int(entry["action_sequence"])
        if sequence in entries:
            raise worker.ReplayWorkerError(f"controller replay records action {sequence} twice")
        entries[sequence] = entry
    return entries


def _program_entry(entries: Mapping[int, Mapping[str, Any]], step: Mapping[str, Any]) -> Mapping[str, Any]:
    entry = entries.get(int(step["action_sequence"]))
    if entry is None:
        raise worker.ReplayWorkerError(f"program step {step['action_sequence']} has no recorded action")
    return entry


def _run_robotwin_program(
    *, plan: Mapping[str, Any], manifest: Mapping[str, Any], manifest_path: Path,
    project_root: Path, output_root: Path, program: Mapping[str, Any], scene_trace: Any = None,
) -> dict[str, Any]:
    """Run a replay program's steps and measure each action's end pose and the success latch.

    ``scene_trace`` (``RoboTwinSceneTrace``) sees the same steps; it reads state only.
    """
    entries = _program_entries(manifest)
    session = worker._robotwin_replay_session(plan, project_root, output_root)
    state: dict[str, Any] = {"first_success_action": None}

    def capture(environment: Any, entry: Any, _phase: str, _step: int | None) -> None:
        # Every physics step is captured; the official predicate reads state only.
        if entry is not None and state["first_success_action"] is None and bool(environment.check_success()):
            state["first_success_action"] = int(entry["action_sequence"])

    rows: list[dict[str, Any]] = []
    planner_patch = None
    if not worker._robotwin_replay_needs_planner(manifest):
        planner_patch = worker._disable_unused_robotwin_planners(session.robotwin_root)
    try:
        session.start()
        environment = session._require_environment()
        if scene_trace is not None:
            scene_trace.begin_round(environment, 0)
        current_round = 0
        for step in program["steps"]:
            entry = _program_entry(entries, step)
            round_index = int(entry.get("round_index", 0))
            if round_index != current_round:
                session.restart_round(round_index)
                current_round = round_index
                environment = session._require_environment()
                if scene_trace is not None:
                    scene_trace.begin_round(environment, round_index)
            if scene_trace is not None:
                scene_trace.begin_action(entry)
            ran = worker._replay_robotwin_entry(
                environment, entry, worker._artifact_arrays(manifest_path, entry), sampling=EVERY_STEP,
                capture=capture, hold_steps=step.get("physics_steps") if step["kind"] == "hold" else None,
            )
            if scene_trace is not None:
                scene_trace.end_action()
            expected_kind = {"skip": "skipped"}.get(step["kind"], step["kind"])
            if ran != expected_kind:
                raise worker.ReplayWorkerError(
                    f"program step {step['action_sequence']} is {step['kind']} but its record replays as {ran}"
                )
            row: dict[str, Any] = {"action_sequence": int(step["action_sequence"]), "kind": step["kind"], "arms": {}}
            for arm, pose in (step.get("expect") or {}).items():
                wxyz = [float(value) for value in session._public_pose_wxyz(environment, arm)]
                row["arms"][arm] = _pose_comparison(wxyz[:3] + wxyz[4:7] + wxyz[3:4], pose)
            rows.append(row)
    finally:
        try:
            if scene_trace is not None:
                scene_trace.close()
        finally:
            try:
                session.close()
            finally:
                if planner_patch is not None:
                    robot_class, original = planner_patch
                    robot_class.set_planner = original
    return {"steps": rows, "first_success_action": state["first_success_action"],
            "official_success": state["first_success_action"] is not None}


def _run_libero_program(
    *, plan: Mapping[str, Any], manifest: Mapping[str, Any], manifest_path: Path,
    output_root: Path, program: Mapping[str, Any], scene_trace: Any = None,
) -> dict[str, Any]:
    """Run a replay program's steps and measure each action's grip pose and the first termination.

    ``scene_trace`` (``LiberoSceneTrace``) sees the same steps; it reads state only. Its tap comes off before a
    round's reset, whose settling steps belong to no round.
    """
    entries = _program_entries(manifest)
    session = worker._libero_replay_session(plan, manifest_path, output_root)
    state: dict[str, Any] = {"first_termination_action": None}

    def capture(entry: Any, _phase: str, _step: int | None) -> None:
        if entry is None or state["first_termination_action"] is not None:
            return
        result = session._worker_attr("harness_replay_step_result")
        if isinstance(result, Mapping) and result.get("terminated"):
            state["first_termination_action"] = int(entry["action_sequence"])

    rows: list[dict[str, Any]] = []
    try:
        session.start()
        if scene_trace is not None:
            scene_trace.begin_round(session, 0)
        current_round = 0
        for step in program["steps"]:
            entry = _program_entry(entries, step)
            round_index = int(entry.get("round_index", 0))
            if round_index != current_round:
                if scene_trace is not None:
                    scene_trace.close()
                session.restart_round(round_index)
                current_round = round_index
                if scene_trace is not None:
                    scene_trace.begin_round(session, round_index)
            if scene_trace is not None:
                scene_trace.begin_action(entry)
            ran = worker._replay_libero_entry(
                session, entry, worker._artifact_arrays(manifest_path, entry), capture=capture,
            )
            if scene_trace is not None:
                scene_trace.end_action()
            expected_kind = {"skip": "skipped"}.get(step["kind"], step["kind"])
            if ran != expected_kind:
                raise worker.ReplayWorkerError(
                    f"program step {step['action_sequence']} is {step['kind']} but its record replays as {ran}"
                )
            row: dict[str, Any] = {"action_sequence": int(step["action_sequence"]), "kind": step["kind"], "arms": {}}
            for arm, pose in (step.get("expect") or {}).items():
                position, orientation = session._grip_site_pose()
                row["arms"][arm] = _pose_comparison(
                    [float(value) for value in [*position, *orientation]], pose,
                )
            rows.append(row)
    finally:
        try:
            if scene_trace is not None:
                scene_trace.close()
        finally:
            session.close()
    return {"steps": rows, "first_termination_action": state["first_termination_action"],
            "official_success": state["first_termination_action"] is not None}


def _robocasa_step(session: Any, step: Mapping[str, Any], entry: Mapping[str, Any], commands: Any,
                   scene_trace: Any) -> dict[str, Any]:
    """Run one RoboCasa365 action's control steps as the run did and compare where it ended.

    The sequence names the action in the session's success latch. The state hashes hold the whole scene, so a
    re-execution that left an object behind fails even where the robot's end pose agrees.
    """
    import math

    sequence = int(step["action_sequence"])
    ran = "controls" if entry.get("replayable") else "skip"
    if step["kind"] != ran:
        raise worker.ReplayWorkerError(f"program step {sequence} is {step['kind']} but its record replays as {ran}")
    if ran == "skip" and entry.get("physics_steps"):
        raise worker.ReplayWorkerError(f"RoboCasa action {sequence} ran {entry['physics_steps']} control steps "
                                       "that its record cannot replay")
    io, expect = session.io, step.get("expect") or {}
    session.action_sequence = sequence
    row: dict[str, Any] = {"action_sequence": sequence, "kind": step["kind"], "arms": {}}
    if "state_before" in expect:
        row["arms"]["state_before"] = {"bit_identical": io.state_sha256() == expect["state_before"]}
    if scene_trace is not None:
        scene_trace.begin_action(entry)
    for command in commands:
        session.after_control_step(io.apply(command))
        if scene_trace is not None:
            scene_trace.after_tick()
    if scene_trace is not None:
        scene_trace.end_action()
    if "right" in expect:
        row["arms"]["right"] = _pose_comparison(io.grip_pose(), expect["right"])
    if "base" in expect:
        base = io.base_pose()
        row["arms"]["base"] = {**_pose_comparison(base, expect["base"]),
                               "position_mm": math.dist(base[:2], expect["base"][:2]) * 1000.0}
    if "state_after" in expect:
        row["arms"]["state_after"] = {"bit_identical": io.state_sha256() == expect["state_after"]}
    return row


def _run_robocasa_program(
    *, plan: Mapping[str, Any], manifest: Mapping[str, Any], manifest_path: Path,
    output_root: Path, program: Mapping[str, Any], scene_trace: Any = None,
) -> dict[str, Any]:
    """Run a replay program's steps and measure each action's grip pose, base pose and scene state, and the verdict.

    The session is the run's without a planner: the recorded action vectors need none, whichever servo sent them.
    ``scene_trace`` (``RoboCasaSceneTrace``) sees the same control steps; it reads state only. A new round builds a
    new kitchen from the seed its reset recorded. The seed does not fix a kitchen, so each is built in the region
    order the run kept (``backends/robocasa/construction.py``); a run recorded before that has none, and the state
    hashes refuse a kitchen that came out differently.
    """
    entries = _program_entries(manifest)
    records = [worker._load(path) for path in sorted(manifest_path.parent.glob("region-order-round-*.json"))]
    session = worker._robocasa_replay_session(plan, output_root,
                                              {int(record["round_index"]): record["regions"] for record in records})

    def enter_round(round_index: int) -> None:
        reset = worker._load(manifest_path.parent.parent / "round-resets" / f"round-{round_index:04d}.json")
        session.restart_round(round_index, seed=int(reset["seed"]))
        if scene_trace is not None:
            scene_trace.begin_round(session, round_index)

    rows: list[dict[str, Any]] = []
    try:
        session.start()
        if scene_trace is not None:
            scene_trace.begin_round(session, 0)
        current_round = 0
        for step in program["steps"]:
            entry = _program_entry(entries, step)
            round_index = int(entry.get("round_index", 0))
            if round_index != current_round:
                enter_round(round_index)
                current_round = round_index
            rows.append(_robocasa_step(session, step, entry, worker._artifact_arrays(manifest_path, entry).get("commands", ()),
                                       scene_trace))
        final_round = (program.get("expect") or {}).get("round_index")
        if isinstance(final_round, int) and final_round != current_round:  # a last round with no action in it
            enter_round(final_round)
        verdict = session.persist_official_verdict()
    finally:
        try:
            if scene_trace is not None:
                scene_trace.close()
        finally:
            session.close()
    first = verdict["first_success"] or {}
    return {"steps": rows, "official_success": verdict["official_success"], "control_steps": verdict["control_steps"],
            "first_success_action": first.get("action_sequence"), "first_success_control_step": first.get("control_step")}


def _run_robolab_program(
    *, session: Any, manifest: Mapping[str, Any], manifest_path: Path, run_root: Path,
    program: Mapping[str, Any], scene: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Run a replay program's steps on the started session and measure each action's TCP and the first success.

    Each native action sends its recorded tensor and physics-step joint targets, and the session's own step
    bookkeeping reads the official result, so the verdict latches at the action the run reached it. The tracer
    (``session.scene_trace``) reads state only.
    """
    entries = _program_entries(manifest)
    tracer = session.scene_trace(Path(str(scene["directory"])), run_root.name) if scene else None
    if tracer is not None:
        tracer.begin_round(session, 0)
    current_round = 0
    rows: list[dict[str, Any]] = []
    for step in program["steps"]:
        entry = _program_entry(entries, step)
        round_index = int(entry.get("round_index", 0))
        if round_index != current_round:
            reset = worker._load(run_root / "native" / "round-resets" / f"round-{round_index:04d}.json")
            session.restart_round(round_index, seed=int(reset["seed"]))
            current_round = round_index
            if tracer is not None:
                tracer.begin_round(session, round_index)
        expected_kind = "controls" if entry.get("replayable") else "skip"
        if step["kind"] != expected_kind:
            raise worker.ReplayWorkerError(
                f"program step {step['action_sequence']} is {step['kind']} but its record replays as {expected_kind}"
            )
        session.action_sequence = int(step["action_sequence"])
        if tracer is not None:
            tracer.begin_action(entry)
        arrays = worker._artifact_arrays(manifest_path, entry)
        for native, targets in zip(arrays.get("native_actions", ()), arrays.get("joint_targets", ())):
            session.after_native_action(session.bridge.replay_tick(native, targets))
            if tracer is not None:
                tracer.after_tick()
        session.bridge.take_ticks()
        if tracer is not None:
            tracer.end_action()
        rows.append({"action_sequence": int(step["action_sequence"]), "kind": step["kind"],
                     "arms": {arm: _pose_comparison(session.bridge.tcp_pose(), pose)
                              for arm, pose in (step.get("expect") or {}).items()}})
    first = session.first_success or {}
    return {"steps": rows, "first_termination_action": first.get("action_sequence"),
            "official_success": bool(session.terminated),
            **({"scene_trace": tracer.finish()} if tracer is not None else {})}


def run_program(job: Mapping[str, Any], *, session: Any = None) -> dict[str, Any]:
    """Run one replay program job; writes ``replay-program-result.json`` under the output root.

    ``session`` is the environment worker's started session, for a backend that replays there.
    """
    from ora.environment.controller_replay import load_controller_replay_manifest

    run_root = Path(str(job["run_root"])).resolve()
    output_root = Path(str(job["output_root"])).resolve()
    project_root = Path(str(job["project_root"])).resolve()
    if output_root == run_root or output_root.is_relative_to(run_root):
        raise worker.ReplayWorkerError("replay program output must remain outside the run")
    program = job.get("program")
    if not isinstance(program, Mapping) or not isinstance(program.get("steps"), list):
        raise worker.ReplayWorkerError("replay program job omits its steps")
    plan = worker._load(run_root / "plan.json")
    manifest_path = run_root / "native/controller-replay/manifest.json"
    manifest = load_controller_replay_manifest(manifest_path)
    if manifest["backend"] != program.get("backend"):
        raise worker.ReplayWorkerError("replay program and run record different backends")
    output_root.mkdir(parents=True, exist_ok=True)
    scene = job.get("scene_trace")
    tracer = None
    if manifest["backend"] == "robotwin":
        tracer = worker._robotwin_scene_trace(Path(str(scene["directory"])), run_root.name) if scene else None
        measured = _run_robotwin_program(
            plan=plan, manifest=manifest, manifest_path=manifest_path,
            project_root=project_root, output_root=output_root, program=program, scene_trace=tracer,
        )
    elif manifest["backend"] == "libero_pro":
        tracer = worker._libero_scene_trace(Path(str(scene["directory"])), run_root.name) if scene else None
        measured = _run_libero_program(
            plan=plan, manifest=manifest, manifest_path=manifest_path,
            output_root=output_root, program=program, scene_trace=tracer,
        )
    elif manifest["backend"] == "robocasa":
        tracer = worker._robocasa_scene_trace(Path(str(scene["directory"])), run_root.name) if scene else None
        measured = _run_robocasa_program(
            plan=plan, manifest=manifest, manifest_path=manifest_path,
            output_root=output_root, program=program, scene_trace=tracer,
        )
    elif manifest["backend"] == "robolab" and session is not None:
        measured = _run_robolab_program(session=session, manifest=manifest, manifest_path=manifest_path,
                                        run_root=run_root, program=program, scene=scene)
    else:
        raise worker.ReplayWorkerError("unsupported controller replay backend")
    if tracer is not None:
        measured["scene_trace"] = tracer.finish()
    result = {"schema": PROGRAM_RESULT_SCHEMA, "backend": manifest["backend"], "source_run_id": run_root.name,
              "source_manifest": {"path": "native/controller-replay/manifest.json", "sha256": worker._hash(manifest_path)},
              **measured}
    worker._atomic_json(output_root / "replay-program-result.json", result)
    return result


__all__ = ["PROGRAM_RESULT_SCHEMA", "run_program"]
