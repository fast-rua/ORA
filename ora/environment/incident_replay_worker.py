"""One incident of a sealed run, inside the simulator worker (``orchestrator/evaluation/incident_replay.py``).

The worker builds the session the job's configuration names (``worker._build_session``,
the path a live episode takes), runs the actions before the incident through the Viewer
replay's per-action code (``replay_worker._replay_robotwin_entry``,
``_replay_libero_entry``), and checks each one's end pose against its receipt. Then one
branch runs the incident action:

* ``recorded``: its recorded controls, as the prefix ran them;
* ``current``: the action's recorded target through ``session.execute``, so this
  checkout's admission, planner, timing and executors decide what happens.

The prefix bypasses ``session.execute``, so before a ``current`` branch the session is told
what those actions left behind (:func:`synchronise`): the step counter, the last receipt and
each arm's gripper directive, without which a held object would be let go. It
never learns the recovery targets: a ``recover_motion`` incident is refused on the host.

Scene poses are private simulator state, for the developer's report only: they are written
under the job's output and reach no policy-readable path.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.contracts.action import MOTION_ACTION_V3_SCHEMA

from . import replay_worker as worker
from .replay_program_worker import EVERY_STEP, _pose_comparison, _program_entries, _program_entry

JOB_SCHEMA = "ora.incident-replay-job.v1"
RESULT_SCHEMA = "ora.incident-replay-result.v1"
BRANCHES = ("recorded", "current")
#: Smaller movements of a body are left out of a report's ``moved`` list.
MOVED_BELOW_MM = 2.0
MOVED_BELOW_DEG = 2.0


def gripper_directives(receipts: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Each arm's last ``set_gripper`` target among ``receipts``, in order.

    A sparse receipt names ``targets``; LIBERO's executor receipt a normalized ``gripper``
    for its one arm, labelled as ``motion_evidence`` labels it.
    """

    directives: dict[str, str] = {}
    for receipt in receipts:
        if receipt.get("action") != "set_gripper":
            continue
        arguments = receipt.get("arguments") or {}
        targets = arguments.get("targets")
        if isinstance(targets, Mapping):
            directives.update({str(arm): str(target) for arm, target in targets.items()})
        elif arguments.get("gripper") in (0.0, 1.0):
            directives[str(arguments.get("arm", "right"))] = "open" if arguments["gripper"] == 1.0 else "close"
    return directives


def synchronise(session: Any, *, backend: str, receipts: Sequence[Mapping[str, Any]],
                previous: Mapping[str, Any] | None, steps_ran: int) -> dict[str, Any]:
    """Give the session what the actions it did not execute left behind; returns what it set.

    Its action counter stays at zero: the incident is the first action this output records,
    and its controller replay is append-only from there.
    """

    directives = gripper_directives(receipts)
    session.last_receipt = None if previous is None else dict(previous)
    if backend == "robotwin":
        session.physics_step_sequence = int(steps_ran)
        session.gripper_command_targets.update(directives)
    else:
        session.sim_steps = int(steps_ran)
        if "right" in directives:
            session._gripper_command_target = directives["right"]
        session.last_raw_observation = session._raw_observation()
    return {"steps": int(steps_ran), "gripper_directives": directives}


def _rotation_deg(first: Sequence[float], second: Sequence[float]) -> float:
    dot = abs(sum(float(a) * float(b) for a, b in zip(first, second)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def scene_changes(before: Mapping[str, Sequence[float]] | None,
                  after: Mapping[str, Sequence[float]] | None) -> dict[str, Any] | None:
    """Bodies that moved between two scene reads, largest first; ``None`` when one read failed."""

    if before is None or after is None:
        return None
    moved = []
    for name in sorted(set(before) & set(after)):
        first, second = before[name], after[name]
        distance = math.dist(first[:3], second[:3]) * 1000.0
        turn = _rotation_deg(first[3:], second[3:])
        if distance >= MOVED_BELOW_MM or turn >= MOVED_BELOW_DEG:
            moved.append({"body": name, "moved_mm": round(distance, 2), "turned_deg": round(turn, 2)})
    moved.sort(key=lambda row: -row["moved_mm"])
    return {"bodies": len(set(before) & set(after)), "moved": moved}


class _RoboTwin:
    backend = "robotwin"

    def __init__(self, session: Any) -> None:
        self.session = session
        self.environment = None

    def started(self) -> None:
        self.environment = self.session._require_environment()

    def replay(self, entry: Mapping[str, Any], arrays: Mapping[str, Any], step: Mapping[str, Any]) -> str:
        return worker._replay_robotwin_entry(
            self.environment, entry, arrays, sampling=EVERY_STEP, capture=lambda *_: None,
            hold_steps=step.get("physics_steps") if step["kind"] == "hold" else None,
        )

    def pose(self, arm: str) -> list[float]:
        wxyz = [float(value) for value in self.session._public_pose_wxyz(self.environment, arm)]
        return wxyz[:3] + wxyz[4:7] + wxyz[3:4]

    def scene(self) -> dict[str, list[float]]:
        scene = self.environment.scene
        poses = {str(actor.get_name()): actor.get_pose() for actor in scene.get_all_actors()}
        # A pot or a drawer is an articulation, no actor: its root pose says where it went.
        poses.update({f"articulation:{body.get_name()}": body.get_root_pose() for body in scene.get_all_articulations()})
        return {name: [*(float(value) for value in pose.p), *(float(value) for value in pose.q)]  # wxyz
                for name, pose in poses.items()}


class _Libero:
    backend = "libero_pro"

    def __init__(self, session: Any) -> None:
        self.session = session

    def started(self) -> None:
        pass

    def replay(self, entry: Mapping[str, Any], arrays: Mapping[str, Any], _step: Mapping[str, Any]) -> str:
        return worker._replay_libero_entry(self.session, entry, arrays, capture=lambda *_: None)

    def pose(self, _arm: str) -> list[float]:
        position, orientation = self.session._grip_site_pose()
        return [float(value) for value in [*position, *orientation]]

    def scene(self) -> dict[str, list[float]]:
        simulation = self.session._worker_attr("sim")
        model, data = simulation.model, simulation.data
        names = [model.joint_id2name(index) for index in range(model.njnt) if int(model.jnt_type[index]) == 0]
        return {str(name): [float(value) for value in data.get_joint_qpos(name)] for name in names}


ADAPTERS = {"robotwin": _RoboTwin, "libero_pro": _Libero}


def _scene(adapter: Any) -> tuple[dict[str, list[float]] | None, str | None]:
    try:
        return adapter.scene(), None
    except Exception as exc:  # noqa: BLE001 - a report without scene poses still says what the planner did
        return None, f"{type(exc).__name__}: {exc}"


def _measure(adapter: Any, step: Mapping[str, Any]) -> dict[str, Any]:
    return {arm: _pose_comparison(adapter.pose(arm), pose) for arm, pose in (step.get("expect") or {}).items()}


def replay_prefix(adapter: Any, steps: Sequence[Mapping[str, Any]], manifest_path: Path,
                  manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Run ``steps`` as recorded; each row says whether the end pose equals the receipt's bit for bit."""

    entries = _program_entries(manifest)
    rows, ran_steps, current_round = [], 0, 0
    for step in steps:
        entry = _program_entry(entries, step)
        round_index = int(entry.get("round_index", 0))
        if round_index != current_round:
            adapter.session.restart_round(round_index)
            adapter.started()
            current_round = round_index
        ran = adapter.replay(entry, worker._artifact_arrays(manifest_path, entry), step)
        expected = {"skip": "skipped"}.get(step["kind"], step["kind"])
        if ran != expected:
            raise worker.ReplayWorkerError(
                f"action {step['action_sequence']} is {step['kind']} but its record replays as {ran}")
        ran_steps += int(step.get("physics_steps") or 0)
        rows.append({"action_sequence": int(step["action_sequence"]), "kind": step["kind"], "arms": _measure(adapter, step)})
    return {"steps": rows, "physics_steps": ran_steps}


def _execute_current(session: Any, job: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """The incident's target through this checkout's ``session.execute``; a raised refusal is its outcome."""

    observation, _paths = session.observe(output / "policy-observations")
    action = {"schema": MOTION_ACTION_V3_SCHEMA, **job["action"],
              "observation_id": str(observation.get("observation_id") or "incident-replay"),
              "bound_observation_sha256": str(observation.get("sha256") or "incident-replay")}
    started = time.monotonic()
    try:
        record = session.execute(action)
    except Exception as exc:  # noqa: BLE001 - a refusal raised by admission is what the report compares
        return {"refused": {"type": type(exc).__name__, "message": str(exc)},
                "elapsed_s": round(time.monotonic() - started, 3)}
    return {"receipt": record.get("receipt", record), "elapsed_s": round(time.monotonic() - started, 3)}


def run(job: Mapping[str, Any]) -> dict[str, Any]:
    """One branch of one incident; writes ``incident-result.json`` under the job's output root."""

    from ora.environment.controller_replay import load_controller_replay_manifest
    from .worker import _build_session

    run_root = Path(str(job["run_root"])).resolve()
    output = Path(str(job["output_root"])).resolve()
    if output == run_root or output.is_relative_to(run_root):
        raise worker.ReplayWorkerError("incident replay output must remain outside the run")
    branch = job.get("branch")
    if branch not in BRANCHES:
        raise worker.ReplayWorkerError(f"incident replay branch must be one of {BRANCHES}")
    manifest_path = run_root / "native/controller-replay/manifest.json"
    manifest = load_controller_replay_manifest(manifest_path)
    if manifest["backend"] not in ADAPTERS:
        raise worker.ReplayWorkerError("unsupported controller replay backend")
    output.mkdir(parents=True, exist_ok=True)
    session = _build_session(job["worker_config"])
    adapter = ADAPTERS[manifest["backend"]](session)
    result: dict[str, Any] = {"schema": RESULT_SCHEMA, "backend": manifest["backend"], "branch": branch,
                              "source_run_id": run_root.name, "action_sequence": int(job["action_sequence"]),
                              "error": None}
    try:
        session.start()
        adapter.started()
        result["prefix"] = replay_prefix(adapter, job["prefix"], manifest_path, manifest)
        before, result["scene_error"] = _scene(adapter)
        if branch == "recorded":
            entry = _program_entry(_program_entries(manifest), job["incident"])
            arrays = worker._artifact_arrays(manifest_path, entry)
            result["ran"] = adapter.replay(entry, arrays, job["incident"])
            result["arms"] = _measure(adapter, job["incident"])
        else:
            result["synchronised"] = synchronise(
                session, backend=manifest["backend"], receipts=job["receipts"], previous=job["previous_receipt"],
                steps_ran=result["prefix"]["physics_steps"])
            result.update(_execute_current(session, job, output))
        after, after_error = _scene(adapter)
        result["scene_error"] = result["scene_error"] or after_error
        result["scene"] = scene_changes(before, after)
    except Exception as exc:  # noqa: BLE001 - recorded for the host report, then the session is closed
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        worker._atomic_json(output / "incident-result.json", result)
        session.close()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one incident branch of a sealed run.")
    parser.add_argument("--job", type=Path, required=True)
    arguments = parser.parse_args(argv)
    job = json.loads(arguments.job.read_text(encoding="utf-8"))
    if job.get("schema") != JOB_SCHEMA:
        raise SystemExit("not an incident replay job")
    result = run(job)
    return 1 if result.get("error") else 0


__all__ = ["BRANCHES", "JOB_SCHEMA", "RESULT_SCHEMA", "gripper_directives", "replay_prefix", "run", "scene_changes",
           "synchronise"]


if __name__ == "__main__":
    raise SystemExit(main())
