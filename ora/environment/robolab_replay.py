"""The host side of a RoboLab replay (``replay_program.py``): the worker that runs it and the trace's clock check."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def replay_in_environment_worker(plan: Mapping[str, Any], job: Path, worker_output: Path, log: Path,
                                 timeout_s: float) -> int:
    """Run the job on the environment worker a run uses, which owns Kit and closes it as a run does.

    The worker gets the plan's runtime without its planner (nothing is planned again) on the caller's card (a router
    lane), plus ``replay_program_job``: its ``start`` runs the run's reset and then the job, so ``worker.py`` and
    ``rpc.py``, members of RoboDojo's pinned kernel, need no replay method.
    """
    from .factory import build_robolab_worker_session

    runtime = {key: value for key, value in plan["runtime"].items() if key not in {"motion_planner", "resource_binding"}}
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit():
        runtime["renderer_gpu"] = int(os.environ["CUDA_VISIBLE_DEVICES"])
    runtime.update(request_timeout_s=timeout_s, replay_program_job=str(job))
    session = build_robolab_worker_session(
        runtime=runtime, output_root=worker_output / "environment", task=str(plan["task"]), seed=int(plan["seed"]),
        action_codec=(plan.get("policy") or {}).get("action_codec"))
    try:
        session.start()
        return 0
    except Exception as exc:  # noqa: BLE001 - a replay that cannot run is a failed check, with the reason beside it
        log.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return 1
    finally:
        session.close()


def receipt_clock_failures(receipts: Mapping[int, Mapping[str, Any]], manifest: Mapping[str, Any]) -> list[str]:
    """RoboLab has no execution capture: its page's clock is the receipts' native actions, accumulated in action order
    within a round. Every action of the trace must span exactly those steps (``backends/robolab/scene_trace.py``)."""
    per_native = manifest["clock"]["stride"]
    cursor: dict[int, int] = {}
    differing = []
    for span in manifest["spans"]:
        start = cursor.get(span["round"], 0)
        steps = (receipts.get(span["action_sequence"]) or {}).get("steps_used")
        end = start + int(steps or 0) * per_native
        if (span["first_step"], span["last_step"]) != (start, end):
            differing.append(span["action_sequence"])
        cursor[span["round"]] = end
    return [f"the scene trace's steps differ from the receipts' at actions {differing}"] if differing else []


__all__ = ["receipt_clock_failures", "replay_in_environment_worker"]
