"""What the Supervisor takes from a LIBERO-Pro run's preflight: the qualified controller binding.

With ``runtime.motion_planner.timing``, the binding also names the timing interpreter
preflight started (``motion_timing``: its path and sha256), which the isolated worker
cannot read from its allowlisted environment and times its moves with.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ora.orchestrator.faults import SupervisorError


def supervisor_binding(run_root: Path, runtime_config: Mapping[str, Any]) -> dict[str, Any] | None:
    """The controller action contract preflight passed, which the worker binds."""

    path = run_root / "preflight.json"
    try:
        preflight = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupervisorError(
            "LIBERO run lacks its passing controller preflight binding"
        ) from exc
    checks = preflight.get("checks") if isinstance(preflight, Mapping) else None
    matches = [
        item for item in checks if isinstance(item, Mapping)
        and item.get("check") == "libero_controller_action_contract"
    ] if isinstance(checks, list) else []
    detail = matches[0].get("detail") if len(matches) == 1 else None
    if (
        len(matches) != 1
        or matches[0].get("passed") is not True
        or not isinstance(detail, Mapping)
        or detail.get("schema")
        != "ora.libero-controller-qualification.v1"
    ):
        raise SupervisorError(
            "LIBERO controller preflight binding is absent or invalid"
        )
    binding = dict(detail)
    planner = runtime_config.get("motion_planner")
    if isinstance(planner, Mapping) and planner.get("timing") is not None:
        timing = [item for item in checks if isinstance(item, Mapping) and item.get("check") == "motion_timing_worker"]
        bound = timing[0].get("detail") if len(timing) == 1 and timing[0].get("passed") is True else None
        if not isinstance(bound, Mapping) or not isinstance(bound.get("python"), str) \
                or not isinstance(bound.get("python_sha256"), str):
            raise SupervisorError("LIBERO motion timing preflight binding is absent or invalid")
        binding["motion_timing"] = {"python": bound["python"], "python_sha256": bound["python_sha256"]}
    return binding


__all__ = ["supervisor_binding"]
