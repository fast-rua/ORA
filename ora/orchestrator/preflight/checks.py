"""The single preflight check record shape, shared by every backend's preflight."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .resources import ExecutionResourceError, require_registered_host


def check(label: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"check": label, "passed": bool(passed), "detail": detail}


def motion_timing_check(timing: Mapping[str, Any], *,
                        environment_overrides: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Start the timing worker a runtime names; the passing detail binds its interpreter (path, sha256) to the run.

    ``timing``: ``{"implementation": "toppra_ruckig", "python_env": variable}``."""
    from ora.environment.controllers.motion_retime import TimingClient
    from ora.environment.controllers.motion_timing import PREFLIGHT_CHECK

    try:
        python_raw = os.environ.get(str(timing["python_env"]))
        if not python_raw:
            raise RuntimeError(f"{timing['python_env']} is missing")
        client = TimingClient(Path(os.path.abspath(os.path.expanduser(python_raw))),
                              environment_overrides=environment_overrides)
        try:
            return check(PREFLIGHT_CHECK, True, client.start())
        finally:
            client.close()
    except Exception as exc:
        return check(PREFLIGHT_CHECK, False, str(exc))


def execution_host_check(runtime: Any, marker: str | None) -> dict[str, Any]:
    """A live run names a registered host; an undeclared one is refused, not assumed."""
    required_host = runtime.get("required_host")
    detail = {"expected": required_host or "runtime.required_host is not declared", "observed": marker}
    passed = bool(required_host) and marker == required_host
    if passed:
        try:
            require_registered_host(required_host)
        except ExecutionResourceError as exc:
            passed, detail["profile"] = False, str(exc)
    return check("execution_host", passed, detail)
