"""What is wrong with the held arms right now, for every front end to show at once.

The program holding the arms (a play worker) watches them even between motions and
publishes its current alerts here: a JSON file beside the cell lock
(``$ORA_CELL_ALERTS``, else ``ora-cell-alerts.json`` next to ``$ORA_CELL_LOCK``). The cell
board shows them as a banner, Pi's play terminal as notices, and the worker adds them to
what the agent reads. A file whose publisher has ended is stale and reads as no alerts,
except the publisher's own last word that it stopped controlling the arms.

An alert: ``{"code", "level": "warning" | "critical", "text"`` (plain Chinese for the
operator) ``, "arm", "motor", "value", "since_unix_s"}``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import lock

SCHEMA = "ora.cell-alerts.v1"
LEVELS = ("warning", "critical")


def path() -> Path:
    named = os.environ.get("ORA_CELL_ALERTS")
    return Path(named).expanduser() if named else lock.LOCK_PATH.parent / "ora-cell-alerts.json"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def publish(alerts: Iterable[Mapping[str, Any]], *, source: str) -> None:
    """Replace the published alerts with ``alerts`` (an empty list clears them)."""

    body = {"schema": SCHEMA, "source": source, "pid": os.getpid(), "updated_unix_s": time.time(),
            "alerts": [dict(alert) for alert in alerts]}
    target = path()
    temporary = target.with_name(f".{target.name}.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(body, ensure_ascii=False, default=float))
        os.chmod(temporary, 0o666)
        os.replace(temporary, target)
    except OSError:
        pass                                   # the alerts are also in the worker's own reports


def current() -> dict[str, Any]:
    """The published alerts, ``{"source", "pid", "alerts", "stale"}``; never raises."""

    try:
        body = json.loads(path().read_text())
    except (OSError, ValueError):
        return {"source": None, "pid": None, "alerts": [], "stale": False}
    pid = body.get("pid")
    stale = not (isinstance(pid, int) and _alive(pid))
    alerts = [a for a in body.get("alerts") or [] if isinstance(a, dict) and a.get("level") in LEVELS]
    if stale:
        alerts = [a for a in alerts if a.get("code") == "arms_not_controlled"]
    return {"source": body.get("source"), "pid": pid, "alerts": alerts, "stale": stale,
            "updated_unix_s": body.get("updated_unix_s")}


def clear(*, source: str) -> None:
    publish([], source=source)
