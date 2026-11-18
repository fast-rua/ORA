"""The run an ``ora-remote`` command left on this host, so the same command attaches to it again.

``ora-remote`` names each command it runs here with ``ORA_REMOTE_SESSION``, a
key derived from the host, checkout and command line. The first time, the
command launches as usual and the run it attaches is recorded under the key
(``.harness-runtime/ora-remote/<key>.json``, outside run evidence). The same
command again, while that run's Supervisor is alive on this host, attaches to
the run instead of launching another one; once the run has finished, it
launches a new one. ``ORA_REMOTE_FRESH=1`` (``ora-remote --new``) always
launches. Records are per host: a shared workspace's record from another
node is ignored, as its process ids mean nothing here.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any

from ora.orchestrator import paths

SESSION_ENV, FRESH_ENV = "ORA_REMOTE_SESSION", "ORA_REMOTE_FRESH"
SCHEMA = "ora.remote-session.v1"
_KEY = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
#: Commands whose run a remote session resumes (``attach`` and ``watch`` name their run).
RESUMABLE = ("play", "interactive")


def key() -> str | None:
    value = os.environ.get(SESSION_ENV)
    if not value:
        return None
    if not _KEY.fullmatch(value):
        raise ValueError(f"{SESSION_ENV} must be 1-64 lowercase letters, digits or dashes")
    return value


def _path(name: str) -> Path:
    return paths.STATE_ROOT / "ora-remote" / f"{name}.json"


def record(root: Path) -> None:
    """Remember that this remote session shows `root` (a no-op outside ``ora-remote``)."""

    name = key()
    if name is None:
        return
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps({
        "schema": SCHEMA, "run_id": root.name, "run_root": str(root), "host": socket.gethostname(),
        "recorded_unix_s": round(time.time(), 3),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resumable_run() -> Path | None:
    """The run this remote session left running on this host, if any."""

    name = key()
    if name is None or os.environ.get(FRESH_ENV) == "1":
        return None
    try:
        value: Any = json.loads(_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("host") != socket.gethostname():
        return None
    root = Path(str(value.get("run_root") or ""))
    if not root.is_absolute() or not root.is_dir():
        return None
    from .interactive import supervisor_running

    return root if supervisor_running(root) else None


def resume_arguments(values: list[str]) -> list[str] | None:
    """``ora attach RUN`` with the command's world options, when this remote session's run is still going."""

    if not values or values[0] not in RESUMABLE:
        return None
    root = resumable_run()
    if root is None:
        return None
    world: list[str] = []
    for index, value in enumerate(values):
        name = value.split("=", 1)[0]
        if name in {"--world-view", "--world-port", "--world-browser", "--world-public-url"}:
            world.append(value)
            if "=" not in value and index + 1 < len(values):
                world.append(values[index + 1])
    print(f"ora: this remote session's run {root.name} is still going; attaching to it "
          "(ora-remote --new starts another)", flush=True)
    return ["attach", str(root), *world]


__all__ = ["FRESH_ENV", "SESSION_ENV", "key", "record", "resumable_run", "resume_arguments"]
