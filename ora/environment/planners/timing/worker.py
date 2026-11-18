"""The timing worker: ``retime`` requests in, answers out, one JSON object per line.

``TimingClient`` (``controllers/motion_planner.py``) starts it as
``python -m ora.environment.planners.timing.worker`` under ``ORA_TIMING_PYTHON``,
a small CPU interpreter with the timing libraries (``requirements.py``), with
this checkout first on ``PYTHONPATH``. It answers ``ready``, ``retime`` and
``close``; it plans nothing, so ``plan`` and ``ik`` are refused commands. The
answers are ``retime.retime``'s; the worker only frames them.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from typing import Any, Callable, Iterable

from ora.environment.controllers.motion_retime import TIMING_IMPLEMENTATION
from ora.environment.controllers.motion_worker import PROTOCOL


def ready() -> dict[str, Any]:
    from . import retime

    return {"implementation": TIMING_IMPLEMENTATION, "versions": retime.versions(), "settings": retime.settings(),
            "python": sys.version.split()[0]}


def serve(lines: Iterable[str], write: Callable[[str], None]) -> int:
    """Answer one JSON request per line until ``close`` or the end of input."""
    from . import retime

    for line in lines:
        request_id: Any = None
        command: Any = None
        try:
            envelope = json.loads(line)
            request_id, command = envelope.get("request_id"), envelope.get("command")
            if envelope.get("protocol") != PROTOCOL:
                raise ValueError("protocol drift")
            reply: dict[str, Any] = {"protocol": PROTOCOL, "request_id": request_id, "command": command}
            if command == "ready":
                reply["ready"] = ready()
            elif command == "retime":
                with contextlib.redirect_stdout(sys.stderr):
                    reply["answer"] = retime.retime(envelope.get("request"))
            elif command == "close":
                write(json.dumps({**reply, "closed": True}))
                return 0
            else:
                raise ValueError("unsupported command")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply = {"protocol": PROTOCOL, "request_id": request_id, "command": command,
                     "error": {"type": type(exc).__name__, "message": str(exc) or type(exc).__name__}}
        write(json.dumps(reply, allow_nan=False))
    return 0


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    with contextlib.redirect_stdout(sys.stderr):
        from . import retime  # noqa: F401  (import the libraries before answering ready)
    return serve(sys.stdin, lambda text: print(text, flush=True))


if __name__ == "__main__":
    sys.exit(main())
