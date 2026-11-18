"""One holder of the real cell at a time: the worker, an ``ora cell`` command or the cell board.

A DISABLE frame sent to an arm another program is holding drops that arm. Everything in
the kit and in a robot's worker that touches the buses or the cameras takes this lock
first; the worker keeps it for its whole episode. Programs outside ORA (a vendor's
scripts, another controller) do not know the lock, which is why the occupancy check still
listens to the buses.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

from .errors import CellError

DEFAULT_LOCK_PATH = Path("/tmp/ora-cell.lock")
LOCK_PATH = Path(os.environ.get("ORA_CELL_LOCK") or DEFAULT_LOCK_PATH)


class CellLock:
    def __init__(self, purpose: str, path: Path = LOCK_PATH) -> None:
        self.purpose, self.path, self.handle = purpose, path, None

    def acquire(self) -> "CellLock":
        handle = open(self.path, "a+")
        try:
            os.chmod(self.path, 0o666)
        except OSError:
            pass
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            holder = handle.read().strip() or "another program"
            handle.close()
            raise CellError(f"the cell is in use by {holder}") from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"{self.purpose} (pid {os.getpid()})")
        handle.flush()
        self.handle = handle
        return self

    def relabel(self, purpose: str) -> None:
        """Say what the holder does now (e.g. that its arms are back at rest), keeping the lock."""

        if self.handle is not None:
            self.purpose = purpose
            self.handle.seek(0)
            self.handle.truncate()
            self.handle.write(f"{purpose} (pid {os.getpid()})")
            self.handle.flush()

    def release(self) -> None:
        if self.handle is not None:
            self.handle.seek(0)
            self.handle.truncate()
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

    def __enter__(self) -> "CellLock":
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()


def holder(path: Path = LOCK_PATH) -> str | None:
    """Who holds the cell now, or None. Reads only; never takes the lock."""

    if not path.exists():
        return None
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            return handle.read().strip() or "unknown"
        fcntl.flock(handle, fcntl.LOCK_UN)
        return None


# A holder that returns its arms to rest on SIGUSR1 says so in its lock label;
# ``operator.actions.stop`` signals only such holders. One that keeps holding its arms at
# rest afterwards (a play session's worker) adds AT_REST to its label once there, and HOMING
# while it drives there (``stop`` then waits longer instead of taking the arms over).
HOMES_ON_SIGNAL = "[homes on SIGUSR1]"
AT_REST = "[at rest, holding]"
HOMING = "[homing]"
# A holder whose arms are no longer under its control (a motor protection stopped a control
# loop) says so; the alert itself is in ``ora.cell.alerts``.
NOT_CONTROLLED = "[arms not controlled]"
