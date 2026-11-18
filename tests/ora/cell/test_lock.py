"""The cell lock: one holder at a time, across processes; ``holder()`` reads who without taking it; ``relabel``
changes what the holder says it does while keeping the lock."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ora.cell.errors import CellError
from ora.cell.lock import AT_REST, HOMES_ON_SIGNAL, CellLock, holder

ROOT = Path(__file__).resolve().parents[3]
PROBE = """
import sys
from pathlib import Path
from ora.cell.errors import CellError
from ora.cell.lock import CellLock, holder

path = Path(sys.argv[1])
print(holder(path))
try:
    CellLock("second process", path).acquire()
except CellError as exc:
    print(exc)
else:
    print("acquired")
"""


def test_nobody_holds_a_cell_without_a_lock_file_or_after_release(cell_lock_path: Path) -> None:
    assert holder() is None
    with CellLock("cell tool (read)"):
        assert holder() == f"cell tool (read) (pid {os.getpid()})"
    assert holder() is None
    assert cell_lock_path.read_text() == ""


def test_a_second_lock_is_refused_and_names_the_holder() -> None:
    first = CellLock("cell board (live view)").acquire()
    try:
        with pytest.raises(CellError, match=rf"in use by cell board \(live view\) \(pid {os.getpid()}\)"):
            CellLock("cell operator").acquire()
    finally:
        first.release()
    with CellLock("cell operator"):
        pass


def test_another_process_sees_the_holder_and_cannot_take_the_lock(cell_lock_path: Path) -> None:
    with CellLock("ora worker"):
        probe = subprocess.run([sys.executable, "-c", PROBE, str(cell_lock_path)], cwd=ROOT,
                               capture_output=True, text=True, timeout=60, check=True)
    seen, refusal = probe.stdout.strip().splitlines()
    assert seen == f"ora worker (pid {os.getpid()})"
    assert refusal.endswith(f"in use by ora worker (pid {os.getpid()})")

    probe = subprocess.run([sys.executable, "-c", PROBE, str(cell_lock_path)], cwd=ROOT,
                           capture_output=True, text=True, timeout=60, check=True)
    assert probe.stdout.strip().splitlines() == ["None", "acquired"]


def test_relabel_keeps_the_lock_and_changes_what_others_read() -> None:
    lock = CellLock(f"ora worker {HOMES_ON_SIGNAL}").acquire()
    try:
        lock.relabel(f"ora worker {HOMES_ON_SIGNAL} {AT_REST}")
        assert AT_REST in holder() and HOMES_ON_SIGNAL in holder()
        assert lock.purpose.endswith(AT_REST)
        with pytest.raises(CellError):
            CellLock("cell operator").acquire()
    finally:
        lock.release()
    lock.relabel("after release")          # a released lock has nothing to relabel
    assert holder() is None
