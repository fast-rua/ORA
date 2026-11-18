"""Occupancy reads only: bus readers counted from the kernel's CAN receive lists, and running programs that
drive arms found by their command lines, this process and the owner's vouched-for programs excepted."""

from __future__ import annotations

import os
from pathlib import Path

from ora.cell.occupancy import MOTION_PROCESS_MARKERS, can_receivers, motion_processes

RECEIVE_ALL = """\
 device   can_id   can_mask  function  userdata   matches  ident
  can_left  000      00000000  ffffffff  ffff8881  0        raw
  can_left  000      00000000  ffffffff  ffff8882  12       raw
 can_right  000      00000000  ffffffff  ffff8883  0        raw
"""
RECEIVE_FILTERED = """\
 device   can_id   can_mask  function  userdata   matches  ident
  can_left  0x07     000007ff  ffffffff  ffff8884  3        raw
"""


def test_receivers_are_rows_of_the_channel_in_every_receive_list(tmp_path: Path) -> None:
    (tmp_path / "rcvlist_all").write_text(RECEIVE_ALL)
    (tmp_path / "rcvlist_fil").write_text(RECEIVE_FILTERED)
    (tmp_path / "rcvlist_sff").write_text("\n  (can_left: no entry)\n  (can_right: no entry)\n")
    (tmp_path / "stats").write_text("can_left can_left can_left\n")       # not a receive list

    assert can_receivers("can_left", root=tmp_path) == 3
    assert can_receivers("can_right", root=tmp_path) == 1
    assert can_receivers("can", root=tmp_path) == 0                      # whole names only
    assert can_receivers("can_left", root=tmp_path / "absent") == 0


# Process ids clear of this process and its parent, which the scan always skips.
BASE = max(os.getpid(), os.getppid()) + 1000


def _process(proc: Path, pid: int, command: list[str]) -> None:
    entry = proc / str(pid)
    entry.mkdir()
    (entry / "cmdline").write_bytes(b"\0".join(part.encode() for part in command) + (b"\0" if command else b""))


def test_programs_that_drive_arms_are_found_by_their_markers(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _process(proc, BASE + 1, ["python", "-m", "i2rt.teleop", "--channel", "can_left"])
    _process(proc, BASE + 2, ["python", "run.py", "--allow-motion", "--teleop"])
    _process(proc, BASE + 3, ["python", "my_teleop.py"])                      # vouched for by the owner
    _process(proc, BASE + 4, ["bash"])
    _process(proc, BASE + 5, [])                                              # a kernel thread: no command line
    _process(proc, os.getpid(), ["python", "--allow-motion"])            # this process
    (proc / "self").mkdir()
    (proc / "self" / "cmdline").write_bytes(b"teleop\0")
    _process(proc, BASE + 6, ["python", "-c", "x" * 400 + " teleop"])

    found = motion_processes(MOTION_PROCESS_MARKERS, ignore=["my_teleop.py"], proc=proc)

    by_pid = {item["pid"]: item for item in found}
    assert sorted(by_pid) == [BASE + 1, BASE + 2, BASE + 6]
    assert by_pid[BASE + 1]["markers"] == ["teleop"]
    assert by_pid[BASE + 1]["command"] == "python -m i2rt.teleop --channel can_left"
    assert by_pid[BASE + 2]["markers"] == ["--allow-motion", "teleop"]
    assert by_pid[BASE + 1]["uid"] == os.getuid()
    assert len(by_pid[BASE + 6]["command"]) == 240


def test_a_robot_brings_its_own_markers(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _process(proc, BASE + 7, ["python", "fake_driver.py"])

    assert motion_processes(MOTION_PROCESS_MARKERS, proc=proc) == []
    assert [item["pid"] for item in motion_processes(("fake_driver",), proc=proc)] == [BASE + 7]
