"""Is the cell free? Other readers or traffic on the buses, programs that drive arms, busy cameras.

Robot-agnostic over socketcan buses and RealSense cameras. Everything here reads only:
``/proc``, ``ip`` and a listening socket; nothing is ever sent. ``check_occupancy`` refuses
the cell before a robot backend sends its first frame. ``SOCKETCAN`` is the default
``Bus`` of ``adapter.py``: a robot whose arms are not on socketcan (a simulated cell) gives
its adapter its own.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .errors import CellError

CAN_PROC = Path("/proc/net/can")
RECEIVE_LISTS = ("rcvlist_all", "rcvlist_fil", "rcvlist_sff", "rcvlist_eff", "rcvlist_inv", "rcvlist_err")
LISTEN_S = 1.0
# Command lines of programs that drive robot arms; a robot adds its own (CellAdapter).
MOTION_PROCESS_MARKERS = ("--allow-motion", "teleop")


def can_receivers(channel: str, root: Path = CAN_PROC) -> int:
    """Sockets other processes hold on ``channel``, from the kernel's receive lists."""

    # An entry is a row whose first column is the device; an empty list prints
    # "(can_left: no entry)" instead.
    count = 0
    for name in RECEIVE_LISTS:
        path = root / name
        if path.is_file():
            count += sum(1 for line in path.read_text().splitlines() if line.split()[:1] == [channel])
    return count


def can_traffic(channel: str, seconds: float = LISTEN_S) -> int:
    """Frames heard on ``channel`` in ``seconds`` of listening; a quiet bus hears none."""

    import can

    bus = can.interface.Bus(channel=channel, interface="socketcan")
    try:
        heard, end = 0, time.monotonic() + seconds
        while time.monotonic() < end:
            if bus.recv(timeout=0.05) is not None:
                heard += 1
        return heard
    finally:
        bus.shutdown()


class SocketCan:
    """The arms' buses as socketcan interfaces (``adapter.Bus``): the functions of this module."""

    def receivers(self, channel: str) -> int:
        return can_receivers(channel)

    def traffic(self, channel: str, seconds: float = LISTEN_S) -> int:
        return can_traffic(channel, seconds)

    def interface(self, channel: str) -> dict[str, Any]:
        return can_interface(channel)


SOCKETCAN = SocketCan()


def motion_processes(markers: Sequence[str] = MOTION_PROCESS_MARKERS, ignore: Iterable[str] = (),
                     proc: Path = Path("/proc")) -> list[dict[str, Any]]:
    """Running processes whose command line says they drive arms, except this worker's own."""

    own = {os.getpid(), os.getppid()}
    ignored = tuple(ignore)
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            owner = entry.stat().st_uid
        except OSError:
            continue
        if not command or any(item in command for item in ignored):
            continue
        hits = [marker for marker in markers if marker in command]
        if hits:
            found.append({"pid": int(entry.name), "uid": owner, "markers": hits, "command": command[:240]})
    return found


def realsense_busy(serials: Iterable[str]) -> dict[str, str]:
    """Serials that cannot be opened now (missing, or streamed by another process)."""

    import pyrealsense2 as rs

    connected = {device.get_info(rs.camera_info.serial_number) for device in rs.context().query_devices()}
    busy = {}
    for serial in serials:
        if serial not in connected:
            busy[serial] = "not connected"
            continue
        pipeline, config = rs.pipeline(), rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth)
        try:
            pipeline.start(config)
            pipeline.stop()
        except RuntimeError as exc:
            busy[serial] = str(exc)
    return busy


def check_occupancy(*, mode: str, channels: Iterable[str], serials: Iterable[str],
                    ignore_processes: Iterable[str] = (),
                    markers: Sequence[str] = MOTION_PROCESS_MARKERS,
                    bus: Any = SOCKETCAN, busy: Callable[[Iterable[str]], dict[str, str]] | None = None) -> dict[str, Any]:
    """Refuse the cell unless nothing else holds it. Returns what was checked.

    ``bus`` is the arms' ``Bus`` and ``busy`` the cameras' check (``realsense_busy`` by default)."""

    report: dict[str, Any] = {"mode": mode, "can": {}, "processes": [], "realsense_busy": {}}
    problems = []
    for channel in channels:
        receivers = bus.receivers(channel)
        heard = bus.traffic(channel, LISTEN_S) if receivers == 0 else None
        report["can"][channel] = {"receivers": receivers, "frames_heard": heard}
        if receivers:
            problems.append(f"{channel} already has {receivers} reader(s): another controller holds it")
        elif heard:
            problems.append(f"{channel} carried {heard} frames in {LISTEN_S:.0f} s: another controller drives it")
    report["processes"] = motion_processes(markers, ignore=ignore_processes)
    if report["processes"] and mode == "on":
        listed = "; ".join(f"pid {p['pid']} ({', '.join(p['markers'])})" for p in report["processes"])
        problems.append(f"processes that can move the arms are running: {listed}. Stop them, or name them in "
                        "ignore_processes once their owner confirms they will not open the CAN buses")
    report["realsense_busy"] = (busy or realsense_busy)(serials)
    for serial, why in report["realsense_busy"].items():
        problems.append(f"RealSense {serial}: {why}")
    report["problems"] = problems
    if problems:
        raise CellError("the cell is not free:\n  " + "\n  ".join(problems))
    return report



def can_interface(channel: str) -> dict[str, Any]:
    """Controller state and packet counters, from ``ip``; opens no CAN socket."""

    try:
        out = subprocess.run(["ip", "-details", "-statistics", "-json", "link", "show", channel],
                             capture_output=True, text=True, timeout=2, check=True).stdout
        link = json.loads(out)[0]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return {"present": False, "error": str(exc)}
    info = link.get("linkinfo", {}).get("info_data", {})
    stats = link.get("stats64", {})
    return {"present": True, "up": "UP" in link.get("flags", []), "state": info.get("state"),
            "bitrate": (info.get("bittiming") or {}).get("bitrate"),
            "rx_packets": stats.get("rx", {}).get("packets"), "tx_packets": stats.get("tx", {}).get("packets"),
            "tx_errors": stats.get("tx", {}).get("errors"), "rx_errors": stats.get("rx", {}).get("errors")}
