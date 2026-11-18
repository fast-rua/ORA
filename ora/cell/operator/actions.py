"""The operator's verbs on a real cell, implemented once for every front end.

The cell board's buttons, ``ora cell <verb>`` and the play terminal's slash commands all
call these. They act **now**, also while another program moves the arms:

- ``home``: the arms go back to the recorded rest pose and only then are released. A
  program that holds the arms and says it homes on SIGUSR1 (an ORA worker, a calibration
  motion) is asked to do it itself; one that does not answer in ``STOP_WAIT_S`` is ended
  and the arms are taken over at once, inside the motors' communication timeout. A blocked
  or refused path holds where the arms are; nothing drops.
- ``hold`` / ``float``: take the arms and hold the measured pose, or float them for hand
  guiding (gravity compensation); switch between the two without letting go.
- ``release``: the arms go limp. ``estop``: every motor disabled whoever holds the cell.
  Both make arms fall; front ends confirm them first.
- ``diagnose``, ``cameras``, ``can``: read-only or device-level; never move an arm.
- ``stop`` / ``kill``: end the program holding the cell (gracefully / by force).

What a robot does to read, hold and release its arms is its ``CellAdapter``'s.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from typing import Any, Callable

import numpy as np

from .. import alerts as cell_alerts
from .. import homing
from ..errors import CellError
from ..geometry import REST_DEVIATION_RAD
from ..lock import AT_REST, HOMES_ON_SIGNAL, HOMING, CellLock, holder
from ..occupancy import motion_processes
from .context import Cell

HOME_SPEED_RAD_S = 0.25
ALERT_PERIOD_S = 0.5
HOME_TOLERANCE_RAD = 0.02
STOP_WAIT_S = 20.0                 # a camera capture (about 4 s) plus the drive back must fit
HOMING_WAIT_S = 60.0               # a holder that says it is driving home is not taken over before this

Confirm = Callable[[str], bool]


def ask_terminal(question: str) -> bool:
    try:
        with open("/dev/tty", "r+", buffering=1) as tty:
            tty.write(question.rstrip() + "\n[yes/no] > ")
            return tty.readline().strip().lower() in {"y", "yes"}
    except OSError:
        return False


def holder_pid() -> tuple[str | None, int | None]:
    who = holder()
    if who is None:
        return None, None
    match = re.search(r"pid (\d+)", who)
    return who, int(match.group(1)) if match else None


# ---------------------------------------------------------------- the held arms
class Holder:
    """Arms this process holds for as long as the person needs: floating (guided by hand) or
    holding a pose, driven home on request. Only ``release`` makes them limp. While it holds
    them a thread publishes their alerts (``ora.cell.alerts``) every ALERT_PERIOD_S."""

    def __init__(self, cell: Cell) -> None:
        self.cell = cell
        self.lock: CellLock | None = None
        self.held: Any = None
        self.mode: str | None = None
        self.since: float | None = None
        self.note: str | None = None
        self._watching: threading.Event | None = None

    def _watch(self, held: Any, stop: threading.Event) -> None:
        source, published = f"cell operator holder (pid {os.getpid()})", None
        while not stop.is_set():
            try:
                found = list(held.alerts()) if hasattr(held, "alerts") else []
            except Exception as exc:  # noqa: BLE001 - the watcher must keep reporting
                found = [{"code": "watch_failed", "level": "warning", "text": f"手臂监测出错：{exc}",
                          "since_unix_s": time.time()}]
            key = [(a.get("code"), a.get("arm"), a.get("motor"), a.get("level")) for a in found]
            if key != published:
                published = key
                cell_alerts.publish(found, source=source)
            stop.wait(ALERT_PERIOD_S)

    @property
    def active(self) -> bool:
        return self.held is not None

    def start(self, mode: str) -> str:
        if self.active:
            return self.switch(mode)
        lock = CellLock("cell operator (holding the arms)").acquire()
        try:
            self.held = self.cell.adapter.take_over(self.cell.adapter.arms, mode=mode,
                                                    gripper_limits=self.cell.gripper_limits())
        except BaseException:
            lock.release()
            raise
        self.lock, self.mode, self.since, self.note = lock, mode, time.time(), None
        self._watching = threading.Event()
        threading.Thread(target=self._watch, args=(self.held, self._watching), name="cell-holder-watch",
                         daemon=True).start()
        return "arms floating (gravity compensation)" if mode == "float" else "arms holding their pose"

    def switch(self, mode: str) -> str:
        self.held.set_mode(mode)
        self.mode, self.since = mode, time.time()
        return f"switched to {mode}"

    def drive_home(self) -> str:
        """Joint-space path to rest at HOME_SPEED_RAD_S (``homing.drive``). Blocked or refused: hold where the arms are."""

        goal = self.cell.rest()
        start = self.held.joints()
        self.switch("hold")
        adapter = self.cell.adapter
        travel = max(float(np.max(np.abs(goal[a] - start[a]))) for a in start)
        settings = homing.HomeSettings(
            speed_rad_s=HOME_SPEED_RAD_S, period_s=1.0 / adapter.control_hz, tracking_stop_rad=adapter.tracking_stop_rad,
            tolerance_rad=HOME_TOLERANCE_RAD, limit_s=travel / HOME_SPEED_RAD_S + 10.0)
        outcome = homing.drive(goal, read=self.held.joints, command=self.held.command, alive=self.held.alive,
                               wait=lambda: time.sleep(settings.period_s), settings=settings,
                               kinematics=self.cell.kinematics(with_table=True))
        if outcome.state == "at_rest":
            self.note = "at rest"
            return "at rest, holding"
        if outcome.state == "path_blocked":
            self.note = "a straight path home is not clear: " + outcome.detail
            return self.note + ". Holding here; float the arms closer to rest by hand and try again"
        if outcome.state == "dead":
            self.note = outcome.detail
            return self.note
        measured = self.held.joints()
        self.held.command({arm: q for arm, q in measured.items() if self.held.alive().get(arm, True)})
        if outcome.state == "blocked":
            self.note = f"{outcome.detail} (blocked?)"
            return self.note + ": holding where it is"
        self.note = "did not reach rest in time: holding where it is"
        return self.note

    def release(self) -> str:
        if not self.active:
            return "nothing held"
        if self._watching is not None:
            self._watching.set()
            self._watching = None
        try:
            self.held.release()
        finally:
            self.held, self.mode, self.since = None, None, None
            self.lock.release()
            self.lock = None
        return "released: the arms are limp"


def at_rest(cell: Cell) -> bool:
    goal = cell.rest()
    with CellLock("cell operator (checking the pose)"):
        readings = cell.read_arms()
    return all(np.max(np.abs(readings[arm].q - goal[arm])) <= HOME_TOLERANCE_RAD for arm in goal)


# ---------------------------------------------------------------- verbs
def home(cell: Cell, held: Holder) -> str:
    """Back to rest from whatever state the cell is in, never letting an arm drop."""

    if held.active:
        outcome = held.drive_home()
    else:
        who = holder() or ""
        if who:
            answer = stop()
            if "went home" in answer:
                return answer
            _, pid = holder_pid()
            if pid is not None and pid != os.getpid():
                os.kill(pid, signal.SIGKILL)
                for _ in range(40):
                    if holder() is None:
                        break
                    time.sleep(0.05)
        elif at_rest(cell):
            return "already at rest: nothing moved"
        held.start("hold")
        outcome = held.drive_home()
    if held.note == "at rest":
        time.sleep(0.5)
        held.release()
        return "at rest and released"
    return outcome


def stop(timeout_s: float = STOP_WAIT_S) -> str:
    """Ask the program holding the cell to return its arms and release them (SIGUSR1)."""

    who, pid = holder_pid()
    if pid is None:
        return "no program holds the cell"
    if HOMES_ON_SIGNAL not in (who or ""):
        return f"{who} does not home on a signal: use its own controls, or 'kill'"
    os.kill(pid, signal.SIGUSR1)
    time.sleep(0.5)
    begun = time.monotonic()
    end = begun + timeout_s
    while time.monotonic() < end:
        now = holder()
        if now is None:
            return f"pid {pid} went home and released the cell"
        if AT_REST in now:
            return f"pid {pid} went home and holds the arms at rest"
        if HOMING in now:
            end = max(end, begun + max(timeout_s, HOMING_WAIT_S))
        time.sleep(0.5)
    return f"pid {pid} still holds the cell after {timeout_s:.0f} s"


def kill(confirm: Confirm = ask_terminal) -> str:
    who, pid = holder_pid()
    if pid is None:
        return "no program holds the cell"
    if not confirm(f"End {who}? If it holds enabled arms they FALL within the motor timeout. Support them first."):
        return "not killed"
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        time.sleep(0.25)
        if holder() is None:
            return f"pid {pid} ended"
    os.kill(pid, signal.SIGKILL)
    return f"pid {pid} killed"


def estop(cell: Cell) -> str:
    """DISABLE to every motor now, whoever holds the cell: the one stop that cannot be refused."""

    who, pid = holder_pid()
    for _ in range(2):                            # a frame lost to the holder's traffic gets a second chance
        cell.adapter.disable_all()
    if pid is not None and pid != os.getpid():
        try:
            # The holder's arms are gone: a calibration motion releases its lock on SIGUSR2;
            # any other holder is ended (the default action), which its disabled arms need anyway.
            os.kill(pid, signal.SIGUSR2)
        except (OSError, ValueError):
            pass
    return "every motor disabled" + (f" (the cell was held by {who})" if who else "")


def release_all(cell: Cell, confirm: Confirm = ask_terminal) -> str:
    if not confirm("Disable every motor now? Enabled arms go limp and fall."):
        return "nothing sent"
    with CellLock("cell operator (releasing)"):
        cell.adapter.disable_all()
    return "every motor disabled"


def diagnose(cell: Cell) -> dict[str, Any]:
    report: dict[str, Any] = {
        "lock_holder": holder(),
        "motion_processes": motion_processes(cell.adapter.foreign_controller_markers,
                                             ignore=cell.config.ignore_processes),
        "buses": {arm: {**cell.bus.interface(channel), "channel": channel, "receivers": cell.bus.receivers(channel)}
                  for arm, channel in cell.adapter.buses.items()}}
    advice = []
    who = report["lock_holder"]
    if who and "live view" in who:
        advice.append("the cell board's live view holds the cell and reads the arms passively; "
                      "stop live view before a play session")
    elif who:
        advice.append(f"the cell is held by {who}: 'home' returns its arms to rest; 'kill' ends it "
                      "(enabled arms then fall within the motor timeout)")
    for arm, bus in report["buses"].items():
        if not bus.get("present") or not bus.get("up"):
            advice.append(f"{bus['channel']} is missing or down: run 'can'")
        elif bus.get("state") not in (None, "ERROR-ACTIVE"):
            advice.append(f"{bus['channel']} is {bus.get('state')}: run 'can'")
    if who is None and not any(b.get("receivers") for b in report["buses"].values()):
        try:
            with CellLock("cell operator (diagnose)"):
                readings = cell.read_arms()
            rest = cell.calibration().get("rest_reference")
            report["arms"] = {}
            for arm, reading in readings.items():
                away = None if not rest else float(np.max(np.abs(reading.q - np.asarray(rest[arm]))))
                report["arms"][arm] = {"q": np.round(reading.q, 3).tolist(), "faults": reading.faults,
                                       "max_temperature_c": reading.max_temperature_c, "from_rest_rad": away}
                if reading.faults:
                    advice.append(f"{arm} motors report faults {reading.faults}: they clear on the next enable "
                                  "or by switching the arm power off and on")
                if away is not None and away > REST_DEVIATION_RAD:
                    advice.append(f"the {arm} arm is {away:.2f} rad from rest: 'home', or 'float' and guide it")
        except CellError as exc:
            report["arms"] = {"error": str(exc)}
    report["advice"] = advice or ["nothing to reset: the cell is free, the buses are quiet and the arms answer"]
    return report


def reset_cameras(cell: Cell) -> str:
    with CellLock("cell operator (camera reset)"):
        return cell.camera_driver.reset(sorted(set(cell.config.serials.values())))


def reset_buses(cell: Cell, bitrate: int = 1_000_000) -> str:
    script = " && ".join(f"ip link set {c} down && ip link set {c} up type can bitrate {bitrate}" for c in cell.channels)
    result = subprocess.run(["sudo", "-n", "sh", "-c", script], capture_output=True, text=True)
    if result.returncode == 0:
        return "CAN interfaces re-initialised"
    return "needs your password; run:\n  sudo sh -c '" + script + "'"
