"""The real YAM cell behind the ``Rig`` protocol: ``read`` and ``on``.

Start-up runs in a fixed order and stops at the first problem, before any motor is enabled:

1. occupancy: no other reader or traffic on either CAN bus, no running process that drives
   arms (``on`` only), every configured RealSense opens;
2. a passive read of every motor with DISABLE frames, and the joints checked against the
   model's limits and the cell's recorded rest pose (an encoder zero that moved shows here);
3. the table seen by each wrist D405, placed by the encoders through the CAD chain: it
   must be level and at the calibrated height. With no calibrated table, the wrists' plane
   becomes the table;
4. ``on`` only: the operator approves, then i2rt enables the arms holding their start pose.

The worker has no terminal, so every question goes through the cell's approval broker
(``ora/cell/operator/approval.py``) and is answered on the cell board, with ``ora cell
approve|decline`` or in the play terminal (``/approve``, ``/decline``): enabling the arms,
each motion plan (unless ``auto_approve``) and, when the way home is blocked at the end,
releasing the arms where they are.

During motion every tick checks the control loop is alive, joint tracking, joint torque
and motor temperature; a stop holds the arms where they are measured. Closing drives the
arms home (the recorded rest pose, else the start pose) and releases them there without a
question, since nothing drops at rest. A blocked way home holds and asks first; see
``_home_for_release`` for what happens when nobody approves.
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from typing import Any, Mapping

import numpy as np

from ora.environment.controllers import yam_station as Y

from ora.cell import alerts as cell_alerts
from ora.cell import calibration as cell_calibration
from ora.cell import config as cell_config
from ora.cell import geometry as cell_geometry
from ora.cell import homing
from ora.cell.cameras.realsense import RealSenseCamera
from ora.cell.lock import AT_REST, HOMES_ON_SIGNAL, HOMING, NOT_CONTROLLED, CellLock
from ora.cell.occupancy import check_occupancy
from ora.cell.operator import approval

from . import arms, devices, external_load, grippers, health, selfcheck
from . import operator_words as words
from .cell_adapter import CAMERA_MOUNTS
from .common import YamError
from .kinematics import StationKinematics
from .rig import TickResult
from .scene import DEFAULT_TABLE, build_cell_model
WRIST_TABLE_RANGE_M = (0.07, 0.6)
RETURN_SPEED_FRACTION = 0.5
HOME_LIMIT_S = 30.0
# Held in position control the arms sag a little under gravity: rest is reached once the
# command is there and every measured joint is within this, or the measurement has settled.
REST_TOLERANCE_RAD = 0.03
SETTLE_S = 1.0
HOME_SETTINGS = homing.HomeSettings(
    speed_rad_s=Y.MAX_JOINT_SPEED_RAD_S * RETURN_SPEED_FRACTION, period_s=Y.CONTROL_PERIOD_S,
    tracking_stop_rad=Y.TRACKING_ERROR_STOP_RAD, tolerance_rad=REST_TOLERANCE_RAD, limit_s=HOME_LIMIT_S,
    settle_s=SETTLE_S)
# How long each question waits for the operator before it counts as declined.
ENABLE_ANSWER_S = 300.0
MOTION_ANSWER_S = 180.0
RELEASE_ANSWER_S = 60.0
# Closing with a blocked way home: home is tried this many times, a release question
# between tries, all within the worker's close budget (``close_timeout_s``, the time its
# parent waits for ``close``, less a margin); a question needs at least MIN_ANSWER_S.
CLOSE_HOME_ATTEMPTS = 3
CLOSE_MARGIN_S = 15.0
MIN_ANSWER_S = 10.0
# How ``home`` names a straight path home it refused (``ora.cell.homing.path_problems``).
PATH_BLOCKED = "path home not clear"
# The arms are watched also between motions (a thread, every MONITOR_PERIOD_S): a stopped
# control loop, a motor warming up, a gripper that keeps pushing (``health.arm_alerts``).
MONITOR_PERIOD_S = 0.5


def _say(text: str) -> None:
    """A line for the person at the cell, in the worker's log (stderr; stdout is not the operator's)."""

    print(f"[yam rig] {text.rstrip()}", file=sys.stderr, flush=True)


class RealRig:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.mode = config.get("rig_mode")
        if self.mode not in {"read", "on"}:
            raise YamError(f"RealRig runs rig_mode 'read' or 'on', not {self.mode!r}")
        self.auto_approve = bool(config.get("auto_approve", False))
        self.close_budget_s = float(config.get("close_timeout_s", 120.0)) - CLOSE_MARGIN_S
        self.cell = cell_config.load(config.get("cell_config"))
        if self.cell.sim_state is not None:
            raise YamError(f"{self.cell.path} is a simulated cell (robot.sim): rig_mode {self.mode} needs a real cell")
        if set(self.cell.cameras) != set(CAMERA_MOUNTS) or set(self.cell.buses) != set(Y.ARMS):
            raise YamError(f"{self.cell.path} must name the buses of {list(Y.ARMS)} and the cameras {list(CAMERA_MOUNTS)}")
        self.buses = dict(self.cell.buses)
        self.calibration_path = self.cell.calibration_path
        self.calibration = cell_calibration.load(self.calibration_path)
        serials = self.cell.serials
        self.startup_report: dict[str, Any] = {"mode": self.mode, "cell": self.cell.cell_id,
                                               "cell_config": str(self.cell.path),
                                               "cell_calibration": str(self.calibration_path)}
        self._home_requested = False
        self.last_approval: dict[str, Any] | None = None
        self._in_step = False
        self._homing = False
        self._home_count = 0                # drives home: a motion plan waiting for approval is void after one
        self._home_signals = 0              # the operator's home requests (SIGUSR1)
        self.at_rest = False
        self.blocked_arms: list[str] = []
        self.readers: dict[str, devices.PassiveArmReader] = {}
        self.robots: dict[str, Any] = {}
        self.cameras_by_role: dict[str, RealSenseCamera] = {}
        self._command_lock = threading.RLock()      # the main thread and the watcher both command
        self._watcher: threading.Thread | None = None
        self._stop_watch = threading.Event()
        self._alerts: list[dict[str, Any]] = []
        self._gripper_pushing_since: dict[str, float] = {}
        self.loads = external_load.LoadMonitor(Y.ARMS)   # what each arm is pushed on with (external_load.py)
        # Smooth arm commands (runtime.smooth_arm_commands): each tick's target is reached along a streamed line with
        # its velocity as the feed-forward (_stream), not as one position step with a zero velocity target.
        self.smooth = config.get("smooth_arm_commands") is True
        self._segments: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}
        self._streamer: threading.Thread | None = None
        self._stop_stream = threading.Event()
        self.label = f"ORA worker yam rig_mode {self.mode} {HOMES_ON_SIGNAL}"
        self.lock = CellLock(self.label).acquire()
        try:
            self._start(config, serials)
        except BaseException:
            self.close(ask=False)
            raise

    # ----------------------------------------------------------- start-up
    def _start(self, config: Mapping[str, Any], serials: Mapping[str, str]) -> None:
        self.startup_report["occupancy"] = check_occupancy(
            mode=self.mode, channels=self.buses.values(), serials=serials.values(),
            ignore_processes=(*self.cell.ignore_processes, *config.get("ignore_processes", ())),
            markers=devices.MOTION_PROCESS_MARKERS)
        for role, serial in serials.items():
            self.cameras_by_role[role] = RealSenseCamera(role=role, serial=serial, mount=CAMERA_MOUNTS[role])
        self.readers = {arm: devices.PassiveArmReader(channel) for arm, channel in self.buses.items()}
        self._passive_refresh()
        self.kinematics = cell_calibration.apply(StationKinematics(build_cell_model(table=None)), self.calibration)
        self.kinematics.set_state(self._q, self._finger)
        joints = cell_geometry.joint_sanity(self._q, self.kinematics.arm_limits, self.calibration.get("rest_reference"))
        self.startup_report["joints"] = joints
        self._decide_table()
        # Every check before anything is enabled, each with its fix (selfcheck.py); on `on` one refusal lists them all.
        checks = selfcheck.startup_checks(
            readings=self._readings, joints=joints, wrist_table=self.startup_report.get("wrist_table"),
            calibration=self.calibration, cameras=self._camera_frames())
        self.startup_report["self_check"] = checks
        if self.mode == "on":
            reason = selfcheck.refusal(checks)
            if reason is not None:
                raise YamError(reason)
            self._enable(at_rest=joints["at_rest"])
        self.start_q = {arm: q.copy() for arm, q in self._q.items()}
        self._commanded = {arm: q.copy() for arm, q in self._q.items()}
        self._over = {arm: {"tracking": 0, "effort": 0} for arm in Y.ARMS}
        self._next_tick = time.monotonic()

    def _camera_frames(self) -> dict[str, dict[str, Any]]:
        """One frame of every camera: its valid depth, or why it gave none (the self-check's camera check)."""

        frames: dict[str, dict[str, Any]] = {}
        for role, camera in self.cameras_by_role.items():
            try:
                data = camera.capture()
            except Exception as exc:  # a camera that gives no frame is reported, the check refuses
                frames[role] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            frames[role] = {"depth_valid_fraction": float(data["depth_valid_fraction"]),
                            "depth_below_floor": bool(data["depth_below_floor"]),
                            "depth_pipeline_restarted": bool(data.get("depth_pipeline_restarted"))}
        return frames

    def _decide_table(self) -> None:
        checks, heights = {}, []
        calibrated = self.calibration.get("table")
        expected = float(calibrated["top_z_m"]) if calibrated else None
        # The cell's owner may widen the check (with a reason) while the arm chain is recalibrated.
        widened = dict(self.cell.wrist_table_check)
        limits = {key: float(widened[key]) for key in ("max_tilt_deg", "max_height_gap_m") if widened.get(key)}
        if widened:
            self.startup_report["wrist_table_widened"] = widened
        for arm in Y.ARMS:
            data = self.cameras_by_role[f"wrist_{arm}"].capture()
            plane = cell_geometry.table_plane(data["depth"], data["intrinsic_cv"], self.kinematics.wrist_camera(arm),
                                              near=WRIST_TABLE_RANGE_M[0], far=WRIST_TABLE_RANGE_M[1])
            checks[arm] = cell_geometry.wrist_table_check(plane, expected, **limits)
            if plane is not None and plane["tilt_deg"] <= limits.get("max_tilt_deg", cell_geometry.WRIST_TILT_LIMIT_DEG):
                heights.append(plane["height_m"])
        self.startup_report["wrist_table"] = checks
        if calibrated:
            table, source = cell_calibration.table_for_model(self.calibration), "cell_calibration"
        elif heights:
            table, source = {**DEFAULT_TABLE, "top_z_m": float(np.median(heights))}, "wrist_cameras"
        elif self.mode == "on":
            raise YamError("no calibrated table and neither wrist camera sees a level table: the planner "
                           "cannot keep the arms off the table")
        else:
            table, source = None, None
        self.startup_report["table"] = {"source": source, **(table or {})}
        self.table_top_z_m = None if table is None else float(table["top_z_m"])
        self.kinematics = cell_calibration.apply(StationKinematics(build_cell_model(table=table)), self.calibration)
        self.kinematics.set_state(self._q, self._finger)

    def _enable(self, *, at_rest: bool | None) -> None:
        try:
            # The operator's ``home`` (board, ``ora cell home``, play's /home) sends SIGUSR1. Before
            # the arms are enabled it withdraws the question below; afterwards it drives them home.
            signal.signal(signal.SIGUSR1, self._on_home_signal)
        except ValueError:
            self.startup_report["home_signal"] = "not installed (rig built off the main thread)"
        if not self.auto_approve:
            limits = self.calibration.get("gripper_limits") or {}
            known = set(limits) == set(Y.ARMS)
            title, text = words.enable(at_rest=at_rest, grippers_known=known)
            signals = self._home_signals
            approved, reason = approval.ask(
                "enable", title, text, {"cell": self.cell.cell_id, "at_rest": at_rest, "gripper_limits_known": known},
                timeout_s=ENABLE_ANSWER_S, asker=self.label,
                cancel=lambda: "the operator asked for home instead" if self._home_signals != signals else None)
            if not approved:
                raise YamError(f"the arms were not enabled: {reason}")
        for reader in self.readers.values():
            reader.close()
        self.readers = {}
        limits = self.calibration.get("gripper_limits") or {}
        for arm, channel in self.buses.items():
            # the gripper's force limit lowered, the gripper relaxed, temperatures recorded (arms.py)
            robot = arms.connect(channel, gripper_limits=limits.get(arm))
            self.robots[arm] = robot
            if arm not in limits:
                self.calibration.setdefault("gripper_limits", {})[arm] = [float(v) for v in robot._gripper_limits]
                cell_calibration.save(self.calibration_path, self.calibration)
        self.startup_report["gripper_limits"] = self.calibration.get("gripper_limits")
        self.startup_report["gripper_force_limit_n"] = {
            arm: getattr(robot, "_limit_gripper_force", None) for arm, robot in self.robots.items()}
        self._live_refresh()
        # i2rt enables holding every motor where it is; jaws at a stop would push into it.
        self._commanded = {}
        for arm in self.robots:
            self._command(arm, self._q[arm], self._finger[arm])
        self._start_watching()
        self._start_streaming()

    def _on_home_signal(self, *_: Any) -> None:
        # Mid-tick the request waits for the tick to end (``step`` reports it as a stop);
        # between motions the arms go home right here, in the main thread, also while a
        # question waits for its answer (the broker polls with short sleeps).
        self._home_signals += 1
        self.lock.relabel(self.label)
        if self._in_step:
            self._home_requested = True
        else:
            self.home()

    def home(self) -> str:
        """Drive the arms back to rest (the recorded rest pose, else the start pose) and keep
        holding them there; the episode goes on. Nothing is released."""

        if self.mode != "on" or not self.robots:
            return "nothing held"
        if getattr(self, "start_q", None) is None:
            return "still starting: the arms hold the pose they were enabled in"
        if self._homing:
            return "already on the way home"           # a signal during the drive
        self._homing, self._home_count = True, self._home_count + 1
        self.lock.relabel(f"{self.label} {HOMING}")
        outcome = None
        try:
            outcome = self._drive_home()
            return outcome
        finally:
            self._homing = False
            if outcome != "at rest, holding":
                self.lock.relabel(self.label)

    def _drive_home(self) -> str:
        """The arms to rest (``ora.cell.homing.drive``: the straight line checked against the table and between the
        arms first, then a ramped joint-space drive); anything short of rest holds them where they are."""

        rest = self.calibration.get("rest_reference")
        goal = {arm: np.asarray(rest[arm], float) if rest else self.start_q[arm] for arm in self.robots}
        self.blocked_arms = []

        def read() -> dict[str, np.ndarray]:
            self._live_refresh()
            return self._q

        def command(targets: Mapping[str, np.ndarray]) -> None:
            for arm, q in targets.items():
                self._command(arm, q, self._finger[arm], smooth=True)

        outcome = homing.drive(
            goal, read=read, command=command, wait=self._pace, settings=HOME_SETTINGS, kinematics=self.kinematics,
            alive=lambda: {arm: bool(getattr(robot.motor_chain, "running", True)) for arm, robot in self.robots.items()})
        self.kinematics.set_state(self._q, self._finger)       # the path check moved the station model
        if outcome.state == "at_rest":
            for robot in self.robots.values():
                grippers.relax(robot)                # at rest nothing is gripped: the jaws give no torque
            self.at_rest = True
            self.lock.relabel(f"{self.label} {AT_REST}")
            return "at rest, holding"
        if outcome.state == "dead" and not outcome.arms:
            _say("home: no arm's control loop runs (a motor stopped or was disabled); nothing to drive")
            return "no control loop running: nothing driven"
        self._hold_measured()
        if outcome.state == "path_blocked":
            _say(f"home refused: {outcome.detail}; holding here")
            return f"{PATH_BLOCKED} ({outcome.detail}): holding where the arms are"
        if outcome.state == "dead":
            _say(f"home stopped: {outcome.detail}; holding the other arms here")
            return "a control loop stopped: holding where the arms are"
        if outcome.state == "blocked":
            self.blocked_arms = outcome.arms
            _say(f"home stopped: {outcome.detail}; holding here")
            return "blocked: holding where the arms are"
        if outcome.state == "not_at_rest":
            _say(f"home: commanded rest but the arms stay {outcome.off_rad:.3f} rad from it; holding here")
            return f"not at rest: {outcome.off_rad:.3f} rad from it, holding"
        _say(f"home did not reach rest in {HOME_LIMIT_S:g} s; holding here")
        return "not at rest in time: holding where the arms are"

    # ----------------------------------------------------------- state
    def _passive_refresh(self) -> None:
        limits = self.calibration.get("gripper_limits") or {}
        self._readings = {arm: devices.arm_reading(reader.read(), limits.get(arm))
                          for arm, reader in self.readers.items()}
        self._q = {arm: reading["q"] for arm, reading in self._readings.items()}
        # Without gripper limits a passive read cannot tell the opening; such a reading is
        # reported as open and named in the start-up report.
        self._finger = {arm: (reading["gripper_opening"] if reading["gripper_opening"] is not None else 1.0)
                        * Y.FINGER_OPEN_M for arm, reading in self._readings.items()}
        self.startup_report["gripper_opening_known"] = {arm: r["gripper_opening"] is not None
                                                       for arm, r in self._readings.items()}

    def _live_refresh(self) -> None:
        self._observations = {arm: robot.get_observations() for arm, robot in self.robots.items()}
        self._q = {arm: np.asarray(obs["joint_pos"], float).copy() for arm, obs in self._observations.items()}
        self._finger = {arm: float(np.clip(obs["gripper_pos"][0], 0.0, 1.0)) * Y.FINGER_OPEN_M
                        for arm, obs in self._observations.items()}

    def arm_q(self, arm: str) -> np.ndarray:
        return self._q[arm].copy()

    def finger_m(self, arm: str) -> float:
        return float(self._finger[arm])

    # ----------------------------------------------------------- control
    def _pace(self) -> None:
        self._next_tick = max(self._next_tick + Y.CONTROL_PERIOD_S, time.monotonic())
        delay = self._next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def _command(self, arm: str, q: np.ndarray, finger_m: float, *, smooth: bool = False) -> None:
        """Command one arm. ``smooth`` (a motion's tick, with smooth arm commands on) reaches ``q`` along the line
        from the last target within one tick (``_stream``); anything else, a hold above all, is commanded at once."""

        opening = float(np.clip(finger_m / Y.FINGER_OPEN_M, *Y.GRIPPER_HOLD_RANGE))
        target = np.asarray(q, float).copy()
        with self._command_lock:
            if smooth and self.smooth and self._streamer is not None:
                start = np.asarray(self._commanded.get(arm, target), float).copy()
                self._segments[arm] = (start, target, time.monotonic(), opening)
                self._send(arm, start, (target - start) / Y.CONTROL_PERIOD_S, opening)
            else:
                self._segments.pop(arm, None)
                self._send(arm, target, None, opening)
            self._commanded[arm] = target

    def _send(self, arm: str, q: np.ndarray, velocity: np.ndarray | None, opening: float) -> None:
        """One command to i2rt (under the command lock): the position, the velocity feed-forward when there is one,
        then the gripper's force-control torque, which every i2rt command zeroes."""

        robot = self.robots[arm]
        command = np.append(np.asarray(q, float), opening)
        if velocity is not None and hasattr(robot, "command_joint_state"):
            robot.command_joint_state({"pos": command, "vel": np.append(np.asarray(velocity, float), 0.0)})
        else:
            robot.command_joint_pos(command)
        grippers.apply_force(robot, opening, float(self._finger[arm]) / Y.FINGER_OPEN_M)

    def _stream(self) -> None:
        """Between ticks: each arm's target moves along its segment at ARM_COMMAND_STREAM_HZ; past the tick it is the
        segment's end with no feed-forward."""

        period = 1.0 / Y.ARM_COMMAND_STREAM_HZ
        while not self._stop_stream.wait(period):
            with self._command_lock:
                now = time.monotonic()
                for arm, (start, end, t0, opening) in list(self._segments.items()):
                    if arm not in self.robots:
                        continue
                    share = (now - t0) / Y.CONTROL_PERIOD_S
                    if share >= 1.0:
                        self._send(arm, end, None, opening)
                        del self._segments[arm]
                    else:
                        self._send(arm, start + (end - start) * share, (end - start) / Y.CONTROL_PERIOD_S, opening)

    def _start_streaming(self) -> None:
        if not self.smooth or self._streamer is not None:
            return
        self._stop_stream.clear()
        self._streamer = threading.Thread(target=self._stream, name="yam-arm-stream", daemon=True)
        self._streamer.start()

    def _stop_streaming(self) -> None:
        self._stop_stream.set()
        if self._streamer is not None:
            self._streamer.join(timeout=1.0)
        self._streamer = None
        with self._command_lock:
            self._segments.clear()

    def step(self, arm_targets: Mapping[str, np.ndarray], finger_targets: Mapping[str, float]) -> TickResult:
        if self.mode == "on" and not self.robots:
            raise YamError("the arms were released (the rig closed): nothing can move them")
        if self.mode == "read":
            self._pace()
            self._passive_refresh()
            return TickResult(self._snapshot_q(), dict(self._finger))
        if AT_REST in self.lock.purpose:
            self.lock.relabel(self.label)
        self.at_rest = False
        self._in_step = True
        try:
            self._pace()
            for arm in Y.ARMS:
                if arm in finger_targets:
                    grippers.take_force_control(self.robots[arm])
                self._command(arm, arm_targets.get(arm, self._commanded[arm]),
                              finger_targets.get(arm, self._finger[arm]), smooth=arm in arm_targets)
            self._live_refresh()
            events = self._guards() + [{**alert, "code": "rig_alert", "alert": alert["code"]}
                                       for alert in self._alerts if alert["level"] == "critical"]
            stop = approval.take_stop_request()
            if stop is not None:
                # The operator's "stop here": the motion ends where the arms are, held there.
                events.append({"code": "operator_hold", "detail": f"the operator stopped the motion here ({stop.get('by')})"})
        except KeyboardInterrupt:
            events = [{"code": "operator_stop", "detail": "Ctrl-C at the cell"}]
        finally:
            self._in_step = False
        if self._home_requested:
            self._home_requested = False
            outcome = self.home()
            self._commanded = {arm: q.copy() for arm, q in self._q.items()}
            events.append({"code": "operator_home", "detail": f"the operator sent the arms home: {outcome}"})
        elif events:
            self._hold_measured()
        return TickResult(self._snapshot_q(), dict(self._finger), events)

    def _snapshot_q(self) -> dict[str, np.ndarray]:
        return {arm: q.copy() for arm, q in self._q.items()}

    def _guards(self) -> list[dict[str, Any]]:
        events = []
        motor_types = [kind for _, kind in Y.ARM_MOTORS]
        for arm, robot in self.robots.items():
            if not getattr(robot.motor_chain, "running", True):
                events.append({"code": "motor_fault", "arm": arm, "detail": "the i2rt control loop stopped"})
                continue
            obs = self._observations[arm]
            temperature = max(np.max(obs.get("temp_mos", [0.0])), np.max(obs.get("temp_rotor", [0.0])))
            if temperature >= Y.MOTOR_TEMPERATURE_STOP_C:
                events.append({"code": "motor_overheat", "arm": arm, "temperature_c": float(temperature)})
            tracking = float(np.max(np.abs(self._commanded[arm] - self._q[arm])))
            effort = np.abs(np.asarray(obs["joint_eff"], float)) / np.array([Y.MOTOR_TORQUE_MAX_NM[k] for k in motor_types])
            for name, over, value in (("tracking", tracking > Y.TRACKING_ERROR_STOP_RAD, tracking),
                                      ("effort", bool(np.max(effort) > Y.JOINT_EFFORT_STOP_FRACTION),
                                       float(np.max(effort)))):
                self._over[arm][name] = self._over[arm][name] + 1 if over else 0
                if self._over[arm][name] > Y.TRACKING_ERROR_GRACE_TICKS:
                    code = "tracking_error" if name == "tracking" else "joint_effort"
                    events.append({"code": code, "arm": arm, "value": value})
            # Pushed on from outside: measured minus i2rt's own model torque (external_load.py).
            load = external_load.residual(robot, obs)
            if load is not None:
                event = self.loads.update(arm, load)
                if event is not None:
                    events.append(event)
        return events

    def _hold_measured(self) -> None:
        for arm, robot in self.robots.items():
            if getattr(robot.motor_chain, "running", True):
                self._command(arm, self._q[arm], self._finger[arm])
            self._over[arm] = {"tracking": 0, "effort": 0}
            self.loads.reset(arm)

    # ----------------------------------------------------------- watching between motions
    def alerts(self) -> list[dict[str, Any]]:
        """What is wrong with the held arms now (``ora.cell.alerts`` entries)."""

        return [dict(alert) for alert in self._alerts]

    def _start_watching(self) -> None:
        self._stop_watch.clear()
        self._watcher = threading.Thread(target=self._watch, name="yam-arm-watch", daemon=True)
        self._watcher.start()

    def _stop_watching(self) -> None:
        self._stop_watch.set()
        if self._watcher is not None and self._watcher is not threading.current_thread():
            self._watcher.join(timeout=2.0)
        self._watcher = None

    def _watch(self) -> None:
        published = None
        while not self._stop_watch.is_set():
            try:
                alerts = self._check_arms(time.monotonic())
            except Exception as exc:  # noqa: BLE001 - the watcher must keep reporting
                alerts = [{"code": "watch_failed", "level": "warning", "text": f"手臂监测出错：{exc}",
                           "since_unix_s": time.time()}]
            self._alerts = alerts
            key = [(a["code"], a.get("arm"), a.get("motor"), a["level"]) for a in alerts]
            if key != published:
                published = key
                cell_alerts.publish(alerts, source=self.label)
                for alert in alerts:
                    _say(f"[{alert['level']}] {alert['text']}")
                not_held = any(a["code"] == "arms_not_controlled" for a in alerts)
                if not_held and NOT_CONTROLLED not in self.lock.purpose:
                    self.lock.relabel(f"{self.label} {NOT_CONTROLLED}")
            self._stop_watch.wait(MONITOR_PERIOD_S)

    def _check_arms(self, now: float) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        stamp = time.time()
        for arm, robot in list(self.robots.items()):
            with self._command_lock:
                alerts += health.arm_alerts(arm, robot, now=now, stamp=stamp, pushing_since=self._gripper_pushing_since,
                                            stopped_text="请用 /quit 结束这次运行，查明原因再重启。",
                                            may_relax=not self._in_step)
        return alerts

    def operator_approval(self, summary: Mapping[str, Any]) -> tuple[bool, str | None]:
        """Ask the operator about one plan; a home while it waits voids it (it starts where the arms were).
        An operator's "stop here" sent while the agent planned it refuses it before anything moves."""

        now = time.time()
        stop = approval.take_stop_request()
        if stop is not None:
            self.last_approval = {"request_id": None, "created_unix_s": now, "answered_unix_s": now,
                                  "by": f"stop here ({stop.get('by')})", "approved": False}
            return False, (f"the operator stopped the robot while this motion was being planned ({stop.get('by')}): "
                           "read the operator's latest message and the fresh observation before moving again")
        if self.auto_approve:
            self.last_approval = {"request_id": None, "created_unix_s": now, "answered_unix_s": now,
                                  "by": "runtime.auto_approve", "approved": True}
            return True, None
        homes = self._home_count

        def moved() -> str | None:
            return ("the operator sent the arms home while this plan waited: it no longer starts where the "
                    "arms are" if self._home_count != homes else None)

        title, text = words.motion(summary)
        record = approval.ask_record("motion", title, text, summary, timeout_s=MOTION_ANSWER_S, asker=self.label,
                                     cancel=moved)
        approved = record.get("approved") is True
        reason = None if approved else (record.get("reason") or f"declined ({record.get('by') or 'the operator'})")
        # When the question was asked and answered, and by whom: the motion's receipt keeps it.
        self.last_approval = {"request_id": record.get("id"), "created_unix_s": record.get("created_unix_s"),
                              "answered_unix_s": record.get("answered_unix_s"), "by": record.get("by"),
                              "approved": approved}
        if approved and moved():
            return False, moved()
        return (True, None) if approved else (False, reason)

    # ----------------------------------------------------------- cameras
    def cameras(self) -> dict[str, dict[str, Any]]:
        if not self.cameras_by_role:
            raise YamError("the rig was closed: no cameras")
        self.kinematics.set_state(self._q, self._finger)
        out = {}
        for role, camera in self.cameras_by_role.items():
            data = camera.capture()
            if camera.mount == "end_effector":
                data["cam2world_cv"] = self.kinematics.wrist_camera(role.removeprefix("wrist_"))
                data["calibration_source"] = "station_cad_and_encoders"
            else:
                fixed = (self.calibration.get("cameras") or {}).get(role)
                data["cam2world_cv"] = None if fixed is None else np.asarray(fixed["cam2world_cv"], float)
                data["calibration_source"] = None if fixed is None else fixed.get("method", "cell_calibration")
            out[role] = data
        return out

    # ----------------------------------------------------------- shutdown
    def _return_to_start(self) -> None:
        """Back to the start pose, slowly; an arm whose control loop died is left to its timeout.
        Only a start-up failure uses this (``close(ask=False)``)."""

        live = [arm for arm, robot in self.robots.items() if getattr(robot.motor_chain, "running", True)]
        step = Y.MAX_JOINT_SPEED_RAD_S * RETURN_SPEED_FRACTION * Y.CONTROL_PERIOD_S
        for _ in range(int(20.0 / Y.CONTROL_PERIOD_S)):
            remaining = {arm: self.start_q[arm] - self._q[arm] for arm in live}
            if all(np.max(np.abs(r)) < REST_TOLERANCE_RAD for r in remaining.values()):
                return
            targets = {arm: self._q[arm] + np.clip(r, -step, step) for arm, r in remaining.items()}
            result = self.step(targets, {})
            stops = [event for event in result.events if event.get("arm") in live or "arm" not in event]
            if stops:
                _say(f"return to the start pose stopped: {stops}")
                return

    def _home_for_release(self) -> None:
        """Closing: drive the arms home and release them there, without a question.

        A blocked way home (or one not finished in time) holds the arms and asks the operator
        (kind ``release``) whether to release them where they are. Declined or unanswered, the
        arms stay held and home is tried again, up to CLOSE_HOME_ATTEMPTS times within the
        close budget. At that bound the arms are released in place anyway, and the last
        question says so: the worker is about to end (its parent stops waiting after
        ``close_timeout_s``), and without its control loop the motors let go after their
        timeout regardless, so an announced release beats an unannounced one. The way to keep
        them held past the bound is the board's 回到起始位 (``ora cell home``, play's /home):
        the board takes the arms over and holds them for as long as needed. A home sent while
        a question waits that reaches rest ends the question; the arms are then released at
        rest. ``auto_approve`` asks nothing: it retries home, then releases in place.
        """

        deadline = time.monotonic() + self.close_budget_s
        for attempt in range(1, CLOSE_HOME_ATTEMPTS + 1):
            outcome = self.home()
            if self.at_rest:
                return
            remaining = deadline - time.monotonic()
            if remaining < MIN_ANSWER_S:
                break
            if self.auto_approve:
                _say(f"closing: {outcome}; auto_approve, so home is tried again without a question")
                continue
            timeout = min(RELEASE_ANSWER_S, remaining)
            last = attempt == CLOSE_HOME_ATTEMPTS or remaining - timeout < MIN_ANSWER_S
            _say(f"closing: {outcome}; asking whether to release in place ({attempt}/{CLOSE_HOME_ATTEMPTS})")
            title, text = words.release(
                blocked=self.blocked_arms, path_blocked=outcome.startswith(PATH_BLOCKED), attempt=attempt,
                attempts=CLOSE_HOME_ATTEMPTS, timeout_s=timeout, home_limit_s=HOME_LIMIT_S, last=last)
            approved, reason = approval.ask(
                "release", title, text,
                {"attempt": attempt, "attempts": CLOSE_HOME_ATTEMPTS, "home": outcome, "blocked_arms": self.blocked_arms,
                 "release_anyway_after": last, "q": {arm: q for arm, q in self._q.items()}},
                timeout_s=timeout, asker=self.label, cancel=lambda: "the arms reached rest" if self.at_rest else None)
            if self.at_rest:
                return
            if approved:
                _say("closing: releasing the arms in place, as approved")
                return
            if last:
                break
            _say(f"closing: not released ({reason}); holding and trying home again")
        _say("closing: the arms did not get home and no release was approved; releasing them in place now, "
             "because this worker is ending")

    def close(self, *, ask: bool = True) -> None:
        try:
            self._stop_watching()
            self._stop_streaming()
            if self.robots:
                if ask:
                    self._home_for_release()
                elif getattr(self, "start_q", None) is not None:
                    self._return_to_start()
                arms.release(self.robots, self.buses, say=_say)
                self.robots = {}
            cell_alerts.clear(source=self.label)
        finally:
            for reader in self.readers.values():
                reader.close()
            self.readers = {}
            for camera in self.cameras_by_role.values():
                camera.close()
            self.cameras_by_role = {}
            self.lock.release()
