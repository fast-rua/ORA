"""Small supervised motions for calibration: the grippers, and the fixed cameras on the moving arms.

``grippers`` lets the robot's driver find each gripper's [closed, open] limits (each opens
and closes once, its arm held at rest). ``calibrate_cameras`` moves every arm together in
joint space at ``SPEED_RAD_S`` through a few poses near rest (the robot's own
``calibration_offsets``, checked here for table and inter-arm clearance along the whole
path), captures the fixed cameras at rest first and at every pose, writing each capture to
disk as it is taken, returns the arms to rest and releases them, then registers each fixed
camera on the moving arms (``geometry.register_on_moving_arms``).

Throughout, the operator's ``home`` reaches this process by SIGUSR1: the motion stops and
the arms drive back to where they started before being released. A tracking error, a
dead control loop, a critical alert (a motor near its protection) or another program
opening a bus does the same on its own. While the arms are held their alerts are
published (``ora.cell.alerts``), so the board shows them.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..adapter import Camera
from ..cameras.charuco import by_colour, chessboard_pose, flip_variants
from .. import alerts as cell_alerts
from ..errors import CellError
from ..geometry import register_on_moving_arms
from ..lock import HOMES_ON_SIGNAL, CellLock
from .context import Cell

SPEED_RAD_S = 0.15
HOLD_S = 1.5
CAPTURE_FRAMES = 5
MIN_TCP_ABOVE_TABLE_M = 0.08
MIN_INTER_ARM_M = 0.15
POSES = 3
ALERT_PERIOD_S = 0.5
RECORD_ROOT = Path("~/.cache/ora/cell-calibration")


class Stop(Exception):
    """The motion must end: asked for by the operator, or a guard tripped."""


def plan(cell: Cell) -> dict[str, Any]:
    """Calibration poses whose straight joint path from rest keeps every TCP above the table
    and the arms apart."""

    rest = cell.rest()
    table = cell.table()
    table_z = table["top_z_m"] if table else 0.0
    kinematics = cell.kinematics()
    arms = cell.adapter.arms
    poses = []
    for offset in cell.adapter.calibration_offsets():
        q = {arm: rest[arm] + offset for arm in arms}
        if any(np.any(q[a] < kinematics.arm_limits[a][:, 0]) or np.any(q[a] > kinematics.arm_limits[a][:, 1])
               for a in arms):
            continue
        clear = True
        for t in np.linspace(0.0, 1.0, 12):
            kinematics.set_state({a: rest[a] + t * offset for a in arms})
            if any(kinematics.tcp(a)[2, 3] - table_z < MIN_TCP_ABOVE_TABLE_M for a in arms) \
                    or (len(arms) > 1 and kinematics.inter_arm_distance() < MIN_INTER_ARM_M):
                clear = False
                break
        if clear:
            kinematics.set_state(q)
            poses.append({"offset": offset.tolist(), "q": {a: q[a].tolist() for a in arms},
                          "tcp": {a: kinematics.tcp(a)[:3, 3].tolist() for a in arms}})
    return {"rest": {a: rest[a].tolist() for a in arms}, "poses": poses[:POSES], "table_z": table_z}


class Mover:
    """Every arm under the driver, moved together by joint interpolation with guards."""

    def __init__(self, cell: Cell) -> None:
        self.cell = cell
        self.stop_reason: str | None = None
        self.released = False
        self.held = cell.adapter.take_over(cell.adapter.arms, mode="hold", gripper_limits=cell.gripper_limits())
        self.own = {channel: cell.bus.receivers(channel) for channel in cell.channels}
        self.source = f"cell operator motion (pid {os.getpid()})"
        self._alerts_at = 0.0
        self._published: list[tuple[Any, ...]] | None = None
        signal.signal(signal.SIGUSR1, self._on_stop)
        signal.signal(signal.SIGUSR2, self._on_estop)
        signal.signal(signal.SIGINT, self._on_stop)

    def _on_stop(self, *_: Any) -> None:
        self.stop_reason = self.stop_reason or "stop requested (operator home, or Ctrl-C)"

    def _on_estop(self, *_: Any) -> None:
        self.stop_reason = "every motor was disabled by the operator"
        self.released = True

    def _guard(self, commanded: dict[str, np.ndarray]) -> None:
        if self.stop_reason:
            raise Stop(self.stop_reason)
        dead = [arm for arm, alive in self.held.alive().items() if not alive]
        if dead:
            raise Stop(f"the {', '.join(dead)} control loop stopped (a motor reported an error or was disabled)")
        measured = self.held.joints()
        for arm, q in commanded.items():
            lag = q - measured[arm]
            if np.max(np.abs(lag)) > self.cell.adapter.tracking_stop_rad:
                joint = int(np.argmax(np.abs(lag)))
                raise Stop(f"the {arm} arm does not follow: joint {joint + 1} lags {lag[joint]:+.3f} rad")
        for channel, own in self.own.items():
            if self.cell.bus.receivers(channel) > own:
                raise Stop(f"another program opened {channel}")
        critical = [a for a in self.watch() if a.get("level") == "critical"]
        if critical:
            raise Stop(critical[0].get("text") or critical[0].get("code"))

    def watch(self, force: bool = False) -> list[dict[str, Any]]:
        """The held arms' alerts, published when they change (every ALERT_PERIOD_S at most)."""

        now = time.monotonic()
        if not force and now - self._alerts_at < ALERT_PERIOD_S:
            return []
        self._alerts_at = now
        found = list(self.held.alerts()) if hasattr(self.held, "alerts") else []
        key = [(a.get("code"), a.get("arm"), a.get("motor"), a.get("level")) for a in found]
        if key != self._published:
            self._published = key
            cell_alerts.publish(found, source=self.source)
            for alert in found:
                print(f"[{alert.get('level')}] {alert.get('text')}", flush=True)
        return found

    def move(self, goal: dict[str, np.ndarray], *, guarded: bool = True) -> None:
        commanded = self.held.joints()
        step = SPEED_RAD_S / self.cell.adapter.control_hz
        while any(np.max(np.abs(goal[a] - commanded[a])) > 1e-4 for a in commanded):
            commanded = {a: q + np.clip(goal[a] - q, -step, step) for a, q in commanded.items()}
            self.held.command(commanded)
            time.sleep(1.0 / self.cell.adapter.control_hz)
            if guarded:
                self._guard(commanded)
        end = time.monotonic() + 1.0
        while time.monotonic() < end and any(np.max(np.abs(goal[a] - q)) > 0.01 for a, q in self.held.joints().items()):
            time.sleep(0.02)

    def hold_here(self) -> None:
        self.watch(force=True)
        if not self.released:
            alive = self.held.alive()
            self.held.command({a: q for a, q in self.held.joints().items() if alive.get(a)})

    def back_and_release(self, start: dict[str, np.ndarray]) -> None:
        try:
            if not self.released:
                self.stop_reason = None
                self.move(start, guarded=False)
                time.sleep(0.5)
        finally:
            if not self.released:
                self.held.release()
                self.released = True
            # The last alerts stay published while this process lives: the person sees why it stopped.


def _median_depth(camera: Camera, frames: int) -> dict[str, Any]:
    shots = [camera.capture() for _ in range(frames)]
    stack = np.stack([s["depth"] for s in shots]).astype(np.float32)
    stack[stack <= 0] = np.nan
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            depth = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0)
    return {**shots[-1], "depth": depth}


def capture(cell: Cell, roles: list[str], frames: int = CAPTURE_FRAMES) -> dict[str, dict[str, Any]]:
    out = {}
    for role in roles:
        camera = cell.open_camera(role)
        try:
            out[role] = _median_depth(camera, frames)
        finally:
            camera.close()
    return out


def grippers(cell: Cell, *, save: bool) -> dict[str, Any]:
    stop = {"asked": False}
    with CellLock(f"cell operator (gripper calibration) {HOMES_ON_SIGNAL}"):
        signal.signal(signal.SIGUSR1, lambda *_: stop.update(asked=True))
        found = cell.adapter.calibrate_grippers(cell.adapter.arms, lambda: stop["asked"])
    if save and found:
        calibration = cell.calibration()
        calibration.setdefault("gripper_limits", {}).update(found)
        cell.save_calibration(calibration)
    return {"gripper_limits": found, "stopped": stop["asked"]}


def calibrate_cameras(cell: Cell, *, save: bool) -> dict[str, Any]:
    planned = plan(cell)
    if len(planned["poses"]) < 2:
        raise CellError("fewer than two safe calibration poses: not moving")
    roles = cell.fixed_cameras()
    views: dict[str, list[tuple[np.ndarray, dict]]] = {role: [] for role in roles}
    intrinsics: dict[str, Any] = {}
    record = RECORD_ROOT.expanduser() / time.strftime("%Y%m%d-%H%M%S")
    record.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"poses": [], "stopped": None, "record": str(record)}

    def take(label: str, q: dict[str, np.ndarray]) -> None:
        for role, data in capture(cell, roles).items():
            views[role].append((data["depth"], {a: v.tolist() for a, v in q.items()}))
            intrinsics[role] = data["intrinsic_cv"]
            np.savez_compressed(record / f"{label}-{role}.npz", depth=data["depth"], rgb=data["rgb"],
                                intrinsic=np.asarray(data["intrinsic_cv"]), **{f"q_{a}": v for a, v in q.items()})
        report["poses"].append({"label": label, "q": {a: v.tolist() for a, v in q.items()}})

    with CellLock(f"cell operator (calibration poses) {HOMES_ON_SIGNAL}"):
        cell.quiet()
        mover = Mover(cell)
        start = mover.held.joints()
        try:
            time.sleep(HOLD_S)
            take("rest", mover.held.joints())
            for i, pose in enumerate(planned["poses"]):
                print(f"moving: offset {np.round(pose['offset'], 2).tolist()} rad from rest", flush=True)
                mover.move({a: np.asarray(q) for a, q in pose["q"].items()})
                time.sleep(HOLD_S)
                take(f"pose-{i}", mover.held.joints())
        except Stop as stop:
            report["stopped"] = str(stop)
            print(f"stopped: {stop}", flush=True)
            mover.hold_here()
        finally:
            mover.back_and_release(start)
    kinematics = cell.kinematics()
    calibration = cell.calibration()
    for role in roles:
        if len(views[role]) < 2:
            continue
        rough = [np.asarray(c["cam2world_cv"], float) for c in
                 [(calibration.get("cameras") or {}).get(role)] +
                 [r for r in calibration.get("rejected_cameras", []) if r.get("serial") == cell.config.cameras[role].serial]
                 if c]
        if not rough:
            report[role] = {"accepted": False, "why": "no rough pose to start from"}
            continue
        result = register_on_moving_arms(views[role], intrinsics[role], kinematics, rough)
        report[role] = {k: v for k, v in result.items() if k != "cam2world_cv"}
        if save and result.get("accepted"):
            calibration.setdefault("cameras", {})[role] = {
                "cam2world_cv": np.asarray(result["cam2world_cv"]).tolist(), "method": "moving_arm_registration",
                "provisional": False, "median_residual_m": result["median_residual_m"],
                "inlier_fraction": result["inlier_fraction"], "record": str(record),
                "serial": cell.config.cameras[role].serial, "intrinsic_cv": intrinsics[role],
                "captured_unix_s": time.time()}
    if save:
        cell.save_calibration(calibration)
    return report


# ---------------------------------------------------------------- a checkerboard seen from the moving wrists
def _mean_pose(poses: list[np.ndarray]) -> np.ndarray:
    u, _, vt = np.linalg.svd(sum(p[:3, :3] for p in poses))
    rotation = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rotation, np.mean([p[:3, 3] for p in poses], axis=0)
    return out


def _gap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    turn = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1.0, 1.0)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(np.degrees(np.arccos(turn)))


def calibrate_with_board(cell: Cell, *, square_mm: float, pattern: tuple[int, int], save: bool,
                         grip: bool = True) -> dict[str, Any]:
    """Fixed cameras from a checkerboard lying on the table, placed in the world by the wrist
    cameras as the arms rise through the calibration poses (where the jaws no longer hide it).

    At the first pose each gripper closes and opens once. Every wrist view of the board
    gives one board pose in the world through the encoders and the CAD chain; their spread
    is reported (it measures that chain), their mean anchors the board, and each fixed
    camera follows from its own view of the board. A plain checkerboard's half turn is
    resolved by its colours where they tell (``charuco.by_colour``), else against the views
    already used and, for a fixed camera, its rough pose."""

    planned = plan(cell)
    if len(planned["poses"]) < 2:
        raise CellError("fewer than two safe calibration poses: not moving")
    square = square_mm / 1000.0
    roles = list(cell.config.cameras)
    record = RECORD_ROOT.expanduser() / time.strftime("board-%Y%m%d-%H%M%S")
    record.mkdir(parents=True, exist_ok=True)
    captures: list[tuple[str, dict[str, np.ndarray], dict[str, dict[str, Any]]]] = []
    report: dict[str, Any] = {"record": str(record), "stopped": None, "grippers": None}

    def take(label: str, q: dict[str, np.ndarray]) -> None:
        shots = capture(cell, roles, frames=3)
        captures.append((label, q, shots))
        for role, data in shots.items():
            np.savez_compressed(record / f"{label}-{role}.npz", rgb=data["rgb"], depth=data["depth"],
                                intrinsic=np.asarray(data["intrinsic_cv"]), **{f"q_{a}": v for a, v in q.items()})

    with CellLock(f"cell operator (board calibration poses) {HOMES_ON_SIGNAL}"):
        cell.quiet()
        mover = Mover(cell)
        start = mover.held.joints()
        try:
            time.sleep(HOLD_S)
            take("rest", mover.held.joints())
            for i, pose in enumerate(planned["poses"]):
                print(f"moving: offset {np.round(pose['offset'], 2).tolist()} rad from rest", flush=True)
                mover.move({a: np.asarray(q) for a, q in pose["q"].items()})
                if grip and i == 0 and hasattr(mover.held, "grip"):
                    print("grippers: close, then open", flush=True)
                    closed = mover.held.grip({a: 0.0 for a in cell.adapter.arms})
                    opened = mover.held.grip({a: 1.0 for a in cell.adapter.arms})
                    report["grippers"] = {"closed": closed, "opened": opened}
                time.sleep(HOLD_S)
                take(f"pose-{i}", mover.held.joints())
        except Stop as stop:
            report["stopped"] = str(stop)
            print(f"stopped: {stop}", flush=True)
            mover.hold_here()
        finally:
            mover.back_and_release(start)

    kinematics = cell.kinematics()
    seen: dict[str, list[dict[str, Any]]] = {role: [] for role in roles}
    anchors: list[tuple[str, np.ndarray]] = []
    for label, q, shots in captures:
        kinematics.set_state(q)
        for role, data in shots.items():
            pose = chessboard_pose(data["rgb"], data["intrinsic_cv"], data["distortion"]["coeffs"], square, pattern,
                                   depth=data["depth"])
            if pose is None:
                continue
            pose["board2cam"], pose["by_colour"] = by_colour(data["rgb"], pose["board2cam"], data["intrinsic_cv"],
                                                             data["distortion"]["coeffs"], square, pattern)
            seen[role].append({"label": label, **pose})
            if role in cell.adapter.wrist_cameras:
                camera2world = kinematics.wrist_camera(cell.adapter.wrist_cameras[role])
                variants = [camera2world @ v for v in flip_variants(pose["board2cam"], square, pattern)]
                if anchors:
                    variants.sort(key=lambda v: _gap(v, anchors[0][1]))
                anchors.append((f"{role}@{label}", variants[0]))
    report["detections"] = {role: [(s["label"], s["corners"], round(s["reprojection_px"], 2),
                                    None if s.get("depth_scale_ratio") is None else round(s["depth_scale_ratio"], 3))
                                   for s in views] for role, views in seen.items()}
    ratios = [s["depth_scale_ratio"] for views in seen.values() for s in views if s.get("depth_scale_ratio")]
    report["depth_scale_ratio"] = None if not ratios else round(float(np.median(ratios)), 3)
    if ratios and abs(np.median(ratios) - 1.0) > 0.03:
        report["why"] = (f"depth puts the board {np.median(ratios):.3f}x as far as {square_mm} mm squares do: "
                         "re-measure one square; nothing saved")
        return report
    if not anchors:
        report["why"] = "no wrist camera saw the whole board at any pose; nothing saved"
        return report
    board2world = _mean_pose([pose for _, pose in anchors])
    spread = [_gap(pose, board2world) for _, pose in anchors]
    report["anchors"] = {name: {"mm": round(g[0] * 1000, 1), "deg": round(g[1], 2)} for (name, _), g in zip(anchors, spread)}
    calibration = cell.calibration()
    placed = {}
    anchored_by_colour = all(s["by_colour"] for w in cell.adapter.wrist_cameras if w in seen for s in seen[w])
    for role in cell.fixed_cameras():
        if not seen[role]:
            continue
        board2cam = _mean_pose([s["board2cam"] for s in seen[role]])
        candidates = [board2world @ np.linalg.inv(v) for v in flip_variants(board2cam, square, pattern)]
        rough = [np.asarray(c["cam2world_cv"], float) for c in [(calibration.get("cameras") or {}).get(role)] +
                 [r for r in calibration.get("rejected_cameras", []) if r.get("serial") == cell.config.cameras[role].serial]
                 if c]
        coloured = anchored_by_colour and all(s["by_colour"] for s in seen[role])
        if rough:
            candidates.sort(key=lambda c: min(_gap(c, r)[1] for r in rough))
        elif coloured:
            pass                                   # every view named the corners alike
        elif candidates[0][2, 3] < candidates[1][2, 3]:
            candidates.reverse()                   # no rough pose: the camera above the board
        placed[role] = candidates[0]
    report["placed"] = {role: np.round(pose[:3, 3], 3).tolist() for role, pose in placed.items()}
    if save and placed:
        for role, cam2world in placed.items():
            calibration.setdefault("cameras", {})[role] = {
                "cam2world_cv": cam2world.tolist(), "method": "checkerboard_wrist_anchor", "provisional": False,
                "square_mm": square_mm, "pattern": list(pattern), "anchors": report["anchors"],
                "median_residual_m": None, "reprojection_px": float(np.median([s["reprojection_px"] for s in seen[role]])),
                "serial": cell.config.cameras[role].serial, "record": str(record), "captured_unix_s": time.time()}
        cell.save_calibration(calibration)
    return report
