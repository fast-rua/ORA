"""Live view of the real cell for the cell board: streams, a fused point cloud, the arms.

While live, the board holds the cell lock, keeps the four RealSense cameras open and
reads both arms passively (DISABLE frames) at ``JOINT_HZ``. An ORA episode cannot start
meanwhile, so live stops by itself when no page has asked for anything for ``IDLE_S``.
A frame on a bus that is not a motor's reply to this reader stops the joint reading at
once: another program drives the arms, and they must not be sent another DISABLE.

The point cloud places each camera's depth in ``yam_station_world``: wrist cameras by the
encoders through the CAD chain, fixed cameras by the cell calibration. A fixed camera
without calibration streams its images but stays out of the cloud.
"""

from __future__ import annotations

import struct
import threading
import time
from typing import Any

import numpy as np

from ..depth.client import Completer, service_info
from ..errors import CellError
from ..geometry import arm_surface_points, voxel_downsample
from ..lock import CellLock
from .context import Cell

JOINT_HZ = 10.0
IDLE_S = 30.0
MODEL_VOXEL_M = 0.008
JPEG_QUALITY = 80
DEPTH_RANGE_M = {"fixed_world": (0.25, 2.5), "end_effector": (0.07, 0.6)}
# A pixel whose depth differs from a neighbour's by more than this share of its own depth
# sits on a depth edge, where sensors and networks both return values between the two
# surfaces ("flying pixels"); in 3D they streak along the camera's rays.
FLYING_PIXEL_RATIO = 0.03


def drop_flying_pixels(depth: np.ndarray, ratio: float = FLYING_PIXEL_RATIO) -> np.ndarray:
    jump = np.zeros(depth.shape, bool)
    for a, b in ((depth[:, 1:], depth[:, :-1]), (depth[1:, :], depth[:-1, :])):
        both = (a > 0) & (b > 0)
        edge = both & (np.abs(a - b) > ratio * np.minimum(a, b))
        if a.shape[1] != depth.shape[1]:
            jump[:, 1:] |= edge
            jump[:, :-1] |= edge
        else:
            jump[1:, :] |= edge
            jump[:-1, :] |= edge
    return np.where(jump, 0.0, depth)


class _Camera(threading.Thread):
    """One camera, captured continuously; the latest frame wins."""

    def __init__(self, cell: Cell, role: str, stop: threading.Event) -> None:
        super().__init__(name=f"live-{role}", daemon=True)
        self.cell, self.role, self.stop = cell, role, stop
        self.mount = cell.config.cameras[role].mount
        self.frame: dict[str, Any] | None = None
        self.sequence = 0
        self.error: str | None = None
        self.times: list[float] = []
        self.lock = threading.Lock()

    def run(self) -> None:
        camera = None
        try:
            camera = self.cell.open_camera(self.role)
            while not self.stop.is_set():
                data = camera.capture()
                now = time.time()
                with self.lock:
                    self.frame, self.sequence = {**data, "at": now}, self.sequence + 1
                    self.times = [t for t in self.times if now - t < 2.0] + [now]
        except Exception as exc:  # noqa: BLE001 - the page shows why a camera stopped
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if camera is not None:
                camera.close()

    def latest(self) -> tuple[int, dict[str, Any] | None]:
        with self.lock:
            return self.sequence, self.frame

    def fps(self) -> float:
        with self.lock:
            return (len(self.times) - 1) / (self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0


class _Arms(threading.Thread):
    """Both arms read passively; forward kinematics of what was read."""

    def __init__(self, cell: Cell, stop: threading.Event) -> None:
        super().__init__(name="live-arms", daemon=True)
        self.cell, self.stop = cell, stop
        self.state: dict[str, Any] | None = None
        self.model_points: tuple[int, bytes] = (0, b"")
        self.error: str | None = None
        self.lock = threading.Lock()

    def run(self) -> None:
        reader = None
        cell, arms = self.cell, self.cell.adapter.arms
        try:
            kinematics = cell.kinematics()
            rest = cell.calibration().get("rest_reference")
            rest_chain = None
            if rest:
                kinematics.set_state({arm: rest[arm] for arm in arms})
                rest_chain = {arm: kinematics.chain(arm) for arm in arms}
            reader = cell.adapter.open_reader(cell.gripper_limits())
            # Our own sockets' entries in the kernel's receive lists; any more is someone else.
            own = {channel: cell.bus.receivers(channel) for channel in cell.channels}
            version, last_q = 0, None
            while not self.stop.is_set():
                started = time.monotonic()
                for channel in cell.channels:
                    if cell.bus.receivers(channel) > own[channel]:
                        raise CellError(f"another program opened {channel}: stopped reading the arms")
                readings = reader.read()
                q = {arm: r.q for arm, r in readings.items()}
                open_m = cell.adapter.finger_open_m
                finger = {arm: (r.gripper_opening if r.gripper_opening is not None else 1.0) * open_m
                          for arm, r in readings.items()}
                kinematics.set_state(q, finger)
                state = {"at": time.time(), "arms": {arm: {
                    "q": r.q.tolist(), "faults": r.faults, "status": r.status,
                    "max_temperature_c": r.max_temperature_c, "torque_nm": r.torque_nm,
                    "gripper_raw_rad": r.gripper_raw, "gripper_opening": r.gripper_opening,
                    "tcp": kinematics.tcp(arm).tolist(), "wrist_camera": kinematics.wrist_camera(arm).tolist(),
                    "chain": kinematics.chain(arm), "rest_chain": None if rest_chain is None else rest_chain[arm],
                } for arm, r in readings.items()}}
                moved = last_q is None or any(np.max(np.abs(q[a] - last_q[a])) > 0.002 for a in arms)
                if moved:
                    points, _ = arm_surface_points(kinematics, per_face=1)
                    points = points[voxel_downsample(points, MODEL_VOXEL_M)]
                    version, last_q = version + 1, {a: v.copy() for a, v in q.items()}
                    packed = np.round(points * 1000).astype("<i2").tobytes()
                with self.lock:
                    self.state = state
                    if moved:
                        self.model_points = (version, packed)
                time.sleep(max(0.0, 1.0 / JOINT_HZ - (time.monotonic() - started)))
        except Exception as exc:  # noqa: BLE001 - the page shows why reading stopped
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if reader is not None:
                reader.close()

    def latest(self) -> dict[str, Any] | None:
        with self.lock:
            return self.state


class LiveCell:
    def __init__(self, cell: Cell, depth_service: str | None = None) -> None:
        self.cell = cell
        self.depth_service = depth_service
        self.completers: dict[str, Completer] = {}
        self.completion_stop: threading.Event | None = None
        self.completion_info: dict[str, Any] | None = None
        self.guard = threading.Lock()
        self.stop_event: threading.Event | None = None
        self.cell_lock: CellLock | None = None
        self.cameras: dict[str, _Camera] = {}
        self.arms: _Arms | None = None
        self.started_at: float | None = None
        self.touched = 0.0
        self.calibration: dict[str, Any] = {}
        self._cloud_cache: tuple[tuple, bytes] | None = None
        self._jpeg_cache: dict[tuple[str, str], tuple[int, bytes]] = {}
        threading.Thread(target=self._idle_watch, name="live-idle", daemon=True).start()

    # ------------------------------------------------------------ lifecycle
    @property
    def running(self) -> bool:
        return self.stop_event is not None

    def touch(self) -> None:
        self.touched = time.monotonic()

    def start(self) -> dict[str, Any]:
        with self.guard:
            if self.running:
                return self.summary()
            self.calibration = self.cell.calibration()
            lock = CellLock("cell board (live view)").acquire()
            try:
                for channel in self.cell.channels:
                    if self.cell.bus.receivers(channel) or self.cell.bus.traffic(channel, 0.3):
                        raise CellError(f"{channel} is in use by another program")
            except BaseException:
                lock.release()
                raise
            self.cell_lock, self.stop_event = lock, threading.Event()
            self.cameras = {role: _Camera(self.cell, role, self.stop_event) for role in self.cell.config.cameras}
            self.arms = _Arms(self.cell, self.stop_event)
            for worker in (*self.cameras.values(), self.arms):
                worker.start()
            self.started_at = time.time()
            self.touch()
            return self.summary()

    def stop(self) -> None:
        with self.guard:
            if not self.running:
                return
            self.set_completion(False)
            self.stop_event.set()
            for worker in (*self.cameras.values(), self.arms):
                worker.join(timeout=10.0)
            self.cell_lock.release()
            self.stop_event, self.cell_lock, self.arms, self.cameras = None, None, None, {}
            self._cloud_cache, self._jpeg_cache = None, {}

    def _idle_watch(self) -> None:
        while True:
            time.sleep(2.0)
            if self.running and time.monotonic() - self.touched > IDLE_S:
                self.stop()

    # ------------------------------------------------------------ products
    def _pose(self, role: str, arms: dict[str, Any] | None) -> np.ndarray | None:
        if role in self.cell.adapter.wrist_cameras:
            arm = self.cell.adapter.wrist_cameras[role]
            return None if arms is None else np.asarray(arms["arms"][arm]["wrist_camera"])
        fixed = (self.calibration.get("cameras") or {}).get(role)
        return None if fixed is None else np.asarray(fixed["cam2world_cv"], float)

    def set_completion(self, on: bool) -> dict[str, Any]:
        """Send every camera's frames through the depth service (on) or show raw depth (off)."""

        if not on:
            if self.completion_stop is not None:
                self.completion_stop.set()
            self.completers, self.completion_stop = {}, None
            return {"completion": False}
        if not self.depth_service:
            raise CellError("no depth service configured: start the board with --depth-service URL")
        if not self.running:
            raise CellError("start live view first")
        self.completion_info = service_info(self.depth_service)
        if self.completion_stop is None:
            self.completion_stop = threading.Event()
            self.completers = {role: Completer(self.depth_service, camera.latest, self.completion_stop)
                               for role, camera in self.cameras.items()}
            for completer in self.completers.values():
                completer.start()
        return {"completion": True, "info": self.completion_info}

    def _depth(self, role: str, frame: dict[str, Any]) -> tuple[np.ndarray, Any]:
        """The depth the cloud uses: fused with completion when it is on and fresh, else raw."""

        completer = self.completers.get(role)
        completed = completer.completed() if completer else None
        if completed is not None:
            return completed[1], ("completed", completed[0])
        return frame["depth"], ("raw", None)

    def summary(self) -> dict[str, Any]:
        arms = self.arms.latest() if self.arms else None
        cameras = {}
        for role, camera in self.cameras.items():
            _, frame = camera.latest()
            pose = self._pose(role, arms)
            cameras[role] = {"fps": camera.fps(), "error": camera.error, "mount": camera.mount,
                             "in_cloud": pose is not None, "cam2world": None if pose is None else pose.tolist(),
                             "intrinsic": None if frame is None else frame["intrinsic_cv"],
                             "size": None if frame is None else list(frame["depth"].shape[::-1]),
                             "depth_valid_fraction": None if frame is None else frame["depth_valid_fraction"],
                             "source": None if frame is None else frame["source_camera"]}
        completion = {"available": bool(self.depth_service), "service": self.depth_service,
                      "on": bool(self.completers), "info": self.completion_info,
                      "cameras": {role: c.status() for role, c in self.completers.items()}}
        return {"running": self.running, "since": self.started_at, "cameras": cameras, "arms": arms,
                "completion": completion,
                "arms_error": self.arms.error if self.arms else None,
                "model_version": self.arms.model_points[0] if self.arms else 0, "idle_stop_s": IDLE_S}

    def jpeg(self, role: str, kind: str) -> tuple[int, bytes] | None:
        import cv2

        camera = self.cameras.get(role)
        if camera is None:
            return None
        sequence, frame = camera.latest()
        if frame is None:
            return None
        cached = self._jpeg_cache.get((role, kind))
        if cached and cached[0] == sequence:
            return cached
        if kind in {"depth", "completed"}:
            depth = frame["depth"]
            if kind == "completed":
                completed = self.completers.get(role).completed() if role in self.completers else None
                if completed is None:
                    return None
                sequence, depth = completed[0] + 10**9, completed[1]
            near, far = DEPTH_RANGE_M[camera.mount]
            scaled = np.clip(255 * (depth - near) / (far - near), 0, 255).astype(np.uint8)
            image = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
            image[depth <= 0] = 0
        else:
            image = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
        encoded = (sequence, cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes())
        self._jpeg_cache[(role, kind)] = encoded
        return encoded

    def cloud(self, stride: int) -> bytes:
        """Little-endian: uint32 n, int16[n,3] millimetres in world, uint8[n,3] RGB, uint8[n] camera index."""

        stride = int(min(max(stride, 2), 16))
        arms = self.arms.latest() if self.arms else None
        key = (stride, tuple(c.latest()[0] for c in self.cameras.values()), None if arms is None else arms["at"],
               tuple(self._depth(role, c.latest()[1])[1] if c.latest()[1] is not None else None
                     for role, c in self.cameras.items()))
        if self._cloud_cache and self._cloud_cache[0] == key:
            return self._cloud_cache[1]
        xyz, rgb, index = [], [], []
        for i, role in enumerate(self.cell.config.cameras):
            camera = self.cameras.get(role)
            pose = self._pose(role, arms)
            if camera is None or pose is None:
                continue
            _, frame = camera.latest()
            if frame is None:
                continue
            near, far = DEPTH_RANGE_M[camera.mount]
            depth, _ = self._depth(role, frame)
            depth = drop_flying_pixels(depth)
            k = np.asarray(frame["intrinsic_cv"], float)
            sampled = depth[::stride, ::stride]
            v, u = np.nonzero((sampled > near) & (sampled < far))
            z = sampled[v, u]
            u, v = u * stride, v * stride
            points = np.c_[(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z]
            world = points @ pose[:3, :3].T + pose[:3, 3]
            xyz.append(np.round(world * 1000).astype("<i2"))
            rgb.append(frame["rgb"][v, u].astype(np.uint8))
            index.append(np.full(len(z), i, np.uint8))
        if xyz:
            body = np.concatenate(xyz).tobytes() + np.concatenate(rgb).tobytes() + np.concatenate(index).tobytes()
            count = sum(len(a) for a in xyz)
        else:
            body, count = b"", 0
        packed = struct.pack("<I", count) + body
        self._cloud_cache = (key, packed)
        return packed

    def model(self) -> tuple[int, bytes]:
        return self.arms.model_points if self.arms else (0, b"")
