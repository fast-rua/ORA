"""The cell's RealSense cameras: aligned metric RGB-D with a depth validity check.

Depth is aligned to the colour stream, so one intrinsic matrix serves both. A frame whose
valid depth falls below the mount's floor is the classic stalled-emitter or USB-reset
failure: the pipeline is restarted once and the frame taken again, and the fraction is
recorded either way so a policy never reads a hole as free space unknowingly.

A camera that stops delivering frames is recovered rather than ending the session: at the
start, by a hardware reset and up to ``RESET_ATTEMPTS`` starts with a growing settle; in a
capture, by restarting its pipeline, then by a hardware reset and a start, before the frame is
taken again (``depth_pipeline_restarted`` true). On the YAM cell the right wrist D405, the third
camera on one 5 Gbps hub, stalled this way at the start of most sessions and once mid-session
(2026-11-14).
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

from ..errors import CellError

IMAGE_SIZE = (848, 480)
FPS = 30
WARMUP_FRAMES = 30
FRAME_TIMEOUT_MS = 2000
# The first frame after a start can take seconds (auto-exposure, USB re-enumeration).
FIRST_FRAME_TIMEOUT_MS = 10000
# A camera that does not stream at the start: hardware resets and starts, each settling longer.
RESET_ATTEMPTS = 2
RESET_SETTLE_S = 2.0
# Least valid-depth fraction per mount. A wrist D405 sees well from 7 cm to about 50 cm,
# so far views legitimately leave much of its frame empty.
MIN_VALID_DEPTH_FRACTION = {"fixed_world": 0.20, "end_effector": 0.05}


class RealSenseCamera:
    def __init__(self, *, role: str, serial: str, mount: str) -> None:
        if mount not in MIN_VALID_DEPTH_FRACTION:
            raise CellError(f"unknown camera mount {mount!r}")
        self.role, self.serial, self.mount = role, serial, mount
        self.pipeline = None
        self.restarts = 0
        self.recovered = False
        try:
            self._start()
        except CellError:
            for attempt in range(1, RESET_ATTEMPTS + 1):
                self._reset_device()
                time.sleep(RESET_SETTLE_S * attempt)
                try:
                    self._start()
                except CellError:
                    if attempt == RESET_ATTEMPTS:
                        raise
                    continue
                self.restarts += 1
                break

    def _reset_device(self) -> None:
        import pyrealsense2 as rs

        for device in rs.context().query_devices():
            if device.get_info(rs.camera_info.serial_number) == self.serial:
                device.hardware_reset()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            time.sleep(1.0)
            try:
                if any(d.get_info(rs.camera_info.serial_number) == self.serial for d in rs.context().query_devices()):
                    time.sleep(1.0)
                    return
            except RuntimeError:
                continue        # a device still re-enumerating answers queries with errors
        raise CellError(f"RealSense {self.role} ({self.serial}) did not come back after a hardware reset")

    def _start(self) -> None:
        import pyrealsense2 as rs

        width, height = IMAGE_SIZE
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, FPS)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, FPS)
        self.pipeline = rs.pipeline()
        try:
            profile = self.pipeline.start(config)
        except RuntimeError as exc:
            raise CellError(f"RealSense {self.role} ({self.serial}) does not start: {exc}") from exc
        device = profile.get_device()
        self.model = device.get_info(rs.camera_info.name)
        self.depth_scale = float(device.first_depth_sensor().get_depth_scale())
        self.align = rs.align(rs.stream.color)
        color = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsic = [[color.fx, 0.0, color.ppx], [0.0, color.fy, color.ppy], [0.0, 0.0, 1.0]]
        self.distortion = {"model": str(color.model), "coeffs": list(color.coeffs)}
        try:
            self.pipeline.wait_for_frames(FIRST_FRAME_TIMEOUT_MS)
            for _ in range(WARMUP_FRAMES):
                self.pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
        except RuntimeError as exc:
            self.pipeline.stop()
            raise CellError(f"RealSense {self.role} ({self.serial}) streams no frames: {exc}") from exc

    def _wait(self) -> Any:
        """The next frameset; a stalled stream is recovered (restart, else hardware reset) and waited for once more."""

        try:
            return self.pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
        except RuntimeError:
            pass
        try:
            self.restart()
        except CellError:
            self._stop()
            self._reset_device()
            time.sleep(RESET_SETTLE_S)
            self._start()
            self.restarts += 1
        self.recovered = True
        try:
            return self.pipeline.wait_for_frames(FIRST_FRAME_TIMEOUT_MS)
        except RuntimeError as exc:
            raise CellError(f"RealSense {self.role} ({self.serial}) streams no frames after a restart: {exc}") from exc

    def _stop(self) -> None:
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except RuntimeError:
            pass

    def _frame(self) -> tuple[np.ndarray, np.ndarray]:
        frames = self.align.process(self._wait())
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise CellError(f"RealSense {self.role} returned an incomplete frame")
        rgb = np.asanyarray(color.get_data()).copy()
        metres = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        return rgb, metres

    def restart(self) -> None:
        self._stop()
        time.sleep(0.5)
        self._start()
        self.restarts += 1

    def capture(self) -> dict[str, Any]:
        self.recovered = False
        rgb, depth = self._frame()
        valid = float(np.count_nonzero(depth > 0) / depth.size)
        restarted = self.recovered
        if valid < MIN_VALID_DEPTH_FRACTION[self.mount]:
            self.restart()
            rgb, depth = self._frame()
            valid, restarted = float(np.count_nonzero(depth > 0) / depth.size), True
        return {"rgb": rgb, "depth": depth, "intrinsic_cv": self.intrinsic, "mount": self.mount,
                "source_camera": f"realsense:{self.model}:{self.serial}",
                "depth_valid_fraction": valid, "depth_pipeline_restarted": restarted,
                "depth_below_floor": valid < MIN_VALID_DEPTH_FRACTION[self.mount],
                "distortion": self.distortion}

    def close(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None


class RealSense:
    """The cell's cameras as RealSense devices (``adapter.CameraDriver``), the kit's default."""

    def open(self, role: str, serial: str, mount: str) -> RealSenseCamera:
        return RealSenseCamera(role=role, serial=serial, mount=mount)

    def connected(self) -> set[str]:
        """Serials of the RealSense devices connected now; refuses without pyrealsense2."""

        try:
            import pyrealsense2 as rs
        except ImportError:
            raise CellError("no pyrealsense2 in this interpreter: the cameras cannot be listed") from None
        return {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}

    def busy(self, serials: Iterable[str]) -> dict[str, str]:
        from ..occupancy import realsense_busy

        return realsense_busy(serials)

    def reset(self, serials: Iterable[str]) -> str:
        """A hardware reset of each named camera; says which are not back yet."""

        import pyrealsense2 as rs

        wanted = set(serials)
        found = [d for d in rs.context().query_devices() if d.get_info(rs.camera_info.serial_number) in wanted]
        for device in found:
            device.hardware_reset()
        time.sleep(5.0)
        back = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
        missing = sorted(wanted - back)
        return f"reset {len(found)} cameras" + (f"; not back yet: {missing}" if missing else "; all back")


REALSENSE = RealSense()
