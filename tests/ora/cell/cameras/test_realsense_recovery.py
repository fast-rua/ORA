"""A RealSense camera that stops streaming is recovered instead of ending the session (cameras/realsense.py)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ora.cell.cameras import realsense
from ora.cell.errors import CellError


class _Fake:
    """pyrealsense2 as the camera uses it; ``waits`` scripts each wait_for_frames: True delivers, False times out."""

    def __init__(self, waits: list[bool]) -> None:
        self.waits, self.resets, self.starts = list(waits), 0, 0
        fake = self
        frame = types.SimpleNamespace(get_data=lambda: np.ones((480, 848), np.uint16))
        color = types.SimpleNamespace(get_data=lambda: np.zeros((480, 848, 3), np.uint8))
        frameset = types.SimpleNamespace(get_color_frame=lambda: color, get_depth_frame=lambda: frame)
        intrinsics = types.SimpleNamespace(fx=1.0, fy=1.0, ppx=0.0, ppy=0.0, model="none", coeffs=[0.0] * 5)
        stream = types.SimpleNamespace(as_video_stream_profile=lambda: types.SimpleNamespace(get_intrinsics=lambda: intrinsics))
        device = types.SimpleNamespace(get_info=lambda key: "SERIAL" if key == "serial" else "D405",
                                       first_depth_sensor=lambda: types.SimpleNamespace(get_depth_scale=lambda: 0.001),
                                       hardware_reset=lambda: setattr(fake, "resets", fake.resets + 1))
        profile = types.SimpleNamespace(get_device=lambda: device, get_stream=lambda kind: stream)

        class Pipeline:
            def start(self, config):
                fake.starts += 1
                return profile

            def stop(self):
                pass

            def wait_for_frames(self, timeout_ms):
                if fake.waits and not fake.waits.pop(0):
                    raise RuntimeError(f"Frame didn't arrive within {timeout_ms}")
                return frameset

        self.module = types.SimpleNamespace(
            config=lambda: types.SimpleNamespace(enable_device=lambda serial: None, enable_stream=lambda *a: None),
            pipeline=Pipeline, align=lambda kind: types.SimpleNamespace(process=lambda frames: frames),
            stream=types.SimpleNamespace(color="color", depth="depth"),
            format=types.SimpleNamespace(rgb8="rgb8", z16="z16"),
            camera_info=types.SimpleNamespace(serial_number="serial", name="name"),
            context=lambda: types.SimpleNamespace(query_devices=lambda: [device]))


def _camera(monkeypatch, waits: list[bool]) -> tuple[realsense.RealSenseCamera, _Fake]:
    fake = _Fake(waits)
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake.module)
    monkeypatch.setattr(realsense.time, "sleep", lambda seconds: None)
    return realsense.RealSenseCamera(role="wrist_right", serial="SERIAL", mount="end_effector"), fake


WARM = [True] * (1 + realsense.WARMUP_FRAMES)


def test_a_stream_that_stalls_mid_session_is_restarted_and_the_frame_taken_again(monkeypatch):
    camera, fake = _camera(monkeypatch, [*WARM, False])          # the capture's first wait times out
    data = camera.capture()
    assert data["depth_pipeline_restarted"] is True and fake.resets == 0 and fake.starts == 2
    assert camera.capture()["depth_pipeline_restarted"] is False


def test_a_restart_that_streams_nothing_falls_back_to_a_hardware_reset(monkeypatch):
    # The capture times out, the restart's first frame times out too; a hardware reset and a start recover it.
    camera, fake = _camera(monkeypatch, [*WARM, False, False])
    assert camera.capture()["depth_pipeline_restarted"] is True
    assert fake.resets == 1 and fake.starts == 3


def test_a_camera_that_does_not_stream_at_the_start_gets_a_second_reset(monkeypatch):
    # The first start and the first reset's start stream nothing; the second reset's does.
    camera, fake = _camera(monkeypatch, [False, False, *WARM])
    assert fake.resets == 2 and camera.restarts == 1


def test_a_camera_that_never_streams_is_still_refused(monkeypatch):
    with pytest.raises(CellError, match="wrist_right .* streams no frames"):
        _camera(monkeypatch, [False] * 10)
