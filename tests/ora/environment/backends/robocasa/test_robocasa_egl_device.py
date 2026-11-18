"""The RoboCasa365 reset records the EGL device its kitchen renders on, on a stand-in kitchen."""

from __future__ import annotations

import json
from typing import Any

from ora.environment.backends.robocasa import construction
from ora.environment.backends.robocasa.robot import OmronIO
from ora.environment.backends.robocasa.session import RoboCasaSession
from ora.environment.graphics_probe import SOFTWARE_EGL_DEVICE_SCHEMA


class Kitchen:
    horizon = 750

    def reset(self) -> None:
        pass

    def get_ep_meta(self) -> dict[str, Any]:
        return {"lang": "Open the left drawer."}

    def _check_success(self) -> bool:
        return False


def test_the_reset_records_the_egl_device_the_kitchen_renders_on(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(OmronIO, "__post_init__", lambda _self: None)
    for name, value in (("grip_pose", [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]), ("arm_qpos", [0.0] * 7),
                        ("base_pose", [0.0, 0.0, 0.0]), ("public_gripper_geometry", {}), ("finger_qpos", [0.0, 0.0])):
        monkeypatch.setattr(OmronIO, name, lambda _self, value=value: value)
    monkeypatch.setattr("ora.environment.backends.robocasa.session.shared_motion.start", lambda _session: None)
    device = {"schema": SOFTWARE_EGL_DEVICE_SCHEMA, "index": 8, "vendor": "Mesa Project",
              "driver": "llvmpipe (LLVM 15.0.7, 256 bits)"}
    monkeypatch.setattr(construction, "_EGL_DEVICE", device)
    session = RoboCasaSession(output_root=tmp_path, config={
        "task": "OpenDrawer", "seed": 3, "environment_factory": lambda *, task, seed: Kitchen()})

    reset = session.start()

    assert reset["egl_device"] == device
    assert json.loads((tmp_path / "round-resets" / "round-0000.json").read_text())["egl_device"] == device
