"""LIBERO-Pro's part of the scene trace: which simulator, robot and TCP the session hands to the MuJoCo reader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ora.environment import mujoco_scene_trace
from ora.environment.backends.libero_pro.scene_trace import LiberoSceneTrace
from ora.environment.scene_trace import Body


class Sim:
    def __init__(self) -> None:
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        self.data = SimpleNamespace()

    def step(self) -> None:
        pass


def test_a_round_reads_the_session_s_simulator_panda_and_grip_site(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def bodies(_model: Any, _data: Any, *, robot_roots: Any) -> list[Body]:
        seen["roots"] = robot_roots
        return [Body("cup", "cup", "object", lambda: [0, 0, 0, 0, 0, 0, 1], None)]

    def links(_model: Any, sites: Any) -> dict[str, Any]:
        seen["sites"] = sites
        return {}

    monkeypatch.setattr(mujoco_scene_trace, "scene_bodies", bodies)
    monkeypatch.setattr(mujoco_scene_trace, "tcp_links", links)
    panda = SimpleNamespace(robot_model=SimpleNamespace(root_body="robot0_base"),
                            gripper=SimpleNamespace(important_sites={"grip_site": "gripper0_grip_site"}))
    sim = Sim()
    session = SimpleNamespace(_worker_attr=lambda name: {"sim": sim, "robots": [panda]}[name])
    trace = LiberoSceneTrace(tmp_path / "t", run_id="r")
    trace.begin_round(session, 0)
    sim.step()
    manifest = trace.finish()

    assert seen == {"roots": ["robot0_base"], "sites": {"right": "gripper0_grip_site"}}
    assert manifest["backend"] == "libero_pro" and manifest["clock"]["unit"] == "mujoco_step_since_round_reset"
    assert manifest["clock"]["stride"] == 10
