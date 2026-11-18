"""The YAM station's original camera pixels for the world view (capture.py through session.py).

The simulated station renders its cameras at every boundary and during a motion; a real rig reads nothing
extra while the arms can move: its frames are the pixels its observations read, a reset's or an action's
end written with the next observation's. Each frame passes the world view's own checks. No MuJoCo, no
hardware: a stand-in rig and a stand-in motion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ora.environment.backends.yam import rig as yam_rig
from ora.environment.backends.yam import session as yam_session
from ora.environment.backends.yam.capture import CAMERAS
from ora.environment.backends.yam.session import YamSession
from ora.orchestrator.world_view import _Observer
from tests.support.cell.fake import FakeKinematics

CAPTURE = {"schema": "ora.execution-capture-config.v1", "enabled": True, "fps": 10, "queue_capacity": 32}


class StandInRig:
    """The rig as the session uses it; ``mode`` sim also renders colour alone, as SimRig does."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.kinematics = FakeKinematics()
        self.table_top_z_m = 0.0
        self.reads = 0
        self.renders = 0
        self.shade = 0
        self.broken = False
        if mode == "sim":
            self.camera_rgb = self._render

    def arm_q(self, arm: str) -> np.ndarray:
        return np.zeros(6)

    def finger_m(self, arm: str) -> float:
        return 0.04

    def _image(self) -> np.ndarray:
        if self.broken:
            raise RuntimeError("the camera did not answer")
        return np.full((6, 8, 3), self.shade, dtype=np.uint8)

    def _render(self) -> dict[str, np.ndarray]:
        self.renders += 1
        return {name: self._image() for name in CAMERAS}

    def cameras(self) -> dict[str, dict]:
        self.reads += 1
        return {name: {"rgb": self._image(), "depth": np.ones((6, 8), np.float32), "intrinsic_cv": np.eye(3).tolist(),
                       "mount": "end_effector" if name.startswith("wrist") else "fixed_world",
                       "source_camera": f"stand-in:{name}", "cam2world_cv": None, "calibration_source": None}
                for name in CAMERAS}

    def home(self) -> str:
        return "at rest"

    def close(self) -> None:
        pass


def motion(ticks: int):
    """A motion of ``ticks`` control ticks, as motion.py runs one (each tick told to the session)."""

    def execute(session, action):
        for _ in range(ticks):
            session.control_ticks += 1
            session.on_control_tick()
        return {"action_id": action["action_id"], "action": action["action"], "execution_status": "reached",
                "steps_used": ticks}

    return execute


@pytest.fixture
def episode(tmp_path, monkeypatch):
    monkeypatch.setenv("ORA_CELL_APPROVALS", str(tmp_path / "approvals"))
    monkeypatch.setenv("ORA_CELL_LOCK", str(tmp_path / "ora-cell.lock"))

    def start(mode: str, *, capture=CAPTURE, ticks: int = 20) -> tuple[YamSession, StandInRig, Path]:
        rig = StandInRig(mode)
        monkeypatch.setattr(yam_rig, "build_rig", lambda config: rig)
        monkeypatch.setattr(yam_session, "execute_motion_v3", motion(ticks))
        run = tmp_path / f"run-{mode}"
        session = YamSession(config={"task_instruction": "play", "rig_mode": mode, "execution_capture": capture},
                             output_root=run / "native")
        session.start()
        return session, rig, run

    return start


def act(session: YamSession, index: int) -> dict:
    return session.execute({"schema": "ora-motion-action.v3", "observation_id": "o", "action_id": f"a{index}",
                            "bound_observation_sha256": "0" * 64, "action": "set_gripper",
                            "arguments": {"targets": {"left": "close"}}})


def written(session: YamSession) -> None:
    """The recorder's writer has written every frame taken so far (it writes on its own thread)."""

    session.capture.recorder._queue.join()


def timeline(run: Path) -> list[dict]:
    path = run / "native" / "execution-capture" / "timeline.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def manifest(run: Path) -> dict:
    return json.loads((run / "native" / "execution-capture" / "manifest.json").read_text())


def world_view_accepts(run: Path) -> _Observer:
    observer = _Observer(run)
    observer.manifest = manifest(run)
    for record in timeline(run):
        observer._frame(record)
    return observer


def phases(run: Path) -> list[tuple]:
    return [(record["round_index"], record["phase"], record["action_id"], record["sim_step"]) for record in timeline(run)]


def test_the_simulated_station_renders_every_boundary_and_its_motions(episode, tmp_path):
    session, rig, run = episode("sim", ticks=20)
    act(session, 0)
    session.observe(tmp_path / "observations")
    session.restart_round(1)
    session.persist_official_verdict()
    session.finalize_operator_video()
    session.close()
    assert phases(run) == [
        (0, "reset", None, 0), (0, "action_start", "a0", 0),
        # 10 frames a second of control time at 50 Hz: one every 5 ticks after the action's start.
        (0, "control", "a0", 5), (0, "control", "a0", 10), (0, "control", "a0", 15), (0, "control", "a0", 20),
        (0, "action_end", "a0", 20), (0, "terminal", None, 20),
        (1, "reset", None, 0), (1, "terminal", None, 0), (1, "close", None, 0)]
    assert rig.reads == 1, "the observation reads the cameras once; the frames are rendered"
    assert manifest(run)["status"] == "complete" and manifest(run)["camera_panels"] == list(CAMERAS)
    assert manifest(run)["identity"] == {"run_id": run.name, "task": "operator_task", "rig_mode": "sim"}
    assert list(world_view_accepts(run).ends) == ["a0"]


@pytest.mark.parametrize("mode", ["read", "on"])
def test_a_real_rig_reads_nothing_more_while_the_arms_can_move(episode, tmp_path, mode):
    session, rig, run = episode(mode)
    written(session)
    assert rig.reads == 0 and timeline(run) == [], "the reset waits for the first observation's pixels"
    session.observe(tmp_path / "observations")
    act(session, 0)
    written(session)
    assert rig.reads == 1 and len(timeline(run)) == 1, "no camera is read for the action"
    rig.shade = 200
    session.observe(tmp_path / "observations")
    session.observe(tmp_path / "observations")
    assert rig.reads == 3
    session.persist_official_verdict()
    session.finalize_operator_video()
    session.close()
    assert phases(run) == [(0, "reset", None, 0), (0, "action_end", "a0", 20), (0, "observation", None, 20),
                           (0, "terminal", None, 20)]
    end = timeline(run)[1]
    seen = np.full((6, 8, 3), 200, dtype=np.uint8)
    assert end["images"]["front"]["pixel_sha256"] == hashlib.sha256(seen.tobytes()).hexdigest(), \
        "an action's end is the next observation's own pixels"
    assert rig.reads == 4, "only the end of the round reads the cameras once more"
    assert manifest(run)["status"] == "complete"
    assert list(world_view_accepts(run).ends) == ["a0"]


def test_an_action_without_an_observation_before_it_drops_the_reset_and_keeps_its_end(episode, tmp_path):
    session, rig, run = episode("on")
    act(session, 0)
    session.observe(tmp_path / "observations")
    session.finalize_operator_video()
    assert [record["phase"] for record in timeline(run)] == ["action_end", "terminal"]
    assert list(world_view_accepts(run).ends) == ["a0"]


@pytest.mark.parametrize("mode,closing", [("sim", ["terminal", "close"]), ("on", ["terminal"])])
def test_a_session_quit_without_a_verdict_still_ends_its_round(episode, tmp_path, mode, closing):
    """An operator's /quit persists no official verdict: closing records the round's end itself."""

    session, rig, run = episode(mode)
    session.observe(tmp_path / "observations")
    session.finalize_operator_video()
    session.close()
    assert [record["phase"] for record in timeline(run)] == ["reset", *closing]
    assert manifest(run)["status"] == "complete"


def test_a_frame_that_cannot_be_taken_is_noted_and_the_episode_goes_on(episode, tmp_path):
    session, rig, run = episode("sim")
    rig.broken = True
    receipt = act(session, 0)
    assert receipt["execution_status"] == "reached"
    failure = json.loads((run / "native" / "execution-capture-failure.json").read_text())
    assert failure["component"] == "original_execution_capture" and "did not answer" in failure["error"]
    rig.broken = False
    act(session, 1)
    session.finalize_operator_video()
    assert manifest(run)["error"] is None, "the recorder keeps recording"
    assert ("action_end" in [record["phase"] for record in timeline(run)])


def test_a_frame_without_every_camera_is_refused_before_it_stops_the_recorder(episode, tmp_path):
    session, rig, run = episode("on")
    session.capture.observation({"front": np.zeros((6, 8, 3), np.uint8)})
    assert "no overhead, wrist_left, wrist_right image" in json.loads(
        (run / "native" / "execution-capture-failure.json").read_text())["error"]
    session.observe(tmp_path / "observations")
    session.finalize_operator_video()
    assert manifest(run)["error"] is None
    assert [record["phase"] for record in timeline(run)] == ["observation", "terminal"]


def test_without_execution_capture_nothing_is_recorded(episode, tmp_path):
    session, rig, run = episode("sim", capture={**CAPTURE, "enabled": False})
    act(session, 0)
    session.close()
    assert session.capture is None and not (run / "native" / "execution-capture").exists()
