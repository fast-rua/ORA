"""The operator view: an environment's picture for the human operator, never for the policy.

An environment session may answer ``operator_view()`` with a PNG, its camera
and its size (``ora.environment.base.validate_operator_view``). The described
session forwards it, the local worker transport passes it as a file under the
run (never inside a JSON line), Synthetic draws one, RoboTwin reads its
observer camera, and a free-play Supervisor stores one per published
observation at ``native/operator-view/<step>.png`` with an
``operator_view_frame`` event that the play evaluation checks.

The Supervisor also tells the operator's terminal and page (display only)
when a message is queued into the running instruction and when it is
delivered, and when the robot starts and finishes an action.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ora.environment.base import EnvironmentContractError, png_size, validate_operator_view


def png(width: int = 6, height: int = 4) -> bytes:
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def view(**change: Any) -> dict[str, Any]:
    return {"png": png(), "camera": "observer_camera", "width": 6, "height": 4, **change}


def test_an_operator_view_is_a_png_its_camera_and_its_size() -> None:
    assert validate_operator_view(None) is None
    assert validate_operator_view(view()) == view()
    assert png_size(png(7, 5)) == (7, 5)
    for bad in (view(width=7), view(png=b"not a png"), view(camera="a camera"), view(camera=""),
                {**view(), "pose": [0, 0, 0]}, {"png": png()}, view(height=True), view(png="text")):
        with pytest.raises(EnvironmentContractError):
            validate_operator_view(bad)


def test_the_described_session_forwards_the_view_or_answers_none() -> None:
    from ora.environment.session import DescribedEnvironmentSession

    described = object.__new__(DescribedEnvironmentSession)
    described.delegate = SimpleNamespace(operator_view=lambda: view())
    assert described.operator_view() == view()
    described.delegate = SimpleNamespace(operator_view=lambda: view(width=5))
    with pytest.raises(EnvironmentContractError):
        described.operator_view()
    described.delegate = SimpleNamespace(observe=None)
    assert described.operator_view() is None


def test_the_worker_writes_the_view_under_the_run_and_the_client_reads_it_back(tmp_path: Path) -> None:
    from ora.environment.rpc import OPERATOR_VIEW_TRANSFER, LocalRpcEnvironmentSession
    from ora.environment.worker import WorkerProtocolError, _dispatch

    native = tmp_path / "run" / "native"
    native.mkdir(parents=True)
    config = {"shared_root": str(tmp_path / "run")}
    destination = native / OPERATOR_VIEW_TRANSFER
    session = SimpleNamespace(operator_view=lambda: view())
    result, close = _dispatch(session, config, "operator_view", {"destination": str(destination)})
    assert close is False and result == {"path": str(destination.resolve()), "camera": "observer_camera",
                                         "width": 6, "height": 4}
    assert destination.read_bytes() == png()
    assert _dispatch(SimpleNamespace(), config, "operator_view", {"destination": str(destination)}) == (None, False)
    with pytest.raises(WorkerProtocolError):
        _dispatch(session, config, "operator_view", {"destination": str(tmp_path / "elsewhere.png")})

    # The client asks with the transfer path, reads the PNG back and removes the file.
    client = object.__new__(LocalRpcEnvironmentSession)
    client.output_root = native

    def request(method: str, params: dict[str, Any]) -> Any:
        assert method == "operator_view"
        return _dispatch(session, config, method, params)[0]

    client._request = request  # type: ignore[method-assign]
    assert client.operator_view() == view()
    assert not destination.exists()
    client._request = lambda method, params: None  # type: ignore[method-assign]
    assert client.operator_view() is None


def test_synthetic_draws_its_front_view_only_while_it_draws_its_world(tmp_path: Path) -> None:
    from ora.environment.execution_capture import resolve_execution_capture
    from ora.environment.synthetic import SyntheticSession

    plain = SyntheticSession(output_root=tmp_path / "plain", task="t", seed=0)
    plain.start()
    assert plain.operator_view() is None
    drawn = SyntheticSession(output_root=tmp_path / "drawn", task="t", seed=0,
                             execution_capture=resolve_execution_capture())
    drawn.start()
    answer = validate_operator_view(drawn.operator_view())
    assert answer is not None and answer["camera"] == "operator_front_view"
    assert (answer["width"], answer["height"]) == (640, 400)
    drawn.close()


def test_robotwin_reads_its_observer_camera_and_nothing_else() -> None:
    import numpy as np

    from ora.environment.backends.robotwin.operator_capture import RoboTwinOperatorCaptureMixin

    class Session(RoboTwinOperatorCaptureMixin):
        environment: Any = None

    session = Session()
    assert session.operator_view() is None  # not started
    asked: list[str] = []

    def observer() -> Any:
        asked.append("observer")
        return np.full((48, 64, 3), 200, dtype=np.uint8)

    session.environment = SimpleNamespace(cameras=SimpleNamespace(get_observer_rgb=observer))
    answer = validate_operator_view(session.operator_view())
    assert answer is not None and answer["camera"] == "observer_camera" and (answer["width"], answer["height"]) == (64, 48)
    assert asked == ["observer"]
    session.environment = SimpleNamespace(cameras=SimpleNamespace())
    assert session.operator_view() is None


class _Loop:
    """The pieces of a PlayLoop that `observation_published` uses."""

    def __init__(self, root: Path, runtime: Any) -> None:
        from ora.orchestrator.episode.play import PlayLoop

        self.loop = object.__new__(PlayLoop)
        loop = self.loop
        loop.run_root, loop.native, loop.events = root, root / "native", root / "events.jsonl"
        loop.native.mkdir(parents=True)
        loop.runtime = runtime
        loop.operator_view, loop.operator_view_failures = None, 0
        loop.live: list[tuple[str, dict[str, Any]]] = []
        loop.failures: list[str] = []
        loop.publish_live = lambda kind, payload: loop.live.append((kind, dict(payload)))
        loop.record_auxiliary_failure = lambda stage, exc: loop.failures.append(f"{stage}: {exc}")

    def events(self) -> list[dict[str, Any]]:
        path = self.loop.events
        return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def test_the_supervisor_stores_one_view_per_observation_and_says_where(tmp_path: Path) -> None:
    from ora.orchestrator.evaluation.play import _operator_view_errors

    answers = iter([view(), None, view(png=png(8, 2), width=8, height=2)])
    harness = _Loop(tmp_path / "run", SimpleNamespace(operator_view=lambda: next(answers)))
    loop = harness.loop
    for step in (0, 1, 2):
        loop.observation_published(step)
    frames = harness.events()
    assert [(event["type"], event["observation_step"], event["path"]) for event in frames] == [
        ("operator_view_frame", 0, "native/operator-view/000000.png"),
        ("operator_view_frame", 2, "native/operator-view/000002.png")]
    assert frames[1]["camera"] == "observer_camera" and (frames[1]["width"], frames[1]["height"]) == (8, 2)
    assert frames[1]["sha256"] == hashlib.sha256((loop.run_root / frames[1]["path"]).read_bytes()).hexdigest()
    assert [kind for kind, _ in loop.live] == ["operator_view_frame"] * 2
    assert loop.operator_view["observation_step"] == 2
    assert _operator_view_errors(loop.run_root, frames) == []
    (loop.run_root / frames[0]["path"]).write_bytes(png(1, 1))
    assert _operator_view_errors(loop.run_root, frames) == ["play_operator_view_frame_invalid"]


def test_a_failing_view_is_recorded_and_asked_for_no_more_after_three(tmp_path: Path) -> None:
    calls: list[int] = []

    def broken() -> Any:
        calls.append(1)
        return view(width=99)

    harness = _Loop(tmp_path / "run", SimpleNamespace(operator_view=broken))
    for step in range(5):
        harness.loop.observation_published(step)
    assert len(calls) == 3 and len(harness.loop.failures) == 3
    assert all(item.startswith("play:operator_view") for item in harness.loop.failures)
    assert harness.events() == []
    # A runtime without the method is simply not asked.
    plain = _Loop(tmp_path / "plain", SimpleNamespace())
    plain.loop.observation_published(0)
    assert plain.events() == [] and plain.loop.failures == []


class _Inbox:
    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []
        self.acknowledged: list[tuple[str, str]] = []

    def poll(self) -> list[dict[str, Any]]:
        return list(self.commands)

    def acknowledge(self, command: dict[str, Any], status: str, **_fields: Any) -> None:
        self.acknowledged.append((command["id"], status))


def _announcing_loop(root: Path) -> tuple[Any, list[dict[str, Any]], _Inbox]:
    """A PlayLoop's message and action paths, with a planner that hears the announcements."""

    from ora.orchestrator.episode.play import PlayLoop

    loop = object.__new__(PlayLoop)
    announced: list[dict[str, Any]] = []
    inbox = _Inbox()
    root.mkdir(parents=True)
    loop.run_root, loop.events, loop.native = root, root / "events.jsonl", root / "native"
    loop.operator_inbox = inbox
    loop.planner = SimpleNamespace(accept_runtime_event=lambda event: None, announce_play_state=announced.append)
    loop.interaction = SimpleNamespace(observation_step=3, observation_id="obs-3")
    loop.play_scenes, loop.round_index = [{"seed": 0}], 0
    loop.segment, loop.segments, loop.play_pending, loop.play_seen = None, [], [], set()
    loop.play_stop, loop.play_controls, loop.operator_message_count = None, 0, 0
    loop.decisions = loop.refused_proposals = loop.model_calls = 0
    loop.instruction_budgets = {"wall_time_s": 3600, "max_model_calls": 80, "max_robot_decisions": 24}
    loop._moving = None
    loop.live_sequence, loop.auxiliary_failures = 0, []
    loop.operator_port = SimpleNamespace(publish=lambda event: None)
    return loop, announced, inbox


def test_the_operator_hears_a_message_queued_then_delivered_into_the_running_instruction(tmp_path: Path) -> None:
    loop, announced, inbox = _announcing_loop(tmp_path / "run")
    loop._begin_segment({"id": "i1", "type": "instruction", "message": "put the cup on the plate"})
    assert announced[-1]["state"] == "instruction"
    assert announced[-1]["message"] == {"id": "i1", "text": "put the cup on the plate", "instruction": 1}
    inbox.commands.append({"id": "m1", "type": "instruction", "message": "how far along are you?"})
    loop._take_commands()
    assert announced[-1]["state"] == "message_queued"
    assert announced[-1]["message"] == {"id": "m1", "text": "how far along are you?", "instruction": 1}
    assert loop._deliver_operator_messages(observation_step=3, before_model_call_id="play-call-000000") is None
    delivered = [record["message"] for record in announced if record["state"] == "message_delivered"]
    assert [(item["id"], item["joined"], item["observation_step"], item["instruction"]) for item in delivered] == [
        ("i1", False, 3, 1), ("m1", True, 3, 1)]
    payloads = [json.loads(line)["payload"] for line in loop.events.read_text().splitlines()
                if json.loads(line)["type"] == "operator_message"]
    assert [(item["id"], item["joined"], item["instruction_id"]) for item in payloads] == [
        ("i1", False, "i1"), ("m1", True, "i1")]
    assert loop.segment["joined_messages"] == ["m1"]
    # Between instructions a message opens the next one: nothing is queued into an instruction.
    loop.segment = None
    inbox.commands.append({"id": "i2", "type": "instruction", "message": "now the bowl"})
    loop._take_commands()
    assert announced[-1]["state"] != "message_queued"


def test_the_operator_hears_the_robot_start_and_finish_moving(tmp_path: Path, monkeypatch) -> None:
    from ora.orchestrator.episode.loop import EpisodeLoop

    loop, announced, _inbox = _announcing_loop(tmp_path / "run")
    monkeypatch.setattr(EpisodeLoop, "_accept_submitted_action",
                        lambda self, increment: ({"action_id": "act-7"}, {"observation_sha256": "x"}))
    monkeypatch.setattr(EpisodeLoop, "_record_receipt", lambda self, action, receipt, duration: None)
    loop._accept_submitted_action(1)
    assert announced[-1]["state"] == "action_started"
    started = announced[-1]["action"]
    assert {k: started[k] for k in ("action_id", "observation_step")} == {"action_id": "act-7", "observation_step": 3}
    assert isinstance(started["started_unix_s"], float) and isinstance(announced[-1]["unix_s"], float)
    loop._record_receipt({"action_id": "act-7"}, {"timestamps": {"approval": {"created_unix_s": 1.0,
                                                                              "answered_unix_s": 3.5}}}, 1.25)
    assert announced[-1]["state"] == "action_finished"
    finished = announced[-1]["action"]
    assert {k: finished[k] for k in ("action_id", "observation_step", "duration_s", "approval_wait_s")} == {
        "action_id": "act-7", "observation_step": 3, "duration_s": 1.25, "approval_wait_s": 2.5}
    assert finished["started_unix_s"] == started["started_unix_s"] <= finished["finished_unix_s"]
