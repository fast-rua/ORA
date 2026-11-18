"""``ora task run``: the task's model-request settings over the Provider profile's."""
from ora.orchestrator.task.run import apply_model_request


def test_the_task_sets_the_model_request_and_the_record_keeps_the_profile_values():
    spec = {"policy": {"provider_profile": "orr_x", "reasoning_effort": "high", "max_output_tokens": 16384}}
    task = {"policy": {"reasoning_effort": "medium", "max_output_tokens": 32768, "baseline": "b"}}
    assert apply_model_request(task, spec) == {
        "reasoning_effort": {"profile": "high", "run": "medium"},
        "max_output_tokens": {"profile": 16384, "run": 32768}}
    assert spec["policy"] == {"provider_profile": "orr_x", "reasoning_effort": "medium", "max_output_tokens": 32768}


def test_without_settings_the_profile_values_stand():
    spec = {"policy": {"reasoning_effort": "high", "max_output_tokens": 16384}}
    assert apply_model_request({"policy": {}}, spec) == {}
    assert spec["policy"] == {"reasoning_effort": "high", "max_output_tokens": 16384}


def test_the_run_transcript_follows_on_the_task_output_unless_switched_off(monkeypatch, tmp_path):
    from ora.orchestrator.task import run

    started = []
    monkeypatch.setattr(run.subprocess, "Popen", lambda command, **kwargs: started.append(command) or "watcher")
    assert run.follow_transcript(tmp_path) == "watcher"
    assert started[0][-7:] == ["watch", "--plain", "--ui", "plain", "--speed", "0", str(tmp_path)]
    monkeypatch.setenv("ORA_TASK_TRANSCRIPT", "off")
    assert run.follow_transcript(tmp_path) is None and len(started) == 1


def test_a_stop_signal_becomes_one_graceful_stop_of_the_run(monkeypatch):
    import os
    import signal

    from ora.orchestrator import launch
    from ora.orchestrator.task import run

    stops, lines = [], []
    monkeypatch.setattr(launch, "request_stop", lambda run_id, report=print: stops.append(run_id))
    before = signal.getsignal(signal.SIGTERM)
    forwarding = run.forward_stop_signals("run-1", lines.append)
    os.kill(os.getpid(), signal.SIGTERM)
    os.kill(os.getpid(), signal.SIGINT)  # a second signal adds no second stop request
    forwarding.restore()
    assert stops == ["run-1"] and forwarding.signal == signal.SIGTERM
    assert "SIGTERM: stopping run run-1" in lines[0]
    assert signal.getsignal(signal.SIGTERM) == before


def test_a_task_without_the_sandbox_composes_a_spec_that_keeps_only_hashes(monkeypatch):
    from types import SimpleNamespace

    from ora.orchestrator.backends import registry
    from ora.orchestrator.task.run import compose_spec

    monkeypatch.setattr(registry, "backend", lambda kind: SimpleNamespace(compose_task=lambda task, binding: {}))
    assert compose_spec({"simulator": "x", "save_sandbox": False}, {}) == {"save_sandbox": False}
    assert compose_spec({"simulator": "x", "save_sandbox": True}, {}) == {}
