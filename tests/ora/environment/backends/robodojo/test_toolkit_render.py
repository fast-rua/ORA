"""Render scheduling is per physics position, including mid-native-tick captures."""

from types import SimpleNamespace
import pytest
from ora.environment.backends.robodojo.toolkit_render import RenderHistory
from ora.environment.backends.robodojo.control_replay import ReplayError


def test_replay_synchronizes_same_clock_multiplicity_and_rounds(tmp_path):
    calls = []
    session = SimpleNamespace(round_index=0, step=10, output_root=tmp_path)
    session._clock = lambda: session.step
    session.environment = SimpleNamespace(
        get_obs=lambda: calls.append((session.round_index, session.step))
    )
    events = [
        dict(index=i, round_index=r, physics_clock=c)
        for i, (r, c) in enumerate(((0, 10), (0, 10), (0, 15), (1, 2)))
    ]
    history = RenderHistory(session, dict(status="recorded", events=events))
    history.attach()
    for clock in range(11, 21):
        session.step = clock
        session.environment.execution_capture_callback()
    previous = session.environment
    session.round_index, session.step = 1, 2
    session.environment = SimpleNamespace(
        get_obs=lambda: calls.append((session.round_index, session.step))
    )
    history.attach()
    report = history.close()
    assert calls == [(0, 10), (0, 10), (0, 15), (1, 2)]
    assert report["complete"] is True
    assert previous.execution_capture_callback is None
    assert session.environment.execution_capture_callback is None


def test_render_that_advances_physics_is_never_tolerated(tmp_path):
    session = SimpleNamespace(round_index=0, step=10, output_root=tmp_path)
    session._clock = lambda: session.step
    session.environment = SimpleNamespace(
        get_obs=lambda: setattr(session, "step", session.step + 1)
    )
    history = RenderHistory(
        session, dict(status="recorded", events=[dict(round_index=0, physics_clock=10)])
    )
    try:
        with pytest.raises(ReplayError, match="advanced physics"):
            history.attach()
    finally:
        history.close()
    assert session.environment.execution_capture_callback is None


def _session(tmp_path, calls):
    session = SimpleNamespace(round_index=0, step=844, output_root=tmp_path)
    session._clock = lambda: session.step
    session.environment = SimpleNamespace(
        get_obs=lambda: calls.append((session.round_index, session.step))
    )
    return session


def test_unavailable_history_still_renders_each_round_start_before_controls(tmp_path):
    """A restore without a render history still needs a render at reset for its wrist cameras.

    The renderer's first transform update after reset fixes the wrist camera's mount on link6
    from link6's pose at that render; first rendered at a moved boundary, the camera keeps its
    reset view and follows the wrist with a wrong offset.
    """
    calls = []
    session = _session(tmp_path, calls)
    history = RenderHistory(session, dict(status="unavailable", events=[], reason="no unique synchronization"))
    history.attach()  # restore start, at the reset clock
    for clock in (845, 900):  # replayed controls, then each action's journal re-attaches
        session.step = clock
        session.environment.execution_capture_callback()
        history.attach()
    session.round_index, session.step = 1, 300  # a later round restarts
    history.attach()
    session.step = 400
    history.attach()
    report = history.close()
    assert calls == [(0, 844), (1, 300)]
    assert [(row["round_index"], row["physics_clock"], row["status"]) for row in report["round_start_renders"]] == [
        (0, 844, "ok"), (1, 300, "ok")]
    assert report["results"] == [] and report["complete"] is False


def test_recorded_history_rendering_at_round_start_adds_no_render(tmp_path):
    calls = []
    session = _session(tmp_path, calls)
    events = [dict(index=0, round_index=0, physics_clock=844), dict(index=1, round_index=0, physics_clock=869)]
    history = RenderHistory(session, dict(status="recorded", events=events))
    history.attach()
    session.step = 869
    session.environment.execution_capture_callback()
    report = history.close()
    assert calls == [(0, 844), (0, 869)]
    assert report["round_start_renders"] == [] and report["complete"] is True


def test_history_starting_after_reset_gets_a_reset_render_first(tmp_path):
    calls = []
    session = _session(tmp_path, calls)
    history = RenderHistory(session, dict(status="recorded", events=[dict(index=0, round_index=0, physics_clock=900)]))
    history.attach()
    session.step = 900
    session.environment.execution_capture_callback()
    report = history.close()
    assert calls == [(0, 844), (0, 900)]
    assert report["complete"] is True
