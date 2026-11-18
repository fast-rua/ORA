"""A RoboLab episode past the official length: the official result stays the limit's."""

from __future__ import annotations

from ora.environment.backends.robolab.session import RobolabSession
from ora.environment.benchmarks.robolab import TASKS

TASK = "BananaInBowlTask"
LIMIT = TASKS[TASK]["native_action_limit"]


def _session(tmp_path, *, used: int, multiple: int = 3) -> RobolabSession:
    session = RobolabSession(config={"task": TASK, "seed": 0, "official_limit_multiple": multiple},
                             output_root=tmp_path)
    session.native_actions = used
    return session


def test_the_limit_is_snapshotted_and_the_episode_runs_on(tmp_path) -> None:
    session = _session(tmp_path, used=LIMIT - 1)
    session.after_native_action({"official_result": False, "truncated": False})
    assert session.official_limit_snapshot == {"official_success": False, "native_actions": LIMIT}
    assert session.must_stop() is False
    session.native_actions = LIMIT * 3 - 1
    session.after_native_action({"official_result": False, "truncated": True})
    assert session.status()["reason"] == "robolab_native_timeout"
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is False
    assert verdict["native_action_limit"] == verdict["official_native_action_limit"] == LIMIT
    assert verdict["extended_native_action_limit"] == LIMIT * 3
    assert verdict["success_after_official_limit"] is False


def test_a_success_past_the_limit_is_recorded_apart_from_the_official_one(tmp_path) -> None:
    session = _session(tmp_path, used=LIMIT)
    session.official_limit_snapshot = {"official_success": False, "native_actions": LIMIT}
    session.after_native_action({"official_result": True, "truncated": False})
    assert session.must_stop() is True
    assert session.status()["reason"] == "robolab_native_timeout"
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is False and verdict["terminated"] is False
    assert verdict["first_success"] is None
    assert verdict["success_after_official_limit"] is True
    assert verdict["success_after_official_limit_native_actions"] == LIMIT + 1


def test_a_success_on_the_last_official_action_stays_official(tmp_path) -> None:
    session = _session(tmp_path, used=LIMIT - 1)
    session.after_native_action({"official_result": True, "truncated": False})
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is True
    assert verdict["official_limit_snapshot"] is None


def test_at_multiple_one_the_limit_ends_the_episode_and_the_verdict_is_unchanged(tmp_path) -> None:
    session = _session(tmp_path, used=LIMIT - 1, multiple=1)
    session.after_native_action({"official_result": False, "truncated": False})
    assert session.must_stop() is True
    assert "official_limit_snapshot" not in session.persist_official_verdict()


def _progress(score: float, completed: int, total: int = 2) -> dict:
    return {"score": score, "completed": completed, "total": total}


def test_the_verdict_carries_robolabs_score_within_the_official_limit(tmp_path) -> None:
    """RoboLab's leaderboard Score: the subtask score at the official limit, never a later one."""

    session = _session(tmp_path, used=LIMIT - 2)
    session.after_native_action({"official_result": False, "truncated": False, "subtask_progress": _progress(0.25, 0)})
    session.after_native_action({"official_result": False, "truncated": False, "subtask_progress": _progress(0.5, 1)})
    session.after_native_action({"official_result": False, "truncated": False, "subtask_progress": _progress(0.75, 1)})
    score = session.persist_official_verdict()["robolab_score"]
    assert score["schema"] == "ora.robolab-subtask-score.v1"
    assert score["at_official_limit"] == {**_progress(0.5, 1), "native_actions": LIMIT}
    assert score["final"] == {**_progress(0.75, 1), "native_actions": LIMIT + 1}
    assert score["official_score"] == 0.5


def test_an_official_success_scores_one_and_an_untracked_task_scores_null(tmp_path) -> None:
    session = _session(tmp_path, used=10)
    session.after_native_action({"official_result": True, "truncated": False, "subtask_progress": _progress(0.5, 1)})
    assert session.persist_official_verdict()["robolab_score"]["official_score"] == 1.0
    untracked = _session(tmp_path / "untracked", used=10)
    untracked.after_native_action({"official_result": False, "truncated": False})
    score = untracked.persist_official_verdict()["robolab_score"]
    assert score["official_score"] is None and score["final"] is None and score["at_official_limit"] is None


def test_an_episode_that_ends_before_the_limit_keeps_its_last_score(tmp_path) -> None:
    session = _session(tmp_path, used=10, multiple=1)
    session.after_native_action({"official_result": False, "truncated": False, "subtask_progress": _progress(0.5, 1)})
    session.abort("policy_finished_episode")
    score = session.persist_official_verdict()["robolab_score"]
    assert score["at_official_limit"] is None and score["official_score"] == 0.5
