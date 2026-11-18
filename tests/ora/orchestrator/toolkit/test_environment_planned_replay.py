"""``ora toolkit replay --tool`` restores tool state on a run whose environment planned its moves.

Such a run has no snapshot chain, which only the owner of RoboDojo's execution state writes, so its tool state comes from
``state/tool-deltas.jsonl`` alone: the journal starts at the owner's ``initialize``, and the rows before a call's ``begin``
are that call's state. A mark from a second camera tells this from re-measuring: it is ``verified`` only against the
first mark, and ``solved`` without it. RoboDojo's plain recorded replay keeps its empty store, so a sealed run answers as
it did: ``motion.environment_plans``, absent from every RoboDojo context, is what selects the journal.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ora.agent.toolkit.interface.types import digest
from ora.orchestrator.toolkit.replay import experiment
from ora.orchestrator.toolkit.snapshot_replay import experiment as snapshot_experiment
from tests.support.agent.stack_parity import SURFACES, on_toolkit, stand_in
from tests.support.agent.toolkit_atomic import complete_cameras
from tests.support.agent.toolkit_recording import proposal
from tests.support.agent.toolkit_tools import install, make_host, use_toolkit_v5

#: The surfaces that are to move, with a second camera to verify a mark from.
MOVING = ("libero", "robotwin", "robocasa", "robolab", "yam")


def tree(root):
    return {str(path.relative_to(root)): digest(path.read_bytes()) for path in root.rglob("*") if path.is_file()}


def call(host, index, name, arguments):
    return host.invoke_with_context(name, arguments, call_context=proposal(host, name, arguments, index=index))


def mark(view):
    return {"view": view, "point_ids": ["block"], "points_uv": [[64, 64]]}


@pytest.fixture
def host(request, tmp_path, monkeypatch):
    return on_toolkit(stand_in(tmp_path / "run", request.param, monkeypatch))


@pytest.mark.parametrize("host", MOVING, indirect=True)
def test_a_mark_replays_against_the_marks_before_it(host, tmp_path, request):
    first_view, second_view = SURFACES[request.node.callspec.params["host"]].cameras[:2]
    first = call(host, 0, "mark_point", mark(first_view))
    second = call(host, 1, "mark_point", mark(second_view))
    assert (first["marks"][0]["status"], second["marks"][0]["status"]) == ("solved", "verified")
    source = host.control_root.parent
    before = tree(source)
    for index, expected in enumerate((first, second)):
        replayed = experiment(source=source, at=str(index), tool="mark_point", output=tmp_path / f"replay-{index}")
        assert replayed["status"] == "ok", replayed.get("error")
        # What the tool returned: the host adds this after it, on RoboDojo too (``perception/feedback.py``).
        assert replayed["result"] == {key: value for key, value in expected.items() if key != "measurement_contract"}
    assert tree(source) == before, "a replay writes on its own copy"


@pytest.mark.parametrize("host", ["libero"], indirect=True)
def test_a_held_intention_replays_from_the_journal(host, tmp_path):
    """``set_gripper`` reads and stages ``tool.held.*``, which an empty store refuses."""
    arguments = {"action": "close"}
    expected = call(host, 0, "set_gripper", arguments)
    source = host.control_root.parent
    replayed = experiment(source=source, at="0", tool="set_gripper", output=tmp_path / "replay")
    assert replayed["status"] == "ok", replayed.get("error")
    assert expected["status"] == "submitted" and replayed["result"] == expected


@pytest.mark.parametrize("host", ["libero"], indirect=True)
def test_a_run_without_its_journal_says_so_rather_than_replaying_no_marks(host, tmp_path):
    call(host, 0, "mark_point", mark("agentview"))
    source = host.control_root.parent
    (source / "state/tool-deltas.jsonl").unlink()
    replayed = experiment(source=source, at="0", tool="mark_point", output=tmp_path / "replay")
    assert replayed["status"] == "error" and replayed["error"]["type"] == "NotRecorded"
    assert "未记录" in replayed["error"]["message"]


@pytest.mark.parametrize("host", ["libero"], indirect=True)
def test_a_snapshot_replay_of_such_a_run_reports_what_it_lacks(host, tmp_path):
    """Its publications carry no snapshot chain, so ``--snapshot`` stays 未记录: the journal is ``--tool``'s."""
    call(host, 0, "mark_point", mark("agentview"))
    replayed = snapshot_experiment(source=host.control_root.parent, at="0", tool="mark_point",
                                   output=tmp_path / "replay")
    assert replayed["status"] == "error" and replayed["error"]["type"] == "NotRecorded"


def test_robodojo_s_plain_replay_keeps_its_empty_store(tmp_path, monkeypatch):
    """A RoboDojo run has the journal too, and its plain replay does not read it."""
    @contextmanager
    def invocation(name, context, budget):
        yield SimpleNamespace()

    host, client, _ = make_host(tmp_path / "run", monkeypatch)
    try:
        complete_cameras(host)
        use_toolkit_v5(host)
        host.toolkit_executor = SimpleNamespace(invocation=invocation)
        install(host)
        arguments = {"view": "head", "point_ids": ["block"], "points_uv": [[300, 195]]}
        call(host, 0, "mark_point", arguments)
        source = host.control_root.parent
        assert (source / "state/tool-deltas.jsonl").is_file()
        replayed = experiment(source=source, at="0", tool="mark_point", output=tmp_path / "replay")
        assert replayed["error"] == {"type": "NotRecorded", "message": "未记录: tool.marks"}
    finally:
        client.close()
