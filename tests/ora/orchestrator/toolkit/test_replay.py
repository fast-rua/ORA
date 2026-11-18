"""CLI experiments use public real pixels and fake clients, never a simulator."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil

import pytest

from ora.app.main import main
from ora.agent.toolkit.examples.ground_geometry import run as ground_geometry
from ora.agent.toolkit.interface.api import EXTERNAL
from ora.agent.toolkit.interface.gaps import write_gap
from ora.agent.toolkit.interface.services import sam3_adapters
from ora.agent.toolkit.interface.types import CallContext, PublicData, canonical, digest
from ora.orchestrator.toolkit import replay
from ora.orchestrator.toolkit.selection import list_calls
from tests.support.agent.toolkit_lib import BINDINGS, Sam, make_runtime

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/data/toolkit-general-pickup"
EXAMPLE = ROOT / "ora/agent/toolkit/examples/ground_geometry.py"


def tree(root):
    return {str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()}


def script(tmp_path, body):
    path = tmp_path / "candidate.py"
    path.write_text(body)
    return path


def fake_services(monkeypatch, adapters=None):
    sam = Sam()
    @contextmanager
    def session(report):
        yield adapters or sam3_adapters(sam), BINDINGS
    monkeypatch.setattr(replay, "service_session", session)
    return sam


def recorded(tmp_path):
    runtime, sam, _ = make_runtime(tmp_path / "source")
    boundary = dict(kind="before_call", call=0, seq=2, tool_call_id="call-0", model_call_id="model-0")
    runtime.context = CallContext.create(**dict(runtime.context.json(), boundary=boundary))
    proposal = dict(type="tool_call_proposed", run_id="run", seq=2, tool_call_id="call-0", model_call_id="model-0",
                    payload=dict(name="vision_ground", arguments=dict(camera="head", text="red cup", max_candidates=2)))
    (runtime.store.root / "agent-events.jsonl").write_bytes(canonical(proposal) + b"\n")
    return runtime, sam, proposal


def comparison(output):
    events = [json.loads(line) for line in (output / "native/toolkit-io/comparisons.jsonl").read_bytes().splitlines()]
    return [json.loads((output / event["reference"]["path"]).read_bytes()) for event in events]


def test_legacy_cli_list_and_geometry_real_pixels_fake_service(tmp_path, monkeypatch, capsys):
    before = tree(FIXTURE)
    assert main(["toolkit", "replay", str(FIXTURE), "--list", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["calls"] == [dict(call=0, tool_call_id="chatcmpl-tool-81c25cefba22c98b", tool="vision_ground",
                                     has_toolkit_io=False, has_gap=False)]
    sam = fake_services(monkeypatch)
    output = tmp_path / "evaluation"
    assert main(["toolkit", "replay", str(FIXTURE), "--at", "0", "--script", str(EXAMPLE),
                 "--output", str(output), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads((output / "result.json").read_bytes())
    assert result["mode"] == "recorded" and result["evidence_mode"] == "legacy" and result["context"]["boundary"]["seq"] == 6
    assert result["arguments"]["text"] == "mint green scissors"
    assert len(result["result"]["geometries"]) == 3
    assert all(g["points"] > 0 and g["frame"] == "world" for g in result["result"]["geometries"])
    assert sam.calls == [("ground", "mint green scissors")]
    assert comparison(output)[0]["reason"] == "legacy_response_scope_not_equivalent"
    assert tree(FIXTURE) == before


def test_exact_context_bytes_and_recomputed_comparison(tmp_path, monkeypatch):
    runtime, sam, proposal = recorded(tmp_path)
    expected = ground_geometry(runtime, proposal["payload"]["arguments"])
    before = tree(runtime.store.root)
    fake = fake_services(monkeypatch)
    result = replay.experiment(source=runtime.store.root, at="call-0", script=EXAMPLE, output=tmp_path / "out")
    assert result["status"] == "ok", result
    assert result["context"] == runtime.context.json()
    assert result["next_interface_ordinal"] == 2
    assert result["result"]["geometries"] == expected["geometries"]
    assert result["result"]["comparison"]["status"] == "equal"
    assert len(fake.calls) == len(sam.calls) == 1
    assert comparison(tmp_path / "out")[0]["status"] == "equal"
    assert tree(runtime.store.root) == before


def test_exact_simulator_envelope_bytes_and_missing_query(tmp_path):
    runtime, _, _ = recorded(tmp_path)
    first = runtime.observation().payload
    runtime.adapters["probe_motion"] = lambda *args: PublicData.from_json(dict(available=True))
    runtime.probe_motion(request=dict(target=[1, 2, 3]))
    path = script(tmp_path, "def run(api,args):\n return {'payload':api.state.observation().payload.decode()}\n")
    result = replay.experiment(source=runtime.store.root, at="0", script=path, output=tmp_path / "out")
    assert result["result"]["payload"].encode() == first
    path.write_text("def run(api,args):\n api.state.observation()\n return api.planner.probe_motion(request={'target':[9,9,9]}).json()\n")
    result = replay.experiment(source=runtime.store.root, at="0", script=path, output=tmp_path / "new-query")
    assert result["error"]["type"] == "NotRecorded"


@pytest.mark.parametrize("action, error", [("old", None), ("new", "Divergence")])
def test_actions_compare_only_never_execute(tmp_path, action, error):
    runtime, _, _ = recorded(tmp_path)
    runtime.adapters["submit_action"] = lambda *args: PublicData.from_json(dict(accepted=True))
    historical = runtime.submit_action(action=dict(id="old")).json()
    path = script(tmp_path, "def run(api,args):\n return api.act.submit_action(action=args).json()\n")
    before = tree(runtime.store.root)
    result = replay.experiment(source=runtime.store.root, at="0", script=path, arguments=dict(id=action), output=tmp_path / "out")
    assert result["error"]["type"] == error if error else result["result"] == historical
    assert tree(runtime.store.root) == before


@pytest.mark.parametrize("gap_id", [None, "chatcmpl-tool-81c25cefba22c98b"])
def test_legacy_gap_refuses_before_executing_script(tmp_path, gap_id):
    source = tmp_path / "source"
    shutil.copytree(FIXTURE, source)
    write_gap(source, tool_call_id=gap_id, method="ground", stage="project", error=ValueError("lost"))
    path = script(tmp_path, "raise AssertionError('must not execute')\n")
    result = replay.experiment(source=source, at="0", script=path, output=tmp_path / "out")
    assert result["error"]["type"] == "NotRecorded" and "gap" in result["error"]["message"]
    listing = list_calls(source)
    assert listing["unbound_gaps"] == (1 if gap_id is None else 0)
    assert listing["calls"][0]["has_gap"] == (gap_id is not None)


@pytest.mark.parametrize("mode", ["legacy", "recorded"])
def test_no_report_explicit_service_unavailable(tmp_path, mode):
    source = FIXTURE
    if mode == "recorded":
        runtime, _, proposal = recorded(tmp_path)
        ground_geometry(runtime, proposal["payload"]["arguments"])
        source = runtime.store.root
    result = replay.experiment(source=source, at="0", script=EXAMPLE, output=tmp_path / "out")
    assert result["error"]["type"] == "ServiceUnavailable"
    assert "service unavailable" in result["error"]["message"]


def test_default_output_args_and_script_stdout(tmp_path):
    source = tmp_path / "runs" / "source"
    shutil.copytree(FIXTURE, source)
    path = script(tmp_path, "print('imported')\ndef run(api,args):\n print('called')\n return args\n")
    result = replay.experiment(source=source, at="0", script=path)
    output = Path(result["output"])
    assert output.parent == tmp_path / "toolkit-replay" / source.name
    assert not output.is_relative_to(source.parent) and not output.is_relative_to(source)
    import re
    assert re.fullmatch(r"[0-9]{8}T[0-9]{6}\.[0-9]{6}Z-[a-z0-9_]{8}", output.name)
    assert list(source.parent.iterdir()) == [source]
    assert result["result"] == result["arguments"]
    assert (output / "script.log").read_text() == "imported\ncalled\n"
    assert not (tmp_path / "__pycache__").exists()


@pytest.mark.parametrize("body, expected", [
    ("def run(api,args):\n return float('nan')", "InterfaceError"),
    ("def run(api,args):\n raise RuntimeError('broken')", "RuntimeError"),
    ("raise SystemExit(7)", "SystemExit"),
    ("run=42", "ValueError"),
])
def test_script_failures_are_saved(tmp_path, body, expected):
    result = replay.experiment(source=FIXTURE, at="0", script=script(tmp_path, body), output=tmp_path / "out")
    assert result["error"]["type"] == expected
    assert json.loads((tmp_path / "out/result.json").read_text()) == result


def test_tool_missing_service_and_no_source_output(tmp_path, capsys):
    assert main(["toolkit", "replay", str(FIXTURE), "--at", "0", "--tool", "vision_ground",
                 "--output", str(tmp_path / "out"), "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["type"] == "NotRecorded"  # the fixture has no frozen tool inputs
    before = tree(FIXTURE)
    with pytest.raises(ValueError, match="disjoint"):
        replay.experiment(source=FIXTURE, at="0", script=EXAMPLE, output=FIXTURE / "out")
    assert tree(FIXTURE) == before


def test_legacy_events_never_take_future_observation(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(FIXTURE, source)
    snapshot = json.loads((source / "snapshot.json").read_bytes())
    oid = snapshot["available_at_boundary"]["observation"]["id"]
    proposal = dict(type="tool_call_proposed", seq=6, run_id=snapshot["run_id"], tool_call_id="call-0",
                    model_call_id="model-0", payload=dict(name="vision_ground", arguments={}))
    events = [dict(type="fresh_observation_published", seq=1, payload=dict(observation_id=oid)), proposal,
              dict(type="fresh_observation_published", seq=7, payload=dict(observation_id="future-missing"))]
    (source / "agent-events.jsonl").write_bytes(b"".join(canonical(e) + b"\n" for e in events))
    path = script(tmp_path, "def run(api,args):\n return api.state.observation().json()\n")
    result = replay.experiment(source=source, at="0", script=path, output=tmp_path / "out")
    assert result["status"] == "ok", result
    assert result["result"]["data"]["observation"]["observation_id"] == oid


def test_duplicate_parent_ids_and_incomplete_journal_refused(tmp_path):
    runtime, _, proposal = recorded(tmp_path)
    runtime.observation()
    path = runtime.store.root / "agent-events.jsonl"
    path.write_bytes(path.read_bytes() + canonical(dict(proposal, seq=3)) + b"\n")
    with pytest.raises(ValueError, match="unique"):
        list_calls(runtime.store.root)
    path.write_bytes(canonical(proposal) + b"\n")
    journal = runtime.store.root / "native/toolkit-io/calls.jsonl"
    journal.write_bytes(journal.read_bytes().splitlines()[0] + b"\n")
    result = replay.experiment(source=runtime.store.root, at="0", script=EXAMPLE, output=tmp_path / "out")
    assert result["error"]["type"] == "NotRecorded"


def test_second_parent_and_repeated_identical_queries_keep_occurrence(tmp_path):
    runtime, _, first_proposal = recorded(tmp_path)
    runtime.observation()
    proposal = dict(first_proposal, seq=4, tool_call_id="call-1")
    boundary = dict(runtime.context.json()["boundary"], seq=4, call=1, tool_call_id="call-1")
    second_context = CallContext.create(**dict(runtime.context.json(), tool_call_id="call-1", boundary=boundary))
    from ora.agent.toolkit.interface.runtime import Runtime
    api = Runtime(context=second_context, store=runtime.store, journal=runtime.journal,
                  adapters={}, bindings={})
    counter = iter([1, 2])
    api.adapters["probe_motion"] = lambda *args: PublicData.from_json(dict(occurrence=next(counter)))
    api.planner.probe_motion(request={})
    api.planner.probe_motion(request={})
    path = runtime.store.root / "agent-events.jsonl"
    path.write_bytes(path.read_bytes() + canonical(proposal) + b"\n")
    candidate = script(tmp_path, "def run(api,args):\n return [api.planner.probe_motion(request={}).json()['data'] for _ in range(2)]\n")
    result = replay.experiment(source=runtime.store.root, at="1", script=candidate, output=tmp_path / "out")
    assert result["status"] == "ok", result
    assert result["context"] == second_context.json()
    assert result["result"] == [dict(occurrence=1), dict(occurrence=2)]


def test_recorded_gap_overrides_io_and_context_drift_is_refused(tmp_path):
    runtime, _, proposal = recorded(tmp_path)
    runtime.observation()
    source = runtime.store.root
    proposal["seq"] = 999
    (source / "agent-events.jsonl").write_bytes(canonical(proposal) + b"\n")
    result = replay.experiment(source=source, at="0", script=EXAMPLE, output=tmp_path / "drift")
    assert result["error"]["type"] == "CorruptRecord"
    write_gap(source, tool_call_id="call-0", method="observation", stage="end", error=OSError("failed"))
    assert list_calls(source)["calls"][0]["has_toolkit_io"] is True
    assert list_calls(source)["calls"][0]["has_gap"] is True
    result = replay.experiment(source=source, at="0", script=EXAMPLE, output=tmp_path / "gap")
    assert result["error"]["type"] == "NotRecorded"
