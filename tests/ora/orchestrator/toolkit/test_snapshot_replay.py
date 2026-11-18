"""Snapshot CLI is explicit, keeps source bytes fixed and never falls back to live."""
import json
from pathlib import Path

import pytest

from ora.app.main import main
from ora.agent.toolkit.interface.types import canonical
from ora.orchestrator.toolkit.snapshot_boundary import select_snapshot
from ora.orchestrator.toolkit.snapshot_replay import experiment
from tests.support.agent.toolkit_snapshot import make_source


def test_cli_step_script_and_before_parent_use_private_state(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    make_source(source, monkeypatch)
    script = tmp_path / "trial.py"
    script.write_text('''from ora.agent.toolkit.lib.observation import observe

def run(api, args):
    observed = observe(api)
    before = api.state.read("tool.marks")
    api.state.write("tool.marks", {"trial": True})
    return {"observed": observed.identity.json(), "marks": before, "args": args}
''')
    before = {str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    for selectors, expected in [(["--step", "0"], {}), (["--snapshot", "--at", "1"], {"rim": [0, 2, 3]})]:
        assert main(["toolkit", "replay", str(source), *selectors, "--script", str(script), "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["mode"] == "snapshot" and result["result"]["marks"] == expected
        assert result["tool_state"]["tool.marks"] == {"trial": True}
        assert Path(result["output"]).is_relative_to(tmp_path / "toolkit-replay")
        assert result["proposed_actions"] == []
    assert before == {str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()}


def test_step_can_compute_and_read_only_recorded_official_check(tmp_path, monkeypatch):
    source = tmp_path / "source"
    make_source(source, monkeypatch)
    computed = experiment(source=source, step=0, tool="compute", arguments={"expressions": "total = 1 + 2"})
    assert computed["status"] == "ok", computed
    checked = experiment(source=source, step=0, tool="check_task", arguments={})
    assert checked["status"] == "ok", checked
    assert checked["result"]["task_success"] is False


def test_step_missing_configuration_never_imports_future_parent_table(tmp_path, monkeypatch):
    source = tmp_path / "source"
    make_source(source, monkeypatch)
    path = source / "native/toolkit-io/tool-inputs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"deliberately": "unrelated future table"}\n')
    result = experiment(source=source, step=0, tool="vision_ground", arguments={"text": "cup"})
    assert result["error"]["type"] == "NotRecorded"
    assert result["missing_requests"][0]["method"] == "tool_context"
    assert not (Path(result["output"]) / "source-tool-inputs.json").exists()
    result = experiment(source=source, step=0, tool="compute")
    assert result["error"]["type"] == "ValueError"
    result = experiment(source=source, step=99, tool="compute", arguments={})
    assert result["error"]["type"] == "NotRecorded"
    assert result["missing_requests"][0]["method"] == "snapshot_boundary"


def test_parent_input_index_uses_model_identity_and_original_request_context(tmp_path, monkeypatch):
    source = tmp_path / "source"
    make_source(source, monkeypatch)
    boundary = select_snapshot(source, at="1")
    context = boundary.context.json()
    context["check_task"] = False
    context["state"]["published_seq"] = 2
    record = dict(schema="ora.toolkit.tool-inputs.v1", context=context,
                  configuration={"marker": "exact-parent"}, inputs={})
    path = source / "native/toolkit-io/tool-inputs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(record) + b"\n")
    script = tmp_path / "context.py"
    script.write_text('def run(api, args):\n    return api.artifacts.tool_context()\n')
    result = experiment(source=source, at="1", script=script)
    assert result["status"] == "ok", result
    assert result["result"] == record["configuration"]
    assert result["context"] == context
    record["context"]["state"]["observation"]["observation_sha256"] = "0" * 64
    path.write_bytes(canonical(record) + b"\n")
    result = experiment(source=source, at="1", script=script)
    assert result["error"]["type"] == "CorruptRecord"


@pytest.mark.parametrize("options", [
    ["--snapshot", "--at", "0", "--live"],
    ["--step", "0", "--execute-action"],
    ["--snapshot"],
])
def test_cli_rejects_mixed_or_incomplete_snapshot_options(tmp_path, monkeypatch, capsys, options):
    source = tmp_path / "source"
    make_source(source, monkeypatch)
    assert main(["toolkit", "replay", str(source), *options, "--tool", "compute", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "ValueError"


def test_snapshot_lib_example_recomputes_real_pixels_with_fake_service_and_comparison(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from ora.agent.toolkit.interface.blobs import BlobStore
    from ora.agent.toolkit.interface.journal import Journal
    from ora.agent.toolkit.interface.runtime import Runtime
    from ora.agent.toolkit.interface.services import sam3_adapters
    from ora.agent.toolkit.lib.observation import observe, load_camera
    from ora.orchestrator.toolkit.snapshot_api import build
    from tests.support.agent.toolkit_lib import BINDINGS, Sam
    source = tmp_path / "source"
    make_source(source, monkeypatch)
    boundary = select_snapshot(source, at="0")
    api, _ = build(boundary, tmp_path / "inspect", services={}, bindings={})
    camera = load_camera(api, observe(api).camera("head"))
    rgb = observe(api).camera("head").rgb.json()
    sam = Sam()
    adapters = sam3_adapters(sam)
    store = BlobStore(source, writable=True)
    store.register(rgb)
    recorder = Runtime(context=boundary.context, store=store, journal=Journal(source),
                       adapters=adapters, bindings=BINDINGS)
    recorder.ground(rgb=recorder.save_array(camera.rgb.numpy()), text="cup", max_candidates=1)
    @contextmanager
    def session(report):
        yield adapters, BINDINGS
    monkeypatch.setattr("ora.orchestrator.toolkit.snapshot_replay.service_session", session)
    example = Path(__file__).resolve().parents[4] / "ora/agent/toolkit/examples/ground_geometry.py"
    result = experiment(source=source, at="0", script=example,
                        arguments={"camera": "head", "text": "cup", "max_candidates": 1})
    assert result["status"] == "ok", result
    assert result["comparison_count"] == 1
    assert result["result"]["comparison"] is None
    output = Path(result["output"])
    comparison = json.loads((output / result["comparisons"]).read_text())
    assert json.loads((output / comparison["reference"]["path"]).read_text())["status"] == "equal"
    assert result["result"]["geometries"][0]["points"] > 0
    assert sam.calls == [("ground", "cup"), ("ground", "cup")]


def test_step_inside_parent_marks_tool_namespace_missing_but_next_parent_restores_rollback(tmp_path, monkeypatch):
    from ora.agent.toolkit.state.snapshot import export_snapshot, file_ref
    from ora.agent.toolkit.interface.types import decode
    source = tmp_path / "source"
    state = make_source(source, monkeypatch)
    tool = state.access(writer="tool")
    with state.tool_journal.boundary(dict(kind="parent", model_call_id="zero", tool_call_id="zero")):
        tool.write("tool.held.left", {"intent": "open"})
        snapshot = source / "state/steps/step-0001/snapshot.json"
        snapshot.unlink()  # Replace this synthetic publication with an in-call publication.
        name = "policy-state/steps/step-0001.json"
        policy = decode((source / name).read_bytes())
        ref = dict(file_ref(source, name, prefix="policy-state/steps/"), step=1,
                   state_sha256=policy["state_sha256"])
        export_snapshot(state, source, ref)
        tool.write("tool.held.left", None)  # 0-spend rollback after the publication.
    with state.tool_journal.boundary(dict(kind="parent", model_call_id="next", tool_call_id="next")):
        pass
    with (source / "agent-events.jsonl").open("ab") as stream:
        stream.write(canonical(dict(seq=4, type="tool_call_proposed", run_id="run", model_call_id="next",
                                   tool_call_id="next", payload=dict(name="compute", arguments={}))) + b"\n")
    script = tmp_path / "read.py"
    script.write_text('''def run(api, args):
    observed = api.state.read("env.observation")
    try:
        held = api.state.read("tool.held.left")
    except Exception as error:
        held = type(error).__name__
    return {"held": held, "observation": observed["observation_id"]}
''')
    result = experiment(source=source, step=1, script=script)
    assert result["status"] == "ok", result
    assert result["result"]["held"] == "NotRecorded"
    assert result["tool_state"] == {}
    assert result["missing_state_fields"][0]["field"] == "tool.*"
    assert "unfinished" in result["missing_state_fields"][0]["reason"]
    settled = experiment(source=source, at="next", script=script)
    assert settled["status"] == "ok", settled
    assert settled["result"]["held"] is None
    assert settled["result"]["observation"] == result["result"]["observation"]


@pytest.mark.parametrize("kind", ["action_proposed", "environment_receipt", "round_started"])
def test_snapshot_parent_rejects_unpublished_mutation_or_reset(tmp_path, monkeypatch, kind):
    from ora.agent.toolkit.interface.types import NotRecorded
    source = tmp_path / "source"
    make_source(source, monkeypatch)
    path = source / "agent-events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["seq"] *= 2
    rows.insert(1, dict(seq=1, type=kind, payload={}))
    path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    with pytest.raises(NotRecorded, match="unpublished mutation/reset"):
        select_snapshot(source, at="0")


def test_legacy_missing_at_keeps_original_message(tmp_path, capsys):
    assert main(["toolkit", "replay", str(tmp_path), "--tool", "compute", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["message"] == "--at is required for replay"


@pytest.mark.parametrize("version", [1, 2, None])
def test_snapshot_protocol_version_never_blocks_decodable_state(tmp_path, monkeypatch, version):
    source = tmp_path / "source"
    make_source(source, monkeypatch)
    if version is not None:
        (source / "toolkit-protocol.json").write_text(json.dumps({"protocol_version": version}))
    script = tmp_path / "read.py"
    script.write_text('def run(api, args):\n    return {"decoded": True}\n')
    result = experiment(source=source, step=0, script=script, output=tmp_path / "out")
    assert result["status"] == "ok", result
    assert result["protocol_compatibility"]["source_version"] == version
    assert result["protocol_compatibility"]["current_version"] == 1
