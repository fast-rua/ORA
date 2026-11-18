"""Contracts at the public data boundary, without simulator or model services."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ora.agent.toolkit.interface.api import EXTERNAL, METHODS
from ora.agent.toolkit.interface.blobs import BlobStore, IO_ROOT
from ora.agent.toolkit.interface.journal import Journal, read_events
from ora.agent.toolkit.interface.replay import Replay
from ora.agent.toolkit.interface.runtime import Runtime, request_wire
from ora.agent.toolkit.interface.types import (
    AccessDenied, AmbiguousRecord, CallContext, CorruptRecord, Divergence,
    InterfaceError, NotRecorded, PublicData, canonical, digest,
)


BINDINGS = {name: dict(backend="fake", service_identity={"id": "fake-service"},
                       model_identity={"id": "fake-model"}, configuration_identity={"version": 1})
            for name in EXTERNAL}


def context(*, run="run", call="call-0", task=True, boundary=None):
    return CallContext.create(run_id=run, round_index=0, tool_call_id=call,
                              boundary=boundary or dict(kind="before_call", call=0),
                              state={"observation": {"observation_id": "obs-0", "observation_sha256": "a" * 64}},
                              check_task=task)


def live(root, *, ctx=None, adapters=None, retries=0):
    return Runtime(context=ctx or context(), store=BlobStore(root, writable=True), journal=Journal(root),
                   adapters=adapters or {}, bindings=BINDINGS, service_retries=retries)


def replay(runtime, tmp_path, *, services=None, ctx=None):
    ctx = ctx or runtime.context
    evaluator = live(tmp_path / "evaluation", ctx=CallContext.create(
        run_id="evaluation", round_index=0, tool_call_id="eval-0", boundary=ctx.json()["boundary"],
        state=ctx.json()["state"], check_task=ctx.json()["check_task"]), adapters=services)
    return Replay(context=ctx, store=BlobStore(runtime.store.root), services=services or {},
                  bindings=BINDINGS, evaluation=evaluator)


@pytest.mark.parametrize('parent', ['call-0', None, 'other', 'corrupt'])
def test_replay_rejects_selected_unbound_or_corrupt_gaps(tmp_path, parent):
    from ora.agent.toolkit.interface.gaps import write_gap
    runtime = live(tmp_path/'run', adapters={
        'check_task': lambda p, i, s: PublicData.from_json({'task_success':False})})
    result = runtime.check_task()
    write_gap(runtime.store.root, tool_call_id=parent, method='check_task',
              stage='project', error=ValueError('fixture failure'))
    if parent == 'corrupt':
        (runtime.store.root/IO_ROOT/'gaps.jsonl').write_bytes(b'{partial')
    if parent == 'other':
        assert replay(runtime, tmp_path).check_task().payload == result.payload
    else:
        with pytest.raises(NotRecorded):
            replay(runtime, tmp_path)


def invoke(api, method, ref=None):
    if method in ("observation", "proprioception", "check_task"):
        return getattr(api, method)()
    if method == "receipt":
        return api.receipt(action_id="action-0")
    if method in ("probe_motion", "align_plan", "reach_check", "kinematics", "finish_episode"):
        return getattr(api, method)(request={"schema": "test.request", "query": [3, 2, 1]})
    if method == "execute_action":
        return api.execute_action(action={"action_id": "action-0"}, budget=20)
    if method == "submit_action":
        return api.submit_action(action={"schema": "ora-action.v3", "action_id": "action-0"})
    if method == "report_issue":
        return api.report_issue(issue={"message": "blocked"})
    if method == "ground":
        return api.ground(rgb=ref, text="cup")
    if method == "segment":
        return api.segment(rgb=ref, depth=ref, point_uv=(1, 2))
    if method == "segment_points":
        return api.segment_points(rgb=ref, depth=ref, points_uv=((1, 2), (2, 1)))
    if method == "grasp_candidates":
        return api.grasp_candidates(scene=ref, target=ref, camera_to_world=np.eye(4).tolist(),
                                    workspace_bounds_m={"x": [-1, 1], "y": [-1, 1], "z": [0, 2]})
    raise AssertionError(method)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_each_method_roundtrip_and_services_recompute(tmp_path, method):
    calls = []
    def adapter(params, inputs, store):
        calls.append((params, inputs))
        return PublicData.from_json(dict(result=[1, 2, 3], method=method))
    runtime = live(tmp_path / "run", adapters={method: adapter})
    ref = runtime.store.put(b"input")
    original = invoke(runtime, method, ref)
    external = method in ("segment", "segment_points", "ground", "grasp_candidates")
    recorded = replay(runtime, tmp_path, services={method: adapter} if external else None)
    result = invoke(recorded, method, ref)
    assert result.payload == original.payload
    assert len(calls) == (2 if external else 1)
    assert recorded.replay_state.cursor == 1
    assert (result.comparison or {}).get("status") == ("equal" if external else None)
    events = read_events(runtime.journal.path)
    assert [e["event"] for e in events] == ["begin", "end"]
    assert events[1]["response_ref"]["sha256"] == digest(original.payload)
    assert events[0]["invocation_id"] == events[1]["invocation_id"]
    assert events[0]["semantic_request_hash"] == digest(canonical(events[0]["request"]["identity"]))


def test_same_parameters_use_occurrences_and_missing_never_uses_first(tmp_path):
    sequence = iter([False, True])
    runtime = live(tmp_path / "run", adapters={"check_task": lambda p, i, s: PublicData.from_json({"task_success": next(sequence)})})
    first, second = runtime.check_task(), runtime.check_task()
    recorded = replay(runtime, tmp_path)
    assert recorded.check_task().payload == first.payload
    assert recorded.check_task().payload == second.payload
    with pytest.raises(NotRecorded):
        recorded.check_task()
    events = runtime.journal.events
    assert events[0]["semantic_request_hash"] == events[2]["semantic_request_hash"]
    assert events[0]["invocation_id"] != events[2]["invocation_id"]


def test_legacy_action_divergence_and_single_proposal_without_online_actions(tmp_path):
    calls = []
    runtime = live(tmp_path / "run", adapters={"submit_action": lambda p, i, s: calls.append(p) or PublicData.from_json({"accepted": True})})
    result = runtime.submit_action(action={"x": 1})
    with pytest.raises(AccessDenied, match="one action proposal per parent tool invocation"):
        runtime.submit_action(action={"x": 1})
    recorded = replay(runtime, tmp_path)
    with pytest.raises(Divergence):
        recorded.submit_action(action={"x": 2})
    assert recorded.replay_state.cursor == 0
    assert recorded.submit_action(action={"x": 1}).payload == result.payload
    with pytest.raises(AccessDenied, match="one action proposal per parent tool invocation"):
        recorded.submit_action(action={"x": 1})
    assert len(calls) == 1
    with pytest.raises(AccessDenied):
        replay(runtime, tmp_path / "illegal", services={"submit_action": lambda *args: None})


def test_service_branch_does_not_skip_cursor_and_comparison_is_separate(tmp_path):
    runtime = live(tmp_path / "run", adapters={"ground": lambda p, i, s: PublicData.from_json({"score": 1})})
    ref = runtime.store.put(b"image")
    runtime.ground(rgb=ref, text="cup")
    recorded = replay(runtime, tmp_path, services={"ground": lambda p, i, s: PublicData.from_json({"score": 2})})
    branch = recorded.ground(rgb=ref, text="bowl")
    assert branch.comparison["status"] == "unavailable" and recorded.replay_state.cursor == 0
    matched = recorded.ground(rgb=ref, text="cup")
    assert matched.comparison["status"] == "different" and recorded.replay_state.cursor == 1
    assert "comparison" not in matched.json()
    assert matched.json()["data"]["score"] == 2
    assert (recorded.evaluation.store.root / IO_ROOT / "comparisons.jsonl").is_file()


def test_retry_records_timeout_and_success_without_error_text(tmp_path):
    calls = []
    def adapter(p, i, s):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("secret must not persist")
        return PublicData.from_json({"score": 1})
    runtime = live(tmp_path / "run", adapters={"ground": adapter}, retries=1)
    result = runtime.ground(rgb=runtime.store.put(b"image"), text="cup")
    assert result.json()["status"] == "ok"
    events = read_events(runtime.journal.path)
    assert [e["attempt"] for e in events] == [0, 0, 1, 1]
    assert events[1]["status"] == "timeout"
    assert b"secret" not in runtime.journal.path.read_bytes()
    assert len({e["invocation_id"] for e in events}) == 1


def test_failed_action_is_not_retried(tmp_path):
    def failure(*args):
        raise TimeoutError("uncertain")
    runtime = live(tmp_path / "run", adapters={"submit_action": failure}, retries=3)
    assert runtime.submit_action(action={"id": 1}).json()["status"] == "timeout"
    assert len(runtime.journal.events) == 2
    # A failed legacy submission still consumes its one proposal opportunity.
    with pytest.raises(AccessDenied, match="one action proposal per parent tool invocation"):
        runtime.submit_action(action={"id": 1})
    assert len(runtime.journal.events) == 2


def test_no_final_checker_backfill_or_future_receipts(tmp_path):
    runtime = live(tmp_path / "run")
    (runtime.store.root / "native/official-verdict.json").write_text('{"task_success": true}')
    recorded = replay(runtime, tmp_path)
    with pytest.raises(NotRecorded):
        recorded.check_task()
    with pytest.raises(NotRecorded):
        recorded.receipt(action_id="future")
    denied = live(tmp_path / "denied", ctx=context(task=False))
    with pytest.raises(AccessDenied):
        denied.check_task()


def test_canonical_defaults_path_independence_and_array_order(tmp_path):
    runtime = live(tmp_path / "run", adapters={"ground": lambda p, i, s: PublicData.from_json({})})
    ref = runtime.store.put(b"image")
    runtime.ground(rgb=ref, text="cup")
    runtime.ground(rgb=ref, text="cup", max_candidates=3)
    assert runtime.journal.events[0]["semantic_request_hash"] == runtime.journal.events[2]["semantic_request_hash"]
    kw = dict(context=runtime.context, bindings={})
    a = request_wire("ground", dict(text="cup", max_candidates=3), {"rgb": ref}, **kw)
    b = request_wire("ground", dict(max_candidates=3, text="cup"), {"rgb": dict(ref, path="relocated")}, **kw)
    assert canonical(a["identity"]) == canonical(b["identity"])
    assert canonical([1, 2]) != canonical([2, 1])
    for bad in (float("nan"), float("inf"), {1: "x"}, object()):
        with pytest.raises(InterfaceError):
            canonical(bad)
    matrix = np.asfortranarray(np.arange(6, dtype=">f4").reshape(2, 3))
    array_ref = runtime.store.put_array(matrix)
    assert array_ref["array"]["order"] == "F"
    assert array_ref["array"]["byteorder"] == "big"
    np.testing.assert_array_equal(runtime.store.array(array_ref), matrix)
    with pytest.raises(CorruptRecord):
        runtime.store.register(dict(array_ref, array=dict(array_ref["array"], shape=[7])))


def test_path_allowlist_and_symlink_and_tampering(tmp_path):
    store = BlobStore(tmp_path / "run", writable=True)
    ref = store.put(b"ok")
    for path in ("../secret", "/tmp/secret", "native/official-verdict.json", "control/policy-task-check.json"):
        with pytest.raises(AccessDenied):
            store.register(dict(ref, path=path))
    target = tmp_path / "outside"
    target.write_bytes(b"ok")
    file = store.root / ref["path"]
    file.unlink()
    file.symlink_to(target)
    with pytest.raises(AccessDenied):
        store.read(ref)
    file.unlink()
    file.write_bytes(b"bad")
    with pytest.raises(CorruptRecord):
        store.read(ref)


def test_incomplete_duplicate_and_wrong_context_are_not_replayed(tmp_path):
    runtime = live(tmp_path / "run", adapters={"check_task": lambda p, i, s: PublicData.from_json({"success": False})})
    runtime.check_task()
    original = runtime.journal.path.read_bytes()
    runtime.journal.path.write_bytes(original.splitlines(keepends=True)[0])
    with pytest.raises(NotRecorded):
        replay(runtime, tmp_path / "incomplete")
    events = [json.loads(line) for line in original.splitlines()]
    duplicate = dict(events[0], sequence=2)
    runtime.journal.path.write_bytes(original + canonical(duplicate) + b"\n")
    with pytest.raises(AmbiguousRecord):
        read_events(runtime.journal.path)
    runtime.journal.path.write_bytes(original)
    with pytest.raises(AmbiguousRecord):
        live(runtime.store.root)
    wrong = replay(runtime, tmp_path / "wrong", ctx=context(call="other"))
    with pytest.raises(NotRecorded):
        wrong.check_task()
    with pytest.raises(NotRecorded):
        replay(runtime, tmp_path / "middle", ctx=context(boundary=dict(kind="after_native_action", native_action=1)))


def test_permanent_service_identity_failure_is_recorded_without_retry(tmp_path):
    from ora.agent.model_services.clients import ModelServiceError
    calls = []
    def adapter(*args):
        calls.append(1)
        raise ModelServiceError("private endpoint secret", service="SAM3", failure="permanent", status=503,
                                detail={"token": "secret"})
    runtime = live(tmp_path / "run", adapters={"ground": adapter}, retries=3)
    response = runtime.ground(rgb=runtime.store.put(b"input"), text="cup")
    assert len(calls) == 1
    assert response.json()["error"] == dict(code="model_service_permanent", failure_class="permanent",
                                            http_status=503, retryable=False)
    assert b"secret" not in response.payload


def test_service_can_use_new_locally_computed_blob_without_consuming_history(tmp_path):
    runtime = live(tmp_path / "run")
    recorded = replay(runtime, tmp_path, services={"ground": lambda p, i, s: PublicData.from_json({"bytes": len(s.read(i["rgb"]))})})
    new = recorded.evaluation.store.put(b"new-input")
    response = recorded.ground(rgb=new, text="new query")
    assert response.json()["data"]["bytes"] == 9
    assert response.comparison["status"] == "unavailable" and recorded.replay_state.cursor == 0


def test_duplicate_json_keys_do_not_silently_change_evidence(tmp_path):
    runtime = live(tmp_path / "run", adapters={"check_task": lambda *args: PublicData.from_json({"task_success": False})})
    runtime.check_task()
    data = runtime.journal.path.read_bytes().replace(b'"attempt":0', b'"attempt":1,"attempt":0', 1)
    runtime.journal.path.write_bytes(data)
    with pytest.raises(CorruptRecord):
        read_events(runtime.journal.path)


def test_alternative_simulator_requests_are_exact_and_pre_effect_only(tmp_path):
    from ora.agent.toolkit.interface.experiment import RecordedExperiment
    calls = []
    def adapter(p, i, s):
        calls.append(p)
        return PublicData.from_json(dict(value=len(calls)))
    runtime = live(tmp_path/'run', adapters={name: adapter for name in ('reach_check','submit_action','check_task')})
    first = runtime.reach_check(request={'point':[1,2,3]})
    runtime.submit_action(action={'action_id':'original'})
    runtime.check_task()
    api = RecordedExperiment(replay(runtime,tmp_path), alternative=True,allow_check_task=True)
    assert api.reach_check(request={'point':[1,2,3]}).payload == first.payload
    with pytest.raises(NotRecorded) as error:
        api.reach_check(request={'point':[1,2,4]})
    assert error.value.missing_requests[0]['method'] == 'reach_check'
    with pytest.raises(NotRecorded):
        api.check_task()  # A post-action verdict is not before-call evidence.
    proposed = api.submit_action(action={'action_id':'new'})
    assert proposed.json()['data']['proposal_only'] is True
    with pytest.raises(AccessDenied, match='one action proposal per parent tool invocation'):
        api.submit_action(action={'action_id':'second'})
    assert len(calls) == 3
    assert api.cursor is None and api.replay.replay_state.cursor == 0


def test_alternative_repeated_simulator_query_does_not_choose_first(tmp_path):
    from ora.agent.toolkit.interface.experiment import RecordedExperiment
    values = iter([False, True])
    runtime = live(tmp_path/'run', adapters={'check_task': lambda p,i,s: PublicData.from_json({'task_success':next(values)})})
    runtime.check_task()
    runtime.check_task()
    api = RecordedExperiment(replay(runtime,tmp_path), alternative=True,allow_check_task=True)
    with pytest.raises(AmbiguousRecord):
        api.check_task()
    assert api.replay.replay_state.cursor == 0


def test_check_override_keeps_original_context_and_does_not_enable_runtime(tmp_path):
    from ora.agent.toolkit.interface.experiment import RecordedExperiment
    runtime = live(tmp_path/'run',ctx=context(task=False),adapters={})
    api = RecordedExperiment(replay(runtime,tmp_path),allow_check_task=True)
    with pytest.raises(NotRecorded) as error:
        api.check_task()
    assert error.value.missing_requests[0]['request']['identity']['capabilities']['check_task'] is True
    assert api.context.json()['check_task'] is False
    with pytest.raises(AccessDenied):
        api.replay.check_task()


def test_synchronous_execute_action_has_no_legacy_proposal_guard(tmp_path):
    requests = []
    api = live(tmp_path, adapters={"execute_action": lambda p, i, s:
        requests.append(p) or PublicData.from_json({"receipt": {"action_id": p["action"]["action_id"]}, "remaining_budget": 10})})
    for index in range(2):
        assert api.execute_action(action={"action_id": f"action-{index}"}, budget=5).json()["status"] == "ok"
    assert len(requests) == 2


def test_live_legacy_second_submission_cannot_overwrite_original_file(tmp_path):
    from ora.agent.toolkit.host.live import LiveRuntime
    path = tmp_path / "submitted-action.json"
    def submit(parameters, inputs, store):
        path.write_bytes(canonical(parameters["action"]))
        return PublicData.from_json({"accepted": True})
    api = LiveRuntime(context=context(), store=BlobStore(tmp_path, writable=True), journal=Journal(tmp_path),
                      adapters={"submit_action": submit}, bindings=BINDINGS)
    api.submit_action(action={"action_id": "first"})
    first = path.read_bytes()
    with pytest.raises(AccessDenied, match="one action proposal per parent tool invocation"):
        api.submit_action(action={"action_id": "second"})
    assert path.read_bytes() == first


def test_a_note_goes_on_a_call_s_events_and_stays_out_of_its_request(tmp_path):
    from ora.agent.toolkit.interface.call_note import noted

    adapters = {"check_task": lambda p, i, s: PublicData.from_json({"ok": True})}
    plain = live(tmp_path / "plain", adapters=adapters)
    plain.check_task()
    runtime = live(tmp_path / "noted", adapters=adapters)
    with noted(route="line", skipped=None):
        expected = runtime.check_task().payload
    runtime.check_task()
    first, second = read_events(runtime.journal.path)[:2], read_events(runtime.journal.path)[2:]
    assert [event["note"] for event in first] == [{"route": "line"}, {"route": "line"}]
    assert all("note" not in event for event in second)         # a call outside the block is as it always was
    base = read_events(plain.journal.path)[0]
    assert first[0]["semantic_request_hash"] == base["semantic_request_hash"] and first[0]["request"] == base["request"]
    assert replay(runtime, tmp_path).check_task().payload == expected


@pytest.mark.parametrize("version", [1, 99, None])
def test_manual_protocol_metadata_is_advisory_and_journal_is_versioned(tmp_path, version):
    from ora.agent.toolkit.interface.protocol import audit
    from ora.contracts.toolkit_protocol import metadata

    runtime = live(tmp_path / "run", adapters={"check_task": lambda p, i, s: PublicData.from_json({"ok": True})})
    expected = runtime.check_task().payload
    events = read_events(runtime.journal.path)
    manifest = json.loads((runtime.journal.path.parent / "manifest.json").read_bytes())
    assert manifest["protocol_version"] == 1
    assert {event["protocol_version"] for event in events} == {1}
    path = runtime.store.root / "toolkit-protocol.json"
    if version is not None:
        path.write_bytes(canonical(dict(metadata(recording_enabled=True), protocol_version=version)))
    before = {str(p): p.read_bytes() for p in runtime.store.root.rglob("*") if p.is_file()}
    recorded = replay(runtime, tmp_path)
    assert recorded.check_task().payload == expected
    report = audit(runtime.store.root)
    assert report["source_version"] == version and report["current_version"] == 1
    assert report["status"] == {1: "matched", 99: "mismatch", None: "unknown"}[version]
    assert bool(report["warnings"]) == (version != 1)
    assert before == {str(p): p.read_bytes() for p in runtime.store.root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("method", ["check_task", "submit_action", "ground"])
def test_decodable_method_version_drift_uses_current_decoder_with_warning(tmp_path, method):
    runtime = live(tmp_path / "run", adapters={method: lambda p, i, s: PublicData.from_json({"ok": True})})
    ref = runtime.store.put(b"image")
    expected = invoke(runtime, method, ref)
    rows = read_events(runtime.journal.path)
    for row in rows:
        row["request"]["identity"]["method_version"] = 42
        row["semantic_request_hash"] = digest(canonical(row["request"]["identity"]))
    runtime.journal.path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    adapters = {method: lambda p, i, s: PublicData.from_json({"ok": True})} if method in EXTERNAL else None
    recorded = replay(runtime, tmp_path, services=adapters)
    assert invoke(recorded, method, ref).payload == expected.payload
    warnings = recorded.protocol_compatibility["warnings"]
    assert any(w.get("field") == "method_version" and w["affected_methods"] == [method] for w in warnings)
    # Version labels do not waive evidence hash checks.
    rows[0]["request"]["identity"]["method_version"] = 43
    runtime.journal.path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    with pytest.raises(CorruptRecord):
        read_events(runtime.journal.path)


def test_an_old_unversioned_journal_is_readable_but_not_claimed_protocol_exact(tmp_path):
    runtime = live(tmp_path / "run", adapters={"check_task": lambda p, i, s: PublicData.from_json({"ok": True})})
    expected = runtime.check_task().payload
    rows = read_events(runtime.journal.path)
    for row in rows:
        row.pop("protocol_version")
    runtime.journal.path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    (runtime.journal.path.parent / "manifest.json").unlink()
    recorded = replay(runtime, tmp_path)
    assert recorded.check_task().payload == expected
    assert recorded.protocol_compatibility["status"] == "unknown"
    assert not recorded.protocol_compatibility["exact_replay_eligible"]
