"""Real artifact bytes and fake resident clients exercise the actual adapters."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.journal import Journal
from ora.agent.toolkit.interface.legacy import LegacyReplay
from ora.agent.toolkit.interface.projections import alignment, kinematics, motion_probe, reach
from ora.agent.toolkit.interface.runtime import Runtime
from ora.agent.toolkit.interface.services import grasp_adapter, sam3_adapters
from ora.agent.toolkit.interface.types import AccessDenied, CallContext, NotRecorded, PublicData, canonical, digest
from ora.agent.toolkit.host.adapter import bind_runtime

FIXTURE = Path(__file__).resolve().parents[4] / "data/toolkit-general-pickup"


def test_real_legacy_observation_bytes_and_no_future_state(tmp_path):
    snapshot = json.loads((FIXTURE / "snapshot.json").read_bytes())
    obs = snapshot["available_at_boundary"]["observation"]
    ctx = CallContext.create(run_id=snapshot["run_id"], round_index=0,
                             tool_call_id=snapshot["identity"]["tool_call_id"],
                             boundary=snapshot["boundary"], state={"observation": {
                                 "observation_id": obs["id"], "observation_sha256": obs["manifest"]["sha256"]}},
                             check_task=True)
    evaluation = Runtime(context=ctx, store=BlobStore(tmp_path, writable=True),
                         journal=Journal(tmp_path), adapters={}, bindings={})
    store = BlobStore(FIXTURE)
    legacy = LegacyReplay(snapshot=snapshot, context=ctx, store=store, evaluation=evaluation, bindings={})
    result = legacy.observation().json()
    assert result["provenance"]["exact_interface_bytes"] is False
    assert result["data"]["manifest"]["sha256"] == obs["manifest"]["sha256"]
    ref = legacy.proprioception().json()["data"]["proprioception"]
    assert store.read(ref) == (FIXTURE / ref["path"]).read_bytes()
    assert digest(store.read(ref)) == ref["sha256"]
    for call in (lambda: legacy.check_task(), lambda: legacy.receipt(action_id="future"),
                 lambda: legacy.probe_motion(request={"queries": []}),
                 lambda: legacy.observation(observation={"observation_id": "future", "observation_sha256": "0" * 64})):
        with pytest.raises(NotRecorded):
            call()


class FakeSam3:
    def __init__(self):
        self.calls = []

    def segment(self, *, rgb_u8, depth_m, point_uv):
        rgb, depth, point = rgb_u8, depth_m, point_uv
        self.calls.append(("segment", rgb.copy(), depth.copy(), point))
        return np.ones(depth.shape, dtype=bool), {"model_score": 0.9, "private": "discard"}

    def segment_points(self, *, rgb_u8, depth_m, points_uv):
        rgb, depth, points = rgb_u8, depth_m, points_uv
        self.calls.append(("segment_points", rgb.copy(), depth.copy(), points))
        return [dict(query_index=n, source_index=0, point_uv=list(point),
                     backend_score=0.7, mask=np.ones(depth.shape, dtype=bool), private="discard")
                for n, point in enumerate(points)], {"image_encodings": 1}

    def ground(self, *, rgb_u8, text, max_candidates):
        rgb = rgb_u8
        self.calls.append(("ground", rgb.copy(), text, max_candidates))
        return [dict(source_index=0, rank=0, bbox_xyxy=[0, 0, 3, 2], backend_score=0.8,
                     mask=np.ones(rgb.shape[:2], dtype=bool))], {"returned_candidate_count": 1}


@pytest.mark.parametrize("method", ["segment", "segment_points", "ground"])
def test_sam3_client_projection_and_array_roundtrip(tmp_path, method):
    store = BlobStore(tmp_path, writable=True)
    rgb_array = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    rgb = store.put_array(rgb_array)
    depth_array = np.ones((2, 3), dtype=np.float32)
    depth = store.put_array(depth_array)
    client = FakeSam3()
    params = {"segment": dict(point_uv=[1, 1]), "segment_points": dict(points_uv=[[1, 1], [2, 0]]),
              "ground": dict(text="cup", max_candidates=3)}[method]
    result = sam3_adapters(client)[method](params, dict(rgb=rgb, depth=depth), store)
    value = json.loads(result.data)
    assert b"discard" not in result.data
    np.testing.assert_array_equal(client.calls[0][1], rgb_array)
    masks = [value["mask"]] if method == "segment" else [row["mask"] for row in value["candidates"]]
    for mask in masks:
        np.testing.assert_array_equal(store.array(mask), np.ones((2, 3), dtype=bool))


def test_grasp_preserves_full_backend_pool_not_compiler_selected_subset(tmp_path):
    store = BlobStore(tmp_path, writable=True)
    scene = store.put_array(np.ones((16, 3), dtype=np.float32))
    raw = dict(poses_camera=np.repeat(np.eye(4)[None], 4, axis=0), scores=np.array([.9, .8, .7, .6]),
               contact_points_camera=np.zeros((4, 3)), openings_m=np.ones(4) * .03)
    def propose(params, points, target):
        assert points.shape == (16, 3) and target.dtype == np.float32
        return [dict(selected="only one")], {"_canonical_backend_output": raw, "returned_candidate_count": 4}
    result = grasp_adapter(propose)({}, dict(scene=scene, target=scene), store)
    data = json.loads(result.data)
    assert data["compiler_applied_to_output"] is False
    assert len(store.array(data["arrays"]["poses_camera"])) == 4
    for key in raw:
        np.testing.assert_array_equal(store.array(data["arrays"][key]), raw[key])


def test_field_projections_exclude_private_worker_objects_and_keep_proof_limits():
    secret = object()
    result = motion_probe(dict(schema="ora.motion-probe-result.v1", observation_id="obs", world_state_changed=False,
                               private=secret, results=[dict(query_id="q", arm="left", plan_ref="plan-1", private=secret,
                               variants=[dict(variant_id="exact", reachable=False, private=secret,
                               planner=dict(planner="sdk_ik_joint_interpolation", parity="non_parity_motion", private=secret),
                               failure=dict(code="ik_failed", message="no candidate", private=secret))])]))
    row = json.loads(result.data)["results"][0]
    assert row["variants"][0]["planner"]["parity"] == "non_parity_motion"
    assert b"private" not in result.data
    aligned = alignment(dict(candidates=[dict(align_pose=dict(position_m=[0, 0, 1], orientation_xyzw=[0, 0, 0, 1], private=secret),
                                             claim_boundary="IK only", private=secret)], private=secret))
    assert b"private" not in aligned.data
    checked = reach(dict(results=[dict(reachable=False, route="global", reason="ik_failed", private=secret)],
                         claim_boundary="no collision proof", motion_planner_called=False, private=secret))
    assert json.loads(checked.data)["claim_boundary"] == "no collision proof"



def test_a_table_screen_answer_keeps_its_clearance_and_limiting_link():
    secret = object()
    answer = kinematics(dict(schema="ora.kinematics-result.v1", operation="screen_path", status="table_collision",
                             first_blocked_segment=3, min_table_clearance_m=-0.004, limiting_link="link6",
                             private=secret))
    data = json.loads(answer.data)
    assert (data["min_table_clearance_m"], data["limiting_link"]) == (-0.004, "link6")
    assert b"private" not in answer.data

def test_live_adapter_uses_existing_boundaries_and_never_dumps_runtime(tmp_path):
    root = tmp_path / "policy-observations/obs-0"
    root.mkdir(parents=True)
    raw = canonical(dict(observation_id="obs-0", cameras={}, round_index=0))
    (root / "observation.json").write_bytes(raw)
    (root / "proprioception.json").write_bytes(b'{"joint_qpos": [0]}')
    observed = dict(observation_id="obs-0", observation_sha256=digest(raw))
    ctx = CallContext.create(run_id="run", round_index=0, tool_call_id="call", boundary=dict(kind="before_call", call=0),
                             state=dict(observation=observed), check_task=True)
    current = lambda: (dict(observed, open=True), root, json.loads(raw))
    calls = []
    backend = SimpleNamespace(_current=current, motion_probe_callback=lambda request: dict(results=[], world_state_changed=False),
                              align_plan_callback=lambda request: dict(candidates=[], motion_planner_called=False),
                              _check_task=lambda p: {"success": True, "task_success": False},
                              _report_issue=lambda p: calls.append(p) or {"success": True},
                              _finish=lambda p: {"success": True, "outcome": p["outcome"]})
    from ora.agent.toolkit.state.store import StateStore
    state = StateStore()
    state.access(reader="supervisor", writer="environment").update({
        "env.observation": observed, "privileged.task_check": {"task_success": False}})
    backend.toolkit_state_store = state
    backend.toolkit_state_access = state.access(privileged_tools=("check_task",),
                                                privileged_keys=("privileged.task_check",))
    store = BlobStore(tmp_path, writable=True)
    interface = bind_runtime(runtime=backend, context=ctx, store=store, journal=Journal(tmp_path), bindings={},
                             completed_receipts={"old": PublicData.from_json({"action_id": "old"})},
                             submit=lambda action: calls.append(action) or {"accepted": True, "action_id": "a"})
    assert interface.observation().json()["data"]["observation"] == observed
    assert store.read(interface.proprioception().json()["data"]["proprioception"]) == (root / "proprioception.json").read_bytes()
    assert interface.receipt(action_id="old").json()["data"]["action_id"] == "old"
    assert interface.check_task().json()["data"]["task_success"] is False
    assert interface.probe_motion(request=dict(observed, schema="ora.motion-probe-request.v1", queries=[])).json()["status"] == "ok"
    assert interface.align_plan(request={}).json()["data"]["motion_planner_called"] is False
    assert interface.report_issue(issue={"message": "blocked"}).json()["status"] == "ok"
    assert interface.submit_action(action={"kind": "skill-sequence"}).json()["data"]["accepted"] is True
    with pytest.raises(AccessDenied, match="one action proposal per parent tool invocation"):
        interface.submit_action(action={"kind": "another"})
    assert len(calls) == 2


def test_snapshot_import_uses_pinned_log_not_appended_source(tmp_path):
    from ora.agent.toolkit.interface.snapshot import open_replay
    import shutil
    run = tmp_path / "source"
    shutil.copytree(FIXTURE, run)
    snapshot = json.loads((run / "snapshot.json").read_bytes())
    observation = snapshot["available_at_boundary"]["observation"]
    ctx = CallContext.create(run_id=snapshot["run_id"], round_index=0,
                             tool_call_id=snapshot["identity"]["tool_call_id"], boundary=snapshot["boundary"],
                             state={"observation": {"observation_id": observation["id"],
                                                     "observation_sha256": observation["manifest"]["sha256"]}}, check_task=True)
    runtime = Runtime(context=ctx, store=BlobStore(run, writable=True), journal=Journal(run), bindings={},
                      adapters={"check_task": lambda *args: PublicData.from_json({"task_success": False})})
    expected = runtime.check_task()
    cache = tmp_path / "cache"
    cache.mkdir()
    original = runtime.journal.path.read_bytes()
    (cache / "pinned.jsonl").write_bytes(original)
    snapshot["toolkit_io"] = dict(schema="ora.toolkit.snapshot-index.v1", snapshot_root=str(cache),
                                  calls=dict(path="pinned.jsonl", sha256=digest(original), bytes=len(original)),
                                  selected_tool_call_id=ctx.json()["tool_call_id"], selected_round_index=0,
                                  next_interface_ordinal=0, exact_interface_replay=True)
    runtime.check_task()  # Later source append must not change the fixed snapshot.
    recorded = open_replay(run_root=run, snapshot=snapshot, output_root=tmp_path / "output",
                           services={}, bindings={}, allow_check_task=True)
    assert recorded.check_task().payload == expected.payload
    with pytest.raises(NotRecorded):
        recorded.check_task()
    with pytest.raises(AccessDenied):
        open_replay(run_root=run, snapshot=snapshot, output_root=tmp_path / "denied",
                    services={}, bindings={}, allow_check_task=False)


def test_service_replay_reads_original_image_and_writes_new_masks(tmp_path):
    from ora.agent.toolkit.interface.replay import Replay
    obs = dict(observation_id="obs", observation_sha256="a" * 64)
    ctx = CallContext.create(run_id="run", round_index=0, tool_call_id="call", boundary=dict(kind="before_call", call=0),
                             state=dict(observation=obs))
    client = FakeSam3()
    service = sam3_adapters(client)
    source = BlobStore(tmp_path / "source", writable=True)
    rgb = source.put_array(np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
    runtime = Runtime(context=ctx, store=source, journal=Journal(source.root), adapters=service, bindings={})
    original = runtime.ground(rgb=rgb, text="cup")
    output = BlobStore(tmp_path / "output", writable=True)
    evaluation = Runtime(context=ctx, store=output, journal=Journal(output.root), adapters=service, bindings={})
    recorded = Replay(context=ctx, store=BlobStore(source.root), evaluation=evaluation, services=service, bindings={})
    result = recorded.ground(rgb=rgb, text="cup")
    assert len(client.calls) == 2 and result.payload == original.payload
    np.testing.assert_array_equal(client.calls[0][1], client.calls[1][1])
    mask = result.json()["data"]["candidates"][0]["mask"]
    assert (output.root / mask["path"]).is_file()
    assert result.comparison["status"] == "unavailable"
    assert result.comparison["reason"] == "service_identity_unknown"


def test_sam3_adapter_calls_actual_keyword_only_client_contract(tmp_path):
    from unittest.mock import create_autospec
    from ora.agent.model_services.clients import Sam3ApiSegmenter
    client = create_autospec(Sam3ApiSegmenter, instance=True)
    fake = FakeSam3()
    client.ground.side_effect = fake.ground
    client.segment.side_effect = fake.segment
    client.segment_points.side_effect = fake.segment_points
    store = BlobStore(tmp_path, writable=True)
    rgb = store.put_array(np.zeros((2, 3, 3), dtype=np.uint8))
    depth = store.put_array(np.ones((2, 3), dtype=np.float32))
    adapters = sam3_adapters(client)
    adapters["ground"](dict(text="cup", max_candidates=3), dict(rgb=rgb), store)
    adapters["segment"](dict(point_uv=[1, 1]), dict(rgb=rgb, depth=depth), store)
    adapters["segment_points"](dict(points_uv=[[1, 1]]), dict(rgb=rgb, depth=depth), store)
    assert len(fake.calls) == 3
