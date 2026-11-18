"""Different tools at an immutable recorded boundary, with exact missing requests."""

from contextlib import contextmanager
import json

import pytest

from ora.agent.context.observation.contact_points import project_policy_public_facts
from ora.agent.toolkit.interface.services import sam3_adapters
from ora.agent.toolkit.interface.types import canonical, digest
from tests.support.agent.toolkit_tools import install
from ora.orchestrator.toolkit.replay import experiment
from tests.support.agent.toolkit_atomic import complete_cameras
from tests.support.agent.toolkit_recording import proposal
from tests.support.agent.toolkit_tools import make_host


def tree(root):
    return {
        p.relative_to(root).as_posix(): digest(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def recorded_host(tmp_path, monkeypatch):
    host, client, _ = make_host(tmp_path / "run", monkeypatch)
    try:
        complete_cameras(host)
        host.public_facts_projector = project_policy_public_facts
        surface = host.surface
        frozen = dict(
            profile_id=surface.profile_id,
            definitions=list(surface.definitions),
            capabilities=list(surface.capabilities),
            view_config=dict(surface.view_config),
            result_schemas=dict(surface.result_schemas),
        )
        (host.control_root.parent / "plan.json").write_bytes(
            canonical(dict(policy={}, policy_composition=dict(tool_surface=frozen)))
        )
        install(host)
        args = dict(camera="head", text="object", max_candidates=1)
        parent = proposal(host, "vision_ground", args)
        grounded = host.invoke_with_context("vision_ground", args, call_context=parent)
        yield host, grounded
    finally:
        client.close()


def test_different_compute_and_observe_use_boundary_data(recorded_host, tmp_path):
    host, _ = recorded_host
    source = host.control_root.parent
    before = tree(source)
    computed = experiment(
        source=source,
        at="0",
        tool="compute",
        arguments=dict(expressions="answer = 6 * 7", outputs=["answer"]),
        output=tmp_path / "compute",
    )
    assert computed["status"] == "ok", computed
    assert computed["result"]["outputs"][0]["value"] == 42
    observed = experiment(
        source=source,
        at="0",
        tool="observe",
        arguments=dict(views=["head"]),
        output=tmp_path / "observe",
    )
    assert observed["status"] == "ok", observed.get("error")
    assert observed["result"]["success"] and observed["result"]["images_base64"]
    assert observed["replay_semantics"] == "alternative_tool"
    assert observed["next_interface_ordinal"] is None
    assert tree(source) == before


def test_changed_tool_requires_args_and_respects_frozen_surface(
    recorded_host, tmp_path
):
    host, _ = recorded_host
    source = host.control_root.parent
    result = experiment(
        source=source, at="0", tool="compute", output=tmp_path / "missing-args"
    )
    assert (
        result["error"]["type"] == "ValueError"
        and "--args" in result["error"]["message"]
    )
    plan = json.loads((source / "plan.json").read_bytes())
    plan["policy"]["tool_allowlist"] = ["vision_ground"]
    (source / "plan.json").write_bytes(canonical(plan))
    result = experiment(
        source=source,
        at="0",
        tool="compute",
        arguments={},
        output=tmp_path / "not-on-surface",
    )
    assert result["error"]["type"] == "AccessDenied"


def test_changed_motion_lists_exact_missing_ik(recorded_host, tmp_path):
    host, _ = recorded_host
    pose = host._current()[2]["proprioception"]["end_effectors"]["left"][
        "pose_xyz_xyzw"
    ]
    args = dict(targets=dict(left=dict(position_m=[pose[0] + 0.01, pose[1], pose[2]])))
    result = experiment(
        source=host.control_root.parent,
        at="0",
        tool="move_to",
        arguments=args,
        output=tmp_path / "motion",
    )
    assert result["error"]["type"] == "NotRecorded", result
    missing = result["missing_requests"]
    assert len(missing) == 1 and missing[0]["method"] == "reach_check"
    assert missing[0]["semantic_request_hash"] == digest(
        canonical(missing[0]["request"]["identity"])
    )
    assert missing[0]["request"]["identity"]["parameters"]["request"]["arm"] == "left"
    assert not result["missing_requests_complete"] and not result["proposed_actions"]


def test_alternative_motion_proposal_never_compares_or_writes_control(
    recorded_host, tmp_path
):
    host, _ = recorded_host
    source = host.control_root.parent
    # Explicit synthetic frozen no-IK capability, not a simulator fallback.
    path = source / "native/toolkit-io/tool-inputs.jsonl"
    row = json.loads(path.read_bytes())
    row["configuration"]["motion"]["reach_available"] = False
    path.write_bytes(canonical(row) + b"\n")
    pose = host._current()[2]["proprioception"]["end_effectors"]["left"][
        "pose_xyz_xyzw"
    ]
    args = dict(targets=dict(left=dict(position_m=[pose[0] + 0.01, pose[1], pose[2]])))
    before = tree(source)
    result = experiment(
        source=source,
        at="0",
        tool="move_to",
        arguments=args,
        output=tmp_path / "proposal",
    )
    assert result["status"] == "ok", result.get("error")
    assert len(result["proposed_actions"]) == 1
    action = result["proposed_actions"][0]
    assert action["action"]["action"] == "move_pose"
    assert action["executed"] is False and action["compared"] is False
    assert tree(source) == before
    assert not (tmp_path / "proposal/control/submitted-action.json").exists()


def test_alternative_ground_recomputes_from_compute_boundary(
    recorded_host, tmp_path, monkeypatch
):
    host, _ = recorded_host
    source = host.control_root.parent
    args = dict(expressions="value = 1", outputs=["value"])
    host.invoke_with_context(
        "compute", args, call_context=proposal(host, "compute", args, index=1)
    )
    requests = []
    original = host.grounding_backend.ground

    def ground(**kw):
        requests.append(kw["text"])
        return original(**kw)

    monkeypatch.setattr(host.grounding_backend, "ground", ground)

    @contextmanager
    def services(report):
        yield sam3_adapters(host.grounding_backend), {}

    monkeypatch.setattr("ora.orchestrator.toolkit.replay.service_session", services)
    before = tree(source)
    result = experiment(
        source=source,
        at="1",
        tool="vision_ground",
        arguments=dict(camera="head", text="changed request", max_candidates=1),
        output=tmp_path / "ground",
    )
    assert result["status"] == "ok", result.get("error")
    assert requests == ["changed request"]
    assert tree(source) == before


def test_check_override_is_explicit_but_never_backfills_final_verdict(
    recorded_host, tmp_path
):
    host, _ = recorded_host
    source = host.control_root.parent
    plan = json.loads((source / "plan.json").read_bytes())
    plan["policy"]["tool_allowlist"] = ["vision_ground"]
    (source / "plan.json").write_bytes(canonical(plan))
    # Rewrite capabilities consistently in the synthetic evidence and its keys.
    path = source / "native/toolkit-io/calls.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    for row in rows:
        row["request"]["identity"]["capabilities"]["check_task"] = False
        row["semantic_request_hash"] = digest(canonical(row["request"]["identity"]))
    path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    path = source / "native/toolkit-io/tool-inputs.jsonl"
    row = json.loads(path.read_bytes())
    row["context"]["check_task"] = False
    path.write_bytes(canonical(row) + b"\n")
    (source / "control/policy-task-check.json").write_bytes(
        canonical(dict(task_success=True, terminal=True))
    )
    before = tree(source)
    result = experiment(
        source=source,
        at="0",
        tool="check_task",
        arguments={},
        output=tmp_path / "check",
    )
    assert result["error"]["type"] == "NotRecorded", result
    assert result["capability_override"] == dict(
        check_task=True, original_policy_available=False, reason="原局策略不可用"
    )
    assert (
        result["context"]["check_task"] is False
    )  # The recorded context is not rewritten.
    assert result["missing_requests"][0]["method"] == "check_task"
    assert (
        result["missing_requests"][0]["request"]["identity"]["capabilities"][
            "check_task"
        ]
        is True
    )
    assert tree(source) == before


def test_check_at_finish_boundary_uses_only_recorded_verdict(recorded_host, tmp_path):
    host, _ = recorded_host
    source = host.control_root.parent
    (source / "control/policy-task-check.json").write_bytes(
        canonical(dict(task_success=True, terminal=True))
    )
    args = dict(
        outcome="success",
        reason="check_task confirms completion at this observation boundary",
    )
    host.invoke_with_context(
        "finish_episode",
        args,
        call_context=proposal(host, "finish_episode", args, index=1),
    )
    # The mutable final file must never override same-boundary recorded evidence.
    (source / "control/policy-task-check.json").write_bytes(
        canonical(dict(task_success=False, terminal=False))
    )
    result = experiment(
        source=source,
        at="1",
        tool="check_task",
        arguments={},
        output=tmp_path / "check-finish",
    )
    assert result["status"] == "ok", result.get("error")
    assert result["result"]["task_success"] is True
    assert result["capability_override"]["check_task"] is False


def test_progress_is_flushed_before_tool_runs_and_timings_are_explicit(
    recorded_host, tmp_path, monkeypatch
):
    host, _ = recorded_host
    output = tmp_path / "progress"
    from ora.agent.toolkit.tools.compute.tool import run as compute

    seen = []

    def inspected(api, arguments):
        rows = [
            json.loads(line)
            for line in (output / "progress.jsonl").read_bytes().splitlines()
        ]
        seen.extend(row["stage"] for row in rows)
        assert rows[-1]["stage"] == "run_tool"
        assert not (output / "result.json").exists()
        return compute(api, arguments)

    from ora.agent.toolkit.host.runtime.dispatch import TOOLS

    monkeypatch.setitem(TOOLS, "compute", inspected)
    result = experiment(
        source=host.control_root.parent,
        at="0",
        tool="compute",
        arguments=dict(expressions="answer = 42", outputs=["answer"]),
        output=output,
    )
    assert result["status"] == "ok", result.get("error")
    assert seen == [
        "resolve",
        "boot_worker",
        "replay_controls",
        "verify_state",
        "run_tool",
    ]
    rows = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_bytes().splitlines()
    ]
    assert rows[-1]["stage"] == "done"
    assert set(result["timing"]) == {
        "resolve",
        "boot_worker",
        "replay_controls",
        "verify_state",
        "run_tool",
        "compare",
        "done",
    }
    assert all(
        type(seconds) is float and seconds >= 0 for seconds in result["timing"].values()
    )
    for row in rows:
        assert set(row) == {"t", "stage", "done", "total", "message"}
        if row["stage"] in {"boot_worker", "replay_controls", "verify_state"}:
            assert row["done"] == row["total"] == 0 and "Skipped" in row["message"]
            assert result["timing"][row["stage"]] == 0
    assert result["mode"] == result["evidence_mode"] == "recorded"


def test_resolve_error_has_progress_result_and_missing_boundary(
    recorded_host, tmp_path
):
    host, _ = recorded_host
    output = tmp_path / "bad-call"
    result = experiment(
        source=host.control_root.parent,
        at="999",
        tool="compute",
        arguments={},
        output=output,
    )
    assert result["error"]["type"] == "NotRecorded"
    assert result["missing_requests"][0]["kind"] == "boundary"
    assert result["missing_requests"][0]["method"] == "replay_boundary"
    rows = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_bytes().splitlines()
    ]
    assert [row["stage"] for row in rows] == ["resolve", "done"]
    assert rows[-1]["message"] == "error"
    assert json.loads((output / "result.json").read_bytes()) == result


def test_model_visible_uses_runtime_projection_and_compares_artifact(
    recorded_host, tmp_path
):
    host, _ = recorded_host
    source = host.control_root.parent
    plan = json.loads((source / "plan.json").read_bytes())
    plan["policy"]["context_composition"] = "native_slots_v3"
    (source / "plan.json").write_bytes(canonical(plan))
    args = dict(expressions="answer = 42", outputs=["answer"])
    parent = proposal(host, "compute", args, index=1)
    raw = host.invoke_with_context("compute", args, call_context=parent)
    # Fixture event uses the same CAS writer and runtime projection as a Pi call.
    from ora.orchestrator.toolkit.visible import Projection
    from ora.agent.pi.hints import StaticFactsState, apply_hints
    from ora.agent.evidence.canonical import ArtifactStore
    from types import SimpleNamespace

    projector = Projection()
    projector.tools = SimpleNamespace(surface=host.surface)
    projector.context_composition = "native_slots_v3"
    projector.supports_images = True
    projector.artifacts = ArtifactStore(source)
    projected = projector._project(
        "compute", raw, call_id=parent["tool_call_id"], step=0
    )
    stored, _ = apply_hints(
        projected["canonical"], projected["canonical_hints"], StaticFactsState()
    )
    ref = projector.artifacts.put_json(stored, kind="native_tool_result")
    events_path = source / "agent-events.jsonl"
    events = [json.loads(line) for line in events_path.read_bytes().splitlines()]
    event = dict(
        seq=events[-1]["seq"] + 1,
        type="tool_result",
        tool_call_id=parent["tool_call_id"],
        artifact_ref=ref,
    )
    with events_path.open("ab") as stream:
        stream.write(canonical(event) + b"\n")
    before = tree(source)
    output = tmp_path / "visible"
    result = experiment(source=source, at="1", tool="compute", output=output)
    assert result["status"] == "ok", result.get("error")
    visible = result["model_visible"]
    expected, _ = apply_hints(
        projected["visible"], projected["visible_hints"], StaticFactsState()
    )
    assert visible["text"] == json.dumps(expected, ensure_ascii=False, sort_keys=True)
    assert visible["sha256"] == digest(visible["text"].encode())
    assert (output / visible["artifact_ref"]["path"]).read_bytes() == visible[
        "text"
    ].encode()
    assert visible["comparison"]["artifact"]["byte_equal"] is True
    assert visible["comparison"]["visible"]["content_equal"] is True
    assert tree(source) == before


def test_model_visible_volatile_filter_does_not_edit_compute_output():
    from ora.orchestrator.toolkit.visible import compare

    left = canonical(dict(diagnostics=dict(request_id="first", model="fixed"), score=1))
    right = canonical(
        dict(diagnostics=dict(request_id="second", model="fixed"), score=1)
    )
    assert compare(left, right, "grasp_plan")["content_equal"] is True
    assert compare(left, right, "grasp_plan")["byte_equal"] is False
    assert compare(left, right, "compute")["content_equal"] is False
    changed = canonical(
        dict(diagnostics=dict(request_id="second", model="changed"), score=1)
    )
    assert compare(left, changed, "grasp_plan")["content_equal"] is False


@pytest.mark.parametrize(
    "tool,index,branch",
    [
        ("vision_ground", "0", ("diagnostics",)),
        ("grasp_plan", "1", ("diagnostics", "learned")),
    ],
)
def test_recomputed_perception_ignores_service_timings(
    recorded_host, tmp_path, monkeypatch, tool, index, branch
):
    from ora.agent.evidence.canonical import ArtifactStore
    from ora.agent.toolkit.interface.services import grasp_adapter
    from ora.orchestrator.toolkit.visible import project
    from ora.orchestrator.toolkit.selection import select

    host, grounded = recorded_host
    source = host.control_root.parent
    plan_path = source / "plan.json"
    plan = json.loads(plan_path.read_bytes())
    plan["policy"]["context_composition"] = "native_slots_v3"
    plan_path.write_bytes(canonical(plan))
    original = grounded
    if tool == "grasp_plan":
        args = dict(
            target_ref=grounded["candidates"][0]["target_ref"],
            arm="left",
            max_candidates=3,
        )
        parent = proposal(host, tool, args, index=1)
        original = host.invoke_with_context(tool, args, call_context=parent)
    selected, events, _ = select(source, index)
    projection_root = tmp_path / "original-projection"
    projection_root.mkdir()
    projected = project(
        source, projection_root, selected, events, tool, original, same_tool=False
    )
    ref = ArtifactStore(source).put_bytes(
        (projection_root / projected["stored_projection"]["path"]).read_bytes(),
        kind="native_tool_result",
        suffix=".json",
        media_type="application/json",
    )
    with (source / "agent-events.jsonl").open("ab") as stream:
        stream.write(
            canonical(
                dict(
                    seq=events[-1]["seq"] + 1,
                    type="tool_result",
                    tool_call_id=selected["tool_call_id"],
                    artifact_ref=ref,
                )
            )
            + b"\n"
        )
    calls = [
        json.loads(line)
        for line in (source / "native/toolkit-io/calls.jsonl").read_bytes().splitlines()
    ]
    bindings = {
        row["request"]["identity"]["method"]: row["request"]["identity"]["binding"]
        for row in calls
    }
    requests = []

    def changed(metadata):
        return dict(
            metadata,
            request_id="fresh-request",
            latency_ms=dict(
                roundtrip=81.0,
                service_inference=65.0,
                service_queue=10.0,
                service_total=75.0,
            ),
            timings=dict(total=75.0),
            queue=dict(wait=10.0),
        )

    ground = host.grounding_backend.ground

    def reground(**kw):
        requests.append(kw)
        candidates, metadata = ground(**kw)
        return candidates, changed(metadata)

    monkeypatch.setattr(host.grounding_backend, "ground", reground)

    @contextmanager
    def services(report):
        def propose(p, scene, target):
            requests.append(p)
            candidates, metadata = host.grasp_backend.contact_backend.propose(
                scene_points_camera_m=scene,
                target_points_camera_m=target,
                camera_to_world=p["camera_to_world"],
                target_ref="toolkit-raw-pool",
                workspace_bounds_m=p["workspace_bounds_m"],
                max_candidates=p["max_candidates"],
            )
            return candidates, changed(metadata)

        yield dict(
            **sam3_adapters(host.grounding_backend),
            grasp_candidates=grasp_adapter(propose),
        ), bindings

    monkeypatch.setattr("ora.orchestrator.toolkit.replay.service_session", services)
    before = tree(source)
    result = experiment(
        source=source, at=index, tool=tool, output=tmp_path / "recomputed"
    )
    assert result["status"] == "ok", result.get("error")
    assert len(requests) == 1
    diagnostics = result["result"]
    for key in branch:
        diagnostics = diagnostics[key]
    assert diagnostics["latency_ms"]["roundtrip"] == 81.0
    for domain in ("artifact", "visible"):
        comparison = result["model_visible"]["comparison"][domain]
        assert comparison["byte_equal"] is False
        assert comparison["content_equal"] is True
        for field in ("request_id", "latency_ms", "timings", "queue"):
            assert "/" + "/".join((*branch, field)) in comparison["volatile_paths"]
    assert tree(source) == before
