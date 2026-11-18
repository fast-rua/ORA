"""CPU-only live workbench tests; no simulator, lease or network is used."""

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from ora.agent.toolkit.interface.types import ActionNotExecuted, canonical, decode, digest, Divergence
from tests.support.agent.toolkit_tools import install, V5_GRANTS
from ora.orchestrator.backends.registry import backend, _LOADED
from ora.orchestrator.toolkit.live_api import Live
from ora.orchestrator.toolkit.replay import experiment, live_state
from ora.orchestrator.toolkit.selection import select
from ora.orchestrator.toolkit.live_boundary import freeze
from tests.support.agent.toolkit_atomic import complete_cameras
from tests.support.agent.toolkit_recording import proposal
from tests.support.agent.toolkit_tools import make_host


def hashes(root):
    return {
        p.relative_to(root).as_posix(): digest(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    host, client, _ = make_host(tmp_path / "source", monkeypatch)
    complete_cameras(host)
    source = host.control_root.parent
    control, root, observation = host._current()
    surface = host.surface
    from ora.agent.memory.artifacts import LOSSLESS_CONTEXT_RETENTION
    from ora.orchestrator.spec.composition import _advertised_tool_surface

    _, frozen = _advertised_tool_surface(
        dict(context_retention=LOSSLESS_CONTEXT_RETENTION), surface, {}, root=source
    )
    plan = dict(
        runtime=dict(kind="robodojo", workspace_bounds_m=dict(host.workspace_bounds_m)),
        policy=dict(context_composition="native_slots_v3"),
        policy_composition=dict(tool_surface=frozen),
        toolkit_grants=V5_GRANTS,
    )
    (source / "plan.json").write_bytes(canonical(plan))
    manifest = dict(
        path=str((root / "observation.json").relative_to(source)),
        sha256=control["observation_sha256"],
    )
    state = dict(
        step=0,
        observation=dict(id=observation["observation_id"], manifest=manifest),
        proprioception=observation["proprioception"],
    )
    state_path = source / "policy-state/steps/step-0000.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_bytes(canonical(state))
    events = [
        dict(
            seq=1,
            type="fresh_observation_published",
            run_id="run",
            payload=dict(observation_id=observation["observation_id"]),
        )
    ]
    events[0]["payload"]["state_ref"] = dict(
        path="policy-state/steps/step-0000.json", sha256=digest(state_path.read_bytes())
    )
    (source / "agent-events.jsonl").write_bytes(
        b"".join(canonical(e) + b"\n" for e in events)
    )
    native = source / "native"
    native.mkdir(exist_ok=True)
    (native / "observation-clocks.jsonl").write_bytes(
        canonical(
            dict(
                observation_sequence=observation["sequence"],
                before_physics_steps=10,
                after_physics_steps=10,
                read_only=True,
            )
        )
        + b"\n"
    )
    (native / "reset.json").write_bytes(canonical(dict(physics_steps_before_policy=10)))
    (native / "source-manifest.json").write_bytes(canonical(dict(source_revisions={})))
    install(host)
    args = dict(camera="head", text="object", max_candidates=1)
    host.invoke_with_context(
        "vision_ground", args, call_context=proposal(host, "vision_ground", args)
    )
    try:
        yield host
    finally:
        client.close()


class FakeSession:
    def __init__(self, host, output, boundary):
        self.host, self.boundary = host, boundary
        ref = boundary["snapshot"]["available_at_boundary"]["observation"]["manifest"]
        self.observation_root = (host.control_root.parent / ref["path"]).parent
        self.manifest = json.loads(
            (self.observation_root / "observation.json").read_bytes()
        )
        self.output_root = output / "live-worker/environment"
        self.output_root.mkdir(parents=True)
        self.calls = []
        self.actions = []
        self.terminal = False

    def diagnostic(self, method, **params):
        self.calls.append(method)
        if method == "prepare":
            root, manifest = self.observation_root, self.manifest
            target = Path(params["destination"]) / manifest["observation_id"]
            shutil.copytree(root, target)
            return dict(observation=manifest, images=[])
        if method in {"verify_state", "state"}:
            return self.boundary["expected"]
        if method == "authorize":
            return dict(verified=True)
        raise AssertionError(method)

    def align_plan(self, request):
        self.calls.append("align_plan")
        assert "authorize" in self.calls
        return self.host.align_plan_callback(request)

    def probe_motion(self, request):
        self.calls.append("probe_motion")
        assert "authorize" in self.calls
        return self.host.motion_probe_callback(request)

    def status(self):
        self.calls.append("status")
        assert "authorize" in self.calls
        return dict(
            schema="ora.environment-status.v1",
            terminal=self.terminal,
            phase="terminal" if self.terminal else "running",
            reason="official_success" if self.terminal else None,
            source="fixture",
            policy_visibility="control_only",
            signals=[],
        )

    def execute(self, action):
        self.calls.append("execute")
        self.actions.append(action)
        self.terminal = True
        return dict(
            receipt=dict(
                action_id=action["action_id"], consumed=True, execution_status="reached"
            )
        )

    def observe(self, destination):
        self.calls.append("observe")
        root, manifest = self.observation_root, self.manifest
        target = destination / manifest["observation_id"]
        shutil.copytree(root, target)
        return manifest, []


@pytest.fixture
def fake_live(recorded, monkeypatch):
    sessions = []

    @contextmanager
    def factory(**kw):
        session = FakeSession(recorded, kw["output"], kw["boundary"])
        sessions.append(session)
        try:
            for stage in ("boot_worker", "replay_controls", "verify_state"):
                kw["progress"].start(stage, "fake CPU worker")
            yield session
        finally:
            session.calls.append("close")

    monkeypatch.setitem(
        _LOADED, "robodojo", replace(backend("robodojo"), toolkit_live=factory)
    )
    return recorded, sessions


def test_live_check_and_new_ik_use_restored_worker(fake_live, tmp_path):
    host, sessions = fake_live
    source = host.control_root.parent
    before = hashes(source)
    result = experiment(
        source=source,
        at="0",
        tool="check_task",
        arguments={},
        live=True,
        resource_profile="fake",
        output=tmp_path / "check",
    )
    assert result["status"] == "ok", result.get("error")
    assert result["mode"] == "live" and result["evidence_mode"] == "recorded"
    assert result["result"]["task_success"] is False
    assert "tool_surface_mismatch" in [warning.get("code") for warning in result["warnings"]]
    audit = json.loads((tmp_path / "check/surface-compatibility.json").read_bytes())
    assert audit["blocking"] is False
    assert {"artifact_read", "observe"} <= set(audit["affected_tools"])
    assert any(d["path"].startswith("/definitions/") for d in audit["differences"])
    assert any(d["path"].startswith("/result_schemas/") for d in audit["differences"])
    assert sessions[-1].calls == [
        "prepare",
        "verify_state",
        "authorize",
        "status",
        "close",
    ]
    pose = host._current()[2]["proprioception"]["end_effectors"]["left"][
        "pose_xyz_xyzw"
    ]
    result = experiment(
        source=source,
        at="0",
        tool="move_to",
        arguments=dict(
            targets=dict(left=dict(position_m=[pose[0] + 0.01, *pose[1:3]]))
        ),
        live=True,
        resource_profile="fake",
        output=tmp_path / "move",
    )
    assert result["status"] == "ok", result.get("error")
    assert len(result["proposed_actions"]) == 1
    assert "align_plan" in sessions[-1].calls
    assert sessions[-1].actions == []
    assert hashes(source) == before


def test_live_explicit_execution_reports_post_state(fake_live, tmp_path):
    host, sessions = fake_live
    pose = host._current()[2]["proprioception"]["end_effectors"]["left"][
        "pose_xyz_xyzw"
    ]
    result = experiment(
        source=host.control_root.parent,
        at="0",
        tool="move_to",
        arguments=dict(
            targets=dict(left=dict(position_m=[pose[0] + 0.01, *pose[1:3]]))
        ),
        live=True,
        resource_profile="fake",
        execute_action=True,
        output=tmp_path / "execute",
    )
    assert result["status"] == "ok", result.get("error")
    assert len(sessions[-1].actions) == 1
    assert result["execution"]["executed"] is True
    assert result["execution"]["check_task"]["task_success"] is True
    assert result["execution"]["state"]["physics_clock"] == 10
    assert sessions[-1].calls[-1] == "close"


def test_sensor_difference_is_reported_but_tools_use_original_bytes(
    fake_live, tmp_path, monkeypatch
):
    host, sessions = fake_live
    original = FakeSession.diagnostic

    def changed(self, method, **params):
        result = original(self, method, **params)
        if method == "prepare":
            result = decode_copy(result)
            from PIL import Image
            import numpy as np

            manifest = result["observation"]
            ref = manifest["cameras"]["head"]["files"]["rgb_lossless_png"]
            path = (
                Path(params["destination"]) / manifest["observation_id"] / ref["path"]
            )
            with Image.open(path) as image:
                pixels = np.array(image)
            pixels[0, 0] = (pixels[0, 0].astype(int) + 32) % 256
            Image.fromarray(pixels).save(path)
            ref.update(sha256=digest(path.read_bytes()), bytes=path.stat().st_size)
        return result

    monkeypatch.setattr(FakeSession, "diagnostic", changed)
    result = experiment(
        source=host.control_root.parent,
        at="0",
        tool="check_task",
        arguments={},
        live=True,
        resource_profile="fake",
        output=tmp_path / "bad",
    )
    assert result["status"] == "ok", result.get("error")
    report = result["sensor_report"]
    assert report["blocking"] is False and report["passed"] is False
    assert report["warning"] == "未逐字节复现的传感器"
    assert report["tool_observation_source"] == "recorded_publication"
    rgb = next(
        row
        for row in report["files"]
        if row["camera"] == "head" and row["role"] == "rgb_lossless_png"
    )
    assert rgb["metrics"]["fraction_pixels_gt_16"] > 0
    assert rgb["metrics"]["psnr_db"] > 0
    original_root = host._current()[1]
    copied = tmp_path / "bad/policy-observations" / original_root.name
    assert (copied / "head/rgb.png").read_bytes() == (
        original_root / "head/rgb.png"
    ).read_bytes()
    assert sessions[-1].calls == [
        "prepare",
        "verify_state",
        "authorize",
        "status",
        "close",
    ]


def decode_copy(value):
    return json.loads(json.dumps(value))


@pytest.mark.parametrize(
    "change",
    ["clock", "duplicate_clock", "publication", "pending_action", "missing_gt"],
)
def test_boundary_refuses_inexact_evidence(recorded, tmp_path, change):
    source = recorded.control_root.parent
    selected, events, _ = select(source, "0")
    if change in {"clock", "duplicate_clock"}:
        path = source / "native/observation-clocks.jsonl"
        row = json.loads(path.read_bytes())
        if change == "clock":
            row["after_physics_steps"] += 1
        path.write_bytes(
            (canonical(row) + b"\n") * (2 if change == "duplicate_clock" else 1)
        )
    elif change == "publication":
        events[0]["payload"]["state_ref"]["sha256"] = "0" * 64
    elif change == "pending_action":
        selected["proposal"]["seq"] += 1
        events.insert(1, dict(seq=2, type="action_proposed", payload={}))
    else:
        path = source / "plan.json"
        plan = json.loads(path.read_bytes())
        plan["ground_truth_capture"] = True
        path.write_bytes(canonical(plan))
    output = tmp_path / "freeze"
    output.mkdir()
    with pytest.raises((ValueError, RuntimeError)):
        freeze(source, output, selected, events)


@pytest.mark.parametrize(
    "kwargs",
    [dict(live=True), dict(execute_action=True), dict(resource_profile="fake")],
)
def test_live_flags_require_explicit_mode_and_lane(recorded, tmp_path, kwargs):
    result = experiment(
        source=recorded.control_root.parent,
        at="0",
        tool="check_task",
        arguments={},
        output=tmp_path / "flags",
        **kwargs,
    )
    assert result["error"]["type"] == "ValueError"


def test_live_perception_recomputes_and_records_comparison(
    fake_live, tmp_path, monkeypatch
):
    from ora.agent.toolkit.interface.services import sam3_adapters

    host, sessions = fake_live
    calls = []
    ground = host.grounding_backend.ground

    def tracked(**kw):
        calls.append(kw)
        return ground(**kw)

    monkeypatch.setattr(host.grounding_backend, "ground", tracked)
    entries = [
        json.loads(l)
        for l in (host.control_root.parent / "native/toolkit-io/calls.jsonl")
        .read_bytes()
        .splitlines()
    ]
    bindings = {
        row["request"]["identity"]["method"]: row["request"]["identity"]["binding"]
        for row in entries
    }

    @contextmanager
    def services(report):
        yield sam3_adapters(host.grounding_backend), bindings

    monkeypatch.setattr("ora.orchestrator.toolkit.replay.service_session", services)
    result = experiment(
        source=host.control_root.parent,
        at="0",
        tool="vision_ground",
        live=True,
        resource_profile="fake",
        output=tmp_path / "perception",
    )
    assert result["status"] == "ok", result.get("error")
    assert len(calls) == 1 and result["comparison_count"] == 1
    comparison = json.loads(
        (tmp_path / "perception" / result["comparisons"]).read_bytes()
    )
    content = json.loads(
        (tmp_path / "perception" / comparison["reference"]["path"]).read_bytes()
    )
    assert content["status"] == "equal"


def test_live_skill_uses_existing_stage_driver(fake_live, tmp_path):
    host, sessions = fake_live

    def probe(request):
        return dict(
            schema="ora.motion-probe-result.v1",
            observation_id=request["observation_id"],
            world_mutation_sequence=0,
            variant_policy=request["variant_policy"],
            robot_decisions_consumed=0,
            world_state_changed=False,
            results=[
                dict(query_id=q["query_id"], arm=q["arm"], reachable=True, variants=[])
                for q in request["queries"]
            ],
        )

    host.motion_probe_callback = probe
    args = dict(camera="head", text="object", max_candidates=1)
    ground = host.invoke_with_context(
        "vision_ground",
        args,
        call_context=proposal(host, "vision_ground", args, index=1),
    )
    args = dict(
        target_ref=ground["candidates"][0]["target_ref"], arm="left", max_candidates=3
    )
    planned = host.invoke_with_context(
        "grasp_plan", args, call_context=proposal(host, "grasp_plan", args, index=2)
    )
    args = dict(
        candidate_id=planned["candidates"][0]["candidate_id"],
        rationale="Use the inspected candidate and its verified approach path.",
    )
    host.invoke_with_context(
        "grasp_execute",
        args,
        call_context=proposal(host, "grasp_execute", args, index=3),
    )
    before = hashes(host.control_root.parent)
    result = experiment(
        source=host.control_root.parent,
        at="3",
        tool="grasp_execute",
        live=True,
        resource_profile="fake",
        execute_action=True,
        output=tmp_path / "skill",
    )
    assert result["status"] == "ok", result.get("error")
    assert result["proposed_actions"][0]["action"]["action"] == "execute_skill_sequence"
    assert result["execution"]["executed"] is True
    assert (
        len(sessions[-1].actions) == 1
    )  # Fake terminal after first stage; executor stops the sequence.
    assert ":00-" in sessions[-1].actions[0]["action_id"]
    assert hashes(host.control_root.parent) == before


def test_live_action_failure_preserves_post_attempt_evidence(
    fake_live, tmp_path, monkeypatch
):
    host, sessions = fake_live

    def failed(self, action):
        self.terminal = True
        raise RuntimeError("uncertain execution")

    monkeypatch.setattr(FakeSession, "execute", failed)
    pose = host._current()[2]["proprioception"]["end_effectors"]["left"][
        "pose_xyz_xyzw"
    ]
    result = experiment(
        source=host.control_root.parent,
        at="0",
        tool="move_to",
        arguments=dict(
            targets=dict(left=dict(position_m=[pose[0] + 0.01, *pose[1:3]]))
        ),
        live=True,
        resource_profile="fake",
        execute_action=True,
        output=tmp_path / "failed-action",
    )
    assert result["status"] == "error"
    assert result["execution"]["attempted"] and result["execution"]["executed"] is None
    assert result["execution"]["check_task"]["task_success"] is True
    assert result["execution"]["state"]["physics_clock"] == 10
    assert sessions[-1].calls[-1] == "close"


def test_live_script_second_proposal_is_refused_before_any_execution(
    fake_live, tmp_path
):
    host, sessions = fake_live
    script = tmp_path / "two.py"
    script.write_text(
        """from ora.agent.toolkit.tools.move_to.tool import run as move
def run(api,args):
    move(api,args)
    return move(api,args)
"""
    )
    pose = host._current()[2]["proprioception"]["end_effectors"]["left"][
        "pose_xyz_xyzw"
    ]
    result = experiment(
        source=host.control_root.parent,
        at="0",
        script=script,
        arguments=dict(
            targets=dict(left=dict(position_m=[pose[0] + 0.01, *pose[1:3]]))
        ),
        live=True,
        resource_profile="fake",
        execute_action=True,
        output=tmp_path / "double",
    )
    assert result["status"] == "error"
    assert len(result["proposed_actions"]) == 1 and sessions[-1].actions == []


def test_live_receipt_keeps_public_keyword_api(fake_live, tmp_path):
    host, sessions = fake_live
    script = tmp_path / "receipt.py"
    script.write_text(
        "def run(api, args):\n" "    return api.state.receipt(action_id='missing').json()\n"
    )
    result = experiment(
        source=host.control_root.parent,
        at="0",
        script=script,
        arguments={},
        live=True,
        resource_profile="fake",
        output=tmp_path / "receipt-output",
    )
    assert result["status"] == "ok", result.get("error")
    assert result["result"]["status"] == "error"
    assert result["result"]["method"] == "receipt"
    assert result["result"]["error"] == dict(code="backend_failure", retryable=False)
    assert sessions[-1].actions == []


@pytest.mark.parametrize("failure", ["capture", "physical"])
def test_sensor_diagnostics_do_not_relax_physical_gate(
    fake_live, tmp_path, monkeypatch, failure
):
    host, sessions = fake_live
    original = FakeSession.diagnostic

    def changed(self, method, **params):
        if method == "prepare" and failure == "capture":
            self.calls.append(method)
            raise RuntimeError("renderer unavailable")
        result = original(self, method, **params)
        if method == "verify_state" and failure == "physical":
            result = decode_copy(result)
            result["physics_clock"] += 1
        return result

    monkeypatch.setattr(FakeSession, "diagnostic", changed)
    result = experiment(
        source=host.control_root.parent,
        at="0",
        tool="check_task",
        arguments={},
        live=True,
        resource_profile="fake",
        output=tmp_path / failure,
    )
    if failure == "capture":
        assert result["status"] == "ok", result.get("error")
        assert result["sensor_report"]["status"] == "unavailable"
        assert result["sensor_report"]["passed"] is False
    else:
        assert result["error"]["type"] == "Divergence"
        assert "authorize" not in sessions[-1].calls
        assert "status" not in sessions[-1].calls


def test_corrupt_frozen_surface_is_not_a_version_warning(fake_live, tmp_path):
    host, sessions = fake_live
    source = host.control_root.parent
    path = source / "plan.json"
    plan = json.loads(path.read_bytes())
    # A changed artifact that still claims the original digest is corrupt evidence.
    plan["policy_composition"]["tool_surface"]["definitions"][0][
        "description"
    ] += "tampered"
    path.write_bytes(canonical(plan))
    result = experiment(
        source=source,
        at="0",
        tool="check_task",
        arguments={},
        live=True,
        resource_profile="fake",
        output=tmp_path / "corrupt",
    )
    assert result["status"] == "error"
    assert result["error"]["type"] == "CorruptRecord"
    assert "hash differs" in result["error"]["message"]
    assert "status" not in sessions[-1].calls
    assert sessions[-1].calls[-1] == "close"


class _SynchronousSession:
    """What ``Live.execute_now`` touches of a restored worker."""

    def __init__(self):
        self.calls, self.limits, self.actions = [], [], []
        self.execution_state_owner = self

    def diagnostic(self, method, **params):
        self.calls.append((method, params))
        return dict(objects=dict(block=dict(position_m=[0.0, 0.0, 0.0], orientation=[1.0, 0.0, 0.0, 0.0])))

    @contextmanager
    def limited_action(self, budget, tool_call):
        self.limits.append((budget, tool_call))
        yield

    def execute(self, action):
        self.actions.append(action)
        return dict(receipt=dict(action_id=action["action_id"], native_actions=7, execution_status="reached",
                                 action="set_gripper"))


def _synchronous_live(*, requested):
    live = Live.__new__(Live)
    live.session, live.execute_requested = _SynchronousSession(), requested
    live.executed, live.objects_before = [], None
    live.context = SimpleNamespace(json=lambda: dict(tool_call_id="call-1"))
    return live


GRIPPER_ACTION = dict(schema="ora-motion-action.v3", observation_id="o", action_id="action-1", action="set_gripper",
                      arguments=dict(targets=dict(right="open")), bound_observation_sha256="s")


def test_a_synchronous_action_runs_under_its_budget_and_returns_the_episode_envelope():
    live = _synchronous_live(requested=True)
    answer = decode(live.execute_now(dict(action=GRIPPER_ACTION, budget=20), None, None).data)
    assert live.session.limits == [(20, dict(tool_call_id="call-1", tool="live_diagnostic"))]
    assert len(live.session.actions) == 1 and answer["remaining_budget"] == 13
    assert answer["receipt"]["native_actions"] == 7 and answer["receipt"]["action_id"] == "action-1"
    assert live.session.calls == [("state", dict(objects=True))]  # the scene before the action, for the report
    assert list(live.objects_before) == ["block"]


def test_a_synchronous_action_without_the_flag_does_not_run():
    live = _synchronous_live(requested=False)
    with pytest.raises(ActionNotExecuted):
        live.execute_now(dict(action=GRIPPER_ACTION, budget=20), None, None)
    assert live.session.actions == [] and live.session.limits == []


def test_a_live_tool_gets_empty_holding_intentions_it_can_stage():
    state = live_state()
    assert state.read("tool.held.left") is None and state.read("tool.marks") == {}
    state.write("tool.held.left", dict(source="grasp"))
    assert state.read("tool.held.left") == dict(source="grasp")


def test_a_host_on_another_stack_replaces_the_recorded_routes(tmp_path, monkeypatch):
    from ora.orchestrator.toolkit import live_host

    host, _, _ = make_host(tmp_path / "source", monkeypatch)
    complete_cameras(host)
    host.motion_planner_available = True
    recorded = dict(surface_id="old", legacy_visible=True, synchronous_execution=False, inputs_kept=1,
                    motion=dict(global_planner=False, joint_trajectory=False, capabilities=[]))
    fresh = live_host.current_motion(recorded, host)
    assert fresh["surface_id"] == host.surface.profile_id
    assert fresh["motion"]["global_planner"] is True
    assert fresh["motion"]["capabilities"] == sorted(host.surface.capabilities)
    assert fresh["synchronous_execution"] == host.surface.supports("toolkit_synchronous_execution_v1")
    assert ("legacy_visible" in fresh) == host.surface.supports("toolkit_legacy_projection_v1")
    assert fresh["inputs_kept"] == 1 and recorded["motion"]["global_planner"] is False
