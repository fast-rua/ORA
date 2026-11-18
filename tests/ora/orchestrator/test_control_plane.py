"""Control-plane, output ownership, experiment and viewer contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ora.agent.composition.baseline import default_baseline_id, load_native_baseline
from ora.app import plan as plan_command
from ora.orchestrator import launch, leases
from ora.app.main import build_parser
from ora.orchestrator.backends.libero_pro.preflight import (
    _libero_controller_contract_check,
    _offscreen_graphics_runtime,
)
from ora.orchestrator.preflight.gates import (
    SERVICE_SET_BINDING_SCHEMA,
    _provider_marker,
    _resident_service_prerequisite,
    run_preflight,
)
from ora.environment import graphics_probe
from ora.environment.backends.libero_pro.session import (
    libero_osc_action_contract,
)
from ora.environment.replay_worker import (
    ReplayWorkerError,
    _FrameWriter,
    _apply_robotwin_control_target,
    _capture_steps,
    _needs_final_capture,
    _uniform_capture_steps,
    _libero_qualified_controller_binding,
    _recorded_control_step_count,
    _robotwin_replay_needs_planner,
    _robotwin_replay_route,
)
from ora.orchestrator.evaluation.evaluate import EvaluationError
from ora.orchestrator.spec import quickstart
from ora.orchestrator.spec.inputs import provider_policy
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from ora.orchestrator.evaluation.context_manifest import context_placeholder
from viewer import app as viewer
from viewer.libero_pro_catalog import (
    CATALOG_SCHEMA as LIBERO_PRO_CATALOG_SCHEMA,
    LIBERO_PRO_CODE_REVISION,
    LIBERO_PRO_DATA_REVISION,
    MIGRATED_FROM_COMMIT as LIBERO_MIGRATED_FROM_COMMIT,
    ROBUSTNESS_CASES as LIBERO_ROBUSTNESS_CASES,
    TASKS as LIBERO_PRO_TASKS,
)
from viewer.libero_previews import (
    EXPECTED_PREVIEWS as LIBERO_EXPECTED_PREVIEWS,
    PREVIEW_SCHEMA as LIBERO_PREVIEW_SCHEMA,
    RENDER_CONFIG as LIBERO_RENDER_CONFIG,
    load_preview_manifest,
    preview_descriptors,
)
from viewer.robotwin_catalog import (
    CATALOG_SCHEMA,
    MIGRATED_FROM_COMMIT,
    ROBOTWIN_REVISION,
    TASKS,
)
from viewer.registry import discover_run_roots, register_run, visible_runs
from viewer.trajectory import (
    ANALYSIS_SCHEMA,
    _controller_replay_source,
    _continuous_execution_source,
    _load_run_analysis,
    _render_continuous_execution_frame,
    _render_trajectory_frame,
    compile_raw_trajectory,
    compile_trajectory,
    compile_trajectory_video,
    raw_trajectory_event,
)
from viewer.replay import (
    ViewerReplayError,
    ensure_controller_replay,
)
from tests.support.orchestrator.control_plane import ROOT


def init_project(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(["git", "-C", tmp_path, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", tmp_path, "config", "user.name", "Test"], check=True)
    (tmp_path / "prompt.md").write_text("prompt v1")
    (tmp_path / "manual.md").write_text("## overview\n\nmanual v1\n")
    (tmp_path / "agents.md").write_text("agent rules v1")
    subprocess.run(["git", "-C", tmp_path, "add", "."], check=True)
    subprocess.run(["git", "-C", tmp_path, "commit", "-qm", "initial"], check=True)
    return tmp_path


def spec() -> dict:
    return {
        "schema": "ora.run-spec.v1",
        "experiment": "runtime-v1", "config": "minimal", "milestone": "M0",
        "output_workspace": "outputs/runtime-v1",
        "task": "synthetic", "task_instruction": "Move once.", "seed": 0,
        "evidence_tier": "diagnostic",
        "policy": {"kind": "scripted", "model": "scripted", "provider": "trusted", "reasoning_effort": "none", "actions": [{"x": 1}]},
        "runtime": {"kind": "synthetic", "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}},
        "budgets": {"max_robot_decisions": 1, "max_model_calls": 1, "max_sim_steps": 1, "wall_time_s": 10},
        "policy_context": {
            "codex_project": ["agents.md"],
            "episode_prefix": ["prompt.md"],
            "tool_manual": ["manual.md"],
        },
        "qualification": {"required": False},
    }


DEFAULT_PROVIDER = "deepseek_flash_openai_low_16k_v1"


def _synthetic_native_spec() -> dict:
    """The quickstart's RunSpec on the default Provider; Synthetic needs no live admission marker."""
    return quickstart.synthetic_spec({**provider_policy(DEFAULT_PROVIDER), "require_live_marker": False})


def _robotwin_native_spec() -> dict:
    """A fresh default RoboTwin RunSpec on the default Provider: no surface, seed manifest or host named."""
    base = {key: value for key, value in spec().items() if key != "policy_context"}
    return {**base, "task": "click_alarmclock", "task_instruction": "Use one arm to press the alarm clock button.",
            "policy": {**provider_policy(DEFAULT_PROVIDER), "supports_images": True}, "runtime": {"kind": "robotwin"}}


def test_plan_is_deterministic_and_resource_change_rekeys(tmp_path: Path) -> None:
    root = init_project(tmp_path)
    one = resolve_run_spec(spec(), root=root)
    two = resolve_run_spec(spec(), root=root)
    assert one["runtime"]["transport"] == "in_process_direct"
    assert one["runtime"]["environment_signal_policy"] == "lifecycle_only"
    assert one["plan_sha256"] == two["plan_sha256"]
    (root / "prompt.md").write_text("prompt v2")
    changed = resolve_run_spec(spec(), root=root)
    assert changed["plan_sha256"] != one["plan_sha256"]
    assert changed["policy_context"]["composition_sha256"] != one["policy_context"]["composition_sha256"]
    assert changed["source"]["dirty"] is True
    invalid = spec()
    invalid["output_workspace"] = "outputs/shared"
    with pytest.raises(RunSpecError, match="output_workspace"):
        resolve_run_spec(invalid, root=root)
    invalid_transport = spec()
    invalid_transport["runtime"]["transport"] = "remote_rpc"
    with pytest.raises(RunSpecError, match="runtime.transport"):
        resolve_run_spec(invalid_transport, root=root)
    robotwin = spec()
    robotwin["runtime"] = {
        "kind": "robotwin", "transport": "local_rpc",
        "root_env": "ROBOTWIN_ROOT", "python_env": "ROBOTWIN_PYTHON",
        "required_revision": "revision", "task_config": "demo_clean",
        "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
    }
    assert resolve_run_spec(robotwin, root=root)["runtime"]["transport"] == "local_rpc"
    coordinated = spec()
    coordinated["task"] = "adjust_bottle"
    coordinated["runtime"] = {
        **robotwin["runtime"],
        "environment_signal_policy": "task_recoverability_v1",
    }
    assert resolve_run_spec(coordinated, root=root)["runtime"][
        "environment_signal_policy"
    ] == "task_recoverability_v1"
    coordinated["task"] = "open_laptop"
    with pytest.raises(RunSpecError, match="defined only for adjust_bottle"):
        resolve_run_spec(coordinated, root=root)
    robotwin["runtime"].pop("python_env")
    with pytest.raises(RunSpecError, match="runtime.python_env"):
        resolve_run_spec(robotwin, root=root)
    libero = spec()
    libero["runtime"] = {
        "kind": "libero_pro", "transport": "local_rpc",
        "python_env": "LIBERO_PRO_PYTHON",
        "libero_pro_root_env": "LIBERO_PRO_ROOT", "data_root_env": "LIBERO_PRO_DATA_ROOT",
        "required_libero_pro_revision": "libero-revision",
        "required_data_revision": "data-revision", "suite": "libero_object_swap",
        "task_id": 2, "expected_task_language": "place the object",
        "main_camera": "agentview",
        "workspace_bounds_m": {"x": [-0.3, 0.3], "y": [-0.3, 0.3], "z": [-0.1, 1.3]},
    }
    resolved_libero = resolve_run_spec(libero, root=root)
    assert resolved_libero["runtime"]["task_id"] == 2
    assert context_placeholder(resolved_libero)["main_camera"] == "agentview"
    robustness = spec()
    robustness["runtime"] = {
        **libero["runtime"],
        "suite": "libero_object",
        "required_libero_pro_revision": LIBERO_PRO_CODE_REVISION,
        "robustness_case": "view_occlusion",
    }
    assert resolve_run_spec(robustness, root=root)["runtime"][
        "robustness_case"
    ] == "view_occlusion"
    robustness["runtime"]["required_libero_pro_revision"] = "unpaired"
    with pytest.raises(RunSpecError, match="paired LIBERO-Pro revision"):
        resolve_run_spec(robustness, root=root)
    slotted_libero = json.loads(json.dumps(libero))
    slotted_libero["runtime"].update({
        "required_host": "test-host",
        "resource_profile": "libero-gpu1",
    })
    slotted = resolve_run_spec(slotted_libero, root=root)
    assert "resource_profile" not in slotted["runtime"]
    assert slotted["runtime"]["resource_binding"]["slot_id"] == "gpu1"
    assert slotted["plan_sha256"] != resolved_libero["plan_sha256"]
    invalid_binding = json.loads(json.dumps(libero))
    invalid_binding["runtime"]["resource_binding"] = {}
    with pytest.raises(RunSpecError, match="resolver-owned"):
        resolve_run_spec(invalid_binding, root=root)
    libero["runtime"]["task_id"] = 10
    with pytest.raises(RunSpecError, match="task_id"):
        resolve_run_spec(libero, root=root)


def test_the_ora_command_and_the_transport_of_each_runtime() -> None:
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) >= {
        "plan", "launch", "retry",
        "status", "providers", "watch", "stop", "metadata", "state", "metrics",
        "evaluate", "seal", "export",
    }
    assert resolve_run_spec(_synthetic_native_spec())["runtime"]["transport"] == "in_process_direct"
    assert resolve_run_spec(_robotwin_native_spec())["runtime"]["transport"] == "local_rpc"
    assert (ROOT / "examples/hosts/example-host.yaml").is_file()
    assert (ROOT / "scripts/ora").is_file()


def test_supervisor_pipeline_writes_runtime_metrics_before_releasing_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "plan.json").write_text(json.dumps({"plan_sha256": "plan-1"}))
    lease_root = tmp_path / "leases"
    monkeypatch.setattr(leases, "LEASE_ROOT", lease_root)
    monkeypatch.setattr(
        leases, "LEASE_LOCK_PATH", lease_root / ".registry.lock",
    )
    monkeypatch.setattr(leases, "LEASE_PATH", tmp_path / "active-run.json")
    monkeypatch.setattr(launch, "_run_root", lambda _run_id: run)
    plan = {
        "plan_sha256": "plan-1",
        "runtime": {"kind": "synthetic"},
    }
    lease = leases._create_lease(plan, "run-1", os.getpid())
    lease = leases._assign_lease(lease, os.getpid())

    def fake_episode(*, run_root: Path, plan: dict, run_id: str) -> dict:
        result = {
            "run_id": run_id, "plan_sha256": plan["plan_sha256"],
            "state": "completed", "started_unix_s": 10.0,
            "finished_unix_s": 12.0, "wall_duration_s": 1.8,
        }
        (run_root / "result.json").write_text(json.dumps(result))
        (run_root / "events.jsonl").write_text(
            json.dumps({"type": "run_started", "time": 10.0}) + "\n"
        )
        return result

    monkeypatch.setattr(launch, "run_episode", fake_episode)
    assert launch.supervise(
        "run-1", str(lease["claim_token"]),
    ) == 0
    metrics = json.loads((run / "runtime-metrics.json").read_text())
    assert metrics["run_id"] == "run-1"
    assert metrics["latency"]["episode_wall_s"] == 1.8
    assert (run / "context-audit.json").is_file()
    assert not Path(str(lease["lease_path"])).exists()
    viewer_args = viewer.build_parser().parse_args([])
    assert (viewer_args.host, viewer_args.port) == ("127.0.0.1", 5012)
    assert viewer_args.open_browser is False
    assert not (ROOT / "scripts/run-exp00").exists()
    for milestone in range(6):
        assert not (ROOT / f"scripts/run-harness-m{milestone}").exists()


def test_libero_preview_cache_is_complete_and_safely_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    code_root = tmp_path / "code"
    data_root = tmp_path / "data"
    for task in LIBERO_PRO_TASKS:
        base_suite = task.base_suite
        base_bddl = (
            code_root / "libero/libero/bddl_files" / base_suite
            / f"{task.name}.bddl"
        )
        init_root = (
            data_root / "init_files"
            if task.pro_anchor
            else code_root / "libero/libero/init_files"
        )
        base_init = init_root / base_suite / f"{task.name}.pruned_init"
        base_bddl.parent.mkdir(parents=True, exist_ok=True)
        base_init.parent.mkdir(parents=True, exist_ok=True)
        base_bddl.write_text(
            f"(:language {task.canonical_instruction})\n", encoding="utf-8",
        )
        base_init.write_bytes(b"init")
        for variant in task.core_variants:
            bddl = data_root / "bddl_files" / variant.suite / f"{task.name}.bddl"
            init = data_root / "init_files" / variant.suite / f"{task.name}.pruned_init"
            bddl.parent.mkdir(parents=True, exist_ok=True)
            init.parent.mkdir(parents=True, exist_ok=True)
            bddl.write_text(
                f"(:language {variant.instruction})\n", encoding="utf-8",
            )
            init.write_bytes(b"init")
        for case in LIBERO_ROBUSTNESS_CASES if task.pro_anchor else ():
            bddl = (
                data_root / "bddl_files" / case["id"] / "bddl" / base_suite
                / f"{task.name}.bddl"
            )
            bddl.parent.mkdir(parents=True, exist_ok=True)
            bddl.write_text(
                f"(:language {task.canonical_instruction})\n", encoding="utf-8",
            )
    descriptors = preview_descriptors(
        libero_pro_root=code_root, data_root=data_root,
    )
    assert len(descriptors) == LIBERO_EXPECTED_PREVIEWS == 570
    assert sum(item["layer"] == "base" for item in descriptors) == 130
    assert sum(item["layer"] == "core" for item in descriptors) == 160
    assert sum(item["layer"] == "robustness" for item in descriptors) == 280
    assert next(item for item in descriptors if item["key"] == (
        "base:libero_10:task-00"
    ))["task_name"].startswith("LIVING_ROOM_SCENE2_")
    assert next(item for item in descriptors if item["key"] == (
        "base:libero_90:task-89"
    ))["task_name"].startswith("STUDY_SCENE4_")

    cache = tmp_path / "cache"
    records = []
    for index, descriptor in enumerate(descriptors):
        relative = (
            f"sets/test/{descriptor['layer']}/{descriptor['suite']}/"
            f"task-{descriptor['task_id']:02d}-seed-0-{index:03d}.png"
        )
        image = cache / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\x89PNG\r\nviewer-test")
        records.append({
            "key": descriptor["key"], "image": relative,
            "layer": descriptor["layer"], "suite": descriptor["suite"],
            "task_id": descriptor["task_id"],
        })
    (cache / "manifest.json").write_text(json.dumps({
        "schema": LIBERO_PREVIEW_SCHEMA,
        "source": {
            "libero_pro_code_revision": "2b910b5b5f53016bef9907632f6f840f1ce2229c",
            "libero_pro_data_revision": "c86fc3b8293185a6f373677018ff3e37f8391602",
        },
        "render": LIBERO_RENDER_CONFIG,
        "records": records,
    }), encoding="utf-8")
    status = load_preview_manifest(cache)
    assert status["ready"] is True
    assert status["available"] == status["expected"] == 570
    monkeypatch.setattr(viewer, "load_preview_manifest", lambda: status)

    client = viewer.app.test_client()
    response = client.get("/gallery/libero-pro/object/0")
    assert response.status_code == 200
    assert b"570 / 570 seed-0 previews loaded" in response.data
    assert b"seed 0 \xc2\xb7 zero action" in response.data
    record = status["index"]["core:libero_object_swap:task-00"]
    image_response = client.get(f"/gallery/libero-pro/preview/{record['image']}")
    assert image_response.status_code == 200
    assert image_response.mimetype == "image/png"
    assert image_response.data == b"\x89PNG\r\nviewer-test"
    assert client.get("/gallery/libero-pro/preview/manifest.json").status_code == 404
    api = client.get("/api/libero-pro/previews").get_json()
    assert api["ready"] is True and api["available"] == 570

    (cache / records[-1]["image"]).unlink()
    incomplete = load_preview_manifest(cache)
    assert incomplete["ready"] is False
    assert incomplete["available"] == 569


def test_the_default_robotwin_composition_names_no_model_and_no_host() -> None:
    catalog = load_native_baseline(ROOT, default_baseline_id("robotwin"))
    runtime = catalog["instances"]["robotwin"]["runtime"]
    assert "model" not in catalog["policy"]
    # The execution host is the marker's registered host, never a default.
    assert "required_host" not in runtime
    assert runtime["transport"] == "local_rpc"
    assert runtime["python_env"] == "ROBOTWIN_PYTHON"


def test_context_preview_does_not_require_execution_gates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """``ora plan context`` previews a plan that could not launch: no seed manifest, host or preflight."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("context preview must not run a launch gate")

    monkeypatch.setattr(launch, "_preflight", forbidden)
    monkeypatch.setattr("ora.orchestrator.preflight.gates.run_preflight", forbidden)
    plan = resolve_run_spec(_robotwin_native_spec())
    assert "manifest" not in plan["qualification"]
    assert plan_command.command_context(plan) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["plan_sha256"] == plan["plan_sha256"]


def test_context_audit_distinguishes_native_visibility_from_manifest_only(
    tmp_path: Path,
) -> None:
    from ora.orchestrator.evaluation.context_audit import policy_delivery_inventory

    def source(path: str) -> dict[str, object]:
        return {"path": path, "bytes": 1, "sha256": "a" * 64}

    plan = {
        "policy": {"kind": "native_api"},
        "policy_context": {
            "schema": "ora.policy-context.v3",
            "harness_system": [source("policy/system.md")],
            "project_instructions": [source("policy/project.md")],
            "episode_task": [source("policy/task.md")],
            "tool_manual": [source("policy/tool-manual.md")],
        },
        "task_instruction": "Do the task.",
    }
    inventory = policy_delivery_inventory(plan)
    by_channel = {
        item["channel"]: item for item in inventory["policy_layers"]
        if item["channel"] != "task"
    }
    assert by_channel["harness_system"]["provider_role"] == "system"
    assert by_channel["project_instructions"]["provider_role"] == "user"
    assert by_channel["episode_task"]["provider_visible"] is True
    assert by_channel["tool_manual"]["provider_visible"] is False
    assert by_channel["tool_manual"]["delivery"] == "manifest_only"
    assert by_channel["tool_manual"]["injection_count"] == 0
    assert inventory["tool_schema"]["provider_visible"] is True
    assert inventory["tool_schema"]["delivery_timing"] == "every_model_call"


def test_seal_at_landing_seals_once_and_records_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    calls: list[str] = []

    def fake_seal(root: Path, *, mode: str) -> dict[str, str]:
        calls.append(mode)
        (root / "seal.json").write_text("{}")
        return {"mode": mode}

    monkeypatch.setattr(launch, "seal", fake_seal)
    launch._seal_at_landing(run_root)
    assert calls == ["landing"]
    assert not (run_root / "seal-refusal.json").exists()

    # Already sealed: no second attempt.
    launch._seal_at_landing(run_root)
    assert calls == ["landing"]

    refused_root = tmp_path / "run2"
    refused_root.mkdir()

    def refusing_seal(root: Path, *, mode: str) -> dict[str, str]:
        raise EvaluationError("invalid runs cannot be sealed as evidence")

    monkeypatch.setattr(launch, "seal", refusing_seal)
    launch._seal_at_landing(refused_root)
    refusal = json.loads((refused_root / "seal-refusal.json").read_text())
    assert refusal["schema"] == "ora.seal-refusal.v1"
    assert refusal["stage"] == "landing"
    assert "cannot be sealed" in refusal["reason"]


def test_supervisor_waits_while_the_launcher_has_not_yet_named_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # The launcher assigns supervisor_pid only after spawning the supervisor,
    # so a supervisor can read its lease while that field is still null.
    import threading
    run = tmp_path / "run"
    run.mkdir()
    (run / "plan.json").write_text(json.dumps({"plan_sha256": "plan-1"}))
    lease_root = tmp_path / "leases"
    monkeypatch.setattr(leases, "LEASE_ROOT", lease_root)
    monkeypatch.setattr(leases, "LEASE_LOCK_PATH", lease_root / ".registry.lock")
    monkeypatch.setattr(leases, "LEASE_PATH", tmp_path / "active-run.json")
    monkeypatch.setattr(launch, "_run_root", lambda _run_id: run)
    plan = {"plan_sha256": "plan-1", "runtime": {"kind": "synthetic"}}
    lease = leases._create_lease(plan, "run-1", os.getpid())
    assert lease["supervisor_pid"] is None
    timer = threading.Timer(0.3, leases._assign_lease, args=(lease, os.getpid()))
    timer.start()

    def fake_episode(*, run_root: Path, plan: dict, run_id: str) -> dict:
        result = {"run_id": run_id, "plan_sha256": plan["plan_sha256"], "state": "completed",
                  "started_unix_s": 10.0, "finished_unix_s": 12.0, "wall_duration_s": 1.8}
        (run_root / "result.json").write_text(json.dumps(result))
        (run_root / "events.jsonl").write_text(json.dumps({"type": "run_started", "time": 10.0}) + "\n")
        return result

    monkeypatch.setattr(launch, "run_episode", fake_episode)
    try:
        assert launch.supervise("run-1", str(lease["claim_token"])) == 0
    finally:
        timer.join()


def test_native_plans_resolve_on_the_default_provider_and_the_pinned_simulator() -> None:
    synthetic, robotwin = resolve_run_spec(_synthetic_native_spec()), resolve_run_spec(_robotwin_native_spec())
    for plan in (synthetic, robotwin):
        assert plan["policy"]["kind"] == "native_api"
        assert plan["policy"]["provider_profile"] == DEFAULT_PROVIDER
    assert robotwin["runtime"]["required_revision"] == "cf4872cc90538fd4bca761c8db912d8f65e49f2b"
