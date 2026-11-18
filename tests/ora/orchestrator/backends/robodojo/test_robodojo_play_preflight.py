"""Launch preflight of a RoboDojo free-play plan (``backends/robodojo/play_preflight.py``), host-safe.

A plan naming one of ORA's play scene sets is admitted by its play-scene
audit, not by infrastructure qualification. Its preflight keeps the
benchmark's host, interpreter, pinned clean checkouts, worker package stack,
GPU memory and driver checks; adds that every scheduled layout regenerates to
the one the plan binds; and requires the driver's Vulkan compatibility
fingerprint to equal the one the audit recorded. Git, the interpreter probe,
``nvidia-smi`` and the layout generator are scripted here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ora.environment.benchmarks.robodojo import REVISION_FIELDS
from ora.orchestrator import robodojo_play_audit, robodojo_qualification
from ora.orchestrator.backends.registry import PreflightContext
from ora.orchestrator.backends.robodojo import preflight
from ora.orchestrator.backends.robodojo.play_preflight import play_runtime_checks
from ora.orchestrator.preflight.checks import check

ROOT = Path(__file__).resolve().parents[5]
LAYOUTS = {"0": "a" * 64, "1": "b" * 64}
STACK = {"isaacsim": "5.1.0.0", "torch": "2.7.0+cu128", "numpy": "1.26.0", "warp-lang": "1.8.0"}
ROOT_CHECKS = ["robodojo_root_env", "robodojo_asset_root_env", "robodojo_python_env"]
SOURCE_CHECKS = [f"robodojo_{name}_source" for name in REVISION_FIELDS]
ALL_CHECKS = ["execution_host", "robodojo_play_scene", *ROOT_CHECKS, *SOURCE_CHECKS, "robodojo_play_layouts",
              "robodojo_python_stack", "robodojo_gpu", "robodojo_driver"]


@pytest.fixture
def host(tmp_path, monkeypatch):
    """Roots, a play plan on scene ``desk`` (CUDA physics) and its audit record in the workspace."""

    for name in ("source", "assets"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    monkeypatch.setenv("ROBODOJO_ROOT", str(tmp_path / "source"))
    monkeypatch.setenv("ROBODOJO_ASSET_ROOT", str(tmp_path / "assets"))
    monkeypatch.setenv("ROBODOJO_PYTHON", sys.executable)
    revisions = {field: f"{index}" * 40 for index, field in enumerate(REVISION_FIELDS.values())}
    runtime = {"required_host": "test-host", "play_scene_set": "isaac-tabletop-v1", "physics_device": "cuda:0",
               "root_env": "ROBODOJO_ROOT", "asset_root_env": "ROBODOJO_ASSET_ROOT", "python_env": "ROBODOJO_PYTHON",
               "play_layouts": dict(LAYOUTS), "cuda_visible_devices": "3",
               "resource_binding": {"minimum_free_memory_mib": 16000}, **revisions}
    audit = tmp_path / "workspace/outputs/play-fixture/qualification/audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({"graphics_compatibility": None, "device_binding": {"gpu": "3"}}), encoding="utf-8")
    plan = {"experiment": "play-fixture", "task": "desk", "runtime": runtime,
            "qualification": {"manifest": {"path": "outputs/play-fixture/qualification/audit.json"}}}
    monkeypatch.setattr(robodojo_play_audit, "generated_layouts",
                        lambda definition, seeds, assets: {str(seed): LAYOUTS[str(seed)] for seed in seeds})
    return {"plan": plan, "workspace": tmp_path / "workspace", "revisions": revisions, "tmp": tmp_path}


def _git(host: dict, dirty: str = ""):
    by_root = {Path(host["tmp"] / "source"): REVISION_FIELDS["robodojo"],
               Path(host["tmp"] / "source/third_party/IsaacLab"): REVISION_FIELDS["isaaclab"],
               Path(host["tmp"] / "source/third_party/curobo"): REVISION_FIELDS["curobo"],
               Path(host["tmp"] / "source/XPolicyLab"): REVISION_FIELDS["xpolicylab"],
               Path(host["tmp"] / "assets"): REVISION_FIELDS["assets"]}

    def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        output = host["revisions"][by_root[Path(root)]] + "\n" if args[0] == "rev-parse" else dirty
        return subprocess.CompletedProcess(["git", *args], 0, stdout=output, stderr="")
    return git


def _runner(*, gpu: str = "3, GPU-1, 30000, 580.65.06\n", stack: dict = STACK, smi_timeouts: int = 0):
    calls = {"smi": 0}

    def run(command, **kwargs):
        if command[0] == "nvidia-smi":
            calls["smi"] += 1
            if calls["smi"] <= smi_timeouts:
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return subprocess.CompletedProcess(command, 0, stdout=gpu, stderr="")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"versions": stack}) + "\n", stderr="")
    run.calls = calls
    return run


def _checks(host: dict, *, runner=None, git=None) -> dict[str, dict]:
    checks = play_runtime_checks(host["plan"], runner or _runner(), project_root=ROOT, check=check,
                                 git=git or _git(host), workspace_root=host["workspace"])
    return {item["check"]: item for item in checks}


def test_a_play_plan_on_an_admitted_host_passes_every_check(host):
    checks = _checks(host)
    assert list(checks) == ALL_CHECKS
    assert [name for name, item in checks.items() if not item["passed"]] == []
    assert checks["robodojo_play_scene"]["detail"]["scene"] == "desk"
    assert checks["robodojo_gpu"]["detail"]["audited_device_binding"] == {"gpu": "3"}


def test_the_robodojo_preflight_hands_a_play_plan_to_these_checks(host):
    context = PreflightContext(project_root=ROOT, check=check, git=_git(host),
                               workspace_root=lambda root: host["workspace"], runner=_runner())
    checks = preflight.preflight_runtime(host["plan"], context)
    # A play plan gets no task-contract or infrastructure-qualification check.
    assert [item["check"] for item in checks] == ALL_CHECKS


def test_an_unknown_scene_or_another_physics_device_is_refused(host):
    host["plan"]["task"] = "garage"
    checks = play_runtime_checks(host["plan"], _runner(), project_root=ROOT, check=check, git=_git(host),
                                 workspace_root=host["workspace"])
    assert [(item["check"], item["passed"]) for item in checks] == [("execution_host", True),
                                                                    ("robodojo_play_scene", False)]
    assert "no scene 'garage'" in checks[-1]["detail"]
    host["plan"]["task"] = "desk"
    host["plan"]["runtime"]["physics_device"] = "cpu"
    assert _checks(host)["robodojo_play_scene"]["passed"] is False


def test_missing_roots_stop_before_any_source_or_device_probe(host, monkeypatch):
    monkeypatch.delenv("ROBODOJO_ASSET_ROOT")
    monkeypatch.setenv("ORA_EXECUTION_HOST", "other-host")
    checks = _checks(host, runner=lambda *args, **kwargs: pytest.fail("probed without roots"))
    assert list(checks) == ["execution_host", "robodojo_play_scene", *ROOT_CHECKS]
    assert [item["passed"] for item in checks.values()] == [False, True, True, False, True]


def test_a_dirty_checkout_or_a_drifted_layout_is_refused(host, monkeypatch):
    checks = _checks(host, git=_git(host, dirty=" M source.py\n"))
    assert {name for name, item in checks.items() if not item["passed"]} == set(SOURCE_CHECKS)
    monkeypatch.setattr(robodojo_play_audit, "generated_layouts", lambda definition, seeds, assets: {"0": "c" * 64,
                                                                                                    "1": "b" * 64})
    assert _checks(host)["robodojo_play_layouts"]["passed"] is False

    def broken(definition, seeds, assets):
        raise ValueError("asset usd missing")
    monkeypatch.setattr(robodojo_play_audit, "generated_layouts", broken)
    assert _checks(host)["robodojo_play_layouts"] == {"check": "robodojo_play_layouts", "passed": False,
                                                       "detail": "asset usd missing"}
    host["plan"]["runtime"]["play_layouts"] = {}
    monkeypatch.setattr(robodojo_play_audit, "generated_layouts", lambda definition, seeds, assets: {})
    assert _checks(host)["robodojo_play_layouts"]["passed"] is False  # a plan binds at least one layout


def test_the_worker_stack_must_be_the_pinned_one(host):
    assert _checks(host, runner=_runner(stack={**STACK, "torch": "2.8.0"}))["robodojo_python_stack"]["passed"] is False

    def unreachable(command, **kwargs):
        raise OSError("interpreter vanished")
    assert _checks(host, runner=unreachable)["robodojo_python_stack"]["detail"] == "interpreter vanished"


def test_the_selected_card_needs_its_free_memory_floor_and_one_slow_answer_is_retried(host):
    assert _checks(host, runner=_runner(gpu="3, GPU-1, 9000, 580.65.06\n"))["robodojo_gpu"]["passed"] is False
    absent = _checks(host, runner=_runner(gpu="0, GPU-0, 30000, 580.65.06\n"))
    assert absent["robodojo_gpu"]["passed"] is False and absent["robodojo_driver"]["passed"] is False
    retried = _runner(smi_timeouts=1)
    assert _checks(host, runner=retried)["robodojo_gpu"]["passed"] is True and retried.calls["smi"] == 2
    assert _checks(host, runner=_runner(smi_timeouts=2))["robodojo_gpu"]["passed"] is False
    host["plan"]["runtime"]["resource_binding"] = {}
    unprofiled = _checks(host, runner=_runner(gpu="3, GPU-1, 19000, 580.65.06\n"))["robodojo_gpu"]
    assert unprofiled["passed"] is False and unprofiled["detail"]["minimum_free_memory_mib"] == 20000


def test_the_driver_needs_the_audits_vulkan_fingerprint_and_a_readable_audit(host, monkeypatch):
    driver_595 = _runner(gpu="3, GPU-1, 30000, 595.45.04\n")
    refused = _checks(host, runner=driver_595)["robodojo_driver"]
    assert refused["passed"] is False  # 595 needs the verified compatibility repair
    monkeypatch.setattr(robodojo_qualification, "inspect_graphics_compatibility",
                        lambda python, runner: {"fingerprint": "repaired"})
    monkeypatch.setattr(robodojo_qualification, "validate_runtime_driver",
                        lambda version, graphics_compatibility: None)
    assert "differs from the play-scene audit" in _checks(host, runner=driver_595)["robodojo_driver"]["detail"]
    audit = host["workspace"] / "outputs/play-fixture/qualification/audit.json"
    audit.write_text(json.dumps({"graphics_compatibility": {"fingerprint": "repaired"}}), encoding="utf-8")
    assert _checks(host, runner=driver_595)["robodojo_driver"]["passed"] is True
    audit.write_text("not json", encoding="utf-8")
    assert "unreadable" in _checks(host)["robodojo_driver"]["detail"]
    host["plan"]["qualification"] = {}
    assert "unreadable" in _checks(host)["robodojo_driver"]["detail"]


def test_a_play_plan_gets_the_benchmarks_timing_worker_and_planner_checks(host, monkeypatch):
    """The play worker binds its timing from this record and starts the planner, as a benchmark worker does."""
    started = {"check": "motion_timing_worker", "passed": True, "detail": {"python": "/p", "python_sha256": "a" * 64}}
    seen = []
    monkeypatch.setattr(preflight, "motion_timing_check", lambda timing, **kwargs: (seen.append((timing, kwargs)),
                                                                                    started)[1])
    runtime = host["plan"]["runtime"]
    runtime.update(motion_timing={"implementation": "toppra_ruckig", "python_env": "ORA_TIMING_PYTHON"},
                   motion_planner={"implementation": "curobo_v2", "python_env": "CUROBO_V2_PYTHON",
                                   "root_env": "CUROBO_V2_ROOT"})
    runtime["resource_binding"]["worker_environment"] = {"CUDA_VISIBLE_DEVICES": "3"}
    monkeypatch.setenv("CUROBO_V2_PYTHON", sys.executable)
    monkeypatch.delenv("CUROBO_V2_ROOT", raising=False)
    checks = _checks(host)
    assert list(checks) == ["execution_host", "motion_timing_worker", "shared_motion_planner_environment",
                            *ALL_CHECKS[1:]]
    assert checks["motion_timing_worker"] == started
    assert seen == [(runtime["motion_timing"], {"environment_overrides": {"CUDA_VISIBLE_DEVICES": "3"}})]
    assert checks["shared_motion_planner_environment"] == {
        "check": "shared_motion_planner_environment", "passed": False, "detail": {"missing": ["CUROBO_V2_ROOT"]}}
    monkeypatch.setenv("CUROBO_V2_ROOT", str(host["tmp"]))
    assert _checks(host)["shared_motion_planner_environment"]["passed"] is True
    # The benchmark preflight's own helper, after its host check too; a runtime that names neither gets none.
    planner = {"check": "shared_motion_planner_environment", "passed": True, "detail": {"missing": []}}
    assert preflight.shared_motion_checks(runtime, check) == [started, planner]
    assert preflight.shared_motion_checks({}, check) == []
