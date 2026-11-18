"""The RoboCasa365 orchestrator backend, host-safe: runtime, RunSpec builder, preflight and qualification script.

``ora/orchestrator/backends/robocasa/`` resolves the kind's runtime fields
(``motion_action_v3`` only, the task's official step limit, the scene's own
language), composes one cell's RunSpec on the default baseline, checks the
worker interpreter and its pinned sources before launch, and drives the
sparse-motion qualification script through one worker session. Here the
interpreter probe, the language probe and the session are scripted; nothing
starts robosuite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ora.contracts.official_limit import TOOL_BUDGET_LIMIT_MULTIPLE
from ora.environment.benchmarks.robocasa import TASKS
from ora.orchestrator.backends.registry import PreflightContext, backend
from ora.orchestrator.backends.robocasa import preflight, qualification, runtime, spec
from ora.orchestrator.preflight.checks import check
from ora.orchestrator.spec.run_spec import resolve_run_spec
from ora.orchestrator.spec.values import RunSpecError
from tests.support.orchestrator.sparse_motion import ScriptedSession

ROOT = Path(__file__).resolve().parents[5]
LIMIT = TASKS["OpenDrawer"]["horizon_control_steps"]


def _resolve(**fields) -> dict:
    values = {"task": "OpenDrawer", "seed": 0, **fields}
    policy = {"action_codec": values.pop("action_codec", "motion_action_v3")}
    instruction = values.pop("instruction", "open the drawer")
    task, seed = values.pop("cell", ("OpenDrawer", 0))
    runtime.resolve_runtime(values, policy, None, task=task, seed=seed, task_instruction=instruction)
    return values


def test_the_runtime_binds_the_official_limit_and_its_timeouts():
    resolved = _resolve(expected_task_language="open the drawer")
    assert resolved["native_action_limit"] == LIMIT == runtime.task_limit("OpenDrawer")
    assert resolved["official_limit_multiple"] == TOOL_BUDGET_LIMIT_MULTIPLE  # the budgets end a run (2026-11-05)
    assert (resolved["startup_timeout_s"], resolved["request_timeout_s"], resolved["close_timeout_s"]) == (300, 600, 60)
    assert _resolve(request_timeout_s=900)["request_timeout_s"] == 900


@pytest.mark.parametrize("fields, message", [
    ({"action_codec": "target_state_v4"}, "motion_action_v3 only"),
    ({"cell": ("OpenDrawer", 1)}, "task/seed must match"),
    ({"cell": ("NavigateKitchen", 0)}, "task/seed must match"),
    ({"native_action_limit": LIMIT + 1}, "official limit"),
    ({"expected_task_language": "close the drawer"}, "environment language"),
])
def test_the_runtime_refuses_what_robocasa_cannot_run(fields, message):
    with pytest.raises(RunSpecError, match=message):
        _resolve(**fields)


def test_the_plan_display_names_every_pinned_checkout():
    label = runtime.revision_label({"required_revisions": {"robosuite": "b" * 40, "robocasa": "a" * 40}})
    assert label == f"robocasa={'a' * 12}/robosuite={'b' * 12}"
    assert backend("robocasa").revision_label is runtime.revision_label


def test_a_composed_cell_resolves_into_a_plan_on_the_default_baseline(simulator_host, workspace):
    cell = spec.build_robocasa_spec("OpenDrawer", seed=0, experiment="robocasa-fixture",
                                    resource_profile="robocasa-slot0", instruction="open the drawer")
    assert cell["runtime"] == {"kind": "robocasa", "required_host": simulator_host,
                               "resource_profile": "robocasa-slot0", "task": "OpenDrawer", "seed": 0,
                               "expected_task_language": "open the drawer"}
    assert cell["config"] == "rc-OpenDrawer-s0" and cell["milestone"] == spec.MILESTONE
    assert cell["qualification"] == {"required": False}
    path = spec.compose(cell)
    assert path == workspace / "outputs/robocasa-fixture/specs/rc-OpenDrawer-s0.json"
    plan = resolve_run_spec(json.loads(path.read_text(encoding="utf-8")), root=ROOT)
    assert plan["runtime"]["kind"] == "robocasa" and plan["runtime"]["native_action_limit"] == LIMIT
    assert plan["runtime"]["resource_binding"]["profile_id"] == "robocasa-slot0"
    assert plan["policy"]["action_codec"] == "motion_action_v3"
    with pytest.raises(ValueError):
        spec.build_robocasa_spec("OpenDrawer", seed=0, experiment="robocasa-fixture",
                                 resource_profile="robolab-slot2", instruction="open the drawer")


def test_a_cell_runs_its_tools_and_its_base_on_the_toolkit_and_the_environment_plans_its_moves(simulator_host,
                                                                                                 workspace):
    cell = spec.build_robocasa_spec("OpenDrawer", seed=0, experiment="robocasa-fixture",
                                    resource_profile="robocasa-slot0", instruction="open the drawer")
    surface = resolve_run_spec(json.loads(spec.compose(cell).read_text(encoding="utf-8")),
                               root=ROOT)["policy_composition"]["tool_surface"]
    assert "toolkit_only_v1" in surface["capabilities"]
    assert "toolkit_synchronous_execution_v1" not in surface["capabilities"]
    assert "move_base" in {item["name"] for item in surface["definitions"]}


def _probe(monkeypatch, stdout: str, returncode: int = 0) -> list:
    calls: list = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="probe failed\n")

    monkeypatch.setattr(spec.subprocess, "run", run)
    return calls


def test_the_cell_language_is_read_from_the_scene_by_the_worker_interpreter(monkeypatch):
    monkeypatch.delenv("ROBOCASA_PYTHON", raising=False)
    with pytest.raises(ValueError, match="missing ROBOCASA_PYTHON"):
        spec.episode_language("OpenDrawer", 0)
    monkeypatch.setenv("ROBOCASA_PYTHON", "/opt/robocasa/bin/python")
    calls = _probe(monkeypatch, 'noise\n{"lang": "  open the drawer  "}\n')
    assert spec.episode_language("OpenDrawer", 3) == "open the drawer"
    (command, options), = calls
    assert command[0] == "/opt/robocasa/bin/python" and command[-2:] == ["OpenDrawer", "3"]
    assert options["env"]["PYTHONNOUSERSITE"] == "1" and options["env"]["MUJOCO_GL"] == "egl"
    _probe(monkeypatch, '{"lang": ""}\n')
    with pytest.raises(ValueError, match="no task language"):
        spec.episode_language("OpenDrawer", 0)
    _probe(monkeypatch, "", returncode=1)
    with pytest.raises(ValueError, match="probe failed"):
        spec.episode_language("OpenDrawer", 0)


def test_the_spec_tool_prints_a_language_and_writes_a_cell(simulator_host, workspace, monkeypatch, capsys):
    monkeypatch.setattr(spec, "episode_language", lambda task, seed: f"{task} seed {seed}")
    assert spec.main(["language", "OpenDrawer", "--seed", "2"]) == 0
    assert capsys.readouterr().out == "OpenDrawer seed 2\n"
    output = workspace / "cell.json"
    assert spec.main(["compose", "NavigateKitchen", "--seed", "1", "--experiment", "robocasa-fixture",
                      "--resource-profile", "robocasa-slot0", "--output", str(output)]) == 0
    assert capsys.readouterr().out.strip() == str(output)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["task_instruction"] == written["runtime"]["expected_task_language"] == "NavigateKitchen seed 1"
    with pytest.raises(SystemExit):
        spec.main(["compose", "NoSuchTask", "--seed", "0", "--experiment", "x", "--resource-profile", "y"])


def _context(runner) -> PreflightContext:
    return PreflightContext(project_root=ROOT, check=check, git=None, workspace_root=lambda root: root,
                            runner=runner)


def test_preflight_probes_the_worker_interpreter_for_its_pinned_sources(simulator_host, monkeypatch):
    plan = {"runtime": {"required_host": simulator_host}}
    monkeypatch.delenv("ROBOCASA_PYTHON", raising=False)
    checks = preflight.preflight_runtime(plan, _context(runner=None))
    assert [(item["check"], item["passed"]) for item in checks] == [("execution_host", True),
                                                                    ("environment_python", False)]
    monkeypatch.setenv("ROBOCASA_PYTHON", sys.executable)
    seen: list = []

    def runner(command, **kwargs):
        seen.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, stdout='{"robocasa": "a9e152e2"}\n', stderr="")

    checks = preflight.preflight_runtime(plan, _context(runner))
    assert [item["passed"] for item in checks] == [True, True, True]
    assert checks[-1] == {"check": "robocasa_source", "passed": True, "detail": {"robocasa": "a9e152e2"}}
    assert seen[0]["PYTHONPATH"] == str(ROOT) and "ORA_EXECUTION_HOST" not in seen[0]
    failed = preflight.robocasa_runtime(plan, project_root=ROOT, runner=lambda command, **kwargs: (
        subprocess.CompletedProcess(command, 1, stdout="", stderr="ImportError: robosuite\n")))
    assert failed[-1]["passed"] is False and failed[-1]["detail"]["stderr"] == "ImportError: robosuite"
    garbled = preflight.robocasa_runtime(plan, project_root=ROOT, runner=lambda command, **kwargs: (
        subprocess.CompletedProcess(command, 0, stdout="not json\n", stderr="")))
    assert garbled[-1]["passed"] is False and garbled[-1]["detail"]["returncode"] == 0


def _qualify(monkeypatch, tmp_path: Path, session: ScriptedSession, argv: list[str]) -> tuple[int, dict]:
    built: dict = {}
    monkeypatch.setattr(qualification, "motion_timing_check", lambda timing: {
        "check": "motion_timing_worker", "passed": True, "detail": {"python": "/p", "python_sha256": "a" * 64}})

    def build(**kwargs):
        built.update(kwargs)
        return session

    monkeypatch.setattr(qualification, "build_robocasa_worker_session", build)
    monkeypatch.setattr(qualification, "motion_planner_check", lambda _planner: {
        "check": "shared_motion_planner", "passed": True, "detail": {"python": "/p", "root": "/r"}})
    return qualification.main([*argv, "--output", str(tmp_path / "record")]), built


def test_the_qualification_script_records_every_check_of_one_kitchen(monkeypatch, tmp_path, capsys):
    session = ScriptedSession(reset={"task_language": "open the drawer", "official_success_at_reset": False},
                              verdict={"official_success": False, "source": "RoboCasa _check_success"},
                              refuse_first_move_pose=True)
    status, built = _qualify(monkeypatch, tmp_path, session, ["--seed", "4"])
    assert status == 0 and session.closed
    assert built["task"] == "OpenDrawer" and built["seed"] == 4 and built["action_codec"] == "motion_action_v3"
    record = json.loads((tmp_path / "record" / qualification.RECORD_NAME).read_text(encoding="utf-8"))
    assert record["schema"] == qualification.SCHEMA and record["passed"] is True
    assert [item["check"] for item in record["checks"]] == [
        "reset", "observation_rgbd", "proprioception", "move_pose_into_fixture_refused", "set_gripper_open",
        "move_pose_precision", "move_free_space", "move_pose_reorient_60deg", "set_gripper_close", "wait",
        "move_base_translate", "move_base_rotate", "move_base_return", "move_pose_torso_raise", "recover_home",
        "probe_motion", "status_running", "official_verdict", "close"]
    assert [action["action"] for action in session.actions][-3:] == ["move_base", "move_pose", "recover_motion"]
    assert all(action["schema"] == "ora-motion-action.v3" for action in session.actions)
    assert session.probes[0]["queries"][0]["arm"] == "right"
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["passed"] is True and len(summary["sha256"]) == 64
    with pytest.raises(FileExistsError):  # a record directory is never reused
        _qualify(monkeypatch, tmp_path, session, [])


def test_a_failed_check_or_close_fails_the_record_and_a_lost_worker_still_closes(monkeypatch, tmp_path, capsys):
    session = ScriptedSession(reset={"task_language": "open the drawer", "official_success_at_reset": False},
                              verdict={"official_success": False, "source": "RoboCasa _check_success"},
                              close_error=RuntimeError("worker did not exit"))
    status, _ = _qualify(monkeypatch, tmp_path, session, [])
    assert status == 1
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("FAIL ") and "move_pose_into_fixture_refused" in line for line in lines)
    assert any(line.startswith("FAIL ") and "worker did not exit" in line for line in lines)
    lost = ScriptedSession(reset={}, verdict={}, fail_at="wait")
    with pytest.raises(RuntimeError, match="lost wait"):
        _qualify(monkeypatch, tmp_path / "lost", lost, [])
    assert lost.closed and not (tmp_path / "lost" / "record" / qualification.RECORD_NAME).exists()


def test_the_default_runtime_binds_the_timing_worker_and_preflight_starts_it(simulator_host, monkeypatch):
    from ora.orchestrator.preflight import checks as preflight_checks

    from ora.orchestrator.spec.inputs import default_instance

    _baseline, instance = default_instance("robocasa")
    plan = {"runtime": {**instance["runtime"], "required_host": simulator_host}}
    assert plan["runtime"]["motion_timing"] == {"implementation": "toppra_ruckig", "python_env": "ORA_TIMING_PYTHON"}
    started = {"check": "motion_timing_worker", "passed": True, "detail": {"python": "/p", "python_sha256": "a" * 64}}
    seen = []
    monkeypatch.setattr(preflight, "motion_timing_check", lambda timing, **kwargs: (seen.append(timing), started)[1])
    monkeypatch.delenv("ROBOCASA_PYTHON", raising=False)
    checks = preflight.preflight_runtime(plan, _context(runner=None))
    assert started in checks and seen == [plan["runtime"]["motion_timing"]]
    assert preflight_checks.motion_timing_check({"python_env": "ORA_NOT_SET"})["passed"] is False
