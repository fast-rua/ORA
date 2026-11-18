"""The RoboLab orchestrator backend, host-safe: runtime, RunSpec builder, preflight and qualification script.

``ora/orchestrator/backends/robolab/`` resolves the kind's runtime fields
(``motion_action_v3`` only, the task's official step limit and instruction,
an integer renderer GPU), composes one cell whose renderer is the leased
slot's single card, checks the interpreter's packages, the clean pinned
checkout and the graphics runtime before launch, and drives the sparse-motion
qualification script through one worker session. Here the probes and the
session are scripted; nothing starts Isaac Sim.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ora.contracts.official_limit import TOOL_BUDGET_LIMIT_MULTIPLE
from ora.environment.benchmarks.robolab import PACKAGES, REVISION, TASKS
from ora.orchestrator.backends.registry import PreflightContext, backend
from ora.orchestrator.backends.robolab import preflight, qualification, runtime, spec
from ora.orchestrator.preflight.checks import check
from ora.orchestrator.spec.run_spec import resolve_run_spec
from ora.orchestrator.spec.values import RunSpecError
from tests.support.orchestrator.sparse_motion import ScriptedSession

ROOT = Path(__file__).resolve().parents[5]
TASK = "BananaInBowlTask"
INSTRUCTION = TASKS[TASK]["instruction"]


def _resolve(**fields) -> dict:
    values = {"task": TASK, "seed": 0, "expected_task_language": INSTRUCTION, "renderer_gpu": 2, **fields}
    policy = {"action_codec": values.pop("action_codec", "motion_action_v3")}
    task, seed = values.pop("cell", (TASK, 0))
    runtime.resolve_runtime(values, policy, None, task=task, seed=seed, task_instruction=INSTRUCTION)
    return values


def test_the_runtime_binds_the_official_limit_and_its_timeouts():
    resolved = _resolve()
    assert resolved["native_action_limit"] == TASKS[TASK]["native_action_limit"] == runtime.task_limit(TASK)
    assert resolved["official_limit_multiple"] == TOOL_BUDGET_LIMIT_MULTIPLE  # as RoboDojo: the budgets end a run
    assert _resolve(official_limit_multiple=1)["official_limit_multiple"] == 1
    assert (resolved["startup_timeout_s"], resolved["request_timeout_s"], resolved["close_timeout_s"]) == (900, 1200, 120)


@pytest.mark.parametrize("fields, message", [
    ({"action_codec": "target_state_v4"}, "motion_action_v3 only"),
    ({"cell": (TASK, 1)}, "task/seed must match"),
    ({"native_action_limit": 1}, "official limit"),
    ({"official_limit_multiple": 0}, "official_limit_multiple"),
    ({"expected_task_language": "stack the cubes"}, "official environment language"),
    ({"renderer_gpu": True}, "integer runtime.renderer_gpu"),
    ({"renderer_gpu": "2"}, "integer runtime.renderer_gpu"),
])
def test_the_runtime_refuses_what_robolab_cannot_run(fields, message):
    with pytest.raises(RunSpecError, match=message):
        _resolve(**fields)


def test_the_plan_display_names_the_pin_and_the_renderer():
    assert runtime.revision_label({"required_revision": REVISION, "renderer_gpu": 3}) == (
        f"robolab={REVISION[:12]} gpu=3")
    assert backend("robolab").revision_label is runtime.revision_label
    assert backend("robolab").capabilities.kit_storage_reason


def test_a_composed_cell_renders_on_its_slots_card_and_resolves_into_a_plan(simulator_host, workspace):
    cell = spec.build_robolab_spec(TASK, seed=0, experiment="robolab-fixture", resource_profile="robolab-slot2")
    assert cell["runtime"] == {"kind": "robolab", "required_host": simulator_host,
                               "resource_profile": "robolab-slot2", "task": TASK, "seed": 0,
                               "expected_task_language": INSTRUCTION, "renderer_gpu": 2}
    assert cell["task_instruction"] == INSTRUCTION and cell["config"] == f"rl-{TASK}-s0"
    plan = resolve_run_spec(cell, root=ROOT)
    assert plan["runtime"]["renderer_gpu"] == 2 and plan["runtime"]["required_revision"] == REVISION
    assert plan["runtime"]["resource_binding"]["worker_environment"] == {"CUDA_VISIBLE_DEVICES": "2"}
    with pytest.raises(ValueError, match="exactly one CUDA device"):
        spec.build_robolab_spec(TASK, seed=0, experiment="robolab-fixture", resource_profile="robolab-two-cards")


def test_a_cell_can_stop_at_the_official_limit(simulator_host, workspace):
    """Unset, the plan runs on past the official limit; 1 stops there, as RoboDojo's composer allows."""

    default = resolve_run_spec(spec.build_robolab_spec(TASK, seed=0, experiment="robolab-fixture",
                                                       resource_profile="robolab-slot2"), root=ROOT)
    assert default["runtime"]["official_limit_multiple"] == TOOL_BUDGET_LIMIT_MULTIPLE
    cell = spec.build_robolab_spec(TASK, seed=0, experiment="robolab-fixture", resource_profile="robolab-slot2",
                                   official_limit_multiple=1)
    assert cell["runtime"]["official_limit_multiple"] == 1
    plan = resolve_run_spec(cell, root=ROOT)
    assert plan["runtime"]["official_limit_multiple"] == 1
    assert plan["runtime"]["native_action_limit"] == TASKS[TASK]["native_action_limit"]
    with pytest.raises(RunSpecError, match="official_limit_multiple"):
        resolve_run_spec(spec.build_robolab_spec(TASK, seed=0, experiment="robolab-fixture",
                                                 resource_profile="robolab-slot2", official_limit_multiple=0),
                         root=ROOT)


def test_a_pending_task_composes_without_its_contract_and_preflight_refuses_it(simulator_host, workspace,
                                                                                  monkeypatch):
    """Admission is not review: the cell resolves, records why no contract was sent, and does not launch."""

    from ora.agent.composition.task_contract import delivery

    cell = spec.build_robolab_spec("GrabAFruitTask", seed=0, experiment="robolab-fixture",
                                   resource_profile="robolab-slot2")
    # GrabAFruitTask was promoted on 2026-11-16 with the last 20 tasks: it now launches with its contract.
    promoted = resolve_run_spec(cell, root=ROOT)
    assert promoted["policy"]["task_contract"]["delivered"] is True
    assert preflight.task_contract_check(promoted)["passed"] is True
    # No task is pending any more, so the pending state is staged by moving it back to pending_review.
    registry = delivery._registry

    def staged(kind, root):
        value, relative = registry(kind, root)
        if kind == "robolab":
            value = json.loads(json.dumps(value))
            value["pending_review"]["GrabAFruitTask"] = value["tasks"].pop("GrabAFruitTask")
        return value, relative

    monkeypatch.setattr(delivery, "_registry", staged)
    plan = resolve_run_spec(cell, root=ROOT)
    disclosure = plan["policy"]["task_contract"]
    assert disclosure["delivered"] is False and disclosure["reason"] == "contract_review_pending"
    assert plan["runtime"]["native_action_limit"] == 450
    assert preflight.task_contract_check(plan)["passed"] is False
    reviewed = resolve_run_spec(spec.build_robolab_spec(TASK, seed=0, experiment="robolab-fixture",
                                                        resource_profile="robolab-slot2"), root=ROOT)
    assert preflight.task_contract_check(reviewed)["passed"] is True


def test_a_cell_runs_its_tools_on_the_toolkit_and_the_environment_plans_its_moves(simulator_host, workspace):
    cell = spec.build_robolab_spec(TASK, seed=0, experiment="robolab-fixture", resource_profile="robolab-slot2")
    capabilities = resolve_run_spec(cell, root=ROOT)["policy_composition"]["tool_surface"]["capabilities"]
    assert "toolkit_only_v1" in capabilities and "toolkit_synchronous_execution_v1" not in capabilities


def test_the_spec_tool_writes_one_cell(simulator_host, workspace, capsys):
    output = workspace / "cell.json"
    assert spec.main([TASK, "--seed", "1", "--experiment", "robolab-fixture", "--resource-profile",
                      "robolab-slot2", "--config", "named", "--output", str(output)]) == 0
    assert capsys.readouterr().out.strip() == str(output)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["config"] == "named" and written["seed"] == 1 and written["runtime"]["renderer_gpu"] == 2
    assert "official_limit_multiple" not in written["runtime"]
    assert spec.main([TASK, "--seed", "1", "--experiment", "robolab-fixture", "--resource-profile",
                      "robolab-slot2", "--official-limit-multiple", "1", "--output", str(output)]) == 0
    capsys.readouterr()
    assert json.loads(output.read_text(encoding="utf-8"))["runtime"]["official_limit_multiple"] == 1
    with pytest.raises(SystemExit):
        spec.main(["NoSuchTask", "--seed", "0", "--experiment", "x", "--resource-profile", "robolab-slot2"])


def _runner(packages: dict, head: str = REVISION, dirty: str = "") -> object:
    def run(command, **kwargs):
        if command[0] == "git":
            stdout = head + "\n" if command[-1] == "HEAD" else dirty
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(packages) + "\n", stderr="")
    return run


def _context(runner) -> PreflightContext:
    return PreflightContext(project_root=ROOT, check=check, git=None, workspace_root=lambda root: root,
                            runner=runner)


def _installation(tmp_path: Path, monkeypatch) -> dict:
    """The three roots the runtime names, all present: interpreter, clean checkout, graphics runtime."""

    (tmp_path / "robolab/.git").mkdir(parents=True)
    (tmp_path / "graphics/graphics-compat").mkdir(parents=True)
    (tmp_path / "graphics/graphics-compat/installation.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ROBOLAB_PYTHON", sys.executable)
    monkeypatch.setenv("ROBOLAB_ROOT", str(tmp_path / "robolab"))
    monkeypatch.setenv("ROBOLAB_GRAPHICS_RUNTIME", str(tmp_path / "graphics"))
    return {"runtime": {"required_host": "sim-host", "python_env": "ROBOLAB_PYTHON", "root_env": "ROBOLAB_ROOT",
                        "graphics_runtime_env": "ROBOLAB_GRAPHICS_RUNTIME"},
            "policy": {"task_contract": {"task": TASK, "delivered": True,
                                         "reason": "reviewed_contract_matches_the_pinned_source"}}}


def test_preflight_admits_the_pinned_packages_checkout_and_graphics_runtime(simulator_host, tmp_path, monkeypatch):
    plan = _installation(tmp_path, monkeypatch)
    checks = preflight.preflight_runtime(plan, _context(_runner(dict(PACKAGES, robolab="0.1"))))
    assert [(item["check"], item["passed"]) for item in checks] == [
        ("execution_host", True), ("environment_python", True), ("robolab_packages", True),
        ("robolab_source", True), ("graphics_runtime", True), ("robolab_task_contract", True)]


@pytest.mark.parametrize("disclosure, passed", [
    ({"delivered": True, "reason": "reviewed_contract_matches_the_pinned_source"}, True),
    ({"delivered": False, "reason": "replaced_by_task_file"}, True),
    ({"delivered": False, "reason": "contract_review_pending", "semantic_review": "reviews/x.json"}, False),
    ({"delivered": False, "reason": "task_absent_from_verified_contract_registry"}, False),
    (None, False),
])
def test_preflight_launches_a_task_only_with_its_reviewed_contract(disclosure, passed):
    """A task whose generated contract still waits for its witness composes but does not launch."""

    plan = {"policy": {} if disclosure is None else {"task_contract": {"task": "AnimalsInBinTask", **disclosure}}}
    check = preflight.task_contract_check(plan)
    assert check["check"] == "robolab_task_contract" and check["passed"] is passed
    if disclosure and disclosure["reason"] == "contract_review_pending":
        assert check["detail"]["semantic_review"] == "reviews/x.json"


def test_preflight_refuses_other_packages_a_dirty_or_moved_checkout_and_a_missing_runtime(
    simulator_host, tmp_path, monkeypatch,
):
    plan = _installation(tmp_path, monkeypatch)
    passed = lambda checks: {item["check"]: item["passed"] for item in checks}  # noqa: E731
    checks = preflight.robolab_runtime(plan, runner=_runner({**PACKAGES, "isaacsim": "4.5.0"}, dirty=" M x.py\n"))
    assert passed(checks) == {"execution_host": True, "environment_python": True, "robolab_packages": False,
                              "robolab_source": False, "graphics_runtime": True}
    assert passed(preflight.robolab_runtime(plan, runner=_runner(PACKAGES, head="0" * 40)))["robolab_source"] is False
    unreadable = preflight.robolab_runtime(plan, runner=lambda command, **kwargs: subprocess.CompletedProcess(
        command, 1, stdout="", stderr="no isaacsim\n"))
    assert unreadable[2]["passed"] is False and unreadable[2]["detail"]["observed"] == {"error": "no isaacsim"}
    for variable in ("ROBOLAB_PYTHON", "ROBOLAB_ROOT", "ROBOLAB_GRAPHICS_RUNTIME"):
        monkeypatch.delenv(variable)
    assert passed(preflight.robolab_runtime(plan, runner=None)) == {
        "execution_host": True, "environment_python": False, "robolab_source": False, "graphics_runtime": False}


class PlanningSession(ScriptedSession):
    """A worker that routes as RoboLab does: a short move on the servo, free moves planned, a far one refused."""

    def execute(self, action):
        receipt = super().execute(action)
        free = [item for item in self.actions if item["action"] == "move_free_space"]
        if action["action"] == "move_pose":
            return {**receipt, "motion_strategy": "cartesian_servo_direct"}
        if action["action"] == "move_free_space" and len(free) == 2:
            return {**receipt, "execution_status": "plan_rejected", "failure_stage": "planner", "steps_used": 0,
                    "failure_code": "goal_unreachable"}
        if action["action"] in ("move_free_space", "recover_motion"):
            return {**receipt, "motion_strategy": "planned_joint_path", "motion": {
                "timing": {"implementation": "toppra_ruckig"}, "wrist": {"start_deg": 0.0}}}
        return receipt


def _qualify(monkeypatch, tmp_path: Path, session: ScriptedSession, argv: list[str]) -> tuple[int, dict]:
    built: dict = {}

    def build(**kwargs):
        built.update(kwargs)
        return session

    monkeypatch.setattr(qualification, "build_robolab_worker_session", build)
    monkeypatch.setattr(qualification, "motion_timing_check",
                        lambda _timing: {"check": "motion_timing_worker", "passed": True, "detail": {}})
    monkeypatch.setenv("CUROBO_V2_PYTHON", "/env/python")
    monkeypatch.setenv("CUROBO_V2_ROOT", str(tmp_path))
    return qualification.main(["--gpu", "2", "--output", str(tmp_path / "record"), *argv]), built


def test_the_qualification_script_records_every_check_on_the_named_card(monkeypatch, tmp_path, capsys):
    session = PlanningSession(reset={"official_success_at_reset": False, "task_language": INSTRUCTION},
                              verdict={"official_success": False, "native_actions": 9})
    status, built = _qualify(monkeypatch, tmp_path, session, ["--descriptor", qualification.DESCRIPTOR])
    assert status == 0 and session.closed
    assert built["task"] == TASK and built["runtime"]["renderer_gpu"] == 2
    assert built["runtime"]["motion_planner"]["robot"] == "franka_robotiq"
    assert built["runtime"]["motion_planner"]["python_executable"] == "/env/python"
    assert built["runtime"]["public_geometry_descriptor"]["path"] == qualification.DESCRIPTOR
    record = json.loads((tmp_path / "record" / qualification.RECORD_NAME).read_text(encoding="utf-8"))
    assert record["schema"] == qualification.SCHEMA and record["passed"] is True and record["renderer_gpu"] == 2
    assert [item["check"] for item in record["checks"]] == [
        "reset", "observation_rgbd", "proprioception", "move_pose_precision", "move_free_space",
        "planned_path_timed", "planner_refuses_unreachable", "set_gripper_close", "set_gripper_open", "wait", "recover_home", "status_running", "official_verdict",
        "close"]
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["passed"] is True


def test_a_failed_close_fails_the_record_and_a_lost_worker_still_closes(monkeypatch, tmp_path, capsys):
    session = ScriptedSession(reset={"official_success_at_reset": False}, verdict={"official_success": False},
                              close_error=RuntimeError("Kit did not exit"))
    status, built = _qualify(monkeypatch, tmp_path, session, [])
    assert status == 1 and "public_geometry_descriptor" not in built["runtime"]
    assert any(line.startswith("FAIL ") and "Kit did not exit" in line for line in capsys.readouterr().out.splitlines())
    lost = ScriptedSession(reset={}, verdict={}, fail_at="move_free_space")
    with pytest.raises(RuntimeError, match="lost move_free_space"):
        _qualify(monkeypatch, tmp_path / "lost", lost, [])
    assert lost.closed
