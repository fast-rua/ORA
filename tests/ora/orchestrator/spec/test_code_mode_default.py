"""Code mode by default where the baseline says so (``code_mode`` of a baseline catalog, ``spec/defaults.py``).

RoboDojo's default baseline, ``robodojo_toolkit_v7``, runs code mode with the extended library, prompt v3 and ORA's
source, without forced preview (decided 2026-11-05 from the code-mode comparison on 41 RoboDojo tasks): a RunSpec that names no ``policy.code_mode`` gets it,
with the code-mode context scheme its baseline selects. A RunSpec's own ``policy.code_mode`` wins; ``null`` (or
``false``) is tool calls, the opt-out. Free play and a context pack get no default. Resolution reads the catalog alone,
never the host's sandbox. Resolution only; nothing is launched.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ora.agent.composition.baseline import default_baseline_id, load_native_baseline
from ora.orchestrator.evaluation.model_visible import model_visible_items
from ora.orchestrator.spec.defaults import expand_native_defaults
from ora.orchestrator.spec.run_spec import resolve_run_spec
from tests.support.orchestrator.run_specs import native_spec, robodojo_native_spec

ROOT = Path(__file__).resolve().parents[4]
EXPECTED = json.loads((ROOT / "tests/data/default-expectations.json").read_text(encoding="utf-8"))["code_mode"]
DEFAULT = {"library": "extended", "prompt": "v3", "source": True}
PLANNER = "ora/policy/planners/native/code-mode-coding-v3.md"
DRY_RUN = "ora/policy/planners/native/code-mode-dry-run-v1.md"


def _plan(**policy):
    spec = robodojo_native_spec("general_pickup")
    spec["policy"].update(policy)
    return resolve_run_spec(spec, root=ROOT)


def _advertised(plan):
    return {item["name"] for item in plan["policy_composition"]["tool_surface"]["definitions"]}


def _system(plan):
    return [layer["path"] for layer in plan["policy_context"]["harness_system"]]


def test_the_contract_names_each_backends_default_code_mode():
    for environment, expected in EXPECTED.items():
        if environment == "description":
            continue
        catalog = load_native_baseline(ROOT, default_baseline_id(environment))
        assert catalog.get("code_mode") == expected, environment


def test_a_robodojo_runspec_that_names_none_runs_the_validated_code_mode():
    plan = _plan()
    policy = plan["policy"]
    contract = policy["code_mode"]
    assert {key: contract[key] for key in DEFAULT} == DEFAULT
    assert "preview" not in contract and contract["script_reads"] == ["dry_run", "query", "source"]
    assert policy["baseline"]["id"] == "robodojo_toolkit_v7" and policy["baseline"]["selection"] == "omitted_tool_surface"
    assert policy["context_scheme"]["id"] == "context_keep_written"
    # Optional dry_run: the v3 planner and its dry-run layer.
    system = _system(plan)
    assert system[0] == PLANNER and DRY_RUN in system
    assert {"read", "write", "edit", "bash"} <= _advertised(plan) and "move_pose" not in _advertised(plan)
    # The same plan, field for field, as a RunSpec that names that code mode itself.
    named = _plan(code_mode=copy.deepcopy(DEFAULT))
    assert named["policy"] == policy and named["policy_context"] == plan["policy_context"]
    assert model_visible_items(named, root=ROOT) == model_visible_items(plan, root=ROOT)


@pytest.mark.parametrize("off", [None, False])
def test_null_keeps_tool_calls(off):
    plan = _plan(code_mode=off)
    policy = plan["policy"]
    assert "code_mode" not in policy  # resolved as every tool-call plan before the default
    assert policy["baseline"]["id"] == "robodojo_toolkit_v7"
    assert policy["context_scheme"]["id"] == "context_keep_written"
    assert {"move_pose", "grasp_plan", "wait_ticks"} <= _advertised(plan) and not {"bash", "edit"} & _advertised(plan)
    assert not any("code-mode" in path for path in _system(plan))
    assert not any(layer["kind"] == "code_mode_v1"
                   for layer in plan["policy_composition"]["tool_surface"]["derived_layers"])


def test_a_runspec_naming_another_code_mode_keeps_it():
    plan = _plan(code_mode={"library": "extended", "prompt": "v3"})
    assert "source" not in plan["policy"]["code_mode"] and plan["policy"]["code_mode"]["script_reads"] == [
        "dry_run", "query"]
    plan = _plan(code_mode=True)
    assert "library" not in plan["policy"]["code_mode"] and plan["policy"]["code_mode"]["prompt"] == "v2"


def test_the_previous_robodojo_baseline_and_its_surface_keep_tool_calls():
    for selection in ({"baseline": "robodojo_toolkit_v6"},
                      {"tool_surface": {"profile": "robodojo_placement_toolkit_v5"}}):
        spec = robodojo_native_spec("general_pickup")
        spec["policy"].update(selection)
        spec["toolkit_grants"] = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
        plan = resolve_run_spec(spec, root=ROOT)
        assert plan["policy"]["baseline"]["id"] == "robodojo_toolkit_v6"
        assert "code_mode" not in plan["policy"] and "move_pose" in _advertised(plan)
    # Naming the default's surface selects the default baseline, code mode included.
    spec = robodojo_native_spec("general_pickup")
    spec["policy"]["tool_surface"] = {"profile": "robodojo_placement_toolkit_v6"}
    assert resolve_run_spec(spec, root=ROOT)["policy"]["code_mode"]["library"] == "extended"


@pytest.mark.parametrize("environment", ["libero_pro", "robotwin"])
def test_the_other_defaults_keep_tool_calls(environment):
    plan = resolve_run_spec(native_spec(environment), root=ROOT)
    assert "code_mode" not in plan["policy"]
    assert plan["policy"]["context_scheme"]["id"] == "context_lossless_v6"


def _expanded(**extra):
    spec = {"policy": {"kind": "native_api", "supports_images": True}, "runtime": {"kind": "robodojo"}, **extra}
    return expand_native_defaults(spec, root=ROOT)


def test_free_play_and_a_context_pack_get_no_default():
    assert "code_mode" in _expanded()["policy"]
    assert "code_mode" not in _expanded(play={})["policy"]
    pack = {"path": "/absolute/pack", "manifest_sha256": "0" * 64}
    spec = {"policy": {"kind": "native_api", "supports_images": True, "context_pack": pack},
            "runtime": {"kind": "robodojo"}}
    assert "code_mode" not in expand_native_defaults(spec, root=ROOT)["policy"]


def test_resolution_does_not_look_at_the_host(monkeypatch, tmp_path):
    """The plan is the RunSpec's and the catalog's: no sandbox probe, the same plan with or without an interpreter."""
    from ora.agent.code_mode import sandbox

    def refuse(*args, **kwargs):
        raise AssertionError("plan resolution probed the code-mode sandbox")

    for name in ("sandbox_status", "prerequisite_checks", "python_path", "bwrap_path", "namespace_refusal"):
        monkeypatch.setattr(sandbox, name, refuse)
    monkeypatch.delenv("ORA_CODE_MODE_PYTHON", raising=False)
    monkeypatch.setenv("ORA_CODE_MODE_BWRAP", str(tmp_path / "no-bwrap"))
    monkeypatch.setenv("ORA_GPU_PREFIX", str(tmp_path / "empty-prefix"))
    bare = [_plan(), _plan(code_mode=None)]
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    monkeypatch.setenv("ORA_CODE_MODE_PYTHON", str(interpreter))
    installed = [_plan(), _plan(code_mode=None)]
    for before, after in zip(bare, installed):
        assert before["policy"] == after["policy"] and before["policy_context"] == after["policy_context"]
        assert before["policy_identity"] == after["policy_identity"]
