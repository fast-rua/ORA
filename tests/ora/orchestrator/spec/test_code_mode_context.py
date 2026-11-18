"""Code mode's context on the default baseline (``code_mode_context_scheme``, ``spec/defaults.py``).

``native_full_explicit_motion_direct_v4`` names ``context_lossless_v6`` (management v6) for tool calls and
``context_keep_written`` (management v10: scripts, messages and reasoning never age, data ages from 100,000 tokens,
L6 by exchange; the context experiment's best arm, the default since 2026-11-05) for code mode, on every instance: a
code-mode policy's calls complete no ActionStep for long stretches. Code mode then delivers the scheme's prompt
after its planner, and the round's static facts once (in that prompt, not in their own layer). Forced preview was
retired on 2026-11-05; a dry run stays optional. Resolution only; nothing is launched.
"""
from __future__ import annotations

import pytest

from ora.orchestrator.spec.run_spec import resolve_run_spec
from tests.support.orchestrator.run_specs import native_spec

#: Code mode as a plan sets it (the default's own value).
CODE = {"library": "extended", "prompt": "v3", "source": True}
BACKENDS = ("libero_pro", "robotwin", "robocasa", "robolab")
SCHEME = "context_keep_written"
SCHEME_PROMPT = "ora/policy/context_schemes/keep-written.md"
PLANNER = "ora/policy/planners/native/code-mode-coding-v3.md"


def _plan(environment: str, **policy):
    spec = native_spec(environment)
    if environment in ("robocasa", "robolab"):
        # Their runtimes refuse a task, seed or language that is not the RunSpec's (the documented cells).
        if environment == "robocasa":
            task, instruction, extra = "OpenDrawer", "Open the left drawer.", {}
        else:
            from ora.environment.benchmarks.robolab import TASKS

            task, extra = "BananaInBowlTask", {"renderer_gpu": 0}
            instruction = TASKS[task]["instruction"]
        spec["runtime"].update(task=task, seed=0, expected_task_language=instruction, **extra)
        spec.update(task=task, task_instruction=instruction, seed=0)
    spec["policy"].update(policy)
    return resolve_run_spec(spec)


def _paths(plan, channel):
    return [layer["path"] for layer in plan["policy_context"][channel]]


@pytest.mark.parametrize("environment", BACKENDS)
def test_tool_calls_keep_the_baselines_v6_context(environment):
    plan = _plan(environment)
    assert plan["policy"]["baseline"]["id"] == "native_full_explicit_motion_direct_v4"
    assert plan["policy"]["context_scheme"]["id"] == "context_lossless_v6"
    assert plan["policy"]["context_management"]["schema"] == "ora.native-context-management.v6"
    assert not any(path.startswith("ora/policy/context_schemes/") for path in _paths(plan, "harness_system"))


@pytest.mark.parametrize("environment", BACKENDS)
def test_code_mode_keeps_what_the_model_wrote_with_the_schemes_prompt(environment):
    plan = _plan(environment, code_mode=CODE)
    policy = plan["policy"]
    assert policy["context_scheme"]["id"] == SCHEME
    management = policy["context_management"]
    assert management["schema"] == "ora.native-context-management.v10"
    # Scripts and reasoning never age; L6 counts exchanges, ratio-only at 0.6.
    assert (management["aged_argument_bytes"], management["reasoning_replay"], management["compaction_unit"]) == (
        None, "all", "exchange")
    assert (management["compaction_trigger_tokens"], management["emergency_fallback_ratio"],
            management["emergency_target_ratio"]) == (None, 0.6, 0.5)
    assert policy["context_tools"] == ["artifact_read"]
    assert "preview" not in policy["code_mode"]
    system = _paths(plan, "harness_system")
    assert system[0] == PLANNER
    assert system[-1] == SCHEME_PROMPT
    # The scheme's prompt states the round's static facts; their own layer is not repeated.
    assert "ora/policy/core/round-static-facts-v1.md" not in _paths(plan, "project_instructions")
    assert _paths(plan, "project_instructions")[0] == "ora/policy/core/code-policy-loop-v2.md"
    names = {item["name"] for item in plan["policy_composition"]["tool_surface"]["definitions"]}
    assert "artifact_read" in names or "artifact.read" in names


def test_a_named_scheme_still_wins_in_code_mode():
    plan = _plan("libero_pro", code_mode=CODE, context_scheme="context_keep_written_unaged_reasoning")
    assert plan["policy"]["context_management"]["reasoning_replay"] == "unaged_exchanges"
    assert _paths(plan, "harness_system")[-1] == "ora/policy/context_schemes/keep-written.md"
    # Its prompt states the static facts too.
    assert "ora/policy/core/round-static-facts-v1.md" not in _paths(plan, "project_instructions")


def test_a_named_scheme_without_a_prompt_keeps_the_static_facts_layer():
    plan = _plan("libero_pro", code_mode=CODE, context_scheme="context_lossless_v6")
    assert plan["policy"]["context_management"]["schema"] == "ora.native-context-management.v6"
    assert not any(path.startswith("ora/policy/context_schemes/") for path in _paths(plan, "harness_system"))
    assert "ora/policy/core/round-static-facts-v1.md" in _paths(plan, "project_instructions")


def test_the_default_activation_contract_names_each_backends_schemes():
    """``tests/data/default-expectations.json`` ``context_schemes``: what each backend's default baseline selects."""
    import json
    from pathlib import Path

    from ora.agent.composition.baseline import default_baseline_id, load_native_baseline

    root = Path(__file__).resolve().parents[4]
    expected = json.loads((root / "tests/data/default-expectations.json").read_text(encoding="utf-8"))["context_schemes"]
    for environment, scheme in expected["tool_call"].items():
        catalog = load_native_baseline(root, default_baseline_id(environment))
        assert catalog["context_scheme"] == scheme, environment
        assert catalog.get("code_mode_context_scheme", catalog["context_scheme"]) == expected["code_mode"][environment]
