"""Unqualified runs remain explicit and do not masquerade as qualified ones."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ora.orchestrator import robodojo_qualification as qualification
from ora.orchestrator.backends.robodojo import qualification_admission as admission
from ora.orchestrator.backends.robodojo.supervision import supervisor_binding
from ora.orchestrator.episode.task_language import bind_task_instruction
from ora.orchestrator.faults import SupervisorError
from tests.support.robodojo.admission import _runtime


@pytest.mark.parametrize("missing", ["binding", "combined", "card", "task"])
def test_missing_record_is_evidence_not_a_live_worker_binding(tmp_path, monkeypatch, missing):
    monkeypatch.setattr(qualification, "worker_kernel_sha256", lambda _: "f" * 64)
    runtime = _runtime()
    ref = runtime.pop("infrastructure_qualification")
    if missing == "combined":
        runtime["infrastructure_qualification"] = ref
    elif missing in {"card", "task"}:
        runtime["infrastructure_qualification"] = {"card": ref, "task": ref}
    record = admission.admission(runtime, workspace_root=tmp_path, harness_commit="c" * 40)
    assert record["qualification_available"] is False
    assert record["plan_worker_kernel_sha256"] == "f" * 64
    assert "passed" not in record
    (tmp_path / "preflight.json").write_text(json.dumps({"checks": [{
        "check": "robodojo_infrastructure_qualification", "passed": True, "detail": record}]}))
    assert supervisor_binding(tmp_path, runtime) is None


@pytest.mark.parametrize("damage", ["json", "hash", "escape"])
def test_damaged_qualification_is_not_downgraded(tmp_path, monkeypatch, damage):
    runtime = _runtime()
    ref = runtime["infrastructure_qualification"]
    path = tmp_path / ref["path"]
    path.parent.mkdir(parents=True)
    path.write_text("broken" if damage == "json" else "{}")
    if damage != "hash":
        ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if damage == "escape":
        ref["path"] = "../outside"
    with pytest.raises(ValueError):
        admission.admission(runtime, workspace_root=tmp_path, harness_commit="c" * 40)


def test_live_language_binds_after_reset_without_rewriting_plan(tmp_path):
    plan = {"task": "general_pickup", "task_instruction": "Pending reset.",
            "runtime": {"kind": "robodojo", "layout_id": 9, "task_language_source": "live_reset_readback"}}
    loop = SimpleNamespace(plan=plan, native=tmp_path / "native", events=tmp_path / "events.jsonl")
    bind_task_instruction(loop, {"task_instruction": "Pick up the small cup."})
    assert loop.task_instruction == "Pick up the small cup."
    assert plan["task_instruction"] == "Pending reset."
    record = json.loads((loop.native / "task-language-binding.json").read_bytes())
    assert record["gate_verified"] is False and record["layout_id"] == 9
    with pytest.raises(SupervisorError, match="no official task instruction"):
        bind_task_instruction(loop, {})


def test_live_language_rebinds_terminal_manifest_and_audit(tmp_path):
    from ora.orchestrator.episode.terminal import _terminal_turn
    from ora.orchestrator.evaluation.context_audit import _policy_layers
    from tests.support.orchestrator.runtime_v1 import synthetic_plan, PROMPT, MANUAL, AGENT_RULES

    plan = synthetic_plan()
    plan["runtime"].update(kind="robodojo", layout_id=9, task_language_source="live_reset_readback")
    plan["task_instruction"] = "Pending official task instruction from environment reset."
    plan["policy"].update(kind="native_api", provider="openai", supports_images=True,
                          context_profile="native_minimal_v1", summary_policy="none",
                          context_window_tokens=10000, max_output_tokens=1000)
    prefix = tmp_path / "task-prefix.md"
    prefix.write_text("Task: {{TASK}}")
    root = tmp_path / "outputs/experiment/runs/run"
    loop = SimpleNamespace(
        plan=plan, native=root / "native", events=root / "events.jsonl", run_root=root,
        run_id="run", compiled_policy={"codex_project": AGENT_RULES, "episode_prefix": prefix, "tool_manual": MANUAL},
        pointer={"main_image": "image.png"}, workspace=root, round_decisions=0, round_model_calls=0,
        interaction=SimpleNamespace(observation_id="obs-0", observation_step=0),
        native_allowance=SimpleNamespace(fields=lambda: {}),
    )
    bind_task_instruction(loop, {"task_instruction": "Pick up the small cup."})
    assert _terminal_turn(loop, terminal_lifecycle=True).task_instruction == "Pick up the small cup."
    manifest = json.loads((root / "context-manifest.json").read_bytes())
    text = manifest["context_order"][0]["exact_text"]
    assert "Pick up the small cup." in text and "Pending official" not in text
    task = next(row for row in _policy_layers(root, plan) if row["channel"] == "task")
    assert task["source_path"] == "native/task-language-binding.json:task_instruction"
    assert task["source_sha256"] == hashlib.sha256(b"Pick up the small cup.").hexdigest()
    assert plan["task_instruction"].startswith("Pending official")
