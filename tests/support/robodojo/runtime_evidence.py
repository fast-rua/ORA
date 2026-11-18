"""Test helpers for robodojo runtime evidence, shared by several test modules.

They were defined in ``tests/ora/orchestrator/backends/robodojo/test_robodojo_runtime.py``;
test modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REVISIONS = {name: str(index) * 40 for index, name in enumerate(
    ("robodojo", "isaaclab", "curobo", "xpolicylab", "assets"), start=1,
)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _source_record() -> dict:
    return {
        "schema": "ora.robodojo-source-manifest.v1",
        "task": "fixture_task", "eval_seed": 0, "layout_id": 0, "num_envs": 1,
        "source_revisions": REVISIONS, "required_revisions": REVISIONS,
        "layout_file": {"path": "/fixture/Assets/Eval_Layout/0/0.json", "sha256": "a" * 64},
        "action_codec": "runtime_action_v3", "sensor_profile": "fixture_rgbd_v1",
        "physics_dt_s": 0.004, "control_hz": 25, "official_native_action_limit": 200,
    }


def _round_isolation(clock: int) -> dict:
    """The measured restore proof a RoboDojo round-reset record must carry."""

    return {
        "schema": "ora.robodojo-round-isolation.v1",
        "passed": True,
        "checks": {
            "arm_home_restored": True,
            "carried_channels_cleared": True,
            "episode_counters_zeroed": True,
            "failure_latch_cleared": True,
            "layout_identity_unchanged": True,
            "official_lifecycle_rebound": True,
            "official_terminal_cleared": True,
        },
        "measured": {
            "arms": ["left", "right"],
            "cleared_channels": ["official_terminal"],
            "end_flag": False,
            "failure_latch_clear": True,
            "layout_sha256": "a" * 64,
            "layout_sha256_at_start": "a" * 64,
            "max_home_position_error_m": 0.0,
            "max_home_rotation_error_deg": 0.0,
            "native_action_limit": 200,
            "physics_steps_before_policy": clock,
            "reward_registered": True,
            "step_lim": 200,
            "stale_channels": [],
        },
        "tolerances": {"home_position_m": 0.02, "home_rotation_deg": 5.0},
        "failures": [],
    }


def _plan() -> dict:
    paths = {
        "codex_project": "ora/policy/planners/native/project-v1.md",
        "episode_prefix": "ora/policy/core/thin-react-v1.md",
        "tool_manual": "ora/policy/embodiments/aloha-agilex/manual-v1.md",
    }
    return {
        "schema": "ora.run-plan.v1", "plan_sha256": "fixture-plan",
        "plan_id": "robodojo-fixture", "experiment": "runtime-v1", "config": "fixture",
        "milestone": "M0", "output_workspace": "outputs/runtime-v1",
        "task": "fixture_task", "task_instruction": "Move safely once.", "seed": 0,
        "evidence_tier": "diagnostic", "source": {"commit": "fixture", "dirty": False},
        "policy": {
            "kind": "scripted", "model": "scripted-policy-v1", "provider": "trusted_scripted",
            "reasoning_effort": "none", "action_codec": "runtime_action_v3",
            # RoboDojo takes no target-state v4; this surface's move_pose compiles ora-action.v3.
            "tool_surface": {"profile": "native_atomic_causal_v1", "component_overrides": {}},
            "tool_allowlist": ["move_pose", "close_gripper", "open_gripper"],
            "actions": [{"tool": "move_pose", "xyz_m": [0, 0, position]} for position in (1.02, 1.04)],
        },
        "runtime": {
            "kind": "robodojo", "transport": "local_rpc", "eval_seed": 0, "layout_id": 0,
            "controlled_end_effector": "left",
            "layout_file": "Eval_Layout/0/0.json", "layout_sha256": "a" * 64,
            "sensor_profile": "fixture_rgbd_v1", "physics_dt": 0.004,
            "control_hz": 25, "native_action_limit": 200,
            "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
            "required_revision": REVISIONS["robodojo"],
            "required_isaaclab_revision": REVISIONS["isaaclab"],
            "required_curobo_revision": REVISIONS["curobo"],
            "required_xpolicylab_revision": REVISIONS["xpolicylab"],
            "required_asset_revision": REVISIONS["assets"],
        },
        "budgets": {"max_robot_decisions": 2, "max_model_calls": 3, "max_sim_steps": 50,
                    "wall_time_s": 30},
        "policy_context": {
            **{name: [{"path": path, "sha256": _sha(ROOT / path)}] for name, path in paths.items()},
            "composition_sha256": "fixture",
        },
        "qualification": {"required": False},
    }


def _multi_round_evidence(
    tmp_path: Path, *, rounds: int = 3, perturbation: str | None = None,
) -> tuple[Path, dict, dict, list[dict]]:
    """Lay out one persistent K-round RoboDojo evidence tree and its trace.

    The Supervisor, RPC transport and worker are exercised by the fixture
    above. This builder only reproduces the artifacts and causal trace a
    finished K-round session leaves behind, so one perturbation at a time can
    be joined against the real evaluator. The PhysX clock keeps advancing
    across a round reset while the wrapper's own counters restart, which is
    exactly the join a session-wide sum would get wrong.
    """

    run_root = tmp_path / "run"
    native = run_root / "native"
    _write(native / "source-manifest.json", _source_record())
    source_hash = _sha(native / "source-manifest.json")
    plan = _plan()
    plan["session"] = {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": rounds,
        "budget_scope": "per_round", "reset_owner": "supervisor",
    }
    reset_clocks = [100 + 400 * index for index in range(rounds)]
    events: list[dict] = [{"type": "run_started", "run_id": "robodojo-rounds"}]
    round_records: list[dict] = []
    sequence = 0
    for index in range(rounds):
        clock = reset_clocks[index]
        reset = {
            "schema": (
                "ora.robodojo-reset.v1" if index == 0
                else "ora.robodojo-round-reset.v1"
            ),
            "task": plan["task"], "eval_seed": 0, "seed": 0, "layout_id": 0,
            "round_index": index, "reward_registered": True,
            "physics_steps_before_policy": clock,
            "source_manifest": {
                "path": "source-manifest.json", "sha256": source_hash,
            },
        }
        if index:
            reset["artifact_sequences_preserved"] = {
                "observation_sequence": sequence,
                "action_sequence": sequence,
                "motion_probe_sequence": 0,
            }
            reset["round_isolation"] = _round_isolation(clock)
            _write(native / "round-resets" / f"round-{index:04d}.json", reset)
        events.append({"type": "reset", "record": reset, "round_index": index})
        crossing = perturbation == "receipts_cross_rounds"
        carried_native = 2 * index if crossing else 0
        carried_physics = 20 * index if crossing else 0
        base_clock = (
            reset_clocks[0]
            if crossing
            or (perturbation == "wrong_round_clock" and index == rounds - 1)
            else clock
        )
        for step in range(2):
            action_id = f"round-{index:04d}-action-{step:04d}"
            journal_path = native / "controller" / f"action-{sequence:04d}.jsonl"
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            journal_path.write_text("".join(
                json.dumps({
                    "action_id": action_id, "segment": "command",
                    "physics_clock": (
                        base_clock + carried_physics + step * 10 + offset + 1
                    ),
                    "native_action": carried_native + step + 1,
                }, sort_keys=True) + "\n"
                for offset in range(10)
            ))
            control_path = native / "controller" / f"action-{sequence:04d}.json"
            _write(control_path, {
                "schema": "ora.robodojo-controller-trace.v1",
                "action_id": action_id,
                "samples": [{
                    "native_action": carried_native + step + 1,
                    "physics_steps": carried_physics + (step + 1) * 10,
                }],
                "physics_journal": {
                    "path": journal_path.relative_to(native).as_posix(),
                    "sha256": _sha(journal_path),
                },
            })
            receipt = {
                "schema": "ora.robodojo-sparse-motion-receipt.v1",
                "action_id": action_id, "action": "move_pose",
                "round_index": index, "consumed": True, "state_changed": True,
                "native_actions": 1, "physics_steps": 10,
                "control_trace": {
                    "path": control_path.relative_to(native).as_posix(),
                    "sha256": _sha(control_path),
                },
            }
            receipt_path = native / "receipts" / f"action-{sequence:04d}.json"
            _write(receipt_path, receipt)
            events.append({
                "type": "action_submitted", "action": {"action_id": action_id},
            })
            events.append({"type": "motion_receipt", "record": {
                "path": receipt_path.relative_to(native).as_posix(),
                "sha256": _sha(receipt_path), "receipt": receipt,
            }})
            sequence += 1
        verdict = {
            "schema": "ora.robodojo-official-verdict.v2",
            "source": "RoboDojo EvalEnv.is_episode_end",
            "source_manifest_sha256": source_hash,
            "task": plan["task"], "eval_seed": 0, "seed": 0, "layout_id": 0,
            "round_index": index, "official_success": index == 0,
            "terminal": True, "abort_reason": None,
            "verdict_basis": "official_terminal_transition", "fault_reason": None,
            "native_actions": 2, "physics_steps": 20,
            "reset_physics_steps": clock,
        }
        if perturbation == "verdict_native_count_mismatch" and index == 1:
            verdict["native_actions"] = 3
        if perturbation in {"budget_round", "faulted_round_claims_boolean"} and index == 1:
            # Round 1 ran out of robot decisions: no terminal transition, but
            # the standing official predicate is still a public Boolean.
            verdict.update({
                "official_success": False, "terminal": False,
                "abort_reason": "robot_decision_budget",
                "verdict_basis": "standing_official_predicate_at_policy_budget",
            })
        if perturbation == "faulted_round_claims_boolean" and index == 1:
            verdict["fault_reason"] = "robodojo_native_execution_failed"
        _write(
            native / "rounds" / f"round-{index:04d}" / "official-verdict.json",
            verdict,
        )
        if index == rounds - 1:
            _write(native / "official-verdict.json", verdict)
        authority = (
            "experiment_budget" if verdict["verdict_basis"].startswith("standing")
            else "environment"
        )
        if perturbation == "faulted_authority_claims_boolean" and index == 1:
            authority = "infrastructure"
        round_records.append({
            "schema": "ora.round-result.v1",
            "round_index": index, "state": "completed",
            "termination_authority": authority,
            "official_success": verdict["official_success"], "verdict": verdict,
        })
        events.append({"type": "round_finished", "round_index": index})
    if perturbation == "missing_archived_verdict":
        (native / "rounds/round-0001/official-verdict.json").unlink()
    result = {
        "schema": "ora.session-result.v1",
        "state": "completed", "official_success": None, "verdict": None,
        "termination": "session_round_limit", "rounds": round_records,
        "round_count": rounds,
        "round_success_count": sum(
            record["official_success"] is True for record in round_records
        ),
    }
    return run_root, plan, result, events
