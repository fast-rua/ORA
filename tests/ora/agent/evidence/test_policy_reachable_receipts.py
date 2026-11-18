"""Nothing the policy can read names a timing or planner library; the sealed receipt keeps it all.

A paced move's receipt carries its timing evidence (``motion.timing``, ``pacing.timing``) and the
planner's own record (``planner.schema``, ``planner.canonical_artifact``). The sealed receipt under
``native/receipts/`` keeps them. Every copy the policy can read -- the environment's observation
files, the policy-state chain, the context a session reads (and a script's robot view, built from
it) -- is ``policy_receipt`` of it, and a receipt the session stores for its own trace is not a
Context Artifact, so ``artifact_read`` cannot open it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.evidence.policy_state import PolicyStateStore
from ora.agent.evidence.receipt_binding import canonical_public_receipt
from ora.agent.context.observation.projector import ContextProjector
from ora.agent.memory.artifact_reader import RefUnknownError
from ora.agent.memory.catalog import ContextArtifactCatalog
from ora.contracts.policy_receipt import names_timing, policy_receipt
from tests.support.agent.atomic_causal_runtime import MANUAL, PROJECT, PROMPT
from tests.support.agent.public_robot_facts import fixture

#: What no policy-readable file may hold: the timing libraries and evidence, and the planner's identity.
FORBIDDEN = ("toppra", "ruckig", "motion_pace", "support.franka.de", '"timing":', "curobo_v2", "robot_sha256",
             "collision_spheres_sha256", "ora.curobo-plan-summary", "curobo/plans")
#: Sealed evidence the policy cannot read.
SEALED = ("native/receipts/", "native/agent-artifacts/")

TIMING = {"implementation": "toppra_ruckig", "versions": {"toppra": "0.6.10", "ruckig": "0.19.4"},
          "jerk_source": "Franka (https://support.franka.de/docs/control_parameters.html)", "attempts": 1}
PLANNER = {"schema": "ora.curobo-plan-summary.v1", "status": "planned", "reachable": True,
           "canonical_artifact": {"path": "curobo/plans/action-0000.json", "sha256": "c" * 64}}
PACING = {"speed": {"speed": "fast", "slow_approaching": False}, "wrist": {"travel_deg": 0.0},
          "planned_ticks": 9, "duration_s": 0.45, "timing": TIMING}


def _receipt() -> dict:
    return {"schema": "ora.libero-sparse-motion-receipt.v1", "action": "move_pose", "action_id": "action-0",
            "motion": dict(PACING), "segment": {"pacing": dict(PACING), "planner": dict(PLANNER)},
            "segments": [{"pacing": dict(PACING), "planner": dict(PLANNER)}]}


def _policy_readable(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not relative.startswith(SEALED) and path.suffix in (".json", ".md", ".jsonl", ""):
            files[relative] = path.read_text(encoding="utf-8", errors="replace")
    return files


def test_no_policy_readable_file_names_a_timing_or_planner_library(tmp_path):
    surface, observation, source = fixture(tmp_path, "libero")
    sealed = tmp_path / "native/receipts/action-0000.json"
    sealed.parent.mkdir(parents=True)
    sealed.write_text(json.dumps(_receipt()))
    wrapper = {"path": "receipts/action-0000.json", "sha256": hashlib.sha256(sealed.read_bytes()).hexdigest(),
               "receipt": _receipt()}
    assert names_timing(wrapper)
    # The environment embeds the policy's copy (as LIBERO's observation does) ...
    observation["previous_execution_receipt"] = policy_receipt(wrapper)
    (source / "observation.json").write_text(json.dumps(observation))
    # ... and the Supervisor hands both writers the receipt as the action returned it.
    store = PolicyStateStore(tmp_path, "public-data", "plan")
    budgets = {"model_calls": 11, "robot_decisions": 7, "wall_time_s": 179.5}
    store.publish(observation=observation, observation_root=source, cause="action",
                  task_instruction="Place the bowl.", remaining_budget=budgets, previous_public_action=None,
                  previous_execution_receipt=wrapper)
    projector = ContextProjector(tmp_path / "workspace", tmp_path / "control", PROMPT, MANUAL, PROJECT,
                                 main_camera=observation["main_camera"], run_root=tmp_path,
                                 view_config=surface.view_config)
    pointer = projector.publish(observation=observation, observation_root=source, task_instruction="Place the bowl.",
                                remaining_budget=budgets, policy_state=store.steps[-1],
                                previous_execution_receipt=wrapper)
    # The session keeps the receipt for its trace (``native/agent-artifacts``), not as a Context Artifact.
    artifacts = ArtifactStore(tmp_path)
    stored = artifacts.put_json(wrapper, kind="environment_receipt")
    catalog = ContextArtifactCatalog(tmp_path, "native-policy-test", artifacts, reader_name="artifact_read")
    with pytest.raises(RefUnknownError):
        catalog.resolve(f"artifact://native-policy-test/{stored['sha256']}")

    readable = _policy_readable(tmp_path)
    assert {"workspace/inputs/current-context.json", "workspace/inputs/current/observation.json"} <= set(readable)
    assert any(name.startswith("policy-state/") for name in readable)
    leaks = {name: [token for token in FORBIDDEN if token in text] for name, text in readable.items()}
    assert {name: found for name, found in leaks.items() if found} == {}
    # What the policy reads is still the move: its speed, its wrist and the planner's verdict.
    public = pointer["previous_execution"]["receipt"]
    assert public["segment"]["pacing"]["planned_ticks"] == 9 and public["segment"]["planner"]["status"] == "planned"
    # The sealed receipt keeps every piece of evidence, and the policy's copy is still bound to it.
    assert "toppra_ruckig" in sealed.read_text() and "curobo/plans" in sealed.read_text()
    state = json.loads((tmp_path / store.steps[-1]["path"]).read_text())
    assert canonical_public_receipt(state, tmp_path) == policy_receipt(wrapper)["receipt"]
