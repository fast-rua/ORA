"""The RoboTwin seed-manifest rules: which seeds a manifest admits, and which it shows the expert failed.

Launch preflight (``seed_qualification``), the RoboTwin RunSpec builder
(``spec.benchmark``), the play-scene listing and the expert-unsolved
qualification exception all apply these rules; the seed scans that write the
manifests (``evaluation_seeds``, ``seed_segments``) name the schema from here.
The module imports nothing of ORA, so a reader loads only the rules.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SEED_MANIFEST_SCHEMA = "ora.robotwin-evaluation-seed-manifest.v1"


def digest(value: Any) -> str:
    """A manifest's own identity: the sha256 of ``value`` as sorted, compact JSON (``evidence_io.digest``)."""

    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


#: An expert-unsolved seed is run only on scan evidence that the expert failed or raised on it.
EXPERT_UNSOLVED_CLASSIFICATIONS = frozenset({"oracle_failure", "oracle_exception"})


def expert_unsolved(manifest, *, task, seed, task_config, revision):
    """Whether an official manifest shows the expert failed ``seed`` of ``task``, and the evidence.

    It is the opposite verdict of ``seed_admission`` on the same manifest:
    the seed was evaluated and not accepted, it is not a conflicting seed, and
    every rejection recorded for it is an expert failure or exception. An
    unstable scene, a reset that already satisfies the task or an
    infrastructure failure is not an expert-unsolved seed and is refused.
    """
    rejections = [item for item in manifest.get("rejected") or []
                  if isinstance(item, dict) and item.get("requested_seed") == seed]
    digest_valid = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) == manifest.get("manifest_sha256")
    passed = (
        manifest.get("schema") == SEED_MANIFEST_SCHEMA
        and manifest.get("task") == task
        and manifest.get("task_config") == task_config
        and manifest.get("robotwin_revision") == revision
        and digest_valid
        and seed in (manifest.get("evaluated_seeds") or [])
        and seed not in (manifest.get("accepted_seeds") or [])
        and seed not in (manifest.get("conflicting_seeds") or [])
        and bool(rejections)
        and all(item.get("classification") in EXPERT_UNSOLVED_CLASSIFICATIONS for item in rejections)
    )
    return passed, {"task": manifest.get("task"), "task_config": manifest.get("task_config"),
                    "robotwin_revision": manifest.get("robotwin_revision"), "manifest_sha256_valid": digest_valid,
                    "rejections": [item.get("classification") for item in rejections]}


def seed_admission(manifest, *, task, seed, task_config, revision):
    """Whether a seed manifest admits ``seed`` of ``task``, and the evidence for the verdict.

    Launch preflight (``seed_qualification``) and the maintained RunSpec
    builder (``spec.benchmark``) apply this one rule. An official-scale
    manifest (``SEED_MANIFEST_SCHEMA``, written by ``qualify``) binds a scene
    configuration and revision and carries per-seed evidence; the seed is
    admitted on its own reset/oracle record. The post-success hold is evidence,
    never admission: upstream collects such seeds and the benchmark's verdict
    latches on success. Any other manifest is the older per-task shape, which
    names no scene configuration or revision: it admits a listed seed when its
    reset/oracle flags hold and the terminal hold lasted 250 steps.
    """
    accepted = manifest.get("accepted_seeds", [])
    if manifest.get("schema") == SEED_MANIFEST_SCHEMA:
        record = (manifest.get("seeds") or {}).get(str(seed)) or {}
        digest_valid = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) == manifest.get("manifest_sha256")
        passed = (
            manifest.get("task") == task
            and manifest.get("task_config") == task_config
            and manifest.get("robotwin_revision") == revision
            and digest_valid
            and seed in accepted
            and record.get("reset_predicate_false") is True
            and record.get("oracle_success") is True
        )
        return passed, {
            "schema": manifest.get("schema"), "task": manifest.get("task"),
            "task_config": manifest.get("task_config"), "expected_task_config": task_config,
            "robotwin_revision": manifest.get("robotwin_revision"), "expected_revision": revision,
            "manifest_sha256_valid": digest_valid, "accepted": seed in accepted, "seed": record,
        }
    passed = (
        manifest.get("task") == task
        and seed in accepted
        and manifest.get("reset_predicate_false") is True
        and manifest.get("oracle_success") is True
        and int(manifest.get("terminal_stability_steps", 0)) >= 250
    )
    return passed, {
        "task": manifest.get("task"), "accepted_seeds": accepted,
        "reset_predicate_false": manifest.get("reset_predicate_false"),
        "oracle_success": manifest.get("oracle_success"),
        "terminal_stability_steps": manifest.get("terminal_stability_steps"),
    }


__all__ = ["EXPERT_UNSOLVED_CLASSIFICATIONS", "SEED_MANIFEST_SCHEMA", "digest", "expert_unsolved", "seed_admission"]
