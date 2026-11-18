"""How launch preflight admits a RoboTwin plan's qualification manifest (``registry.Admitter``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ora.orchestrator.backends.registry import Verdict
from ora.orchestrator.backends.robotwin import play_audit, seed_manifest


def admit_seed_manifest(manifest: Mapping[str, Any], plan: Mapping[str, Any], *, seed: int, root: Path,
                        memo: dict[str, Any]) -> Verdict:
    """An official seed manifest, or an older per-task one (``seed_manifest.seed_admission``)."""

    runtime = plan.get("runtime", {})
    return seed_manifest.seed_admission(manifest, task=plan["task"], seed=seed, task_config=runtime.get("task_config"),
                                        revision=runtime.get("required_revision"))


def admit_play_audit(manifest: Mapping[str, Any], plan: Mapping[str, Any], *, seed: int, root: Path,
                     memo: dict[str, Any]) -> Verdict:
    """A scene of ORA's own scene sets (free play only): its play-scene audit."""

    runtime = plan.get("runtime", {})
    return play_audit.play_scene_admission(
        manifest, scene_set=runtime.get("play_scene_set"), scene=plan["task"], seed=seed,
        task_config=runtime.get("task_config"), revision=runtime.get("required_revision"), root=root)


__all__ = ["admit_play_audit", "admit_seed_manifest"]
