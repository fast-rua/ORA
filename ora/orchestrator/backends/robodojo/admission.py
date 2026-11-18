"""How launch preflight admits a RoboDojo play plan's play-scene audit (``registry.Admitter``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ora.orchestrator import robodojo_play_audit, robodojo_play_kernel
from ora.orchestrator.backends.registry import Verdict


def admit_play_audit(manifest: Mapping[str, Any], plan: Mapping[str, Any], *, seed: int, root: Path,
                     memo: dict[str, Any]) -> Verdict:
    """One of ORA's RoboDojo play scenes: its audit, for the play kernel and scene sources the plan binds."""

    runtime = plan.get("runtime", {})
    if "sha256" not in memo:
        # The play kernel at the plan's own commit: an audit of another kernel admits nothing.
        try:
            memo["sha256"] = robodojo_play_kernel.play_kernel_sha256(plan["source"]["commit"], project_root=root)
        except (OSError, ValueError, RuntimeError) as exc:
            memo["sha256"], memo["error"] = None, str(exc)
    passed, detail = robodojo_play_audit.play_scene_admission(
        manifest, scene_set=runtime.get("play_scene_set"), scene=plan["task"], seed=seed,
        play_kernel_sha256=memo["sha256"],
        layout_sha256=(runtime.get("play_layouts") or {}).get(str(seed)), root=root)
    if runtime.get("play_kernel_sha256") != memo["sha256"]:
        passed, detail = False, {**detail, "plan_play_kernel_sha256": runtime.get("play_kernel_sha256"),
                                 **({"play_kernel_error": memo["error"]} if "error" in memo else {})}
    # The scene's own source binding at the plan's commit (the clean tree source_identity checks).
    if runtime.get("play_scene_source_sha256") != detail.get("current_scene_source_sha256"):
        passed, detail = False, {**detail,
                                 "plan_play_scene_source_sha256": runtime.get("play_scene_source_sha256")}
    return passed, detail


__all__ = ["admit_play_audit"]
