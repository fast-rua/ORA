"""The Synthetic backend: ORA's in-process world, which needs no simulator, record or qualification."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ora.orchestrator.play import scenes as play_scenes
from ora.orchestrator.backends.registry import BackendCapabilities, Check, OrchestratorBackend, PreflightContext
from ora.orchestrator.play import binding
from ora.orchestrator.play.scenes import PlayScene, PlaySceneError
from ora.orchestrator.spec.values import RunSpecError, _positive_int

if TYPE_CHECKING:
    from ora.agent.composition import ResolvedToolSurface


def resolve_runtime(
    runtime: dict[str, Any], policy: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface, *, task: str, seed: int,
    task_instruction: str,
) -> None:
    if runtime["transport"] != "in_process_direct":
        raise RunSpecError("synthetic runtime.transport must be in_process_direct")
    runtime.setdefault("terminal_actions", 1)
    runtime["terminal_actions"] = _positive_int(
        runtime["terminal_actions"], "runtime.terminal_actions"
    )


def preflight_runtime(plan: Mapping[str, Any], context: PreflightContext) -> list[Check]:
    return [context.check("runtime", True, "synthetic policy-visible boundary")]


#: The Synthetic world ends a play scene after this many accepted actions.
PLAY_ACTIONS = 400
PLAY_WORKSPACE_BOUNDS_M = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}


class SyntheticPlay:
    """``registry.PlayBackend`` of Synthetic: its one built-in world."""

    environment = "synthetic"
    display_name = "Synthetic"
    scene_sets = False
    bound_record = False

    def list_scenes(self, *, task: str | None, task_config: str | None, scene_set: str | None, root: Path,
                    **_sources: Any) -> list[PlayScene]:
        scenes = [PlayScene("synthetic", "synthetic-world", 0, {"kind": "synthetic_world"})]
        return [scene for scene in scenes if task in (None, scene.task)]

    def draw(self, rng: random.Random, listing: dict[str, Any]) -> PlayScene:
        scenes = play_scenes.list_play_scenes(self.environment, **listing)
        if not scenes:
            raise PlaySceneError(f"Synthetic has one scene, synthetic-world, not {listing.get('task')!r}")
        return play_scenes.draw_from(scenes, rng)

    def label(self, scene: PlayScene) -> str:
        return "Synthetic world"

    def build_spec(self, spec: dict[str, Any], chosen: PlayScene, *, host: str | None,
                   resource_profile: str | None, service_report: Path | None, root: Path) -> None:
        from ora.orchestrator.spec.quickstart import synthetic_spec

        spec["budgets"]["max_sim_steps"] = max(spec["budgets"]["max_sim_steps"], PLAY_ACTIONS)
        quickstart = synthetic_spec({})
        spec["runtime"] = {"kind": "synthetic", "terminal_actions": PLAY_ACTIONS,
                           "workspace_bounds_m": json.loads(json.dumps(PLAY_WORKSPACE_BOUNDS_M)),
                           # The quickstart's drawn world: the scene's cameras in the world view.
                           "execution_capture": dict(quickstart["runtime"]["execution_capture"])}
        spec["policy_context"] = quickstart["policy_context"]
        spec["policy"] = {**dict(spec.get("policy") or {}), "tool_surface": quickstart["policy"]["tool_surface"]}

    def refuse_benchmark(self, runtime: Mapping[str, Any]) -> None:
        binding.refuse_scene_set(runtime)

    def bind_runtime(self, runtime: Mapping[str, Any], scene: Mapping[str, Any],
                     scenes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        binding.require_scene(scene, {"environment": self.environment}, scene_set_allowed=False)
        return dict(runtime)

    def require_scenes(self, runtime: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]]) -> None:
        return None


BACKEND = OrchestratorBackend(
    kind="synthetic",
    resolve_runtime=resolve_runtime,
    preflight_runtime=preflight_runtime,
    # Synthetic draws its world only when the RunSpec asks (spec.quickstart.synthetic_spec does).
    capabilities=BackendCapabilities(captures_execution=False, persistent_session=True),
    play=SyntheticPlay(),
)

__all__ = ["BACKEND", "PLAY_ACTIONS", "PLAY_WORKSPACE_BOUNDS_M", "SyntheticPlay", "preflight_runtime",
           "resolve_runtime"]
