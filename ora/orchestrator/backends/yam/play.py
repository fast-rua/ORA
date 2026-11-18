"""The YAM station's free play (``ora play --env yam``): the cell as it stands, no task.

A real cell has no scene draw: the scene is whatever is on the table. The
station offers one play scene, ``operator_task`` at seed 0, whose record names
the cell rather than a qualification manifest (``bound_record`` is false). A
session's ``/reset`` loads the next of the plan's scenes, all of them this one,
through ``runtime.restart_round``: a new round from the arms' rest pose.

The YAM station is no baseline instance (``baseline_instance`` is false), so
``build_spec`` composes the whole explicit policy: the default baseline's
policy semantics, the ``yam_perception_v1`` ToolSurface (SAM3 segmentation and
grounding, Contact-GraspNet proposals) without its lifecycle tool, its resident
services bound from the ``--service-report`` readiness report, and the typed
prompt layers of ``YAM_PROMPT_BUNDLE`` with the free-play episode task. Without
a report it composes ``yam_direct_v1``, which hosts no services
(``NO_PERCEPTION_SURFACE``); ``ora play`` asks for that explicitly
(``--no-perception``). ``with_rig`` binds the operator's rig choices (``ora play
--rig-mode``, ``--operator-present``, ``--cell``, ``--auto-approve``) into that
runtime. No
evaluator exists: the verdict is null and the operator judges.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

import yaml

from ora.orchestrator.play import binding
from ora.orchestrator.play import scenes as play_scenes
from ora.orchestrator.play.scenes import PlayScene, PlaySceneError
from ora.orchestrator.spec.values import RunSpecError

from . import switches

TASK = "operator_task"
SURFACE = switches.PERCEPTION_SURFACE
YAM_PROMPT_BUNDLE = "ora/policy/prompt_bundles/current_v4/yam-station-perception-native_api-v4.yaml"
#: ``ora play --scripted-moves`` (off by default): a pick and place the station can already plan runs as one codemode
#: script, checking each receipt, instead of a turn per move (15-22 model calls one move at a time, 2026-11-14).
YAM_PLAY_SCRIPTED_MOVES = switches.SCRIPTED_MOVES.layer
#: ``ora play --reach-query`` (off by default): an unreachable target's refusal names the nearest position that solves
#: (``runtime.reach_query``), and this layer tells the model to read it rather than search a grid of dry runs.
YAM_PLAY_REACH_QUERY = switches.REACH_QUERY.layer
#: ``ora play --env yam --code-mode``: the code-mode contract (extended library, prompt v3), as robodojo_toolkit_v7.
YAM_CODE_MODE = {"library": "extended", "prompt": "v3", "source": True}
#: The station without the perception services (``ora play --env yam --no-perception``).
#: The station's free play runs as the real-time cell it is; ``ora play --latency-profile`` overrides it.
LATENCY_PROFILE = "realtime"
NO_PERCEPTION_SURFACE = "yam_direct_v1"
NO_PERCEPTION_BUNDLE = "ora/policy/prompt_bundles/current_v4/yam-station-native_api-v4.yaml"
ENVIRONMENT_PROMPT_PROFILE = "yam_public_direct_v1"
WORKSPACE_PROFILE = "yam_station_tabletop_v1"
#: The runtime fields the operator chooses at launch (``with_rig``).
RIG_FIELDS = ("rig_mode", "operator_present", "cell_config", "required_host", "auto_approve_motions")


class YamPlay:
    """``registry.PlayBackend`` of the YAM station: its one cell, as it stands."""

    environment = "yam"
    display_name = "YAM station"
    scene_sets = False
    bound_record = False

    def list_scenes(self, *, task: str | None, task_config: str | None, scene_set: str | None, root: Path,
                    **_sources: Any) -> list[PlayScene]:
        scene = PlayScene("yam", TASK, 0, {"kind": "yam_operator_cell", "scene": "whatever_is_on_the_table"})
        return [scene] if task in (None, TASK) else []

    def draw(self, rng: random.Random, listing: dict[str, Any]) -> PlayScene:
        scenes = play_scenes.list_play_scenes(self.environment, **listing)
        if not scenes:
            raise PlaySceneError(f"the YAM station has one scene, {TASK}, not {listing.get('task')!r}")
        return play_scenes.draw_from(scenes, rng)

    def label(self, scene: PlayScene) -> str:
        return "YAM station"

    def build_spec(self, spec: dict[str, Any], chosen: PlayScene, *, host: str | None,
                   resource_profile: str | None, service_report: Path | None, root: Path) -> None:
        surface, bundle = (SURFACE, YAM_PROMPT_BUNDLE) if service_report is not None else (
            NO_PERCEPTION_SURFACE, NO_PERCEPTION_BUNDLE)
        spec["runtime"] = yam_runtime(root=root)
        # A real cell does not wait for the harness, and the simulated cell rehearses it: no execution
        # capture and no preview pictures on the control path (docs/reference/latency-profile.md).
        spec["latency_profile"] = LATENCY_PROFILE
        spec["policy"] = yam_policy(root=root, surface=surface)
        spec["policy_context"] = yam_policy_context(root=root, bundle=bundle)
        if service_report is not None:
            try:
                spec["prerequisite"] = yam_service_prerequisite(service_report, root=root)
            except (OSError, ValueError) as exc:
                raise PlaySceneError(f"--service-report does not bind the YAM station's services: {exc}") from None

    def refuse_benchmark(self, runtime: Mapping[str, Any]) -> None:
        binding.refuse_scene_set(runtime)

    def bind_runtime(self, runtime: Mapping[str, Any], scene: Mapping[str, Any],
                     scenes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        binding.require_scene(scene, {"environment": self.environment, "task": TASK}, scene_set_allowed=False)
        if runtime.get("task") != TASK:
            raise RunSpecError(f"a YAM free-play session runs the station's {TASK}")
        return dict(runtime)

    def require_scenes(self, runtime: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]]) -> None:
        # restart_round starts a new round of the same cell; there is nothing else to load.
        if any(scene.get("environment") != self.environment or scene.get("task") != TASK for scene in scenes):
            raise RunSpecError(f"a YAM free-play session's scenes are rounds of the station's {TASK}")


def _bundle(root: Path, bundle: str = YAM_PROMPT_BUNDLE) -> dict[str, Any]:
    return yaml.safe_load((root / bundle).read_text(encoding="utf-8"))


def yam_runtime(*, root: Path) -> dict[str, Any]:
    """The simulated cell's runtime; ``with_rig`` binds another rig mode."""

    from ora.agent.composition.embodiment.environment_context import resolve_environment_prompt_profile
    from ora.environment.benchmarks.yam import public_context

    public = public_context()
    binding_record = resolve_environment_prompt_profile(public, project_root=root,
                                                        profile_id=ENVIRONMENT_PROMPT_PROFILE)
    return {
        "kind": "yam", "transport": "local_rpc", "python_env": "YAM_PYTHON", "task": TASK, "seed": 0,
        "rig_mode": "sim", "workspace_profile": WORKSPACE_PROFILE,
        "public_environment_context": public,
        "environment_prompt_profile": {key: binding_record[key] for key in (
            "schema", "profile_id", "profile_path", "profile_sha256", "metadata_sha256")},
    }


def yam_policy(*, root: Path, surface: str = SURFACE) -> dict[str, Any]:
    """The explicit policy semantics: the default baseline's, on a YAM surface without its lifecycle tools."""

    from ora.agent.composition.baseline import baseline_policy, load_native_baseline
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from ora.orchestrator.spec.defaults import play_tool_allowlist

    catalog = load_native_baseline(root)
    resolved = resolve_tool_surface(surface)
    names = play_tool_allowlist([item["name"] for item in resolved.definitions], resolved)
    return {**baseline_policy(root, catalog), "tool_surface": {"profile": surface, "component_overrides": {}},
            "tool_allowlist": names, "supports_images": True}


def yam_service_prerequisite(report: Path, *, root: Path) -> dict[str, Any]:
    """Bind a readiness report (``ora services qualify``) to ``yam_perception_v1``.

    The station is no baseline instance, so the binding names its surface directly and takes
    the default baseline's required service capabilities, as ``spec.inputs.service_prerequisite``
    does for a simulator; ``operator.binding`` refuses a stale, failed or foreign report.
    """

    from ora.agent.composition.baseline import load_native_baseline
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from ora.agent.model_services import operator as services

    catalog = load_native_baseline(root)
    binding = services.binding(Path(report), SURFACE, catalog["required_service_capabilities"])
    missing = set(binding["capability_scope"]) - set(resolve_tool_surface(SURFACE).capabilities)
    if missing:
        raise ValueError(f"the YAM perception surface lacks qualified service capabilities: {sorted(missing)}")
    return binding


def yam_policy_context(*, root: Path, bundle: str = YAM_PROMPT_BUNDLE) -> dict[str, Any]:
    """The bundle's typed prompt layers, with the free-play episode task in place of the task template."""

    from ora.orchestrator.spec.defaults import PLAY_EPISODE_TASK

    layers = _bundle(root, bundle)["layers"]
    return {
        "schema": "ora.policy-context.v3",
        "harness_system": list(layers["planner"]["sources"]),
        "project_instructions": [*layers["core"]["sources"], layers["environment"]["source"],
                                 layers["embodiment"]["source"]],
        "episode_task": [PLAY_EPISODE_TASK],
        "tool_manual": list(layers["tool_manual"]["sources"]),
    }


def with_rig(spec: dict[str, Any], *, rig_mode: str, operator_present: bool, cell: Path | None,
             host: str | None, auto_approve: bool = False) -> dict[str, Any]:
    """The play RunSpec with the operator's rig choices bound into its runtime.

    ``read`` and ``on`` run on the real cell, so they name its registered
    execution host (``host``, else ``ORA_EXECUTION_HOST``); ``on`` moves the
    real arms and needs ``operator_present``. The cell config is the real
    cell's (``ora.cell-config.v1``), read by the worker. ``auto_approve``
    (``on`` only) sets ``auto_approve_motions``: the worker switches the cell's
    approval broker to answer its motion questions itself, while enabling and
    releasing the arms are still asked (``runtime.auto_approve`` would skip
    those too, so it is never set here).
    """

    import os

    from ora.environment.benchmarks.yam import RIG_MODES

    if rig_mode not in RIG_MODES:
        raise ValueError(f"--rig-mode must be one of {', '.join(RIG_MODES)}")
    if rig_mode == "on" and not operator_present:
        raise ValueError("--rig-mode on moves the real arms: pass --operator-present only with a person "
                         "beside the cell and a hand on the stop")
    if operator_present and rig_mode != "on":
        raise ValueError("--operator-present applies to --rig-mode on only")
    if auto_approve and rig_mode != "on":
        raise ValueError("--auto-approve applies to --rig-mode on only")
    runtime = {key: value for key, value in spec["runtime"].items() if key not in RIG_FIELDS}
    runtime["rig_mode"] = rig_mode
    if rig_mode == "on":
        runtime["operator_present"] = True
        if auto_approve:
            runtime["auto_approve_motions"] = True
    if cell is not None:
        path = cell.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"--cell names no file: {path}")
        runtime["cell_config"] = str(path)
    if rig_mode != "sim":
        required_host = host or os.environ.get("ORA_EXECUTION_HOST")
        if required_host:
            runtime["required_host"] = required_host
    runtime["task_instruction"] = spec["task_instruction"]
    return {**spec, "runtime": runtime}


def with_station_options(spec: dict[str, Any], *, chosen: Collection[str] = (),
                         code_mode: bool = False) -> dict[str, Any]:
    """The play RunSpec with the station's switches ``chosen`` (their names, ``switches.SWITCHES``) turned on, each
    off by default: a switch sets its ``runtime`` key to its one value (``benchmarks.yam.RUNTIME_SWITCHES``) and adds
    its episode-task layer, if it has them; ``code_mode`` names the station's code-mode contract."""

    rows = [switch for switch in switches.SWITCHES if switch.name in chosen]
    unknown = set(chosen) - {switch.name for switch in rows}
    if unknown:
        raise ValueError(f"unknown YAM station switches: {sorted(unknown)}")
    if code_mode:
        # ORA's code mode (``ora play --code-mode``): the station is no baseline instance, so the contract is named
        # here, the one the RoboDojo toolkit baseline defaults to. A stop at the cell ends the whole script, not only
        # its current motion: the robot socket refuses the script's later world-changing calls in that turn.
        spec = {**spec, "policy": {**spec["policy"], "code_mode": dict(YAM_CODE_MODE)}}
    if not rows:
        return spec
    surface = (spec.get("policy") or {}).get("tool_surface", {}).get("profile")
    for switch in rows:
        if switch.surface is not None and surface != switch.surface:
            raise ValueError(f"{switch.flag} perceives with the {switch.surface} tools: drop --no-perception")
    from ora.environment.benchmarks.yam import RUNTIME_SWITCHES

    context = dict(spec["policy_context"])
    layers = list(context["episode_task"])
    runtime = dict(spec["runtime"])
    for switch in rows:
        if switch.layer is not None:
            layers.append(switch.layer)
        if switch.runtime_key is not None:
            value = RUNTIME_SWITCHES[switch.runtime_key]
            runtime[switch.runtime_key] = value if value is True else dict(value)
    context["episode_task"] = layers
    return {**spec, "policy_context": context, "runtime": runtime}


PLAY = YamPlay()

__all__ = ["LATENCY_PROFILE", "NO_PERCEPTION_BUNDLE", "NO_PERCEPTION_SURFACE", "PLAY", "RIG_FIELDS", "SURFACE", "TASK",
           "YAM_CODE_MODE", "YAM_PLAY_REACH_QUERY", "YAM_PLAY_SCRIPTED_MOVES", "YAM_PROMPT_BUNDLE", "YamPlay", "with_rig",
           "with_station_options", "yam_policy", "yam_service_prerequisite", "yam_policy_context", "yam_runtime"]
