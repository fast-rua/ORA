"""``ora play``: free play in one random qualified scene, the operator's instructions in turn.

    ora play --env synthetic [--provider PROFILE|env]
    ora play --env libero|robotwin [--provider PROFILE] [--host HOST] [--service-report REPORT] --confirm-launch
    ora play --env robotwin --scene-set tabletop-v1 [--scene NAME] ...   one of ORA's own audited tabletop scenes
    ora play --env robodojo [--scene-set isaac-tabletop-v1] [--scene NAME] --service-report REPORT --confirm-launch
                                       one of ORA's own audited Isaac Sim tabletops (RoboDojo has no other scenes)
    ora play --env yam --rig-mode sim|read|on --service-report REPORT|--no-perception [--operator-present]
            [--cell CELL.json] [--provider PROFILE]
                                       the YAM station as it stands: its simulated model, the real cell read
                                       only (every motion answered with its plan), or the real arms, each
                                       motion approved by the operator beside the cell; SAM3 and
                                       Contact-GraspNet from REPORT, or none with --no-perception
    ora play --env yam --rig-mode on --operator-present --auto-approve ...
                                       the real arms, motions run without asking (the cell's auto-approve
                                       switch; enabling the arms and releasing them are still asked)
    ora play ... --prepare-only     write the free-play RunSpec under outputs/ora-play/specs/ and stop
    ora play --env ENV --spec PLAY.json --confirm-launch     launch a prepared free-play RunSpec
    ora play audit --env robotwin|robodojo ...   qualify ORA's own free-play scenes on this host (the
                                       backend's play-scene audit: backends/robotwin/play_audit.py,
                                       robodojo_play_audit.py)

The scene comes from ``play.scenes``: one scene this host has qualified,
drawn at random (``draw_play_scene``; ``--scene-set`` draws among the
audited scenes of one of ORA's RoboTwin scene sets instead of the official
tasks, ``--scene`` names one task or scene; RoboDojo offers only ORA's own
scene sets, ``isaac-tabletop-v1`` by default) and loaded without its task
(``build_play_spec``, ``play.scene``). The scenes ``/reset`` may load are the
same task's and scene configuration's scenes that ``list_play_scenes`` lists
and the same record admits, put in a random order here and bound in the plan
(``play.scenes``); ``plan.play.draw`` records the draw. Synthetic offers its
one world, the YAM station its one cell as it stands (``--rig-mode``,
``--operator-present``, ``--cell`` and ``--auto-approve`` bind its runtime, ``--service-report`` or
``--no-perception`` its surface, backends/yam/play.py). The RunSpec is written by ``write_play_spec``, which imports a
RoboTwin seed manifest by content.

The run is Pi's own agent (ora-pi/src/play/): Pi's coding tools, commands,
skills, context files and model registry, with ORA's robot tools and the
Supervisor's authority over the world. The policy never receives the scene's
task: ``task_instruction`` is the play framing, the task contract is withheld
(episode-task-play-v2.md), check_task and the lifecycle tools are not
advertised. Runs land under ``outputs/ora-play``, unscored (official_success
null) and never learnable; each instruction is its own event (episode/play.py,
docs/usage/play.md).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.orchestrator.backends.yam import switches as station_switches

EXPERIMENT = "ora-play"
#: What the policy reads as its task: the session's framing (the operator's instructions follow).
PLAY_TASK = ("Free play in this scene: carry out the operator's instructions, one at a time; "
             "the operator may also talk to you while you work.")
ENVIRONMENTS = ("synthetic", "libero", "robotwin", "robodojo", "yam")
#: Each instruction, and the whole session (the plan's budgets).
INSTRUCTION_BUDGETS = {"max_robot_decisions": 24, "max_model_calls": 80, "wall_time_s": 1200}
SESSION_BUDGETS = {"max_robot_decisions": 400, "max_model_calls": 1600, "wall_time_s": 14400}


AUDIT = """Qualify ORA's own free-play scenes on this host.

    ora play audit --env robotwin --scene-set tabletop-v1 [--scene NAME ...] ...
    ora play audit --env robodojo [--scene-set isaac-tabletop-v1] ...

The backend's play-scene audit (``play_audit`` in ``orchestrator.backends.registry``)
gets the rest of the line (``ora play audit --env robotwin --help``).
"""


def register(sub: Any) -> None:
    from .passthrough import add_tool
    from .world import add_world_options

    parser = sub.add_parser(
        "play", help="free play: a random qualified scene in Pi's terminal UI, no task; type instructions in turn",
        description="Free play: a random qualified scene, no task; type instructions in turn.",
        epilog="Keys in the session: Enter sends an instruction, Esc stops the running one, /reset loads "
               "another scene, /quit ends and seals the session, Ctrl+D detaches.")
    parser.add_argument("--env", choices=ENVIRONMENTS, dest="environment", help="the environment (required)")
    parser.add_argument("--spec", type=Path, help="launch this free-play RunSpec (from --prepare-only) as it is")
    parser.add_argument("--provider", metavar="PROFILE",
                        help="provider profile (`ora providers`), or env for ORA_BASE_URL/ORA_MODEL/ORA_API_KEY")
    parser.add_argument("--host", help="execution host of a real environment (default: ORA_EXECUTION_HOST)")
    parser.add_argument("--resource-profile", metavar="SLOT",
                        help="the host's simulator slot to run on (default: its first), e.g. the slot a router "
                             "allocation leased")
    parser.add_argument("--service-report", type=Path,
                        help="a real environment's perception readiness report (`ora services qualify`)")
    parser.add_argument("--scene-set", metavar="SET",
                        help="draw among the audited scenes of one of ORA's scene sets: RoboTwin's tabletop-v1 "
                             "(instead of the official tasks) or RoboDojo's isaac-tabletop-v1 (its default)")
    parser.add_argument("--scene", metavar="NAME",
                        help="the task or scene to play (a LIBERO or RoboTwin task, or a scene of a scene set); "
                             "its seeds or initial states are still drawn")
    parser.add_argument("--scenes", type=int, default=8, help="scenes drawn for /reset (1-32, default 8)")
    parser.add_argument("--rig-mode", choices=("sim", "read", "on"),
                        help="YAM: the station model (sim), the real cell read only (read), or the real arms "
                             "with each motion approved by the operator (on); required with --env yam")
    parser.add_argument("--operator-present", action="store_true",
                        help="YAM --rig-mode on: a person stands beside the cell with a hand on the stop")
    parser.add_argument("--auto-approve", action="store_true",
                        help="YAM --rig-mode on: motions run without asking; the operator still approves "
                             "enabling the arms once")
    parser.add_argument("--setup", action="store_true",
                        help="YAM --rig-mode on: first finish the cell's setup (ora cell setup), its questions "
                             "answered in the play terminal, then start the arms")
    parser.add_argument("--cell", type=Path, metavar="CELL.json",
                        help="YAM: the real cell's config (ora.cell-config.v1), read by the worker")
    parser.add_argument("--no-perception", action="store_true",
                        help="YAM: run yam_direct_v1, which has no SAM3 or Contact-GraspNet tools, instead of the "
                             "default yam_perception_v1 and its --service-report")
    for switch in station_switches.SWITCHES:          # off by default (backends/yam/switches.py)
        parser.add_argument(switch.flag, action="store_true", help=switch.help)
    parser.add_argument("--code-mode", action="store_true",
                        help="the agent works in ORA's code mode (bash and the robot library in ORA's sandbox, "
                             "the environment baseline's policy.code_mode) instead of Pi's own tools and codemode")
    parser.add_argument("--latency-profile", choices=("sim_benchmark", "realtime"),
                        help="what the run keeps on its control path (docs/reference/latency-profile.md): "
                             "realtime turns execution capture and preview pictures off (default: realtime on "
                             "--env yam, sim_benchmark elsewhere)")
    parser.add_argument("--pipeline-predict", action="store_true",
                        help="while a motion runs, predict the next step and use it only when the receipt confirms "
                             "the motion (off by default; not with --code-mode)")
    parser.add_argument("--request-priority", metavar="JSON",
                        help="the priority field each model request carries, for a self-hosted server's priority "
                             "scheduling: '{\"correction\": -1, \"default\": 0}' (correction: the call that first "
                             "reads an operator message joined to the running instruction; default: none sent)")
    parser.add_argument("--draw-seed", type=int, help="seed of the random draw (default: fresh)")
    parser.add_argument("--prepare-only", action="store_true", help="write the free-play RunSpec and stop")
    parser.add_argument("--confirm-launch", action="store_true",
                        help="authorize launching a real environment after its normal preflight passes")
    parser.add_argument("--no-attach", action="store_true", help="launch and print the `ora attach` command")
    add_world_options(parser)  # the scene's cameras in a browser while attached (ora-remote forwards them)
    actions = parser.add_subparsers(dest="play_command", metavar="audit")
    add_tool(actions, "audit", "ora.app.play:audit", "audit ORA's own free-play scenes on this host", AUDIT)


def draw_scenes(pool: Sequence[Any], count: int, rng: random.Random, *, first: Any = None) -> list[Any]:
    """``count`` scenes from ``pool``, starting at ``first`` (default: a random one); each one
    different from the one before where the pool allows."""

    pool = list(pool)
    if not pool:
        raise RuntimeError("no qualified scene to draw from")
    scenes = [first if first is not None else rng.choice(pool)]
    while len(scenes) < count:
        others = [scene for scene in pool if scene != scenes[-1]]
        scenes.append(rng.choice(others or [scenes[-1]]))
    return scenes


def play_block(scenes: Sequence[Mapping[str, Any]], draw: Mapping[str, Any]) -> dict[str, Any]:
    return {"scene": dict(scenes[0]), "scenes": [dict(scene) for scene in scenes],
            "instruction_budgets": dict(INSTRUCTION_BUDGETS), "draw": dict(draw)}


def provider_policy_for(provider: str | None, environment: str = "synthetic") -> dict[str, Any]:
    """The native_api policy of a provider profile, or the quickstart environment's (``env``, Synthetic)."""

    from .interactive import ENV_PROVIDER, choose_provider
    from ora.orchestrator.spec.quickstart import environment_policy

    chosen = choose_provider(provider)
    if chosen == ENV_PROVIDER:
        if environment != "synthetic":
            # The baseline of a real environment requires a provider's live qualification marker.
            raise ValueError("a real environment runs a qualified provider profile: --provider PROFILE "
                             "(`ora providers`); --provider env is for Synthetic")
        return environment_policy()
    from ora.orchestrator.spec.inputs import provider_policy

    policy = provider_policy(chosen)
    _default_base_url(policy)
    return policy


def _default_base_url(policy: Any) -> None:
    """The profile's base-URL variable defaults to its endpoint, as `ora run` sets it.

    Also for a RunSpec prepared earlier (``--spec``): its policy names the
    variable and the endpoint, and launch preflight requires the variable.
    """
    import os

    if (isinstance(policy, Mapping) and isinstance(policy.get("base_url_env"), str)
            and isinstance(policy.get("endpoint"), str) and policy["endpoint"]):
        os.environ.setdefault(policy["base_url_env"], policy["endpoint"])


# -- The scene (ora.orchestrator.play.scenes) ---------------------------------------------

def play_spec(environment: str, *, policy: Mapping[str, Any], host: str | None, count: int,
              rng: random.Random, draw: Mapping[str, Any], service_report: Path | None = None,
              scene_set: str | None = None, scene: str | None = None, resource_profile: str | None = None,
              code_mode: bool = False) -> tuple[dict[str, Any], Any]:
    """A qualified scene from play.scenes, as this session's free-play RunSpec, and the drawn scene.

    ``scene_set`` draws among a RoboTwin scene set's audited scenes, ``scene``
    names the task or scene (a scene of a set implies its set).
    """

    from ora.orchestrator.play import scenes as play_scenes

    listing: dict[str, Any] = {}
    if scene is not None and scene_set is None and environment == "robotwin":
        from ora.environment.benchmarks.robotwin_play_scenes import find_scene

        named = find_scene(scene)
        scene_set = named.scene_set if named is not None else None
    if environment == "robodojo":
        # RoboDojo has only ORA's own scene sets: a named scene implies its set, else the default set.
        from ora.environment.benchmarks.robodojo_play_scenes import DEFAULT_SCENE_SET, find_scene as find_robodojo

        named = find_robodojo(scene) if scene is not None and scene_set is None else None
        scene_set = scene_set or (named.scene_set if named is not None else DEFAULT_SCENE_SET)
        from ora.environment.benchmarks.robodojo_play_scenes import PlaySceneSetError, scene_definitions

        try:
            scene_definitions(scene_set)
        except PlaySceneSetError as exc:
            raise ValueError(str(exc)) from None
    if scene_set is not None:
        if environment not in {"robotwin", "robodojo"}:
            raise ValueError("--scene-set names one of ORA's RoboTwin or RoboDojo scene sets: use it with "
                             "--env robotwin or --env robodojo")
        listing["scene_set"] = scene_set
    if scene is not None:
        listing["task"] = scene
    chosen = play_scenes.draw_play_scene(environment, rng, **listing)
    spec = copy.deepcopy(dict(play_scenes.build_play_spec(environment, host=host, rng=rng, scene=chosen,
                                                          service_report=service_report,
                                                          **({"resource_profile": resource_profile}
                                                             if resource_profile else {}))))
    scene = chosen.record()
    # A backend resets in place to another seed or initial state of the same task and scene
    # configuration; each candidate is admitted by the record the plan binds (the drawn scene's).
    chosen_set = getattr(chosen, "scene_set", None)
    pool = [item.record() for item in play_scenes.list_play_scenes(
                environment, task=chosen.task, task_config=chosen.task_config,
                **({"scene_set": chosen_set} if chosen_set is not None else {}))
            if (item.task, item.suite, item.task_id, item.task_config)
            == (chosen.task, chosen.suite, chosen.task_id, chosen.task_config)
            and _record(item.qualification) == _record(chosen.qualification)]
    scenes = draw_scenes(pool or [scene], count, rng, first=scene)
    spec.update({
        "experiment": EXPERIMENT, "output_workspace": f"outputs/{EXPERIMENT}", "episode_mode": "play",
        "task_instruction": PLAY_TASK,
        "policy": {**dict(spec.get("policy") or {}), **dict(policy), "agent_core": "pi"},
        "operator_control": {"mode": "file_inbox_v1"},
        "budgets": {**SESSION_BUDGETS, **{key: value for key, value in (spec.get("budgets") or {}).items()
                                           if key == "max_sim_steps"}},
    })
    spec["play"] = {**dict(spec["play"]), **play_block(scenes, {**draw, "pool_size": len(pool) or 1}),
                    **({"code_mode": True} if code_mode else {})}
    spec["config"] = f"play-{environment}-{spec['task']}-{uuid.uuid4().hex[:8]}"
    if environment == "synthetic":
        # The quickstart's layers, then the free-play episode task (a baseline instance composes it itself).
        from ora.orchestrator.spec.defaults import PLAY_EPISODE_TASK

        spec["policy_context"] = {**spec["policy_context"],
                                  "episode_prefix": [*spec["policy_context"]["episode_prefix"], PLAY_EPISODE_TASK]}
    return spec, chosen


def _record(qualification: Mapping[str, Any]) -> dict[str, Any]:
    """The record a scene's qualification names, without its per-seed layout identity (RoboDojo's)."""

    return {key: value for key, value in qualification.items() if key != "layout_sha256"}


# -- The command ---------------------------------------------------------------------------

def compose_play(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None]:
    """The free-play RunSpec this invocation launches, and the file it was written to (if any)."""

    if not 1 <= args.scenes <= 32:
        raise ValueError("--scenes must be between 1 and 32")
    _require_rig_options(args)
    if args.code_mode and args.spec is not None:
        raise ValueError("--spec launches a prepared RunSpec as it is: its play.code_mode already decides")
    if args.latency_profile is not None and args.spec is not None:
        raise ValueError("--spec launches a prepared RunSpec as it is: its latency_profile already decides")
    if args.pipeline_predict and args.spec is not None:
        raise ValueError("--spec launches a prepared RunSpec as it is: its play.pipeline_predict already decides")
    if args.request_priority is not None and args.spec is not None:
        raise ValueError("--spec launches a prepared RunSpec as it is: its play.request_priority already decides")
    if args.spec is not None:
        from ora.orchestrator.spec.run_spec import load_run_spec

        spec = load_run_spec(args.spec)
        if spec.get("episode_mode") != "play":
            raise ValueError(f"{args.spec} is not a free-play RunSpec (episode_mode play)")
        _default_base_url(spec.get("policy"))
        return dict(spec), args.spec
    draw_seed = args.draw_seed if args.draw_seed is not None else random.SystemRandom().randrange(2**31)
    rng = random.Random(draw_seed)
    draw = {"draw_seed": draw_seed, "drawn_unix_s": round(time.time(), 3)}
    policy = provider_policy_for(args.provider, args.environment)
    from ora.orchestrator.play.scenes import write_play_spec

    draw.update({key: value for key, value in (("scene_set", args.scene_set), ("scene", args.scene))
                 if value is not None})
    spec, scene = play_spec(args.environment, policy=policy, host=args.host, count=args.scenes, rng=rng,
                            draw=draw, service_report=args.service_report, scene_set=args.scene_set,
                            scene=args.scene, resource_profile=args.resource_profile, code_mode=args.code_mode)
    if args.environment == "yam":
        from ora.orchestrator.backends.yam.play import with_rig

        spec = with_rig(spec, rig_mode=args.rig_mode, operator_present=args.operator_present, cell=args.cell,
                        host=args.host, auto_approve=args.auto_approve)
        if args.setup:
            spec["play"]["cell_setup"] = True
        from ora.orchestrator.backends.yam.play import with_station_options

        spec = with_station_options(spec, code_mode=args.code_mode, chosen=[
            switch.name for switch in station_switches.SWITCHES if getattr(args, switch.name)])
    if args.latency_profile is not None:
        spec["latency_profile"] = args.latency_profile
    if args.request_priority is not None:
        spec["play"]["request_priority"] = _request_priority(args.request_priority)
    if args.pipeline_predict:
        spec["play"]["pipeline_predict"] = True
    # Written for a launch too: a RoboTwin plan reads the seed manifest it imports.
    return spec, write_play_spec(spec, scene)


def _request_priority(text: str) -> dict[str, int]:
    """``--request-priority`` as JSON text, validated as the plan validates ``play.request_priority``."""

    from ora.orchestrator.spec.play import request_priority
    from ora.orchestrator.spec.values import RunSpecError

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--request-priority is not valid JSON: {exc}") from None
    try:
        priority = request_priority(value)
    except RunSpecError as exc:
        raise ValueError(f"--request-priority: {exc}") from None
    if priority is None:
        raise ValueError("--request-priority names correction and default")
    return priority


#: The options that bind the YAM station's rig, before its switches in ``ora play``'s refusal.
RIG_OPTIONS = ("--rig-mode", "--operator-present", "--cell", "--auto-approve", "--setup", "--no-perception")


def _require_rig_options(args: argparse.Namespace) -> None:
    """The rig options (``RIG_OPTIONS``) and the station's switches (``backends/yam/switches.py``) belong to the YAM
    station, which needs a rig mode and either a service report or ``--no-perception``."""

    chosen = (args.rig_mode or args.operator_present or args.cell or args.auto_approve or args.setup
              or args.no_perception or any(getattr(args, switch.name) for switch in station_switches.SWITCHES))
    if args.environment == "yam":
        if args.spec is None and args.rig_mode is None:
            raise ValueError("ora play --env yam needs --rig-mode sim|read|on")
        if args.spec is None and args.no_perception and args.service_report is not None:
            raise ValueError("--no-perception runs yam_direct_v1, which hosts no services: drop --service-report")
        if args.spec is None and not args.no_perception and args.service_report is None:
            raise ValueError("ora play --env yam runs yam_perception_v1 (SAM3 and Contact-GraspNet): pass "
                             "--service-report REPORT (ora services qualify), or --no-perception for "
                             "yam_direct_v1 without them")
        if args.setup and args.rig_mode != "on":
            raise ValueError("--setup moves the arms: it needs --rig-mode on --operator-present")
        if args.spec is not None and chosen:
            raise ValueError("--spec launches a prepared RunSpec as it is: its runtime already binds the rig")
    elif chosen:
        raise ValueError(station_switches.flags_sentence(RIG_OPTIONS) + " apply to --env yam only")


def _needs_launch_confirmation(environment: str, spec: Mapping[str, Any]) -> bool:
    """A real environment launches only with --confirm-launch; Synthetic and the simulated YAM cell do not."""

    if environment == "synthetic":
        return False
    return not (environment == "yam" and (spec.get("runtime") or {}).get("rig_mode") == "sim")


def scene_description(scene: Mapping[str, Any]) -> str | None:
    """What the operator is told a scene of ORA's own scene sets holds; never sent to the policy."""

    if scene.get("environment") == "robodojo" and scene.get("scene_set"):
        from ora.environment.benchmarks.robodojo_play_scenes import PlaySceneSetError, scene_definition

        try:
            return scene_definition(str(scene["scene_set"]), str(scene["task"])).description
        except PlaySceneSetError:
            return None
    if scene.get("environment") == "robotwin" and scene.get("scene_set"):
        from ora.environment.benchmarks.robotwin_play_scenes import PlaySceneSetError as RoboTwinSetError
        from ora.environment.benchmarks.robotwin_play_scenes import scene_definition as robotwin_scene

        try:
            return robotwin_scene(str(scene["scene_set"]), str(scene["task"])).description
        except RoboTwinSetError:
            return None
    return None


def audit(argv: list[str], environment: str | None = None) -> int:
    """``ora play audit --env ENV ...``; ``ora play --env ENV audit ...`` names the environment as well."""

    from .passthrough import play_audit

    return play_audit(argv, environment=environment, doc=AUDIT)


def run(args: argparse.Namespace) -> int:
    if getattr(args, "play_command", None) == "audit":
        return audit(args.tool_args, args.environment)
    if args.environment is None:
        raise ValueError("ora play needs --env " + "|".join(ENVIRONMENTS))
    try:
        from .interactive import launch, node_binary

        node = None if args.prepare_only else node_binary()
        spec, path = compose_play(args)
        scenes = ", ".join(str(scene["seed"]) for scene in spec["play"]["scenes"])
        print(f"ora play: {args.environment} {spec['task']}, scenes {scenes}"
              + (f"; RunSpec {path}" if path else ""), flush=True)
        description = scene_description(spec["play"]["scene"])
        if description:
            # The operator's view of the scene; the plan and the policy never carry it.
            print(f"ora play: {spec['play']['scene'].get('label', spec['task'])}: {description}", flush=True)
        if args.prepare_only:
            print(path)
            return 0
        if _needs_launch_confirmation(args.environment, spec) and not args.confirm_launch:
            raise ValueError("a real environment launches only with --confirm-launch")
        from .world import terminal_observer
        return launch(spec, node=node, no_attach=args.no_attach, world=terminal_observer(args))
    except (OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
        print(f"ora play: {exc}", file=sys.stderr)
        return 2


__all__ = ["AUDIT", "EXPERIMENT", "PLAY_TASK", "audit", "compose_play", "draw_scenes", "play_spec", "register",
           "run", "scene_description"]
