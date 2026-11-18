"""Fresh RoboDojo RunSpecs from the default baseline plus task, provider and resource inputs.

New RoboDojo experiments start here, the way new RoboTwin and LIBERO-Pro
experiments start from the default resolver: the RunSpec this module writes
names no ToolSurface, prompt bundle, Context or semantic manifest, so
``defaults.expand_native_defaults`` composes RoboDojo's default baseline
(``robodojo_toolkit_v7``, ``baseline.ENVIRONMENT_BASELINES``: the
``robodojo_placement_toolkit_v6`` surface, the ``toolkit_v7`` bundle and the
``context_keep_written`` scheme)
and its reviewed task contract. What a RunSpec
owns comes from here: the task's reviewed runtime
(``backends.robodojo.runtime_config.runtime_configuration``), its instruction from the
default task-contract registry, the GPU and execution resource, an optional
K-round persistent session with its per-round layout schedule, the Provider
profile, budgets, the infrastructure-qualification binding and the
resident-service binding. A one-round episode may instead run on another
official layout of evaluation seed 0 (``layout_id``): only where
the reviewed layout-portability table extends the task's contract to that
layout, and only with the scene-smoke lane gate of that layout, which also
supplies the instruction of a task whose instruction the layout generates.

A cell may instead select its policy: a registered baseline, a tool surface
(which brings its own baseline when it has one, ``TOOLKIT_BASELINES``) and a
named context scheme replacing the baseline's context half
(``ora.agent.composition.context_scheme``). A file-backed surface's required
toolkit grants are filled in from its ``surface.json``. The written RunSpec
names exactly these choices; resolution checks them like any other RunSpec.

It replaces an earlier RoboDojo builder whose composition came from an
experiment archive.
Composing a RunSpec is not host qualification, service
startup or authority to launch: ``scripts/ora plan show|preflight`` and ``launch``
still resolve and check the written spec.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.composition.task_contract.robodojo_layouts import (
    INSTRUCTION_LAYOUT_READBACK, portable_layout,
)
from ora.agent.composition.task_contract.robodojo import REGISTRY_PATH, load_task_contract
from ora.contracts.official_limit import TOOL_BUDGET_LIMIT_MULTIPLE
from ora.environment.benchmarks.robodojo import NATIVE_ACTION_LIMITS, physics_step_ceiling
from ora.environment.benchmarks.robodojo_layouts import REPEAT_ROUND0_LAYOUT, layout_cell_tag
from ora.orchestrator.paths import PROJECT_ROOT, WORKSPACE_ROOT, experiment_workspace
from ora.orchestrator.backends.robodojo.runtime_config import (
    robodojo_layout_schedule, runtime_configuration, with_layout_schedule,
)
# The Provider, service and admission inputs every maintained builder shares;
# their names stay importable from here.
from ora.orchestrator.spec.inputs import (  # noqa: F401
    PROVIDER_FIELDS, SERVICE_VARIABLES, SPEC_SCHEMA, bind_service_environment, provider_policy,
    single_attempt_admission,
)
from ora.orchestrator.spec import inputs as run_spec_inputs

ENVIRONMENT = "robodojo"
#: The cadence of a K-round persistent session (``rounds`` given).
SESSION_CADENCE = "persistent_agent_k_rounds_v1"
#: Budgets of the earlier RoboDojo builder, the ones recorded runs used; per
#: round in a persistent session. ``max_sim_steps`` defaults to the task's
#: ceiling at the spec's ``official_limit_multiple`` (``physics_step_ceiling``).
DEFAULT_BUDGETS = {"max_robot_decisions": 80, "max_model_calls": 200, "wall_time_s": 21600}


def default_instance(root: Path = PROJECT_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """The default catalog and its RoboDojo instance; refused when it has none."""
    return run_spec_inputs.default_instance(ENVIRONMENT, root)


def contract_registry(root: Path = PROJECT_ROOT) -> dict[str, dict[str, Any]]:
    """The default task-contract registry's entries, in registry order."""
    return json.loads((root / REGISTRY_PATH).read_text(encoding="utf-8"))["tasks"]


def task_roster(root: Path = PROJECT_ROOT) -> tuple[str, ...]:
    """Every task with a reviewed contract in the default registry and a Runtime limit."""
    return tuple(task for task in contract_registry(root) if task in NATIVE_ACTION_LIMITS)


def task_contract(task: str, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """The task's record and completion layer as the default registry binds them."""
    entry = contract_registry(root).get(task)
    if entry is None:
        raise ValueError(f"{task} has no reviewed contract in {REGISTRY_PATH}")
    return load_task_contract(task, bindings={task: entry}, project_root=root)


def single_layout(task: str, layout_id: int | None, *, root: Path = PROJECT_ROOT) -> dict[str, Any] | None:
    """None for the reviewed layout 0; the audited portability row of another single-episode layout.

    Refuses an ordinal that is not a non-negative integer, and a layout the
    reviewed layout-portability table does not extend the task's contract to.
    """
    if layout_id is None or (type(layout_id) is int and layout_id == 0):
        return None
    if type(layout_id) is not int or layout_id < 0:
        raise ValueError(f"a RoboDojo layout id is a non-negative official ordinal, not {layout_id!r}")
    try:
        return portable_layout(task, layout_id, project_root=root, registry=REGISTRY_PATH)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def runtime_for(task: str, *, gpu: str, resource_profile: str | None = None,
                rounds: int = 1, layout_schedule: Sequence[int] | str | None = None,
                layout_id: int | None = None, task_instruction: str | None = None,
                eval_seed: int = 0, task_file: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """The task's reviewed runtime on physical GPU ``gpu`` and its execution slot.

    ``task_file``: the RunSpec names its own task file in place of the reviewed
    contract (``policy.task_file``), so any official layout of any evaluation
    seed of the pinned Assets may run, and the instruction is read back from the
    official reset. Without it only evaluation seed 0 and the reviewed layouts do.

    The default instance owns the workspace profile and the resolver derives
    the public environment context, so neither is written here. ``rounds`` of
    two or more start each round on a new reviewed layout: ``layout_schedule``
    None takes the task's default schedule, a sequence of layout ids selects
    inside its reviewed pool, and ``repeat_round0_layout`` is the explicit
    historical treatment that replays layout 0 (no ``layout_schedule``).

    ``layout_id`` k > 0 runs a one-round episode on official layout k
    (``single_layout``). Its instruction is the record's when the table calls
    it ``fixed``; a ``layout_readback`` instruction is ``task_instruction``
    (the lane gate's readback), or is left unbound so the scene smoke reads it
    back from the official reset.
    """
    if task not in task_roster(root):
        raise ValueError(f"unknown or uncontracted RoboDojo task: {task}")
    if task_file:
        if rounds >= 2 or layout_schedule is not None:
            raise ValueError("a RoboDojo task file runs one-round episodes")
        runtime = runtime_configuration(str(gpu), task, layout_id or 0, eval_seed=eval_seed)
        for owned_elsewhere in ("workspace_bounds_m", "public_environment_context"):
            runtime.pop(owned_elsewhere)
        runtime["resource_profile"] = resource_profile or f"robodojo-gpu{gpu}"
        from ora.orchestrator.preflight.resources import resolve_execution_resource
        binding = resolve_execution_resource(
            runtime["resource_profile"], runtime_kind=ENVIRONMENT, required_host=runtime["required_host"])
        if binding["worker_environment"].get("CUDA_VISIBLE_DEVICES") != runtime["cuda_visible_devices"]:
            raise ValueError("the resource profile and the runtime GPU differ")
        if task_instruction is not None:
            runtime["expected_task_language"] = task_instruction
        return runtime
    if eval_seed != 0:
        raise ValueError("only evaluation seed 0 has reviewed task contracts; name a task file (policy.task_file)")
    portable = single_layout(task, layout_id, root=root)
    if portable is not None and (rounds >= 2 or layout_schedule is not None):
        raise ValueError("a single-layout RoboDojo episode is one round; a persistent session selects its "
                         "layouts with a layout schedule")
    runtime = (runtime_configuration(str(gpu), task) if portable is None
               else runtime_configuration(str(gpu), task, layout_id))
    for owned_elsewhere in ("workspace_bounds_m", "public_environment_context"):
        runtime.pop(owned_elsewhere)
    runtime["resource_profile"] = resource_profile or f"robodojo-gpu{gpu}"
    from ora.orchestrator.preflight.resources import resolve_execution_resource
    binding = resolve_execution_resource(
        runtime["resource_profile"], runtime_kind=ENVIRONMENT, required_host=runtime["required_host"])
    if binding["worker_environment"].get("CUDA_VISIBLE_DEVICES") != runtime["cuda_visible_devices"]:
        raise ValueError("the resource profile and the runtime GPU differ")
    instruction = task_contract(task, root=root)["record"]["instruction"]["text"]
    if portable is not None and portable["instruction"] == INSTRUCTION_LAYOUT_READBACK:
        instruction = task_instruction
    elif task_instruction not in (None, instruction):
        raise ValueError(f"{task} renders its reviewed instruction on every layout, not {task_instruction!r}")
    if instruction is not None:
        runtime["expected_task_language"] = instruction
    if layout_schedule == REPEAT_ROUND0_LAYOUT or rounds < 2:
        if layout_schedule not in (None, REPEAT_ROUND0_LAYOUT):
            raise ValueError("a layout schedule needs two or more rounds")
        return runtime
    ids = None if layout_schedule is None else [int(layout_id) for layout_id in layout_schedule]
    return with_layout_schedule(runtime, robodojo_layout_schedule(task, rounds, layout_ids=ids))


def qualification_runtime(task: str, *, gpu: str, resource_profile: str | None = None, rounds: int = 1,
                          layout_schedule: Sequence[int] | str | None = None, layout_id: int | None = None,
                          root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """The runtime a RunSpec built here resolves to, in the shape the infrastructure probes run.

    The card and task records and the scene smoke must measure the
    configuration the RunPlan will carry: ``robodojo_qualification`` compares
    their configuration binding (``workspace_bounds_m`` included) and
    ``robodojo_revalidation`` their public geometry descriptor with the plan's.
    So this applies the default instance exactly as the resolver does
    (``expand_native_defaults``), then binds the workspace profile's bounds,
    the default surface's geometry descriptor and the slot's resource binding.
    It carries no qualification binding: the probes produce one. On a
    single layout whose instruction the layout generates it carries no
    ``expected_task_language`` either: the scene smoke reads it back.
    """
    from ora.agent.composition import resolve_tool_surface
    from ora.environment.workspace_profiles import resolve_workspace_profile
    from ora.orchestrator.spec.defaults import expand_native_defaults
    from ora.orchestrator.preflight.resources import resolve_execution_resource

    runtime = runtime_for(task, gpu=gpu, resource_profile=resource_profile, rounds=rounds,
                          layout_schedule=layout_schedule, layout_id=layout_id, root=root)
    expanded = expand_native_defaults(
        {"policy": {"kind": "native_api", "supports_images": True}, "runtime": runtime}, root=root)
    runtime = expanded["runtime"]
    runtime["workspace_bounds_m"] = resolve_workspace_profile(runtime.pop("workspace_profile"))["bounds_m"]
    surface = resolve_tool_surface(expanded["policy"]["tool_surface"]["profile"],
                                   toolkit_grants=expanded.get("toolkit_grants"))
    runtime["public_geometry_descriptor"] = copy.deepcopy(surface.view_config["public_geometry_descriptor"])
    runtime["resource_binding"] = resolve_execution_resource(
        runtime["resource_profile"], runtime_kind=ENVIRONMENT, required_host=runtime["required_host"])
    return runtime


def bind_services(spec: dict[str, Any], report: Path, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Bind a readiness report (``ora services qualify``) to the surface ``spec`` runs, as its prerequisite.

    The policy is the one source of the surface: its ``tool_surface``, else its
    ``baseline``'s, else the default. Preflight refuses a binding to any other
    surface, so a composer binds after ``select_policy``.
    """
    policy = spec["policy"]
    surface = policy.get("tool_surface")
    spec["prerequisite"] = run_spec_inputs.service_prerequisite(
        report, ENVIRONMENT, root=root, baseline=policy.get("baseline"),
        tool_surface=surface.get("profile") if isinstance(surface, Mapping) else None)
    return spec


def _reference(path: Path) -> dict[str, str]:
    from ora.orchestrator.paths import workspace_record_locator

    locator = workspace_record_locator(path, label="a qualification record")
    return {"path": locator, "sha256": hashlib.sha256(Path(path).resolve().read_bytes()).hexdigest()}


def layered_qualification(task_record: Path) -> dict[str, Any]:
    """The card/task pair a task record names (``backends.robodojo.lane_qualification task``).

    Records from ``backends.robodojo.card_qualification`` without a supplied runtime
    measure the wider compat workspace bounds, so a default RunPlan refuses
    their card configuration at preflight; produce them with
    ``backends.robodojo.lane_qualification``, which measures the plan's own runtime.
    """
    record = json.loads(Path(task_record).read_text(encoding="utf-8"))
    if record.get("passed") is not True or not isinstance(record.get("card_qualification"), Mapping):
        raise ValueError(f"{task_record} is not a passing RoboDojo task qualification record")
    return {"card": dict(record["card_qualification"]), "task": _reference(Path(task_record))}


def qualification_reference(path: Path) -> dict[str, Any]:
    """A binding stored in a file: a ``{path, sha256}`` record or a ``{card, task}`` pair (a lane gate)."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) not in ({"path", "sha256"}, {"card", "task"}):
        raise ValueError(f"{path} holds no RoboDojo qualification binding")
    return dict(value)


def gate_instruction(binding: Mapping[str, Any], *, task: str, layout_id: int) -> str:
    """The official instruction a lane gate's scene smoke read back on ``task``'s layout ``layout_id``.

    ``binding`` is the gate's ``{path, sha256}`` revalidation certificate;
    its configuration must be this task and layout. Launch admission
    re-evaluates the same certificate against the RunPlan's instruction.
    """
    from ora.orchestrator.robodojo_revalidation import SCHEMA, read_reference

    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"layout {layout_id} of {task} needs the scene-smoke lane gate of that layout "
                         "(backends.robodojo.lane_qualification smoke + revalidate --layout), not a card/task pair")
    try:
        record, _ = read_reference(dict(binding), WORKSPACE_ROOT)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"the lane gate of layout {layout_id} of {task} is unreadable: {exc}") from exc
    configuration = record.get("configuration") or {}
    instruction = record.get("task_instruction")
    if (record.get("schema") != SCHEMA or configuration.get("task_variant") != task
            or configuration.get("layout_id") != layout_id or not isinstance(instruction, str)
            or not instruction.strip()):
        raise ValueError(f"the lane gate binds no scene-smoke instruction of layout {layout_id} of {task}")
    return instruction


def surface_required_grants(baseline: str | None, tool_surface: str | None, *,
                            root: Path = PROJECT_ROOT) -> dict[str, Any] | None:
    """The grants a file-backed surface declares, for the surface a cell selects; else None."""
    from ora.agent.composition.baseline import load_native_baseline
    from ora.agent.composition.surface.toolkit_surface import declared_required_grants

    profile = tool_surface
    if profile is None and baseline is not None:
        profile = load_native_baseline(root, baseline)["instances"][ENVIRONMENT]["tool_surface"]
    return None if profile is None else declared_required_grants(profile)


def select_policy(spec: dict[str, Any], *, baseline: str | None = None, tool_surface: str | None = None,
                  context_scheme: str | None = None, toolkit_grants: Mapping[str, Any] | None = None,
                  toolkit_receipt_delivery: str | None = None, record_toolkit_io: bool = False,
                  ground_truth_capture: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Write a cell's policy choices into ``spec``; resolution validates the combination.

    Select before ``bind_services``: a bound prerequisite names the surface it was bound to.
    """
    if (baseline is not None or tool_surface is not None) and "prerequisite" in spec:
        raise ValueError("select the policy before binding the resident services (bind_services)")
    if baseline is not None:
        spec["policy"]["baseline"] = baseline
    if tool_surface is not None:
        spec["policy"]["tool_surface"] = {"profile": tool_surface}
    if context_scheme is not None:
        spec["policy"]["context_scheme"] = context_scheme
    grants = toolkit_grants if toolkit_grants is not None else surface_required_grants(
        baseline, tool_surface, root=root)
    if grants is not None:
        spec["toolkit_grants"] = copy.deepcopy(dict(grants))
    if toolkit_receipt_delivery is not None:
        spec["toolkit_receipt_delivery"] = toolkit_receipt_delivery
    if record_toolkit_io:
        spec["record_toolkit_io"] = True
    if ground_truth_capture:
        spec["runtime"]["ground_truth_capture"] = True
    return spec


def build_spec(task: str, *, experiment: str, gpu: str, infrastructure_qualification: Mapping[str, Any] | None = None,
               config: str | None = None, resource_profile: str | None = None,
               provider_profile: str | None = None, provider_catalog: Path | None = None,
               provider_recovery_window_s: float | None = None, rounds: int | None = None,
               layout_schedule: Sequence[int] | str | None = None, layout_id: int | None = None,
               budgets: Mapping[str, int] | None = None,
               cohort_id: str | None = None, milestone: str = "ROBODOJO",
               evidence_tier: str = "behavioral", eval_seed: int = 0, task_file: str | None = None,
               official_limit_multiple: int = TOOL_BUDGET_LIMIT_MULTIPLE,
               root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """One RoboDojo cell on the default baseline; only the RunSpec-owned choices are inputs.

    ``rounds`` None is the standard single episode; an integer is a K-round
    persistent session whose budgets are per round. ``layout_id`` k > 0 runs
    the single episode on official layout k. Qualification is optional;
    layout-dependent language without a gate is read from live reset. ``cohort_id`` adds a single-attempt launch admission for the cell.
    The resident services bind after policy selection (``select_policy``, then ``bind_services``).
    """
    default_instance(root)
    portable = None if task_file else single_layout(task, layout_id, root=root)
    instruction = None
    if portable is not None and infrastructure_qualification is not None:
        if set(infrastructure_qualification) != {"path", "sha256"}:
            # A task record qualifies layout 0 only.
            raise ValueError(f"layout {layout_id} of {task} needs the scene-smoke lane gate of that layout "
                             "(backends.robodojo.lane_qualification smoke + revalidate --layout), not a card/task pair")
        if portable["instruction"] == INSTRUCTION_LAYOUT_READBACK:
            try:
                instruction = gate_instruction(infrastructure_qualification, task=task, layout_id=layout_id)
            except FileNotFoundError:
                pass  # Absent qualification: official language is bound at live reset.
    runtime = runtime_for(task, gpu=gpu, resource_profile=resource_profile, rounds=rounds or 1,
                          layout_schedule=layout_schedule, layout_id=layout_id, task_instruction=instruction,
                          eval_seed=eval_seed, task_file=task_file is not None, root=root)
    if "layout_schedule" in runtime and infrastructure_qualification is not None and set(infrastructure_qualification) == {"card", "task"}:
        # A task record qualifies the round-0 layout only; a scheduled session
        # needs the lane gate of a scene smoke that relaunched every layout.
        raise ValueError("a layout-scheduled RoboDojo session needs a scene-smoke lane gate "
                         "(backends.robodojo.lane_qualification smoke + revalidate), not a card/task pair")
    if infrastructure_qualification is not None:
        runtime["infrastructure_qualification"] = copy.deepcopy(dict(infrastructure_qualification))
    if "expected_task_language" not in runtime:
        runtime["task_language_source"] = "live_reset_readback"
    runtime["official_limit_multiple"] = official_limit_multiple
    policy = provider_policy(provider_profile, catalog=provider_catalog)
    if provider_recovery_window_s is not None:
        policy["provider_recovery_window_s"] = provider_recovery_window_s
    cell = config or f"{task}-{layout_cell_tag(runtime.get('layout_schedule'), runtime['layout_id'])}-s{eval_seed}"
    spec: dict[str, Any] = {
        "schema": SPEC_SCHEMA, "experiment": experiment, "config": cell,
        "output_workspace": f"outputs/{experiment}", "milestone": milestone,
        "task": task, "task_instruction": runtime.get("expected_task_language",
            "Pending official task instruction from environment reset."), "seed": eval_seed,
        "evidence_tier": evidence_tier, "policy": policy, "runtime": runtime,
        "budgets": {**DEFAULT_BUDGETS, "max_sim_steps": physics_step_ceiling(task, official_limit_multiple),
                    **dict(budgets or {})},
        "qualification": {"required": False},
    }
    if rounds is not None:
        spec["session"] = {"cadence": SESSION_CADENCE, "rounds": int(rounds)}
    if cohort_id is not None:
        spec["launch_admission"] = single_attempt_admission(cohort_id, cell)
    if task_file is not None:
        policy["task_file"] = task_file
    return spec


def compose_task(task: Mapping[str, Any], binding: Mapping[str, Any], *,
                 root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """One RoboDojo run of an ``ora.task.v1`` task on its run binding (``ora task run``)."""
    policy = task.get("policy") or {}
    spec = build_spec(
        task["task"], experiment=task["experiment"], gpu=str(binding["gpu"]), config=binding.get("config"),
        resource_profile=binding.get("resource_profile"), provider_profile=binding["provider_profile"],
        provider_catalog=Path(binding["provider_catalog"]), layout_id=task["layout"] or None,
        budgets=task.get("budgets") or None,
        cohort_id=binding.get("cohort"), eval_seed=task["seed"], task_file=policy.get("task_file"), root=root,
    )
    select_policy(
        spec, baseline=policy.get("baseline"), tool_surface=policy.get("tool_surface"),
        context_scheme=policy.get("context_scheme"), toolkit_grants=policy.get("toolkit_grants"),
        toolkit_receipt_delivery=policy.get("toolkit_receipt_delivery"),
        record_toolkit_io=bool(task.get("record_toolkit_io")),
        ground_truth_capture=bool(task.get("ground_truth_capture")), root=root,
    )
    if binding.get("service_report"):
        bind_services(spec, Path(binding["service_report"]), root=root)
    return spec


def compose(spec: Mapping[str, Any]) -> Path:
    """Write ``spec`` under its experiment workspace, where plan/preflight/launch read it."""
    bind_service_environment(spec)
    path = experiment_workspace(spec["experiment"]) / "specs" / f"{spec['config']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _qualification_argument(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.task_qualification is not None:
        return layered_qualification(args.task_qualification)
    return (qualification_reference(args.qualification_reference)
            if args.qualification_reference is not None else None)


def main(argv: list[str] | None = None) -> int:
    """Operator entry point: ``ora spec robodojo``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("tasks", help="list the contracted tasks with their physics device and limits")
    for name, text in (("spec", "print the RunSpec"), ("compose", "write the RunSpec under outputs/<experiment>/specs")):
        command = commands.add_parser(name, help=text)
        command.add_argument("task")
        command.add_argument("--experiment", required=True, help="the new experiment's execution namespace")
        command.add_argument("--gpu", required=True, choices=list("01234567"))
        command.add_argument("--resource-profile", help="a registered RoboDojo slot (default robodojo-gpu<N>)")
        command.add_argument("--provider-profile", help="default: the shared catalog's default profile")
        command.add_argument("--provider-catalog", type=Path, help="an experiment-owned Provider catalog")
        command.add_argument("--rounds", type=int, help="a K-round persistent session (default: one episode)")
        command.add_argument("--layouts", type=int, nargs="+", help="per-round layout ids inside the reviewed pool")
        command.add_argument("--layout", type=int, dest="layout_id",
                             help="one episode on this official layout of seed 0 (default 0); qualification is optional")
        command.add_argument("--service-report", type=Path, help="readiness report from `ora services qualify`")
        command.add_argument("--cohort", help="single-attempt launch admission cohort id")
        command.add_argument("--config", help="cell id (default <task>-l<layouts>-s0)")
        command.add_argument("--official-limit-multiple", type=int, default=TOOL_BUDGET_LIMIT_MULTIPLE,
                             help="run on to this multiple of the official native-action limit; "
                                  "1 stops at the official limit")
        command.add_argument("--baseline", help="a registered baseline id (default: the tool surface's, else the default)")
        command.add_argument("--tool-surface", help="a registered tool surface profile id")
        command.add_argument("--context-scheme", help="a named context scheme replacing the baseline's context")
        command.add_argument("--toolkit-grants", type=json.loads,
                             help="toolkit grants as JSON (default: what a file-backed surface requires)")
        command.add_argument("--toolkit-receipt-delivery", choices=("observation", "tool_output"))
        command.add_argument("--record-toolkit-io", action="store_true", help="record each toolkit call's interface I/O")
        command.add_argument("--ground-truth-capture", action="store_true",
                             help="capture ground truth outside the sealed run")
        source = command.add_mutually_exclusive_group(required=False)
        source.add_argument("--task-qualification", type=Path,
                            help="task-qualification.json from backends.robodojo.lane_qualification task (one round)")
        source.add_argument("--qualification-reference", type=Path,
                            help="a lane gate from backends.robodojo.lane_qualification revalidate, or another file "
                                 "holding a {path, sha256} or {card, task} binding")
    args = parser.parse_args(argv)
    if args.command == "tasks":
        from ora.environment.benchmarks.robodojo_physics import physics_device_for
        print(json.dumps({task: {"physics_device": physics_device_for(task),
                                 "native_action_limit": NATIVE_ACTION_LIMITS[task],
                                 "official_limit_multiple": TOOL_BUDGET_LIMIT_MULTIPLE,
                                 "max_sim_steps": physics_step_ceiling(task, TOOL_BUDGET_LIMIT_MULTIPLE)}
                          for task in task_roster()},
                         indent=2))
        return 0
    spec = build_spec(
        args.task, experiment=args.experiment, gpu=args.gpu,
        infrastructure_qualification=_qualification_argument(args), config=args.config,
        resource_profile=args.resource_profile, provider_profile=args.provider_profile,
        provider_catalog=args.provider_catalog, rounds=args.rounds, layout_schedule=args.layouts,
        layout_id=args.layout_id, cohort_id=args.cohort,
        official_limit_multiple=args.official_limit_multiple,
    )
    select_policy(
        spec, baseline=args.baseline, tool_surface=args.tool_surface, context_scheme=args.context_scheme,
        toolkit_grants=args.toolkit_grants, toolkit_receipt_delivery=args.toolkit_receipt_delivery,
        record_toolkit_io=args.record_toolkit_io, ground_truth_capture=args.ground_truth_capture,
    )
    if args.service_report is not None:
        bind_services(spec, args.service_report)
    if args.command == "spec":
        print(json.dumps(spec, ensure_ascii=False, indent=2))
    else:
        print(compose(spec))
    return 0
