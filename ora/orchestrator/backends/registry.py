"""The orchestrator's simulator backends, by ``runtime.kind``.

The orchestrator core never imports a backend module. It asks this registry
for the backend of a plan's ``runtime.kind`` and calls what that backend
declares: ``BACKENDS`` maps each kind to the ``"module:attribute"`` string of
its ``OrchestratorBackend``, imported on first use and cached, the same form
``ora.environment.definitions`` uses for its isolated workers. A backend may
import the core freely; nothing the core imports at module level imports a
backend, so the two never form an import cycle.

An ``OrchestratorBackend`` has two required hooks, ``resolve_runtime`` (the
RunSpec resolver of the kind's own runtime fields) and ``preflight_runtime``
(its launch admission checks), plus ``BackendCapabilities``, plain data the
core reads instead of naming kinds. Every other hook is optional: only one or
two backends implement it, and the core applies its own default (no check, no
extra field, nothing) when the hook is ``None``. Each hook's comment below
names the core module that calls it.
"""

from __future__ import annotations

import importlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

Check = dict[str, Any]
#: An admission verdict and its evidence.
Verdict = tuple[bool, dict[str, Any]]

#: runtime.kind -> the ``module:attribute`` of its backend, in the order the
#: RunSpec resolver names the kinds.
BACKENDS = {
    "synthetic": "ora.orchestrator.backends.synthetic:BACKEND",
    "robotwin": "ora.orchestrator.backends.robotwin.backend:BACKEND",
    "libero_pro": "ora.orchestrator.backends.libero_pro.backend:BACKEND",
    "robodojo": "ora.orchestrator.backends.robodojo.backend:BACKEND",
    "robocasa": "ora.orchestrator.backends.robocasa.backend:BACKEND",
    "robolab": "ora.orchestrator.backends.robolab.backend:BACKEND",
    "yam": "ora.orchestrator.backends.yam.backend:BACKEND",
}
#: Other names of a kind: ``libero`` is LIBERO-Pro's name in ``ora run``, ``ora play`` and ``ora setup``.
ALIASES = {"libero": "libero_pro"}
#: The key of a backend's default manifest admitter: it admits a manifest whose
#: schema no backend names (RoboTwin's older per-task seed manifests carry none).
ANY_SCHEMA = "*"


class UnknownBackend(KeyError):
    """No backend is registered under this name."""


@dataclass(frozen=True)
class PreflightContext:
    """What a backend's preflight hook receives from ``preflight/gates.py``.

    ``gates.preflight_context`` builds it at call time from its own module
    globals, so a test that replaces ``gates._git`` or
    ``gates.resolve_workspace_root`` reaches the backend checks too.
    """

    project_root: Path
    check: Callable[[str, bool, Any], Check]
    git: Callable[..., subprocess.CompletedProcess[str]]
    workspace_root: Callable[[Path], Path]
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run


@dataclass(frozen=True)
class BackendCapabilities:
    """Plain data the core reads instead of naming kinds; the defaults are the conservative choice."""

    #: The runtime always resolves ``execution_capture``; without it, only when the RunSpec asks
    #: (spec.runtime).
    captures_execution: bool = True
    #: Persistent multi-round sessions are supported (spec.runtime).
    persistent_session: bool = False
    #: A completed run must carry a Boolean official verdict (evaluation.checks).
    boolean_verdict_required: bool = False
    #: A native_api policy must accept images (spec.runtime).
    native_requires_images: bool = False
    #: ``runtime.environment_signal_policy`` values the backend implements (spec.runtime).
    signal_policies: frozenset[str] = frozenset({"lifecycle_only"})
    #: The reach check's inter-arm screen applies (spec.runtime).
    #: A baseline catalog may carry an instance of this kind (defaults).
    baseline_instance: bool = False
    #: The maintained baseline runs only on an explicit qualified seed manifest (preflight).
    baseline_requires_seed_manifest: bool = False
    #: Physics steps one native action advances, for a backend with an official
    #: native-action limit (native_allowance).
    physics_steps_per_native_action: int | None = None
    #: The plan's own step cap (``budgets.max_sim_steps``) is the episode's native-action
    #: limit, counted in control ticks, when the runtime registers no official one
    #: (native_allowance): LIBERO, whose evaluation protocol is a per-suite step limit.
    step_cap_is_native_limit: bool = False
    #: Why the run writes outside TMPDIR, so the system partitions are checked (host_storage).
    kit_storage_reason: str | None = None
    #: Repository-relative policy files the backend binds outside the policy tree's own
    #: references (policy_pins).
    pinned_descriptors: tuple[str, ...] = ()


class Admitter(Protocol):
    """Admit one seed of a qualification manifest for a plan (preflight)."""

    def __call__(self, manifest: Mapping[str, Any], plan: Mapping[str, Any], *, seed: int, root: Path,
                 memo: dict[str, Any]) -> Verdict: ...


class PlayBackend(Protocol):
    """A backend's free-play scenes (play.scenes) and their plan binding (play.binding, spec.play)."""

    #: The ``ora play --env`` name, and how a refusal names the backend.
    environment: str
    display_name: str
    #: The backend offers ORA's own scene sets (``ora play --scene-set``).
    scene_sets: bool
    #: The plan imports and binds the scene's record as its qualification manifest.
    bound_record: bool

    def list_scenes(self, *, task: str | None, task_config: str | None, scene_set: str | None, root: Path,
                    **sources: Any) -> list[Any]: ...

    def draw(self, rng: Any, listing: dict[str, Any]) -> Any: ...

    def label(self, scene: Any) -> str: ...

    def build_spec(self, spec: dict[str, Any], chosen: Any, *, host: str | None, resource_profile: str | None,
                   service_report: Path | None, root: Path) -> None: ...

    def refuse_benchmark(self, runtime: Mapping[str, Any]) -> None: ...

    def bind_runtime(self, runtime: Mapping[str, Any], scene: Mapping[str, Any],
                     scenes: Sequence[Mapping[str, Any]]) -> dict[str, Any]: ...

    def require_scenes(self, runtime: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]]) -> None: ...


class RunCell(Protocol):
    """One ``ora run --env`` environment: its slot, inputs and composer (ora.app.run).

    ``slot`` is the single-gpu host's slot on GPU 0; ``compose`` takes ``args.gpu``, the GPU index
    ``ora run --gpu`` names (None for 0), and runs on that GPU's slot."""

    slot: str
    required_env: tuple[str, ...]
    example: str

    def compose(self, args: Any, *, common: Mapping[str, Any], host: str, services: Callable[[], Path],
                confirm: Callable[[str], None], stream: Any) -> Path: ...


@dataclass(frozen=True)
class Tool:
    """One operator tool of a backend: ``ora spec KIND NAME`` or ``ora qualify KIND NAME`` (ora.app).

    ``ora spec robotwin compose TASK ...`` runs the backend's ``compose`` tool:
    ``function([*arguments, TASK, ...])``. A function that serves several tools
    (one parser with a subcommand per tool) gets its subcommand in ``arguments``.
    A run cell starts a tool as the subprocess an operator would type with
    ``command()``, which needs the module's ``python -m`` entry point.
    """

    #: ``"module:function"``; ``function(argv)`` returns the exit status.
    target: str
    #: One line for ``ora spec KIND`` / ``ora qualify KIND`` without a tool.
    help: str
    #: Put before the rest of the line: the tool's subcommand in a shared parser.
    arguments: tuple[str, ...] = ()

    @property
    def module(self) -> str:
        return self.target.partition(":")[0]

    def command(self, python: str, *argv: str) -> list[str]:
        """``python -m MODULE [ARGUMENTS] ARGV``: the tool in its own process."""

        return [python, "-m", self.module, *self.arguments, *argv]


@dataclass(frozen=True)
class DoctorProbe:
    """What ``ora doctor --env KIND`` discovers without starting the simulator (doctor)."""

    #: The variable naming the isolated simulator interpreter, and the module it must find.
    python_env: str
    module: str
    #: Directories that must exist.
    directory_envs: tuple[str, ...] = ()
    #: A variable whose ``<value>/<subdirectory>`` the probe puts on ``sys.path`` first.
    source_env: tuple[str, str] | None = None
    #: Further read-only checks: ``(interpreter, nvidia_smi) -> [check]``.
    extra_checks: Callable[[str, str | None], list[dict[str, Any]]] | None = None


@dataclass(frozen=True)
class OrchestratorBackend:
    """One simulator backend's orchestration: two required hooks, data, and optional hooks."""

    kind: str
    #: ``(runtime, policy, resolved_surface, *, task, seed, task_instruction)``: validate and
    #: default the kind's own runtime fields (spec.runtime).
    resolve_runtime: Callable[..., None]
    #: ``(plan, context) -> [check]``: launch admission of the runtime (preflight).
    preflight_runtime: Callable[[Mapping[str, Any], PreflightContext], list[Check]]
    capabilities: BackendCapabilities = BackendCapabilities()
    #: ``(runtime, session) -> refusal | None`` for a resolved session, None for one episode
    #: (spec.runtime).
    validate_session: Callable[[Mapping[str, Any], Mapping[str, Any] | None], str | None] | None = None
    #: ``(runtime, template, requested_baseline) -> template``: adjust the baseline instance's
    #: runtime template for this RunSpec (defaults).
    expand_template: Callable[[Mapping[str, Any], dict[str, Any], str | None], dict[str, Any]] | None = None
    #: ``(policy, runtime_spec)``: provider defaults the backend's RunSpecs inherit (spec.native_provider).
    provider_defaults: Callable[[dict[str, Any], Mapping[str, Any]], None] | None = None
    #: ``runtime -> text`` for the plan display; the default is ``runtime.required_revision`` (cli).
    revision_label: Callable[[Mapping[str, Any]], str | None] | None = None
    #: manifest schema (or ``ANY_SCHEMA``) -> admitter of a required qualification manifest (preflight).
    manifest_admitters: Mapping[str, Admitter] = field(default_factory=dict)
    #: A labelled exception to seed qualification: ``resolve_exception(qualification, bind)``,
    #: raising its ``QualificationExceptionError``, at resolve time (spec), and
    #: ``check_exception(plan, workspace_root)`` at preflight (preflight).
    qualification_exception: Any = None
    #: The backend's seed-manifest rules (``seed_admission``, ``expert_unsolved``) for its RunSpec
    #: builders (spec.benchmark).
    seed_manifest: Any = None
    #: ``(run_root, runtime) -> binding | None``: the preflight binding the Supervisor hands the
    #: environment (episode.bindings).
    supervisor_binding: Callable[[Path, Mapping[str, Any]], dict[str, Any] | None] | None = None
    #: ``require_round_scene(plan, reset, index)``, ``round_scene_field(plan, reset)`` and
    #: ``layout_change_sentence(plan, index)`` of a scheduled round (episode.loop, episode.rounds).
    round_scene: Any = None
    #: ``(official, round_results, *, multi_round, round_limit) -> bool``: whether a finished run
    #: carries its official outcome; an incomplete one is marked so (episode.result).
    official_complete: Callable[..., bool] | None = None
    #: ``(run_root, plan, result, events) -> [reason]``: backend evidence a valid run needs (evaluation).
    evidence_errors: Callable[..., list[str]] | None = None
    #: ``(plan, record, index) -> [reason]``: one round record of a multi-round session (evaluation).
    round_record_errors: Callable[[Mapping[str, Any], Mapping[str, Any], int], list[str]] | None = None
    #: ``(plan, run_root) -> {field: value}`` added to ``evaluation.json`` (evaluation).
    evaluation_extras: Callable[[Mapping[str, Any], Path], dict[str, Any]] | None = None
    #: ``evaluation -> {field: value}`` added to ``seal.json`` (evaluation).
    seal_extras: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None
    #: Free-play scenes (play.scenes, play.binding).
    play: PlayBackend | None = None
    #: The play-scene audit module: ``KIND``, ``AUDIT_SCHEMA``, ``audit_root`` and
    #: ``play_scene_admission`` (play.scenes); its ``main(argv)`` audits scenes
    #: (``ora play audit --env KIND``, ora.app.play).
    play_audit: Any = None
    #: ``ora run --env`` (ora.app.run).
    run_cell: RunCell | None = None
    #: ``(task, binding) -> RunSpec``: one run of an ``ora.task.v1`` task on its run binding
    #: (orchestrator.task.run).
    compose_task: Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]] | None = None
    #: ``ora spec KIND NAME``: the RunSpec composer's commands, such as ``tasks`` and ``compose`` (ora.app.spec).
    spec_tools: Mapping[str, Tool] = field(default_factory=dict)
    #: ``ora qualify KIND NAME``: the backend's qualification tools (ora.app.qualify); a run cell
    #: starts its own with ``Tool.command``.
    qualify_tools: Mapping[str, Tool] = field(default_factory=dict)
    #: Toolkit-only live diagnostic context manager; never the episode worker route.
    toolkit_live: Callable[..., Any] | None = None
    #: ``ora doctor --env`` (doctor).
    doctor: DoctorProbe | None = None
    #: The ``ora setup --env`` components the backend runs on, in install order (ora.app.gpu_recipes).
    gpu_setup: tuple[str, ...] | None = None


_LOADED: dict[str, OrchestratorBackend] = {}


def canonical(kind: Any) -> str | None:
    """The registered kind ``kind`` names (itself or through an alias), or None."""

    name = ALIASES.get(kind, kind) if isinstance(kind, str) else None
    return name if name in BACKENDS else None


def backend(kind: str) -> OrchestratorBackend:
    """The backend registered for ``kind`` (or its alias), imported on first use."""

    name = canonical(kind)
    if name is None:
        raise UnknownBackend(f"no orchestrator backend is registered for runtime.kind {kind!r}")
    if name not in _LOADED:
        module, _, attribute = BACKENDS[name].partition(":")
        loaded = getattr(importlib.import_module(module), attribute)
        if not isinstance(loaded, OrchestratorBackend) or loaded.kind != name:
            raise TypeError(f"{BACKENDS[name]} is not the OrchestratorBackend of {name}")
        _LOADED[name] = loaded
    return _LOADED[name]


def find(kind: Any) -> OrchestratorBackend | None:
    """``backend(kind)``, or None for a kind no backend registers (an unresolved or foreign plan)."""

    name = canonical(kind)
    return None if name is None else backend(name)


def backends() -> tuple[OrchestratorBackend, ...]:
    """Every registered backend, in ``BACKENDS`` order (imports them all)."""

    return tuple(backend(name) for name in BACKENDS)


def manifest_admitter(schema: Any, *, kind: Any) -> Admitter | None:
    """The admitter of a qualification manifest of ``schema`` for a plan of ``kind``.

    The plan's own backend is asked first, then the others: a manifest is
    judged by the backend that names its schema. A schema no backend names
    falls to a default admitter (``ANY_SCHEMA``), the plan's own first.
    """

    own = find(kind)
    ordered = [own] if own is not None else []
    if own is None or str(schema) not in own.manifest_admitters:
        ordered += [item for item in backends() if item is not own]
    for key in (str(schema), ANY_SCHEMA):
        for item in ordered:
            if key in item.manifest_admitters:
                return item.manifest_admitters[key]
    return None


__all__ = [
    "ALIASES", "ANY_SCHEMA", "Admitter", "BACKENDS", "BackendCapabilities", "Check", "DoctorProbe",
    "OrchestratorBackend", "PlayBackend", "PreflightContext", "RunCell", "Tool", "UnknownBackend", "Verdict",
    "backend", "backends", "canonical", "find", "manifest_admitter",
]
