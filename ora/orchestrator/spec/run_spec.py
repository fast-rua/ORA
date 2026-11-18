"""RunSpec validation, resource binding and deterministic identity.

``resolve_run_spec`` turns one RunSpec into a content-addressed RunPlan as a
fixed sequence of section resolvers:

- ``policy.py``: the fields every policy kind shares, the ToolSurface and
  its action codec; ``POLICY_KIND_RESOLVERS`` then selects the scripted or
  native_api (``native_policy.py``, ``native_provider.py``) resolver;
- ``runtime.py``: the runtime by kind, budgets and session;
- ``composition.py``: prompt layers and the hashed policy composition.

This module keeps the episode fields, qualification, launch admission and the
RunPlan identity, and verifies persisted plans. Several names imported below
are used only by the section modules; they remain importable from here.
"""

from __future__ import annotations

from ora.orchestrator.spec.experiment_identity import bind_identity, validate_identity
from ora.orchestrator.backends.registry import backend, find
from ora.orchestrator.spec.values import (
    RunSpecError, canonical_bytes, json_sha256, file_sha256, official_task_language, _mapping, _positive_int,
    _stable_id, _resource, _resource_list,
)

import copy
import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, NamedTuple
from urllib.parse import urlsplit

import yaml

from ..paths import PROJECT_ROOT, resolve_workspace_root
from . import latency
from .environment_context import resolve_typed_environment_prompt_profile
from .defaults import expand_native_defaults
from ora.agent.composition.surface import tool_names
from ora.agent.memory.artifacts import (
    LOSSLESS_CONTEXT_RETENTION,
    recoverable_artifact_definitions,
    recoverable_artifact_result_schemas,
)
from ora.agent.composition import (
    POLICY_CONTEXT_V3_SCHEMA, V2_CHANNELS,
    ToolSurfaceError,
    resolve_tool_surface,
)
from ora.agent.composition.embodiment.environment_context import (
    EnvironmentContextError,
    resolve_environment_prompt_profile,
)
from ora.contracts.workspace import WorkspaceError, normalize_workspace_bounds
from ora.environment.definitions import (
    SPARSE_MOTION_QUALIFIED_ENVIRONMENTS,
)
from ora.environment.workspace_profiles import resolve_workspace_profile
from ..preflight.resources import ExecutionResourceError, resolve_execution_resource
from .native_provider import (
    resolve_native_provider_endpoint, resolve_native_provider_protocol,
)
from .composition import bind_policy_context, compose_policy
from .context_compactor import resolve_context_compactor
from .native_policy import resolve_native_policy
from .operator_control import resolve_operator_control, validate_persisted_operator_control
from .play import bind_play_runtime, resolve_play, verify_play_qualification
from .policy import (
    POLICY_KINDS, resolve_policy_surface, resolve_scripted_policy,
)
from .runtime import (
    ENVIRONMENT_SIGNAL_POLICIES, ENVIRONMENT_TRANSPORTS, RUNTIME_KINDS,
    SESSION_CADENCES, resolve_budgets, resolve_runtime, resolve_session,
)
from .source import bind_policy_identity, valid_plan_identity
from .partition import resolve_partition
from .result_tier import resolve_result_tier


SCHEMA = "ora.run-spec.v1"
PLAN_SCHEMA = "ora.run-plan.v1"
EVIDENCE_TIERS = {"diagnostic", "infrastructure", "behavioral"}
LAUNCH_ADMISSION_SCHEMA = "ora.run-launch-admission.v1"

# One resolver per policy.kind for the fields only that kind has. Each runs
# after ``resolve_policy_surface`` and normalizes the policy copy in place.
POLICY_KIND_RESOLVERS = {
    "scripted": resolve_scripted_policy,
    "native_api": resolve_native_policy,
}


def validate_run_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Verify one persisted RunPlan's complete content-addressed identity."""

    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise RunSpecError("persisted RunPlan schema is invalid")
    validate_persisted_operator_control(plan)
    if "experiment_identity" in plan:
        try:
            validate_identity(plan["experiment_identity"])
        except ValueError as exc:
            raise RunSpecError(str(exc)) from exc
    source = plan.get("source")
    if not valid_plan_identity(plan):
        raise RunSpecError("persisted RunPlan source identity is invalid")
    if source["dirty"]:
        raise RunSpecError("persisted RunPlan was resolved from dirty source")
    if "launch_admission" in plan and _launch_admission(plan["launch_admission"]) != plan["launch_admission"]:
        raise RunSpecError("persisted RunPlan launch admission is invalid")
    portable = {
        str(key): copy.deepcopy(value)
        for key, value in plan.items()
        if key not in {"plan_sha256", "plan_id", "confirmation_code"}
    }
    portable["source"] = {"commit": source["commit"]}
    expected_hash = json_sha256(portable)
    expected_id = (
        f"{plan.get('experiment')}-{str(plan.get('milestone', '')).lower()}-"
        f"{expected_hash[:12]}"
    )
    if plan.get("plan_sha256") != expected_hash:
        raise RunSpecError("persisted RunPlan hash does not match its content")
    if plan.get("plan_id") != expected_id:
        raise RunSpecError("persisted RunPlan id does not match its content")
    if plan.get("confirmation_code") != expected_hash[-6:]:
        raise RunSpecError("persisted RunPlan confirmation code is invalid")
    return copy.deepcopy(dict(plan))


def _git(*args: str, root: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True,
        timeout=60, check=False,
    )
    if result.returncode:
        raise RunSpecError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _launch_admission(value: Any) -> dict[str, Any] | None:
    """Normalize launch/retry admission: disabled (context only) or one single-attempt cohort cell.

    A plan recorded with a launch qualification ``gate`` keeps it, so such a
    plan still validates; a new RunSpec may not bind one
    (``_bind_launch_admission``).
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RunSpecError("launch_admission must be an object")
    mode = value.get("mode")
    if mode == "disabled":
        if set(value) != {"schema", "mode"}:
            raise RunSpecError("disabled launch_admission fields are not closed")
        if value.get("schema") != LAUNCH_ADMISSION_SCHEMA:
            raise RunSpecError("launch_admission schema is invalid")
        return {"schema": LAUNCH_ADMISSION_SCHEMA, "mode": "disabled"}
    allowed_fields = {
        "schema", "mode", "cohort_id", "cell_id", "retry_allowed",
    }
    if "gate" in value:
        allowed_fields.add("gate")
    if mode != "single_attempt_cell" or set(value) != allowed_fields:
        raise RunSpecError("launch_admission fields are invalid")
    if value.get("schema") != LAUNCH_ADMISSION_SCHEMA:
        raise RunSpecError("launch_admission schema is invalid")
    if value.get("retry_allowed") is not False:
        raise RunSpecError(
            "single_attempt_cell launch_admission must disable retry"
        )
    normalized = {
        "schema": LAUNCH_ADMISSION_SCHEMA,
        "mode": "single_attempt_cell",
        "cohort_id": _stable_id(value.get("cohort_id"), "launch cohort"),
        "cell_id": _stable_id(value.get("cell_id"), "launch cell"),
        "retry_allowed": False,
    }
    if "gate" in value:
        normalized["gate"] = copy.deepcopy(value["gate"])
    return normalized


def _generated_resource(
    root: Path, experiment: str, value: Any, label: str,
) -> tuple[dict[str, Any], Path]:
    """Bind an ignored resource in the Git-common experiment workspace."""

    if not isinstance(value, str) or not value:
        raise RunSpecError(f"{label} must be a project-relative path")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:2] != ("outputs", experiment)
    ):
        raise RunSpecError(f"{label} must remain under outputs/{experiment}")
    workspace_root = resolve_workspace_root(root)
    path = (workspace_root / relative).resolve()
    experiment_root = (workspace_root / "outputs" / experiment).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.is_relative_to(experiment_root)
    ):
        raise RunSpecError(f"{label} is unavailable: {value}")
    return ({
        "path": relative.as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }, path)


def load_run_spec(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RunSpecError(f"RunSpec is unavailable: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RunSpecError(f"cannot load RunSpec: {exc}") from exc
    if not isinstance(value, dict):
        raise RunSpecError("RunSpec root must be an object")
    return value


class _Episode(NamedTuple):
    """The validated episode fields of a RunSpec."""

    experiment: str
    config: str
    milestone: str
    task: str
    task_instruction: str
    seed: int
    evidence_tier: str
    output_workspace: str


def resolve_run_spec(spec: Mapping[str, Any], *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Validate one RunSpec and bind it into a content-addressed RunPlan.

    Sections resolve in this fixed order, each raising the first error it
    finds, so a RunSpec with several invalid fields always reports the same
    one. The policy and runtime sections default and normalize shallow copies
    of ``spec["policy"]`` and ``spec["runtime"]``.
    """

    root = root.resolve()
    if spec.get("schema") != SCHEMA:
        raise RunSpecError(f"RunSpec schema must be {SCHEMA}")
    spec = _expand_defaults(spec, root=root)
    if type(spec.get("record_toolkit_io", False)) is not bool:
        raise RunSpecError("record_toolkit_io must be true or false")
    if type(spec.get("save_sandbox", True)) is not bool:
        raise RunSpecError("save_sandbox must be true or false")
    episode = _resolve_episode(spec)

    policy, surface, profile_id = resolve_policy_surface(spec)
    legacy_visible = surface.supports("toolkit_legacy_projection_v1")
    if "toolkit_receipt_delivery" in spec and not legacy_visible:
        raise RunSpecError("toolkit_receipt_delivery requires a legacy-visible toolkit surface")
    if legacy_visible:
        from ora.agent.composition.surface.toolkit_surface import declared_receipt_delivery

        # A file-backed surface declares its delivery; the layered ones default to observation.
        delivery = spec.get("toolkit_receipt_delivery", declared_receipt_delivery(profile_id) or "observation")
        if delivery not in {"observation", "tool_output"}:
            raise RunSpecError("toolkit_receipt_delivery must be observation or tool_output")
        if "toolkit_observation_receipts" in spec and spec["toolkit_observation_receipts"] != (delivery == "observation"):
            raise RunSpecError("toolkit_observation_receipts conflicts with toolkit_receipt_delivery")
        spec = dict(spec, toolkit_observation_receipts=delivery == "observation")
    if surface.supports("toolkit_synchronous_execution_v1") and spec.get("runtime", {}).get("kind") != "robodojo":
        raise RunSpecError("synchronous toolkit surface requires RoboDojo")
    if spec.get("record_toolkit_io", False) and surface.runtime_id != "native_atomic_causal":
        raise RunSpecError("record_toolkit_io requires the manipulation tool runtime")
    POLICY_KIND_RESOLVERS[policy["kind"]](policy, surface, spec.get("runtime"))
    if legacy_visible and policy.get("code_mode") and delivery != "tool_output":
        # A script reads each action's receipt from the tool's return value; with observation delivery the
        # legacy-visible surface's action tools return only a submitted confirmation.
        raise RunSpecError(f"policy.code_mode on {surface.profile_id} needs toolkit_receipt_delivery: "
                           "tool_output, or a script's actions return no receipt")
    _refuse_memory(policy)
    if "context_compactor" in policy:  # only a plan that names one carries the key
        resolve_context_compactor(policy, spec, root=root)
    operator_control = resolve_operator_control({**spec, "policy": policy}) if "operator_control" in spec else None

    latency_selection = latency.select(spec.get("latency_profile"))
    selected_backend = find((spec.get("runtime") if isinstance(spec.get("runtime"), Mapping) else {}).get("kind"))
    runtime_spec = latency.with_capture(spec, latency_selection, captures=bool(
        selected_backend is not None and selected_backend.capabilities.captures_execution))
    runtime, surface = resolve_runtime(
        runtime_spec, policy, surface, task=episode.task, seed=episode.seed,
        # A free-play policy never sees the scene's task language; the backend's
        # identity checks still compare it (values.official_task_language).
        task_instruction=official_task_language({**spec, "task_instruction": episode.task_instruction}),
        root=root,
    )
    synchronous = surface.supports("toolkit_synchronous_execution_v1")
    budgets = resolve_budgets(spec, policy, runtime, synchronous_tools=synchronous)
    if type(spec.get("toolkit_observation_receipts", not synchronous)) is not bool:
        raise RunSpecError("toolkit_observation_receipts must be true or false")
    play = resolve_play(spec, policy, runtime, budgets, seed=episode.seed, task=episode.task)
    result_tier = resolve_result_tier(spec, policy, surface, play)
    runtime = bind_play_runtime(runtime, play)
    session = resolve_session(spec, policy, runtime, surface, seed=episode.seed)

    policy_context = bind_policy_context(
        spec, policy, runtime, surface, profile_id=profile_id, root=root,
    )
    composition = compose_policy(policy, surface, policy_context, root=root)
    qualification = _resolve_qualification(spec, episode.experiment, kind=runtime["kind"], root=root)
    verify_play_qualification(play, qualification)

    from .source_overlay import OVERLAY_FILE, source_state

    overlay = None
    if (root / OVERLAY_FILE).is_file():
        source = source_state(root)
        if source.error is not None:
            raise RunSpecError(source.error)
        source_commit, source_dirty, overlay = source.commit, source.dirty, source.overlay
    else:
        source_commit = _git("rev-parse", "HEAD", root=root)
        source_dirty = bool(_git("status", "--porcelain", "--untracked-files=normal", root=root))
    launch_admission = _bind_launch_admission(spec)
    portable = {
        "schema": PLAN_SCHEMA,
        "experiment": episode.experiment,
        "config": episode.config,
        "milestone": episode.milestone,
        "task": episode.task,
        "task_instruction": episode.task_instruction.strip(),
        "seed": episode.seed,
        "evidence_tier": episode.evidence_tier,
        "result_tier": result_tier,
        "output_workspace": episode.output_workspace,
        "policy": policy,
        "runtime": runtime,
        "budgets": budgets,
        "policy_context": policy_context,
        "policy_composition": composition,
        "qualification": qualification,
        "prerequisite": spec.get("prerequisite"),
        # The switches the run's control path keeps (spec/latency.py); every plan records them.
        "latency_profile": latency.record(latency_selection, spec, runtime),
        "source": {"commit": source_commit},
    }
    if overlay is not None:
        # Files replaced on the base commit (source_overlay.py); the plan hash covers them.
        portable["source_overlay"] = overlay
    if spec.get("record_toolkit_io", False):
        portable["record_toolkit_io"] = True
    if spec.get("save_sandbox", True) is False:
        # Only an opted-out plan carries the key: sandbox/ is reduced to its hashes after sealing
        # (agent/evidence/sandbox_retention.py).
        portable["save_sandbox"] = False
    if legacy_visible:
        portable["toolkit_receipt_delivery"] = delivery
    if synchronous:
        from ora.agent.composition.surface.toolkit_grants import validate_grants
        portable["toolkit_grants"] = validate_grants(spec.get("toolkit_grants"), surface.capabilities)
        portable["toolkit_observation_receipts"] = spec.get("toolkit_observation_receipts", False)
    if launch_admission is not None:
        portable["launch_admission"] = launch_admission
    if session is not None:
        portable["session"] = session
    if operator_control is not None:
        portable["operator_control"] = operator_control
    if play is not None:
        portable["episode_mode"] = "play"  # only a free-play plan carries the key
        portable["play"] = play
    portable["partition"] = resolve_partition(spec, portable)
    portable["policy_identity"] = bind_policy_identity(portable, root=root)
    plan = _seal_plan(
        spec, portable, episode=episode, source_commit=source_commit,
        source_dirty=source_dirty, root=root,
    )
    return plan


def _expand_defaults(spec: Mapping[str, Any], *, root: Path) -> Mapping[str, Any]:
    """Apply the maintained Native defaults to ``spec``."""

    try:
        spec = expand_native_defaults(spec, root=root)
    except (ValueError, OSError, EnvironmentContextError) as exc:
        raise RunSpecError(str(exc)) from exc
    return spec


def _resolve_episode(spec: Mapping[str, Any]) -> _Episode:
    experiment = _stable_id(spec.get("experiment"), "experiment")
    config = _stable_id(spec.get("config"), "config")
    milestone = _stable_id(spec.get("milestone"), "milestone")
    task = _stable_id(spec.get("task"), "task")
    task_instruction = spec.get("task_instruction")
    if not isinstance(task_instruction, str) or not task_instruction.strip():
        raise RunSpecError("task_instruction must be non-empty")
    seed = spec.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RunSpecError("seed must be a non-negative integer")
    evidence_tier = spec.get("evidence_tier")
    if evidence_tier not in EVIDENCE_TIERS:
        raise RunSpecError(f"unsupported evidence_tier: {evidence_tier}")
    output_workspace = spec.get("output_workspace")
    expected_workspace = (Path("outputs") / experiment).as_posix()
    if output_workspace != expected_workspace:
        raise RunSpecError(f"output_workspace must be {expected_workspace}")
    return _Episode(
        experiment, config, milestone, task, task_instruction, seed,
        evidence_tier, output_workspace,
    )


def _refuse_memory(policy: dict[str, Any]) -> None:
    """Resolve the only Memory selection: ``memory_policy: disabled``.

    The field stays in every plan (baselines pin it); Task Memory is not part
    of this tree, so a binding or management block is refused.
    """

    policy.setdefault("memory_policy", "disabled")
    if policy["memory_policy"] != "disabled":
        raise RunSpecError("policy.memory_policy must be disabled")
    for field in ("memory_binding", "memory_management"):
        if field in policy:
            raise RunSpecError(f"policy.{field} is not supported")


def _resolve_qualification(
    spec: Mapping[str, Any], experiment: str, *, kind: str, root: Path,
) -> dict[str, Any]:
    """Bind a required qualification manifest, tracked or generated, or the runtime backend's exception."""

    qualification = dict(_mapping(spec.get("qualification", {}), "qualification"))
    qualification.setdefault("required", False)
    if not isinstance(qualification["required"], bool):
        raise RunSpecError("qualification.required must be boolean")
    if qualification["required"]:
        raw_manifest = qualification.get("manifest")
        relative_manifest = Path(raw_manifest) if isinstance(raw_manifest, str) else None
        if (
            relative_manifest is not None
            and relative_manifest.parts[:2] == ("outputs", experiment)
        ):
            qualification["manifest"], _manifest_path = _generated_resource(
                root, experiment, raw_manifest, "qualification.manifest"
            )
        else:
            qualification["manifest"] = _resource(
                root, raw_manifest, "qualification.manifest"
            )
    # A labelled exception to seed qualification is the runtime backend's
    # (RoboTwin's expert-unsolved seed, backends/robotwin/qualification_exception.py).
    hooks = backend(kind).qualification_exception
    if hooks is None:
        if qualification.get("exception") is not None:
            raise RunSpecError(f"qualification.exception: runtime.kind {kind} has no qualification exception")
        return qualification
    try:
        exception = hooks.resolve_exception(
            qualification, lambda value, label: _generated_resource(root, experiment, value, label)[0])
    except hooks.QualificationExceptionError as exc:
        raise RunSpecError(str(exc)) from exc
    if exception is not None:
        qualification["exception"] = exception
    return qualification


def _bind_launch_admission(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize the RunSpec's launch admission; a launch qualification gate is not bound."""

    launch_admission = _launch_admission(spec.get("launch_admission"))
    if launch_admission is not None and "gate" in launch_admission:
        raise RunSpecError(
            "launch qualification gates require a source-registered cell"
        )
    return launch_admission


def _seal_plan(
    spec: Mapping[str, Any], portable: dict[str, Any], *, episode: _Episode,
    source_commit: str, source_dirty: bool, root: Path,
) -> dict[str, Any]:
    """Add the experiment identity, then the plan hash, id and confirmation."""

    try:
        identity = bind_identity(spec, root=root)
    except ValueError as exc:
        raise RunSpecError(str(exc)) from exc
    if identity is not None:
        portable["experiment_identity"] = identity
    plan_hash = json_sha256(portable)
    return {
        **portable,
        "plan_sha256": plan_hash,
        "plan_id": f"{episode.experiment}-{episode.milestone.lower()}-{plan_hash[:12]}",
        "source": {"commit": source_commit, "dirty": source_dirty},
        "confirmation_code": plan_hash[-6:],
    }
