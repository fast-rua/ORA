"""The ``RobotToolRuntime`` base: one ToolRuntime per policy session.

It owns the lifecycle, the per-round control files and their isolation
proof, ``invoke``'s dispatch of the ``robot_runtime`` surfaces' five tools with
one terminal, state-changing submission, and the observation and camera
selection every tool family shares. The toolkit serves the five tools
(``dispatch.install``); ``RobotToolsMixin`` (``robot_tools.py``) keeps
what it reads of this host. ``ManipulationToolRuntime`` (``core.py``) extends this class.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ora.agent.composition import (
    ResolvedToolSurface,
    resolve_tool_surface,
    validate_schema_instance,
)
from ora.agent.evidence.policy_state import PolicyStateError, load_policy_state
from ora.agent.toolkit.host.shared.artifact_io import _append, _hash
from ora.agent.error_classes import ToolError
from . import mutation_compensation
from .robot_tools import RobotToolsMixin


ROUND_START_ISOLATION_SCHEMA = "ora.round-start-isolation.v1"
#: Control-root files that hold exactly one round of policy-visible state.
#: ``RobotToolRuntime.reset_round`` archives every one of them and
#: ``round_start_isolation`` proves none of them survived; a surface that adds
#: such a file extends ``per_round_control_files`` rather than either list.
PER_ROUND_CONTROL_FILES = (
    "current-observation.json",
    "submitted-action.json",
    "policy-task-check.json",
    "policy-finish.json",
)


def manual_sections(path: Path) -> tuple[str, ...]:
    """Return the exact ordered level-two headings from one bound manual."""

    text = path.read_text(encoding="utf-8")
    sections = tuple(
        line[3:].strip() for line in text.splitlines() if line.startswith("## ")
    )
    if not sections or any(not section for section in sections):
        raise ToolError("manual must contain non-empty level-two sections")
    if len(sections) != len(set(sections)):
        raise ToolError("manual level-two sections must be unique")
    return sections


def tool_definitions(
    allowed_tools: Sequence[str] | None = None, *, manual_path: Path | None = None,
    surface: ResolvedToolSurface | None = None,
    profile_id: str = "robot_runtime_v2",
    component_overrides: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the advertised tool surface for one concrete Runtime manual."""

    resolved = surface or resolve_tool_surface(
        profile_id, component_overrides=component_overrides,
    )
    base_definitions = resolved.definitions
    selected = (
        {item["name"] for item in base_definitions}
        if allowed_tools is None else set(allowed_tools)
    )
    definitions = [
        copy.deepcopy(item) for item in base_definitions if item["name"] in selected
    ]
    if manual_path is not None and any(
        definition["name"] == "robot.read_manual" for definition in definitions
    ):
        sections = list(manual_sections(manual_path))
        for definition in definitions:
            if definition["name"] == "robot.read_manual":
                definition["inputSchema"]["properties"]["section"]["enum"] = sections
                break
    return definitions


@dataclass
class RobotToolRuntime(RobotToolsMixin):
    control_root: Path
    manual_path: Path
    event_path: Path
    workspace_bounds_m: Mapping[str, list[float]]
    allowed_tools: Sequence[str] | None = None
    state_root: Path | None = None
    tool_surface_profile: str = "robot_runtime_v2"
    tool_surface_overrides: Mapping[str, Any] | None = None
    toolkit_grants: Mapping[str, Any] | None = None
    toolkit_receipt_delivery: str | None = None
    controlled_end_effector: str | None = None
    motion_probe_callback: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    align_plan_callback: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    #: The run's backend answers ``kinematics.plan_path`` (the shared motion planner).
    motion_planner_available: bool = False
    #: The plan's motion profile as the tool context carries it (``motion.profile_entry``), or None: the toolkit's
    #: built-in fast and jaw forms (profile v1), as every run before profiles.
    motion_profile: Mapping[str, Any] | None = None
    #: The plan's model-facing latency switches that are off (``orchestrator/spec/latency.py``), or None when all are on;
    #: every tool call's configuration carries them as ``latency_switches`` (``dispatch.py``).
    latency_switches: Mapping[str, bool] | None = None

    def __post_init__(self) -> None:
        self.control_root = self.control_root.resolve()
        self.manual_path = self.manual_path.resolve()
        self.event_path = self.event_path.resolve()
        self.state_root = None if self.state_root is None else self.state_root.resolve()
        if self.controlled_end_effector is not None and not self.controlled_end_effector:
            raise ToolError("controlled_end_effector must be a non-empty name")
        self.surface = resolve_tool_surface(
            self.tool_surface_profile,
            component_overrides=self.tool_surface_overrides,
            toolkit_grants=self.toolkit_grants,
            toolkit_receipt_delivery=self.toolkit_receipt_delivery,
        )
        from ora.agent.composition.surface.public_parameters import bind_public_workspace
        self.surface = bind_public_workspace(self.surface, self.workspace_bounds_m)
        # The manual is read only by robot.read_manual: a surface that advertises it needs the file,
        # any other surface runs without one (a toolkit surface, for one).
        reads_manual = any(
            item["name"] == "robot.read_manual" for item in self.surface.definitions
        )
        if reads_manual and not self.manual_path.is_file():
            raise ToolError(f"manual is unavailable: {self.manual_path}")
        self._manual_sections = manual_sections(self.manual_path) if reads_manual else ()
        known = {item["name"] for item in self.surface.definitions}
        allowed = known if self.allowed_tools is None else set(self.allowed_tools)
        required_mutations = set(self.surface.mutation_tools)
        if not allowed.issubset(known) or not required_mutations.issubset(allowed):
            raise ToolError(
                "allowed_tools must be known and include the surface mutation tools"
            )
        self.allowed_tools = tuple(
            item["name"] for item in self.surface.definitions if item["name"] in allowed
        )

    def definitions(self) -> list[dict[str, Any]]:
        return tool_definitions(
            self.allowed_tools, manual_path=self.manual_path, surface=self.surface,
        )

    def per_round_control_files(self) -> tuple[str, ...]:
        """Control-root files that hold exactly one round's policy state.

        ``reset_round`` archives every name this returns and
        ``round_start_isolation`` proves none of them survived into the new
        round, so a surface that adds a policy-visible control file declares
        it once here instead of in two places that can drift apart. The
        canonical example is ``policy-task-check.json``: ``check_task`` reads
        whatever that file says, so a copy left behind by a round reset would
        answer the new round with the previous round's official outcome.
        """

        return PER_ROUND_CONTROL_FILES

    def round_registry_names(self) -> tuple[str, ...]:
        """In-memory attributes that must be empty when a round starts."""

        return ()

    def residual_round_registries(self) -> list[str]:
        """Which of ``round_registry_names`` still hold a value."""

        return sorted(
            name for name in self.round_registry_names() if getattr(self, name, None)
        )

    def round_start_isolation(self) -> dict[str, Any]:
        """Measure whether any policy-visible channel survived a round reset.

        Read-only: it inspects the Control root and this runtime's own
        registries and reports what it found. The Supervisor calls it before
        the new round's first observation is published and refuses to let the
        policy act when it does not pass.
        """

        residual_files = sorted(
            name for name in self.per_round_control_files()
            if (self.control_root / name).exists()
        )
        residual_registries = self.residual_round_registries()
        failures = []
        if residual_files:
            failures.append(
                "control files carrying a previous round: "
                + ", ".join(residual_files)
            )
        if residual_registries:
            failures.append(
                "tool registries carrying a previous round: "
                + ", ".join(residual_registries)
            )
        return {
            "schema": ROUND_START_ISOLATION_SCHEMA,
            "passed": not failures,
            "audited_control_files": list(self.per_round_control_files()),
            "audited_runtime_registries": list(self.round_registry_names()),
            "residual_control_files": residual_files,
            "residual_runtime_registries": residual_registries,
            "failures": failures,
        }

    def reset_round(self, next_round_index: int) -> None:
        """Archive ephemeral control state before a Supervisor-owned world reset."""

        if (
            isinstance(next_round_index, bool)
            or not isinstance(next_round_index, int)
            or next_round_index <= 0
        ):
            raise ToolError("next_round_index must be a positive integer")
        archive = self.control_root / "rounds" / f"round-{next_round_index - 1:04d}"
        archive.mkdir(parents=True, exist_ok=True)
        for name in self.per_round_control_files():
            source = self.control_root / name
            if source.is_file():
                target = archive / name
                if target.exists():
                    raise ToolError(f"refusing to overwrite round control state: {target}")
                os.replace(source, target)

    def classify_call_effect(
        self, name: str, arguments: Mapping[str, Any]
    ) -> str:
        """Classify calls for surfaces that advertise argument-aware effects."""

        del arguments
        if name in self.surface.lifecycle_tools:
            return "lifecycle"
        if name in self.surface.mutation_tools:
            return "mutation"
        return "read_only"

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self.surface.supports("toolkit_only_v1") and not getattr(
            self, "_toolkit_tools_enabled", False,
        ):
            raise ToolError(
                "toolkit-only surface requires the toolkit runtime factory",
                code="tool_unavailable", retryable=False,
            )
        started = time.time()
        started_monotonic_ns = time.monotonic_ns()

        taken = mutation_compensation.snapshot(self, name, arguments)
        try:
            if name not in self.allowed_tools:
                raise ToolError(
                    f"tool is disabled by this experiment config: {name}",
                    code="tool_unavailable", retryable=False, field="tool",
                    received=name, constraints={"allowed": list(self.allowed_tools)},
                )
            if name == "observation.dump_state":
                result = self._dump_state(arguments)
            elif name == "vision.inspect":
                result = self._inspect(arguments)
            elif name == "geometry.query":
                result = self._geometry(arguments)
            elif name == "robot.read_manual":
                result = self._manual(arguments)
            elif name == "robot.submit_action":
                result = self._submit(arguments)
            else:
                raise ToolError(
                    f"unknown tool: {name}", code="unknown_tool", retryable=False,
                    field="tool", received=name,
                    constraints={"allowed": list(self.allowed_tools)},
                )
            public_result = {
                key: value for key, value in result.items()
                if key not in {"image_base64", "_runtime"}
            }
            result_schema = self.surface.result_schemas.get(name)
            if result_schema is not None:
                validate_schema_instance(public_result, result_schema)
        except Exception as exc:
            # Remove only what this call persisted; a pending action survives.
            mutation_compensation.restore(self, taken)
            finished = time.time()
            duration_s = round(
                max(0, time.monotonic_ns() - started_monotonic_ns) / 1e9, 6
            )
            structured = (
                exc.as_dict() if isinstance(exc, ToolError)
                else ToolError(str(exc), code="internal_tool_error", retryable=False).as_dict()
            )
            _append(self.event_path, {
                "type": "tool_error", "tool": name, "arguments": dict(arguments),
                "error": structured, "time": finished,
                "started": started, "finished": finished,
                "duration_s": duration_s, "duration_clock": "monotonic",
            })
            raise
        finished = time.time()
        duration_s = round(
            max(0, time.monotonic_ns() - started_monotonic_ns) / 1e9, 6
        )
        try:
            _append(self.event_path, {
                "type": "tool_result", "tool": name, "arguments": dict(arguments),
                "result": public_result,
                "started": started, "finished": finished,
                "duration_s": duration_s, "duration_clock": "monotonic",
            })
        except OSError:
            # The action file is only an unexecuted proposal at this boundary.
            # Never let a mutation survive if its pre-dispatch evidence failed.
            mutation_compensation.restore(self, taken)
            raise
        return result

    def _current(self) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        pointer = self.control_root / "current-observation.json"
        if not pointer.is_file():
            raise ToolError(
                "no current observation", code="environment_unavailable",
                retryable=False,
            )
        control = json.loads(pointer.read_text(encoding="utf-8"))
        if control.get("open") is not True:
            # Its own code: stale_observation elsewhere refuses a value bound to
            # an earlier observation and is retryable with a changed value.
            raise ToolError(
                "the current observation is closed: an action was submitted on it "
                "or its round ended, and no fresh observation is published yet",
                code="action_pending_fresh_observation", retryable=False,
            )
        root = Path(control["source_root"]).resolve()
        if not root.is_dir() or not root.is_relative_to(self.control_root.parent.resolve()):
            raise ToolError("observation root escapes the job boundary", code="artifact_integrity_fault")
        manifest = root / "observation.json"
        if _hash(manifest) != control.get("observation_sha256"):
            raise ToolError("observation manifest hash drift", code="artifact_hash_mismatch")
        observation = json.loads(manifest.read_text(encoding="utf-8"))
        return control, root, observation

    def _select_observation(
        self, step: Any,
    ) -> tuple[int | None, dict[str, Any] | None, Path, dict[str, Any]]:
        if step is None:
            control, root, observation = self._current()
            reference = control.get("policy_state", {})
            selected = reference.get("step") if isinstance(reference, Mapping) else None
            if self.state_root is None or not isinstance(selected, int):
                raise ToolError(
                    "current policy-state evidence is unavailable",
                    code="artifact_not_found", retryable=False,
                )
            try:
                state = load_policy_state(self.state_root.parent, selected)
            except PolicyStateError as exc:
                raise ToolError(
                    str(exc), code="artifact_not_found", retryable=False,
                ) from exc
            if (
                state.get("state_sha256") != reference.get("state_sha256")
                or state.get("observation", {}).get("id") != control.get("observation_id")
            ):
                raise ToolError(
                    "current policy-state reference drift",
                    code="artifact_hash_mismatch", retryable=False,
                )
            return selected, state, root, observation
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ToolError(
                "step must be a non-negative integer or null", field="step",
                received=step, constraints={"minimum": 0, "nullable": True},
            )
        if self.state_root is None:
            raise ToolError(
                "policy-state evidence is unavailable", code="artifact_not_found",
                retryable=False,
            )
        try:
            state = load_policy_state(self.state_root.parent, step)
        except PolicyStateError as exc:
            from ora.agent.toolkit.host.shared.saved_steps import saved_step_refusal
            raise saved_step_refusal(self, step, exc) from exc
        from ora.agent.toolkit.host.visibility import check_selection
        check_selection(self, state)
        manifest = state.get("observation", {}).get("manifest", {})
        relative = manifest.get("path") if isinstance(manifest, Mapping) else None
        if not isinstance(relative, str):
            raise ToolError("saved observation reference is invalid",
                            code="artifact_integrity_fault", retryable=False)
        run_root = self.state_root.parent.resolve()
        manifest_path = (run_root / relative).resolve()
        if not manifest_path.is_file() or not manifest_path.is_relative_to(run_root):
            raise ToolError("saved observation is unavailable",
                            code="artifact_integrity_fault", retryable=False)
        if _hash(manifest_path) != manifest.get("sha256"):
            raise ToolError("saved observation hash drift", code="artifact_hash_mismatch", retryable=False)
        observation = json.loads(manifest_path.read_text(encoding="utf-8"))
        return step, state, manifest_path.parent, observation

    @staticmethod
    def _camera(root: Path, observation: Mapping[str, Any], camera: Any) -> tuple[dict[str, Any], Path, Path, Path]:
        if not isinstance(camera, str) or camera not in observation.get("cameras", {}):
            raise ToolError("camera is not policy-visible")
        record = observation["cameras"][camera]
        files = record.get("files", {})
        roles = ("rgb_lossless_png", "depth_numpy_m_f32", "calibration")
        paths = []
        for role in roles:
            ref = files.get(role)
            if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
                raise ToolError(f"camera lacks {role}", code="artifact_integrity_fault")
            path = (root / ref["path"]).resolve()
            if not path.is_file() or not path.is_relative_to(root):
                raise ToolError("camera artifact escapes observation root", code="artifact_integrity_fault")
            if _hash(path) != ref.get("sha256"):
                raise ToolError("camera artifact hash drift", code="artifact_hash_mismatch")
            paths.append(path)
        return record, paths[0], paths[1], paths[2]
