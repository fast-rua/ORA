"""The default manipulation tool runtime over immutable Harness observations.

``ManipulationToolRuntime`` keeps its dataclass fields, construction,
tool dispatch (``invoke``) and per-round reset here. The toolkit serves
every tool (``dispatch.install``); the mixin modules beside this
one (``observe``, ``grasp``, ``skills``, ``motion_submit`` and
``lifecycle``) keep the host state it reads, and ``common`` holds what
several of them share. It serves the
``native_atomic_causal`` runtime id that ToolSurfaces and RunPlans name.
"""

from __future__ import annotations

from ora.agent.toolkit.host.backend_selection import PER_ROUND_RUNTIME_REGISTRIES

import copy
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ora.agent.composition import validate_schema_instance
from ora.agent.composition.surface.tool_surface import ToolSurfaceError
from ora.agent.evidence.policy_state import (  # noqa: F401  read through ``common``'s forwarders
    load_policy_state,
    verify_policy_state,
)
from ora.agent.model_services.binding import service_client_arguments
from ora.agent.toolkit.host.runtime.base import RobotToolRuntime
from ora.agent.error_classes import ToolError
from ora.agent.toolkit.host.shared.artifact_io import _append
from ora.agent.toolkit.host.finish import outcomes as finish_outcomes, task_check as finish_task_check
from ora.agent.toolkit.interface import dry_run
from ora.agent.toolkit.host.runtime import mutation_compensation
from ora.agent.toolkit.host.perception import feedback as perception_feedback
from ora.agent import error_classes as tool_errors
from ora.agent.toolkit.host.grasp import panda_topdown
from ora.agent.toolkit.host.grasp import x5_topdown
from ora.agent.composition.surface import tool_names
from ora.agent.toolkit.host.grasp.compiler_registry import _grasp_hover_offset_m
from ora.agent.toolkit.host.grasp.proposers import (
    AlohaHybridGraspProposer,
    GraspProposalBackend,
    PcaTopDownGraspProposer,
)
from ora.agent.toolkit.host.grasp.contact_graspnet_client import ContactGraspNetApiProposer
from ora.agent.toolkit.host.perception.fallback_segmenter import DepthPromptSegmenter
from ora.agent.toolkit.host.perception.backends import PointSegmentationBackend, TextGroundingBackend
from ora.agent.model_services.clients import Sam3ApiSegmenter


#: The ToolSurface capability whose surfaces check a call's public arguments against the declared input schema before
#: its handler runs (the capability of ``set_gripper``'s close-intent annotation).
ADVISORY_CAPABILITY = "public_rgbd_retention_advisory_v1"

LEGACY_OPENETA_VIEW_CONFIG = {
    "schema": "ora.policy-view-config.v1",
    "rgb_resolution": [512, 512],
    "derived_resolution": [512, 512],
    "automatic_actual_views": ["agentview", "wrist"],
    "default_observe_views": ["agentview", "wrist"],
    "coordinate_convention": "opencv_top_left_xy",
}

MODEL_SERVICE_SET_BINDING_SCHEMA_V2 = (
    "ora.model-service-set-binding.v2"
)


from ora.agent.toolkit.host.runtime.observe import ObserveRuntimeMixin
from ora.agent.toolkit.host.runtime.grasp import GraspRuntimeMixin
from ora.agent.toolkit.host.runtime.skills import SkillRuntimeMixin
from ora.agent.toolkit.host.runtime.motion_submit import MotionSubmissionRuntimeMixin
from ora.agent.toolkit.host.runtime.lifecycle import LifecycleRuntimeMixin


@dataclass
class ManipulationToolRuntime(
    RobotToolRuntime,
    ObserveRuntimeMixin,
    GraspRuntimeMixin,
    SkillRuntimeMixin,
    MotionSubmissionRuntimeMixin,
    LifecycleRuntimeMixin,
):
    """Shared perception/motion dispatcher across versioned compatible surfaces."""

    segmentation_backend: PointSegmentationBackend | None = None
    grounding_backend: TextGroundingBackend | None = None
    grasp_backend: GraspProposalBackend | None = None
    prerequisite_binding: Mapping[str, Any] | None = None
    #: ``context.observation.contact_points.project_policy_public_facts``, or a
    #: stand-in: what observe attaches as ``public_robot_facts`` on a surface
    #: declaring ``current_public_robot_facts_v1``.
    public_facts_projector: Callable[..., Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.surface.runtime_id != "native_atomic_causal":
            raise ToolError(
                "ManipulationToolRuntime requires a compatible perception surface"
            )
        self.artifact_root = self.control_root.parent / "openeta-tool-artifacts"
        self.point_path = self.control_root / "openeta-points.json"
        self.preview_path = self.control_root / "openeta-previews.json"
        self._active_mark_point_batch: tuple[dict[str, Any], ...] | None = None
        self._known_plan_refs: dict[str, dict[str, Any]] = {}
        # Ordered log of submitted mutation names; plan references record their
        # position in it so gripper-only intervals can keep a probed plan alive
        # The environment remains the execution
        # authority and re-validates joints before consuming any plan_ref.
        self._submitted_mutations: list[str] = []
        self.view_config = dict(self.surface.view_config or LEGACY_OPENETA_VIEW_CONFIG)
        # allowed_tools carries whatever this surface advertises, so a literal
        # dotted membership test turns the capability off on a renamed surface
        # without raising anything.
        segmentation_enabled = tool_names.declares(self.allowed_tools, "vision_segment")
        grounding_enabled = tool_names.declares(self.allowed_tools, "vision_ground")
        grasp_plan_enabled = tool_names.declares(self.allowed_tools, "grasp_plan")
        grasp_enabled = grasp_plan_enabled
        prerequisite = (
            dict(self.prerequisite_binding)
            if isinstance(self.prerequisite_binding, Mapping) else {}
        )
        expected_segmentation: Mapping[str, Any] | None = None
        expected_grasp: Mapping[str, Any] | None = None
        binding_scope_verified = False
        if isinstance(self.prerequisite_binding, Mapping) and (
            segmentation_enabled or grounding_enabled or grasp_enabled
        ):
            prerequisite_schema = prerequisite.get("schema")
            if prerequisite_schema == "ora.segmentation-service-binding.v1":
                expected_segmentation = dict(prerequisite)
            elif prerequisite_schema == MODEL_SERVICE_SET_BINDING_SCHEMA_V2:
                # A capability-scoped set binding is valid
                # for any surface that supports every named capability, so a
                # successor profile no longer needs the experiment-side
                # binding rename hack. Service identity (endpoint, process,
                # checkpoint hashes, qualification) stays fully enforced by
                # the service clients.
                services = prerequisite.get("services")
                scope = prerequisite.get("capability_scope")
                if (
                    set(prerequisite) != {
                        "schema", "capability_scope", "services",
                    }
                    or not isinstance(scope, list)
                    or not scope
                    or not all(
                        isinstance(value, str) and value for value in scope
                    )
                    or not isinstance(services, Mapping)
                    or set(services) != {"segmentation", "grasp_proposals"}
                    or not all(
                        isinstance(value, Mapping) for value in services.values()
                    )
                ):
                    raise ValueError(
                        "capability-scoped model-service set binding is invalid"
                    )
                unsupported = [
                    value for value in scope if not self.surface.supports(value)
                ]
                if unsupported:
                    raise ToolError(
                        "model-service binding capability scope is not "
                        "supported by the tool surface: "
                        + ", ".join(sorted(unsupported)),
                        code="tool_surface_capability_mismatch",
                        retryable=False,
                    )
                binding_scope_verified = True
                expected_segmentation = dict(services["segmentation"])
                expected_grasp = dict(services["grasp_proposals"])
            elif prerequisite_schema == "ora.model-service-set-binding.v1":
                services = prerequisite.get("services")
                if (
                    set(prerequisite) != {
                        "schema", "tool_surface", "services",
                    }
                    or prerequisite.get("tool_surface") != self.surface.profile_id
                    or not isinstance(services, Mapping)
                    or set(services) != {"segmentation", "grasp_proposals"}
                    or not all(
                        isinstance(value, Mapping) for value in services.values()
                    )
                ):
                    raise ValueError("model-service set binding is invalid")
                expected_segmentation = dict(services["segmentation"])
                expected_grasp = dict(services["grasp_proposals"])
            else:
                raise ValueError("unsupported perception prerequisite binding")
        for binding in (expected_segmentation, expected_grasp):
            if (
                binding is not None
                and not binding_scope_verified
                and binding.get("tool_surface") != self.surface.profile_id
            ):
                raise ToolError(
                    "model-service prerequisite is bound to a different tool surface"
                )
        if self.segmentation_backend is None:
            self.segmentation_backend = (
                Sam3ApiSegmenter(**service_client_arguments(
                    expected_segmentation, "ORA_SEGMENTATION_URL", "http://127.0.0.1:8114",
                ))
                if segmentation_enabled or grounding_enabled else DepthPromptSegmenter()
            )
        if self.grounding_backend is None:
            if grounding_enabled and callable(getattr(self.segmentation_backend, "ground", None)):
                self.grounding_backend = self.segmentation_backend  # type: ignore[assignment]
            elif grounding_enabled:
                self.grounding_backend = Sam3ApiSegmenter(**service_client_arguments(
                    expected_segmentation, "ORA_SEGMENTATION_URL", "http://127.0.0.1:8114",
                ))
        if self.grasp_backend is None:
            contact_backend = (
                ContactGraspNetApiProposer(**service_client_arguments(
                    expected_grasp, "ORA_GRASP_URL", "http://127.0.0.1:8115",
                ), hover_offset_m=_grasp_hover_offset_m(
                    self.surface, grasp_plan_enabled=grasp_plan_enabled,
                ))
                if grasp_enabled else None
            )
            hybrid = (panda_topdown.hybrid_backend_arguments(self.surface)  # Panda fallback family
                      or x5_topdown.hybrid_backend_arguments(self.surface))  # ALOHA or X5 fused family
            self.grasp_backend = (
                AlohaHybridGraspProposer(contact_backend=contact_backend, **hybrid)
                if contact_backend is not None and hybrid is not None
                else contact_backend
                if contact_backend is not None
                else PcaTopDownGraspProposer()
            )
        service_bindings: dict[str, Any] = {}
        if segmentation_enabled or grounding_enabled:
            readiness = getattr(self.segmentation_backend, "require_ready", None)
            if callable(readiness):
                service_bindings["segmentation"] = readiness()
        if grounding_enabled and self.grounding_backend is not self.segmentation_backend:
            readiness = getattr(self.grounding_backend, "require_ready", None)
            if callable(readiness):
                service_bindings["grounding"] = readiness()
        if grasp_enabled:
            readiness = getattr(self.grasp_backend, "require_ready", None)
            if callable(readiness):
                service_bindings["grasp_proposals"] = readiness()
        if service_bindings:
            _append(self.event_path, {
                "type": "resident_model_services_bound",
                "tool_surface": self.surface.profile_id,
                "services": service_bindings,
                "time": time.time(),
            })

    def definitions(self) -> list[dict[str, Any]]:
        return self.surface.select(self.allowed_tools)

    def per_round_control_files(self) -> tuple[str, ...]:
        """The base control files plus this surface's own per-round artifacts."""

        return (
            *super().per_round_control_files(),
            self.point_path.name,
            self.preview_path.name,
        )

    def round_registry_names(self) -> tuple[str, ...]:
        """In-memory references a Supervisor-owned world reset invalidates.

        Every name below refers to something the reset just destroyed: plans
        the environment has dropped, the mutation ledger those references are
        judged fresh against, and the open mark-point batch. Keeping one would let a stale ``plan_ref``
        pass this runtime's freshness rule and reach a backend that can only
        reject it, and that rejection is raised outside the recoverable-error
        classification, so a leaked reference becomes a Runtime fault rather
        than a retryable receipt.
        """

        return tuple(PER_ROUND_RUNTIME_REGISTRIES)

    def reset_round(self, next_round_index: int) -> None:
        super().reset_round(next_round_index)
        archive = self.control_root / "rounds" / f"round-{next_round_index - 1:04d}"
        for source in (self.point_path, self.preview_path):
            if source.is_file():
                if (archive / source.name).exists():
                    raise ToolError(f"refusing to overwrite round control state: {source.name}")
                os.replace(source, archive / source.name)
        self._active_mark_point_batch = None
        # Every reference below names something the Supervisor-owned world
        # reset just destroyed: plans the environment has dropped and the
        # mutation ledger those references are judged fresh against.
        # Keeping them would let a
        # stale plan_ref pass this runtime's freshness rule and reach a
        # backend that can only reject it, and the rejection is raised
        # outside the recoverable-error classification, so a leaked
        # reference becomes a Runtime fault rather than a retryable receipt.
        self._known_plan_refs.clear()
        self._submitted_mutations.clear()
        # Cleared through the declared roster so a new per-round registry is
        # named once and is provably empty at the next round's first turn.
        for name, empty in PER_ROUND_RUNTIME_REGISTRIES.items():
            setattr(self, name, copy.deepcopy(empty))

    def classify_call_effect(
        self, name: str, arguments: Mapping[str, Any]
    ) -> str:
        """Classify the actual call, not merely its advertised tool name."""

        del arguments
        if name in self.surface.lifecycle_tools:
            return "lifecycle"
        if dry_run.requested():
            return "read_only"
        return "mutation" if name in self.surface.mutation_tools else "read_only"

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        arguments, previewed = dry_run.split(name, arguments)
        if previewed:
            with dry_run.dry():
                return self.invoke(name, arguments)
        if self.surface.supports("toolkit_only_v1") and not getattr(
            self, "_toolkit_tools_enabled", False,
        ):
            raise ToolError(
                "toolkit-only surface requires the toolkit runtime factory",
                code="tool_unavailable", retryable=False,
            )
        started = time.time()
        taken = mutation_compensation.snapshot(self, name, arguments)
        try:
            if name not in self.allowed_tools:
                raise ToolError(
                    f"tool is disabled: {name}", code="tool_unavailable", retryable=False,
                    field="tool", received=name, constraints={"allowed": list(self.allowed_tools)},
                )
            portable_finish = name == "finish_episode" and getattr(self, "_toolkit_tools_enabled", False)
            if name == "finish_episode" and not portable_finish:
                # A latched success first: once check_task confirmed it, no
                # other finish refusal can help.
                finish_task_check.refuse_latched(self, arguments)
                finish_outcomes.admit(self, arguments)
            if self.surface.supports(ADVISORY_CAPABILITY) and not portable_finish:
                self._validate_public_arguments(name, arguments)
            handlers = {
                "observe": lambda values: self._observe(values),
                "mark_point": lambda values: self._mark_point(values),
                "geometry_query": lambda values: self._geometry(values),
                "compute": lambda values: self._restricted_compute(values),
                "vision_segment": lambda values: self._vision_segment(values),
                "vision_ground": lambda values: self._vision_ground(values),
                "grasp_plan": lambda values: self._grasp_plan(values),
                "grasp_execute": lambda values: self._closed_skill("grasp_execute", values),
                "align_plan": lambda values: self._align_plan(values),
                "move_to": lambda values: self._move_to(values),
                "rotate_wrist": lambda values: self._motion_rotate_wrist_v2(values),
                "rotate_pitch": lambda values: self._motion_rotate_pitch_v2(values),
                "preview_move": lambda values: self._atomic_preview_move(values),
                "move_pose": lambda values: self._atomic_move_pose(values),
                "set_gripper": lambda values: self._motion_set_gripper(values),
                "close_gripper": lambda values: self._atomic_close_gripper(values),
                "open_gripper": lambda values: self._atomic_open_gripper(values),
                "recover_pose": lambda values: self._motion_recover_pose_v2(values),
                "return_home": lambda values: self._motion_return_home(values),
                "wait_ticks": lambda values: self._wait_ticks(values),
                "move_base": lambda values: self._move_base(values),
                "report_issue": lambda values: self._report_issue(values),
                "check_task": lambda values: self._check_task(values),
                "finish_episode": lambda values: self._finish(values),
            }
            # A frozen surface advertises the dotted spelling and an active
            # one the canonical spelling; both reach the same handler. Handlers
            # are registered once, under the canonical name, so a tool added
            # later cannot be reachable under one spelling and missing under the
            # other. Each is the host attribute ``dispatch.install``
            # sets, read only when its tool is called.
            handler = handlers.get(tool_names.canonical(name))
            if handler is None and hasattr(self, "_toolkit_handlers"):
                handler = self._toolkit_handlers.get(name)  # a toolset's own tool (dispatch.install)
            if handler is None:
                raise ToolError(f"unknown tool: {name}", code="unknown_tool", retryable=False,
                                field="tool", received=name)
            try:
                result = handler(arguments)
            except dry_run.Proposed as proposed:  # the environment plans this move: it answers for it
                from ora.agent.toolkit.host.action_preview import answer

                result = answer(self, name, proposed)
            perception_feedback.enrich(self, name, arguments, result)
            visible = {
                key: value for key, value in result.items()
                if key not in {"image_base64", "images_base64", "_runtime"}
            }
            if self.surface.result_schemas.get(name) is not None and not dry_run.requested():
                validate_schema_instance(visible, self.surface.result_schemas[name])
        except Exception as exc:
            mutation_compensation.restore(self, taken)
            structured = exc.as_dict() if isinstance(exc, ToolError) else ToolError(
                str(exc), code="internal_tool_error", retryable=False,
            ).as_dict()
            _append(self.event_path, {
                "type": "tool_error", "tool": name, "arguments": dict(arguments),
                "error": structured, "time": time.time(),
            })
            raise
        try:
            _append(self.event_path, {
                "type": "tool_result", "tool": name, "arguments": dict(arguments),
                "result": visible, "started": started, "finished": time.time(),
            })
        except OSError:
            mutation_compensation.restore(self, taken)
            raise
        return result

    def _validate_public_arguments(self, name: str, arguments: Mapping[str, Any]) -> None:
        """Keep schema/arm validation independent of holding admission.

        Callers name a tool canonically. This surface may advertise the dotted
        spelling, and a mismatch here raises StopIteration rather than a
        ToolError, so the lookup compares canonical forms on both sides.
        """
        definition = tool_names.definition_for(self.surface.definitions, name)
        try:
            validate_schema_instance(
                dict(arguments), definition["inputSchema"], root_path="$input",
            )
        except ToolSurfaceError as error:
            raise tool_errors.schema_refusal(error) from error
