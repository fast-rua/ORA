"""The internal read-only motion probe.

The reachability probe the toolkit's queries run through the
environment's read-only bridge. The toolkit serves ``grasp_plan``
(``dispatch.install``).

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ora.contracts.pose import PoseCompilationError, canonicalize_quaternion
from ora.contracts.refusal import EnvironmentRefusal
from ora.contracts.workspace import workspace_state

from ora.agent.error_classes import ToolError, _finite_vector
from ora.agent.toolkit.host.reach.probe_feedback import probe_request_fields


class GraspRuntimeMixin:
    """The internal read-only motion probe."""

    def _motion_probe(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self.motion_probe_callback is None:
            raise ToolError(
                "motion.probe requires the Native environment read-only bridge",
                code="tool_unavailable", retryable=False,
            )
        if set(arguments) != {"queries", "variant_policy"}:
            raise ToolError(
                "motion.probe requires queries and variant_policy",
                code="invalid_arguments",
            )
        queries = arguments.get("queries")
        policy = arguments.get("variant_policy")
        if (
            not isinstance(queries, list) or not 1 <= len(queries) <= 12
            or policy not in {"exact", "parallel_jaw_equivalents"}
        ):
            raise ToolError("motion.probe request is invalid", code="invalid_arguments")
        control, root, observation, _pose, _gripper = self._current_grip_state()
        normalized: list[dict[str, Any]] = []
        query_ids: set[str] = set()
        for query in queries:
            if not isinstance(query, Mapping) or set(query) - {
                "query_id", "arm", "pose",
            }:
                raise ToolError("motion.probe query fields are invalid")
            query_id = query.get("query_id")
            arm = query.get("arm")
            if arm is None and self.surface.supports(
                "single_arm_flat_targets_v1"
            ):
                arm = "right"
                query = {**query, "arm": "right"}
            if (
                not isinstance(query_id, str) or not query_id
                or query_id in query_ids or arm not in {"left", "right"}
                or "pose" not in query
            ):
                raise ToolError("motion.probe query identity is invalid")
            pose = query.get("pose")
            if not isinstance(pose, Mapping) or set(pose) != {
                "position_m", "orientation_xyzw",
            }:
                raise ToolError("motion.probe pose is invalid")
            position = _finite_vector(pose["position_m"], 3, "position_m")
            workspace = workspace_state(position, self.workspace_bounds_m)
            if not workspace["inside"]:
                raise ToolError(
                    "motion.probe target is outside the action workspace; "
                    "select a position within the declared bounds",
                    code="invalid_arguments",
                    field=f"queries.{query_id}.pose.position_m", received=position,
                    constraints={
                        "allowed": dict(self.workspace_bounds_m),
                        "violations": workspace["violations"],
                    },
                )
            try:
                orientation = canonicalize_quaternion(
                    pose["orientation_xyzw"], "orientation_xyzw",
                )
            except PoseCompilationError as exc:
                raise ToolError(
                    str(exc), code="invalid_arguments",
                    field="orientation_xyzw",
                    received=pose.get("orientation_xyzw"),
                    constraints={"norm_min": 0.98, "norm_max": 1.02},
                ) from exc
            normalized.append({
                "query_id": query_id, "arm": arm,
                "pose": {
                    "position_m": position,
                    "orientation_xyzw": orientation,
                },
            })
            query_ids.add(query_id)
        try:
            result = dict(self.motion_probe_callback({
                **probe_request_fields(self.surface),
                "observation_id": control["observation_id"],
                "observation_sha256": control["observation_sha256"],
                "queries": normalized,
                "variant_policy": policy,
            }))
        except EnvironmentRefusal as error:
            raise ToolError(
                str(error), code=error.code, retryable=error.retryable,
                field=error.field, received=error.received,
                constraints={"allowed": error.allowed},
            ) from error
        if result.get("schema") != "ora.motion-probe-result.v1":
            raise ToolError(
                "environment motion probe result schema drifted",
                code="environment_contract", retryable=False,
            )
        query_results = result.get("results")
        if not isinstance(query_results, list) or len(query_results) != len(normalized):
            raise ToolError("motion probe query result count differs",
                            code="environment_contract", retryable=False)
        registrations: dict[str, dict[str, Any]] = {}
        for query, query_result in zip(normalized, query_results):
            if (not isinstance(query_result, Mapping)
                    or query_result.get("query_id") != query["query_id"]
                    or query_result.get("arm") != query["arm"]):
                raise ToolError("motion probe query result identity differs",
                                code="environment_contract", retryable=False)
            variants = query_result.get("variants", [])
            if not isinstance(variants, list) or any(not isinstance(v, Mapping) for v in variants):
                raise ToolError("motion probe variant results are invalid",
                                code="environment_contract", retryable=False)
            handles = [(query_result.get("plan_ref"),
                        query_result.get("selected_variant_id"),
                        query_result.get("selected_target_pose_xyzw"))]
            handles.extend((v.get("plan_ref"), v.get("variant_id"),
                            v.get("target_pose_xyzw")) for v in variants
                           if v.get("reachable") is True)
            for plan_ref, selected_variant, target_pose in handles:
                if plan_ref is None:
                    continue
                if (not isinstance(plan_ref, str) or not plan_ref
                        or not isinstance(selected_variant, str) or not selected_variant
                        or not isinstance(target_pose, list) or len(target_pose) != 7
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in target_pose)):
                    raise ToolError("motion probe plan handle target is invalid",
                                    code="environment_contract", retryable=False)
                binding = {
                    "observation_id": control["observation_id"],
                    "observation_sha256": control["observation_sha256"],
                    "arm": query["arm"],
                    "variant_id": selected_variant,
                    "target_pose_xyzw": list(target_pose),
                    "submitted_mutation_index": len(self._submitted_mutations),
                }
                if plan_ref in registrations and registrations[plan_ref] != binding:
                    raise ToolError("motion probe plan handle has conflicting targets",
                                    code="environment_contract", retryable=False)
                registrations[plan_ref] = binding
        self._known_plan_refs.update(registrations)
        return result
