"""Observation and tool-result projections every ``native_api`` policy session shares.

``PolicyDomainMixin`` holds methods only, no state of its own. The Pi core's
``PiPlanner`` (``ora.agent.pi.policy_session``) binds them as its own domain
helpers. A host provides
``tools`` (with its ``surface`` and ``control_root``), ``run_root``,
``supports_images``, ``initial_observation``, ``pending_receipt``,
``pending_receipt_body``, ``last_actual_snapshot`` and the
``last_action_id``/``last_action_public_name`` pair, and
``_actual_snapshot`` (``actual_snapshot.NativeActualSnapshotMixin``).

The methods keep the exact bodies they had in the Python Native session ORA
removed (``NativePlanner``), so what the model sees is unchanged: the legacy and v3 rewrite/compact observation
messages, the minimal action effect of a receipt, the prepared-grasp skill
observation blocks, the declared delivery-trim rules, the turn instruction and
the strict causal projection of a tool result.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.code_mode import answers as code_answers
from ora.agent.context import causal_provider_value, episode_index_keyframe
from ora.agent.evidence.image_binding import (
    PostActionObservationError, resolve_post_action_images,
)
from ora.agent.context.delivery.trim import declared_rules
from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.canonical import image_data_url
from ora.contracts import (
    POSE_ORIENTATION_TOLERANCE_RAD,
    POSE_POSITION_TOLERANCE_M,
)

from .constants import NATIVE_CAUSAL_SLOTS
from .receipt_projection import (
    ReceiptProjectionContext,
    resolve_receipt_effect_projector,
    public_motion_effect_envelope,
    public_motion_failure_code,
)
from .stage_guard import native_turn_instruction
from ora.agent.context.observation.current_visual import visual_feedback
from ora.agent.context.observation.skill_views import materialize_skill_observations, native_blocks
from ora.agent.composition.surface import tool_names


def _declared_result_properties(schema: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect the declared top-level result fields across schema branches."""

    found: set[str] = set()
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        found.update(str(key) for key in properties)
    for keyword in ("oneOf", "anyOf", "allOf"):
        branches = schema.get(keyword)
        if not isinstance(branches, list):
            continue
        for branch in branches:
            if isinstance(branch, Mapping):
                found.update(_declared_result_properties(branch))
    return tuple(sorted(found))


class PolicyDomainMixin:
    """Observation messages, action effects and tool-result projections of one session."""

    @staticmethod
    def _observation_step(context: Mapping[str, Any]) -> int:
        state = context.get("policy_state")
        if not isinstance(state, Mapping) or not isinstance(state.get("step"), int):
            raise PolicySessionError("native context lacks a bound policy-state step")
        return int(state["step"])

    @staticmethod
    def _post_action_view_feedback(current: Mapping[str, Any]) -> dict[str, Any]:
        full = current.get("current_visual_observation")
        feedback = visual_feedback(full) if isinstance(full, Mapping) else {}
        record = current.get("post_action_observation")
        if not isinstance(record, Mapping):
            return feedback
        return {**feedback, "current_views": {
            "mode": record["mode"], "selected_arms": record["selected_arms"],
            "status": record["status"], "missing_wrist_views": record["missing_wrist_views"],
            "views": [image["camera"] for image in record["images"]],
            "fresh_for_this_action": record.get("action_id") is not None,
        }}

    @staticmethod
    def _receipt_record(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
        nested = receipt.get("receipt")
        return nested if isinstance(nested, Mapping) else receipt

    @staticmethod
    def _pose_residual(
        target: Sequence[Any], actual: Sequence[Any],
    ) -> tuple[float, float]:
        translation = math.sqrt(sum(
            (float(target[index]) - float(actual[index])) ** 2
            for index in range(3)
        ))
        dot = abs(sum(
            float(target[index]) * float(actual[index]) for index in range(3, 7)
        ))
        orientation = 2.0 * math.acos(min(1.0, max(0.0, dot)))
        return translation, orientation

    def _minimal_action_effect(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        record = self._receipt_record(receipt)
        raw_action = record.get("action")
        if isinstance(raw_action, Mapping):
            action_name = raw_action.get("action")
            arguments = raw_action.get("arguments")
        else:
            action_name = raw_action
            arguments = record.get("arguments")
        arguments = arguments if isinstance(arguments, Mapping) else {}
        projector = resolve_receipt_effect_projector(record.get("schema"))
        if projector is not None:
            # Environment-specific receipt contracts project through the
            # schema-keyed registry; the shared session keeps no per-backend
            # branch (cross-embodiment parity boundary).
            surface = getattr(getattr(self, "tools", None), "surface", None)
            stated = getattr(getattr(self, "gateway", None), "stated_prose", None)
            return projector(record, ReceiptProjectionContext(
                last_action_id=getattr(self, "last_action_id", None),
                last_action_public_name=getattr(
                    self, "last_action_public_name", None,
                ),
                surface_supports=(
                    surface.supports if surface is not None else None
                ),
                stated_prose=stated() if callable(stated) else None,
            ))
        if action_name in {"move_pose", "move_free_space", "servo_delta", "recover_motion"}:
            target = record.get("commanded_pose_xyz_xyzw")
            actual = record.get("reached_pose_xyz_xyzw")
            if not (
                isinstance(target, list) and len(target) == 7
                and isinstance(actual, list) and len(actual) == 7
            ):
                motion_status = record.get("motion_status")
                feedback = record.get("feedback")
                reason = (
                    feedback.get("planning_failure_reason")
                    if isinstance(feedback, Mapping) else None
                )
                if motion_status == "refused" and isinstance(reason, str) and reason:
                    return causal_provider_value(public_motion_effect_envelope(
                        action=str(action_name), status="plan_rejected", executor=None,
                        motion_strategy=record.get("motion_strategy"),
                        failure_stage="planner",
                        failure_code=public_motion_failure_code("planner", reason),
                        round_terminated=record.get("terminal_after_action"),
                    ))
                return {"action": str(action_name), "status": "world_state_unknown"}
            translation, orientation = self._pose_residual(target, actual)
            return {
                "action": str(action_name),
                "status": (
                    "executed"
                    if translation <= POSE_POSITION_TOLERANCE_M
                    and orientation <= POSE_ORIENTATION_TOLERANCE_RAD
                    else "not_reached"
                ),
                "target_pose_xyz_xyzw": [round(float(value), 6) for value in target],
                "actual_pose_xyz_xyzw": [round(float(value), 6) for value in actual],
                "position_error_m": round(translation, 6),
                "orientation_error_rad": round(orientation, 6),
            }
        if action_name == "set_gripper":
            gripper = record.get("gripper_evidence")
            before = gripper.get("measured_before") if isinstance(gripper, Mapping) else None
            after = gripper.get("measured_after") if isinstance(gripper, Mapping) else None
            target = arguments.get("gripper")
            label = "close_gripper" if target == 0.0 else "open_gripper"
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in (before, after)
            ):
                return {"action": label, "status": "world_state_unknown"}
            return {
                "action": label,
                "status": "executed",
                "before_aperture_mm": round(float(before) * 80.0, 3),
                "after_aperture_mm": round(float(after) * 80.0, 3),
            }
        return {"action": "unknown", "status": "world_state_unknown"}

    def _skill_observation_blocks(self) -> list[dict[str, Any]]:
        surface = getattr(self.tools, "surface", None)
        if not getattr(self, "toolkit_observation_receipts", True):
            return []
        if surface is None or not surface.supports("prepared_grasp_execution_v2"):
            return []
        images = materialize_skill_observations(
            receipt=self.pending_receipt_body, run_root=self.run_root,
            workspace=self.run_root / "native",
            camera_names=surface.view_config["automatic_actual_views"],
        )
        return native_blocks(images, run_root=self.run_root)

    def _delivery_trim_rules(self) -> frozenset[str]:
        """The trim rules the bound surface declares (``observation_delivery_trim_v2``)."""

        surface = getattr(getattr(self, "tools", None), "surface", None)
        return declared_rules(getattr(surface, "capabilities", ()) or ())

    def _delivery_trim(self) -> bool:
        """Whether the bound surface declares a delivery-trim generation."""

        return bool(self._delivery_trim_rules())

    def _rewrite_causal_observation_message(
        self, *, context: Mapping[str, Any], terminal_lifecycle: bool,
    ) -> dict[str, Any]:
        """Preserve the original v3 rewrite/compact projection byte-for-byte."""

        current = self._actual_snapshot(context)
        show_receipts = getattr(self, "toolkit_observation_receipts", True)
        receipt = self.pending_receipt_body if show_receipts else None
        before = self.last_actual_snapshot if receipt is not None else None
        if receipt is not None and before is None:
            raise PolicySessionError("mutation receipt lacks its pre-action actual snapshot")
        effect = self._minimal_action_effect(receipt) if receipt is not None else None
        current_actual = self._post_action_view_feedback(current) if show_receipts else {}
        if "public_robot_facts" in current:
            current_actual["public_robot_facts"] = copy.deepcopy(current["public_robot_facts"])
        elif "end_effectors" in current:
            current_actual["end_effectors"] = copy.deepcopy(
                current["end_effectors"]
            )
        else:
            current_actual.update({
                "eef_pose_xyz_xyzw": current["eef_pose_xyz_xyzw"],
                "aperture_mm": current["aperture_mm"],
            })
        if getattr(
            getattr(self.tools, "surface", None), "view_config", None,
        ) is not None:
            current_actual["view_dimensions"] = {
                item["camera"]: {
                    "width": item["width"], "height": item["height"],
                }
                for item in current["images"]
            }
        envelope = {
            "current_actual": current_actual,
            "last_action_effect": effect,
            "comparison": (
                {
                    "available": True,
                    "instruction": (
                        "Compare BEFORE ACTUAL with AFTER ACTUAL / CURRENT before "
                        "choosing the next tool."
                    ),
                }
                if before is not None else {
                    "available": False,
                    "instruction": "Use CURRENT ACTUAL evidence to choose the next tool.",
                }
            ),
            "remaining_budget": {
                key: context.get("remaining_budget", {}).get(key)
                for key in (
                    "robot_decisions", "model_calls", "wall_time_s",
                    "native_actions_limit", "native_actions_remaining",
                )
                if key in context.get("remaining_budget", {})
            },
            "terminal_lifecycle": terminal_lifecycle,
        }
        if not show_receipts:
            envelope.pop("last_action_effect")
            envelope.pop("comparison")
        round_notice = context.get("runtime_notice")
        if (
            isinstance(round_notice, Mapping)
            and round_notice.get("kind") == "session_round_start"
        ):
            envelope["session_round"] = {
                key: copy.deepcopy(round_notice[key])
                for key in (
                    "kind", "round", "total_rounds", "budget_scope", "instruction",
                )
                if key in round_notice
            }
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": json.dumps(envelope, ensure_ascii=False, sort_keys=True),
        }]
        current_images = {
            item["camera"]: item["path"] for item in current["images"]
        }
        camera_order = [item["camera"] for item in current["images"]]
        if before is not None:
            before_images = {
                item["camera"]: item["path"] for item in before["images"]
            }
            if set(before_images) != set(current_images) and not (
                current.get("post_action_observation") or before.get("post_action_observation")
            ):
                raise PolicySessionError("causal actual camera set changed across mutation")
            for camera in camera_order:
                if camera in before_images:
                    content.extend([
                        {"type": "text", "text": f"BEFORE ACTUAL · {camera}"},
                        {"type": "image_url", "image_url": {
                            "url": image_data_url(before_images[camera]),
                        }},
                    ])
                content.extend([
                    {"type": "text", "text": f"AFTER ACTUAL / CURRENT · {camera}"},
                    {"type": "image_url", "image_url": {
                        "url": image_data_url(current_images[camera]),
                    }},
                ])
        else:
            for camera in camera_order:
                content.extend([
                    {"type": "text", "text": f"CURRENT ACTUAL · {camera}"},
                    {"type": "image_url", "image_url": {
                        "url": image_data_url(current_images[camera]),
                    }},
                ])
        content.extend(self._skill_observation_blocks())
        message = {
            "role": "user",
            "content": content,
            "_observation": {
                "step": current["step"],
                "state_ref": dict(context["policy_state"]),
                "observation_id": self._current_observation_id(),
                "keyframe": True,
                "receipt": self.pending_receipt if show_receipts else None,
            },
            "_full_current_visual": "current_visual_observation" in current,
            "_causal_observation": {
                "schema": "ora.causal-observation.v1",
                "step": current["step"],
                "paired": before is not None,
                "cameras": camera_order,
                **({"post_action_observation": current["post_action_observation"]}
                   if show_receipts and "post_action_observation" in current else {}),
            },
        }
        self.last_actual_snapshot = current
        return message

    def _turn_instruction(self) -> str:
        return native_turn_instruction(self.tools.surface)

    def _observation_message(
        self, *, task_instruction: str, context: Mapping[str, Any], image_path: Path,
        terminal_lifecycle: bool = False,
    ) -> dict[str, Any]:
        if self.context_composition in NATIVE_CAUSAL_SLOTS:
            return self._causal_observation_message(
                context=context, terminal_lifecycle=terminal_lifecycle,
            )
        step = self._observation_step(context)
        state = context["policy_state"]
        previous_execution = context.get("previous_execution")
        previous_receipt = (
            previous_execution.get("receipt")
            if isinstance(previous_execution, Mapping) else None
        )
        previous_action = (
            previous_receipt.get("action")
            if isinstance(previous_receipt, Mapping) else None
        )
        keyframe = episode_index_keyframe(
            step=step, previous_action=previous_action,
        )
        text = {
            "task_instruction": task_instruction,
            "current_observation": context,
            "instruction": self._turn_instruction(),
        }
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": json.dumps(text, ensure_ascii=False, sort_keys=True),
        }]
        try:
            post_action_images = resolve_post_action_images(context, root=self.run_root, path_key="run_path")
        except PostActionObservationError as exc:
            raise PolicySessionError(str(exc)) from exc
        if post_action_images:
            if not self.supports_images:
                raise PolicySessionError("current post-action views require image support")
            for item in post_action_images:
                content.extend([
                    {"type": "text", "text": f"ACTION RESULT ACTUAL · {item['camera']}"},
                    {"type": "image_url", "image_url": {"url": image_data_url(item["path"])}},
                ])
        elif self.supports_images and self.initial_observation == "automatic":
            if not image_path.is_file():
                raise PolicySessionError(f"native observation image is unavailable: {image_path}")
            content.append({"type": "image_url", "image_url": {"url": image_data_url(image_path)}})
        message = {
            "role": "user",
            "content": content,
            "_observation": {
                "step": step,
                "state_ref": dict(state),
                "observation_id": self._current_observation_id(),
                "keyframe": keyframe,
                "receipt": self.pending_receipt,
            },
        }
        return message

    def _current_observation_id(self) -> str | None:
        path = self.tools.control_root / "current-observation.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value.get("observation_id") if isinstance(value, Mapping) else None

    def _causal_tool_result(
        self, name: str, result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Strict model projection; canonical result remains in the artifact store."""

        error = result.get("error")
        if isinstance(error, Mapping):
            # Public recovery fields apply to every active consumer. The shared
            # filter still removes private state and host provenance recursively.
            visible_error = {
                key: copy.deepcopy(error[key])
                for key in (
                    "code", "message", "retryable", "retry_requires_changed_arguments",
                    "infrastructure", "field", "received", "constraints", "repetition",
                )
                if error.get(key) is not None
            }
            return {"error": causal_provider_value(visible_error)}
        # result_schemas is keyed by what this surface advertises. A miss here
        # does not raise -- it falls through to the unprojected result -- so the
        # lookup resolves the spelling rather than depending on the caller's.

        schemas = self.tools.surface.result_schemas
        result_schema = schemas.get(tool_names.as_advertised(schemas, name))
        if result.get("dry_run") is True:  # one record for every moving tool, which no tool's result schema describes
            # Each arm's predicted pose with its world axes, as a script reads it (RO-27; ``code_mode/answers.py``).
            return code_answers.predicted_axes(causal_provider_value(dict(result)),
                                               getattr(self.tools.surface, "view_config", None))
        if isinstance(result_schema, Mapping):
            properties = _declared_result_properties(result_schema)
            if not properties:
                raise PolicySessionError(
                    f"public result schema for {name} declares no result properties"
                )
            # ToolRuntime already validates the complete public result against
            # this schema. Native causal projection mechanically follows the
            # union of its declared branches, then applies the shared causal
            # provenance filter. This supports new surfaces without adding a
            # second tool-name switch while keeping audit IDs out of prompts.
            visible = causal_provider_value(
                {key: result[key] for key in properties if key in result},
                declared=frozenset(properties),
            )
            return visible
        if name in {
            "move_pose", "move_free_space", "servo_delta", "set_gripper",
            "close_gripper", "open_gripper", "recover_motion",
        }:
            return {
                key: copy.deepcopy(result[key])
                for key in ("status", "action") if key in result
            }
        if name == "check_task":
            if result.get("success") is False:
                return {
                    "status": "unavailable",
                    "reason": result.get("reason"),
                    "retryable": bool(result.get("retryable", False)),
                }
            return {"task_success": bool(result.get("task_success", False))}
        if name == "report_issue":
            return {"status": "recorded"}
        if name == "finish_episode":
            return {"outcome": result.get("outcome")}
        return {"status": str(result.get("status", "returned"))}
