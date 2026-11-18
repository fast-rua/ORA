"""Static context and observation drafts for the Pi agent core (a ``PiPlanner`` mixin).

Python owns what an observation says about the robot: the envelope built from
the policy-state snapshot and the last receipt, its images as content-addressed
references, and the delivery trims as generic retention hints the TypeScript
core applies. The draft is otherwise the message the Python Native session built
before ORA removed it (``ora.agent.session_core.domain`` keeps its projections).
"""

from __future__ import annotations

import base64
import binascii
import copy
from pathlib import Path
from typing import Any, Mapping

from ora.agent.context.delivery import round_static_facts
from ora.agent.context.delivery.trim_envelope import trim_observation_envelope, trim_view_freshness
from ora.agent.context.delivery.dedup import CAPABILITY as DEDUP_CAPABILITY
from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.memory.markers import LOSSLESS_CONTEXT_RETENTION
from ora.agent.session_core.composition import assemble_native_v3_static_context
from ora.agent.session_core.constants import NATIVE_CAUSAL_SLOTS, NATIVE_SLOTS_V4

from . import hints as retention_hints


def _image_ref_block(store: ArtifactStore, url: str) -> dict[str, Any]:
    """A data-URL image block as a content-addressed reference (bytes stay on disk)."""

    header, _, encoded = url.partition(",")
    if not header.startswith("data:") or not header.endswith(";base64"):
        raise PolicySessionError("observation image is not a base64 data URL")
    media_type = header[len("data:"):-len(";base64")]
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PolicySessionError("observation image is not valid base64") from exc
    ref = store.put_bytes(
        data, kind="context_image", suffix=".png" if media_type == "image/png" else ".img",
        media_type=media_type,
    )
    return {"type": "image_ref", "ref": ref}


def _refs_for_images(store: ArtifactStore, content: Any) -> Any:
    if not isinstance(content, list):
        return content
    blocks = []
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "image_url":
            blocks.append(_image_ref_block(store, str(block["image_url"]["url"])))
        else:
            blocks.append(copy.deepcopy(dict(block)) if isinstance(block, Mapping) else block)
    return blocks


class PiObservationDrafts:
    """The native_api static context and observation messages, trims as hints."""

    def _static_context(self, task_instruction: str) -> dict[str, Any] | None:
        task = task_instruction.strip()
        if not task:
            raise PolicySessionError("native task instruction must be non-empty")
        if self.bound_task_instruction is not None:
            if task != self.bound_task_instruction:
                raise PolicySessionError("native task instruction changed inside one session")
            return None
        self.bound_task_instruction = task
        if self.context_composition in NATIVE_CAUSAL_SLOTS:
            assembled = assemble_native_v3_static_context(
                harness_system=str(self.harness_system_template),
                project_instructions=self.project_template,
                episode_task_template=self.episode_prefix_template,
                task_instruction=task, task_id=self.task_id or self.run_id,
            )
            return {
                "system_text": assembled["system_text"], "task_sha256": assembled["task_sha256"],
                "messages": assembled["messages"], "sources": assembled["sources"],
            }
        prefix = self.episode_prefix_template.replace("{{TASK}}", task)
        system_text = self.project_template + "\n\n" + prefix
        return {"system_text": system_text, "task_sha256": None,
                "messages": [{"role": "system", "content": system_text}], "sources": []}

    def _observation_draft(self, *, task_instruction: str, context: Mapping[str, Any],
                           image_path: Path, terminal_lifecycle: bool) -> dict[str, Any]:
        if self.context_composition in NATIVE_CAUSAL_SLOTS and self.context_retention == LOSSLESS_CONTEXT_RETENTION:
            return self._causal_observation_draft(context=context, terminal_lifecycle=terminal_lifecycle)
        # The original v3 rewrite/compact projection and the legacy one carry no
        # trims or slots: the shared domain message, its images as references.
        if self.context_composition in NATIVE_CAUSAL_SLOTS:
            message = self._rewrite_causal_observation_message(
                context=context, terminal_lifecycle=terminal_lifecycle,
            )
        else:
            message = self._legacy_observation_message(
                task_instruction=task_instruction, context=context, image_path=image_path,
                terminal_lifecycle=terminal_lifecycle,
            )
        message["content"] = _refs_for_images(self.artifacts, message["content"])
        return {"message": message, "hints": []}

    def _causal_observation_draft(self, *, context: Mapping[str, Any], terminal_lifecycle: bool,
                                  ) -> dict[str, Any]:
        """The causal observation message the Native session built, with trims and slot as hints."""

        current = self._actual_snapshot(context)
        show_receipts = getattr(self, "toolkit_observation_receipts", True)
        receipt = self.pending_receipt_body if show_receipts else None
        before = self.last_actual_snapshot if receipt is not None else None
        if receipt is not None and before is None:
            raise PolicySessionError("mutation receipt lacks its pre-action actual snapshot")
        current_sequence = self.actual_sequence + 1
        current["actual_sequence"] = current_sequence
        before_sequence = before.get("actual_sequence") if isinstance(before, Mapping) else None
        if before is not None and not isinstance(before_sequence, int):
            raise PolicySessionError("causal input actual lacks a stable sequence")
        effect = self._minimal_action_effect(receipt) if receipt is not None else None
        current_actual: dict[str, Any] = {"actual_sequence": current_sequence}
        if show_receipts:
            current_actual.update(self._post_action_view_feedback(current))
        hints: list[dict[str, Any]] = []
        if self._delivery_trim():
            hints += retention_hints.omit_hints(
                current_actual, trim_view_freshness(current_actual, paired=before is not None),
                prefix=("current_actual",), rule="observation_delivery_trim_v2:view_freshness",
            )
        if "public_robot_facts" in current:
            current_actual["public_robot_facts"] = copy.deepcopy(current["public_robot_facts"])
        elif "end_effectors" in current:
            current_actual["end_effectors"] = copy.deepcopy(current["end_effectors"])
        else:
            current_actual.update({
                "eef_pose_xyz_xyzw": current["eef_pose_xyz_xyzw"],
                "aperture_mm": current["aperture_mm"],
            })
        surface = getattr(self.tools, "surface", None)
        if getattr(surface, "view_config", None) is not None:
            current_actual["view_dimensions"] = {
                item["camera"]: {"width": item["width"], "height": item["height"]}
                for item in current["images"]
            }
        if surface is not None and surface.supports(DEDUP_CAPABILITY):
            hints.append(retention_hints.hoist_hint(("current_actual",)))
        if self.context_composition == NATIVE_SLOTS_V4:
            round_index = context.get("remaining_budget", {}).get("session_round")
            if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 1:
                raise PolicySessionError("native_slots_v4 observation lacks its Session round")
            hints.append(retention_hints.round_static_facts_hint(
                at=("current_actual",), scope=round_index,
                listing=(round_static_facts.LISTING_ABSENT if self._delivery_trim()
                         else round_static_facts.LISTING_UNCHANGED),
            ))
        envelope = {
            "current_actual": current_actual,
            "last_action_effect": effect,
            "comparison": (
                {
                    "available": True,
                    "action_input_actual_sequence": before_sequence,
                    "action_result_actual_sequence": current_sequence,
                    "instruction": (
                        "Compare the retained ACTION INPUT ACTUAL with this ACTION "
                        "RESULT ACTUAL before choosing the next tool."
                    ),
                }
                if before is not None else {
                    "available": False,
                    "instruction": "Use this initial actual evidence to choose the next tool.",
                }
            ),
            "remaining_budget": {
                key: context.get("remaining_budget", {}).get(key)
                for key in ("robot_decisions", "model_calls", "wall_time_s",
                            "native_actions_limit", "native_actions_remaining")
                if key in context.get("remaining_budget", {})
            },
            "terminal_lifecycle": terminal_lifecycle,
        }
        if not show_receipts:
            envelope.pop("last_action_effect")
            envelope.pop("comparison")
        round_notice = context.get("runtime_notice")
        if isinstance(round_notice, Mapping) and round_notice.get("kind") == "session_round_start":
            envelope["session_round"] = {
                key: copy.deepcopy(round_notice[key])
                for key in ("kind", "round", "total_rounds", "budget_scope", "instruction")
                if key in round_notice
            }
        if self._delivery_trim():
            hints += retention_hints.omit_hints(
                envelope, trim_observation_envelope(envelope), rule="observation_delivery_trim_v2:envelope_defaults",
            )
        content: list[dict[str, Any]] = [{"type": "ora_json", "value": envelope, "hints": hints}]
        current_images = {item["camera"]: item["path"] for item in current["images"]}
        camera_order = [item["camera"] for item in current["images"]]
        if before is not None:
            before_images = {item["camera"]: item["path"] for item in before["images"]}
            if set(before_images) != set(current_images) and not (
                current.get("post_action_observation") or before.get("post_action_observation")
            ):
                raise PolicySessionError("causal actual camera set changed across mutation")
            label = "ACTION RESULT ACTUAL"
        else:
            label = "INITIAL ACTUAL" if show_receipts else "CURRENT ACTUAL"
        for camera in camera_order:
            path = Path(current_images[camera])
            content.extend([
                {"type": "text", "text": f"{label} · sequence {current_sequence} · {camera} · latest at this message"},
                {"type": "image_ref", "ref": self.artifacts.put_bytes(
                    path.read_bytes(), kind="context_image",
                    suffix=".png" if path.suffix.lower() == ".png" else ".img",
                    media_type="image/png" if path.suffix.lower() == ".png" else "image/jpeg",
                )},
            ])
        if show_receipts:
            content.extend(_refs_for_images(self.artifacts, self._skill_observation_blocks()))
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
                "input_actual_sequence": before_sequence,
                "result_actual_sequence": current_sequence,
                "new_actual_image_blocks": len(current_images),
                **({"post_action_observation": current["post_action_observation"]}
                   if show_receipts and "post_action_observation" in current else {}),
            },
            "_session_round": context.get("remaining_budget", {}).get("session_round"),
        }
        self.last_actual_snapshot = current
        self.actual_sequence = current_sequence
        self._dump_hints("observation", {"envelope": envelope, "hints": hints})
        return {"message": message}
