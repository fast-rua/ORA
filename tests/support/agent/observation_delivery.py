"""What the Pi agent core delivers for an observation or a tool result, computed in Python.

The Python half of the Pi core drafts each observation
(``PiPlanner._causal_observation_draft``) and projects each tool result
(``PiPlanner._project``), with the delivery rules (trims, once-per-payload
conventions, the round static-facts slot) as generic retention hints. The
TypeScript core resolves a draft (``resolveDraft`` in
``ora-pi/src/core/session.ts``): hints applied, the envelope serialized with
sorted keys and non-ASCII kept, image references read back as data URLs.
``Delivery`` does the same with the Python reference applier
(``ora.agent.pi.hints.apply_hints``, which ``tests/support/agent_core/hint_check.py``
holds to the TypeScript one), so a test sees the message the model receives
without starting node.

``host`` builds a ``PiPlanner`` without its process: the attributes the draft
and projection read, nothing else.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.pi.hints import StaticFactsState, apply_hints
from ora.agent.pi.policy_session import PiPlanner

LOSSLESS = "action_step_lossless_v4"


def host(run_root: Path, *, tools: Any, context_composition: str | None = "native_slots_v3",
         context_retention: str = LOSSLESS, supports_images: bool = True,
         observation_id: str | None = None, actual_sequence: int = -1) -> PiPlanner:
    """A ``PiPlanner`` with the state an observation draft and a tool projection read."""

    planner = PiPlanner.__new__(PiPlanner)
    planner.run_root = Path(run_root)
    planner.tools = tools
    planner.policy = {}
    planner.context_composition = context_composition
    planner.context_retention = context_retention
    planner.supports_images = supports_images
    planner.initial_observation = "automatic"
    planner.artifacts = ArtifactStore(planner.run_root)
    planner.pending_receipt = planner.pending_receipt_body = planner.last_actual_snapshot = None
    planner.actual_sequence = actual_sequence
    planner.last_action_step = None
    if observation_id is not None:
        planner._current_observation_id = lambda: observation_id
    return planner


def _data_url(store: ArtifactStore, ref: Mapping[str, Any]) -> str:
    data = (store.run_root / str(ref["path"])).read_bytes()
    return f"data:{ref['media_type']};base64,{base64.b64encode(data).decode('ascii')}"


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Delivery:
    """One session's deliveries: the round static-facts state persists across observations."""

    def __init__(self, planner: PiPlanner) -> None:
        self.planner = planner
        self.static_facts = StaticFactsState()
        self.slots: list[dict[str, Any]] = []
        self.envelopes: list[dict[str, Any]] = []

    def resolve(self, draft_message: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """``resolveDraft``: the delivered message and the slot message a new round produced."""

        message = copy.deepcopy(dict(draft_message))
        slot = None
        if isinstance(message.get("content"), list):
            blocks = []
            for block in message["content"]:
                if isinstance(block, Mapping) and block.get("type") == "ora_json":
                    delivered, slot = apply_hints(block["value"], block.get("hints") or [], self.static_facts)
                    self.envelopes.append(delivered)
                    blocks.append({"type": "text", "text": _dumps(delivered)})
                elif isinstance(block, Mapping) and block.get("type") == "image_ref":
                    blocks.append({"type": "image_url",
                                   "image_url": {"url": _data_url(self.planner.artifacts, block["ref"])}})
                else:
                    blocks.append(block)
            message["content"] = blocks
        if slot is not None:
            self.slots.append(slot)
        return message, slot

    def observation(self, context: Mapping[str, Any], *, terminal_lifecycle: bool = False,
                    task_instruction: str = "", image_path: Path | None = None,
                    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """The delivered observation message for ``context`` and its round slot, if any.

        The draft is the one ``PiPlanner.act`` sends for the planner's
        composition and retention; ``task_instruction`` and ``image_path`` only
        matter to a composition without causal slots.
        """

        draft = self.planner._observation_draft(
            task_instruction=task_instruction, context=context, image_path=image_path,
            terminal_lifecycle=terminal_lifecycle,
        )
        return self.resolve(draft["message"])

    def tool_result(self, name: str, result: Mapping[str, Any], *, call_id: str = "call-0",
                    step: int = 0) -> dict[str, Any]:
        """The delivered tool message (``recordToolResult``): the visible result with its
        trims applied, and the user message carrying its images as data URLs (``None``
        without images, or when the model takes no images)."""

        projected = self.planner._project(name, result, call_id=call_id, step=step)
        visible, _ = apply_hints(projected["visible"], projected["visible_hints"], StaticFactsState())
        message: dict[str, Any] = {"role": "tool", "tool_call_id": call_id, "name": name,
                                   "content": _dumps(visible)}
        images = [{"type": "image_url", "image_url": {"url": _data_url(self.planner.artifacts, ref)}}
                  for ref in projected["images"]]
        image_message = None
        if images and projected["image_text"] is not None:
            image_message = {"role": "user", "content": [{"type": "text", "text": projected["image_text"]}, *images],
                             "_tool_image_step": step, "_tool_image_name": name, "_tool_image_call_id": call_id}
        return {"message": message, "visible": visible, "images": images, "image_message": image_message,
                "image_text": projected["image_text"], "canonical": projected["canonical"]}


def observation_text(message: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON envelope of a delivered observation message."""

    return json.loads(message["content"][0]["text"])
