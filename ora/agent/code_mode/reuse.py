"""Reused perception results (``policy.code_mode.library: extended``): the same look at the same observation.

The policy session keeps the answers of a script's read-only perception calls for the observation they were
made on (``agent/pi/code_mode.py``). The same call with the same arguments on the same observation returns the
kept answer marked ``reused: true``, and no tool runs; a new observation starts afresh.

Which calls: those whose answer is a function of their arguments and the published observation alone, the
toolkit tools that read it (``REUSABLE``), and only where the surface itself classes the call read-only. Tools
that persist something (``mark_point``), ask the checker (``check_task``) or the planner or a learned service for
a plan (``grasp_plan``, ``align_plan``), return the images (``observe``) or compute (``compute``) are not kept.
A kept answer is the script's view of the result (its causal projection), not a second tool call: it is not
recorded as one, counts against no call limit, and carries no images again.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

#: The script functions whose answers are kept. A legacy tool's dotted name is spelled with underscores (``wire_name``).
REUSABLE = frozenset({"vision_ground", "vision_segment", "geometry_query"})
#: The marker a reused answer carries (``robot.Result.reused``).
REUSED_FIELD = "reused"


class KeptResults:
    """The kept answers of the current observation: keyed by the call (tool and arguments), dropped when the
    observation changes."""

    def __init__(self) -> None:
        self.observation_id: str | None = None
        self.answers: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def key(name: str, arguments: Mapping[str, Any]) -> tuple[str, str] | None:
        """The call's identity, or None for a tool that is not kept (or arguments that are not JSON)."""

        wire = name.replace(".", "_")
        if wire not in REUSABLE:
            return None
        try:
            return wire, json.dumps(arguments, sort_keys=True)
        except (TypeError, ValueError):
            return None

    def get(self, observation_id: str | None, name: str, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
        """The kept answer to this call on this observation, marked reused."""

        key = self.key(name, arguments)
        if key is None or observation_id is None or observation_id != self.observation_id:
            return None
        kept = self.answers.get(key)
        return None if kept is None else {**copy.deepcopy(kept), REUSED_FIELD: True}

    def keep(self, observation_id: str | None, name: str, arguments: Mapping[str, Any],
             answer: Mapping[str, Any]) -> None:
        """Keep the answer of a call that succeeded (a refusal or an error is asked again) on this observation."""

        key = self.key(name, arguments)
        if key is None or observation_id is None or not answer.get("ok") or "error" in answer:
            return
        if observation_id != self.observation_id:
            self.observation_id, self.answers = observation_id, {}
        self.answers[key] = copy.deepcopy(dict(answer))


__all__ = ["REUSABLE", "REUSED_FIELD", "KeptResults"]
