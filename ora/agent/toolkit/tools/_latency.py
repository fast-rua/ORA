"""The plan's model-facing latency switches, as one tool call reads them.

``orchestrator/spec/latency.py`` resolves a run's ``latency_profile``; only a plan that turns a model-facing switch
off gives its tool calls ``latency_switches`` (``host/runtime/dispatch.py``), so a call without the key, every
benchmark one and every sealed one, has every switch on.

- ``reach_reference``: ``move_pose``, ``move_to``, ``rotate_wrist`` and ``rotate_pitch`` ask the planner's IK for
  ``compiled_pose.targets.<arm>.reach_along_request`` (``move_pose/_reach.py``). Off, they ask nothing and say so.
- ``preview_images``: a dry run's record and a real call refused by its plan's verdict carry the rendered picture
  (``_dry_run.py``). Off, a dry run keeps its numbers and says it has no picture; a refusal stands alone.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..lib.action_preview import PICTURE_OFF

KEY = "latency_switches"
#: What a target's ``reach_along_request`` says when the run asks no reach reference.
REACH_OFF = {"status": "off", "reason": "this run asks no reach reference before a move (latency profile)"}


def on(config: Mapping[str, Any], name: str) -> bool:
    """Whether switch ``name`` is on for a call with tool configuration ``config``."""

    return (config.get(KEY) or {}).get(name, True) is not False


def host_on(host: Any, name: str) -> bool:
    """Whether switch ``name`` is on for a tool host (``RobotToolRuntime.latency_switches``)."""

    return (getattr(host, KEY, None) or {}).get(name, True) is not False


__all__ = ["KEY", "PICTURE_OFF", "REACH_OFF", "host_on", "on"]
