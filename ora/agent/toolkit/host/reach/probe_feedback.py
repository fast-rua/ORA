"""Bounded decision summaries of immutable motion-probe results."""
from __future__ import annotations

from typing import Any

from ora.contracts.planning_diagnostics import (
    PROBE_FEEDBACK_REQUEST_SCHEMA, PROBE_IK_CANDIDATE_FEEDBACK,
)
from ora.agent.toolkit.tools.grasp_execute._probe_feedback import summarize_motion_probe

#: A surface declaring this sends the v2 probe request and receives, per query,
#: the bounded IK candidate evidence the backend kept for each variant.
IK_CANDIDATE_FEEDBACK_CAPABILITY = "motion_probe_ik_candidate_feedback_v1"


def probe_request_fields(surface: Any) -> dict[str, str]:
    """The request schema (and feedback selection) this surface's probes send."""

    if surface is not None and surface.supports(IK_CANDIDATE_FEEDBACK_CAPABILITY):
        return {"schema": PROBE_FEEDBACK_REQUEST_SCHEMA,
                "planner_feedback": PROBE_IK_CANDIDATE_FEEDBACK}
    return {"schema": "ora.motion-probe-request.v1"}
