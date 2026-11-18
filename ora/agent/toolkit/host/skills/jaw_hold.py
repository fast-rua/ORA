"""A held/not-held signal from the jaw-stall evidence a backend already reports.

``closure_evidence`` compares a metric jaw separation with the target's visible
extents. RoboDojo publishes no metric separation: the ARX pad aperture is
uncalibrated (``localization.measured_jaw_separation_m`` returns ``None`` on
purpose), so every verify stage of one recorded batch's ten completed
``grasp_execute`` calls reported ``closure_strength: unavailable`` while the
close stage beside it had measured exactly the fact in question.

Under the ARX drive limiter a jaw that closes on something stops short of the
closed floor (``stalled_before_target``) and holds a drive target 0.2 below
that encoder reading; a jaw that closes on nothing reaches the floor; a jaw
whose object leaves during the lift travels down to the held drive target.
This module turns those normalized encoder readings into ``likely``,
``unlikely`` or ``unknown``, and returns the readings next to the thresholds.
It never converts them to a width, and none of its answers is contact,
attachment or retention.

A close that stalls without leaving the open position is not the first case:
jaws that never moved were stopped by something outside them (a fingertip on
the rim, the table) or never driven, and nothing between them pushes the pads
past fully open. A recorded stack_bowls run (+31.7 min) stalled at 0.99999999
after 3 native actions -- the fingers at their 44 mm stop -- and the released
rule read the unchanged reading after the lift as "still held apart". Such a
close is ``unlikely`` with reason ``jaws_never_closed``: its reading lies
within the arrival tolerance of the opening the open stage commanded, the band
that stage itself counted as open. On the X5 no admitted candidate needs that
band: the 80 mm admission ceiling is a normalized opening of 0.918.

The readings are the two-finger mean the backend publishes. The rules above assume the fingers move as a symmetric pair; a pair
shifted to one side says nothing about holding either way (recorded: one finger at
its 44 mm stop, the other 30 mm in, read as "fully open"). When the close or
after-lift reading has a lateral finger offset beyond ``SYMMETRIC_OFFSET_BOUND_M``
the verdict is ``unknown`` with reason ``finger_offset_exceeds_symmetric_bound``.
Only the readings a verdict compares count: a close that reached the floor or
never left the open position is judged on the close reading alone.
"""

from __future__ import annotations

from ora.agent.toolkit.tools.grasp_execute._jaw_hold import (
    CAPABILITY, CLAIM_BOUNDARY, OPEN_POSITION_NORMALIZED, SCHEMA, SYMMETRIC_OFFSET_BOUND_M, after_lift_offset,
    after_lift_opening, close_outcome, hold_evidence, open_position, verify_stage_evidence,
)


__all__ = ["CAPABILITY", "CLAIM_BOUNDARY", "OPEN_POSITION_NORMALIZED", "SCHEMA", "SYMMETRIC_OFFSET_BOUND_M",
           "after_lift_offset", "after_lift_opening", "close_outcome", "hold_evidence", "open_position",
           "verify_stage_evidence"]
