"""A receipt's copy for the policy leaves out the evidence that names an implementation, and nothing else."""

from __future__ import annotations

import copy

from ora.contracts.policy_receipt import names_timing, policy_receipt

TIMING = {"implementation": "toppra_ruckig", "versions": {"toppra": "0.6.10"}, "jerk_source": "Franka, a URL"}
PLANNER = {"schema": "ora.curobo-plan-summary.v1", "status": "planned", "reachable": True,
           "canonical_artifact": {"path": "curobo/plans/action-0000.json", "sha256": "a" * 64},
           "planning_diagnostics": {"phase": "trajectory_planning"}}
PACING = {"speed": {"speed": "fast"}, "wrist": {"travel_deg": 1.0}, "planned_ticks": 9, "duration_s": 0.45,
          "timing": TIMING}
RECEIPT = {"schema": "ora.libero-sparse-motion-receipt.v1", "action_id": "a-0",
           "executor": ["libero_curobo_joint_position"],
           "motion": {**PACING}, "segment": {"pacing": PACING, "planner": PLANNER},
           "segments": [{"pacing": PACING, "planner": PLANNER}],
           # A timing anywhere else is not this evidence.
           "timing_s": {"total": 1.0}, "feedback": {"timing": "kept"}}


def test_the_policy_copy_drops_timing_and_the_planner_record_and_keeps_the_rest():
    original = copy.deepcopy(RECEIPT)
    public = policy_receipt({"path": "receipts/action-0000.json", "sha256": "b" * 64, "receipt": RECEIPT})
    assert RECEIPT == original  # never edited
    inner = public["receipt"]
    for pacing in (inner["motion"], inner["segment"]["pacing"], inner["segments"][0]["pacing"]):
        assert "timing" not in pacing and pacing["speed"] == {"speed": "fast"} and pacing["planned_ticks"] == 9
    for planner in (inner["segment"]["planner"], inner["segments"][0]["planner"]):
        assert planner == {"status": "planned", "reachable": True,
                           "planning_diagnostics": {"phase": "trajectory_planning"}}
    assert inner["timing_s"] == {"total": 1.0} and inner["feedback"] == {"timing": "kept"}
    assert inner["executor"] == ["libero_curobo_joint_position"]
    assert policy_receipt(public) == public


def test_names_timing_finds_a_copy_that_still_holds_timing():
    assert names_timing(RECEIPT) and names_timing({"receipt": RECEIPT})
    assert not names_timing(policy_receipt(RECEIPT)) and not names_timing(None)
    # The planner's record alone is pre-existing evidence a backend may still embed.
    assert not names_timing({"segment": {"planner": PLANNER}})
