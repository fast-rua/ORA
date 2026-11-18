"""The tool-runtime half of a closed skill: commit, drive, verify, release.

One policy call has to turn into several Supervisor executions without asking
the policy again, and the parts of that are split across three owners. This file
covers the part the tool runtime owns: resolving the chosen candidate against
the observation that produced it, committing one action for the whole sequence,
producing each stage's target from the committed candidate, and advancing and
releasing the ledger.

The releases are the tests worth reading. A sequence that stalls holding a
decision the policy cannot take back is worse than the three explicit calls
these skills replace, so the ledger is reconciled by any later call rather than
only by the writer that committed it, and `build_skill_receipt` releases
every terminal path.
"""

from __future__ import annotations


import pytest

from ora.agent.toolkit.host.perception import localization


def test_a_jaw_separation_under_the_near_closed_bound_is_a_strong_negative():
    """The negative branch the LIBERO runs were losing, given both inputs.

    2.35 mm of separation is below the 3 mm near-closed bound, so the jaws met:
    evidence that nothing of the target's measured size is between them.
    """

    evidence = localization.closure_evidence(
        gripper_opening_m=0.0023495691745634946,
        localization={
            "schema": localization.SCHEMA,
            "principal_extents_m": [0.1586882214457184,
                                    0.06582657322217246,
                                    0.044610149665408064],
        },
    )
    assert evidence["strength"] == "strong_negative"
    assert "missing" not in evidence


def test_the_measured_jaw_separation_reads_either_backend_and_never_the_command():
    from_robotwin = {"gripper_opening_m": 0.0412}
    from_libero = {"gripper_finger_qpos": [0.04000015011914127,
                                           -0.03999216510420057],
                   "gripper": 0.9999}
    assert localization.measured_jaw_separation_m(from_robotwin) == pytest.approx(0.0412)
    assert localization.measured_jaw_separation_m(from_libero) == pytest.approx(0.07999231522334184)
    # A normalized command is not a length and must not stand in for one.
    assert localization.measured_jaw_separation_m({"gripper": 0.5099}) is None
    assert localization.measured_jaw_separation_m({"gripper_finger_qpos": [0.02]}) is None
    assert localization.measured_jaw_separation_m(None) is None
