"""The live boundary carries the publication's execution state, so a restore resumes the IK seed generators."""

import json

import pytest

from ora.agent.toolkit.interface.types import CorruptRecord
from ora.orchestrator.toolkit.live_boundary import _exec_values

PUBLISHED = {"path": "policy-state/steps/step-0001.json", "sha256": "a" * 64}
IK_STATE = {"schema": "ora.robodojo-ik-sampler-state.v1", "arms": {"left": {"generator_state_hex": "00"}}}


def make_run(root):
    """A run whose step-1 state snapshot was published with ``PUBLISHED`` and recorded the IK generators."""
    snapshot = root / "run/state/steps/step-0001/snapshot.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps({"policy_state": PUBLISHED, "values": {
        "exec.ik_sampler_state": IK_STATE, "exec.native_actions": {"limit": 50, "used": 7},
        "tool.marks": {}, "env.observation": {"sequence": 1}}}))
    return root / "run"


def test_exec_values_of_the_published_snapshot(tmp_path):
    run = make_run(tmp_path)
    values = _exec_values(run, PUBLISHED)
    assert values == {"exec.ik_sampler_state": IK_STATE, "exec.native_actions": {"limit": 50, "used": 7}}
    assert _exec_values(run, {"step": 1, "sha256": "a" * 64}) == values
    with pytest.raises(CorruptRecord):
        _exec_values(run, dict(PUBLISHED, sha256="0" * 64))
    assert _exec_values(run, dict(PUBLISHED, step=9)) == {}
