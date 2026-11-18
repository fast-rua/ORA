"""The routes a RoboDojo runtime may name, and the SDK executor's actions it now refuses.

``runtime.recovery_route`` and ``runtime.contact_approach_route`` are derived by
the RunSpec from the surface that declares them, and the session still
validates both: plans sealed with them carry the keys. The moves themselves
are ``follow_joint_trajectory`` knots the toolkit plans, so ``move_pose``,
``move_free_space`` and ``recover_motion`` are refused before anything is
written, typed as a correctable argument.
"""
from __future__ import annotations

import pytest

from ora.agent.session_core.receipt_projection import public_motion_failure_code
from ora.contracts.refusal import EnvironmentRefusal
from ora.environment.backends.robodojo import recovery
from ora.environment.backends.robodojo.bridge import RobodojoError
from ora.environment.backends.robodojo.sparse_motion import STAGED_CONTACT_STRATEGY
from tests.support.robodojo.backend import pose, sparse_session, v3


def test_an_unknown_recovery_route_is_refused_at_session_construction(tmp_path):
    with pytest.raises(RobodojoError, match="recovery_route"):
        sparse_session(tmp_path, recovery_route="teleport")


def test_an_unknown_contact_route_is_refused_at_session_construction(tmp_path):
    with pytest.raises(RobodojoError, match="contact_approach_route"):
        sparse_session(tmp_path, contact_approach_route="global")


@pytest.mark.parametrize("raw,public", [("controller_stalled", "controller_stalled"),
                                         ("native_action_budget_exhausted", "controller_timeout"),
                                         ("recovery_not_reached", "execution_fault")])
def test_public_codes_no_longer_call_a_blocked_recovery_an_execution_fault(raw, public):
    # The last row is the released code: it read as an infrastructure fault.
    assert public_motion_failure_code("execution", raw) == public


def test_the_runspec_derives_both_routes_only_from_the_declaring_surface():
    from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
    from tests.support.robodojo.closed_skills_default import ROOT, spec_for

    # The Dojo default declares both routes; a runtime value the surface does not
    # declare is refused.
    spec = spec_for("stack_blocks")
    runtime = resolve_run_spec(spec, root=ROOT)["runtime"]
    assert runtime["recovery_route"] == recovery.RETRACT_THEN_UNWIND
    assert runtime["contact_approach_route"] == STAGED_CONTACT_STRATEGY
    spec["runtime"]["recovery_route"] = "some_other_route"
    with pytest.raises(RunSpecError, match="recovery_route"):
        resolve_run_spec(spec, root=ROOT)


@pytest.mark.parametrize("kind,targets", [
    ("move_pose", {"left": pose([0.05, 0.0, 0.8])}),
    ("move_free_space", {"left": pose([0.05, 0.0, 0.8])}),
    ("recover_motion", {"left": "home"}),
])
def test_the_sdk_executor_s_actions_are_refused_before_any_write(tmp_path, kind, targets):
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND)
    try:
        action = v3(session, tmp_path, kind=kind, targets=targets)
        before = (session.environment.take_action_cnt[0], session.sim_steps, session.action_sequence)
        with pytest.raises(EnvironmentRefusal) as refused:
            session.execute(action)
        assert refused.value.field == "action" and refused.value.received == kind
        assert refused.value.allowed == ["follow_joint_trajectory", "set_gripper", "wait"]
        assert (session.environment.take_action_cnt[0], session.sim_steps, session.action_sequence) == before
        assert not (tmp_path / "native" / "receipts").exists()
        assert not (tmp_path / "native" / "controller").exists()
        # Nothing consumed the observation: the next action may still bind it.
        assert session.last_observation_id == action["observation_id"]
    finally:
        session.close()
