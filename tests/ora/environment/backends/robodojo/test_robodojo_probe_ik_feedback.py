"""A RoboDojo motion probe is one IK solve per variant, and the grasp precommit gate reads its reasons.

``probe_motion`` answers from ``reachability.solve_batch``: the shared planner's IK
in the world a plan to the pose would see; a session with no planner answers no
variant (``no_planner``). One solve has no per-seed counts, so the v2 request a surface
declaring ``motion_probe_ik_candidate_feedback_v1`` still sends gets the v1 record:
no candidate counts, no diagnostics, no artifact and no plan reference. Sealed runs
keep the SDK candidate evidence they recorded (D3L-R002).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tests.support.robodojo.backend as backend
from tests.support.agent.reach_check import planner_ik
from tests.support.agent.surface_catalog import surface_of
from ora.agent.toolkit.host.reach.probe_feedback import (
    IK_CANDIDATE_FEEDBACK_CAPABILITY, probe_request_fields, summarize_motion_probe,
)
from ora.agent.toolkit.interface.projections import motion_probe
from ora.agent.toolkit.interface.types import Response
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.tools.grasp_execute import _precommit
from ora.contracts.planning_diagnostics import PROBE_FEEDBACK_REQUEST_SCHEMA, PROBE_IK_CANDIDATE_FEEDBACK
from ora.environment.backends.robodojo import shared_motion
from ora.environment.backends.robodojo.bridge import RobodojoError

pytestmark = pytest.mark.usefixtures("fake_sdk_candidate_boundary")

DOJO_CLOSED = "robodojo_placement_toolkit_v5"
EXACT, FLIP = "variant-00-exact-+0deg", "variant-01-jaw-flip-180deg"
IK_COUNTS = {"constraint_rejection_counts", "planning_diagnostics", "ik_validation_artifact"}


@pytest.fixture
def sparse(tmp_path):
    instance = backend.sparse_session(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


def _v2(request):
    return {**request, "schema": PROBE_FEEDBACK_REQUEST_SCHEMA,
            "planner_feedback": PROBE_IK_CANDIDATE_FEEDBACK}


def _probe(sparse, tmp_path, *, feedback=False, policy="exact"):
    _observation, request = backend.probe_request(sparse, tmp_path, [0.05, 0, 0.8], policy=policy)
    return sparse.probe_motion(_v2(request) if feedback else request)


def _flipped(pose):
    """The jaw flip of the identity orientation: the fake plant's exact target is ``[0, 0, 0, 1]``."""
    return abs(pose[3]) > 0.5


def test_the_planners_ik_answers_every_variant_from_the_measured_joints(sparse, tmp_path, monkeypatch):
    requests = planner_ik(sparse, monkeypatch)
    clock = sparse._clock()
    result = _probe(sparse, tmp_path, policy="parallel_jaw_equivalents")
    assert result["schema"] == "ora.motion-probe-result.v1"
    assert result["world_state_changed"] is False and result["robot_decisions_consumed"] == 0
    assert [query["from_index"] for query in requests[0]["queries"]] == [None, None]
    assert requests[0]["world"]["input"] == {"path": "world.npz"} and requests[0]["held_object"] == {"source": "world"}
    row = result["results"][0]
    assert row["reachable"] is True and row["plan_ref"] is None and row["selected_variant_id"] == EXACT
    assert row["selected_target_pose_xyzw"] == [0.05, 0, 0.8, 0, 0, 0, 1]
    exact, flip = row["variants"]
    assert (exact["variant_id"], flip["variant_id"]) == (EXACT, FLIP)
    assert flip["target_pose_xyzw"][3:] == pytest.approx([1, 0, 0, 0])
    assert exact["planner"] == {"kind": "shared_ik", "plan": "none"}
    assert exact["plan_ref"] is None and exact["trajectory_steps"] == 0 and "failure" not in exact
    assert exact["joint_delta_rad"] == pytest.approx(0.05)
    assert exact["endpoint_error"]["translation_m"] == pytest.approx(0.0, abs=1e-9)
    assert sparse.environment.commands == [] and sparse._clock() == clock
    assert not (sparse.output_root / "motion-probes").exists()


def test_the_selected_variant_is_the_admitted_one_nearest_the_measured_joints(sparse, tmp_path, monkeypatch):
    planner_ik(sparse, monkeypatch, refuse=lambda pose: None if _flipped(pose) else "unsolved")
    row = _probe(sparse, tmp_path, policy="parallel_jaw_equivalents")["results"][0]
    assert row["reachable"] is True and row["selected_variant_id"] == FLIP
    assert [variant["reachable"] for variant in row["variants"]] == [False, True]


@pytest.mark.parametrize("refusal,code", [("goal_below_support", "goal_below_support"), ("unsolved", "ik_failed")])
def test_a_refused_variant_names_its_reason(sparse, tmp_path, monkeypatch, refusal, code):
    planner_ik(sparse, monkeypatch, refuse=lambda pose: refusal)
    row = _probe(sparse, tmp_path)["results"][0]
    assert row["reachable"] is False and row["selected_variant_id"] is None and row["plan_ref"] is None
    assert row["variants"][0]["failure"]["code"] == code and row["variants"][0]["joint_delta_rad"] is None


@pytest.mark.parametrize("reason,arrange", [
    ("stale_observation", lambda session: setattr(shared_motion, "world_input", lambda session, arm: (None, []))),
    ("planner_error", lambda session: setattr(session.motion_planner, "solve_ik", lambda request: {
        "status": "error", "reason": "world_build_failed"})),
])
def test_a_probe_with_no_answer_is_unreachable_with_its_reason(sparse, tmp_path, monkeypatch, reason, arrange):
    planner_ik(sparse, monkeypatch)
    _observation, request = backend.probe_request(sparse, tmp_path, [0.05, 0, 0.8])
    arrange(sparse)
    row = sparse.probe_motion(request)["results"][0]
    assert row["reachable"] is False and row["variants"][0]["failure"]["code"] == reason


def test_the_v2_request_gets_the_same_record(sparse, tmp_path, monkeypatch):
    planner_ik(sparse, monkeypatch)
    plain = _probe(sparse, tmp_path)["results"]
    assert _probe(sparse, tmp_path, feedback=True)["results"] == plain
    assert not IK_COUNTS & set(plain[0]["variants"][0]["planner"])


@pytest.mark.parametrize("damage", [
    {"planner_feedback": "everything"},
    {"schema": "ora.motion-probe-request.v1"},
    {"planner_feedback": None},
])
def test_a_malformed_feedback_request_is_refused(sparse, tmp_path, damage):
    _observation, request = backend.probe_request(sparse, tmp_path, [0.05, 0, 0.8])
    with pytest.raises(RobodojoError, match="drifted"):
        sparse.probe_motion({**_v2(request), **damage})


def test_a_session_with_no_planner_answers_no_variant(sparse, tmp_path, monkeypatch):
    from ora.environment.backends.robodojo import ik

    assert sparse.motion_planner is None
    monkeypatch.setattr(ik, "sdk_candidates", lambda *arguments, **keywords: pytest.fail("the SDK's IK ran"))
    row = _probe(sparse, tmp_path, policy="parallel_jaw_equivalents")["results"][0]
    assert row["reachable"] is False and row["selected_variant_id"] is None and row["plan_ref"] is None
    assert [variant["failure"]["code"] for variant in row["variants"]] == ["no_planner", "no_planner"]
    assert row["variants"][0]["planner"] == {"kind": "shared_ik", "plan": "none"}


def test_the_summary_counts_the_reasons_and_no_ik_candidates(sparse, tmp_path, monkeypatch):
    planner_ik(sparse, monkeypatch, refuse=lambda pose: "goal_below_support")
    result = _probe(sparse, tmp_path, feedback=True, policy="parallel_jaw_equivalents")
    row = summarize_motion_probe(result, ik_candidates=True)[0]
    assert row["variant_failure_counts"] == {"goal_below_support": 2}
    assert row["admitted_count"] == 0 and row["plan_ref"] is None
    assert not {"ik_candidate_counts", "ik_outcome_counts", "constraint_rejection_counts"} & set(row)


@pytest.mark.parametrize("profile,schema", [
    (DOJO_CLOSED, PROBE_FEEDBACK_REQUEST_SCHEMA),
    ("robotwin_toolchain_b2_closed_noservo_v2", PROBE_FEEDBACK_REQUEST_SCHEMA),
    ("libero_placement_direct_overhead_v1", "ora.motion-probe-request.v1"),
])
def test_only_the_declaring_surface_sends_the_feedback_request(profile, schema):
    fields = probe_request_fields(surface_of(profile))
    assert fields["schema"] == schema
    assert ("planner_feedback" in fields) is (schema == PROBE_FEEDBACK_REQUEST_SCHEMA)


def _gate(sparse, tmp_path):
    """The toolkit's precommit check, over the session's probe and the public projection."""
    surface = surface_of(DOJO_CLOSED)
    assert surface.supports(IK_CANDIDATE_FEEDBACK_CAPABILITY)
    observation, _request = backend.probe_request(sparse, tmp_path, [0.05, 0, 0.8])

    def probe_motion(*, request):
        data = json.loads(motion_probe(sparse.probe_motion(request)).data)
        return Response(json.dumps({"status": "ok", "method": "probe_motion", "data": data}).encode())

    pose = {"position_m": [0.05, 0, 0.8], "orientation_xyzw": [0, 0, 0, 1]}
    lower = {**pose, "position_m": [0.05, 0, 0.76]}
    commitment = {"arm": "left", "observation_id": observation["observation_id"], "candidate_id": "grasp-a1",
                  "poses": {"pregrasp": pose, "approach": lower, "lift": pose}}
    config = {"probe_request_fields": probe_request_fields(surface), "ik_candidate_feedback": True,
              "workspace_bounds_m": sparse.config["workspace_bounds_m"], "lift_m": 0.03}
    observed = {"observation_id": observation["observation_id"], "observation_sha256": "a" * 64}
    return lambda: _precommit.check_pregrasp(SimpleNamespace(planner=SimpleNamespace(probe_motion=probe_motion)),
                                             commitment, config, observed), commitment


def test_the_gate_refuses_with_the_reasons_of_the_ik(sparse, tmp_path, monkeypatch):
    planner_ik(sparse, monkeypatch, refuse=lambda pose: "goal_below_support" if pose[2] < 0.78 else None)
    check, _commitment = _gate(sparse, tmp_path)
    with pytest.raises(ToolRefusal) as refused:
        check()
    assert refused.value.code == "grasp_pregrasp_unreachable"
    constraints = refused.value.constraints
    assert constraints["unreachable_stages"] == ["approach"]
    pregrasp, approach = constraints["stage_summaries"]
    assert pregrasp["reachable"] is True and pregrasp["plan_ref"] is None
    assert approach["variant_failure_counts"] == {"goal_below_support": 2}
    assert not {"ik_candidate_counts", "ik_outcome_counts", "constraint_rejection_counts"} & set(approach)


def test_the_gate_still_adopts_the_jaw_flip_every_stage_admitted(sparse, tmp_path, monkeypatch):
    planner_ik(sparse, monkeypatch, refuse=lambda pose: None if _flipped(pose) else "unsolved")
    check, commitment = _gate(sparse, tmp_path)
    evidence = check()
    assert evidence["selected_variant_id"] == FLIP
    assert evidence["pose_selection"]["decision"] == "adopted_equivalent"
    assert commitment["poses"]["approach"]["orientation_xyzw"] == pytest.approx([1, 0, 0, 0])
