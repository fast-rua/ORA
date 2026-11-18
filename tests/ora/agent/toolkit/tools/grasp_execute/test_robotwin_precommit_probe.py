"""RoboTwin's precommit motion probe on the toolkit, from a recorded smoke of the switch.

The run (``tests/data/robotwin/grasp-execute-precommit-probe.json``) chose ``grasp_plan``'s first candidate, and both
``grasp_execute`` calls came back ``grasp_proposal_failed: undeclared nested object in public scalar field``. A variant
the worker's planner failed carries the public planner error object (``stage``, ``type``, ``message``), and the toolkit's
probe projection took that field for a scalar; the handlers read the worker's answer as it is. The run did not record
the worker's answer, so the worker's own ``probe_motion`` rebuilds it for the recorded request
(``tests/support/agent/stack_parity.RoboTwinWorkerProbe``). The parity rows run the whole call on both stacks against
the same worker probe: ``robotwin grasp_execute`` and the stage loop in ``tests/ora/agent/toolkit/test_stack_parity.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ora.agent.toolkit.interface.projections import motion_probe
from ora.agent.toolkit.tools.grasp_execute._probe_feedback import summarize_motion_probe
from tests.support.agent.stack_parity import RoboTwinWorkerProbe

ROOT = Path(__file__).resolve().parents[6]
RECORD = json.loads((ROOT / "tests/data/robotwin/grasp-execute-precommit-probe.json").read_text(encoding="utf-8"))


def _errors(answer):
    return [variant["planner"]["error"] for query in answer["results"] for variant in query["variants"]
            if variant["planner"].get("error") is not None]


def test_the_recorded_probe_asks_for_the_recorded_candidate():
    candidate = RECORD["grasp_plan"]["candidate"]
    assert RECORD["grasp_execute"]["arguments"]["candidate_id"] == candidate["candidate_id"]
    assert RECORD["grasp_execute"]["refusal"]["message"] == "undeclared nested object in public scalar field"
    assert {query["query_id"]: (query["arm"], query["pose"]) for query in RECORD["probe_request"]["queries"]} == {
        "pregrasp": (RECORD["grasp_plan"]["arm"], candidate["pregrasp_pose"]),
        "approach": (RECORD["grasp_plan"]["arm"], candidate["contact_pose"])}


def test_both_stacks_read_the_worker_s_answer_to_the_recorded_probe_alike(tmp_path):
    answer = RoboTwinWorkerProbe(tmp_path)(RECORD["probe_request"])
    errors = _errors(answer)
    assert errors and all(set(error) == {"stage", "type", "message"} for error in errors)
    projected = json.loads(motion_probe(answer).data)  # what the toolkit's precommit check reads
    assert _errors(projected) == errors
    assert summarize_motion_probe(projected, ik_candidates=True) == summarize_motion_probe(answer, ik_candidates=True)


def test_libero_pro_s_planner_error_projects_as_robotwin_s_does():
    """LIBERO-Pro advertises no ``grasp_execute``, so no tool of its surface probes, but its worker's rows carry the
    same error object: a planner that raised is ``{"stage": "planner", **error}`` (``libero_pro/motion_probe.py``)."""
    planner = {"called": True, "status": "error", "stage": "FAILED", "valid_query": None, "attempts": None,
               "trajopt_attempts": None, "used_graph": None, "goalset_index": None, "path_steps": 0,
               "interpolation_dt_s": None, "errors": {"position": None, "rotation": None, "cspace": None},
               "timings_s": {"total": None},
               "error": {"stage": "planner", "type": "RuntimeError", "message": "the planner raised"}}
    answer = {"schema": "ora.motion-probe-result.v1", "observation_id": "o", "variant_policy": "exact",
              "robot_decisions_consumed": 0, "world_state_changed": False, "results": [{
                  "query_id": "q", "arm": "right", "reachable": False, "selected_variant_id": None, "plan_ref": None,
                  "selected_target_pose_xyzw": None, "variants": [{"variant_id": "exact", "reachable": False,
                                                                    "planner": planner}]}]}
    assert _errors(json.loads(motion_probe(answer).data)) == [planner["error"]]
