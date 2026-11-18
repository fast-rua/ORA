"""A code-mode script's legacy action through the Supervisor's executor (``ora/agent/pi/code_mode.py``).

A committed skill sequence (RoboTwin's closed-skill ``grasp_execute``) runs its stage loop inside the script's
parent: its call carries the staged receipt and the post-close views, and its time is the Supervisor's. An action
that did not start leaves nothing behind, a committed skill's ledger included; any other failure is the parent's.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ora.agent.pi.code_mode import PiCodeMode
from ora.agent.toolkit.interface.types import ActionNotStarted
from ora.agent.toolkit.host.skills import driver as skill_driver
from ora.agent.toolkit.host.skills import ledger as skill_ledger

SEQUENCE_RECEIPT = "ora.manipulation-skill-sequence-receipt.v1"
STAGED = {"action": "run_python", "status": "completed",
          "detail": {"stages": [{"stage": "close", "status": "executed", "reason": "executed"}],
                     "stop_reason": "sequence_completed", "verification": {"jaws_hold": True}}}


def _grasp(observation_id: str = "obs-3") -> dict:
    return {"schema": "ora-skill-sequence-action.v1", "action": "execute_skill_sequence", "action_id": "action-3",
            "observation_id": observation_id, "bound_observation_sha256": "a" * 64,
            "arguments": {"skill": "grasp_execute", "candidate_id": "cand-1", "rationale": "grasp the cube",
                          "stop_before_commit": False, "sequence": ["open", "close", "lift"], "resume_from": 0}}


class Executor:
    def __init__(self, outcome):
        self.outcome, self.calls = outcome, []

    def native_actions_per_action(self) -> int:
        return 1

    def execute_action(self, action, budget):
        self.calls.append(action["action_id"])
        if isinstance(self.outcome, Exception):
            raise self.outcome
        time.sleep(0.05)  # the stage loop
        return self.outcome


def _session(tmp_path: Path, executor: Executor, action: dict, *, observation_receipts: bool = False) -> PiCodeMode:
    control = tmp_path / "control"
    control.mkdir()
    (control / "submitted-action.json").write_text(json.dumps(action))
    (control / "current-observation.json").write_text(json.dumps({"observation_id": action["observation_id"],
                                                                   "open": False}))
    session = PiCodeMode()
    session.run_root = tmp_path
    session.tool_action_executor = executor
    session.toolkit_observation_receipts = observation_receipts
    session.pending_receipt_body = None
    host = SimpleNamespace(control_root=control, surface=SimpleNamespace(
        supports=lambda capability: capability == "prepared_grasp_execution_v2",
        view_config={"automatic_actual_views": ["head_camera", "left_camera"]}))
    host.build_skill_receipt = lambda **kwargs: skill_driver.build_skill_receipt(host, **kwargs)
    session.tools = host
    return session


def _post_close_receipt(tmp_path: Path) -> dict:
    """A grasp's raw receipt binding its post-close baseline observation, as the stage loop records it."""

    root = tmp_path / "observations" / "obs-close"
    root.mkdir(parents=True)
    cameras = {}
    for index, camera in enumerate(("head_camera", "left_camera")):
        Image.new("RGB", (4, 4), (index * 40, 10, 10)).save(root / f"{camera}.png")
        digest = hashlib.sha256((root / f"{camera}.png").read_bytes()).hexdigest()
        cameras[camera] = {"files": {"rgb": {"path": f"{camera}.png", "sha256": digest}}}
    (root / "observation.json").write_text(json.dumps({"observation_id": "obs-close", "cameras": cameras}))
    return {"schema": SEQUENCE_RECEIPT, "action_id": "action-3", "skill": "grasp_execute",
            "committed_sequence": ["open", "close", "lift"],
            "skill_receipt": {"observation_evidence": [{
                "arm": "left", "role": "post_close_before_lift", "observation_root": str(root),
                "observation_id": "obs-close",
                "observation_sha256": hashlib.sha256((root / "observation.json").read_bytes()).hexdigest()}]}}


@pytest.mark.parametrize("observation_receipts", [False, True])
def test_a_script_s_grasp_brings_its_post_close_views_and_its_stage_loop_time(tmp_path, observation_receipts):
    """Tool-call mode shows the post-close views with the next observation; where code mode's does not
    (``toolkit_observation_receipts`` false, RoboTwin), they come back as the grasp call's images."""

    session = _session(tmp_path, Executor({"receipt": STAGED, "remaining_budget": 11}), _grasp(),
                       observation_receipts=observation_receipts)
    session.pending_receipt_body = _post_close_receipt(tmp_path)
    result = session._code_mode_action()
    assert result["receipt"] == STAGED and result["sequence"] is True
    assert result["uncounted_s"] >= 0.05, "the stage loop's time is the Supervisor's"
    if observation_receipts:
        assert result["images"] == [], "the next observation shows them"
        return
    assert [Path(item["path"]).name.split("-")[0] for item in result["images"]] == ["00", "01"]
    for item in result["images"]:
        path = Path(item["path"])
        assert path.is_relative_to(tmp_path / "native/inputs/skill-observations")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_a_primitive_is_no_skill_sequence(tmp_path):
    primitive = {"schema": "ora-motion-action.v3", "action": "wait", "arguments": {"ticks": 20},
                 "action_id": "action-3", "observation_id": "obs-3", "bound_observation_sha256": "a" * 64}
    session = _session(tmp_path, Executor({"receipt": {"status": "executed"}, "remaining_budget": 11}), primitive)
    assert session._code_mode_action() == {"receipt": {"status": "executed"}, "remaining_budget": 11}


@pytest.mark.parametrize("refusal", ["not_started", "refused"])
def test_a_grasp_refused_before_it_starts_releases_its_skill_ledger(tmp_path, refusal):
    """Otherwise the ledger holds the skill as running and every later grasp_execute is refused
    (skill_sequence_active): the proposal is gone and the observation is reopened, so nothing reconciles it."""

    outcome = (ActionNotStarted("toolkit execution accepts one primitive action, not a composite")
               if refusal == "not_started" else
               {"receipt": {"schema": "ora.tool-action-refusal.v1", "action_id": "action-3", "status": "refused",
                            "reason": "robot_decision_budget", "native_actions": 0}, "remaining_budget": 12})
    session = _session(tmp_path, Executor(outcome), _grasp())
    skill_driver.store_ledger(session.tools, skill_ledger.begin(
        skill="grasp_execute", candidate_id="cand-1", rationale="grasp the cube", observation_id="obs-3",
        observation_sha256="a" * 64))
    assert skill_driver.reconcile(session.tools)["phase"] == "running"
    result = session._code_mode_action()
    assert result["action_error"]["code"] == ("action_not_started" if refusal == "not_started"
                                              else "robot_decision_budget")
    control = session.tools.control_root
    assert not (control / "submitted-action.json").exists()
    assert json.loads((control / "current-observation.json").read_text())["open"] is True
    assert skill_driver.load_ledger(session.tools)["phase"] == "aborted", "a later grasp_execute may commit"


def test_only_the_supervisor_s_refusal_withdraws_an_executed_action(tmp_path):
    """The refusal is the Supervisor's receipt (``ora.tool-action-refusal.v1``), not any projected receipt
    whose own status reads ``refused``: an executed action keeps its proposal and its closed observation."""

    primitive = {"schema": "ora-motion-action.v3", "action": "wait", "arguments": {"ticks": 20},
                 "action_id": "action-3", "observation_id": "obs-3", "bound_observation_sha256": "a" * 64}
    receipt = {"action": "run_python", "action_id": "action-3", "status": "refused", "native_actions": 1}
    session = _session(tmp_path, Executor({"receipt": receipt, "remaining_budget": 11}), primitive)
    result = session._code_mode_action()
    assert "action_error" not in result and result["receipt"] == receipt
    control = session.tools.control_root
    assert (control / "submitted-action.json").exists()
    assert json.loads((control / "current-observation.json").read_text())["open"] is False


def test_a_failure_after_dispatch_is_not_reported_as_not_started(tmp_path):
    """Only ActionNotStarted proves nothing reached the world; any other error interrupts the parent."""

    error = ValueError("worker consumption exceeds the admitted action budget")
    session = _session(tmp_path, Executor(error), _grasp())
    with pytest.raises(ValueError, match="worker consumption"):
        session._code_mode_action()
    control = session.tools.control_root
    assert (control / "submitted-action.json").exists(), "the proposal is evidence of an uncertain action"
    assert json.loads((control / "current-observation.json").read_text())["open"] is False


def test_a_grasp_s_call_entry_carries_its_staged_receipt_and_its_time_is_uncounted():
    """The model's list of calls carries the staged receipt even when the script never prints it."""

    class Session(PiCodeMode):
        code_mode = {"max_calls_per_script": 24}
        _script_names: dict = {}

        def _project(self, name, result, *, call_id, step):
            return {"visible": dict(result)}

        def _script_call(self, call, *, ids, origin, preset):
            ledger["actions"].append({"action_id": "action-3"})
            return {"ok": True, "call_id": ids[1], "observation_step": 3, "result": {"status": "submitted"},
                    "images": [{"path": "/run/native/inputs/skill-observations/x/00-y.png"}],
                    "receipt": STAGED, "remaining_budget": 11, "sequence": True, "uncounted_s": 170.0}

    session = Session()
    session.gateway = SimpleNamespace(arguments=lambda name, arguments: arguments,
                                      classify_effect=lambda name, arguments: "mutation")
    ledger: dict = {"actions": []}
    state = {"seq": 0, "actions": 0, "calls": 0, "cap": 12, "log": [], "frames": [], "uncounted_s": 0.0,
             "parent": SimpleNamespace(log=SimpleNamespace(value=ledger))}
    answer = session._code_mode_call("grasp_execute", {"candidate_id": "cand-1", "rationale": "grasp the cube"},
                                     model_call="model", parent="call", state=state)
    assert answer["receipt"] == STAGED and answer["action"] == 1
    assert state["log"] == [{"call": 1, "tool": "grasp_execute", "ok": True, "images": 1, "action": 1,
                             "receipt": STAGED}]
    assert state["uncounted_s"] == 170.0 and state["frames"] == ["/run/native/inputs/skill-observations/x/00-y.png"]


def _move_session(receipt, *, seconds=0.0):
    """The state of a script after its ``move_pose``, whose action the parent executed and published with ``receipt``."""

    class Session(PiCodeMode):
        code_mode = {"max_calls_per_script": 24}
        _script_names: dict = {}

        def _project(self, name, result, *, call_id, step):
            return {"visible": dict(result)}

        def _script_call(self, call, *, ids, origin, preset):
            time.sleep(seconds)
            ledger["actions"].append({"action_id": "action-4"})
            return {"ok": True, "call_id": ids[1], "observation_step": 4, "result": {"status": "submitted"},
                    "receipt": receipt, "remaining_budget": 11}

    session = Session()
    session.gateway = SimpleNamespace(arguments=lambda name, arguments: arguments,
                                      classify_effect=lambda name, arguments: "mutation")
    ledger: dict = {"actions": []}
    state = {"seq": 0, "actions": 0, "calls": 0, "cap": 12, "log": [], "frames": [], "uncounted_s": 0.0,
             "parent": SimpleNamespace(log=SimpleNamespace(value=ledger))}
    session._code_mode_call("move_pose", {"targets": {}}, model_call="model", parent="call", state=state)
    return state


def test_the_time_a_move_takes_is_not_the_script_s_wall_time():
    assert _move_session({"status": "reached"}, seconds=0.05)["uncounted_s"] >= 0.05


def test_a_refused_move_s_call_entry_carries_its_line_with_the_planner_s_words():
    """A script that prints only a code still shows the model why the planner refused and what near it plans."""
    line = ("right arm 240 mm from target, planner failed, TCP moved 0.0 mm. With the fingertips at the same point, "
            "an approach tilted 10 degrees plans: (0.17, 0.00, -0.98) in world coordinates instead of (0.00, 0.00, -1.00).")
    refused = {"action": "move_pose", "status": "planner_failed", "failure_code": "planner_failed",
               "action_summary": line, "detail": {"target_arms": ["right"]}}
    assert _move_session(refused)["log"] == [{"call": 1, "tool": "move_pose", "ok": True, "action": 1, "summary": line}]
    reached = {**refused, "status": "reached", "failure_code": None, "action_summary": "right arm 2.0 mm from target, reached"}
    assert "summary" not in _move_session(reached)["log"][0]
