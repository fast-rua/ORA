"""Reused perception results (``ora/agent/pi/code_mode.py``, ``ora/agent/code_mode/reuse.py``, ``library: extended``).

The same perception call with the same arguments on the same observation returns the answer kept from the first,
marked ``reused``; no tool runs, no call limit counts it, and a new observation starts afresh. A world-changing
call, a refused or failed one, and the read-only tools that persist something, plan or return images are not
kept. A base contract keeps nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ora.agent.code_mode.contract import resolve_contract
from ora.agent.code_mode.reuse import REUSABLE, KeptResults
from ora.agent.pi.code_mode import PiCodeMode

MUTATIONS = {"move_pose"}


class Gateway:
    def arguments(self, name, arguments):
        return {key: value for key, value in arguments.items() if value is not None}  # null means omitted

    def classify_effect(self, name, arguments):
        return "mutation" if name in MUTATIONS else "read_only"


class Session(PiCodeMode):
    """The policy session's script path with the robot socket's ``_script_call`` stood in for: it records which
    tools ran, answers each with the number of calls so far, and publishes a new observation for an action."""

    def __init__(self, library="extended", **limits) -> None:
        contract = resolve_contract(limits or True)
        self.code_mode = {**contract, **({"library": library} if library else {})}
        # Every tool a program may name, as the robot socket lists them (an unlisted name is refused unknown_tool).
        self.gateway, self._script_names = Gateway(), {"vision_ground": "vision.ground", **{name: name for name in (
            "geometry_query", "vision_segment", "mark_point", "grasp_plan", "align_plan", "observe", "move_pose")}}
        self.observation, self.ran, self.fail = "observation-1", [], set()
        self.parent = SimpleNamespace(log=SimpleNamespace(value={"actions": []}))
        self.artifacts = SimpleNamespace(put_json=lambda value, kind: {"kind": kind})

    def _current_observation_id(self):
        return self.observation

    def _project(self, name, result, *, call_id, step):
        return {"visible": dict(result)}

    def _script_call(self, call, *, ids, origin, preset=None, result_fields=None):
        outcome = self._outcome(call, ids, preset)
        if result_fields is not None:  # the explain inputs the real one records with the call's result
            result_fields(outcome)
        return outcome

    def _outcome(self, call, ids, preset):
        if preset is not None:
            return {"ok": False, "call_id": ids[1], "result": {"error": {"code": preset[1].code}}, "images": []}
        self.ran.append((call.name, dict(call.arguments)))
        if call.name in self.fail:
            self.fail.discard(call.name)
            return {"ok": False, "call_id": ids[1], "result": {"error": {"code": "service_unavailable"}}, "images": []}
        if call.name in MUTATIONS:
            self.parent.log.value["actions"].append({"refusal": None})
            self.observation = f"observation-{len(self.ran) + 1}"
        return {"ok": True, "call_id": ids[1], "observation_step": 0, "images": [],
                "result": {"echo": dict(call.arguments), "ran": len(self.ran), "candidates": [{"target_ref": "t"}]}}

    def script(self, *calls, state=None):
        """The answers to ``calls`` ``(name, arguments)`` made by one script; its state."""

        state = state or {"seq": 0, "actions": 0, "calls": 0, "cap": 12, "log": [], "frames": [], "uncounted_s": 0.0,
                          "parent": self.parent}
        return [self._code_mode_call(name, arguments, model_call="m", parent="p", state=state)
                for name, arguments in calls], state


LOOK = ("vision_ground", {"camera": "head", "text": "cup"})


def test_a_repeat_on_the_same_observation_returns_the_kept_answer_marked_reused() -> None:
    session = Session()
    (first, second, third), state = session.script(LOOK, LOOK, LOOK)
    assert "reused" not in first and second["reused"] is True and third["reused"] is True
    assert {key: value for key, value in second.items() if key != "reused"} == first
    assert [name for name, _ in session.ran] == ["vision_ground"], "the tool ran once"
    assert state["calls"] == 1 and state["seq"] == 3, "a reused answer counts against no limit"
    assert state["log"] == [{"call": 1, "tool": "vision.ground", "ok": True},
                            {"call": 2, "tool": "vision.ground", "ok": True, "reused": True},
                            {"call": 3, "tool": "vision.ground", "ok": True, "reused": True}]
    second["result"]["echo"]["text"] = "changed by the script"
    assert session.script(LOOK)[0][0]["result"]["echo"]["text"] == "cup", "the script cannot change what is kept"


def test_other_arguments_other_tools_and_null_as_omitted() -> None:
    session = Session()
    answers, _ = session.script(LOOK, ("vision_ground", {"camera": "head", "text": "cup", "max_candidates": 1}),
                                ("geometry_query", {"camera": "head", "points": [[1, 2]]}),
                                ("geometry_query", {"camera": "head", "points": [[1, 2]], "step": None}),
                                ("geometry_query", {"camera": "head", "points": [[1, 3]]}))
    assert [bool(answer.get("reused")) for answer in answers] == [False, False, False, True, False]
    assert len(session.ran) == 4


@pytest.mark.parametrize("name", ["mark_point", "grasp_plan", "align_plan", "observe"])
def test_a_read_only_tool_that_persists_plans_or_returns_images_is_not_kept(name: str) -> None:
    assert name not in REUSABLE
    session = Session()
    answers, _ = session.script((name, {"a": 1}), (name, {"a": 1}))
    assert not any(answer.get("reused") for answer in answers) and len(session.ran) == 2


def test_a_new_observation_starts_afresh_and_a_script_after_another_reuses() -> None:
    session = Session()
    session.script(LOOK)
    (again,), _ = session.script(LOOK)  # a second script on the same observation
    assert again["reused"] is True and len(session.ran) == 1
    session.observation = "observation-2"
    (fresh, repeat), _ = session.script(LOOK, LOOK)
    assert "reused" not in fresh and repeat["reused"] is True and len(session.ran) == 2
    session.observation = "observation-1"  # an old observation is not kept once the session moved on
    assert "reused" not in session.script(LOOK)[0][0]


def test_an_action_publishes_a_new_observation_so_the_next_look_runs_again() -> None:
    session = Session()
    answers, state = session.script(LOOK, ("move_pose", {"targets": {}}), LOOK, LOOK)
    assert [bool(answer.get("reused")) for answer in answers] == [False, False, False, True]
    assert [name for name, _ in session.ran] == ["vision_ground", "move_pose", "vision_ground"]
    assert state["actions"] == 1 and "action" in answers[1]
    assert state["log"][-1]["reused"] is True and "action" not in state["log"][-1]


def test_a_reused_answer_is_given_at_the_call_limit_and_a_new_look_is_refused() -> None:
    session = Session(max_calls_per_script=2)
    other = ("geometry_query", {"camera": "head", "points": [[1, 2]]})
    (a, b, repeat, reuse, refused), state = session.script(
        LOOK, other, ("geometry_query", {"camera": "head", "points": [[9, 9]]}), LOOK, other)
    assert repeat["error"]["code"] == "script_call_limit"
    assert state["calls"] == 3, "two looks and the refused one count (as ever); the two reused answers do not"
    assert reuse["reused"] is True and refused["reused"] is True, "kept answers need no call"
    assert not a.get("reused") and not b.get("reused")


def test_a_refused_or_failed_call_is_not_kept() -> None:
    session = Session()
    session.fail.add("vision_ground")
    first, second, third = session.script(LOOK, LOOK, LOOK)[0]
    assert first["ok"] is False and "reused" not in second and third["reused"] is True
    assert len(session.ran) == 2, "the failure was asked again, the success kept"


@pytest.mark.parametrize("library", [None, "base"])
def test_a_base_contract_keeps_and_marks_nothing(library) -> None:
    session = Session(library=library)
    answers, state = session.script(LOOK, LOOK)
    assert not any("reused" in answer for answer in answers) and len(session.ran) == 2
    assert state["calls"] == 2 and all("reused" not in entry for entry in state["log"])
    assert session._code_mode_kept is None


def test_a_legacy_tool_s_dotted_name_is_the_same_tool() -> None:
    kept = KeptResults()
    kept.keep("o", "vision.ground", {"a": 1}, {"ok": True, "tool": "vision.ground"})
    assert kept.get("o", "vision_ground", {"a": 1}) == {"ok": True, "tool": "vision.ground", "reused": True}
    assert kept.get("o", "vision.ground", {"a": 2}) is None and kept.get("p", "vision.ground", {"a": 1}) is None
    assert kept.get(None, "vision.ground", {"a": 1}) is None
    kept.keep(None, "geometry_query", {}, {"ok": True})
    kept.keep("o", "geometry_query", {}, {"ok": True, "error": {"code": "x"}})
    assert kept.get("o", "geometry_query", {}) is None
