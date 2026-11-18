"""``explain`` in the policy session (``ora/agent/pi/code_mode.py``, ``library: extended``).

The session collects the stage events of each script call, keeps the call's answer, serves the read-only ``explain`` op
by call id, and gives a failed call of a tool that has a template its explanation by default: in the answer the script
reads (``explain``, with the call's ``call_id``) and in its entry of the script result's ``calls``. A ``base`` contract
collects nothing, adds nothing to an answer and serves no such op.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ora.agent.code_mode import docs
from ora.agent.code_mode.contract import resolve_contract
from ora.agent.pi import code_mode as session_module
from ora.agent.pi.code_mode import PiCodeMode
from ora.agent.toolkit import explain
from ora.agent.toolkit.interface import stages

ROOT = Path(__file__).resolve().parents[4]
ANSWERS = json.loads((ROOT / "tests/data/code_mode/perception-answers.json").read_text(encoding="utf-8"))["answers"]


def _recorded(tool: str, *, refused: bool = False, **where: Any) -> dict[str, Any]:
    for item in ANSWERS:
        if item["tool"] == tool and ("error" in item["result"]) == refused and all(
                (key in item["result"]) == (value is not None) for key, value in where.items()):
            return item
    raise LookupError((tool, refused, where))


class Gateway:
    def classify_effect(self, name: str, arguments: Any) -> str:
        return "read_only"

    def arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return arguments

    def definitions(self) -> list[dict[str, Any]]:
        return []


class Session(PiCodeMode):
    """Script calls answered from recorded results, each with the stage events its tool would record."""

    def __init__(self, contract: dict[str, Any], root: Path, answers: dict[str, dict[str, Any]],
                 events: dict[str, list[tuple[str, dict[str, Any]]]]) -> None:
        self.code_mode, self.run_root, self.gateway = contract, root, Gateway()
        self._script_names = {name: name for name in answers}
        self.answers, self.events, self.active_inside = answers, events, []
        self.observation_id: str | None = None  # none: nothing is kept for reuse (test_code_mode_reuse.py has that)
        self.artifacts, self.recorded = SimpleNamespace(put_json=lambda value, kind: {"kind": kind}), []

    def _current_observation_id(self) -> str | None:
        return self.observation_id

    def _script_call(self, call, *, ids, origin, preset, result_fields=None):
        self.active_inside.append(stages.active())
        for name, fields in self.events.get(call.name, ()):
            stages.stage(name, **fields)  # what the tool's own code does
        result = self.answers[call.name]
        outcome = {"ok": "error" not in result, "call_id": ids[1], "tool": call.name, "result": result,
                   "observation_step": 0, "images": []}
        if result_fields is not None:  # what the call's tool_result records besides its own fields
            self.recorded.append(result_fields(outcome))
        return outcome

    def _project(self, name, result, *, call_id, step):
        return {"visible": result}


def _session(tmp_path: Path, tools: list[str], library: str | None = "extended", **events: Any) -> Session:
    items = {tool: _recorded(tool, candidates=True) if tool == "vision_ground" else _recorded(tool, points=True)
             if tool == "geometry_query" else _recorded(tool) for tool in tools}
    contract = {**resolve_contract(True),
                **({"library": library, "script_reads": list(docs.READS)} if library else {})}
    session = Session(contract, tmp_path, {tool: item["result"] for tool, item in items.items()}, events)
    session.items = items
    return session


def _call(session: Session, name: str, arguments: dict[str, Any] | None = None, seq: int = 1) -> tuple[dict, dict]:
    state = {"seq": seq - 1, "actions": 0, "calls": 0, "cap": 4, "log": [], "frames": [], "uncounted_s": 0.0}
    answer = session._code_mode_call(name, arguments if arguments is not None else session.items[name]["arguments"],
                                     model_call="model-1", parent="call-1", state=state)
    return answer, state["log"][-1]


def test_a_call_that_worked_is_remembered_and_explained_on_request(tmp_path: Path) -> None:
    session = _session(tmp_path, ["geometry_query"], geometry_query=[("depth_window", {"window_px": 2})])
    answer, entry = _call(session, "geometry_query")
    assert answer["call_id"] == "call-1#001" and "explain" not in answer and "explain" not in entry
    assert session.active_inside == [True], "the session collects the tool's stage events around the call"
    reply = session._code_mode_op({"op": "explain", "call_id": "call-1#001"})
    arguments = session.items["geometry_query"]["arguments"]
    expected = explain.render("geometry_query", arguments=arguments, result=session.items["geometry_query"]["result"],
                              events=[{"stage": "depth_window", "fields": {"window_px": 2}}])
    assert reply == {"ok": True, "call_id": "call-1#001", "text": expected.text}
    assert "2. depth window: each pixel takes the median of its valid depths within 2 px" in reply["text"]
    assert not stages.active()


def test_a_failed_call_carries_its_explanation_by_default(tmp_path: Path) -> None:
    refused = _recorded("geometry_query", refused=True)
    session = _session(tmp_path, [])
    session.answers["geometry_query"], session._script_names["geometry_query"] = refused["result"], "geometry_query"
    answer, entry = _call(session, "geometry_query", refused["arguments"])
    assert answer["ok"] is False and answer["call_id"] == "call-1#001"
    expected = explain.render("geometry_query", arguments=refused["arguments"], result=refused["result"]).text
    assert answer["explain"] == entry["explain"] == expected.rstrip()
    assert entry["error"] == refused["result"]["error"]["code"]
    # the same text when asked for
    assert session._code_mode_op({"op": "explain", "call_id": "call-1#001"})["text"] == expected


def test_a_result_that_is_ok_but_empty_is_a_failed_call_of_its_template(tmp_path: Path) -> None:
    base = _recorded("vision_ground", candidates=True)
    empty = {**base["result"], "status": "no_candidate", "candidates": []}
    session = _session(tmp_path, [])
    session.answers["vision_ground"], session._script_names["vision_ground"] = empty, "vision_ground"
    answer, entry = _call(session, "vision_ground", base["arguments"])
    assert answer["ok"] is True and answer["explain"].startswith("vision_ground(") and "no candidate" in answer["explain"]
    assert entry["explain"] == answer["explain"]


def test_an_explanation_attached_to_a_result_is_bounded(tmp_path: Path, monkeypatch) -> None:
    refused = _recorded("geometry_query", refused=True)
    session = _session(tmp_path, [])
    session.answers["geometry_query"], session._script_names["geometry_query"] = refused["result"], "geometry_query"
    monkeypatch.setattr(explain, "ATTACHED_CHARS", 120)
    answer, _entry = _call(session, "geometry_query", refused["arguments"])
    assert len(answer["explain"]) < 220 and "characters cut" in answer["explain"]
    full = session._code_mode_op({"op": "explain", "call_id": "call-1#001"})["text"]
    assert len(full) > 220, "a request is answered whole"


def test_the_calls_of_a_result_carry_a_bounded_share_of_the_explanations(tmp_path: Path, monkeypatch) -> None:
    refused = _recorded("geometry_query", refused=True)
    session = _session(tmp_path, [])
    session.answers["geometry_query"], session._script_names["geometry_query"] = refused["result"], "geometry_query"
    size = len(explain.render("geometry_query", arguments=refused["arguments"], result=refused["result"]).text.rstrip())
    monkeypatch.setattr(explain, "SCRIPT_CHARS", 2 * size + 10)
    state = {"seq": 0, "actions": 0, "calls": 0, "cap": 4, "log": [], "frames": [], "uncounted_s": 0.0}
    answers = [session._code_mode_call("geometry_query", refused["arguments"], model_call="m", parent="c", state=state)
               for _ in range(3)]
    assert all(answer["explain"] for answer in answers), "a script always reads its own call's explanation"
    assert ["explain" in entry for entry in state["log"]] == [True, True, False]


def test_a_tool_without_a_template_is_remembered_but_has_nothing_to_explain(tmp_path: Path) -> None:
    session = _session(tmp_path, [])
    session.answers["set_gripper"], session._script_names["set_gripper"] = {"error": {"code": "ik_failed"}}, "set_gripper"
    answer, entry = _call(session, "set_gripper", {"targets": {}})
    assert answer["call_id"] == "call-1#001" and "explain" not in answer and "explain" not in entry
    reply = session._code_mode_op({"op": "explain", "call_id": "call-1#001"})
    assert reply["ok"] is False and reply["error"]["code"] == "no_explanation"
    assert reply["error"]["constraints"]["allowed"] == list(explain.tools())
    assert f"explain covers {', '.join(explain.tools())}" in reply["error"]["message"]
    assert explain.tools() == ("geometry_query", "grasp_plan", "move_pose", "move_to", "recover_pose", "return_home",
                               "rotate_pitch", "rotate_wrist", "vision_ground", "vision_segment")


def test_an_unknown_call_is_refused_and_only_the_latest_calls_are_kept(tmp_path: Path, monkeypatch) -> None:
    session = _session(tmp_path, ["geometry_query"])
    assert session._code_mode_op({"op": "explain", "call_id": "call-1#001"})["error"]["code"] == "unknown_call"
    assert session._code_mode_op({"op": "explain"})["error"]["code"] == "unknown_call"
    monkeypatch.setattr(session_module, "EXPLAIN_CALLS", 3)
    for seq in range(1, 6):
        _call(session, "geometry_query", seq=seq)
    kept = [session._code_mode_op({"op": "explain", "call_id": f"call-1#{seq:03d}"})["ok"] for seq in range(1, 6)]
    assert kept == [False, False, True, True, True]
    assert session._code_mode_op({"op": "elsewhere"})["error"]["code"] == "op_unavailable"


def test_a_base_contract_collects_nothing_adds_nothing_and_serves_no_op(tmp_path: Path) -> None:
    session = _session(tmp_path, [], library=None)
    refused = _recorded("geometry_query", refused=True)
    session.answers["geometry_query"], session._script_names["geometry_query"] = refused["result"], "geometry_query"
    session.events["geometry_query"] = [("depth_window", {"window_px": 2})]
    answer, entry = _call(session, "geometry_query", refused["arguments"])
    assert session.active_inside == [False] and session._code_mode_calls is None
    assert set(answer) == {"ok", "tool", "result", "error"} and set(entry) == {"call", "tool", "ok", "error"}
    assert session._code_mode_ops() == {} and not stages.active()


def test_the_documentation_lists_the_explain_stages_of_the_tools_that_have_them(tmp_path: Path) -> None:
    definitions, results = [], {}
    for tool in (*explain.tools(), "observe"):
        path = ROOT / "ora/agent/toolkit/tools" / tool / "description.json"
        description = json.loads(path.read_text(encoding="utf-8"))
        definitions.append({"name": tool, "description": description["description"], "inputSchema": description["inputSchema"]})
        results[tool] = description["resultSchema"]
    session = _session(tmp_path, [])
    session.tools = SimpleNamespace(surface=SimpleNamespace(lifecycle_tools=(), result_schemas=results),
                                    script_definitions=lambda: definitions)
    index, entries = session._code_mode_docs()
    assert "explain" not in entries["observe"], "no template, no part"
    for tool in explain.tools():
        assert list(entries[tool]) == [part for part in docs.PARTS
                                       if part != "examples" or docs.examples(tool, reads=docs.READS)
                                       if part != "dry_run" or docs.accepts_dry_run(tool)]
        assert entries[tool]["explain"] == explain.documentation(explain.template(tool))
        assert entries[tool]["explain"].startswith("Explain (robot.explain(r); a failed call carries it as r.explain)")
    assert not [line for line in index.splitlines() if line.startswith("- tools.") and "robot.explain" in line]
    reply = session._code_mode_op({"op": "docs", "name": "grasp_plan", "part": "explain"})
    assert reply["ok"] and "1. request:" in reply["text"] and "- display_cap(compiled, shown)" in reply["text"]


def test_a_reused_answer_explains_the_call_it_was_kept_from(tmp_path: Path) -> None:
    session = _session(tmp_path, ["geometry_query"], geometry_query=[("depth_window", {"window_px": 2})])
    session.observation_id = "observation-0001"
    first, _ = _call(session, "geometry_query")
    again, entry = _call(session, "geometry_query", seq=2)
    assert again["reused"] is True and entry.get("reused") is True and again["call_id"] == first["call_id"]
    assert session.active_inside == [True], "the second look ran no tool, so no stage events were collected"
    reply = session._code_mode_op({"op": "explain", "call_id": again["call_id"]})
    assert reply["ok"] and "2. depth window" in reply["text"]
