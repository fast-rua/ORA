"""Code mode in a benchmark episode (plan ``policy.code_mode``) on the Pi core.

Synthetic episodes whose scripted model answers with a script call. The script
runs as one parent invocation of the Supervisor's executor: every action it
causes is recorded as an ``action_proposed`` of the script's own call and in its
``ora.tool-execution.v1`` ledger, which the agent core ends the turn on; the
script's own tool calls are recorded under the reply's model call as
``<call>#NNN`` with origin ``code_mode``. A refusal raises in the script and the
turn continues, and one of a tool the model has (the completion check, finishing)
says to call it directly; after the trusted terminal boundary
the script tool itself is refused (``episode_terminal``), naming the direct tools
left, where tool-call mode keeps its own refusal; an action the Supervisor refused
leaves the observation to the script's next action; ``handoff`` hands back after an
action; a coding agent writes, runs, edits and reads; with ``library: extended`` a script looks its functions'
documentation up (``robot.docs``, ``help``) and asks what a call did (``robot.explain``) without a tool call.
The command runs without bubblewrap here (through the real ``robot`` library and
protocol); ``tests/ora/agent/code_mode/test_sandbox.py`` covers the sandbox itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from tests.support.agent_core import episodes
from tests.support.agent_core.code_mode_shell import local_run_command

pytestmark = pytest.mark.usefixtures("pi_core")


def finish() -> dict[str, Any]:
    return episodes.tool_reply(("f0", "finish_episode", {
        "outcome": "failure", "reason": "scripted end", "attempts": ["scripted"]}))


def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *replies: dict[str, Any], code_mode: Any = True,
        terminal_actions: int | None = None) -> dict[str, Any]:
    """The episode in code mode (``code_mode`` None: tool-call mode); with ``terminal_actions`` the Synthetic world
    reaches its trusted terminal boundary on that action."""

    import ora.agent.pi.code_mode as session
    from ora.agent.providers.fake import FakeProvider

    def configure(plan: dict[str, Any]) -> dict[str, str]:
        plan["runtime"]["terminal_actions"] = terminal_actions
        return {}

    monkeypatch.setattr(session, "run_command", local_run_command)
    return episodes.run_core("pi", tmp_path, provider=FakeProvider(responses=[*replies, finish()]), actions=4,
                             configure=configure if terminal_actions is not None else None,
                             policy_overrides={} if code_mode is None else {"code_mode": code_mode})


def program(code: str) -> tuple[str, dict[str, Any]]:
    """The ``bash`` call that runs ``code`` with the ``robot`` library's names imported."""

    return "bash", {"command": f"python - <<'EOF'\nimport robot\nfrom robot import tools, handoff, ToolRefused\n"
                               f"{code.rstrip()}\nEOF"}


def script(call_id: str, code: str) -> dict[str, Any]:
    return episodes.tool_reply((call_id, *program(code)))


def result(run_: Mapping[str, Any], call_id: str) -> dict[str, Any]:
    (event,) = [item for item in run_["events"] if item["type"] == "tool_result" and item["tool_call_id"] == call_id]
    return json.loads((run_["root"] / event["artifact_ref"]["path"]).read_text(encoding="utf-8"))


def execution(run_: Mapping[str, Any], call_id: str) -> dict[str, Any] | None:
    (event,) = [item for item in run_["events"] if item["type"] == "tool_result" and item["tool_call_id"] == call_id]
    return event["payload"].get("execution")


def read_by_model(run_: Mapping[str, Any], call_id: str) -> dict[str, Any]:
    """A call's result as the model read it: its tool message in the last request (the finishing reply's)."""

    (message,) = [message for message in json.loads(run_["requests"][-1])["messages"]
                  if message.get("role") == "tool" and message["tool_call_id"] == call_id]
    return json.loads(message["content"]) if isinstance(message["content"], str) else message["content"]


def test_a_script_acts_through_one_parent_invocation_and_ends_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_ = run(tmp_path, monkeypatch, script("g0", "first = tools.close_gripper()\nsecond = tools.open_gripper()\n"
                                                    "print(first['action'], second['action'])\n"
                                                    "tools.observe(views=['agentview'])\n"))
    assert run_["evaluation"]["status"] == "valid", run_["evaluation"]["reasons"]
    events = run_["events"]
    nested = [event["tool_call_id"] for event in events if event["type"] == "tool_call_proposed"
              and (event["payload"].get("origin") or {}).get("kind") == "code_mode"]
    assert nested == ["g0#001", "g0#002", "g0#003"]
    actions = [event for event in events if event["type"] == "action_proposed"]
    assert [event["tool_call_id"] for event in actions] == ["g0", "g0"], "actions belong to the script's call"
    ledger = execution(run_, "g0")
    assert ledger["schema"] == "ora.tool-execution.v1" and len(ledger["actions"]) == 2
    assert ledger["native_actions_used"] == 2
    outcome = result(run_, "g0")
    assert outcome["status"] == "completed" and outcome["actions_executed"] == 2 and outcome["output"] == "1 2\n"
    # One robot decision per parent call that acted, as for a synchronous toolkit tool's primitives.
    assert run_["result"]["decisions"] == 1


def test_a_refused_call_raises_in_the_script_and_the_turn_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_ = run(tmp_path, monkeypatch, script("g0", "tools.check_task()\ntools.close_gripper()"))
    assert run_["evaluation"]["status"] == "valid", run_["evaluation"]["reasons"]
    assert not [event for event in run_["events"] if event["type"] == "action_proposed"]
    outcome = result(run_, "g0")
    assert outcome["status"] == "failed" and outcome["actions_executed"] == 0
    assert outcome["calls"][0]["error"] == "script_tool_unavailable"
    # The completion check is the model's own call: the refusal says how to make it.
    assert outcome["output"].splitlines()[-1] == ("robot.ToolRefused: check_task: check_task cannot be called from a "
                                                  "script; call check_task directly, as its own tool call outside the "
                                                  "script")
    assert (execution(run_, "g0") or {}).get("native_actions_used", 0) == 0
    assert run_["result"]["termination"] == "policy_finished_episode"


#: A program that calls the completion check and finishing, printing each refusal's code and message.
REFUSED = ("for call in (tools.check_task, tools.finish_episode):\n"
           "    try:\n"
           "        call()\n"
           "    except Exception as error:\n"
           "        print(error.code, '|', error)\n")


def test_a_script_is_told_to_call_the_completion_check_and_finishing_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_ = run(tmp_path, monkeypatch, script("g0", REFUSED))
    assert run_["evaluation"]["status"] == "valid", run_["evaluation"]["reasons"]
    outcome = result(run_, "g0")
    assert [entry["error"] for entry in outcome["calls"]] == ["script_tool_unavailable", "script_lifecycle_unavailable"]
    assert outcome["output"].splitlines() == [
        "script_tool_unavailable | check_task: check_task cannot be called from a script; call check_task directly, "
        "as its own tool call outside the script",
        "script_lifecycle_unavailable | finish_episode: a lifecycle tool is the agent's own decision; a script cannot "
        "call it; call finish_episode directly, as its own tool call outside the script"]
    # Finishing is still recorded as the lifecycle call it is.
    (proposal,) = [event for event in run_["events"]
                   if event["type"] == "tool_call_proposed" and event["tool_call_id"] == "g0#002"]
    assert proposal["payload"]["effect"] == "lifecycle"
    assert run_["result"]["termination"] == "policy_finished_episode"

#: The direct tools the Synthetic surface leaves the model in code mode, in the order it is given them.
DIRECT = ["report_issue", "check_task", "finish_episode"]


def test_after_the_terminal_boundary_a_script_is_refused_naming_the_direct_tools_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_ = run(tmp_path, monkeypatch, script("g0", "tools.close_gripper()\ntools.open_gripper()"),  # its second action
               script("t0", "tools.observe(views=['agentview'])"),                                   # ends the world
               episodes.tool_reply(("c0", "check_task", {})), terminal_actions=2)
    assert run_["evaluation"]["status"] == "valid", run_["evaluation"]["reasons"]
    assert [event["tool_call_id"] for event in run_["events"] if event["type"] == "action_proposed"] == ["g0", "g0"]
    error = read_by_model(run_, "t0")["error"]
    # The code stays the core's own; the words and constraints say what is left.
    assert (error["code"], error["retryable"]) == ("episode_terminal", False)
    assert error["message"] == ("world mutation is unavailable after the trusted terminal boundary, so no script or "
                                "command runs any more (a read-only one included); call the tools left directly: "
                                + ", ".join(DIRECT))
    assert error["constraints"] == {"allowed": DIRECT}
    assert not [event for event in run_["events"] if event["type"] == "tool_call_proposed"
                and event["tool_call_id"].startswith("t0#")], "the script never ran"
    assert "error" not in read_by_model(run_, "c0"), "the completion check runs directly"


def test_tool_call_mode_keeps_its_terminal_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_ = run(tmp_path, monkeypatch, episodes.tool_reply(("m0", "close_gripper", {})),
               episodes.tool_reply(("m1", "open_gripper", {})), episodes.tool_reply(("t0", "close_gripper", {})),
               code_mode=None, terminal_actions=2)
    assert run_["evaluation"]["status"] == "valid", run_["evaluation"]["reasons"]
    error = read_by_model(run_, "t0")["error"]
    assert (error["code"], error["retryable"]) == ("episode_terminal", False)
    assert error["message"] == "world mutation is unavailable after the trusted terminal boundary"
    assert error["constraints"] == {}


def test_a_script_acts_again_after_the_supervisor_refused_its_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.orchestrator.episode import code_mode_execution

    refusals = ["stale_or_mismatched_recovery_target"]  # the executor refuses the script's first action
    monkeypatch.setattr(code_mode_execution._NoExecutionState, "recovery_refusal",
                        lambda self, action: refusals.pop(0) if refusals else None)
    run_ = run(tmp_path, monkeypatch, script("g0", "try:\n    tools.close_gripper()\nexcept ToolRefused as error:\n"
                                                    "    refused = error.code\n"
                                                    "print(refused, tools.open_gripper()['action'])"))
    assert run_["evaluation"]["status"] == "valid", run_["evaluation"]["reasons"]
    actions = [event for event in run_["events"] if event["type"] == "action_proposed"]
    assert [event["tool_call_id"] for event in actions] == ["g0", "g0"]
    assert actions[0]["payload"] == actions[1]["payload"], "one observation, one action identity"
    ledger = execution(run_, "g0")
    assert [(item["refusal"], item["native_actions_used"]) for item in ledger["actions"]] == [
        ("stale_or_mismatched_recovery_target", 0), (None, 1)]
    outcome = result(run_, "g0")
    assert outcome["status"] == "completed" and outcome["actions_executed"] == 1
    assert outcome["output"] == "stale_or_mismatched_recovery_target 1\n"
    assert run_["result"]["decisions"] == 1
    # The trace's readers take the refused proposal as not executed.
    from ora.orchestrator.evaluation.metrics import build_runtime_metrics
    from viewer.process_timeline import compile_process_timeline

    blocks = [(event["phase"], event["title"], event.get("action_id"), event["issues"])
              for event in compile_process_timeline(run_["root"])["events"]
              if event["phase"] in {"execution", "outcome", "unmatched"}]
    action_id = actions[1]["payload"]["action_id"]
    assert blocks == [("execution", "Action proposal", None, []), ("outcome", "Action refused", None, []),
                      ("execution", "set_gripper", action_id, []), ("outcome", "Measured outcome", action_id, [])]
    pairing = build_runtime_metrics(run_["root"])["latency"]["turn_complete_to_action_publish_s"]
    assert pairing["proposing_turns"] == pairing["outer_actions"] == 1


def test_a_script_that_acts_and_hands_back_ends_the_turn_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_ = run(tmp_path, monkeypatch, script("g0", "tools.close_gripper()\nhandoff('need a look', step=1)\n"
                                                    "tools.open_gripper()"))
    assert run_["evaluation"]["status"] == "valid", run_["evaluation"]["reasons"]
    outcome = result(run_, "g0")
    assert outcome["status"] == "handed_back" and outcome["ok"] is True and outcome["actions_executed"] == 1
    assert outcome["handed_back"] == {"reason": "need a look", "data": {"step": 1}}
    assert execution(run_, "g0")["native_actions_used"] == 1


def test_a_coding_agent_writes_a_module_and_runs_it_with_bash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    program = "from robot import tools\nprint('action', tools.close_gripper()['action'])\n"
    run_ = run(tmp_path, monkeypatch,
               episodes.tool_reply(("w0", "write", {"path": "grip.py", "content": program})),
               episodes.tool_reply(("b0", "bash", {"command": "python grip.py"})),
               episodes.tool_reply(("e0", "edit", {"path": "grip.py", "edits": [
                   {"oldText": "close_gripper", "newText": "open_gripper"}]})),
               episodes.tool_reply(("r0", "read", {"path": "grip.py"})))
    assert run_["evaluation"]["status"] == "valid", run_["evaluation"]["reasons"]
    assert result(run_, "w0")["written"] == ["grip.py"]
    bash = result(run_, "b0")
    assert bash["status"] == "completed" and bash["actions_executed"] == 1 and "action 1" in bash["output"]
    assert [event["tool_call_id"] for event in run_["events"] if event["type"] == "action_proposed"] == ["b0"]
    assert execution(run_, "b0")["native_actions_used"] == 1
    (entry,) = bash["workspace"]  # the program as it stood, kept in the run's evidence
    assert (run_["root"] / "native" / "code-workspace-objects" / entry["sha256"]).read_text() == program
    assert "open_gripper" in result(run_, "r0")["content"]
    assert run_["result"]["decisions"] == 1


#: A program that looks a function up every way a script can, then asks for a tool the model has itself.
LOOKUP = ("print(robot.docs('observe'))\n"
          "print('=====')\n"
          "print(robot.docs('observe', 'arguments'))\n"
          "print('=====')\n"
          "print(robot.docs())\n"
          "print('=====')\n"
          "help(tools.observe)\n"
          "print('=====')\n"
          "try:\n"
          "    robot.docs('check_task')\n"
          "except ToolRefused as error:\n"
          "    print(error.code, '|', error)\n"
          "print('=====')\n"
          "print(robot.docs('target'))\n"
          "print('=====')\n"
          "try:\n"
          "    robot.docs('Pointt')\n"
          "except ToolRefused as error:\n"
          "    print(error.code, '|', error)\n")


def test_with_the_extended_library_a_script_looks_a_function_up_without_a_tool_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``robot.docs`` is served over the script RPC by the policy session from the entries generated from the same
    definitions the description's index lists: the default lookup (this entry fits, so whole), one part, the index and
    the card; ``help`` shows the default lookup; a name on the card is looked up too; a tool the model has itself is no
    function, and an unknown name is refused with the names nearest to it. A lookup is no tool call: nothing is
    recorded or counted."""
    from ora.agent.code_mode import docs
    from tests.support.agent_core.episodes import synthetic_plan

    code_mode = {"library": "extended"}
    run_ = run(tmp_path, monkeypatch, script("g0", LOOKUP), code_mode=code_mode)
    assert run_["evaluation"]["status"] == "valid", run_["evaluation"]["reasons"]
    outcome = result(run_, "g0")
    assert outcome["status"] == "completed" and outcome["calls"] == [] and outcome["actions_executed"] == 0
    assert not [event for event in run_["events"] if event["type"] == "tool_call_proposed"
                and event["tool_call_id"].startswith("g0#")], "a lookup is no tool call"
    plan = synthetic_plan(actions=4, policy_overrides={"code_mode": code_mode})
    surface = plan["policy_composition"]["tool_surface"]
    (layer,) = [item for item in surface["derived_layers"] if item["kind"] == "code_mode_v1"]
    tool_call = synthetic_plan(actions=4)["policy_composition"]["tool_surface"]
    full = {item["name"]: item for item in tool_call["definitions"]}
    functions = [full[name] for name in layer["script_functions"]]
    entry = docs.entry(full["observe"], tool_call["result_schemas"].get("observe"))
    whole, arguments, index, shown, refused, library, near = outcome["output"].split("=====\n")
    assert whole == entry + "\n"
    assert arguments == docs.entry_parts(full["observe"])["arguments"] + "\n\n"
    assert index == docs.listing(functions, surface["result_schemas"], reads=()) + "\n"
    assert surface["definitions"][-1]["description"].endswith(index[:-1])
    assert all(line.strip() in shown for line in entry.splitlines()), "help shows the same text"
    assert refused == ("unknown_function | docs: check_task is one of your own tools, not a function; its description "
                       "is in your tool list\n")
    assert library.startswith("robot.target(position=None, *, approach=None") and "typed values" in library
    assert near.startswith("unknown_function | docs: no function 'Pointt'; the nearest: Point")


#: A program that asks what its own calls did: by result, by id (its own, one made up) and with nothing to go on.
ASKS = ("r = tools.observe(views=['agentview'])\n"
        "print(r.call_id, r.explain)\n"
        "for ask in (r, r.call_id, 'g0#999', {'call_id': r.call_id}):\n"
        "    try:\n"
        "        robot.explain(ask)\n"
        "    except ToolRefused as error:\n"
        "        print(error.code, '|', error)\n"
        "try:\n"
        "    robot.explain(3)\n"
        "except ValueError as error:\n"
        "    print('ValueError |', error)\n")


def test_with_the_extended_library_a_script_asks_what_a_call_did_without_a_tool_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``robot.explain`` is served over the script RPC by the policy session: a result carries the id of its call
    (``<call>#NNN``), a tool without a stage template (the Synthetic world's) has nothing to explain and says which
    tools do, a call the session never made is unknown, and no request is a tool call: none is counted, each is on
    the record as a nested ``robot.explain`` call of the script."""
    from ora.agent.evidence.agent_events import library_op

    run_ = run(tmp_path, monkeypatch, script("g0", ASKS), code_mode={"library": "extended"})
    assert run_["evaluation"]["status"] == "valid", run_["evaluation"]["reasons"]
    outcome = result(run_, "g0")
    assert outcome["status"] == "completed" and [entry["tool"] for entry in outcome["calls"]] == ["observe"]
    assert outcome["allowance_left"]["calls"] == 23 and "explain" not in outcome["calls"][0]
    nothing = ("no_explanation | explain: observe has no explanation yet; explain covers geometry_query, grasp_plan, "
               "move_pose, move_to, recover_pose, return_home, rotate_pitch, rotate_wrist, vision_ground, "
               "vision_segment")
    assert outcome["output"].splitlines() == [
        "g0#001 None", nothing, nothing,
        "unknown_call | explain: no script call 'g0#999' in this episode's record; explain takes the result of a "
        "call or its call_id", nothing,
        "ValueError | explain needs the result of a call (what tools.<name>(...) returned, or the .result of its "
        "ToolRefused) or a call id"]
    assert [(event["tool_call_id"], library_op(event)) for event in run_["events"]
            if event["type"] == "tool_call_proposed" and event["tool_call_id"].startswith("g0#")] == [
        ("g0#001", None), ("g0#002", "explain"), ("g0#003", "explain"), ("g0#004", "explain"),
        ("g0#005", "explain")], "each request is recorded, as no tool call"


#: A stage template for the Synthetic ``observe``, which has none, and the stage event its call notes here: what a
#: toolkit tool's own template and ``stage()`` give ``explain``.
OBSERVE_TEMPLATE = {
    "schema": "ora.toolkit-explain.v1", "tool": "observe", "events": {"looked": ["views"]},
    "call": ["{args.views|join}"],
    "outcomes": [{"if": {"has": "error"}, "text": "refused: {error.code}", "failed": True}, {"text": "looked"}],
    "stages": [{"name": "request", "doc": "the views asked for", "line": "show {args.views|count|n:view}"},
               {"name": "views", "doc": "the views rendered (stage event looked)",
                "when": {"has": "events.looked[0].views"}, "line": "rendered {events.looked[0].views|join}"}]}
#: Turn one looks, asks what the look did, reads the robot model (which the Synthetic world cannot answer), looks
#: again and acts; turn two asks again what turn one's look did and looks where no camera is (a failed call, which
#: carries its explanation).
LOOK_ASK_ACT = ("r = tools.observe(views=['agentview'])\n"
                "print(robot.explain(r), end='')\n"
                "try:\n    robot.query('robot_state')\nexcept ToolRefused as error:\n    print(error.code)\n"
                "print(tools.observe(views=['agentview']).call_id)\n"
                "tools.close_gripper()\n")
ASK_AGAIN = ("print(robot.explain('g0#001'), end='')\n"
             "try:\n    tools.observe(views=['nowhere'])\nexcept ToolRefused:\n    pass\n")


def _staged_observe(monkeypatch: pytest.MonkeyPatch) -> None:
    from ora.agent.code_mode.runtime import CodeModeToolRuntime
    from ora.agent.toolkit import explain as toolkit_explain
    from ora.agent.toolkit.interface import stages

    template, invoke = toolkit_explain.template, CodeModeToolRuntime.invoke_with_context

    def staged(self: Any, name: str, arguments: Mapping[str, Any], **kwargs: Any) -> Any:
        if name == "observe":
            stages.stage("looked", views=list(arguments.get("views") or []))
            if arguments.get("views") == ["nowhere"]:
                return {"ok": False, "error": {"code": "unknown_view", "message": "no camera nowhere",
                                               "retryable": False}}
        return invoke(self, name, arguments, **kwargs)

    monkeypatch.setattr(toolkit_explain, "template", lambda tool: (
        json.loads(json.dumps(OBSERVE_TEMPLATE)) if tool == "observe" else template(tool)))
    monkeypatch.setattr(CodeModeToolRuntime, "invoke_with_context", staged)


def test_what_a_script_asks_of_the_library_is_on_the_record_and_explain_is_reproduced_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each ``query`` and ``explain`` request is a nested ``robot.<op>`` call pair of the script, with what was asked
    and what was returned, refused or answered, numbered among its calls; each call's result records its explain
    inputs. The run seals and verifies (which renders every explanation a script read again from the record), no
    counter counts a request, the model reads only the script's result, and turn two's ``explain`` of turn one's call,
    with nothing kept in memory, is answered from the record alone."""
    from ora.agent.code_mode import record
    from ora.agent.evidence.agent_events import library_op
    from ora.agent.pi import code_mode as session

    _staged_observe(monkeypatch)
    monkeypatch.setattr(session, "EXPLAIN_CALLS", 0)  # nothing kept in memory: every explain reads the record
    run_ = run(tmp_path, monkeypatch, script("g0", LOOK_ASK_ACT), script("e0", ASK_AGAIN),
               code_mode={"library": "extended"})
    assert run_["evaluation"]["status"] == "valid", run_["evaluation"]["reasons"]
    root, events = run_["root"], run_["events"]
    nested = [(event["tool_call_id"], event["payload"]["name"], library_op(event)) for event in events
              if event["type"] == "tool_call_proposed" and "#" in event["tool_call_id"]]
    assert nested == [("g0#001", "observe", None), ("g0#002", "robot.explain", "explain"),
                      ("g0#003", "robot.query", "query"), ("g0#004", "observe", None),
                      ("g0#005", "close_gripper", None), ("e0#001", "robot.explain", "explain"),
                      ("e0#002", "observe", None)]
    results = {event["tool_call_id"]: event for event in events if event["type"] == "tool_result"}
    asked = {event["tool_call_id"]: event["payload"]["arguments"] for event in events
             if event["type"] == "tool_call_proposed" and library_op(event)}
    assert asked == {"g0#002": {"name": None, "arguments": {"call_id": "g0#001"}},
                     "g0#003": {"name": "robot_state", "arguments": {}},
                     "e0#001": {"name": None, "arguments": {"call_id": "g0#001"}}}
    returned = {call: result(run_, call) for call in asked}
    assert returned["g0#003"]["ok"] is False and returned["g0#003"]["error"]["code"] == "query_unavailable"
    whole = record.render(record.recorded_inputs(root, "g0#001")).text
    assert whole.startswith("observe(agentview)  looked\n") and "rendered agentview" in whole
    assert returned["g0#002"] == returned["e0#001"] == {"ok": True, "call_id": "g0#001", "text": whole}
    assert all(record.EXPLAIN_INPUTS_FIELD in results[call]["payload"] for call in ("g0#001", "g0#004", "g0#005"))
    outputs = [result(run_, call)["output"] for call in ("g0", "e0")]
    assert outputs == [whole + "query_unavailable\ng0#004\n", whole], "the text each script read"
    (failed,) = result(run_, "e0")["calls"]
    attached = record.render(record.recorded_inputs(root, "e0#002")).text
    assert attached.startswith("observe(nowhere)  refused: unknown_view")
    assert failed["error"] == "unknown_view" and failed["explain"] == record.attached(attached)
    assert record.check_explanations(root) == []
    # Counters: one action, two other calls, no query spent; one robot decision; the model reads the result only.
    outcome = result(run_, "g0")
    assert [(entry["call"], entry["tool"]) for entry in outcome["calls"]] == [(1, "observe"), (4, "observe"),
                                                                             (5, "close_gripper")]
    assert outcome["actions_executed"] == 1 and outcome["allowance_left"]["calls"] == 22
    assert outcome["allowance_left"]["queries"]["count"] == 32
    assert [event["tool_call_id"] for event in events if event["type"] == "action_proposed"] == ["g0"]
    assert execution(run_, "g0")["native_actions_used"] == 1 and run_["result"]["decisions"] == 1
    assert read_by_model(run_, "g0")["calls"] == outcome["calls"]
    tool_messages = [message for message in json.loads(run_["requests"][-1])["messages"] if message.get("role") == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["g0", "e0"]
    assert not any(name in json.dumps(tool_messages) for name in ("robot.query", "robot.explain", "explain_inputs"))
    # No reader of the trace takes a request for a tool call: the run report, `ora toolkit replay`'s list and the
    # live index that numbers the calls it records, the Viewer's process timeline and the terminal transcript.
    from ora.agent.toolkit.host.event_index import EventIndex
    from ora.agent.tui.transcript import Transcript
    from ora.orchestrator.reports.events import scan_agent_events
    from ora.orchestrator.toolkit.selection import list_calls
    from viewer.process_timeline import compile_process_timeline

    calls = ["g0", "g0#001", "g0#004", "g0#005", "e0", "e0#002", "f0"]
    assert set(scan_agent_events(root)["tool_calls"]) == {"close_gripper", "finish_episode", "observe", "bash"}
    index = EventIndex(root / "agent-events.jsonl")
    index.refresh()
    assert [(row["call"], row["tool_call_id"]) for row in index.proposals] == list(enumerate(calls))
    assert [(item["call"], item["tool_call_id"]) for item in list_calls(root)["calls"]] == list(enumerate(calls))
    shown = [item.get("tool_call_id") for item in compile_process_timeline(root)["events"] if item.get("tool")]
    assert set(shown) <= set(calls) and "g0#002" not in shown
    transcript = Transcript(root)
    fed = [op for event in events for op in transcript.feed(event) if op["op"] in ("tool_call", "tool_result")]
    assert {op["id"] for op in fed} == set(calls)


#: What a ``base`` script's call (a look, then an action) wrote before the library's requests were recorded (commit
#: 3de663f24, the same episode): its nested calls field by field, and the order of the events under its call.
ORIGIN = {"kind": "code_mode", "parent_tool_call_id": "g0"}
BASE_NESTED = [
    ("tool_call_proposed", "g0#001", {"name": "observe", "arguments": {"views": ["agentview"]}, "effect": "read_only",
                                      "batch_valid": True, "origin": {**ORIGIN, "seq": 1}}),
    ("tool_result", "g0#001", {"name": "observe", "ok": True, "origin": {**ORIGIN, "seq": 1}}),
    ("tool_image", "g0#001", {"image_index": 0, "origin": {**ORIGIN, "seq": 1}}),
    ("tool_call_proposed", "g0#002", {"name": "close_gripper", "arguments": {}, "effect": "mutation",
                                      "batch_valid": True, "origin": {**ORIGIN, "seq": 2}}),
    ("tool_result", "g0#002", {"name": "close_gripper", "ok": True, "origin": {**ORIGIN, "seq": 2}}),
]
BASE_ORDER = [("tool_call_proposed", "g0"), ("tool_call_proposed", "g0#001"), ("tool_result", "g0#001"),
              ("tool_image", "g0#001"), ("tool_call_proposed", "g0#002"), ("tool_result", "g0#002"),
              ("action_proposed", "g0"), ("environment_receipt", None), ("fresh_observation_published", None),
              ("tool_result", "g0")]


def _under(run_: Mapping[str, Any], parent: str) -> tuple[list[Any], list[Any]]:
    """The nested calls of the script call ``parent``, field by field, and the order of the events under it."""

    events = run_["events"]
    nested = [(event["type"], event["tool_call_id"], event["payload"]) for event in events
              if str(event.get("tool_call_id")).startswith(parent + "#")]
    order = [(event["type"], event.get("tool_call_id")) for event in events]
    return nested, order[order.index(("tool_call_proposed", parent)):order.index(("tool_result", parent)) + 1]


def test_a_base_script_writes_what_it_wrote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``library`` absent: no request is served or recorded and no call records explain inputs; the nested calls and the
    order of the events under the script's call are those the same episode wrote before. With ``library: extended`` the
    same calls add their explain inputs, and the action's call records its result once the action ran."""

    look_act = "r = tools.observe(views=['agentview'])\ntools.close_gripper()\n"
    base = run(tmp_path / "base", monkeypatch, script("g0", look_act))
    assert base["evaluation"]["status"] == "valid", base["evaluation"]["reasons"]
    assert _under(base, "g0") == (BASE_NESTED, BASE_ORDER)
    extended = run(tmp_path / "extended", monkeypatch, script("g0", look_act), code_mode={"library": "extended"})
    assert extended["evaluation"]["status"] == "valid", extended["evaluation"]["reasons"]
    nested, order = _under(extended, "g0")
    assert [(kind, call, {key: value for key, value in payload.items() if key != "explain_inputs"})
            for kind, call, payload in nested] == BASE_NESTED
    assert [kind for kind, call, payload in nested if "explain_inputs" in payload] == ["tool_result"] * 2
    assert order == [*BASE_ORDER[:5], *BASE_ORDER[6:9], BASE_ORDER[5], BASE_ORDER[9]]


def test_a_script_that_prints_and_hands_back_a_contact_key_keeps_the_context_and_the_run_valid(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The names are the model's: three runs of 2026-10-27 aborted on a printed ``contact`` before their next model
    call (``causal_context_forbidden_fields``), and one of 2026-11-01 was refused sealing on the receipt its script
    handed back. Later requests repeat both, so the check and the audit read them as the result itself."""

    run_ = run(tmp_path, monkeypatch, script("printed", """
import json
print(json.dumps({"lift": {"contact": [0.1, 0.2, 0.3], "contact_motion": {"contact_estimated": True}}}))
"""), script("handed", """
tools.close_gripper()
handoff("lifted", rc={"detail": {"contact_motion": {"contact_estimated": True}}, "contact": None})
"""))
    assert run_["result"]["termination"] == "policy_finished_episode"
    assert run_["evaluation"]["status"] == "valid", run_["evaluation"]["reasons"]
    assert result(run_, "printed")["status"] == "completed"
    assert result(run_, "handed")["handed_back"]["data"]["rc"]["contact"] is None
