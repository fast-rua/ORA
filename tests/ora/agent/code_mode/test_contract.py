"""``policy.code_mode`` resolves to the code-mode contract, tools and prompt layers (``ora/agent/code_mode``)."""

from __future__ import annotations

import pytest

from ora.agent.code_mode.contract import with_reads

import json

from ora.agent.code_mode import coding
from ora.agent.code_mode import contract as contract_module
from ora.agent.code_mode import docs
from ora.agent.code_mode.contract import (CONTRACT, DEFAULT_LIBRARY, DEFAULT_PROMPT, DIRECT_TOOLS, LIMITS, QUERY_LIMITS,
                                         prompt_generation, query_allowance, resolve_contract, script_library,
                                         wire_name)
from ora.agent.code_mode.runtime import CodeModeToolRuntime, validate_command
from ora.agent.error_classes import ToolError

CODING_PLANNER = "ora/policy/planners/native/code-mode-coding-v2.md"
CODE_CORE = "ora/policy/core/code-policy-loop-v2.md"
STATIC_FACTS = "ora/policy/core/round-static-facts-v1.md"
TRAJECTORY_BUDGET = "ora/policy/planners/native/code-mode-trajectory-budget-v1.md"


def test_off_unless_asked_one_switch_and_limits_only_tighten() -> None:
    assert resolve_contract(False) is None and resolve_contract(None) is None
    coding = resolve_contract(True)
    assert coding == {"contract": CONTRACT, "code_only": True, "handoff": True, "script_math": True,
                      "workspace": True, "coding_tools": True, "prompt": DEFAULT_PROMPT, **LIMITS}
    assert DEFAULT_PROMPT == "v2"
    assert resolve_contract({"interface": "coding"}) == coding
    assert resolve_contract(coding) == coding, "a resolved contract resolves to itself"
    # A contract without prompt is v1, what code-mode runs sealed before v2 read; v1 was retired on 2026-10-31.
    sealed = {key: value for key, value in coding.items() if key != "prompt"}
    assert prompt_generation(sealed) == "v1"
    for retired in (sealed, {"prompt": "v1"}):
        with pytest.raises(ValueError, match="v1 was retired"):
            resolve_contract(retired)
    assert resolve_contract({"max_actions_per_script": 4})["max_actions_per_script"] == 4
    for bad, match in (({"max_actions_per_script": 40}, "from 1 to 12"), ({"timeout": 5}, "unsupported"),
                       ({"interface": "repl"}, "interface"), ({"interface": "coding", "kernel": True}, "unsupported"),
                       ({"prompt": "v4"}, "prompt")):
        with pytest.raises(ValueError, match=match):
            resolve_contract(bad)


def test_the_script_interface_was_retired_and_a_contract_of_it_is_refused_with_why() -> None:
    """``interface: script`` (``run_python``, ``read_file``) ran in none of 119 cohorts: a RunSpec that names it, and a
    resolved contract of it (no ``coding_tools``), are refused, in resolution and when a persisted plan composes."""
    from ora.orchestrator.spec.values import RunSpecError
    from tests.support.agent_core.episodes import synthetic_plan

    resolved = {key: value for key, value in resolve_contract(True).items() if key != "coding_tools"}
    for retired in ({"interface": "script"}, {"interface": "script", "library": "extended", "prompt": "v3"}, resolved):
        with pytest.raises(ValueError, match="interface script .* was retired on 2026-11-01.* Use interface coding"):
            resolve_contract(retired)
    with pytest.raises(ValueError, match="was retired on 2026-11-01"):
        contract_module.current_generation(resolved)
    with pytest.raises(RunSpecError, match="was retired on 2026-11-01"):
        synthetic_plan(actions=2, policy_overrides={"code_mode": {"interface": "script"}})


def test_the_library_is_recorded_only_when_extended_and_an_unknown_one_is_refused(monkeypatch) -> None:
    """``library`` absent is ``base``, today's contract: a base contract names no library, so it is the contract
    every plan resolved before the switch carries; ``extended`` is recorded and resolves to itself."""
    assert DEFAULT_LIBRARY == "base"
    base = resolve_contract(True)
    assert "library" not in base and script_library(base) == "base"
    assert resolve_contract({"library": "base"}) == base
    extended = resolve_contract({"library": "extended"})
    assert extended == {**base, "library": "extended", **QUERY_LIMITS} and extended.get("library") == "extended"
    assert resolve_contract(extended) == extended and script_library(extended) == "extended"
    for bad in ("full", "", None, True):
        with pytest.raises(ValueError, match="policy.code_mode.library must be one of base, extended"):
            resolve_contract({"library": bad})
    # When the default becomes extended, a contract resolved before it (no library) stays base.
    monkeypatch.setattr(contract_module, "DEFAULT_LIBRARY", "extended")
    assert resolve_contract(True)["library"] == "extended"
    assert resolve_contract(base) == base and "library" not in resolve_contract(base)


def test_the_query_allowance_is_a_limit_of_the_extended_library_and_only_tightens() -> None:
    """``robot.query`` (and the dry runs that share it) spend a count and seconds per script. They are recorded with
    ``library: extended`` only: a base contract names neither, so it is the contract it always was."""
    assert QUERY_LIMITS == {"max_queries_per_script": 32, "query_time_s": 180}
    for contract in (resolve_contract(True), resolve_contract({"prompt": "v2"}), resolve_contract({"library": "base"})):
        assert not set(QUERY_LIMITS) & set(contract), "a base contract records no query allowance"
        assert query_allowance(contract) == (32, 180), "and a reader of one still gets the default"
    extended = resolve_contract({"library": "extended"})
    assert {key: extended[key] for key in QUERY_LIMITS} == QUERY_LIMITS
    tight = resolve_contract({"library": "extended", "max_queries_per_script": 5, "query_time_s": 10})
    assert query_allowance(tight) == (5, 10) and resolve_contract(tight) == tight
    assert resolve_contract({"library": "extended", "max_queries_per_script": 0})["max_queries_per_script"] == 0
    for bad, match in (({"max_queries_per_script": 33}, "from 0 to 32"), ({"query_time_s": 181}, "from 1 to 180"),
                       ({"query_time_s": 0}, "from 1 to 180"), ({"max_queries_per_script": True}, "from 0 to 32")):
        with pytest.raises(ValueError, match=match):
            resolve_contract({"library": "extended", **bad})
    for stray in ({"max_queries_per_script": 4}, {"query_time_s": 4}):
        with pytest.raises(ValueError, match="query allowance of library extended"):
            resolve_contract(stray)
        with pytest.raises(ValueError, match="query allowance of library extended"):
            resolve_contract({"library": "base", **stray})


def test_forced_preview_is_retired_and_a_dry_run_stays_optional() -> None:
    """``preview`` (plans first, run with ``.go()``) was retired on 2026-11-05: asked or recorded, it is refused with
    the retirement, also as the read a resolved contract of a sealed preview run lists in its ``script_reads``."""

    plain = resolve_contract({"library": "extended", "prompt": "v3"})
    assert "preview" not in plain
    assert with_reads(plain, toolkit=True, moves=True)["script_reads"] == ["dry_run", "query"]
    for asked in ({"library": "extended", "prompt": "v3", "preview": True}, {"preview": False}, {"preview": "yes"}):
        with pytest.raises(ValueError, match="preview .* was retired on 2026-11-05"):
            resolve_contract(asked)
    recorded = {**plain, "preview": True, "script_reads": ["preview", "query"]}
    with pytest.raises(ValueError, match="preview .* was retired on 2026-11-05"):
        resolve_contract(recorded)
    with pytest.raises(ValueError, match="preview .* was retired on 2026-11-05"):
        resolve_contract({key: value for key, value in recorded.items() if key != "preview"})


def test_the_source_is_recorded_only_when_asked_needs_extended_and_is_served_on_the_toolkit() -> None:
    assert "source" not in resolve_contract({"library": "extended", "prompt": "v3"})
    asked = resolve_contract({"library": "extended", "prompt": "v3", "source": True})
    assert asked["source"] is True and resolve_contract(asked) == asked
    assert with_reads(dict(asked), toolkit=True, moves=True)["script_reads"] == ["dry_run", "query", "source"]
    assert "script_reads" not in with_reads(dict(asked), toolkit=False, moves=False)
    with pytest.raises(ValueError, match="source needs library extended"):
        resolve_contract({"source": True})
    with pytest.raises(ValueError, match="source must be true or false"):
        resolve_contract({"library": "extended", "prompt": "v3", "source": "yes"})


def test_what_a_surface_serves_is_decided_once_and_recorded_with_the_extended_library_only() -> None:
    extended = resolve_contract({"library": "extended"})
    assert "script_reads" not in extended and contract_module.script_reads(extended) == frozenset()
    served = contract_module.with_reads(extended, toolkit=True, moves=True)
    assert served["script_reads"] == ["dry_run", "query"]
    assert resolve_contract(served) == served, "a resolved contract resolves to itself"
    released = contract_module.with_reads(served, toolkit=True, moves=False)
    assert released["script_reads"] == ["query"], "the surface decides, not what the contract said before"
    assert "script_reads" not in contract_module.with_reads(served, toolkit=False, moves=False)
    base = resolve_contract({"library": "base"})
    assert contract_module.with_reads(base, toolkit=True, moves=True) == base, "base records nothing new"
    with pytest.raises(ValueError, match="unsupported policy.code_mode keys: script_reads"):
        resolve_contract({"library": "extended", "script_reads": ["query"]})
    with pytest.raises(ValueError, match="script_reads must name only"):
        resolve_contract({**extended, "script_reads": ["teleport"]})
    with pytest.raises(ValueError, match="inspect was retired on 2026-11-01"):
        resolve_contract({**extended, "script_reads": ["inspect", "query"]})


def test_a_base_bash_is_described_as_it_always_was() -> None:
    """The model-visible text of a code-mode contract without ``library: extended`` names no query and lists no more
    than the functions it did: the index and the queries are the extended library's."""
    contract = resolve_contract({"prompt": "v2"})
    text = json.dumps(coding.definitions(contract, [
        {"name": "observe", "description": "Look.", "inputSchema": {"type": "object"}}]))
    assert "robot.query" not in text and "query_" not in text and "robot.docs" not in text


def _plan(code_mode):
    from tests.support.agent_core.episodes import synthetic_plan

    return synthetic_plan(actions=2, policy_overrides={"code_mode": code_mode} if code_mode is not None else None)


def _layers(plan) -> list[str]:
    return [layer["path"] for channel in plan["policy_context"].values() if isinstance(channel, list)
            for layer in channel]


def test_the_coding_interface_gives_the_model_a_coding_agent_s_tools() -> None:
    full = [item["name"] for item in _plan(None)["policy_composition"]["tool_surface"]["definitions"]]
    plan = _plan(True)
    surface = plan["policy_composition"]["tool_surface"]
    names = [item["name"] for item in surface["definitions"]]
    assert names[-4:] == ["read", "write", "edit", "bash"] and "run_python" not in names and "read_file" not in names
    assert set(names[:-4]) <= DIRECT_TOOLS | set(surface["lifecycle_tools"])
    (layer,) = [item for item in surface["derived_layers"] if item["kind"] == "code_mode_v1"]
    assert set(names[:-4]) | set(layer["script_functions"]) | {"compute"} >= set(full)
    bash = surface["definitions"][-1]["description"]
    assert "from robot import tools, handoff" in bash
    assert all(f"### tools.{wire_name(name)}\n" in bash for name in layer["script_functions"])
    assert {"bash", "read", "write", "edit"} <= set(surface["result_schemas"])
    paths = _layers(plan)
    assert CODING_PLANNER in paths and "ora/policy/planners/native/policy-v5.md" not in paths
    plain = _plan(None)
    assert "code_mode" not in plain["policy"] and "bash" not in full


def _functions(plan) -> list[dict]:
    """The definitions of the plan's script functions, as the tool-call plan of the same RunSpec advertises them."""
    full = {item["name"]: item for item in _plan(None)["policy_composition"]["tool_surface"]["definitions"]}
    (layer,) = [item for item in plan["policy_composition"]["tool_surface"]["derived_layers"]
                if item["kind"] == "code_mode_v1"]
    return [full[name] for name in layer["script_functions"]]


def test_a_base_library_composes_byte_for_byte_as_a_contract_without_one() -> None:
    """``library: base`` is the control: tool definitions, prompt layers, the derived layer and every identity are
    those of the same contract without the field, whose script tool lists each function's description and schema."""
    absent, base = _plan(True), _plan({"library": "base"})
    assert base["policy"]["code_mode"] == absent["policy"]["code_mode"] and "library" not in base["policy"]["code_mode"]
    assert json.dumps(base["policy_composition"], sort_keys=True) == json.dumps(absent["policy_composition"],
                                                                                sort_keys=True)
    assert json.dumps(base["policy_context"], sort_keys=True) == json.dumps(absent["policy_context"], sort_keys=True)
    assert base["policy_identity"]["model_visible_sha256"] == absent["policy_identity"]["model_visible_sha256"]
    description = base["policy_composition"]["tool_surface"]["definitions"][-1]["description"]
    for function in _functions(base):
        schema = json.dumps(function.get("inputSchema") or {}, sort_keys=True, separators=(",", ":"))
        assert (f"\n### tools.{wire_name(function['name'])}\n{function['description'].strip()}\nArguments: {schema}"
                in description)
    assert "robot.docs" not in description
    shown = json.dumps(base["policy_composition"], sort_keys=True) + json.dumps(base["policy_context"], sort_keys=True)
    # What the extended library's fragments add (perception fragments, reused results, the pose after an action)
    # reaches the model through the library and its documentation only: a base contract's text names none of it.
    assert not [word for word in ("object_geometry", "ora_lib", "reused", "pose_after", "depth_inliers")
                if word in shown]


def test_an_extended_library_lists_one_line_per_function_instead_of_their_documentation() -> None:
    """With ``library: extended`` the script tool's description ends with the index generated from the same
    definitions; nothing else the model is shown changes (the planner and core layers are the base ones)."""
    base, plan = _plan(True), _plan({"library": "extended"})
    before, surface = base["policy_composition"]["tool_surface"], plan["policy_composition"]["tool_surface"]
    assert plan["policy"]["code_mode"] == {**base["policy"]["code_mode"], "library": "extended", **QUERY_LIMITS}
    assert surface["definitions"][:-1] == before["definitions"][:-1]
    assert surface["definitions"][-1]["inputSchema"] == before["definitions"][-1]["inputSchema"]
    assert plan["policy_context"] == base["policy_context"]
    (layer,) = [item for item in surface["derived_layers"] if item["kind"] == "code_mode_v1"]
    assert layer["contract"]["library"] == "extended"
    functions = _functions(plan)
    assert functions and [item["name"] for item in functions] == [item["name"] for item in _functions(base)]
    description = surface["definitions"][-1]["description"]
    results = base["policy_composition"]["tool_surface"]["result_schemas"]
    assert description.endswith(":\n" + docs.listing(functions, results, reads=()))
    assert 'robot.docs("<name>")' in description and "help(tools.<name>)" in description
    assert "### tools." not in description and "Arguments: {" not in description
    assert len(description) < len(before["definitions"][-1]["description"])
    assert surface["advertised_composition_sha256"] != before["advertised_composition_sha256"]


def test_on_a_toolkit_surface_the_index_comes_from_each_tool_s_description_json() -> None:
    """The ToolSurface definitions a toolkit plan resolves are the tools' ``description.json`` files: the index
    line of every script function is generated from that file (``observe`` with the Artifact reader's ``ref`` mode
    where the plan's context scheme provides the reader)."""
    from ora.agent.memory.artifacts import context_artifact_reader, recoverable_artifact_definitions
    from ora.orchestrator.spec.run_spec import resolve_run_spec
    from tests.support.orchestrator.run_specs import ROOT, robodojo_native_spec

    spec = robodojo_native_spec("general_pickup")
    spec["policy"].update(code_mode={"library": "extended"}, tool_surface={"profile": "robodojo_placement_toolkit_v5"},
                          baseline="robodojo_toolkit_v6")
    spec["toolkit_grants"] = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
    plan = resolve_run_spec(spec)
    surface = plan["policy_composition"]["tool_surface"]
    (layer,) = [item for item in surface["derived_layers"] if item["kind"] == "code_mode_v1"]
    bash = surface["definitions"][-1]
    assert bash["name"] == "bash"
    lines = bash["description"].splitlines()
    for name in layer["script_functions"]:
        tool = json.loads((ROOT / f"ora/agent/toolkit/tools/{name}/description.json").read_text(encoding="utf-8"))
        if name == "observe" and context_artifact_reader(plan["policy"]):
            tool = recoverable_artifact_definitions([tool])[0]  # followed by the reader it appends
        assert docs.index_line(tool, surface["result_schemas"][name], reads=docs.READS) in lines, name
    assert len([line for line in lines if line.startswith("- tools.")]) == len(layer["script_functions"])


class _Base:
    allowed_tools = ("observe", "compute", "check_task")
    surface = type("Surface", (), {"result_schemas": {"observe": {}}, "lifecycle_tools": ()})()

    def definitions(self):
        return [{"name": name, "description": name.title() + ".", "inputSchema": {"type": "object"}}
                for name in ("observe", "compute", "check_task")]

    def classify_call_effect(self, name, arguments):
        return "read_only"

    def invoke(self, name, arguments):
        return {"ok": True}


def test_the_runtime_shows_the_model_less_than_scripts_can_call_and_never_runs_a_script() -> None:
    contract = resolve_contract(True)
    runtime = CodeModeToolRuntime(base=_Base(), contract=contract)
    assert runtime.allowed_tools == ("check_task", "read", "write", "edit", "bash")
    description = runtime.definitions()[-1]["description"]
    assert "### tools.observe\nObserve." in description and "### tools.compute" not in description
    assert [item["name"] for item in runtime.script_definitions()] == ["observe", "compute", "check_task"]
    assert runtime.classify_call_effect("bash", {"command": "true"}) == "mutation"
    assert runtime.classify_call_effect("read", {"path": ""}) == "read_only"
    with pytest.raises(ToolError, match="policy session"):
        runtime.invoke("bash", {"command": "true"})
    for bad in ({}, {"command": ""}, {"command": "true", "x": 1}, {"command": "x" * (contract["max_script_chars"] + 1)}):
        with pytest.raises(ToolError):
            validate_command(bad, contract)
    extended = CodeModeToolRuntime(base=_Base(), contract={**contract, "library": "extended"})
    description = extended.definitions()[-1]["description"]
    assert ":\n- tools.observe(): Observe.\n`ora_lib` (" in description and "### tools." not in description
    assert "robot.query" not in description, "no query where the surface serves none"
    assert 'robot.query("ik", arm: ' in CodeModeToolRuntime(
        base=_Base(), contract={**contract, "library": "extended", "script_reads": ["query"]}).definitions()[-1]["description"]


def test_code_mode_replaces_the_planner_and_core_on_any_surface() -> None:
    plan = _plan(True)
    paths = _layers(plan)
    assert paths.count(CODING_PLANNER) == 1
    assert CODE_CORE in paths or "ora/policy/core/evidence-action-loop-v8.md" not in _layers(_plan(None))


def test_the_code_tool_promises_no_robot_state_and_names_the_toolkit_receipt() -> None:
    """v2 promises no robot state the session does not return (the retired v1 did) and says where a tool executing
    inside its own call carries its receipt."""
    current = _plan(True)["policy_composition"]["tool_surface"]["definitions"][-1]["description"]
    for stale in ("its receipt and robot state", "`observation` (the public robot state after it)",
                  "and the public robot state"):
        assert stale not in current
    assert "result.receipt" in current


def test_code_mode_refuses_a_surface_whose_skills_the_stage_loop_drives() -> None:
    """RoboDojo's closed-skill grasp_execute commits a sequence for after the turn; its toolkit surface runs it."""
    from types import SimpleNamespace

    from ora.orchestrator.spec.native_policy import _resolve_code_mode
    from ora.orchestrator.spec.values import RunSpecError

    def surface(*capabilities):
        return SimpleNamespace(profile_id="robodojo_test", capabilities=capabilities,
                               definitions=[{"name": "grasp_plan"}, {"name": "grasp_execute"}],
                               supports=lambda capability: capability in capabilities)

    for runtime in (None, {"kind": "robodojo"}):  # a RunSpec naming no backend is refused too
        with pytest.raises(RunSpecError, match="synchronous toolkit surface: grasp_execute"):
            _resolve_code_mode({"agent_core": "pi", "code_mode": True}, surface(), runtime)
    policy = {"agent_core": "pi", "code_mode": True}
    _resolve_code_mode(policy, surface("toolkit_synchronous_execution_v1"), {"kind": "robodojo"})
    assert policy["code_mode"]["coding_tools"] is True
    policy = {"agent_core": "pi", "code_mode": True}
    _resolve_code_mode(policy, surface(), {"kind": "robotwin"})  # the script's parent runs the stage loop
    assert policy["code_mode"]["coding_tools"] is True


def test_robotwin_runs_code_mode_on_its_closed_skill_surface() -> None:
    """RoboTwin's only surface keeps grasp_execute, as a script function; the tool-call plan is unchanged."""
    import copy

    from ora.orchestrator.spec.run_spec import resolve_run_spec
    from tests.support.orchestrator.run_specs import native_spec

    spec = native_spec("robotwin")
    plain = resolve_run_spec(copy.deepcopy(spec))
    spec["policy"]["code_mode"] = True
    plan = resolve_run_spec(spec)
    surface = plan["policy_composition"]["tool_surface"]
    assert plan["policy"]["code_mode"]["contract"] == CONTRACT
    assert surface["profile_id"] == plain["policy_composition"]["tool_surface"]["profile_id"] \
        == "robotwin_toolchain_b2_closed_noservo_v2"
    (layer,) = [item for item in surface["derived_layers"] if item["kind"] == "code_mode_v1"]
    assert "grasp_execute" in layer["script_functions"]
    assert "code_mode" not in plain["policy"]


def test_the_code_core_describes_the_round_static_facts_only_where_they_are_sent() -> None:
    """From ``v2`` the ``round_static_facts`` message is described once, only on a composition that sends it
    (``native_slots_v4``, RoboTwin's default): by the context scheme's prompt where code mode delivers one (the default
    baseline's ``code_mode_context_scheme``), else in its own layer right after the code core."""
    from ora.orchestrator.spec.composition import _code_policy_layers
    from ora.orchestrator.spec.run_spec import resolve_run_spec
    from tests.support.orchestrator.run_specs import ROOT, native_spec

    spec = native_spec("robotwin")
    spec["policy"]["code_mode"] = {"prompt": "v2"}
    plan = resolve_run_spec(spec)
    assert plan["policy"]["context_composition"] == "native_slots_v4"
    paths = _layers(plan)
    sentence = "is stated once per round, in the round_static_facts message"
    texts = {path: (ROOT / path).read_text() for path in paths}
    assert STATIC_FACTS not in paths and sentence not in texts[CODE_CORE]
    assert [path for path in paths if sentence in texts[path]] == ["ora/policy/context_schemes/keep-written.md"]
    # Without a scheme prompt the layer follows the core.
    bound = {"harness_system": [{"path": "ora/policy/planners/native/policy-v5.md"}],
             "project_instructions": [{"path": "ora/policy/core/evidence-action-loop-v8.md"}]}
    layered = [layer["path"] for layer in _code_policy_layers(bound, root=ROOT, prompt="v2", round_static_facts=True)[
        "project_instructions"]]
    assert layered == [CODE_CORE, STATIC_FACTS]
    assert sentence in (ROOT / STATIC_FACTS).read_text()


def test_robodojo_runs_code_mode_on_its_synchronous_toolkit_surface() -> None:
    """On RoboDojo's default surface (toolkit v6) every tool a script may act with is an execution tool the executor knows."""
    from ora.agent.composition.schemas.toolkit_execution import EXECUTION_TOOLS
    from ora.orchestrator.spec.run_spec import resolve_run_spec
    from tests.support.orchestrator.run_specs import robodojo_native_spec

    spec = robodojo_native_spec("general_pickup")
    spec["policy"]["code_mode"] = True
    plan = resolve_run_spec(spec)
    assert plan["policy"]["baseline"]["id"] == "robodojo_toolkit_v7"
    paths = _layers(plan)
    assert CODING_PLANNER in paths
    assert not any(path.startswith("ora/policy/prompt_bundles/toolkit_v7/") and path.endswith(("/planner.md", "/core.md"))
                   for path in paths), paths
    surface = plan["policy_composition"]["tool_surface"]
    (layer,) = [item for item in surface["derived_layers"] if item["kind"] == "code_mode_v1"]
    # v5 adds return_home (2026-10-28), an action tool its toolset declares.
    assert set(surface["mutation_tools"]) <= EXECUTION_TOOLS | {"return_home"}
    assert set(surface["mutation_tools"]) <= set(layer["script_functions"])

def test_on_toolkit_v5_code_mode_states_the_trajectory_budget_rules() -> None:
    """On a surface whose moves are joint trajectories (v5) one primitive's 20 native actions is no rule: from
    ``v2`` the trajectory budget layer follows the code planner instead, wherever the planner it replaced came
    from (here the toolkit_v6 bundle's native planner; the current_v4 one before baseline robodojo_toolkit_v5 was
    retired on 2026-10-31)."""
    from ora.orchestrator.spec.run_spec import resolve_run_spec
    from tests.support.orchestrator.run_specs import robodojo_native_spec

    spec = robodojo_native_spec("general_pickup")
    spec["policy"].update(code_mode={"prompt": "v2"}, tool_surface={"profile": "robodojo_placement_toolkit_v5"},
                          baseline="robodojo_toolkit_v6")
    spec["toolkit_grants"] = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
    plan = resolve_run_spec(spec)
    paths = _layers(plan)
    assert paths.index(TRAJECTORY_BUDGET) == paths.index(CODING_PLANNER) + 1, "the budget rules follow the planner"
    functions = [item for item in plan["policy_composition"]["tool_surface"]["derived_layers"]
                 if item["kind"] == "code_mode_v1"][0]["script_functions"]
    assert {"move_pose", "recover_pose", "return_home", "set_gripper", "grasp_execute"} <= set(functions)


V9_CORE = "ora/policy/core/evidence-action-loop-v9.md"
V9_CODE_CORE = "ora/policy/core/code-policy-loop-v3.md"
SURFACE_V5_PROMPT = "ora/policy/tool_surfaces/robodojo_placement_toolkit_v5/prompt.md"


def test_code_mode_runs_on_robodojo_toolkit_v6_with_the_code_forms_of_its_text() -> None:
    """Baseline ``robodojo_toolkit_v6`` delivers core v9 and surface v5's ``prompt.md``. From ``v2``
    code mode puts v9's code form in its place and leaves the surface prompt out: the code planner, the trajectory
    budget layer and the code core state its rules for a program. The baseline itself is not overridden."""
    import copy

    from ora.orchestrator.spec.run_spec import resolve_run_spec
    from tests.support.orchestrator.run_specs import ROOT, robodojo_native_spec

    def channels(plan):
        return {channel: [layer["path"] for layer in layers]
                for channel, layers in plan["policy_context"].items() if isinstance(layers, list)}

    spec = robodojo_native_spec("general_pickup")
    spec["policy"]["baseline"] = "robodojo_toolkit_v6"
    spec["toolkit_grants"] = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
    plain = channels(resolve_run_spec(copy.deepcopy(spec)))
    (core_channel,) = [channel for channel, paths in plain.items() if V9_CORE in paths]
    assert any(SURFACE_V5_PROMPT in paths for paths in plain.values()), "tool-call mode delivers both"
    spec["policy"]["code_mode"] = True
    plan = resolve_run_spec(spec)
    assert plan["policy"]["baseline"]["id"] == "robodojo_toolkit_v6"
    coded = channels(plan)
    assert coded[core_channel].index(V9_CODE_CORE) == plain[core_channel].index(V9_CORE), "in the core's place"
    paths = _layers(plan)
    assert not {V9_CORE, SURFACE_V5_PROMPT, CODE_CORE, STATIC_FACTS} & set(paths)
    assert paths.index(TRAJECTORY_BUDGET) == paths.index(CODING_PLANNER) + 1, "the budget rules follow the planner"
    text = (ROOT / V9_CODE_CORE).read_text()
    assert "Your program is the calculation capability" in text and "round_static_facts" not in text


def test_code_mode_refuses_tool_call_text_it_has_no_code_form_of() -> None:
    """Delivered tool-call text without a code form (here core v2, ``ora local``'s) is refused rather than composed
    beside the code loop, so no prompt states both loops."""
    from ora.orchestrator.spec.composition import _code_policy_layers
    from ora.orchestrator.spec.values import RunSpecError
    from tests.support.orchestrator.run_specs import ROOT

    bound = {"harness_system": [{"path": "ora/policy/planners/native/policy-v5.md"}],
             "project_instructions": [{"path": "ora/policy/core/evidence-action-loop-v2.md"}]}
    with pytest.raises(RunSpecError, match="no code form of the tool-call text ora/policy/core/evidence-action-loop-v2.md"):
        _code_policy_layers(bound, root=ROOT, prompt="v2")


def test_a_not_started_script_action_reopens_its_observation(tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from ora.agent.pi.code_mode import PiCodeMode

    pointer = tmp_path / "current-observation.json"
    pointer.write_text(json.dumps({"observation_id": "obs-1", "open": False}))
    session = PiCodeMode()
    session.tools = SimpleNamespace(control_root=tmp_path)
    session._code_mode_reopen({"observation_id": "obs-0"})
    assert json.loads(pointer.read_text())["open"] is False, "a later observation is left alone"
    session._code_mode_reopen({"observation_id": "obs-1"})
    assert json.loads(pointer.read_text()) == {"observation_id": "obs-1", "open": True}


#: The Synthetic plan's identities at bc2cd3936 (the base of the code-mode library work, 2026-10-31), recorded before
#: ``explain``: ``(model_visible_sha256, advertised_composition_sha256)`` per code-mode request.
RECORDED_IDENTITIES = {
    "none": (None, "49b866f2d7cb99cae9dd9f43ab695e7ee1fec8848bd7de69692b46cfefad5c61",
             "efe72d9011806b1a71a943b7e2161abddbfd85fbc00b7c089284082f0e9cd189"),
    "coding": ({"interface": "coding"}, "3154a5a4f57b76ca2f2c1f6a5869cca99984960e7c77255d902d77694eefa276",
               "20bf0d57dea98681f7ba6780f00dd41075aa749d5915a6d8862777b0bef32087"),
    "coding, library base": ({"interface": "coding", "library": "base"},
                             "3154a5a4f57b76ca2f2c1f6a5869cca99984960e7c77255d902d77694eefa276",
                             "20bf0d57dea98681f7ba6780f00dd41075aa749d5915a6d8862777b0bef32087"),
    "no interface": (True, "3154a5a4f57b76ca2f2c1f6a5869cca99984960e7c77255d902d77694eefa276",
                     "20bf0d57dea98681f7ba6780f00dd41075aa749d5915a6d8862777b0bef32087"),
}


@pytest.mark.parametrize("name", list(RECORDED_IDENTITIES))
def test_a_plan_without_the_extended_library_keeps_the_identity_it_was_sealed_with(name) -> None:
    """Whatever the library adds (the index, ``explain``, its result fields) is gated on ``library: extended``:
    tool-call mode and every code-mode contract without it compose to the hashes recorded before the additions."""
    code_mode, model_visible, advertised = RECORDED_IDENTITIES[name]
    plan = _plan(code_mode)
    assert plan["policy_identity"]["model_visible_sha256"] == model_visible
    assert plan["policy_composition"]["tool_surface"]["advertised_composition_sha256"] == advertised
