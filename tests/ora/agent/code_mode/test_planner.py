"""Code mode's prompt generation ``v3`` (``policy.code_mode.prompt``): the planner written for ``library: extended``.

``v3`` is for the extended library, so a ``base`` contract is refused it. It rewrites the planner
(``code-mode-coding-v3.md``) as a procedure with a list of trigger lines and one worked
script, and keeps ``v2``'s core and every layer after it; the trigger lines of what only some surfaces serve are layers
of their own, composed there. ``v1`` and ``v2`` compose byte for byte as they did before
``v3`` existed (pinned below), and so does every ``library: base`` contract. The planner's worked script runs against
recorded tool answers, so the example cannot rot.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import re
import shutil
from pathlib import Path

import pytest

from ora.agent.code_mode import coding
from ora.agent.code_mode import contract as contract_module
from ora.agent.code_mode import ora_lib, robot
from ora.agent.code_mode.contract import resolve_contract, script_library
from ora.agent.code_mode.library import ObservationMount, observation_mount
from ora.agent.code_mode.ora_lib import observation as library_observation
from ora.agent.code_mode.runtime import ALLOWANCE_FIELD, extra_result_schemas
from ora.agent.code_mode.sandbox import sandbox_status
from ora.agent.code_mode.shell import run_command
from ora.agent.toolkit.interface.types import canonical, digest
from tests.support.orchestrator.run_specs import ROOT

V2 = "ora/policy/planners/native/code-mode-coding-v2.md"
V3 = "ora/policy/planners/native/code-mode-coding-v3.md"
CODE_CORE = "ora/policy/core/code-policy-loop-v2.md"
#: The trigger lines of what only some surfaces serve (``composition.CODE_MODE_READ_LAYERS``).
QUERY_LAYER = "ora/policy/planners/native/code-mode-query-v1.md"
DRY_RUN_LAYER = "ora/policy/planners/native/code-mode-dry-run-v1.md"


def _plan(code_mode):
    from tests.support.agent_core.episodes import synthetic_plan

    return synthetic_plan(actions=2, policy_overrides={"code_mode": code_mode})


def _layers(plan) -> list[str]:
    return [layer["path"] for channel in plan["policy_context"].values() if isinstance(channel, list)
            for layer in channel]


# --- the generation and the library ---------------------------------------------------------------------------

def test_v3_needs_the_extended_library_and_says_so() -> None:
    """The text of ``v3`` names ``robot.docs``, typed results and ``ora_lib``, which a ``base`` script lacks."""
    for bad in ({"prompt": "v3"}, {"prompt": "v3", "library": "base"}):
        with pytest.raises(ValueError, match=r"prompt v3 is written for library extended.*set "
                                             r"policy.code_mode.library to extended, or use prompt v2 with base"):
            resolve_contract(bad)
    with pytest.raises(ValueError, match="prompt must be one of v2, v3"):
        resolve_contract({"prompt": "v4", "library": "extended"})
    contract = resolve_contract({"prompt": "v3", "library": "extended"})
    assert contract["prompt"] == "v3" and contract["library"] == "extended"
    assert resolve_contract(contract) == contract, "a resolved v3 contract resolves to itself"
    assert resolve_contract({"library": "extended", "prompt": "v2"})["prompt"] == "v2", "v2 keeps its extended form"


def test_the_defaults_can_become_extended_and_v3_without_changing_a_base_plan_or_a_sealed_contract(monkeypatch) -> None:
    """When ``DEFAULT_LIBRARY`` is ``extended`` and ``DEFAULT_PROMPT`` ``v3``, a new plan gets both; a plan that asks
    for ``library: base`` is the control it always was (``v2``, no library recorded); and a contract resolved before
    either field existed, or before ``v3``, still resolves to itself."""
    sealed = resolve_contract({"prompt": "v2"})
    monkeypatch.setattr(contract_module, "DEFAULT_LIBRARY", "extended")
    monkeypatch.setattr(contract_module, "DEFAULT_PROMPT", "v3")
    new = resolve_contract(True)
    assert new["prompt"] == "v3" and new["library"] == "extended"
    control = resolve_contract({"library": "base"})
    assert control == sealed and "library" not in control
    assert resolve_contract(sealed) == sealed and script_library(sealed) == "base"
    with pytest.raises(ValueError, match="prompt v3 is written for library extended"):
        resolve_contract({"library": "base", "prompt": "v3"})


# --- v2 and every base contract compose as before --------------------------------------------------------------

#: sha256 of what a synthetic code-mode plan composes (its ``policy_composition`` and ``policy_context``: tool
#: definitions with their descriptions, the derived layer, prompt layers with their content hashes), at the commit
#: before ``prompt: v3`` existed. A change to any v2 text, or to how a ``base`` contract lists its functions,
#: changes one of these; that is the point: sealed runs and the control arm keep composing byte for byte.
#: Re-pinned once, by decision, when the Synthetic surface was written out as files and declared the toolkit
#: (2026-11-03): only its component paths and hashes, manifest and composition hashes, provenance and the
#: ``toolkit_only_v1`` capability moved; every definition, prompt layer and the derived layer are unchanged.
BEFORE_V3 = "09e5aff8cf1d0f59137dd379c8d0c6ef05d32179e97b273e643e785c89ba6336"


def composed(plan) -> str:
    value = {"composition": plan["policy_composition"], "context": plan["policy_context"]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_v2_and_the_base_library_compose_byte_for_byte_as_before_v3() -> None:
    for library in (None, "base"):
        plan = _plan({"prompt": "v2", **({"library": library} if library else {})})
        assert composed(plan) == BEFORE_V3, library
        assert "library" not in plan["policy"]["code_mode"]
        assert V3 not in _layers(plan)


# --- what v3 composes ------------------------------------------------------------------------------------------

def test_v3_replaces_the_planner_only_and_lists_the_functions_as_an_index() -> None:
    """Against ``v2`` with the extended library, the layers differ in the planner alone, and the tool definitions not
    at all: the core, the static facts and the descriptions are the same text."""
    v2 = _plan({"prompt": "v2", "library": "extended"})
    v3 = _plan({"prompt": "v3", "library": "extended"})
    assert v3["policy"]["code_mode"] == {**v2["policy"]["code_mode"], "prompt": "v3"}
    paths = _layers(v3)
    assert paths.count(V3) == 1
    assert [V3 if path == V2 else path for path in _layers(v2)] == paths
    surface = v3["policy_composition"]["tool_surface"]
    assert surface["definitions"] == v2["policy_composition"]["tool_surface"]["definitions"]
    (layer,) = [item for item in surface["derived_layers"] if item["kind"] == "code_mode_v1"]
    assert layer["contract"]["prompt"] == "v3" and layer["contract"]["library"] == "extended"
    description = surface["definitions"][-1]["description"]
    assert 'robot.docs("<name>")' in description and "### tools." not in description


def _robotwin(code_mode):
    from tests.support.orchestrator.run_specs import native_spec

    spec = native_spec("robotwin")
    spec["policy"]["code_mode"] = code_mode
    return spec


def _robodojo(baseline, profile):
    def build(code_mode):
        from tests.support.orchestrator.run_specs import robodojo_native_spec

        spec = robodojo_native_spec("general_pickup")
        spec["policy"].update(code_mode=code_mode, baseline=baseline)
        if profile:
            spec["policy"]["tool_surface"] = {"profile": profile}
        spec["toolkit_grants"] = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
        return spec
    return build


@pytest.mark.parametrize("build, reads", [
    pytest.param(_robotwin, [DRY_RUN_LAYER], id="robotwin-static-facts"),
    pytest.param(_robodojo("robodojo_toolkit_v6", None), [QUERY_LAYER, DRY_RUN_LAYER], id="robodojo-v6-code-forms"),
])
def test_v3_keeps_v2_s_layers_after_the_planner_on_every_surface(build, reads) -> None:
    """The static-facts layer (RoboTwin), the trajectory budget layer and the code forms of core v9 and surface v5's
    prompt (RoboDojo) follow the v3 planner as they follow the v2 one, and the layers of what the surface serves follow
    those, in ``v3`` only."""
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    v2 = _layers(resolve_run_spec(build({"prompt": "v2", "library": "extended"})))
    v3 = _layers(resolve_run_spec(build({"prompt": "v3", "library": "extended"})))
    assert V2 in v2 and V3 in v3 and V2 not in v3
    assert build is not _robotwin or CODE_CORE in v3, "v2's core follows the v3 planner"
    assert [path for path in v3 if path in (QUERY_LAYER, DRY_RUN_LAYER)] == reads
    assert not {QUERY_LAYER, DRY_RUN_LAYER} & set(v2)
    assert [V3 if path == V2 else path for path in v2] == [path for path in v3 if path not in reads]
    start = v3.index(V3)
    assert v3[start + 1:start + 1 + len(reads)] == reads or v3[start + 2:start + 2 + len(reads)] == reads, "after the budget"


# --- the interaction label -------------------------------------------------------------------------------------

def _libero(code_mode):
    from tests.support.orchestrator.run_specs import native_spec

    spec = native_spec("libero_pro")
    spec["policy"]["code_mode"] = code_mode
    return spec


def _description(build) -> str:
    """The code tool's description a surface composes, its index and card."""
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    plan = resolve_run_spec(build({"prompt": "v3", "library": "extended"}))
    return plan["policy_composition"]["tool_surface"]["definitions"][-1]["description"]


@pytest.mark.parametrize("build, call", [
    pytest.param(_robodojo("robodojo_toolkit_v6", None), 'targets={ARM: target(p_world, interaction="precision")}',
                 id="robodojo-in-each-arm"),
    pytest.param(_robotwin, 'targets={ARM: target(p_world, interaction="precision")}', id="robotwin-in-each-arm"),
    pytest.param(_libero, '**target(p_world, interaction="precision")', id="libero-flat"),
])
def test_a_surface_that_takes_the_label_says_to_set_it_and_shows_it_in_its_own_spelling(build, call) -> None:
    """The description follows ``move_pose``'s line with the instruction, the three values, when each applies and the
    call as that surface spells it; the card's ``target`` keeps the parameter."""
    text = _description(build)
    lines = text.splitlines()
    (at,) = [index for index, line in enumerate(lines) if line.startswith("- tools.move_pose(")]
    note = lines[at + 1]
    assert note.startswith("  Label every `move_pose` target, as `interaction`, by physical intent, not by controller name: ")
    for value in ("free_space for ", "precision for ", "contact_approach for ", "omitted label means"):
        assert value in note, value
    assert f"`tools.move_pose({call})`" in note
    assert "interaction=None" in next(line for line in lines if line.startswith("- target("))


def test_the_card_says_what_speed_costs_in_the_tool_s_own_words_and_opens_with_a_call_that_runs() -> None:
    """The arguments a call cannot guess follow ``move_pose``'s note with their schema's descriptions, and on a surface
    with ``targets`` the opening example is such a call; a moving tool's ordinary call is shown, not only the labelled one."""
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    plan = resolve_run_spec(_robodojo("robodojo_toolkit_v6", None)({"prompt": "v3", "library": "extended"}))
    surface = plan["policy_composition"]["tool_surface"]["definitions"]
    text = surface[-1]["description"]
    lines = text.splitlines()
    (at,) = [index for index, line in enumerate(lines) if line.startswith("- tools.move_pose(")]
    assert "`tools.move_pose(targets={ARM: target(p_world)})` through open space" in lines[at + 1]
    assert lines[at + 2].startswith("  `speed`: Optional; default fast") and "Slower speeds use more native actions" in lines[at + 2]
    assert lines[at + 3].startswith("  `slow_approaching`: Optional; default false")
    assert "Null means omitted" not in lines[at + 2] + lines[at + 3]
    assert 'r = tools.move_pose(targets={"left": target(p_world)})' in text and "position_m=[...]" not in text


def test_forced_preview_is_refused_and_the_optional_dry_run_runs_nothing_through_go() -> None:
    """``preview`` was retired on 2026-11-05: a RunSpec that asks for it is refused, and the plan without it states the
    optional ``dry_run`` (its own layer after the v3 planner) and never ``.go()``."""
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    build = _robodojo("robodojo_toolkit_v6", None)
    with pytest.raises(ValueError, match="preview .* was retired on 2026-11-05"):
        resolve_run_spec(build({"prompt": "v3", "library": "extended", "preview": True}))
    plan = resolve_run_spec(build({"prompt": "v3", "library": "extended"}))
    layers = _layers(plan)
    assert V3 in layers and DRY_RUN_LAYER in layers
    text = plan["policy_composition"]["tool_surface"]["definitions"][-1]["description"]
    assert "dry_run?" in text and ".go()" not in text


def test_the_base_library_keeps_its_opening_example() -> None:
    """``base`` lists every schema after the sketch it has always had: its text does not change."""
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    plan = resolve_run_spec(_robodojo("robodojo_toolkit_v6", None)({"prompt": "v2", "library": "base"}))
    text = plan["policy_composition"]["tool_surface"]["definitions"][-1]["description"]
    assert "from robot import tools, handoff\n    r = tools.move_pose(position_m=[...], ...)" in text


def test_the_extended_description_states_every_per_command_limit_from_the_contract() -> None:
    """Scripts ran into the query allowance (248 refusals), kept files in /tmp, ran out of memory and called check_task,
    none of which the description said: with the extended library it states each, from the contract; a base
    description keeps its text (``BEFORE_V3`` pins it)."""
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    build = _robodojo("robodojo_toolkit_v6", None)
    plan = resolve_run_spec(build({"prompt": "v3", "library": "extended", "max_queries_per_script": 20,
                                   "query_time_s": 90, "script_cpu_time_s": 60, "script_memory_mib": 1024}))
    surface = plan["policy_composition"]["tool_surface"]
    text = " ".join(surface["definitions"][-1]["description"].split())
    assert ("Per command: at most 12 actions (and at most half of the robot decisions left), 24 other robot calls, 20 "
            "queries and dry runs (90 s of them), 300 s of wall time, 60 s of CPU time and 1024 MiB of memory for its "
            "programs. `/tmp` is a fresh, empty directory for every command: keep what a later command reads in /work.") \
        in text
    direct = [item["name"] for item in surface["definitions"] if item["name"] not in coding.CODING_TOOLS
              and item["name"] in contract_module.NOT_FROM_SCRIPTS | set(surface["lifecycle_tools"])]
    assert "check_task" in direct and f"A script cannot call {' or '.join(f'`{name}`' for name in direct)}: call" in text
    for stated in (20, 90, 60, 1024):
        assert str(stated) in text
    base = resolve_run_spec(build({"prompt": "v2", "library": "base"}))
    told = " ".join(base["policy_composition"]["tool_surface"]["definitions"][-1]["description"].split())
    assert "24 other robot calls, 300 s. If a command's programs executed any action" in told
    assert "/tmp` is a fresh" not in told and "cannot call `check_task`" not in told and "MiB" not in told


def test_a_surface_that_does_not_take_the_label_is_told_nothing_of_it() -> None:
    """The synthetic surface's ``move_pose`` has fields of its own: no line, prompt layer or part of the card names
    the label, and the planners name it nowhere, so that holds on every surface."""
    from ora.agent.code_mode import docs

    plan = _plan({"prompt": "v3", "library": "extended"})
    definitions = plan["policy_composition"]["tool_surface"]["definitions"]
    text = definitions[-1]["description"]
    assert "- tools.move_pose(" in text, "this surface has a move_pose"
    layers = [(ROOT / path).read_text() for path in _layers(plan)]
    for shown in (json.dumps(definitions), *layers):
        assert not re.search(r"\binteraction\b|free_space|contact_approach", shown)
    assert not re.search(r"interaction|free_space", docs.fragment_parts("target", interaction=False)["description"])
    assert "interaction" in docs.fragment_parts("target")["description"], "and is the library's on a surface that has it"


def test_what_a_surface_is_told_goes_by_its_own_schema_on_every_registered_surface() -> None:
    """Every profile: the note and the card's parameter are there exactly where the composed ``move_pose`` schema names
    ``interaction``, in the spelling of that schema, and ``move_to`` takes it nowhere."""
    from ora.agent.code_mode import docs
    from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from tests.support.agent.surface_catalog import catalog_grants

    forms = {}
    for profile in PROFILE_CATALOG:
        definitions = resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile)).definitions
        tools = {definition["name"]: definition for definition in definitions}
        declared = '"interaction"' in json.dumps(tools.get("move_pose", {}).get("inputSchema", {}))
        assert docs.takes_interaction(definitions) is declared, profile
        assert '"interaction"' not in json.dumps(tools.get("move_to", {}).get("inputSchema", {})), profile
        entries = {"move_pose": docs.entry_parts(tools["move_pose"], reads=())} if "move_pose" in tools else {}
        assert docs.labelled(entries) is declared, profile
        listing = docs.listing(definitions, reads=())
        assert ("Label every `move_pose`" in listing) is declared, profile
        assert ("interaction=None" in listing) is declared, profile
        forms[profile] = docs.interaction_form(tools.get("move_pose", {}))
    assert {form for form in forms.values()} == {"arm", "flat", None}
    assert {profile for profile, form in forms.items() if form == "flat"} == {
        "libero_placement_direct_overhead_v1", "robocasa_mobile_direct_v1", "robolab_direct_v1"}
    assert {profile for profile, form in forms.items() if form is None} == {
        "robot_runtime_v2", "native_atomic_causal_v1", "local_atomic_cpu_v1"}


# --- the planners ----------------------------------------------------------------------------------------------

def _text() -> str:
    return (ROOT / V3).read_text()


def _section(text: str, heading: str) -> str:
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert match, heading
    return match.group(1)


def _triggers() -> list[str]:
    section = _section(_text(), "What to reach for")
    return [" ".join(item.split()) for item in re.findall(r"^- (\*\*.*?)(?=^- \*\*|\Z)", section, re.S | re.M)]


def test_the_planner_is_a_procedure_and_a_short_list_of_trigger_lines() -> None:
    """The procedure, the numbers and frames rule, the triggers, when to hand back and one worked script. The triggers
    are one bullet each, ``**when it applies:** what to call``: a capability that arrives later adds its own line to
    this list, beside the library's."""
    text = _text()
    headings = re.findall(r"^## (.*)$", text, re.M)
    for heading in ("Numbers and frames", "What to reach for", "Hand back to think", "Example"):
        assert heading in headings
    triggers = _triggers()
    assert len(triggers) == 11
    for trigger in triggers:
        assert re.fullmatch(r"\*\*[^*]+:\*\* .+", trigger) and len(trigger) < 400, trigger
    assert len(re.findall(r"^```", text, re.M)) == 2, "one worked script"
    for step in ("Measure", "Compute", "Act", "Re-measure"):
        assert f"**{step}**" in text
    assert "allowance_left" in text and "frames_card" in text and "robot.docs" in text
    assert re.findall(r"^- \*\*[^*]+:\*\*", _section(text, "Hand back to think"), re.M) == [
        "- **perception is ambiguous:**", "- **a plan is infeasible:**", "- **a check fails:**"]


def test_the_planner_says_the_description_settles_a_call_and_a_lookup_goes_with_the_work() -> None:
    text = " ".join(_text().split())
    assert "Look a function up" not in text and "before you first call it" not in text
    assert "take the argument shapes from `robot.docs`" not in text
    assert "Each line gives the call's arguments and its result" in text
    assert "is for what a line does not settle" in text
    assert "a lookup goes in the same command as the work that needs it, several in one command" in text
    assert "`target` is `robot`'s, not `ora_lib`'s" in text
    header = coding.BASH_INDEX_HEADER
    assert "Look a function up" not in header and "in the same command as the work" in " ".join(header.split())
    assert "comes from `robot`, not `ora_lib`" in " ".join(header.split()) and "`r[\"result\"]`" in header
    assert "`Result`" in header


@pytest.mark.parametrize("layer, says", [(QUERY_LAYER, ("robot.query", "`robot.docs(\"query_ik\")`")),
                                         (DRY_RUN_LAYER, ("dry_run=True", "`r.feasible`", "`r.predicted_pose`"))])
def test_a_read_a_surface_serves_is_a_trigger_line_of_its_own_layer(layer, says) -> None:
    text = (ROOT / layer).read_text()
    for trigger in re.findall(r"^- (\*\*.*?)(?=^- \*\*|\Z)", text, re.S | re.M):
        assert re.fullmatch(r"\*\*[^*]+:\*\* .+", " ".join(trigger.split()))
    for word in says:
        assert word in " ".join(text.split()), word
    assert "dry_run_unavailable" not in text, "the layer is composed only where it holds"
    assert not re.search(r"dry_run|robot\.query", _text()), "the planner names none of them"


def test_the_two_ways_to_ask_before_a_move_say_which_question_each_answers() -> None:
    """``reach_check`` holds a pose, ``dry_run`` runs the call's plan. Each layer also composes without the other (every
    surface that moves a robot serves dry runs, only the toolkit's queries), so neither names the other."""
    query = " ".join((ROOT / QUERY_LAYER).read_text().split())
    dry = " ".join((ROOT / DRY_RUN_LAYER).read_text().split())
    assert '**You are choosing among poses and ask which the arm can hold:** `robot.query("reach_check"' in query
    assert "no motion planned, so not whether a move there will run" in query and "dry_run" not in query
    assert "**You have the call and want to see it before it runs:** pass `dry_run=True`" in dry
    assert "It plans that call as the real one does" in dry and "reach_check" not in dry
    assert "It predicts the arm, not the world" in dry and "`r.failure_code`" in dry
    assert "The same call on the same observation then runs exactly that path." in dry
    assert "near objects, while holding something, or for a long move" in dry
    assert contract_module.with_reads({"library": "extended"}, toolkit=True, moves=True)["script_reads"] == [
        "dry_run", "query"]
    assert contract_module.with_reads({"library": "extended"}, toolkit=True, moves=False)["script_reads"] == [
        "query"]


def test_the_planner_names_only_capabilities_that_exist() -> None:
    """``images.annotate`` does not exist yet; every function, method and result field the planner mentions is the
    library's, and each capability the library has is named, or a model never finds it."""
    text = _text()
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    assert "images.annotate" not in text
    for capability in ("robot.explain", "r.explain", "object_geometry", "skimage", "shapely", "`queries`"):
        assert capability in " ".join(text.split()), capability
    calls = {name for span in re.findall(r"`([^`]*)`", prose) for name in re.findall(r"\b([A-Za-z_]+)\(", span)}
    methods = {name for cls in (ora_lib.Point, ora_lib.Direction, ora_lib.Pose, ora_lib.Pixel) for name in dir(cls)}
    known = set(ora_lib.__all__) | set(robot.__all__) | methods | {"print", "help", "bash"}
    assert calls <= known, sorted(calls - known)
    assert {"pixels_to_world", "project", "direction", "offset", "frames_card"} <= calls
    fields = set(re.findall(r"`r\.(\w+)", prose))
    assert fields == {"ok", "status", "failure_code", "summary", "target_error_m", "arms", "explain"}
    result = robot.Result("move_pose", {"ok": True, "result": {}})
    assert all(hasattr(result, field) for field in fields | {"feasible", "predicted_pose"})
    schema = extra_result_schemas(resolve_contract({"library": "extended"}))["bash"]
    assert ALLOWANCE_FIELD == "allowance_left" and ALLOWANCE_FIELD in schema["properties"]


# --- the worked scripts run ------------------------------------------------------------------------------------

STATUS = sandbox_status()
sandboxed = pytest.mark.skipif(not STATUS["ok"], reason=f"code-mode sandbox unavailable: {STATUS.get('reason')}")
LIMITS = {"wall_s": 60, "cpu_s": 30, "memory_mib": 2048}
RECORDED = json.loads((ROOT / "tests/data/code_mode/script-answers.json").read_text())["answers"]
CAMERA, ARM = "head", "left"  # the camera of the RoboDojo fixture, and the arm of the recorded move_pose answers
#: ``<move>`` as each form of ``move_pose`` takes the target ``t``: under the arm in ``targets``, or one arm's fields.
MOVES = {"targets": "targets={ARM: t}", "flat": "**t"}
BOX = [300, 280, 340, 320]    # the block's box in the head camera: its centre pixel is (320, 300)


def recorded_move(*, failed: bool) -> dict:
    """A recorded RoboDojo ``move_pose`` answer for the left arm: reached (error 1.9 mm), or ``planner_failed``."""
    for item in RECORDED:
        answer = item["answer"]
        if item["backend"] != "robodojo" or answer["tool"] != "move_pose":
            continue
        receipt = answer["result"]["receipt"]
        if receipt["detail"]["arms"].get(ARM, {}).get("targeted") and bool(receipt.get("failure_code")) == failed:
            return json.loads(json.dumps(answer))
    raise AssertionError("no recorded move_pose answer")


class Host:
    """The host of a program: ``vision_ground`` answers one box; ``move_pose`` answers a recorded answer and, as the
    session does, publishes the observation after the move with the arm ``off`` metres from the target."""

    def __init__(self, published, *, failed: bool = False, off=(0.0, 0.0, 0.0)) -> None:
        self.published, self.failed, self.off, self.calls = published, failed, off, []

    def __call__(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name == "vision_ground":
            return {"ok": True, "tool": name, "result": {"candidates": [{"rank": 0, "bbox_xyxy": BOX}]}}
        assert name == "move_pose", name
        target = arguments["targets"][ARM] if "targets" in arguments else arguments
        if not self.failed:
            self.publish_after([a + b for a, b in zip(target["position_m"], self.off)])
        return recorded_move(failed=self.failed)

    def publish_after(self, xyz) -> None:
        pointer_path = self.published.control_root / "current-observation.json"
        pointer = json.loads(pointer_path.read_bytes())
        source = Path(pointer["source_root"])
        root = source.parent / "observation-0002"
        shutil.copytree(source, root)
        manifest = json.loads((root / "observation.json").read_bytes())
        manifest.update(observation_id="observation-0002", sequence=2)
        manifest["proprioception"]["end_effectors"][ARM]["pose_xyz_xyzw"][:3] = list(xyz)
        (root / "observation.json").write_bytes(canonical(manifest))
        (root / "proprioception.json").write_bytes(canonical(manifest["proprioception"]))
        pointer.update(source_root=str(root), observation_id="observation-0002", policy_state={"step": 2},
                       observation_sha256=digest((root / "observation.json").read_bytes()))
        pointer_path.write_bytes(canonical(pointer))


def worked(form: str = "targets") -> str:
    """The planner's worked program with its three placeholders filled, the move in ``form`` (``MOVES``)."""
    (block,) = re.findall(r"^```bash\n(.*?)^```", _text(), re.S | re.M)
    match = re.fullmatch(r"cat > /work/hover.py <<'EOF'\n(.*?)^EOF\npython /work/hover.py\n", block, re.S | re.M)
    assert match, "a heredoc that writes the program, and the python that runs it"
    block = match.group(1)
    assert set(re.findall(r"<(\w+)>", block)) == {"camera", "arm", "move"}
    assert "targets=" not in _text(), "the planner is one text for every surface: neither form is written into it"
    return block.replace("<camera>", CAMERA).replace("<arm>", ARM).replace("<move>", MOVES[form])


def execute(published, tmp_path, host, form: str = "targets"):
    """The worked program run in the sandbox as the session runs it: the file a ``bash`` command's ``python`` runs.
    Returns its outcome and the dict it printed."""
    code = worked(form)
    with observation_mount({"library": "extended"}, published.control_root) as mount:
        (tmp_path / "work").mkdir()
        (tmp_path / "work" / "hover.py").write_text(code)
        outcome = run_command("python hover.py", handle_call=host, workspace=tmp_path / "work", extended=True,
                              observation=mount, **LIMITS)
    lines = outcome.output.strip().splitlines()
    return outcome, ast.literal_eval(lines[-1]) if outcome.status == "completed" else None


def expected_target(published, tmp_path):
    """Where the program must send the arm: the box's centre pixel lifted over the published depth, 12 cm up."""
    mount = ObservationMount(published.control_root, tmp_path / "expected")
    mount.root.mkdir()
    mount.refresh()
    lifted = ora_lib.pixels_to_world(CAMERA, [[(BOX[0] + BOX[2]) / 2, (BOX[1] + BOX[3]) / 2]],
                                     observation=library_observation.load(str(mount.root)))
    return (lifted.xyz[0] + [0.0, 0.0, 0.12]).tolist()


def _move_pose_schema(profile: str) -> dict:
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from tests.support.agent.surface_catalog import catalog_grants

    definitions = resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile)).definitions
    return next(definition for definition in definitions if definition["name"] == "move_pose")["inputSchema"]


@sandboxed
@pytest.mark.parametrize("profile, call, compiler", [
    pytest.param("robodojo_placement_toolkit_v5", 'tools.move_pose(targets={"left": target(p_world, interaction=LABEL)})',
                 "ora.agent.toolkit.tools.move_pose._arguments", id="in-each-arm"),
    pytest.param("libero_placement_direct_overhead_v1", "tools.move_pose(**target(p_world, interaction=LABEL))",
                 "ora.agent.toolkit.host.motion_rules.arguments", id="flat"),
])
def test_a_labelled_target_reaches_the_tool_as_its_schema_takes_it(profile, call, compiler, tmp_path) -> None:
    """The call each surface's description shows, from a program in the real sandbox: what reaches the host is valid
    against that surface's own ``move_pose`` schema, and the tool's compiler reads the label from it."""
    from ora.agent.composition.surface.schema_validation import validate_schema_instance

    calls = []

    def host(name, arguments):
        calls.append((name, dict(arguments)))
        return {"ok": True, "tool": name, "result": {}}

    head = ("from ora_lib import Point\nfrom robot import target, tools\nLABEL = 'precision'\n"
            "p_world = Point([0.3, 0.1, 0.9], 'world')\n")
    (tmp_path / "move.py").write_text(head + call + "\n")
    outcome = run_command("python move.py", handle_call=host, workspace=tmp_path, extended=True, **LIMITS)
    assert outcome.status == "completed", outcome.output
    ((name, arguments),) = calls
    validate_schema_instance(arguments, _move_pose_schema(profile), root_path="$input")
    written = arguments["targets"]["left"] if "targets" in arguments else arguments
    assert name == "move_pose" and importlib.import_module(compiler).arm_target(written)["interaction"] == "precision"


@sandboxed
@pytest.mark.parametrize("form", list(MOVES))
def test_the_worked_program_lifts_a_pixel_moves_with_a_typed_target_and_checks_where_the_arm_ended(
        published, tmp_path, form) -> None:
    """In either form of ``move_pose``: under the arm in ``targets``, or one arm's fields (a single-arm surface)."""
    host = Host(published)
    outcome, reported = execute(published, tmp_path, host, form)
    assert outcome.status == "completed", (outcome.error, outcome.output)
    (ground_name, ground), (move_name, move) = host.calls
    assert (ground_name, ground) == ("vision_ground", {"camera": CAMERA, "text": "red block"})
    target = move["targets"][ARM] if form == "targets" else move  # the typed Point became the tool's own mode fields
    assert move_name == "move_pose" and (set(move) == {"targets"} or form == "flat")
    assert target == {"position_mode": "absolute", "position_m": pytest.approx(expected_target(published, tmp_path),
                                                                               abs=1e-9),
                      "orientation_mode": "hold_current"}
    assert reported["miss_m"] == pytest.approx(0.0, abs=1e-9)
    assert reported["above_world"] == pytest.approx(target["position_m"], abs=1e-9)


@sandboxed
def test_the_worked_program_hands_back_on_a_failed_move(published, tmp_path) -> None:
    """``r.ok`` is what stops it: a recorded ``planner_failed`` answer, read as ``r.status``, ``r.failure_code`` and
    ``r.target_error_m`` (from the receipt's ``pose_error``)."""
    outcome, _ = execute(published, tmp_path, Host(published, failed=True))
    assert outcome.status == "handed_back" and outcome.handed_back["reason"] == "move failed"
    assert outcome.handed_back["data"] == {"status": "planner_failed", "failure": "planner_failed",
                                           "error_m": pytest.approx(0.0448173, abs=1e-6)}


@sandboxed
def test_the_worked_program_re_measures_and_hands_back_when_the_arm_ended_off_target(published, tmp_path) -> None:
    """A move the receipt calls reached, from an arm the next observation shows 5 cm off: the re-measurement reads
    the observation the move published and compares it with the stated tolerance."""
    outcome, _ = execute(published, tmp_path, Host(published, off=(0.0, 0.0, 0.05)))
    assert outcome.status == "handed_back" and outcome.handed_back["reason"] == "gripper missed its target"
    assert outcome.handed_back["data"] == {"miss_m": pytest.approx(0.05, abs=1e-6), "tol_m": 0.02}
