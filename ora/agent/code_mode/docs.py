"""The documentation of code mode's script functions (``policy.code_mode.library: extended``), from one source.

The source is each function's ToolSurface definition as the plan resolves it -- for a toolkit tool its
``description.json`` (``ora/agent/toolkit/tools/<name>/``): name, description and input schema -- and the
surface's result schema for it. Two forms are generated from it and nothing is written by hand:

- ``index_line``: one line per function, which the script tool's description carries instead of every tool's full
  description and schema (``coding.py``): the call with its arguments' nested shape (``?`` marks an optional one,
  enums spelled out, an array's length, ``...`` for what a depth or character limit left out, ``shape``), the result's
  shape to ``RESULT_DEPTH`` and the description's first sentence, its purpose, enough to write the common call without
  a lookup. ``move_pose``'s line is followed by ``interaction_note`` where its schema takes ``interaction``, and
  ``move_base``'s by the rule that joins its two forms (``argument_notes``).
- ``entry``: a function's documentation, which a script looks up (``robot.docs(name)``, ``help(tools.<name>)``,
  served by the policy session, ``agent/pi/code_mode.py``). It is a sequence of named parts (``PARTS``), each
  looked up alone with ``robot.docs(name, part)``; a lookup with no part is ``short`` (the call, the description and the
  arguments within ``SHORT_CHARS``, then the name and size of every other part), and ``robot.docs(name, "all")`` the
  whole entry. The parts: ``description`` (the call and the description verbatim),
  ``arguments`` (every argument at every depth, ``a.b``, ``a[]``, with its type, constraints and the schema's own
  description: units and frames as the schema states them), ``result`` (the fields of the tool's result, nested by
  indentation), ``dry_run`` for the arm-moving tools that take ``dry_run=True`` (what a dry run returns instead, from
  ``toolkit/interface/dry_run.py``), ``explain`` for a function whose tool has a stage template (the stages
  ``robot.explain`` renders its calls by, ``toolkit/explain.py``) and, where given, ``examples`` (``EXAMPLES``; those
  tools' carry one with ``dry_run``), written as the surface's own schema takes them (``examples``). The index ends
  with one line per query (``robot.query``, ``query_spec.py``) and each query has an entry (``query_entries``), named
  ``query.<name>`` and looked up as ``query_<name>``; their arguments and results are described by the schemas of
  ``query_spec.py``.
- ``listing``: the index, then the library card (``library_card.py``, by introspection), whose names are looked up as
  a function is (``fragment_parts``).

What a script can do only where its surface serves it (``READS``, a resolved contract's ``script_reads``, decided once
in resolution, ``contract.with_reads``) appears only there: the generators take ``reads`` and leave the rest out, a
``dry_run`` argument, part and example, the query lines and entries, the card's ``query`` line and a dry
run's fields. With no ``reads`` they show everything; the session and the descriptions always pass the contract's.
``interaction`` is read off the definitions themselves (``takes_interaction``): a surface whose ``move_pose`` has none
is told nothing of it, in the index, the card or ``robot.docs("target")``.

Tool-call mode reads the same definitions unchanged: nothing here rewrites them.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Collection, Iterable, Mapping, Sequence

from ora.agent.composition.surface.schema_validation import SchemaViolation, validate_schema_instance
from ora.agent.toolkit.interface import dry_run

from . import library_card, query_spec
from .library_card import READS, card, fragment_parts, library_names, members

#: An entry's parts, in order; a part without content (a tool without a result schema or examples) is left out.
PARTS = ("description", "arguments", "result", "dry_run", "explain", "examples")
#: The longest purpose an index line carries (a first sentence longer than this is cut at a word).
PURPOSE_CHARS = 300
#: How deep an index line opens a function's arguments and its result (a level is one nested object), and the most
#: characters it spends on each: past them a deeper level closes as ``{...}``, then the members that do not fit go,
#: said by ``...``.
ARGUMENT_DEPTH, RESULT_DEPTH = 3, 2
ARGUMENT_CHARS, RESULT_CHARS = 620, 640
#: The part name of a whole entry (``robot.docs(name, "all")``).
ALL = "all"
#: The most characters a lookup with no part shows (``short``).
SHORT_CHARS = 6000
#: Schema keywords a line states in words; any other keyword is stated as ``keyword=value``.
_STATED = frozenset({"type", "description", "properties", "items", "prefixItems", "required", "additionalProperties",
                     "oneOf", "anyOf", "allOf", "$defs", "definitions", "$ref", "enum", "const", "default",
                     "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems",
                     "minLength", "maxLength", "minProperties", "maxProperties", "pattern", "uniqueItems", "format"})
_ALTERNATIVES = (("oneOf", "one of these forms"), ("anyOf", "any of these forms"), ("allOf", "all of these"))
#: What a tool's entry shows besides what its definition generates: where the library has a fragment for what the
#: tool returns, a use of it (``examples``, by the function's wire name). ``{camera}`` is the first camera the tool's
#: schema names (``head`` without a definition).
EXAMPLES: dict[str, tuple[str, ...]] = {
    "vision_ground": ('from ora_lib import object_geometry\n'
                      'candidate = tools.vision_ground(camera="{camera}", text="red cup")["result"]["candidates"][0]\n'
                      'cup = object_geometry(candidate)  # its mask on the camera\'s depth, in world metres\n'
                      'print(cup.extent_xyz_m, cup.height_above_support_m, cup.mid_height_z_m)',),
    "vision_segment": ('from ora_lib import object_geometry\n'
                       'seen = tools.vision_segment(camera="{camera}", points_uv=[[320, 240], [330, 250]])["result"]\n'
                       'cup = object_geometry(seen["candidates"][0])  # the first mask of the first point\n'
                       'print(cup.extent_xyz_m, cup.support_z_m)',),
}
#: The camera ``EXAMPLES`` name where no definition gives one.
EXAMPLE_CAMERA = "head"
#: A motion tool's ``dry_run`` argument and what it returns, as an entry states them (``accepts_dry_run`` tools).
DRY_RUN_ARGUMENT = (
    "boolean, default false. true returns what this call would do without doing it: the route, the screening, the "
    "predicted end pose, per arm the native actions, how far the hand travels and how "
    "much it turns, how high its path rises above its ends and where, and the least clearance between the jaws and "
    "what the depth images show, and a picture of the path "
    "on the cameras that see it, in the shape the dry_run part gives. It runs the planner the real call runs before it "
    "acts and stops there: nothing is executed or proposed and no native action is spent. It predicts the robot's own "
    "motion on the measured state, never the world's response: no object moves, no contact, no grasp outcome, nothing "
    "after the first contact, and a dashed line is a servo's commanded line, not a planned path. The same call on the "
    "same observation then runs exactly the path shown. The numbers and the picture reach you with the script's result "
    "whether it prints them or not. Worth a call near objects, while holding something, or for a long move. A dry run "
    "counts against the script's query allowance, not its calls or actions. Where a move cannot be planned ahead without "
    "moving, the call is refused dry_run_unavailable and constraints.reason says why.")
#: Introduces the ``dry_run`` part (its fields follow, ``interface/dry_run.RESULT_SCHEMA``).
DRY_RUN_RESULT = (
    "Dry run (dry_run=True): the call returns this record instead of its result, and executes nothing. A plan the real "
    "call would refuse (an unreachable pose, a blocked path, a target outside the workspace) is an answer, "
    "feasible false with the real call's failure_code, not an error; a malformed argument still raises ToolRefused. "
    "Poses are world [x, y, z, qx, qy, qz, qw] in metres, angles in degrees; each arm's record also gives its "
    "predicted pose's world axes, predicted_approach_world and predicted_jaw_line_world (as move_pose's world_axes "
    "takes them), where the robot's controlled frame is published. As a Result (a Plan): .dry_run, .feasible, "
    ".predicted_pose and .end_joints_deg (the one arm it moves), .predicted_poses, .route, .screening, .summary, "
    ".failure_code, .failure_stage and .explain; a run's fields (.ok, .status, .pose_after, ...) raise PlanNotRun, "
    "since nothing has run: .feasible says whether it would.\n")
#: The dry run's own example of each tool that takes ``dry_run``: ask first, read ``feasible`` and ``explain``, then act.
_DRY_EXAMPLE = (
    'r = tools.{call}, dry_run=True)  # the plan the real call would run; nothing moves, no action is spent\n'
    'if r.feasible:\n'
    '    print(r.summary)  # then make the same call without dry_run: it runs this path\n'
    'else:\n'
    '    print(r.failure_code, r.failure_stage)\n'
    '    print(robot.explain(r))  # what failed, why, and what to change')
#: Each such tool's example arguments, one per form a surface's schema may take (two arms under ``targets``, one arm's
#: fields directly): the example shows the first that this surface's schema validates, the first without a definition.
_DRY_CALLS = {
    "move_pose": ('targets={"left": {"position_mode": "delta", "translation_m": [0.0, 0.0, -0.05], '
                  '"position_frame": "world", "orientation_mode": "hold_current"}}',
                  'position_mode="delta", translation_m=[0.0, 0.0, -0.05], position_frame="world", '
                  'orientation_mode="hold_current"'),
    "move_to": ('targets={"left": {"position_m": [0.30, 0.10, 0.95]}}', 'position_m=[0.30, 0.10, 0.95]'),
    "rotate_wrist": ('targets={"left": 30.0}', 'angle_deg=30.0'),
    "rotate_pitch": ('targets={"left": -20.0}', 'angle_deg=-20.0'),
    "recover_pose": ('arms=["left"]', 'targets={"left": "previous"}', 'strategy="previous"'),
    "return_home": ('arms=["left"]',),
    "set_gripper": ('targets={"left": "close"}', 'action="close"'),
    "grasp_execute": ('candidate_id="g0", rationale="the nearest candidate"',),
    "move_base": ('forward_m=0.3',),
}
#: After ``move_pose``'s index line where its schema takes ``interaction``: ``target()`` writes the label only when it is
#: asked to and an omitted one means free_space, so the index says when each applies, in the terms of the tool's
#: description. What each one does differs by robot and stays in that description.
INTERACTION_NOTE = (
    "  Label every `move_pose` target, as `interaction`, by physical intent, not by controller name: free_space for a "
    "move through open space (what an omitted label means), precision for a small exact move near objects, "
    'contact_approach for the move into contact. The label decides how the arm gets there: `robot.docs("move_pose", '
    '"description")` says how on this robot. In use: `{open}` through open space, `{near}` for the small exact move.')
#: The plain and the labelled call, by where the schema takes the label: in each arm's target, or beside the other
#: arguments.
_LABELLED_CALLS = {
    "arm": ('tools.move_pose(targets={ARM: target(p_world)})',
            'tools.move_pose(targets={ARM: target(p_world, interaction="precision")})'),
    "flat": ('tools.move_pose(**target(p_world))', 'tools.move_pose(**target(p_world, interaction="precision"))'),
}
#: ``move_pose``'s arguments whose meaning a call cannot guess from their names: each one's own schema description
#: follows the index line, so the text stays the tool's, and speed's cost in native actions is read before a move.
_NOTED_ARGUMENTS = ("speed", "slow_approaching")
#: ``move_base``'s two forms (``toolkit/tools/move_base/tool.py``: ``ABSOLUTE``, ``RELATIVE``): its line marks all six
#: optional, so ``BASE_NOTE`` follows it with the rule the tool refuses by.
BASE_FORMS = (("x_m", "y_m", "yaw_deg"), ("forward_m", "left_m", "turn_deg"))
BASE_NOTE = ("  `move_base` takes either all of x_m, y_m and yaw_deg (a world target) or any of forward_m, left_m and "
             "turn_deg (relative to the current base pose), never a mix: `tools.move_base(x_m=..., y_m=..., "
             "yaw_deg=...)` or `tools.move_base(turn_deg=...)`.")
_SCALARS = {"string": "str", "integer": "int", "number": "number", "boolean": "bool"}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[])")
_MAX_DEPTH = 24


def wire_name(name: str) -> str:
    """The function's name in a script: a dotted tool name with underscores (as ``contract.wire_name``)."""

    return name.replace(".", "_")


def sentences(text: str) -> list[str]:
    """The text's sentences, each as written."""

    return [part for chunk in text.strip().split("\n") for part in _SENTENCE_END.split(chunk.strip()) if part]


def purpose(description: str) -> str:
    """The description's first sentence on one line: what the function is for."""

    first = " ".join(sentences(description)[0].split()) if description.strip() else ""
    if len(first) > PURPOSE_CHARS:
        first = first[:PURPOSE_CHARS].rsplit(" ", 1)[0] + " ..."
    return first


def accepts_dry_run(name: str) -> bool:
    """Whether the function takes ``dry_run`` (an arm-moving tool, ``interface/dry_run.TOOLS``)."""

    return str(name).rsplit(".", 1)[-1] in dry_run.TOOLS


def signature(definition: Mapping[str, Any], *, reads: Collection[str] = READS) -> str:
    """``tools.<name>(required, optional?)``: the top-level arguments, required ones first, in the schema's order; an
    arm-moving tool where dry runs are served also takes ``dry_run?`` (the library's, not the tool's schema), and a
    library function that is not a tool, a query, states its own call. A function's entry starts with this line and
    ``robot.py`` binds positional arguments by parsing it, so it holds names, ``, `` and ``?`` only: the shapes are
    ``call_shape``'s."""

    if isinstance(definition.get("signature"), str):
        return definition["signature"]
    schema = _tool_schema(definition, reads)
    properties = list(schema.get("properties") or {}) if isinstance(schema.get("properties"), Mapping) else []
    required = set(schema.get("required") or ())
    names = [name for name in properties if name in required] + [f"{name}?" for name in properties
                                                                 if name not in required]
    if takes_dry_run(definition["name"], reads):
        names.append(f"{dry_run.ARGUMENT}?")
    return f"tools.{wire_name(str(definition['name']))}({', '.join(names)})"


def _tool_schema(definition: Mapping[str, Any], reads: Collection[str]) -> Mapping[str, Any]:
    """The tool's input schema, without ``dry_run`` where these docs state it themselves."""

    schema = definition.get("inputSchema")
    if not isinstance(schema, Mapping):
        return {}
    properties = schema.get("properties")
    if not takes_dry_run(definition.get("name"), reads) or not isinstance(properties, Mapping):
        return schema
    return {**schema, "properties": {key: value for key, value in properties.items() if key != dry_run.ARGUMENT}}


def takes_dry_run(name: Any, reads: Collection[str]) -> bool:
    """Whether the function takes ``dry_run`` on this surface: it is an arm-moving tool and dry runs are served."""

    return accepts_dry_run(str(name)) and dry_run.ARGUMENT in reads


@dataclass
class _Shape:
    """A schema on one line: ``{a, b?: [3], c?: x|y}``, equal neighbours merged (``left?, right?: {...}``), an array's
    length (``[3]``, three numbers; ``[int; 2]``, two integers; ``[x; 1-8]``), ``{...}`` below ``depth``. ``typed``
    False (a result) names the members and leaves their scalar types and values out."""

    root: Mapping[str, Any]
    typed: bool = True

    def of(self, node: Any, depth: int, seen: tuple[str, ...] = ()) -> str:
        node, seen = _Walk(self.root, "", False).resolve(node, seen)
        if not isinstance(node, Mapping):
            return ""
        kinds = [kind for kind in _types(node) if kind != "null"]
        if isinstance(node.get("properties"), Mapping) or isinstance(node.get("additionalProperties"), Mapping):
            return "{" + ", ".join(self.members(node, depth, seen)) + "}" if depth > 0 else "{...}"
        for keyword, _word in _ALTERNATIVES:
            if isinstance(node.get(keyword), list):
                forms = dict.fromkeys(self.of(branch, depth, seen) or "..." for branch in node[keyword])
                return " or ".join(forms)
        if "enum" in node or "const" in node:
            values = node["enum"] if "enum" in node else [node["const"]]
            shown = "|".join(item if isinstance(item, str) else json.dumps(item) for item in values if item is not None)
            return shown if self.typed else ""
        if "array" in kinds:
            return self.array(node, depth, seen)
        return self.scalar(node, kinds)

    def rows(self, node: Mapping[str, Any], depth: int, seen: tuple[str, ...] = ()) -> list[tuple[str, str]]:
        """``node``'s members, required ones first in the schema's order: ``(name, shape)``, ``?`` on an optional one."""

        properties = node.get("properties") if isinstance(node.get("properties"), Mapping) else {}
        required = set(node.get("required") or ())
        rows = [(name if name in required else f"{name}?", self.of(properties[name], depth - 1, seen))
                for name in sorted(properties, key=lambda name: name not in required)]
        values = node.get("additionalProperties")
        return [*rows, ("<key>", self.of(values, depth - 1, seen))] if isinstance(values, Mapping) else rows

    def members(self, node: Mapping[str, Any], depth: int, seen: tuple[str, ...] = ()) -> list[str]:
        return _merged(self.rows(node, depth, seen))

    def array(self, node: Mapping[str, Any], depth: int, seen: tuple[str, ...]) -> str:
        if isinstance(node.get("prefixItems"), list):
            return "[" + ", ".join(self.of(item, depth, seen) or "..." for item in node["prefixItems"]) + "]"
        items, seen = _Walk(self.root, "", False).resolve(node.get("items"), seen)
        low, high = node.get("minItems"), node.get("maxItems")
        number = isinstance(items, Mapping) and set(_types(items)) & {"number", "integer"} and "enum" not in items
        if number and low is not None and low == high:  # integers said as such: a validator refuses 88.97 there
            return f"[int; {low}]" if self.typed and "number" not in _types(items) else f"[{low}]"
        inner = self.of(items, depth, seen)
        length = "" if not self.typed or (low is None and high is None) else (
            f"; {low or 0}-{high}" if high is not None else f"; {low}+")
        return f"[{inner}{length}]"

    def scalar(self, node: Mapping[str, Any], kinds: list[str]) -> str:
        if not self.typed:
            return ""
        word = "|".join(_SCALARS.get(kind, "dict" if kind == "object" else kind) for kind in kinds)
        low, high = node.get("minimum"), node.get("maximum")
        bounds = f"{low}..{high}" if low is not None and high is not None else (
            f">={low}" if low is not None else f"<={high}" if high is not None else "")
        return f"{word} {bounds}".strip()


def _merged(rows: Sequence[tuple[str, str]]) -> list[str]:
    """``name`` or ``name: shape`` per row, neighbours of one shape as one member: ``left?, right?: {...}``."""

    merged: list[str] = []
    names: list[str] = []
    for position, (name, shape) in enumerate(rows):
        names.append(name)
        if shape and position + 1 < len(rows) and rows[position + 1][1] == shape:
            continue
        merged.append(", ".join(names) + (f": {shape}" if shape else ""))
        names = []
    return merged


def _within(rows: Sequence[tuple[str, str]], limit: int) -> list[str]:
    """As many ``rows`` as fit ``limit`` characters, merged, then ``...`` if any did not."""

    for count in range(len(rows), 0, -1):
        shown = [*_merged(rows[:count]), *([] if count == len(rows) else ["..."])]
        if len(", ".join(shown)) <= limit:
            return shown
    return ["..."] if rows else []


def call_shape(definition: Mapping[str, Any], *, reads: Collection[str] = READS) -> str:
    """The call with its arguments' shape: ``tools.move_pose(targets: {left?, right?: {position_mode: a|b, ...}})``; a
    query says its name as its first argument."""

    schema = _tool_schema(definition, reads)
    shape = _Shape(schema)
    resolved = _Walk(schema, "", False).resolve(schema, ())[0]
    levels = (shape.members(resolved, level) for level in range(ARGUMENT_DEPTH, 0, -1))
    members = next((item for item in levels if len(", ".join(item)) <= ARGUMENT_CHARS), None)
    members = _within(shape.rows(resolved, 1), ARGUMENT_CHARS) if members is None else members
    if takes_dry_run(definition["name"], reads):
        members = [*members, f"{dry_run.ARGUMENT}?: bool"]
    if isinstance(definition.get("signature"), str):
        head = definition["signature"].split(",")[0].removesuffix(")")
        return ", ".join([head, *members]) + ")"
    return f"tools.{wire_name(str(definition['name']))}({', '.join(members)})"


def result_shape(schema: Mapping[str, Any]) -> str:
    """What a call's ``r["result"]`` holds, by name, to ``RESULT_DEPTH``: ``{status, candidates: [{bbox_xyxy: [4]}]}``;
    alternative forms share ``RESULT_CHARS``."""

    walk = _Walk(schema, "", False)
    resolved = walk.resolve(schema, ())[0]
    forms = [resolved]
    for keyword, _word in _ALTERNATIVES:
        forms = resolved[keyword] if isinstance(resolved.get(keyword), list) else forms
    return " or ".join(dict.fromkeys(
        "{" + (", ".join(_result_members(schema, walk.resolve(form, ())[0], RESULT_CHARS // len(forms))) or "...") + "}"
        for form in forms))


def _result_members(schema: Mapping[str, Any], node: Mapping[str, Any], limit: int) -> list[str]:
    """``node``'s members within ``limit`` characters: each one closed, then opened where it still fits, the fullest
    first: the arrays, for what a script reads first of a result is its list of candidates or marks, and the objects
    of a result with no array to open."""

    shape = _Shape(schema, typed=False)
    closed, opened = shape.rows(node, 1), shape.rows(node, RESULT_DEPTH)
    rows = dict(closed)
    spare = limit - len(", ".join(_merged(closed)))
    gains = sorted(((name, wide) for name, wide in opened if wide != rows[name]), key=lambda row: -len(row[1]))
    arrays = [row for row in gains if row[1].startswith("[")]
    for name, wide in arrays or gains:
        if len(wide) - len(rows[name]) <= spare:
            spare -= len(wide) - len(rows[name])
            rows[name] = wide
    return _within(list(rows.items()), limit)


def index_line(definition: Mapping[str, Any], result_schema: Mapping[str, Any] | None = None, *,
               reads: Collection[str] = READS) -> str:
    """One line: the call, its result, the purpose (and, for a query, the name its documentation is looked up by)."""

    text = purpose(str(definition.get("description") or ""))
    lookup = f' (robot.docs("{definition["lookup"]}"))' if definition.get("lookup") else ""
    result = f" -> {result_shape(result_schema)}" if result_schema else ""
    return f"- {call_shape(definition, reads=reads)}{result}" + (f": {text}" if text else "") + lookup


def interaction_form(definition: Mapping[str, Any]) -> str | None:
    """Where a ``move_pose`` definition takes ``interaction``: ``arm`` (in each arm's target, a surface with ``targets``),
    ``flat`` (with the other arguments, a single-arm surface) or None (another tool, or a ``move_pose`` without it)."""

    schema = definition.get("inputSchema")
    if definition.get("name") != "move_pose" or not isinstance(schema, Mapping):
        return None
    properties = schema.get("properties") or {}
    arms = (properties.get("targets") or {}).get("properties") or {}
    return ("flat" if library_card.INTERACTION in properties
            else "arm" if any(library_card.INTERACTION in (target.get("properties") or {}) for target in arms.values())
            else None)


def takes_interaction(functions: Iterable[Mapping[str, Any]]) -> bool:
    """Whether this surface's ``move_pose`` takes ``interaction``."""

    return any(interaction_form(definition) for definition in functions)


def labelled(entries: Mapping[str, Mapping[str, str]]) -> bool:
    """``takes_interaction``, from the entries the session keeps: their ``arguments`` name every argument at every depth."""

    return library_card.INTERACTION in entries.get("move_pose", {}).get("arguments", "")


def interaction_note(definition: Mapping[str, Any]) -> str:
    """What follows the function's index line about ``interaction``; nothing for any other function, or a ``move_pose``
    that does not take it."""

    form = interaction_form(definition)
    if form is None:
        return ""
    plain, near = _LABELLED_CALLS[form]
    return INTERACTION_NOTE.format(open=plain, near=near)


def argument_notes(definition: Mapping[str, Any]) -> str:
    """``move_pose``'s ``_NOTED_ARGUMENTS`` this schema has, each with its own description; for a ``move_base`` whose
    schema has both of ``BASE_FORMS``, the rule that joins them (``BASE_NOTE``); nothing for another tool."""

    schema = definition.get("inputSchema")
    if not isinstance(schema, Mapping):
        return ""
    properties = schema.get("properties") or {}
    if definition.get("name") == "move_base":
        return BASE_NOTE if all(name in properties for form in BASE_FORMS for name in form) else ""
    if definition.get("name") != "move_pose":
        return ""
    notes = (f"  `{name}`: {' '.join(sentences(str(properties[name].get('description') or '')))}"
             for name in _NOTED_ARGUMENTS if name in properties)
    return "\n".join(note.removesuffix(" Null means omitted.") for note in notes)


def index(functions: Iterable[Mapping[str, Any]], result_schemas: Mapping[str, Any] | None = None, *,
          reads: Collection[str] = READS) -> str:
    """One line per function, in the order given (``move_pose``'s with its ``interaction_note`` and ``argument_notes``),
    then one per query where they are served (``robot.query``, ``query_spec.py``)."""

    results = result_schemas or {}
    return "\n".join([*(line for definition in functions
                        for line in (index_line(definition, results.get(definition["name"]), reads=reads),
                                     interaction_note(definition), argument_notes(definition)) if line),
                      *(index_line(query_spec.definition(name), query_spec.result_schema(name), reads=reads)
                        for name in (query_spec.NAMES if "query" in reads else ()))])


def query_entries() -> dict[str, dict[str, str]]:
    """The entry of each query by part, named ``query.<name>`` (a script looks it up as ``query_<name>``)."""

    return {query_spec.entry_name(name): entry_parts(
        query_spec.definition(name), query_spec.result_schema(name), examples=[query_spec.example(name)])
        for name in query_spec.NAMES}


def entry_parts(definition: Mapping[str, Any], result_schema: Mapping[str, Any] | None = None, *,
                examples: Sequence[str] = (), explain: str | None = None,
                reads: Collection[str] = READS) -> dict[str, str]:
    """The entry's parts by name, in ``PARTS`` order (``description`` and ``arguments`` always); ``explain`` is the
    text of the ``explain`` part (the tool's stage template described, ``toolkit/explain.documentation``); an
    arm-moving tool has the ``dry_run`` part and argument where ``reads`` serves them."""

    arguments = schema_lines(_tool_schema(definition, reads), required_word="required")
    result = schema_lines(result_schema, required_word="always", relative=True) if result_schema else []
    shown = [example.strip("\n") for example in examples if example.strip()]
    dry = takes_dry_run(definition["name"], reads)
    if dry:
        arguments = [*arguments, f"- {dry_run.ARGUMENT}: {DRY_RUN_ARGUMENT}"]
    parts = {
        "description": f"{signature(definition, reads=reads)}\n\n{str(definition.get('description') or '').strip()}".rstrip(),
        "arguments": "Arguments (by keyword):\n" + "\n".join(arguments) if arguments else "Arguments: none.",
        "result": "Result (the tool's result, as a direct call of it returns it):\n" + "\n".join(result)
                  if result else "",
        "dry_run": DRY_RUN_RESULT
                   + "\n".join(schema_lines(dry_run.RESULT_SCHEMA, required_word="always", relative=True))
                   if dry else "",
        "explain": explain.strip() if explain else "",
        "examples": "Examples:\n" + "\n\n".join(shown) if shown else "",
    }
    return {name: parts[name] for name in PARTS if parts[name]}


def examples(name: str, *, reads: Collection[str] = READS,
             definition: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """The examples ``EXAMPLES`` gives the function ``name`` (a dotted tool name is read with underscores); an
    arm-moving tool's also shows ``dry_run`` where it is served. Given the function's ``definition``, they are written
    as this surface takes them: the first camera its schema names, the first form of ``_DRY_CALLS`` its schema
    validates (no dry-run example where it validates none)."""

    shown = tuple(example.replace("{camera}", _camera(definition)) for example in EXAMPLES.get(wire_name(name), ()))
    if not takes_dry_run(name, reads):
        return shown
    base = str(name).rsplit(".", 1)[-1]
    forms = _DRY_CALLS[base] if definition is None else [
        form for form in _DRY_CALLS[base] if _takes(_tool_schema(definition, reads), form)]
    return (*shown, _DRY_EXAMPLE.format(call=f"{base}({forms[0]}")) if forms else shown


def _camera(definition: Mapping[str, Any] | None) -> str:
    """The first camera the definition's ``camera`` argument names, else ``EXAMPLE_CAMERA``."""

    schema = (definition or {}).get("inputSchema")
    camera = ((schema.get("properties") or {}) if isinstance(schema, Mapping) else {}).get("camera")
    named = [item for item in (camera.get("enum") or ()) if isinstance(item, str)] if isinstance(camera, Mapping) else []
    return named[0] if named else EXAMPLE_CAMERA


def _takes(schema: Mapping[str, Any], keywords: str) -> bool:
    """Whether the call's keyword arguments, as Python source (``a=1, b="x"``), validate against ``schema``."""

    call = ast.parse(f"f({keywords})", mode="eval").body
    arguments = {item.arg: ast.literal_eval(item.value) for item in call.keywords}
    try:
        validate_schema_instance(arguments, schema, root_path="$input")
    except SchemaViolation:
        return False
    return True


def listing(functions: Iterable[Mapping[str, Any]], result_schemas: Mapping[str, Any] | None = None, *,
            reads: Collection[str] = READS) -> str:
    """The script tool's list: the index, then the card (``robot.docs()`` answers the same)."""

    functions = list(functions)
    return (index(functions, result_schemas, reads=reads) + "\n"
            + card(reads, interaction=takes_interaction(functions)))


def nearest(name: str, names: Iterable[str]) -> list[str]:
    """The ``names`` nearest to ``name``, nearest first, by spelling or by a word they share (``pose_move``,
    ``move_pose``): for a refusal to say what was probably meant."""

    wanted = str(name).lower()
    words = set(re.split(r"[._]", wanted))

    def score(item: str) -> float:
        shared = len(words & set(re.split(r"[._]", item.lower()))) / len(words)
        spelled = difflib.SequenceMatcher(None, wanted, item.lower()).ratio() - 0.1
        return max(spelled, shared, 0.9 if wanted in item.lower() else 0)

    return [item for item in sorted(names, key=score, reverse=True) if score(item) >= 0.5][:5]


def _cut_arguments(text: str, limit: int) -> str:
    """The ``arguments`` part at the deepest level that fits ``limit`` characters; if even the top level does not, as
    many of its lines as do."""

    lines = re.split(r"\n(?= *- )", text)
    depth = lambda line: (len(line) - len(line.lstrip(" "))) // 2  # noqa: E731
    kept: list[str] = []
    for level in sorted({depth(line) for line in lines}, reverse=True):
        kept = [line for line in lines if depth(line) <= level]
        if len("\n".join(kept)) <= limit:
            return "\n".join(kept)
    fitting: list[str] = []
    for line in kept:
        if len("\n".join([*fitting, line])) > limit:
            break
        fitting.append(line)
    return "\n".join(fitting)


def _cut_description(text: str, limit: int) -> str:
    """The description's first sentences, as written, within ``limit`` characters."""

    if len(text) <= limit:
        return text
    cut = text[:limit]
    ends = [match.start() for match in _SENTENCE_END.finditer(cut)] + [index for index, char in enumerate(cut)
                                                                       if char == "\n"]
    return cut[:max(ends, default=limit)].rstrip()


def short(name: str, parts: Mapping[str, str]) -> str:
    """What ``robot.docs(name)`` and ``help(tools.<name>)`` show: the call, the description and the arguments, at most
    ``SHORT_CHARS`` characters (the arguments cut in depth, then the description in sentences), then the name and size
    of every part not shown whole. An entry that fits is shown whole."""

    if len(whole(parts)) <= SHORT_CHARS:
        return whole(parts)
    listed = [f'- robot.docs("{name}", "{part}"): {len(text)} chars' for part, text in parts.items()]
    head = "The rest of this entry, part by part:"
    tail = [f'- robot.docs("{name}", "{ALL}"): the whole entry']
    described, arguments = parts["description"], parts.get("arguments", "")
    room = SHORT_CHARS - len("\n".join([head, *listed, *tail])) - 6
    shown_arguments = _cut_arguments(arguments, max(room * 3 // 5, room - len(described)))
    shown_description = _cut_description(described, room - len(shown_arguments))
    shown = {"description": shown_description, "arguments": shown_arguments}
    rest = [line for part, line in zip(parts, listed) if shown.get(part) != parts[part]]
    return "\n\n".join(item for item in (shown_description, shown_arguments, "\n".join([head, *rest, *tail]))
                        if item) + "\n"


def entry(definition: Mapping[str, Any], result_schema: Mapping[str, Any] | None = None, *,
          examples: Sequence[str] = (), explain: str | None = None, reads: Collection[str] = READS) -> str:
    """The function's whole documentation: its parts, one after another."""

    return whole(entry_parts(definition, result_schema, examples=examples, explain=explain, reads=reads))


def whole(parts: Mapping[str, str]) -> str:
    """An entry from its parts (``entry_parts``)."""

    return "\n\n".join(parts.values()) + "\n"


def schema_lines(schema: Any, *, required_word: str = "required", relative: bool = False) -> list[str]:
    """One line per member of ``schema`` at every depth, nested by indentation: ``a``, ``a.b``, ``a[]``, ``a[].b``,
    ``a.<key>`` (a map's values); with ``relative`` a nested member is named from its parent (``.b``, ``[]``)."""

    if not isinstance(schema, Mapping):
        return []
    walk = _Walk(schema, required_word, relative)
    resolved, seen = walk.resolve(schema, ())
    if isinstance(resolved, Mapping):
        about = ", ".join(_constraints(resolved))  # what the schema says of the whole value, first
        description = resolved.get("description")
        if isinstance(description, str) and description.strip():
            about = f"{about}. {description}" if about else description
        if about:
            walk.lines.append(about)
        for keyword, word in _ALTERNATIVES:  # the whole value has alternative forms
            if isinstance(resolved.get(keyword), list):
                walk.lines.append(f"- {word}:")
                walk.forms(resolved[keyword], "", "", 1, seen)
        walk.members(resolved, "", "", 0, seen, forms=False)
    return walk.lines


@dataclass
class _Walk:
    root: Mapping[str, Any]
    required_word: str
    relative: bool
    lines: list[str] = field(default_factory=list)

    def resolve(self, node: Any, seen: tuple[str, ...]) -> tuple[Any, tuple[str, ...]]:
        """A local ``$ref`` (``#/$defs/x``) resolved against the root schema, its sibling keywords kept."""

        ref = node.get("$ref") if isinstance(node, Mapping) else None
        if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen:
            return node, seen
        target: Any = self.root
        for part in ref[2:].split("/"):
            target = target.get(part) if isinstance(target, Mapping) else None
        if not isinstance(target, Mapping):
            return node, seen
        return {**target, **{key: value for key, value in node.items() if key != "$ref"}}, (*seen, ref)

    def label(self, path: str, shown: str) -> str:
        return path[len(shown):] if self.relative and shown and path.startswith(shown) else path

    def line(self, node: Any, path: str, shown: str, depth: int, seen: tuple[str, ...], *, required: bool) -> None:
        node, seen = self.resolve(node, seen)
        label = self.label(path, shown)
        if not isinstance(node, Mapping):
            self.lines.append(f"{'  ' * depth}- {label}: {json.dumps(node)}")
            return
        summary = _summary(node, required=required, word=self.required_word)
        self.lines.append(f"{'  ' * depth}- {label}" + (f": {summary}" if summary else ""))
        self.members(node, path, path, depth + 1, seen)

    def members(self, node: Mapping[str, Any], path: str, shown: str, depth: int, seen: tuple[str, ...], *,
                forms: bool = True) -> None:
        """``node``'s properties, map values, array items and alternative forms, at ``depth``."""

        if depth > _MAX_DEPTH:
            return
        properties = node.get("properties")
        required = set(node.get("required") or ())
        if isinstance(properties, Mapping):
            for name, child in properties.items():
                self.line(child, f"{path}.{name}" if path else str(name), shown, depth, seen,
                          required=name in required)
        values = node.get("additionalProperties")
        if isinstance(values, Mapping):
            self.line(values, f"{path}.<key>" if path else "<key>", shown, depth, seen, required=False)
        items = node.get("items")
        if isinstance(items, Mapping) and path:
            items, item_seen = self.resolve(items, seen)
            if isinstance(items, Mapping) and _says_more(items):
                self.line(items, f"{path}[]", shown, depth, item_seen, required=False)
            elif isinstance(items, Mapping):
                self.members(items, f"{path}[]", shown, depth, item_seen)
        positional = node.get("prefixItems") or (items if isinstance(items, list) else [])
        for position, item in enumerate(positional if path else []):
            self.line(item, f"{path}[{position}]", shown, depth, seen, required=False)
        if forms:
            for keyword, _word in _ALTERNATIVES:
                if isinstance(node.get(keyword), list):
                    self.forms(node[keyword], path, shown, depth, seen)

    def forms(self, branches: list[Any], path: str, shown: str, depth: int, seen: tuple[str, ...]) -> None:
        for number, branch in enumerate(branches, start=1):
            branch, branch_seen = self.resolve(branch, seen)
            if not isinstance(branch, Mapping):
                self.lines.append(f"{'  ' * depth}- form {number}: {json.dumps(branch)}")
                continue
            summary = _summary(branch, required=False, word="")
            self.lines.append(f"{'  ' * depth}- form {number}" + (f": {summary}" if summary else ""))
            self.members(branch, path, shown, depth + 1, branch_seen)


def _says_more(node: Mapping[str, Any]) -> bool:
    """Whether an array's items get a line of their own: they say more than a bare type and their members."""

    return bool(set(node) - {"type", "required", "additionalProperties", "properties"})


def _summary(node: Mapping[str, Any], *, required: bool, word: str) -> str:
    """``type, constraints. description`` for one schema node."""

    facts = [_kind(node)] if _kind(node) else []
    if required and word:
        facts.append(word)
    facts.extend(_constraints(node))
    facts.extend(word for keyword, word in _ALTERNATIVES if isinstance(node.get(keyword), list))
    text = ", ".join(facts)
    description = node.get("description")
    if isinstance(description, str) and description.strip():
        text = f"{text}. {description}" if text else description
    return text


def _types(node: Mapping[str, Any]) -> list[str]:
    declared = node.get("type")
    return [str(item) for item in declared] if isinstance(declared, list) else [str(declared)] if declared else []


def _kind(node: Mapping[str, Any]) -> str:
    """The value's type in words: ``number``, ``array of number``, ``string or null``."""

    words = []
    for name in _types(node):
        items = node.get("items")
        words.append(f"array of {' or '.join(_types(items)) or 'value'}" if name == "array"
                     and isinstance(items, Mapping) else name)
    return " or ".join(words)


def _count(low: Any, high: Any, one: str, many: str) -> str | None:
    def counted(number: Any) -> str:
        return f"{number} {one if number == 1 else many}"

    if low is not None and low == high:
        return f"exactly {counted(low)}"
    if low is not None and high is not None:
        return f"{low} to {counted(high)}"
    if low is not None:
        return f"at least {counted(low)}"
    if high is not None:
        return f"at most {counted(high)}"
    return None


def _constraints(node: Mapping[str, Any]) -> list[str]:
    facts: list[str] = []
    if "enum" in node:
        facts.append("one of " + ", ".join(json.dumps(value) for value in node["enum"]))
    if "const" in node:
        facts.append(f"constant {json.dumps(node['const'])}")
    for low, high, one, many in (("minItems", "maxItems", "item", "items"),
                                 ("minLength", "maxLength", "character", "characters"),
                                 ("minProperties", "maxProperties", "property", "properties")):
        counted = _count(node.get(low), node.get(high), one, many)
        if counted:
            facts.append(counted)
    for keyword, sign in (("minimum", ">="), ("exclusiveMinimum", ">"), ("maximum", "<="),
                          ("exclusiveMaximum", "<")):
        if isinstance(node.get(keyword), (int, float)) and not isinstance(node.get(keyword), bool):
            facts.append(f"{sign} {node[keyword]}")
    if node.get("uniqueItems"):
        facts.append("unique items")
    if "pattern" in node:
        facts.append(f"matching {json.dumps(node['pattern'])}")
    if "format" in node:
        facts.append(f"format {node['format']}")
    if "default" in node:
        facts.append(f"default {json.dumps(node['default'])}")
    if node.get("additionalProperties") is True and isinstance(node.get("properties"), Mapping):
        facts.append("may carry further fields")
    facts.extend(f"{keyword}={json.dumps(value, sort_keys=True)}" for keyword, value in node.items()
                 if keyword not in _STATED)
    return facts


__all__ = ["ALL", "BASE_FORMS", "BASE_NOTE", "DRY_RUN_ARGUMENT", "EXAMPLES", "EXAMPLE_CAMERA", "INTERACTION_NOTE",
           "PARTS", "READS", "SHORT_CHARS", "accepts_dry_run", "call_shape", "card", "entry", "entry_parts", "examples",
           "fragment_parts", "index", "index_line", "interaction_form", "interaction_note", "labelled", "library_names",
           "listing", "members", "nearest", "purpose", "query_entries", "result_shape", "schema_lines", "sentences",
           "short", "signature", "takes_dry_run", "takes_interaction", "whole"]
