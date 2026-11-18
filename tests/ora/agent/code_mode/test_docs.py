"""Code mode's function documentation (``ora/agent/code_mode/docs.py``) is generated from the definitions, faithfully.

Every toolkit tool's ``description.json`` -- the definition a toolkit surface resolves, unchanged -- gives an entry
that keeps every sentence of its description, every argument at every depth with its own description and allowed
values, and every field of its result, and an index line that gives the call's nested shape, the result's and the
description's first sentence on one line. The entry is its parts in order, each of which a script may look up alone,
and a lookup with no part is short. After the functions a card lists the library by introspection, and what a surface
does not serve is told nowhere.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Iterator

import pytest

from ora.agent.code_mode import coding, docs, library_card, ora_lib
from ora.agent.code_mode.ora_lib import observation
from ora.agent.toolkit.interface import dry_run

ROOT = Path(__file__).resolve().parents[4]
DESCRIPTIONS = sorted((ROOT / "ora/agent/toolkit/tools").glob("*/description.json"))


def _tool(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk(node: Any, root: Any = None, seen: frozenset[str] = frozenset()) -> Iterator[tuple[str, Any]]:
    """Every (keyword, value) of the schema the value follows, at every depth: local ``$ref`` followed, unreferenced
    ``$defs`` left out."""

    root = node if root is None else root
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"$defs", "definitions"}:
                continue
            yield key, value
            if key == "$ref" and isinstance(value, str) and value.startswith("#/") and value not in seen:
                target = root
                for part in value[2:].split("/"):
                    target = target[part]
                yield from _walk(target, root, seen | {value})
            else:
                yield from _walk(value, root, seen)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value, root, seen)


def _names(node: Any) -> set[str]:
    return {name for key, value in _walk(node) if key == "properties" and isinstance(value, dict) for name in value}


def test_the_card_names_the_source_only_where_it_is_mounted() -> None:
    assert library_card.SOURCE_LINE in docs.card(("query", "source"))
    assert library_card.SOURCE_LINE not in docs.card(("dry_run", "query"))


def test_every_toolkit_tool_is_covered() -> None:
    assert len(DESCRIPTIONS) >= 20


@pytest.mark.parametrize("path", DESCRIPTIONS, ids=lambda path: path.parent.name)
def test_an_entry_keeps_every_sentence_argument_and_result_field(path: Path) -> None:
    tool = _tool(path)
    definition = {"name": tool["name"], "description": tool["description"], "inputSchema": tool["inputSchema"]}
    text = docs.entry(definition, tool["resultSchema"])
    assert tool["description"].strip() in text, "the description verbatim"
    for sentence in docs.sentences(tool["description"]):
        assert sentence in text
    arguments = docs.entry_parts(definition, tool["resultSchema"])["arguments"]
    for name in _names(tool["inputSchema"]):
        assert f".{name}:" in arguments or f"- {name}:" in arguments, name
    for key, value in _walk(tool["inputSchema"]):
        if key == "description" and isinstance(value, str):
            assert value in arguments, value
        if key == "enum":
            assert ", ".join(json.dumps(item) for item in value) in arguments
        if key in {"minimum", "maximum", "default"} and not isinstance(value, (dict, list)):
            assert json.dumps(value) in arguments
    result = docs.entry_parts(definition, tool["resultSchema"])["result"]
    for name in _names(tool["resultSchema"]):
        assert f".{name}" in result or f"- {name}" in result, name
    for key, value in _walk(tool["resultSchema"]):
        if key == "description" and isinstance(value, str):
            assert value in result, value


@pytest.mark.parametrize("path", DESCRIPTIONS, ids=lambda path: path.parent.name)
def test_an_index_line_is_the_call_s_shape_its_result_and_the_first_sentence(path: Path) -> None:
    tool = _tool(path)
    line = docs.index_line(tool, tool["resultSchema"], reads=())
    assert "\n" not in line and line.startswith(f"- tools.{tool['name']}(") and " -> {" in line
    first = docs.sentences(tool["description"])[0]
    assert line.endswith(": " + " ".join(first.split()))
    schema = tool["inputSchema"]
    required = set(schema.get("required") or ())
    call = docs.call_shape(tool, reads=())
    for name in schema.get("properties", {}):
        assert re.search(rf"\b{name}{'' if name in required else '[?]'}[,:)]", call), name
    assert docs.result_shape(tool["resultSchema"]) in line
    assert "dry_run" not in call and ("dry_run?: bool" in docs.call_shape(tool)) == (
        tool["name"] in dry_run.TOOLS), "an arm-moving tool also takes dry_run, where it is served"


@pytest.mark.parametrize("path", DESCRIPTIONS, ids=lambda path: path.parent.name)
def test_a_signature_names_the_arguments_required_first_and_dry_run_where_it_is_served(path: Path) -> None:
    tool = _tool(path)
    schema = tool["inputSchema"]
    required = schema.get("required") or []
    call = docs.signature(tool)
    named = call[call.index("(") + 1:-1].split(", ") if schema.get("properties") else []
    assert named == [*(name for name in schema.get("properties", {}) if name in required),
                     *(f"{name}?" for name in schema.get("properties", {}) if name not in required),
                     *(["dry_run?"] if tool["name"] in dry_run.TOOLS else [])]
    assert "dry_run" not in docs.signature(tool, reads=()), "not where dry runs are not served"


def test_the_description_part_starts_with_the_plain_signature_a_positional_call_binds_to() -> None:
    """``robot.py`` binds ``tools.vision_ground("head", "cup")`` by parsing this line, so it stays names, ``, `` and
    ``?``: the shapes belong to the index line (``call_shape``) and never reach it."""
    tool = _tool(ROOT / "ora/agent/toolkit/tools/vision_ground/description.json")
    assert docs.call_shape(tool).startswith("tools.vision_ground(camera: ")
    for reads in ((), docs.READS):
        for text in (docs.entry_parts(tool, tool["resultSchema"], reads=reads)["description"],
                     docs.short("vision_ground", docs.entry_parts(tool, tool["resultSchema"], reads=reads))):
            line = text.split("\n", 1)[0]
            assert line == "tools.vision_ground(camera, text, max_candidates?)"
            assert re.fullmatch(r"tools\.\w+\(\w+\??(, \w+\??)*\)", line)
            names = [name.rstrip("?") for name in line.partition("(")[2].rpartition(")")[0].split(", ")]
            assert names == ["camera", "text", "max_candidates"]


def test_a_shape_spells_out_enums_and_lengths_and_merges_equal_neighbours() -> None:
    arm = {"type": ["object", "null"], "required": ["mode"], "properties": {
        "mode": {"type": "string", "enum": ["hold", "absolute"]},
        "position_m": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "interaction": {"type": ["string", "null"], "enum": ["free_space", "precision", None]}}}
    tool = {"name": "robot.move", "description": "Move it. More.", "inputSchema": {
        "type": "object", "required": ["targets"], "$defs": {"arm": arm}, "properties": {
            "speed": {"type": ["string", "null"], "enum": ["fast", "slow", None]},
            "targets": {"type": "object", "properties": {"left": {"$ref": "#/$defs/arm"}, "right": {"$ref": "#/$defs/arm"}}},
            "ticks": {"type": "integer", "minimum": 1, "maximum": 20},
            "tilt": {"type": "number", "minimum": 0},
            "points": {"type": "array", "minItems": 1, "maxItems": 8,
                       "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}},
            "arms": {"type": "array", "items": {"type": "string", "enum": ["left", "right"]}, "minItems": 1},
            "note": {"type": "string"}, "free": {"type": "object"}, "flag": {"type": "boolean"}}}}
    assert docs.call_shape(tool) == (
        "tools.robot_move(targets: {left?, right?: {mode: hold|absolute, position_m?: [3], "
        "interaction?: free_space|precision}}, speed?: fast|slow, ticks?: int 1..20, tilt?: number >=0, "
        "points?: [[int; 2]; 1-8], arms?: [left|right; 1+], note?: str, free?: dict, flag?: bool)")
    assert docs.index_line(tool, {"type": "object", "properties": {"status": {"type": "string"}}}).endswith(
        ", flag?: bool) -> {status?}: Move it.")
    query = docs.index_line(docs.query_spec.definition("robot_state"), docs.query_spec.result_schema("robot_state"))
    assert query.startswith('- robot.query("robot_state") -> {') and query.endswith('(robot.docs("query_robot_state"))')
    assert docs.call_shape(docs.query_spec.definition("fk")).startswith('robot.query("fk", arm: str, joints_deg: [[6]; 1-')


def _integer_items(node: Any) -> bool:
    """Whether an argument is an array (of arrays) of integers: its validator refuses a float there."""

    def kinds(value: Any) -> list[str]:
        declared = value.get("type") if isinstance(value, dict) else None
        return list(declared) if isinstance(declared, list) else [declared]

    nested = False
    while "array" in kinds(node):
        node, nested = node.get("items"), True
    return nested and "integer" in kinds(node) and "number" not in kinds(node)


def test_an_integer_array_says_int_on_every_surface() -> None:
    """Pixels are integers: ``[[2]; 1-8]`` read as numbers, and 40 RoboCasa365 calls sent a projection's 88.97."""
    from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from tests.support.agent.surface_catalog import catalog_grants

    seen = set()
    for profile in PROFILE_CATALOG:
        for definition in resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile)).definitions:
            line = docs.index_line(definition, reads=())
            for name, schema in ((definition.get("inputSchema") or {}).get("properties") or {}).items():
                if _integer_items(schema):
                    assert re.search(rf"\b{name}\??(, \w+\??)*: \[+int\b", line), (profile, definition["name"], name)
                    seen.add((profile, definition["name"], name))
    assert {("robocasa_mobile_direct_v1", "vision_segment", "points_uv"), ("robocasa_mobile_direct_v1", "mark_point",
            "points_uv"), ("robocasa_mobile_direct_v1", "geometry_query", "points")} <= seen
    assert docs.call_shape({"name": "f", "inputSchema": {"type": "object", "properties": {
        "xyz": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}}}}) == "tools.f(xyz?: [3])"
    assert "`[int; 2]` two integers" in coding.BASH_INDEX_HEADER


def test_move_base_s_line_says_its_two_forms_where_it_has_both() -> None:
    """Six optional arguments on one line, and a validator that takes all three world ones or relative ones only: 27
    RoboCasa365 calls gave a part of the world target."""
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from ora.agent.toolkit.tools.move_base import tool

    assert [frozenset(form) for form in docs.BASE_FORMS] == [tool.ABSOLUTE, tool.RELATIVE]
    definition = next(item for item in resolve_tool_surface("robocasa_mobile_direct_v1").definitions
                      if item["name"] == "move_base")
    line, note = docs.index([definition], reads=()).splitlines()
    assert line.startswith("- tools.move_base(x_m?, ") and note == docs.BASE_NOTE
    assert "all of x_m, y_m and yaw_deg (a world target) or any of forward_m, left_m and turn_deg" in note
    assert "never a mix" in note
    schema = definition["inputSchema"]
    relative = {**definition, "inputSchema": {**schema, "properties": {
        name: value for name, value in schema["properties"].items() if name in tool.RELATIVE}}}
    assert docs.argument_notes(relative) == "" and docs.index([relative], reads=()).count("\n") == 0


def test_one_huge_schema_cannot_take_over_a_line_and_says_what_it_left_out() -> None:
    leaf = {"type": "string", "enum": [f"value_{number}" for number in range(40)]}
    wide = {"type": "object", "properties": {f"field_{number}": leaf for number in range(40)}}
    tool = {"name": "huge", "description": "Huge.", "inputSchema": {"type": "object", "properties": {
        f"argument_{number}": {"type": "object", "properties": {"inner": wide}} for number in range(90)}}}
    result = {"type": "object", "required": ["kept"], "properties": {
        "kept": {"type": "string"}, **{f"field_{number}": wide for number in range(80)}}}
    call, shown = docs.call_shape(tool), docs.result_shape(result)
    assert len(call) <= docs.ARGUMENT_CHARS + 20 and call.endswith(", ...)") and "{...}" in call
    assert len(shown) <= docs.RESULT_CHARS + 2 and shown.startswith("{kept, field_0?, field_1?") and shown.endswith(
        ": {...}, ...}")


def test_a_result_shows_what_a_script_reads_first() -> None:
    """The list of candidates or marks, opened; alternative forms share the characters."""
    ground = docs.result_shape(_tool(ROOT / "ora/agent/toolkit/tools/vision_ground/description.json")["resultSchema"])
    item = re.search(r"candidates: \[\{(.*?)\}\]", ground).group(1)
    for field in ("bbox_xyxy: [4]", "localization", "candidate_id", "target_ref"):
        assert field in item, field
    query = docs.result_shape(_tool(ROOT / "ora/agent/toolkit/tools/geometry_query/description.json")["resultSchema"])
    assert query.count(" or ") == 2 and "world_points: [{" in query and "world_xyz_m" in query
    assert len(query) <= docs.RESULT_CHARS + 20


def test_a_lookup_is_short_by_default_and_names_the_rest_with_its_size() -> None:
    long = " ".join(f"Sentence number {number} says one thing." for number in range(400))
    tool = {"name": "move_pose", "description": "Move the arm.\n" + long, "inputSchema": {
        "type": "object", "properties": {"targets": {"type": "object", "description": "The targets.", "properties": {
            f"arm_{number}": {"type": "object", "description": "One arm. " * 30, "properties": {
                "position_m": {"type": "array", "description": "Where. " * 30}}} for number in range(20)}}}}}
    parts = docs.entry_parts(tool, {"type": "object", "properties": {"status": {"type": "string"}}},
                             examples=docs.examples("move_pose", reads=docs.READS), explain="Explain (r):\n1. x",
                             reads=docs.READS)
    text = docs.short("move_pose", parts)
    assert len(text) <= docs.SHORT_CHARS < len(docs.whole(parts))
    assert text.startswith("tools.move_pose(targets?, dry_run?)\n\nMove the arm.\nSentence number 0 says one thing.")
    assert text.count("Sentence number") < 400 and text.split("Arguments (by keyword):")[0].rstrip().endswith(".")
    assert "- targets: object." in text and "targets.arm_0.position_m" not in text, "the depth is cut first"
    for part in ("description", "arguments", "result", "dry_run", "explain", "examples"):
        assert f'- robot.docs("move_pose", "{part}"): {len(parts[part])} chars\n' in text
    assert text.endswith('- robot.docs("move_pose", "all"): the whole entry\n')
    small = docs.entry_parts({"name": "wait", "description": "Wait.", "inputSchema": {"type": "object"}})
    assert docs.short("wait", small) == docs.whole(small), "an entry that fits is shown whole"


def test_the_entry_is_its_parts_in_order_and_each_part_stands_alone() -> None:
    tool = _tool(ROOT / "ora/agent/toolkit/tools/geometry_query/description.json")
    example = "tools.geometry_query(camera='head', region=[0, 0, 8, 8])"
    parts = docs.entry_parts(tool, tool["resultSchema"], examples=[example], explain="Explain (r):\n1. request: x")
    assert list(parts) == [part for part in docs.PARTS if part != "dry_run"] and parts["examples"] == f"Examples:\n{example}"
    assert parts["explain"] == "Explain (r):\n1. request: x"
    assert docs.entry(tool, tool["resultSchema"], examples=[example], explain=parts["explain"]) == (
        "\n\n".join(parts.values()) + "\n")
    assert "explain" not in docs.entry_parts(tool, tool["resultSchema"]), "a function without a template has no part"
    assert parts["description"].startswith("tools.geometry_query(camera, points?, region?, step?, window_px?, "
                                           "world_points_m?)\n\n")
    assert parts["arguments"].startswith("Arguments (by keyword):\n- camera: string, required, one of ")
    assert "  - points[]: array of integer, exactly 2 items. One pixel as [u,v]" in parts["arguments"]
    assert parts["result"].startswith("Result (the tool's result, as a direct call of it returns it):\n"
                                      "- one of these forms:\n  - form 1: object\n    - step: integer or null, always")
    assert "region_result" in parts["result"] and "      - .depth_m: object" in parts["result"]
    bare = docs.entry_parts({"name": "robot.wait", "description": "Wait.", "inputSchema": {"type": "object"}})
    assert bare == {"description": "tools.robot_wait()\n\nWait.", "arguments": "Arguments: none."}


def test_references_maps_tuples_and_forms_are_spelled_out() -> None:
    schema = {
        "type": "object", "required": ["pose"], "$defs": {"xyz": {
            "type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
            "description": "Metres, world frame."}},
        "properties": {
            "pose": {"$ref": "#/$defs/xyz", "description": "Target position."},
            "labels": {"type": "object", "additionalProperties": {"type": "string", "description": "A label."}},
            "pair": {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string", "const": "px"}]},
            "mode": {"oneOf": [{"type": "string", "enum": ["fast"]}, {"type": "number", "exclusiveMinimum": 0}],
                     "x-unit": "m/s"},
        },
    }
    lines = docs.schema_lines(schema)
    assert lines == [
        "- pose: array of number, required, exactly 3 items. Target position.",
        "- labels: object",
        "  - labels.<key>: string. A label.",
        "- pair: array",
        "  - pair[0]: integer",
        '  - pair[1]: string, constant "px"',
        '- mode: x-unit="m/s", one of these forms',
        '  - form 1: string, one of "fast"',
        "  - form 2: number, > 0",
    ]
    assert docs.schema_lines({"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/a"}}}) == [], "no endless loop"


def test_purpose_is_the_first_sentence_on_one_line_and_bounded() -> None:
    assert docs.purpose("Move the arm. Then stop.") == "Move the arm."
    assert docs.purpose("Measure e.g. a cup.\nUse when: x.") == "Measure e.g. a cup."
    assert docs.purpose("") == ""
    long = "word " * 200 + "end."
    assert len(docs.purpose(long)) <= docs.PURPOSE_CHARS + 4 and docs.purpose(long).endswith(" ...")


def test_a_perception_tool_s_entry_shows_the_library_fragment_for_what_it_returns() -> None:
    definition = {"name": "vision.ground", "description": "Find things. More.", "inputSchema": {"type": "object"}}
    parts = docs.entry_parts(definition, None, examples=docs.examples("vision.ground"))
    assert list(parts) == ["description", "arguments", "examples"]
    assert "object_geometry(candidate)" in parts["examples"] and 'tools.vision_ground(camera="head"' in parts["examples"]
    assert docs.examples("vision_ground") == docs.examples("vision.ground") and docs.examples("wait_ticks") == ()
    assert "examples" not in docs.entry_parts(definition), "an entry given none has none"


def test_a_library_name_is_looked_up_from_its_signature_and_docstring() -> None:
    parts = docs.fragment_parts("object_geometry")
    assert list(parts) == ["description"]
    assert parts["description"].startswith("ora_lib.object_geometry(target, *, observation") \
        and "support height" in parts["description"] and "vision_ground" in parts["description"]
    assert docs.fragment_parts("move_pose") is None
    for name, head, says in (("pixels_to_world", "ora_lib.pixels_to_world(camera, uv", "pinhole"),
                             ("offset", "ora_lib.offset(value, dx=0.0", "moved by"),
                             ("target", "robot.target(position=None, *, approach=None", "typed values"),
                             ("handoff", "robot.handoff(reason, **data)", "hand control back"),
                             ("explain", "robot.explain(call)", "causal order"),
                             ("Point", "ora_lib.Point(xyz, frame)\n.frame .xyz", "NaN marks a point"),
                             ("Pose", "ora_lib.Pose(xyz, xyzw, frame)", ".axis(name): "),
                             ("Observation", "ora_lib.observation.Observation(", ".camera(name): A camera by name"),
                             ("Camera", "ora_lib.observation.Camera(", "OpenCV intrinsics"),
                             ("ToolRefused", "robot.ToolRefused(name, answer)", ".code"),
                             ("Result", "robot.Result(tool, answer)", "``call_id``"),
                             ("geometry.rigid", "ora_lib.geometry.rigid(matrix)", "rigid 4x4"),
                             ("mask_to_cloud", "ora_lib.geometry.mask_to_cloud(depth_m", "")):
        text = docs.fragment_parts(name)["description"]
        assert text.startswith(head) and says in text, name
    assert docs.fragment_parts("query", reads=()) is None
    assert docs.fragment_parts("query")["description"].startswith("robot.query(name, **arguments)")
    assert "dry run" not in docs.fragment_parts("Result", reads=())["description"].lower()
    assert "A dry run" in docs.fragment_parts("Result")["description"]


def test_the_card_lists_every_public_name_of_the_library_by_introspection() -> None:
    text = docs.card(docs.READS)
    listed = text.splitlines()
    for name in ora_lib.__all__:
        assert any(re.match(rf"- {name}\b", line) for line in listed), name
    assert all(any(f"- {name}" in line for line in listed) for name in observation.__all__ if name[0].isupper()
               and name not in ("MOUNT", "SCHEMA"))
    point = next(line for line in listed if line.startswith("- Point("))
    assert point.startswith("- Point(xyz, frame): ")
    assert {".xyz", ".x", ".y", ".z", ".frame", ".valid", ".to(frame)", ".mean()", ".distance(other)"} <= set(point.split())
    camera = next(line for line in listed if line.startswith("- Camera"))
    assert camera.startswith("- Camera: ") and {".K", ".extrinsic", ".camera_to_world", ".depth_m", ".rgb", ".mount"} <= set(
        camera.split()), "a class a script is handed shows no constructor"
    seen = next(line for line in listed if line.startswith("- Observation: "))
    assert {".cameras", ".end_effectors", ".camera(name)", ".pose(arm)", ".arms()", ".target(reference)"} <= set(seen.split())
    geometry = next(line for line in listed if line.startswith("- ObjectGeometry"))
    assert {".extent_xyz_m", ".height_above_support_m", ".support_z_m", ".center", ".largest_axis"} <= set(geometry.split())
    assert "- current() -> Observation" in listed and "- geometry.mask_to_cloud(depth_m, intrinsic, mask, *, " in text
    assert "observation=None" not in text, "said once, at the head of the list"
    head, own = text.split("`robot`, not `ora_lib`")
    assert "- target(" not in head and "- target(position=None, *, approach=None" in own
    assert "`from robot import <name>`" in own and "- handoff(reason, **data)" in own
    assert re.search(r"^- ToolRefused: .*\.code .*\.answer", own, re.M)
    result = next(line for line in own.splitlines() if line.startswith("- Result: "))
    assert result.startswith("- Result: .ok .status .failure_code") and ".feasible" not in result and ".arms" in result
    plan = next(line for line in own.splitlines() if line.startswith("- Plan: "))
    assert {".feasible", ".summary", ".predicted_pose", ".failure_code", ".route"} <= set(plan.split())
    assert ".ok" not in plan.split() and ".go()" not in plan, "a dry run's plan has no run's fields and runs nothing"
    assert "- PlanNotRun" in own
    assert not any(name in own for name in library_card.ROBOT_UNLISTED - {"docs"})


def test_the_card_follows_the_classes_it_describes() -> None:
    class Later:
        """What a later commit adds is on the card without a line of documentation."""

        def __init__(self, v) -> None:
            self.v, self.xyz = v, v
            self._private = 1

        @property
        def norm(self):
            return 0

        def scaled(self, factor, *, observation=None):
            return self

    assert docs.members(Later) == [".v", ".xyz", ".norm", ".scaled(factor)"]
    card = docs.card(docs.READS)
    assert "- target(position=None" in card and "- handoff(reason, **data)" in card
    assert "`from robot import <name>`" in card and "- Result: .ok" in card and "- query(name, **arguments)" in card


def test_the_docs_op_answers_a_fragment_beside_the_tools_and_refuses_the_rest() -> None:
    from ora.agent.pi.code_mode import PiCodeMode

    class Session(PiCodeMode):
        code_mode = {"library": "extended"}

        def _code_mode_docs(self):
            return "- tools.observe()", {"observe": {"description": "tools.observe()\n\nLook.", "arguments": "none"},
                                         "move_pose": {"description": "tools.move_pose()\n\nMove.", "arguments": "x"}}

        def _code_mode_model_tools(self):
            return ["check_task"]

    session = Session()
    ask = lambda **message: session._code_mode_op({"op": "docs", **message})  # noqa: E731
    found = ask(name="object_geometry")
    assert found["ok"] and found["name"] == "object_geometry" and found["text"].startswith("ora_lib.object_geometry(")
    assert ask(name="object_geometry", part="description")["text"].startswith("ora_lib.object_geometry(")
    assert ask(name="object_geometry", part="arguments")["error"]["code"] == "unknown_part"
    assert ask(name="observe", part="arguments")["text"] == "none\n", "a tool's entry as before"
    assert ask(name="observe", part="all")["text"] == "tools.observe()\n\nLook.\n\nnone\n", "and the whole of it"
    assert ask(name="observe")["text"] == ask(name="observe", part="all")["text"], "a short one is whole by default"
    assert ask(name="target")["text"].startswith("robot.target(position=None")
    assert ask(name="observe", part="nothing")["error"]["constraints"] == {"allowed": ["description", "arguments", "all"]}
    refused = ask(name="nothing")["error"]
    assert refused["code"] == "unknown_function" and "observe" in refused["constraints"]["allowed"]
    assert "Point" in refused["constraints"]["allowed"], "the library's names too"
    assert refused["message"] == "no function 'nothing'; robot.docs() lists them" and refused["constraints"]["nearest"] == []
    near = ask(name="pose_move")["error"]
    assert near["code"] == "unknown_function" and near["constraints"]["nearest"][0] == "move_pose"
    assert near["message"].startswith("no function 'pose_move'; the nearest: move_pose, ")
    assert {"Pixel", "pixels_to_world"} <= set(ask(name="Pixels")["error"]["constraints"]["nearest"])
    assert "your own tools" in ask(name="check_task")["error"]["message"]


DRY = [path for path in DESCRIPTIONS if path.parent.name in dry_run.TOOLS]


def test_the_tools_that_take_dry_run_are_the_arm_moving_ones() -> None:
    assert len(DRY) == len(dry_run.TOOLS) == 9
    assert not any(docs.accepts_dry_run(name) for name in ("observe", "wait_ticks", "vision_ground", "check_task"))
    assert docs.accepts_dry_run("move_pose") and docs.accepts_dry_run("robot.move_to")
    assert docs.takes_dry_run("move_pose", docs.READS) and not docs.takes_dry_run("move_pose", ("query",))


@pytest.mark.parametrize("path", DRY, ids=lambda path: path.parent.name)
def test_a_tool_that_takes_dry_run_documents_it_with_its_result_and_one_example(path: Path) -> None:
    tool = _tool(path)
    definition = {"name": tool["name"], "description": tool["description"], "inputSchema": tool["inputSchema"]}
    parts = docs.entry_parts(definition, tool["resultSchema"], examples=docs.examples(tool["name"], reads=docs.READS),
                             reads=docs.READS)
    assert list(parts) == list(docs.PARTS[:3]) + ["dry_run", "examples"]
    assert parts["arguments"].splitlines()[-1].startswith("- dry_run: boolean, default false. true returns what this call")
    assert "tools." + tool["name"] + "(" in parts["description"] and "dry_run?)" in parts["description"].splitlines()[0]
    assert "dry_run_unavailable" in parts["arguments"] and "query allowance" in parts["arguments"]
    assert parts["dry_run"].startswith("Dry run (dry_run=True): the call returns this record instead of its result")
    for field in ("feasible", "predicted_pose", "end_joints_deg", "route", "screening", "failure_code", "failure_stage",
                  "refusal", "claim_boundary", "predicted_approach_world", "predicted_jaw_line_world"):
        assert f"- {field}:" in parts["dry_run"] or f".{field}:" in parts["dry_run"], field
    assert "_rad" not in parts["dry_run"], "every angle is in degrees"
    (example,) = parts["examples"].removeprefix("Examples:\n").split("\n\n")
    assert f"tools.{tool['name']}(" in example and ", dry_run=True)" in example
    compile(example.replace("robot.explain", "str"), "<example>", "exec")  # it is Python
    assert "dry_run" not in json.dumps(tool["inputSchema"]), "the tool's own schema never carries it"


@pytest.mark.parametrize("path", DRY, ids=lambda path: path.parent.name)
def test_where_dry_runs_are_not_served_an_entry_never_mentions_one(path: Path) -> None:
    tool = _tool(path)
    definition = {"name": tool["name"], "description": tool["description"], "inputSchema": tool["inputSchema"]}
    for reads in ((), ("query",)):
        parts = docs.entry_parts(definition, tool["resultSchema"], examples=docs.examples(tool["name"], reads=reads),
                                 reads=reads)
        assert "dry_run" not in parts and "dry_run" not in json.dumps(parts) and "dry run" not in json.dumps(parts).lower()


def test_a_tool_without_a_dry_run_has_neither_the_argument_nor_the_part() -> None:
    tool = _tool(ROOT / "ora/agent/toolkit/tools/wait_ticks/description.json")
    parts = docs.entry_parts(tool, tool["resultSchema"], examples=docs.examples("wait_ticks"))
    assert "dry_run" not in parts and "dry_run" not in json.dumps(parts) and docs.examples("wait_ticks") == ()
    assert "dry_run" not in docs.index_line(tool, reads=docs.READS)


def test_the_queries_are_listed_only_where_they_are_served() -> None:
    tool = _tool(ROOT / "ora/agent/toolkit/tools/wait_ticks/description.json")
    assert docs.index([tool], reads=()) == docs.index_line(tool, reads=()), "no query line"
    assert "robot.query" not in docs.listing([tool], reads=())
    assert "query(" not in docs.card(())
    assert '- robot.query("ik", arm: str, poses: [[7]; 1-' in docs.index([tool])


# --- what a surface composes -------------------------------------------------------------------------------------

# 31,012 on RoboDojo since its default scheme (context_keep_written) brings artifact_read and
# observe(ref=...); 31,3xx since the description states the query allowance, the CPU time, memory, /tmp and the
# tools a script cannot call, and integer arrays say int (RoboCasa365 run audit, 2026-11-06).
#: The most the first request's tool schema may hold with ``library: extended`` (every definition, as JSON).
SCHEMA_BUDGET = 31_400


def _robotwin(code_mode):
    from tests.support.orchestrator.run_specs import native_spec

    return {**(spec := native_spec("robotwin")), "policy": {**spec["policy"], "code_mode": code_mode}}


def _libero(code_mode):
    from tests.support.orchestrator.run_specs import native_spec

    return {**(spec := native_spec("libero_pro")), "policy": {**spec["policy"], "code_mode": code_mode}}


def _robodojo(code_mode):
    from tests.support.orchestrator.run_specs import robodojo_native_spec

    spec = robodojo_native_spec("general_pickup")
    spec["policy"].update(code_mode=code_mode, baseline="robodojo_toolkit_v6")
    spec["toolkit_grants"] = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
    return spec


SURFACES = [pytest.param(_robodojo, id="robodojo"), pytest.param(_robotwin, id="robotwin"),
            pytest.param(_libero, id="libero")]


def _extended(build):
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    return resolve_run_spec(build({"library": "extended", "prompt": "v3"}))


def _functions_of(plan) -> tuple[list, dict]:
    """The surface's script functions and result schemas, as the session reads them."""
    from ora.agent.code_mode.contract import script_functions
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface

    surface = resolve_tool_surface(plan["policy"]["tool_surface"]["profile"], toolkit_grants=plan.get("toolkit_grants"),
                                   toolkit_receipt_delivery=plan.get("toolkit_receipt_delivery"))
    return script_functions(list(surface.definitions), tuple(surface.lifecycle_tools)), dict(surface.result_schemas)


@pytest.mark.parametrize("build", SURFACES)
def test_the_script_tool_keeps_to_its_budget(build) -> None:
    """The index and the card, against ~100 KB in the control: the first request's tool schema stays within budget and
    no function's line takes over."""
    plan = _extended(build)
    definitions = plan["policy_composition"]["tool_surface"]["definitions"]
    assert len(json.dumps(definitions)) <= SCHEMA_BUDGET
    lines = [line for line in definitions[-1]["description"].splitlines() if line.startswith(("- tools.", "- robot.query"))]
    assert lines and max(len(line) for line in lines) <= (docs.ARGUMENT_CHARS + docs.RESULT_CHARS + docs.PURPOSE_CHARS + 120)


@pytest.mark.parametrize("build", SURFACES)
def test_the_runtime_describes_the_script_tool_as_the_plan_does(build, tmp_path) -> None:
    """A run refuses definitions that differ from its plan's, and the index reads the result schemas: both must read the
    same ones, the Artifact reader's change to ``observe`` included."""
    from ora.agent.code_mode.runtime import build_code_mode_runtime
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from ora.agent.memory.tool_runtime import build_recoverable_artifact_runtime

    plan = _extended(build)
    resolved = resolve_tool_surface(plan["policy"]["tool_surface"]["profile"], toolkit_grants=plan.get("toolkit_grants"),
                                    toolkit_receipt_delivery=plan.get("toolkit_receipt_delivery"))

    class Base:
        surface = resolved

        def definitions(self):
            return resolved.select(plan["policy"].get("tool_allowlist"))

    tools = build_recoverable_artifact_runtime(plan, run_root=tmp_path, event_path=tmp_path / "events.jsonl",
                                               base_runtime=Base())
    tools = build_code_mode_runtime(plan, base_runtime=tools)
    assert tools.definitions() == plan["policy_composition"]["tool_surface"]["definitions"]


@pytest.mark.parametrize("build", SURFACES)
def test_a_default_lookup_of_every_function_on_every_surface_is_short(build) -> None:
    """``robot.docs("move_pose")`` brought 16-36 KB back into the history; it keeps to ``SHORT_CHARS``, first line the
    signature (a positional call binds to it), and says what the rest costs."""
    plan = _extended(build)
    functions, results = _functions_of(plan)
    reads = docs.READS if "dry_run" in plan["policy"]["code_mode"].get("script_reads", ()) else ()
    for definition in functions:
        parts = docs.entry_parts(definition, results.get(definition["name"]), reads=reads)
        text = docs.short(definition["name"], parts)
        assert len(text) <= docs.SHORT_CHARS, definition["name"]
        assert text.splitlines()[0] == docs.signature(definition, reads=reads) == parts["description"].splitlines()[0]
        if len(docs.whole(parts)) > docs.SHORT_CHARS:
            assert f'robot.docs("{definition["name"]}", "all")' in text


def _told(plan) -> str:
    """Everything a plan shows the model before its first turn: the tool definitions and the prompt layers' text."""
    layers = [layer for channel in plan["policy_context"].values() if isinstance(channel, list) for layer in channel]
    return json.dumps(plan["policy_composition"]["tool_surface"]["definitions"]) + "\n".join(
        (ROOT / layer["path"]).read_text(encoding="utf-8") for layer in layers)


@pytest.mark.parametrize("build", [pytest.param(_robotwin, id="robotwin"), pytest.param(_libero, id="libero")])
def test_a_surface_without_the_toolkit_s_data_interface_is_told_of_its_dry_run_and_no_query(build) -> None:
    plan = _extended(build)
    assert plan["policy"]["code_mode"]["script_reads"] == ["dry_run"]
    told = _told(plan)
    assert "dry_run" in told
    for word in ("robot.query", "query_ik"):
        assert word not in told, word


def test_the_robodojo_surface_is_told_what_it_serves() -> None:
    """Toolkit v5 plans each move on the policy side, so a script may both query and dry-run it."""
    assert _extended(_robodojo)["policy"]["code_mode"]["script_reads"] == ["dry_run", "query"]
    told = _told(_extended(_robodojo))
    assert "robot.query" in told and "dry_run=True" in told


def _robocasa(code_mode):
    from tests.support.orchestrator.run_specs import native_spec

    spec = native_spec("robocasa")  # its runtime refuses a task, seed or language that is not the RunSpec's
    spec["runtime"].update(task="OpenDrawer", seed=0, expected_task_language="Open the left drawer.")
    spec.update(task="OpenDrawer", task_instruction="Open the left drawer.", seed=0)
    spec["policy"]["code_mode"] = code_mode
    return spec


def _tool_calls(example: str) -> list[tuple[str, dict]]:
    """The ``tools.<name>(...)`` calls of an example program, with their keyword arguments but ``dry_run``."""
    return [(node.func.attr, {item.arg: ast.literal_eval(item.value) for item in node.keywords if item.arg != "dry_run"})
            for node in ast.walk(ast.parse(example)) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and getattr(node.func.value, "id", None) == "tools"]


@pytest.mark.parametrize("build", [*SURFACES, pytest.param(_robocasa, id="robocasa")])
def test_every_example_call_is_one_the_surface_s_own_schema_takes(build) -> None:
    """The examples are written per surface: a dry run in the form the tool's schema takes (RoboTwin's recover_pose
    takes targets, RoboCasa365's strategy) and the cameras the surface has."""
    from ora.agent.composition.surface.schema_validation import validate_schema_instance

    plan = _extended(build)
    functions, _results = _functions_of(plan)
    reads = frozenset(plan["policy"]["code_mode"].get("script_reads", ()))
    schemas = {definition["name"]: definition["inputSchema"] for definition in functions}
    shown = 0
    for definition in functions:
        examples = docs.examples(definition["name"], reads=reads, definition=definition)
        if docs.takes_dry_run(definition["name"], reads):
            assert any(", dry_run=True)" in example for example in examples), definition["name"]
        for example in examples:
            for name, arguments in _tool_calls(example):
                validate_schema_instance(arguments, schemas[name], root_path="$input")
                shown += 1
    assert shown >= 5


def test_a_single_arm_surface_is_never_shown_a_targets_wrapper() -> None:
    """RoboCasa365's tools take one arm's fields directly: the prompt layers, the script tool's description and every
    part of every entry, the library's included, show no ``targets=`` (the prompt's example, ``target()``'s docstring and
    the dry-run examples taught it, and 7 runs sent it)."""
    plan = _extended(_robocasa)
    functions, results = _functions_of(plan)
    reads = frozenset(plan["policy"]["code_mode"].get("script_reads", ()))
    assert reads == {"dry_run"} and not any(docs.interaction_form(item) == "arm" for item in functions)
    interaction = docs.takes_interaction(functions)
    texts = [_told(plan), *(docs.whole(docs.entry_parts(
        definition, results.get(definition["name"]), reads=reads,
        examples=docs.examples(definition["name"], reads=reads, definition=definition))) for definition in functions),
        *(docs.fragment_parts(name, reads=reads, interaction=interaction)["description"]
          for name in docs.library_names(reads))]
    for text in texts:
        assert "targets=" not in text, text[:200]
    assert "tools.move_pose(**target(p_world))" in _told(plan)
    assert 'camera="agentview"' in docs.whole(docs.entry_parts(
        next(item for item in functions if item["name"] == "vision_ground"), reads=reads,
        examples=docs.examples("vision_ground", reads=reads,
                               definition=next(item for item in functions if item["name"] == "vision_ground"))))
