"""``explain``: what one tool call did, in causal order, rendered from its tool's stage template.

A tool that has a template, ``tools/<name>/explain.json`` (``ora.toolkit-explain.v1``), beside its ``description.json``,
can be explained: what it did at each stage, the criterion, the measured value and the verdict, then the conclusion
and, for a failed call, what can be done next. The rendering is deterministic and calls no model: the same inputs give
the same text on every replay. It reads, and only reads:

- ``args``: the call's arguments; ``result``: the tool's result as the policy reads it (its causal projection); a
  refused call has ``error`` instead, and ``decision`` says who refused it (``by`` the ``tool`` or the
  ``supervisor``, with the ``code``);
- ``events``: the stage events the tool recorded (``interface/stages.py``), by stage name, each a list of field maps;
- ``receipt``: the environment receipt of an action the call ran (the motion tools' templates read it);
- ``dry``: the record of a dry run (``interface/dry_run.py``, a motion tool called with ``dry_run=True``), which is
  its ``result`` too; absent for any other call, so a stage reads ``dry`` only under ``when: {"has": "dry"}``.

Template (everything is data, so the fields it reads can be checked against the tool's schemas, ``problems``):

- ``events``: the stage events the tool records, ``{name: [field, ...]}``;
- ``call``: the header's arguments (text clauses); ``outcomes``: the header's verdict, the first entry whose ``if``
  holds (``failed: true`` marks a failed call, which also attaches the explanation to the result by default);
- ``stages``: each with a ``name``, a ``doc`` line for the documentation, an optional ``when``, the ``line`` it
  renders, optional ``each`` (``over`` a list, ``line`` per item, at most ``limit``) and a ``verdict`` (the first
  entry whose ``if`` holds); ``conclusion`` and ``next`` (shown for a failed call only), the first entry whose ``if``
  holds.

A text is a string, or a list of clauses (strings, or ``{"if": condition, "text": ...}``) joined with ``; ``. A
placeholder is ``{path|op|op}`` over the roots above (and ``item``, ``index`` inside ``each``). A path is dotted keys,
``[n]``, ``[]`` (every element), ``[?key=value]`` or ``[?key!=value]`` (the elements whose key does, or does not,
equal the value). A clause or line whose placeholder cannot be read is left out (``strict`` raises ``ExplainMiss``
instead, which is how the tests hold a template to the results it reads); ``|or:value`` stands in for a missing value
and goes through the ops after it. Ops, applied to each element of a ``[]`` list: ``f1`` ``f2`` ``f3`` ``pct``
``pct1`` ``cm`` ``mm`` ``mm1`` ``m`` ``deg`` ``yn`` (numbers) and ``xyz`` ``uv`` ``bbox`` (vectors); over a whole
list: ``count`` ``max`` ``min`` ``sum`` ``join``; ``n:word`` makes ``3 candidates``. A condition is one of ``has``,
``empty``, ``eq``, ``ne``, ``gt``, ``ge``, ``lt``, ``le``, ``in``, ``all``, ``any``, ``not``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .interface import dry_run

TEMPLATE_SCHEMA = "ora.toolkit-explain.v1"
TOOLS_ROOT = Path(__file__).resolve().parent / "tools"
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
#: Items an ``each`` shows when its template gives no ``limit``.
EACH_LIMIT = 5
#: The most characters an explanation attached to a result carries, and all the explanations one script result's
#: ``calls`` carry (a failed call past that still has its own in the answer the script reads).
ATTACHED_CHARS = 1800
SCRIPT_CHARS = 6000
#: Refusal codes the code-mode host or the Supervisor gives (every other code is the tool's own).
SUPERVISOR_CODES = frozenset({
    "action_not_started", "action_pending_fresh_observation", "execution_unavailable", "script_action_limit",
    "script_action_refused", "script_call_limit", "script_lifecycle_unavailable", "script_query_limit",
    "script_tool_unavailable", "turn_ended", "world_changed_by_script"})
#: What can be done next after a refusal by the host, for every tool (a template's own ``next`` comes first).
HOST_NEXT = {
    "script_call_limit": "end this script and make the remaining calls in the next one, which starts with a fresh "
                         "allowance; batch what can be batched",
    "script_action_limit": "end this script; the next one starts with a fresh allowance of world-changing calls",
    "script_query_limit": "end this script; the next one starts with a fresh allowance of queries and dry runs",
    "action_pending_fresh_observation": "an action closed the current observation: end the script (hand back) and "
                                        "perceive again on the fresh observation of the next turn",
    "turn_ended": "an action ended this turn: end the script; the next turn starts from a fresh observation",
    "execution_unavailable": "the Supervisor could not run the tool just now: retry the call once, and hand back if "
                             "it fails again",
}
_ERROR_FIELDS = frozenset({"code", "message", "retryable", "retry_requires_changed_arguments", "infrastructure",
                           "field", "received", "constraints", "repetition"})
_MISSING = object()


class ExplainMiss(KeyError):
    """A placeholder or condition read a field the inputs do not have (``strict`` rendering)."""


@dataclass(frozen=True)
class Explanation:
    text: str
    failed: bool


class _Mapped(list):
    """The elements a ``[]`` or ``[?...]`` step selected: later steps and formats apply to each one."""


# --- templates ------------------------------------------------------------------------------------------------------

def tools() -> tuple[str, ...]:
    """The tools that have a template, sorted by name."""

    return tuple(sorted(path.parent.name for path in TOOLS_ROOT.glob("*/explain.json") if _NAME.match(path.parent.name)))


@lru_cache(maxsize=None)
def _load(name: str) -> str | None:
    path = TOOLS_ROOT / name / "explain.json"
    return path.read_text(encoding="utf-8") if path.is_file() else None


def template(tool: str) -> dict[str, Any] | None:
    """The tool's stage template, or None for a tool without one (a copy: the caller may keep it)."""

    text = _load(tool) if isinstance(tool, str) and _NAME.match(tool) else None
    if text is None:
        return None
    value = json.loads(text)
    if not isinstance(value, dict) or value.get("schema") != TEMPLATE_SCHEMA or value.get("tool") != tool:
        raise ValueError(f"tools/{tool}/explain.json is not an {TEMPLATE_SCHEMA} template for {tool}")
    return value


# --- paths ----------------------------------------------------------------------------------------------------------

_STEP = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]|(\[\])|\[\?([A-Za-z0-9_.]+)(!?=)([^\]]*)\]")
_ROOT = re.compile(r"^[a-z_]+")


def _tokens(path: str) -> tuple[str, list[tuple[Any, ...]]]:
    match = _ROOT.match(path)
    if match is None:
        raise ValueError(f"bad explain path {path!r}")
    steps, position = [], match.end()
    while position < len(path):
        step = _STEP.match(path, position)
        if step is None:
            raise ValueError(f"bad explain path {path!r}")
        key, index, each, where, comparison, literal = step.groups()
        steps.append(("key", key) if key else ("index", int(index)) if index else ("each",) if each
                     else ("where", where, _literal(literal), comparison == "!="))
        position = step.end()
    return match.group(0), steps


def _literal(text: str) -> Any:
    if text in ("true", "false"):
        return text == "true"
    if text == "null":
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _get(value: Any, key: str) -> Any:
    return value.get(key, _MISSING) if isinstance(value, Mapping) else _MISSING


def _resolve(path: str, scope: Mapping[str, Any]) -> Any:
    root, steps = _tokens(path)
    value = scope.get(root, _MISSING)
    for step in steps:
        if value is _MISSING or value is None:
            return _MISSING
        if isinstance(value, _Mapped) and step[0] in ("key", "where"):
            if step[0] == "key":
                found = [_get(item, step[1]) for item in value]
                value = _Mapped(item for item in found if item is not _MISSING and item is not None)
            else:
                value = _Mapped(item for item in value if (_resolve_in(item, step[1]) == step[2]) != step[3])
        elif step[0] == "key":
            value = _get(value, step[1])
        elif step[0] == "index":
            value = value[step[1]] if isinstance(value, list) and step[1] < len(value) else _MISSING
        elif step[0] == "each":
            value = _Mapped(value) if isinstance(value, list) else _MISSING
        elif isinstance(value, list):
            value = _Mapped(item for item in value if (_resolve_in(item, step[1]) == step[2]) != step[3])
        else:
            value = _MISSING
    return value


def _resolve_in(item: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        item = _get(item, key)
    return item


# --- conditions -----------------------------------------------------------------------------------------------------

def _present(value: Any) -> bool:
    return value is not _MISSING and value is not None


def _empty(value: Any) -> bool:
    return not _present(value) or (isinstance(value, (list, dict, str)) and len(value) == 0)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _holds(condition: Any, scope: Mapping[str, Any]) -> bool:
    if not isinstance(condition, Mapping) or len(condition) != 1:
        raise ValueError(f"an explain condition has exactly one key: {condition!r}")
    (op, argument), = condition.items()
    if op == "has":
        return _present(_resolve(argument, scope))
    if op == "empty":
        return _empty(_resolve(argument, scope))
    if op == "not":
        return not _holds(argument, scope)
    if op == "all":
        return all(_holds(item, scope) for item in argument)
    if op == "any":
        return any(_holds(item, scope) for item in argument)
    value = _resolve(argument[0], scope)
    if op == "eq":
        return value is not _MISSING and value == argument[1]
    if op == "ne":
        return value is _MISSING or value != argument[1]
    if op == "in":
        return value is not _MISSING and value in argument[1]
    if op in ("gt", "ge", "lt", "le"):
        if not _number(value):
            return False
        return {"gt": value > argument[1], "ge": value >= argument[1], "lt": value < argument[1],
                "le": value <= argument[1]}[op]
    raise ValueError(f"unknown explain condition {op!r}")


# --- text -----------------------------------------------------------------------------------------------------------

def _trim(text: str) -> str:
    return text.rstrip("0").rstrip(".") if "." in text else text


def _plain(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return _trim(f"{value:.3f}")
    if isinstance(value, list):
        return ", ".join(_plain(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _pixel(value: Any) -> str:
    return str(round(value)) if isinstance(value, float) else str(value)


#: Formats of one number; applied to each element of a ``[]`` list.
SCALAR_OPS = {
    "f1": lambda v: f"{v:.1f}", "f2": lambda v: f"{v:.2f}", "f3": lambda v: f"{v:.3f}",
    "pct": lambda v: f"{v * 100:.0f}%", "pct1": lambda v: f"{v * 100:.1f}%",
    "cm": lambda v: f"{v * 100:.1f} cm", "mm": lambda v: f"{v * 1000:.0f} mm", "mm1": lambda v: f"{v * 1000:.1f} mm",
    "m": lambda v: f"{v:.3f} m",
    "deg": lambda v: f"{v:.1f} deg", "yn": lambda v: "yes" if v else "no",
}
#: Formats of one vector.
VECTOR_OPS = {
    "xyz": lambda v: "(" + ", ".join(f"{item:.3f}" for item in v) + ")",
    "uv": lambda v: "(" + ", ".join(_pixel(item) for item in v) + ")",
    "bbox": lambda v: f"({_pixel(v[0])}, {_pixel(v[1])})-({_pixel(v[2])}, {_pixel(v[3])})",
}
LIST_OPS = frozenset({"count", "max", "min", "sum", "join"})
OPS = frozenset(SCALAR_OPS) | frozenset(VECTOR_OPS) | LIST_OPS | {"n", "or"}


def _apply(op: str, argument: str | None, value: Any) -> Any:
    if op == "count":
        return len(value) if isinstance(value, list) else 1
    if op in ("max", "min", "sum"):
        numbers = [item for item in (value if isinstance(value, list) else [value]) if _number(item)]
        if not numbers:
            raise ExplainMiss(op)
        return {"max": max, "min": min, "sum": sum}[op](numbers)
    if op == "join":
        return ", ".join(_plain(item) for item in value) if isinstance(value, list) else _plain(value)
    if op == "n":
        count = len(value) if isinstance(value, list) else value
        singular, _, plural = (argument or "").partition("/")
        return f"{_plain(count)} {singular if count == 1 else plural or singular + 's'}"
    table = SCALAR_OPS if op in SCALAR_OPS else VECTOR_OPS
    if isinstance(value, _Mapped):
        return _Mapped(_apply(op, argument, item) for item in value)
    return table[op](value)


def _placeholder(body: str, scope: Mapping[str, Any]) -> str:
    path, *operations = body.split("|")
    path = path.strip()
    steps = [item.partition(":")[::2] for item in operations]
    for op, _argument in steps:
        if op not in OPS:
            raise ValueError(f"unknown explain op {op!r} in {{{body}}}")
    value = _resolve(path, scope)
    if not _present(value):
        fallback = next((argument for op, argument in steps if op == "or"), None)
        if fallback is None:
            raise ExplainMiss(path)
        value = _literal(fallback)
    try:
        for op, argument in steps:
            if op != "or":
                value = _apply(op, argument or None, value)
    except (TypeError, ValueError, IndexError, ExplainMiss) as error:
        raise ExplainMiss(path) from error
    return _plain(value)


_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def _fill(text: str, scope: Mapping[str, Any]) -> str:
    text = text.replace("{{", "\0").replace("}}", "\1")
    return _PLACEHOLDER.sub(lambda match: _placeholder(match.group(1), scope), text).replace("\0", "{").replace("\1", "}")


def _clauses(line: Any, scope: Mapping[str, Any], strict: bool, joiner: str = "; ") -> str:
    """A text rendered: a string, or its clauses joined with ``joiner`` (one that cannot be read is left out)."""

    parts = [line] if isinstance(line, (str, Mapping)) else list(line)
    rendered = []
    for part in parts:
        text = part if isinstance(part, str) else part["text"]
        if isinstance(part, Mapping) and "if" in part and not _holds(part["if"], scope):
            continue
        try:
            rendered.append(_fill(text, scope))
        except ExplainMiss:
            if strict:
                raise
    return joiner.join(item for item in rendered if item)


def _first(entries: Sequence[Mapping[str, Any]] | None, scope: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return next((entry for entry in entries or () if "if" not in entry or _holds(entry["if"], scope)), None)


# --- rendering ------------------------------------------------------------------------------------------------------

def decision_of(error: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Who refused a call: the Supervisor (or code mode's limits) for a host code, else the tool; None if none did."""

    if not isinstance(error, Mapping):
        return None
    code = error.get("code")
    return {"by": "supervisor" if code in SUPERVISOR_CODES else "tool", "code": code}


def _stage_events(events: Iterable[Mapping[str, Any]]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for event in events or ():
        grouped.setdefault(str(event.get("stage")), []).append(dict(event.get("fields") or {}))
    return grouped


def render(tool: str, *, arguments: Mapping[str, Any] | None = None, result: Mapping[str, Any] | None = None,
           events: Iterable[Mapping[str, Any]] = (), receipt: Mapping[str, Any] | None = None,
           decision: Mapping[str, Any] | None = None, strict: bool = False) -> Explanation | None:
    """The tool's explanation of one call, or None for a tool without a template.

    ``result`` is the tool's result as the policy reads it, or ``{"error": ...}`` for a refused call; ``events`` the
    call's stage events; ``decision`` defaults to who refused the call, from its error.
    """

    spec = template(tool)
    if spec is None:
        return None
    return render_spec(spec, tool=tool, arguments=arguments, result=result, events=events, receipt=receipt,
                       decision=decision, strict=strict)


def render_spec(spec: Mapping[str, Any], *, tool: str, arguments: Mapping[str, Any] | None = None,
                result: Mapping[str, Any] | None = None, events: Iterable[Mapping[str, Any]] = (),
                receipt: Mapping[str, Any] | None = None, decision: Mapping[str, Any] | None = None,
                strict: bool = False) -> Explanation:
    """``render`` for a template in hand (the tests', a tool author's draft)."""

    result = dict(result or {})
    error = result.get("error") if isinstance(result.get("error"), Mapping) else None
    scope: dict[str, Any] = {
        "args": dict(arguments or {}), "result": result, "error": error, "receipt": receipt,
        "decision": dict(decision) if decision is not None else decision_of(error), "events": _stage_events(events),
        "dry": result if result.get("dry_run") is True else None,
    }
    scope["events"] = _EventsView(scope["events"])
    lines: list[str] = []
    outcome = _first(spec.get("outcomes"), scope)
    failed = error is not None or bool(outcome and outcome.get("failed"))
    header = f"{tool}({_clauses(spec.get('call', ''), scope, strict, ', ')})"
    verdict = _clauses(outcome["text"], scope, strict) if outcome else (f"refused: {error.get('code')}" if error else "")
    lines.append(f"{header}  {verdict}".rstrip())
    number = 0
    for stage in spec.get("stages", ()):
        if "when" in stage and not _holds(stage["when"], scope):
            continue
        head = _clauses(stage["line"], scope, strict)
        if not head:
            continue
        number += 1
        closing = _first(stage.get("verdict"), scope)
        tail = _clauses(closing["text"], scope, strict) if closing else ""
        lines.append(f"{number}. {stage['name']}: {head}" + (f"  -> {tail}" if tail else ""))
        each = stage.get("each")
        if each:
            lines.extend(_items(each, scope, strict))
    if error is not None:  # a call that did not run: who refused it, and why
        who = "the Supervisor" if (scope["decision"] or {}).get("by") == "supervisor" else "the tool"
        number += 1
        lines.append(f"{number}. refusal: {who} refused the call: {error.get('message') or error.get('code')}"
                     "  -> not run")
    last = _first(spec.get("conclusion"), scope) if error is None else None
    if last is not None:
        text = _clauses(last["text"], scope, strict)
        if text:
            lines.append(f"conclusion: {text}")
    if failed:
        step = _first(spec.get("next"), scope)
        text = _clauses(step["text"], scope, strict) if step else HOST_NEXT.get(str((error or {}).get("code")), "")
        if text:
            lines.append(f"next: {text}")
    return Explanation("\n".join(lines) + "\n", failed)


class _EventsView(dict):
    """``events``: a stage's events by name; a stage that recorded nothing reads as an empty list."""

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, [])


def _items(each: Mapping[str, Any], scope: Mapping[str, Any], strict: bool) -> list[str]:
    values = _resolve(each["over"], scope)
    values = list(values) if isinstance(values, list) else []
    limit = int(each.get("limit", EACH_LIMIT))
    lines = []
    for index, item in enumerate(values[:limit]):
        text = _clauses(each["line"], {**scope, "item": item, "index": index}, strict)
        if text:
            lines.append(f"   {text}")
    if len(values) > limit:
        lines.append(f"   ... and {len(values) - limit} more")
    return lines


def render_answer(tool: str, arguments: Mapping[str, Any], answer: Mapping[str, Any],
                  events: Iterable[Mapping[str, Any]] = ()) -> Explanation | None:
    """The explanation of a call from the answer a script reads for it (``agent/pi/code_mode.py``): its projected
    ``result`` (a refused call's is its ``error``), its ``receipt`` where an action ran, and the stage events."""

    result = answer.get("result") if isinstance(answer.get("result"), Mapping) else {}
    error = result.get("error") if isinstance(result.get("error"), Mapping) else answer.get("error")
    if isinstance(error, Mapping) and "error" not in result:
        result = {**result, "error": error}
    receipt = answer.get("receipt") if isinstance(answer.get("receipt"), Mapping) else None
    return render(tool, arguments=arguments, result=result, events=events, receipt=receipt)


# --- what a template reads ------------------------------------------------------------------------------------------

def _texts(line: Any) -> Iterable[str]:
    for part in [line] if isinstance(line, (str, Mapping)) else line or ():
        yield part if isinstance(part, str) else part["text"]


def _condition_paths(condition: Any) -> Iterable[str]:
    if isinstance(condition, Mapping):
        for op, argument in condition.items():
            if op in ("all", "any"):
                for item in argument:
                    yield from _condition_paths(item)
            elif op == "not":
                yield from _condition_paths(argument)
            else:
                yield argument if isinstance(argument, str) else argument[0]


def reads(spec: Mapping[str, Any]) -> list[tuple[str, str | None]]:
    """Every path the template reads, as ``(path, over)``: ``over`` is the ``each`` list an ``item`` path is read
    from (None otherwise), in the order first read."""

    found: list[tuple[str, str | None]] = []

    def text(line: Any, over: str | None) -> None:
        for chunk in _texts(line):
            for body in _PLACEHOLDER.findall(chunk.replace("{{", "").replace("}}", "")):
                found.append((body.split("|")[0].strip(), over))
        for part in [line] if isinstance(line, (str, Mapping)) else line or ():
            if isinstance(part, Mapping) and part.get("if"):
                condition(part["if"], over)

    def condition(value: Any, over: str | None) -> None:
        found.extend((path, over) for path in _condition_paths(value))

    text(spec.get("call", ""), None)
    for section in ("outcomes", "conclusion", "next"):
        for entry in spec.get(section) or ():
            text(entry["text"], None)
            condition(entry.get("if"), None)
    for stage in spec.get("stages", ()):
        if stage.get("when"):
            condition(stage["when"], None)
        text(stage["line"], None)
        for entry in stage.get("verdict") or ():
            text(entry["text"], None)
            condition(entry.get("if"), None)
        if stage.get("each"):
            found.append((stage["each"]["over"], None))
            text(stage["each"]["line"], stage["each"]["over"])
    seen: set[tuple[str, str | None]] = set()
    return [pair for pair in found if not (pair in seen or seen.add(pair))]


# --- checking a template against the schemas ------------------------------------------------------------------------

def _expand(schema: Any, root: Mapping[str, Any], seen: tuple[str, ...] = ()) -> list[Mapping[str, Any]]:
    """The object schemas a schema allows: ``$ref`` followed, ``oneOf``/``anyOf``/``allOf`` branches included."""

    if not isinstance(schema, Mapping):
        return []
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/") and ref not in seen:
        target: Any = root
        for part in ref[2:].split("/"):
            target = target.get(part) if isinstance(target, Mapping) else None
        return _expand(target, root, (*seen, ref))
    found = [schema]
    for keyword in ("oneOf", "anyOf", "allOf"):
        for branch in schema.get(keyword) or ():
            found.extend(_expand(branch, root, seen))
    return found


def _follow(schemas: list[Mapping[str, Any]], steps: list[tuple[Any, ...]], root: Mapping[str, Any]
            ) -> list[Mapping[str, Any]]:
    """The schemas a path ends at (empty: no such field); ``[]``, ``[n]`` and ``[?...]`` step into the items."""

    for step in steps:
        following: list[Mapping[str, Any]] = []
        for schema in schemas:
            if step[0] == "key":
                properties = schema.get("properties")
                if isinstance(properties, Mapping) and step[1] in properties:
                    following.extend(_expand(properties[step[1]], root))
                elif isinstance(schema.get("additionalProperties"), Mapping):
                    following.extend(_expand(schema["additionalProperties"], root))
                elif schema.get("additionalProperties") is True and not properties:  # an open object (a receipt)
                    following.append({"additionalProperties": True})
            elif isinstance(schema.get("items"), Mapping):
                following.extend(_expand(schema["items"], root))
            elif step[0] == "where":  # a filter on the elements already selected
                following.append(schema)
        schemas = following
        if not schemas:
            return []
    return schemas


def _element(schema: Mapping[str, Any], over: str) -> list[Mapping[str, Any]]:
    """The schemas of the elements of the list ``over`` names in the result schema."""

    _root, steps = _tokens(over)
    found = _follow(_expand(schema, schema), steps, schema)
    return found if steps and steps[-1][0] in ("each", "where") else _follow(found, [("each",)], schema)


def problems(spec: Mapping[str, Any], input_schema: Mapping[str, Any], result_schema: Mapping[str, Any],
             dry_schema: Mapping[str, Any] = dry_run.RESULT_SCHEMA) -> list[str]:
    """What in the template reads a field that does not exist: a path outside the tool's input or result schema (or a
    dry run's, ``dry``), a stage event or event field it does not declare, an ``item`` outside its ``each`` list's
    elements. Empty when sound."""

    issues: list[str] = []
    events = spec.get("events") or {}
    for path, over in reads(spec):
        root, steps = _tokens(path)
        if root == "args":
            ok = bool(_follow(_expand(input_schema, input_schema), steps, input_schema))
        elif root == "result":
            ok = bool(_follow(_expand(result_schema, result_schema), steps, result_schema))
        elif root == "dry":
            ok = not steps or bool(_follow(_expand(dry_schema, dry_schema), steps, dry_schema))
        elif root == "item" and over is not None and over.startswith("events."):
            name = _tokens(over)[1][0][1]
            ok = len(steps) == 1 and steps[0][0] == "key" and steps[0][1] in events.get(name, ())
        elif root == "item" and over is not None and over.startswith("dry"):
            ok = bool(_follow(_element(dry_schema, over), steps, dry_schema))
        elif root == "item":
            ok = over is not None and over.startswith("result") and bool(_follow(_element(result_schema, over), steps,
                                                                                result_schema))
        elif root == "events":
            rest = steps[1:]
            ok = bool(steps) and steps[0][0] == "key" and steps[0][1] in events and (
                not rest or (len(rest) == 2 and rest[0][0] in ("index", "each") and rest[1][0] == "key"
                             and rest[1][1] in events[steps[0][1]]))
        elif root == "error":
            ok = not steps or (steps[0][0] == "key" and steps[0][1] in _ERROR_FIELDS)
        elif root == "decision":
            ok = not steps or (steps[0][0] == "key" and steps[0][1] in ("by", "code"))
        else:
            ok = root in ("receipt", "index")
        if not ok:
            issues.append(f"{path} is not a field of the call's {root}")
        for position, step in enumerate(steps):  # a filter's key is a field of the elements it selects
            if step[0] == "where" and root == "result":
                selected = _follow(_expand(result_schema, result_schema), steps[:position], result_schema)
                elements = [element for schema in selected for element in (
                    _expand(schema["items"], result_schema) if isinstance(schema.get("items"), Mapping) else [schema])]
                if not _follow(elements, [("key", key) for key in step[1].split(".")], result_schema):
                    issues.append(f"{path} filters on {step[1]}, which the elements do not have")
    return issues


def documentation(spec: Mapping[str, Any]) -> str:
    """The template as the tool's ``explain`` documentation part: its stages and the stage events it reads."""

    lines = ["Explain (robot.explain(r); a failed call carries it as r.explain), in causal order:"]
    for number, stage in enumerate(spec.get("stages", ()), start=1):
        used = sorted({path for path, over in reads({"stages": [stage]}) if over is None})
        lines.append(f"{number}. {stage['name']}: {stage.get('doc', '')}" + (f" (reads {', '.join(used)})" if used else ""))
    lines.append("Then the conclusion and, for a failed call, what can be done next.")
    events = spec.get("events") or {}
    if events:
        lines.append("Stage events the tool records for it:")
        lines.extend(f"- {name}({', '.join(fields)})" for name, fields in sorted(events.items()))
    return "\n".join(lines)


__all__ = ["ATTACHED_CHARS", "EACH_LIMIT", "SCRIPT_CHARS", "ExplainMiss", "Explanation", "HOST_NEXT", "SUPERVISOR_CODES",
           "TEMPLATE_SCHEMA", "decision_of", "documentation", "problems", "reads", "render", "render_answer", "render_spec",
           "template", "tools"]
