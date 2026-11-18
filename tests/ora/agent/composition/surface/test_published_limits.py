"""Every enforced number in policy-visible tool help equals the code that enforces it.

One table, `tests/data/published-limits.json`, maps each limit to its owner --
the constant, profile value or source literal the handler or backend enforces --
and lists every sentence that states it, per surface. The test composes each
surface, finds each sentence in the named tool's delivered description or
input-schema text, parses the stated number and compares it with the owner after
unit conversion.

Surfaces: the current defaults. Each is file-backed (written out of the
published-limits layers that ended their manifests), so the text is the
surface's own tool files. A default this table does not cover
is recorded under `not_covered` with its reason.

A census then requires every digit in every composed tool text to belong to a
registered statement or a reviewed informational entry, so a new number cannot
enter the help without an owner.
"""

from __future__ import annotations

import fnmatch
import functools
import importlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterator

import pytest

from ora.agent.composition.surface import tool_surface
from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from tests.support.agent.published_limits import ROOT, TABLE, SURFACES, _surface

WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "sixteen": 16, "twenty": 20, "pi": math.pi,
}
#: Every unit a statement or owner may use, as a factor to one base unit.
UNITS = {
    "m": ("length", 1.0), "cm": ("length", 1e-2), "mm": ("length", 1e-3),
    "rad": ("angle", 1.0), "deg": ("angle", math.pi / 180.0),
    "s": ("time", 1.0), "ms": ("time", 1e-3), "Hz": ("frequency", 1.0),
    "rad/s": ("angular_speed", 1.0), "deg/s": ("angular_speed", math.pi / 180.0),
}
#: A digit run that is not part of an identifier (T0, link6, x0, sha256, v3).
DIGITS = re.compile(r"(?<![A-Za-z_\d.])\d+(?:\.\d+)?(?:e-?\d+)?")


# --------------------------------------------------------------- composition

def current_surfaces() -> tuple[str, ...]:
    """The surfaces a fresh or live consumer binds: every default instance and the defaults they replaced.

    The replaced defaults are read from tests/data/backend-parity.json previous_defaults;
    one that is no longer registered cannot be bound and is not listed.
    """

    activation = json.loads((ROOT / "tests/data/default-expectations.json").read_text(encoding="utf-8"))
    found = {instance["tool_surface"] for instance in activation["instances"].values()}
    parity = json.loads((ROOT / "tests/data/backend-parity.json").read_text(encoding="utf-8"))
    found |= {record["profile_id"] for record in parity["previous_defaults"].values()
              if record["profile_id"] in PROFILE_CATALOG}
    return tuple(sorted(found))


def _walk(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(node, dict):
        for name, value in node.items():
            if name == "description" and isinstance(value, str):
                yield path + "/description", value
            else:
                yield from _walk(value, f"{path}/{name}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}/{index}")


@functools.cache
def _fields(key: str) -> dict[str, tuple[tuple[str, str], ...]]:
    """Every policy-visible text of every tool: its description, then its schema texts."""

    return {
        item["name"]: (("description", item["description"]),
                       *_walk(item.get("inputSchema", {}), "inputSchema"))
        for item in _surface(key).definitions
    }


def _keys(patterns: list[str]) -> list[str]:
    found = [key for key in SURFACES if any(fnmatch.fnmatchcase(key, p) for p in patterns)]
    assert found, f"no table surface matches {patterns}"
    return found


# --------------------------------------------------------------- owners

def _python(reference: str, path: list[Any]) -> Any:
    module, name = reference.split(":")
    value: Any = getattr(importlib.import_module(module), name)
    for step in path:
        if step == "()":
            value = value()
        elif isinstance(value, (dict, list, tuple)):
            value = value[step]
        else:
            value = getattr(value, step)
    return value


def _pointer(document: Any, pointer: str) -> Any:
    for part in pointer.split("/")[1:]:
        document = document[int(part)] if isinstance(document, list) else document[part]
    return document


def _convert(value: float, unit: str, into: str) -> float:
    if unit == into:
        return float(value)
    (dimension, scale), (target, target_scale) = UNITS[unit], UNITS[into]
    assert dimension == target, f"cannot convert {unit} into {into}"
    return float(value) * scale / target_scale


def _types(node: dict[str, Any]) -> list[Any]:
    declared = node.get("type")
    return declared if isinstance(declared, list) else [declared]


def owner_value(owner: dict[str, Any], key: str) -> float:
    """The enforcing value of one owner, in the owner's own unit."""

    if "python" in owner:
        value = _python(owner["python"], owner.get("path", []))
    elif "json" in owner:
        value = _pointer(json.loads((ROOT / owner["json"]).read_text(encoding="utf-8")), owner["pointer"])
    elif "source" in owner:
        found = re.findall(owner["pattern"], (ROOT / owner["source"]).read_text(encoding="utf-8"))
        assert len(found) == 1, f"{owner['source']}: {owner['pattern']!r} matched {len(found)} times"
        value = float(found[0])
    elif "count_excluding" in owner:
        spec = owner["count_excluding"]
        items = list(_python(spec["items"], []))
        if "after" in spec:
            items = items[items.index(spec["after"]) + 1:]
        value = sum(1 for item in items if item not in set(_python(spec["exclude"], [])))
    elif "schema_keyword" in owner:
        spec = owner["schema_keyword"]
        tool = next(item for item in _surface(key).definitions if item["name"] == spec["tool"])
        value = _pointer(tool["inputSchema"], spec["pointer"])
    elif "schema_array_depth" in owner:
        spec = owner["schema_array_depth"]
        tool = next(item for item in _surface(key).definitions if item["name"] == spec["tool"])
        node, value = _pointer(tool["inputSchema"], spec["pointer"]), 0
        while isinstance(node, dict) and "array" in _types(node):
            value, node = value + 1, node.get("items")
    elif "derive" in owner:
        # Each part in its base unit (a count stays a count); the result is read
        # in the owner's unit.
        values = [owner_value(part, key) * UNITS.get(part["unit"], ("", 1.0))[1]
                  for part in owner["of"]]
        operation = owner["derive"]
        if operation == "difference":
            value = values[0] - sum(values[1:])
        elif operation == "product":
            value = math.prod(values)
        elif operation == "reciprocal":
            value = 1.0 / values[0]
        elif operation == "negative":
            value = -values[0]
        else:
            raise AssertionError(f"unknown derivation {operation!r}")
        value /= UNITS.get(owner["unit"], ("", 1.0))[1]
    else:
        raise AssertionError(f"owner has no recognised kind: {sorted(owner)}")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AssertionError(f"owner value is not a finite number: {value!r}")
    return float(value)


def stated_value(literal: str) -> float:
    sign, word = (-1.0, literal[1:].lower()) if literal.startswith("-") else (1.0, literal.lower())
    return sign * WORDS[word] if word in WORDS else float(literal.replace(",", ""))


def _split_dimensions(text: str) -> str:
    """Read a card size such as 1024x576 as two numbers: that x is not part of an identifier."""

    return re.sub(r"(?<=\d)x(?=\d)", " ", text)


def _literal_offset(statement: dict[str, Any]) -> int:
    """Where the stated number sits in its sentence: the n-th whole-token occurrence."""

    pattern = rf"(?<![A-Za-z_\d.]){re.escape(statement['literal'])}(?![A-Za-z_\d])"
    found = [match.start() for match in re.finditer(pattern, _split_dimensions(statement["text"]))]
    index = statement.get("literal_index", 0)
    assert index < len(found), f"{statement['literal']!r} is not a number in {statement['text']!r}"
    return found[index]


# --------------------------------------------------------------- cases

def _statement_cases() -> Iterator[Any]:
    for limit in TABLE["limits"]:
        for index, statement in enumerate(limit["statements"]):
            for key in _keys(statement["surfaces"]):
                yield pytest.param(key, limit, statement, id=f"{limit['key']}-{index}-{key}")


@pytest.mark.parametrize("key,limit,statement", list(_statement_cases()))
def test_the_stated_number_is_the_enforced_number(key, limit, statement):
    fields = _fields(key).get(statement["tool"])
    assert fields is not None, f"{key} declares no {statement['tool']}"
    texts = [text for _path, text in fields if statement["text"] in text]
    assert texts, f"{key} {statement['tool']} no longer says: {statement['text']!r}"
    assert _literal_offset(statement) >= 0
    capabilities = set(_surface(key).capabilities)
    if "requires_capability" in limit:
        assert limit["requires_capability"] in capabilities, f"{key} lacks {limit['requires_capability']}"
    if "absent_capability" in limit:
        assert limit["absent_capability"] not in capabilities, f"{key} declares {limit['absent_capability']}"
    unit = statement.get("unit", limit["unit"])
    stated = stated_value(statement["literal"])
    actual = owner_value(limit["owner"], key)
    if unit in UNITS and limit["unit"] in UNITS:
        actual = _convert(actual, limit["unit"], unit)
    else:
        assert unit == limit["unit"], f"{limit['key']}: {unit} against {limit['unit']}"
    if statement.get("rounded"):
        decimals = len(statement["literal"].partition(".")[2])
        tolerance = 0.5 * 10 ** -decimals + 1e-12
    else:
        tolerance = 1e-9 * max(1.0, abs(actual))
    assert abs(stated - actual) <= tolerance, (
        f"{key} {statement['tool']}: the help says {statement['literal']} {unit}; "
        f"{limit['key']} enforces {actual:g} {unit}"
    )


def _schema_cases() -> Iterator[Any]:
    for limit in TABLE["limits"]:
        for index, check in enumerate(limit.get("schema", [])):
            for key in _keys(check["surfaces"]):
                yield pytest.param(key, limit, check, id=f"{limit['key']}-schema{index}-{key}")


@pytest.mark.parametrize("key,limit,check", list(_schema_cases()))
def test_the_schema_keyword_is_the_enforced_number(key, limit, check):
    tool = next(item for item in _surface(key).definitions if item["name"] == check["tool"])
    declared = _pointer(tool["inputSchema"], check["pointer"])
    actual = owner_value(limit["owner"], key)
    unit = check.get("unit", limit["unit"])
    if unit in UNITS and limit["unit"] in UNITS:
        actual = _convert(actual, limit["unit"], unit)
    assert math.isclose(float(declared) * check.get("sign", 1), actual, rel_tol=1e-9), (
        f"{key} {check['tool']}{check['pointer']} = {declared}; {limit['key']} enforces {actual:g}"
    )


# --------------------------------------------------------------- census

def _covered(key: str, tool: str, text: str) -> list[tuple[int, int]]:
    spans = []
    for limit in TABLE["limits"]:
        for statement in limit["statements"]:
            if statement["tool"] != tool or key not in _keys(statement["surfaces"]):
                continue
            start = text.find(statement["text"])
            while start >= 0:
                offset = start + _literal_offset(statement)
                spans.append((offset, offset + len(statement["literal"])))
                start = text.find(statement["text"], start + 1)
    for entry in TABLE["informational"]:
        if tool not in entry["tools"] or key not in _keys(entry["surfaces"]):
            continue
        start = text.find(entry["text"])
        while start >= 0:
            spans.append((start, start + len(entry["text"])))
            start = text.find(entry["text"], start + 1)
    return spans


@pytest.mark.parametrize("key", list(SURFACES))
def test_every_number_in_the_help_is_registered(key):
    unregistered = []
    for tool, fields in _fields(key).items():
        for path, text in fields:
            spans = _covered(key, tool, text)
            for match in DIGITS.finditer(_split_dimensions(text)):
                if not any(start <= match.start() and match.end() <= end for start, end in spans):
                    context = text[max(0, match.start() - 50):match.end() + 30]
                    unregistered.append(f"{tool} {path}: {match.group(0)} in ...{context}...")
    assert not unregistered, "numbers without a registered owner:\n" + "\n".join(unregistered)


# --------------------------------------------------------------- table rules

def test_the_table_covers_every_current_surface():
    profiles = {spec["profile"] for spec in SURFACES.values()}
    for profile in current_surfaces():
        assert profile in profiles or profile in TABLE["not_covered"], profile
    assert all(reason.strip() for reason in TABLE["not_covered"].values())
    assert set(TABLE["not_covered"]) <= set(current_surfaces())
    activation = json.loads((ROOT / "tests/data/default-expectations.json").read_text(encoding="utf-8"))
    # A default the table records as not covered (RoboDojo's file-backed toolkit v6) has no key.
    covered = {backend: instance["tool_surface"] for backend, instance in activation["instances"].items()
               if instance["tool_surface"] not in TABLE["not_covered"]}
    assert {spec["backend"] for spec in SURFACES.values()} == set(covered)
    # Every default instance is in the table, unless the table records why it is not covered.
    assert set(covered.values()) == profiles


def test_keys_are_unique_and_every_limit_is_stated():
    keys = [limit["key"] for limit in TABLE["limits"]]
    assert len(keys) == len(set(keys))
    for limit in TABLE["limits"]:
        assert limit["statements"], limit["key"]
    for entry in TABLE["informational"]:
        assert entry["reason"].strip() and entry["text"].strip()


# --------------------------------------------------------------- current texts

@pytest.mark.parametrize("key", list(SURFACES))
def test_every_successor_tool_states_what_it_refuses(key):
    """N2: every tool's help names its ex-ante refusals, as rules and nothing more.

    A tool whose text a binder writes after every layer (``finish_episode``)
    states them too: its binder writes the sentence.
    """

    for item in _surface(key).definitions:
        on_error = item["description"].split("On error:", 1)[1]
        assert "Refused:" in on_error, f"{key} {item['name']}"
