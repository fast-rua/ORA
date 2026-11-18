"""Render Python reference text for `::: module.Object` lines on the Python API pages.

Usage: prepare-python-docs.py IDENTIFIER... > reference.json

Griffe reads the source statically (allow_inspection=False): ORA and its runtime
dependencies are never imported. The output maps each identifier to Markdown that
replaces its `:::` line: a level-2 heading pinned to the full dotted path (the
anchor the earlier site used), the signature, the docstring, a table of public
attributes, and a level-3 entry per public method, in source order. Members whose
names start with "_" are left out.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import griffe

ROOT = Path(__file__).resolve().parents[2]


def _expr(value: object) -> str:
    return "" if value is None else str(value)


def _signature(function: griffe.Function, *, drop_self: bool) -> str:
    parts: list[str] = []
    seen_keyword_only = False
    parameters = list(function.parameters)
    if drop_self and parameters and parameters[0].name in {"self", "cls"}:
        parameters = parameters[1:]
    for parameter in parameters:
        kind = parameter.kind
        if kind is griffe.ParameterKind.keyword_only and not seen_keyword_only:
            parts.append("*")
            seen_keyword_only = True
        prefix = {
            griffe.ParameterKind.var_positional: "*",
            griffe.ParameterKind.var_keyword: "**",
        }.get(kind, "")
        if kind is griffe.ParameterKind.var_positional:
            seen_keyword_only = True
        text = prefix + parameter.name
        if parameter.annotation is not None:
            text += f": {_expr(parameter.annotation)}"
        if parameter.default is not None:
            text += f" = {_expr(parameter.default)}" if parameter.annotation is not None else f"={_expr(parameter.default)}"
        parts.append(text)
    returns = f" -> {_expr(function.returns)}" if function.returns is not None else ""
    one_line = f"{function.name}({', '.join(parts)}){returns}"
    if len(one_line) <= 88:
        return one_line
    body = "".join(f"    {part},\n" for part in parts)
    return f"{function.name}(\n{body}){returns}"


def _docstring(obj: griffe.Object) -> str:
    if obj.docstring is None:
        return ""
    blocks: list[str] = []
    for section in obj.docstring.parse("google"):
        kind = section.kind.value
        if kind == "text":
            blocks.append(section.value.strip())
        elif kind in {"parameters", "other parameters", "attributes"}:
            rows = [
                f"| `{item.name}` | {_cell(_expr(item.annotation))} | {_cell(item.description)} |"
                for item in section.value
            ]
            title = section.title or kind.capitalize()
            blocks.append(f"**{title}**\n\n| Name | Type | Description |\n| --- | --- | --- |\n" + "\n".join(rows))
        elif kind in {"returns", "yields", "raises", "receives", "warns"}:
            lines = []
            for item in section.value:
                label = _expr(getattr(item, "annotation", None)) or getattr(item, "name", "")
                lines.append(f"- {f'`{label}`: ' if label else ''}{item.description.strip()}")
            blocks.append(f"**{section.title or kind.capitalize()}**\n\n" + "\n".join(lines))
        else:
            value = section.value
            blocks.append(value.strip() if isinstance(value, str) else str(value))
    return "\n\n".join(block for block in blocks if block)


def _cell(text: str) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def _public_members(cls: griffe.Class) -> list[griffe.Object]:
    members = [
        member
        for name, member in cls.members.items()
        if not name.startswith("_") and not member.is_alias and not member.inherited
    ]
    return sorted(members, key=lambda member: member.lineno or 0)


def _render_function(function: griffe.Function, level: int, *, method: bool) -> str:
    heading = "#" * level
    labels = [label for label in sorted(function.labels) if label in {"property", "classmethod", "staticmethod", "abstractmethod", "async"}]
    badge = f" <small>{', '.join(labels)}</small>" if labels else ""
    out = [f"{heading} {function.name}{badge} [#{function.path}]", "", "```python", _signature(function, drop_self=method), "```"]
    text = _docstring(function)
    if text:
        out += ["", text]
    return "\n".join(out)


def _render_class(cls: griffe.Class) -> str:
    bases = ", ".join(_expr(base) for base in cls.bases)
    out = [f"## {cls.name} [#{cls.path}]", "", "```python", f"class {cls.name}({bases})" if bases else f"class {cls.name}", "```"]
    text = _docstring(cls)
    if text:
        out += ["", text]
    members = _public_members(cls)
    attributes = [member for member in members if member.is_attribute]
    if attributes:
        rows = []
        for attribute in attributes:
            default = _expr(attribute.value)
            rows.append(
                f"| `{attribute.name}` | {_cell(_expr(attribute.annotation))} | "
                f"{_cell(f'`{default}`' if default else '')} | {_cell(_docstring(attribute))} |"
            )
        out += ["", "| Attribute | Type | Default | Description |", "| --- | --- | --- | --- |", *rows]
    for member in members:
        if member.is_function:
            out += ["", _render_function(member, 3, method=True)]
        elif member.is_class:
            out += ["", f"### {member.name} [#{member.path}]", "", _docstring(member) or "Nested class."]
    return "\n".join(out)


def render(identifier: str, loader: griffe.GriffeLoader) -> str:
    package = identifier.split(".", 1)[0]
    module = loader.load(package)
    obj = module[identifier.split(".", 1)[1]] if "." in identifier else module
    if obj.is_alias:
        obj = obj.final_target
    if obj.is_class:
        return _render_class(obj)
    if obj.is_function:
        return _render_function(obj, 2, method=False)
    if obj.is_attribute:
        value = _expr(obj.value)
        annotation = _expr(obj.annotation)
        lines = [f"## {obj.name} [#{obj.path}]", "", "```python", f"{obj.name}{': ' + annotation if annotation else ''}{' = ' + value if value else ''}", "```"]
        text = _docstring(obj)
        return "\n".join(lines + (["", text] if text else []))
    text = _docstring(obj)
    return f"## {obj.path} [#{obj.path}]" + (f"\n\n{text}" if text else "")


def main(argv: list[str]) -> int:
    loader = griffe.GriffeLoader(search_paths=[str(ROOT)], allow_inspection=False)
    result = {identifier: render(identifier, loader) for identifier in argv}
    json.dump(result, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
