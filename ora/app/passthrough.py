"""Commands that hand the rest of their line to an operator tool's own parser.

``ora report``, ``ora suite``, ``ora dev``, ``ora spec``,
``ora qualify`` and ``ora play audit`` front tools that keep their own argument
parser. Their leaf in ``ora``'s parser is a ``Parser`` built with
``tool="module:function"``: it parses nothing and passes every argument after
the leaf's name, ``--help`` included, to ``function(argv)``, imported only when
the command runs.

``ora spec`` and ``ora qualify`` name a backend first: ``ora spec robotwin
compose ...`` runs the ``compose`` tool the RoboTwin backend declares
(``registry.Tool``) and ``ora play audit --env robotwin ...`` its play-scene
audit, so this package names no backend module (``backend_tool``).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any, Mapping, Sequence


class Parser(argparse.ArgumentParser):
    """``ora``'s parser class; with ``tool`` it is a leaf that passes its arguments through unparsed."""

    def __init__(self, *args: Any, tool: str | None = None, **kwargs: Any) -> None:
        self.tool = tool
        super().__init__(*args, **kwargs)

    def parse_known_args(self, args: Sequence[str] | None = None,
                         namespace: argparse.Namespace | None = None) -> tuple[argparse.Namespace, list[str]]:
        if self.tool is None:
            return super().parse_known_args(args, namespace)
        namespace = argparse.Namespace() if namespace is None else namespace
        namespace.tool, namespace.tool_prog = self.tool, self.prog
        namespace.tool_args = list(sys.argv[1:] if args is None else args)
        return namespace, []


def add_tool(subparsers: Any, name: str, tool: str, help_text: str, description: str | None = None) -> Parser:
    """A leaf ``name`` under ``subparsers`` that runs ``tool`` (``"module:function"``) on its arguments."""

    return subparsers.add_parser(name, tool=tool, help=help_text, description=description or help_text,
                                 formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)


def call(target: str, argv: Sequence[str], prog: str) -> int:
    """``function(argv)`` of ``target`` (``"module:function"``); its usage lines name ``prog``."""

    module, _, function = target.partition(":")
    entry = getattr(importlib.import_module(module), function)
    program, sys.argv[0] = sys.argv[0], prog
    try:
        result = entry(list(argv))
    finally:
        sys.argv[0] = program
    return int(result or 0)


def run_tool(args: argparse.Namespace) -> int:
    """Run the leaf's tool on its arguments; the tool's usage lines name the ``ora`` command."""

    return call(args.tool, args.tool_args, args.tool_prog)


def _help(argv: Sequence[str]) -> bool:
    return bool(argv) and argv[0] in ("-h", "--help")


def _environments(field: str) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """Every name ``ora spec`` / ``ora qualify`` accepts with tools in ``field``: kind -> (shown name, tools)."""

    from ora.orchestrator.backends import registry

    names = {}
    for item in registry.backends():
        tools = getattr(item, field)
        if tools:
            aliases = [alias for alias, kind in registry.ALIASES.items() if kind == item.kind]
            names[item.kind] = (" | ".join([*aliases, item.kind]), tools)
    return names


def backend_tool(command: str, field: str, argv: Sequence[str], *, doc: str) -> int:
    """``ora COMMAND KIND NAME ...``: the backend tool NAME that ``registry`` field ``field`` of KIND declares.

    ``ora COMMAND`` and ``ora COMMAND KIND`` without a tool print what there is
    (on stdout and exit 0 with ``--help``, else on stderr and exit 2).
    """

    from ora.orchestrator.backends import registry

    def listing(lines: list[str], status: int) -> int:
        print("\n".join(lines), file=sys.stdout if status == 0 else sys.stderr)
        return status

    argv = list(argv)
    if not argv or _help(argv):
        lines = [f"usage: ora {command} ENVIRONMENT TOOL [ARGUMENTS ...]", "", doc.strip(), "",
                 "environments and their tools:"]
        lines += [f"  {shown:<22}  {', '.join(tools)}" for shown, tools in _environments(field).values()]
        return listing(lines, 0 if argv else 2)
    kind, *rest = argv
    try:
        tools = getattr(registry.backend(kind), field)
    except registry.UnknownBackend:
        known = ", ".join(shown for shown, _ in _environments(field).values())
        return listing([f"ora {command}: unknown environment {kind!r} (choose from {known})"], 2)
    if not tools:
        return listing([f"ora {command}: {kind} has no {command} tools"], 2)
    if not rest or _help(rest) or rest[0] not in tools:
        width = max(len(name) for name in tools)
        lines = [f"usage: ora {command} {kind} TOOL [ARGUMENTS ...]", "", "tools:"]
        lines += [f"  {name:<{width}}  {tool.help}" for name, tool in tools.items()]
        lines += ["", f"ora {command} {kind} TOOL --help shows one tool's arguments."]
        if rest and not _help(rest):
            lines.insert(0, f"ora {command}: {kind} has no tool {rest[0]!r}")
        return listing(lines, 0 if _help(rest) else 2)
    name, *rest = rest
    tool = tools[name]
    # A tool's own subcommand adds its name to the usage line; a whole-parser tool needs it in the prog.
    return call(tool.target, [*tool.arguments, *rest], f"ora {command} {kind}" + ("" if tool.arguments else f" {name}"))


def play_audit(argv: Sequence[str], *, environment: str | None, doc: str) -> int:
    """``ora play audit --env KIND ...``: the backend's play-scene audit (``registry`` ``play_audit``)."""

    from ora.orchestrator.backends import registry

    parser = argparse.ArgumentParser(prog="ora play audit", add_help=False, allow_abbrev=False)
    parser.add_argument("--env", dest="environment")
    known, rest = parser.parse_known_args(list(argv))
    kind = known.environment or environment
    chosen = registry.find(kind) if kind else None
    if chosen is None or chosen.play_audit is None:
        audited = ", ".join(item.kind for item in registry.backends() if item.play_audit is not None)
        lines = ["usage: ora play audit --env ENVIRONMENT [ARGUMENTS ...]", "", doc.strip(), "",
                 f"environments: {audited}"]
        if kind is not None:
            lines.insert(0, f"ora play audit: {kind} has no play-scene audit")
        status = 0 if kind is None and _help(rest) else 2
        print("\n".join(lines), file=sys.stdout if status == 0 else sys.stderr)
        return status
    return call(f"{chosen.play_audit.__name__}:main", rest, f"ora play audit --env {chosen.kind}")


__all__ = ["Parser", "add_tool", "backend_tool", "call", "play_audit", "run_tool"]
