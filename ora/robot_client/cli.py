"""``ora-robot``: the robot of an ORA free-play run, from a shell (standard library only).

    ora-robot tools [--json]
    ora-robot call TOOL [--json '{...}'] [KEY=VALUE ...] [--save PATH]
    ora-robot observe [--save PATH] [--camera NAME]
    ora-robot status

``call`` sends one tool call through the run's Supervisor and prints its
outcome as JSON; ``KEY=VALUE`` arguments are read as JSON when they parse
(``views='["agentview"]'``, ``width=0.04``) and as text otherwise, and override
``--json``. ``--save`` copies the call's (or the observation's) image to PATH;
several images become PATH-1, PATH-2, ... Exit status: 0 when the call ran,
1 when the tool failed or the Supervisor refused it, 2 when the socket is
unreachable or the request was malformed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from .client import Robot, RobotError, origin_record, pick_image


def _arguments(raw_json: str | None, pairs: Sequence[str]) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    if raw_json:
        decoded = json.loads(raw_json)
        if not isinstance(decoded, dict):
            raise ValueError("--json must be a JSON object")
        arguments.update(decoded)
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise ValueError(f"expected KEY=VALUE, got {pair!r}")
        try:
            arguments[key] = json.loads(value)
        except json.JSONDecodeError:
            arguments[key] = value
    return arguments


def _save(paths: Sequence[str], target: str) -> list[str]:
    if not paths:
        raise RobotError("the call returned no image to save", "no_image")
    destination = Path(target)
    saved = []
    for index, path in enumerate(paths, start=1):
        out = destination if len(paths) == 1 else destination.with_name(f"{destination.stem}-{index}{destination.suffix}")
        shutil.copyfile(path, out)
        saved.append(str(out))
    return saved


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="ora-robot", description=__doc__.split("\n\n")[0])
    parser.add_argument("--socket", help="the robot socket (default: $ORA_ROBOT_SOCKET)")
    commands = parser.add_subparsers(dest="command", required=True)
    tools = commands.add_parser("tools", help="list the run's tools")
    tools.add_argument("--json", action="store_true", help="print the full definitions, schemas included")
    call = commands.add_parser("call", help="call one tool through the Supervisor")
    call.add_argument("tool")
    call.add_argument("pairs", nargs="*", metavar="KEY=VALUE")
    call.add_argument("--json", dest="raw_json", metavar="JSON", help="the arguments as one JSON object")
    call.add_argument("--save", metavar="PATH", help="copy the returned image(s) to PATH")
    observe = commands.add_parser("observe", help="the latest published observation")
    observe.add_argument("--save", metavar="PATH", help="copy one camera image to PATH")
    observe.add_argument("--camera", help="the camera --save copies (default: the main camera)")
    commands.add_parser("status", help="whether a turn is active, the observation step, call counts")
    options, extra = parser.parse_known_args(argv)
    if extra:  # KEY=VALUE pairs after --json or --save
        if options.command != "call" or any(item.startswith("-") for item in extra):
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        options.pairs = [*options.pairs, *extra]
    try:
        robot = Robot(options.socket, client="ora-robot", origin=origin_record("ora-robot", ["ora-robot", *argv]))
        with robot:
            if options.command == "tools":
                listed = robot.tools()
                if options.json:
                    _print(listed)
                else:
                    width = max((len(item["name"]) for item in listed), default=0)
                    for item in listed:
                        summary = (item.get("description") or "").strip().splitlines()
                        print(f"{item['name']:<{width}}  {item.get('effect', '?'):<9}  {summary[0] if summary else ''}")
                return 0
            if options.command == "call":
                outcome = robot.call(options.tool, _arguments(options.raw_json, options.pairs))
                if options.save and outcome.get("ok"):
                    outcome["saved"] = _save([item["path"] for item in outcome.get("images") or []], options.save)
                _print(outcome)
                return 0 if outcome.get("ok") else 1
            if options.command == "observe":
                view = robot.observe()
                if options.save:
                    view["saved"] = _save([pick_image(view, options.camera)], options.save)
                _print(view)
                return 0
            _print(robot.status())
            return 0
    except (RobotError, ValueError, OSError) as exc:
        code = getattr(exc, "code", type(exc).__name__)
        print(f"ora-robot: {exc} ({code})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
