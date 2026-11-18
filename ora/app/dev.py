"""``ora dev``: authoring tools for embodiments, prompt bundles, task contracts and this CLI.

    ora dev embodiment ...          validate and compile embodiment descriptors into cards
    ora dev prompt-bundle ...       create or statically audit one modular prompt bundle
    ora dev task-contracts ...      the offline task-contract review inventory and rendering
    ora dev robolab-tasks ...       RoboLab-120's roster, contracts and review drafts from the pinned source
    ora dev cli-reference ...       docs/usage/cli-reference.md, generated from ora's parser

The rest of the line goes to the tool's own parser (``ora dev prompt-bundle --help``).
The Viewer's own build steps are ``ora-viewer compile previews|videos`` (``viewer/``).
"""

from __future__ import annotations

import argparse
from typing import Any

from .passthrough import add_tool, run_tool

TOOLS = {
    "embodiment": ("ora.agent.composition.embodiment.descriptor:main",
                   "validate and compile policy-safe embodiment descriptors into cards"),
    "prompt-bundle": ("ora.agent.composition.prompt.bundle:main",
                      "create or statically audit one modular robot prompt bundle"),
    "task-contracts": ("ora.agent.composition.task_contract.review:main",
                       "offline task-contract review inventory, private template and public rendering"),
    "robolab-tasks": ("ora.orchestrator.backends.robolab.task_catalog:main",
                      "generate RoboLab-120's roster, completion contracts and review drafts; promote witnessed ones"),
    "cli-reference": ("ora.app.cli_reference:main", "print, write or check docs/usage/cli-reference.md"),
}


def register(sub: Any) -> None:
    dev = sub.add_parser("dev", help="authoring tools (embodiment, prompt bundle, task contract, CLI reference)",
                         description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tools = dev.add_subparsers(dest="dev_tool", required=True, metavar="<tool>")
    for name, (tool, text) in TOOLS.items():
        add_tool(tools, name, tool, text)


def run(args: argparse.Namespace) -> int:
    return run_tool(args)


__all__ = ["TOOLS", "register", "run"]
