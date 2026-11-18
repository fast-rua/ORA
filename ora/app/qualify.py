"""``ora qualify``: environment, lane and seed qualification, from the tools each backend declares.

    ora qualify robotwin {configs,prepare,qualify,merge,seal} ...   expert seed scans and their seed manifests
    ora qualify robodojo {card,task,smoke,revalidate} TASK ...     one lane for the composed RunSpecs
    ora qualify robodojo {physics-probe,motion-replay,motion-metrics} ...
    ora qualify libero motion --run RUN --output RECORD ...       scripted sparse-motion record
    ora qualify robocasa motion --output RECORD ...               scripted sparse-motion record
    ora qualify robolab motion --gpu N --output RECORD ...        scripted sparse-motion record
    ora qualify robolab contract-witness --task T --gpu N ...     seed-0 task-contract witnesses

The backend's ``qualify_tools`` (``orchestrator.backends.registry``) name each
tool; the rest of the line goes to its own parser
(``ora qualify robotwin prepare --help``). ``ora qualify ENVIRONMENT`` lists
them. Free-play scene audits are ``ora play audit``.
"""

from __future__ import annotations

import argparse
from typing import Any

from .passthrough import add_tool, backend_tool

TOOL = "ora.app.qualify:main"


def register(sub: Any) -> None:
    add_tool(sub, "qualify", TOOL, "environment, lane and seed qualification", __doc__)


def main(argv: list[str]) -> int:
    return backend_tool("qualify", "qualify_tools", argv, doc=__doc__)


def run(args: argparse.Namespace) -> int:
    return main(args.tool_args)


__all__ = ["TOOL", "main", "register", "run"]
