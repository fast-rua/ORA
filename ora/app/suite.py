"""``ora suite``: the official-suite queue (``orchestrator.suite.queue``).

    ora suite run --cells CELLS.json --experiment EXP --cohort ID --group G --lanes PROFILE ...
                  --service-report REPORT --yes        one single-attempt launch per listed cell
    ora suite status|drain --experiment EXP --group G

The rest of the line goes to the queue's own parser (``ora suite --help``).
"""

from __future__ import annotations

import argparse
from typing import Any

from .passthrough import run_tool

TOOL = "ora.orchestrator.suite.queue:main"


def register(sub: Any) -> None:
    sub.add_parser("suite", tool=TOOL, add_help=False, help="the official-suite queue: run, status, drain",
                   description=__doc__)


def run(args: argparse.Namespace) -> int:
    return run_tool(args)


__all__ = ["TOOL", "register", "run"]
