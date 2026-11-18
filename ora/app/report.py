"""``ora report``: the read-only stop-reason and cost report of runs (``orchestrator.reports.run_report``).

    ora report RUN [RUN ...] [--json FILE] [--markdown FILE]
    ora report --namespace NS --cohort ID [--cohort ID ...]

The rest of the line goes to the report's own parser (``ora report --help``).
"""

from __future__ import annotations

import argparse
from typing import Any

from .passthrough import run_tool

TOOL = "ora.orchestrator.reports.run_report:main"


def register(sub: Any) -> None:
    sub.add_parser("report", tool=TOOL, add_help=False, help="stop-reason and cost report of runs (read-only)",
                   description=__doc__)


def run(args: argparse.Namespace) -> int:
    return run_tool(args)


__all__ = ["TOOL", "register", "run"]
