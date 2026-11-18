"""``ora spec``: tasks and RunSpecs of one environment, from the RunSpec composer its backend declares.

    ora spec robotwin {tasks,spec,compose} ...        RoboTwin (orchestrator.spec.benchmark)
    ora spec libero {tasks,spec,compose} ...          a LIBERO base or LIBERO-Pro core suite cell, by its suite
    ora spec robodojo {tasks,spec,compose} ...        RoboDojo (backends.robodojo.spec)
    ora spec robocasa {language,compose} ...          RoboCasa365 (backends.robocasa.spec)
    ora spec robolab {tasks,compose} ...              RoboLab (backends.robolab.spec, task_catalog)

The backend's ``spec_tools`` (``orchestrator.backends.registry``) name each
tool; the rest of the line goes to its own parser
(``ora spec robotwin compose --help``). ``ora spec ENVIRONMENT`` lists them.
"""

from __future__ import annotations

import argparse
from typing import Any

from .passthrough import add_tool, backend_tool

TOOL = "ora.app.spec:main"


def register(sub: Any) -> None:
    add_tool(sub, "spec", TOOL, "compose tasks and RunSpecs per environment", __doc__)


def main(argv: list[str]) -> int:
    return backend_tool("spec", "spec_tools", argv, doc=__doc__)


def run(args: argparse.Namespace) -> int:
    return main(args.tool_args)


__all__ = ["TOOL", "main", "register", "run"]
