"""The policy catalog: ``ora baselines``, ``ora lock`` and ``ora experiments``.

These commands only register here: the default composition and its explanation
are ``orchestrator.spec.baseline_explain``'s, ``policy.lock`` is
``orchestrator.spec.policy_lock``'s and the experiment registry is
``orchestrator.spec.experiment_identity``'s (``experiments`` is a development
view and stays out of ``ora --help``).
"""

from __future__ import annotations

import argparse
from typing import Any

COMMANDS = ("baselines", "lock", "experiments")


def register(sub: Any) -> None:
    baselines = sub.add_parser("baselines", help="inspect the host-independent Native robot default composition")
    baselines.add_argument("--json", action="store_true")
    baselines.add_argument("--explain", action="store_true",
                           help="show how the default resolves and what each tool does")
    baselines.add_argument("--environment", metavar="KIND",
                           help="with --explain, show only this environment instance")
    experiments = sub.add_parser("experiments", description="Inspect experiment families and generation lineage.")
    experiments.add_argument("family", nargs="?")
    experiments.add_argument("--json", action="store_true")
    lock = sub.add_parser("lock", help="write policy.lock, the sha256 of every tracked policy asset")
    lock.add_argument("--check", action="store_true", help="exit 1 and list drift instead of writing")
    lock.add_argument("--strip-pins", action="store_true",
                      help="rewrite composable assets to path-only references, then write policy.lock")


def run(args: argparse.Namespace) -> int:
    from ora.orchestrator.paths import PROJECT_ROOT

    if args.command == "baselines":
        from ora.orchestrator.spec.baseline_explain import command_baselines

        return command_baselines(args, PROJECT_ROOT)
    if args.command == "lock":
        from ora.orchestrator.spec.policy_lock import command_lock

        return command_lock(args, PROJECT_ROOT)
    from ora.orchestrator.spec.experiment_identity import command_experiments

    return command_experiments(args)


__all__ = ["COMMANDS", "register", "run"]
