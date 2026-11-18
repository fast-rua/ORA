"""``ora services``: operate the resident SAM3 and Contact-GraspNet services from the shared workspace.

    ora services start|status|stop|qualify [--state DIR]

`start` reads the `ORA_SAM_*` / `ORA_GRASP_*` selectors and the `ORA_SAM_PORT` /
`ORA_GRASP_PORT` ports from the environment, `status` reports readiness,
`stop` refuses while Runtime leases exist (and exits 1 naming any service it
left running because it could not prove that process's identity), and `qualify` writes the
hash-named readiness report a launch binds. The host layer supplies the
workspace and Runtime state roots; the service module resolves no paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def state_directory(explicit: Path | None) -> Path:
    """The `model-services` workspace under the configured output storage."""
    if explicit is not None:
        return explicit
    from ora.orchestrator import paths

    return paths.experiment_workspace("model-services")


def register(sub: Any) -> None:
    services = sub.add_parser("services", help="start, check, stop or qualify the resident perception services",
                              description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    services.add_argument("action", choices=("start", "status", "stop", "qualify"))
    services.add_argument("--state", type=Path,
                          help="service state and log directory (default: model-services under the configured "
                               "output storage; services started by the retired scripts/b2 used b2-model-services)")


def run(args: argparse.Namespace) -> int:
    from ora.agent.model_services import operator
    from ora.orchestrator import paths

    state = state_directory(args.state)
    if args.action == "qualify":
        print(operator.qualify(state / "qualification"))
        return 0
    result = operator.lifecycle(args.action, root=paths.PROJECT_ROOT, state=state,
                                runtime_state_root=paths.STATE_ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.action == "status":
        return 0 if all(item.get("ready") is True for item in result.values()) else 1
    if args.action == "stop":
        # A service whose identity this process could not prove was left running.
        return 1 if any("refused" in item for item in result.values()) else 0
    return 0


__all__ = ["register", "run", "state_directory"]
