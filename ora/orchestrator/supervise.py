"""The supervisor process of one launched run, started by ``launch._spawn``; never run it by hand.

    python -m ora.orchestrator.supervise RUN_ID CLAIM_TOKEN
"""

from __future__ import annotations

import argparse
import sys

from ora.agent.providers.user_store import apply_stored_credentials

from .launch import supervise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ora.orchestrator.supervise", description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("claim_token")
    args = parser.parse_args(argv)
    apply_stored_credentials()  # the user's stored provider keys, where the environment sets none, as `ora` does
    try:
        return supervise(args.run_id, args.claim_token)
    except (RuntimeError, ValueError, OSError) as exc:  # RunSpecError, EvaluationError and JSON errors among them
        print(f"ora: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
