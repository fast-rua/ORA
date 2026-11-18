"""Feed recorded Chat Completions events through Python's complete() parse phase.

Usage: python tests/support/agent_core/provider/accumulate.py IN.json OUT.json

IN.json: {"cases": [{"id", "request", "raw_events", "api_key"?, "reasoning_field"?,
"model_call_id"?, "recovery_policy"?}]}. OUT.json: per case the canonical
response (timing normalized) or the error, plus attempt-failure records.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import canonical_text, python_complete  # noqa: E402


def main() -> None:
    cases = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["cases"]
    results = []
    for case in cases:
        options = {key: case[key] for key in ("api_key", "reasoning_field", "model_call_id", "recovery_policy")
                   if key in case}
        outcome = python_complete(case["request"], case["raw_events"], **options)
        outcome["id"] = case["id"]
        outcome["records"] = [canonical_text(record) for record in outcome["records"]]
        results.append(outcome)
    Path(sys.argv[2]).write_text(json.dumps({"results": results}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cases": len(results), "errors": sum("error" in r for r in results)}))


if __name__ == "__main__":
    main()
