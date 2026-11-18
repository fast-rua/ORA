"""Retention-hint check: the TypeScript engine, acting only on generic hints, must
produce exactly what Python's own delivery rules produce.

Per record of an ``ORA_PI_HINT_DUMP`` file (written by a Pi-core run):

* tool results: Python's direct delivery (delivery.tool_results trims applied by the
  domain code) == Python reference applier on the hints == TypeScript engine;
* observations: Python reference applier (the real round_static_facts.reduce,
  delivery.dedup hoisting and the envelope trims) == TypeScript engine, for the
  delivered envelope text and every round static-facts slot message.

    python -m tests.support.agent_core.hint_check HINT_DUMP.jsonl
"""
from __future__ import annotations

import collections
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ora.agent.pi.hints import StaticFactsState, apply_hints

ROOT = Path(__file__).resolve().parents[3]


def check(dump: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in dump.read_text().splitlines() if line.strip()]
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node is not on PATH")
    result = subprocess.run([node, str(ROOT / "ora-pi/test/hint_apply.ts"), str(dump)],
                            capture_output=True, text=True, check=True)
    ts = [json.loads(line) for line in result.stdout.splitlines()]
    if len(ts) != len(records):
        raise AssertionError(f"TypeScript answered {len(ts)} of {len(records)} records")
    state = StaticFactsState()
    stats: collections.Counter[str] = collections.Counter()
    saved: collections.Counter[str] = collections.Counter()
    mismatches = []
    for index, (record, theirs) in enumerate(zip(records, ts)):
        if record["kind"] == "observation":
            delivered, slot = apply_hints(record["envelope"], record["hints"], state)
            untrimmed_text = json.dumps(record["envelope"], ensure_ascii=False, sort_keys=True)
        else:
            delivered, slot = apply_hints(record["visible"], record["hints"], StaticFactsState())
            untrimmed_text = json.dumps(record["visible"], ensure_ascii=False, sort_keys=True)
            if json.dumps(delivered, ensure_ascii=False, sort_keys=True) != record["python_delivered_text"]:
                mismatches.append((index, "python hints != python direct delivery"))
        text = json.dumps(delivered, ensure_ascii=False, sort_keys=True)
        slot_text = None if slot is None else slot["content"][0]["text"]
        if text != theirs["delivered_text"]:
            mismatches.append((index, f"{record['kind']}: TypeScript delivered text differs"))
        if slot_text != theirs["slot_text"]:
            mismatches.append((index, f"{record['kind']}: TypeScript static-facts slot differs"))
        stats[record["kind"]] += 1
        for hint in record["hints"]:
            stats[f"{record['kind']}:{hint['op']}" + (f":{hint['rule']}" if hint['op'] == 'omit' else '')] += 1
        stats[f"{record['kind']}:slots"] += slot is not None
        saved[record["kind"]] += len(untrimmed_text.encode()) - len(text.encode())
    return {"records": len(records), "mismatches": mismatches[:20], "mismatch_count": len(mismatches),
            "hinted_records": sum(1 for record in records if record["hints"]),
            "hint_counts": dict(sorted(stats.items())), "bytes_saved_by_hints": dict(saved)}


def main() -> None:
    report = check(Path(sys.argv[1]))
    print(json.dumps(report, indent=1))
    sys.exit(1 if report["mismatch_count"] else 0)


if __name__ == "__main__":
    main()
