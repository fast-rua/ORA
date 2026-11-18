"""Persist recomputation comparisons independently of historical response bytes."""
from .blobs import IO_ROOT
from .types import canonical


def append_comparison(evaluation, comparison: dict) -> None:
    ref = evaluation.store.put(canonical(comparison), media_type="application/json")
    event = dict(schema="ora.toolkit.comparison.v1", reference=ref,
                 source_invocation_id=comparison["historical_invocation_id"],
                 evaluation_invocation_id=evaluation.journal.events[-1]["invocation_id"])
    with (evaluation.store.root / IO_ROOT / "comparisons.jsonl").open("ab") as stream:
        stream.write(canonical(event) + b"\n")
