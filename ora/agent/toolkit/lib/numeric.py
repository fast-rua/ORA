"""Bounded numeric execution over caller-supplied values only."""

from ..interface.api import DataInterface
from .values import JsonValue


def evaluate(
    api: DataInterface, source: str, bindings: JsonValue, *, outputs: list | None = None
) -> JsonValue:
    """Pure computation: numeric recomputation, including in Replay; no simulator or service access.

    Units and frames are those explicitly supplied in bindings. The evaluator
    admits only the published geometry grammar and retains CPU/memory/time limits.
    Example: evaluate(api, "x = np.array([1, 2]) * 2", JsonValue.of({}), outputs=["x"]).
    """
    return JsonValue.of(
        api.compute.evaluate_numeric(source=source, bindings=bindings.json(), outputs=outputs)
    )
