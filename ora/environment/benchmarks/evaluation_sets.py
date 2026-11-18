"""Which episodes official evaluation draws from, for refusing practice on them.

``evaluation-sets-v1.json`` beside this module is the machine-readable list:
per benchmark, the evaluation set, its practice side and the RunPlan field
that tells them apart. :func:`evaluation_set_hit` applies those rules to one
RunPlan; a practice launcher refuses any plan it returns a reason for. Host
safe: no simulator imports.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "ora.evaluation-sets.v1"
MANIFEST = Path(__file__).with_name("evaluation-sets-v1.json")


@lru_cache(maxsize=1)
def evaluation_sets() -> dict[str, Any]:
    """The parsed manifest (schema-checked)."""

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{MANIFEST.name} is not {SCHEMA}")
    return manifest


def evaluation_set_hit(plan: Mapping[str, Any]) -> str | None:
    """Why this RunPlan's episode belongs to an evaluation set, or ``None`` when it is practice-eligible.

    An environment the manifest does not list has no practice side, so it is
    always a hit.
    """

    runtime = plan.get("runtime") or {}
    kind = runtime.get("kind")
    rules = evaluation_sets()["environments"].get(kind)
    if rules is None:
        return f"{kind}: no practice side is defined for this environment"
    match = rules["match"]
    if kind == "libero_pro":
        driver = runtime.get("environment_driver", "direct_libero_v1")
        if driver in match["practice_values"]:
            return None
        return f"libero_pro: driver {driver} loads an official initial state (seed {plan.get('seed')} % trial count)"
    if kind == "robotwin":
        seed = plan.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool) and seed < match["evaluation_minimum"]:
            return None
        return f"robotwin: seed {seed} is in the evaluation range (>= {match['evaluation_minimum']})"
    if kind == "robocasa":
        split = runtime.get("split", match["default"])
        if split not in match["evaluation_values"]:
            return None
        return f"robocasa: split {split} is the evaluation split"
    return f"{kind}: the manifest lists no match rule this module applies"


__all__ = ["MANIFEST", "SCHEMA", "evaluation_set_hit", "evaluation_sets"]
