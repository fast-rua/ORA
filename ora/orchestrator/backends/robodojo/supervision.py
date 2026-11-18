"""What the Supervisor takes from a RoboDojo run's preflight, and when the run carries its official outcome."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.orchestrator.faults import SupervisorError
from ora.orchestrator.episode.policy_finish import admitted_abort_round


def supervisor_binding(run_root: Path, runtime_config: Mapping[str, Any]) -> dict[str, Any] | None:
    """The infrastructure qualification preflight admitted, which the worker rechecks at reset."""

    if runtime_config.get("play_scene_set") is not None:
        # A play scene is admitted by its play-scene audit (backends.robodojo.play_preflight), not by
        # an infrastructure qualification the worker would recheck.
        return None
    try:
        record = json.loads((run_root / "preflight.json").read_text(encoding="utf-8"))
        matching = [
            check for check in record["checks"]
            if check.get("check") == "robodojo_infrastructure_qualification"
        ]
        if len(matching) != 1 or matching[0].get("passed") is not True:
            raise ValueError("missing qualification")
        binding = matching[0]["detail"]
        from .qualification_admission import WARNING_SCHEMA
        if binding.get("schema") == WARNING_SCHEMA and binding.get("qualification_available") is False:
            return None
        # Either admitted contract: the combined single record, or the
        # card/task layered view, whose merged projection carries the
        # same source/layout/configuration/device/instruction fields the
        # worker rechecks against its own live readback at reset.
        from ora.environment.benchmarks.robodojo import (
            QUALIFICATION_SCHEMA,
        )
        from ora.orchestrator.robodojo_qualification import LAYERED_SCHEMA
        from ora.orchestrator.robodojo_revalidation import SCHEMA as REVALIDATION_SCHEMA

        if binding.get("schema") not in {QUALIFICATION_SCHEMA, LAYERED_SCHEMA, REVALIDATION_SCHEMA}:
            raise ValueError("invalid qualification schema")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SupervisorError("RoboDojo run lacks its passing infrastructure binding") from exc
    return dict(binding)


def official_complete(
    official: Any, round_results: Sequence[Mapping[str, Any]], *,
    multi_round: bool, round_limit: int,
) -> bool:
    """Whether a RoboDojo run actually carries its official Boolean outcome.

    A round the Supervisor stopped on its own budget or on a policy lifecycle
    request now carries the standing official predicate — false, with the
    verdict disclosing that the simulator never declared a terminal
    transition — exactly as LIBERO reports a truncated episode. Only an
    infrastructure fault, or a backend that produced no verdict at all, keeps
    the Boolean null and the run incomplete. A persistent session keeps its
    aggregate verdict null by contract, so the rule is applied at the round
    boundary; applying it to the aggregate would mark every completed K-round
    RoboDojo session incomplete. A round an abort admitted on recorded
    infrastructure evidence ended (``policy_finish.admitted_abort_round``) has
    no official Boolean by design: it is an invalid sample that ends only
    itself, so it does not make the session incomplete.
    """

    if not multi_round:
        return isinstance(official, bool)
    return len(round_results) == round_limit and all(
        isinstance(record.get("official_success"), bool) or admitted_abort_round(record)
        for record in round_results
    )


__all__ = ["official_complete", "supervisor_binding"]
