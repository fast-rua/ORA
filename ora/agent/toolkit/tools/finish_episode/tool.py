"""Episode exit policy over public evidence and gated same-boundary judgment."""

import re
from ...interface.api import DataInterface
from ..errors import ToolRefusal
from ...interface.types import NotRecorded
from ...lib.values import PublicResult
from ._arguments import (
    _check_arguments,
    _check_category,
    reason_problem,
    _refuse_reason,
)
from . import _released
from ._verdict import refuse_latched, refuse_unconfirmed


def run(api: DataInterface, args: dict) -> dict:
    """Replay: compare a recorded exit request; never end an online episode.

    Official judgment comes only from check_task, when that capability is
    enabled. Public budget and infrastructure evidence are frozen before call.
    Missing simulator evidence fails; a different exit request is Divergence.
    """
    config = api.artifacts.tool_context()["lifecycle"]
    checked = None
    if config["faithful_check_task"] and not config["official_verdict_finish"]:
        data = PublicResult.of(api.state.check_task()).data.json()
        if "verdict" not in data:
            raise NotRecorded("未记录: boundary verdict metadata required for finish")
        checked = data["verdict"]
        refuse_latched(checked, args)
    if config.get("released_arguments") is not None:  # a surface with no outcome contract
        return _released.run(api, args, config, checked)
    allowed = tuple(config["outcomes"])
    outcome = _check_arguments(args, allowed)
    _check_category(outcome, args.get("category"), config.get("categories"))
    problem = reason_problem(outcome, args.get("reason"))
    if problem is not None:
        raise _refuse_reason(
            outcome,
            args.get("reason"),
            problem,
            without_check_task=config["official_verdict_finish"],
        )
    signals = config["infrastructure_signals"]
    if outcome == "abort" and not signals:
        raise ToolRefusal(
            "abort is admitted only after the Harness records an infrastructure failure in this round, and none "
            "is recorded so far; the record is re-read on every call, so the same abort is admitted once one is "
            "recorded in this round",
            code="abort_without_infrastructure_evidence",
            retryable=True,
            field="outcome",
            received="abort",
            constraints={
                "allowed": list(allowed),
                "abort_requires": config["abort_requires"],
                "counted": config["counted_classes"],
                "recorded_infrastructure_signals": signals,
                "remaining_budget": config["remaining_budget"],
            },
        )
    if config["action_pending"]:
        raise ToolRefusal(
            "an action proposal is pending; wait for the fresh observation before finishing",
            code="action_pending_fresh_observation",
            retryable=False,
        )
    if checked is not None:
        refuse_unconfirmed(checked, outcome)
    budget = config["remaining_budget"]
    remaining = budget.get("robot_decisions")
    record = dict(
        schema="ora.openeta-policy-finish.v1",
        **args,
        runtime_remaining_budget=budget,
        early_exit=(outcome != "success" and type(remaining) is int and remaining > 0),
        early_exit_argument_required=False,
        finish_contract="policy_finish_outcomes_v2",
    )
    record["category"] = args.get("category")
    if outcome == "abort":
        record.update(
            abort_admission="infrastructure_evidence",
            abort_infrastructure_evidence=signals,
        )
    if outcome == "give_up":
        ledger = config["candidates"]
        named = sorted(
            identity
            for identity in ledger["untried_candidate_ids"]
            if re.search(
                rf"(?<![A-Za-z0-9_-]){re.escape(identity)}(?![A-Za-z0-9_-])",
                args["reason"],
            )
        )
        ledger.update(
            named_untried_candidate_ids=named, reason_names_untried=bool(named)
        )
        record["untried_candidates"] = ledger
    return PublicResult.of(api.episode.finish_episode(request=record)).data.json()
