"""Lazy public façade for trusted episode orchestration."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "EpisodeCoordinator": (".episode.coordinator", "EpisodeCoordinator"),
    "SupervisorError": (".faults", "SupervisorError"),
    "run_episode": (".episode.coordinator", "run_episode"),
    "HarnessFault": (".faults", "HarnessFault"),
    "HarnessFaultError": (".faults", "HarnessFaultError"),
    "classify_harness_fault": (".faults", "classify_harness_fault"),
    "EpisodeInteractionState": (".episode.interaction", "EpisodeInteractionState"),
    "InteractionStateError": (".episode.interaction", "InteractionStateError"),
    "HarnessLiveEvent": (".episode.live", "HarnessLiveEvent"),
    "LiveInteractionPort": (".episode.live", "LiveInteractionPort"),
    "NullLiveInteractionPort": (".episode.live", "NullLiveInteractionPort"),
    "TerminationAuthority": (".episode.termination", "TerminationAuthority"),
    "TerminationDecision": (".episode.termination", "TerminationDecision"),
    "policy_outcome_decision": (".episode.termination", "policy_outcome_decision"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
