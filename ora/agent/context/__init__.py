"""Lazy façade for typed context sources and provider projections."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    # Produced by the policy layers, so they live in composition below context.
    "ContextBundle": ("ora.agent.composition.prompt.slots", "ContextBundle"),
    "ContextSlot": ("ora.agent.composition.prompt.slots", "ContextSlot"),
    "ContextSource": ("ora.agent.composition.prompt.slots", "ContextSource"),
    "causal_provider_value": (".causal_value", "causal_provider_value"),
    "episode_index_keyframe": (".causal_value", "episode_index_keyframe"),
    "ContextError": (".observation.projector", "ContextError"),
    "ContextPolicy": (".observation.projector", "ContextPolicy"),
    "ContextProjector": (".observation.projector", "ContextProjector"),
    "CurrentObservationContextPolicy": (".observation.projector", "CurrentObservationContextPolicy"),
    "build_context_policy": (".observation.projector", "build_context_policy"),
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
