"""RunSpec -> RunPlan: validation, defaults, composition, resources and plan identity.

``run_spec`` resolves one RunSpec into a content-addressed RunPlan through the
section resolvers beside it (``policy``, ``native_policy``, ``native_provider``,
``runtime``, ``composition``, ``context_compactor``, ``context_pack``,
``operator_control``, ``play``, ``partition``, ``result_tier``, ``source``,
``environment_context``, ``latency``); ``values`` holds the shared value rules and hashes.
``benchmark`` and ``inputs`` compose benchmark cells and fresh RunSpecs,
``defaults`` the default instance, ``quickstart`` the Synthetic quickstart's
RunSpec, ``policy_pins`` and ``policy_lock`` the
pinned policy data, ``experiment_identity`` the experiment registry and
``baseline_explain`` with ``baseline_tool_routes`` ``ora baselines --explain``.

The package exports the documented plan API lazily, so importing one section
module does not load the others; code inside the orchestrator imports the
module that defines a name.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "load_run_spec": (".run_spec", "load_run_spec"),
    "resolve_run_spec": (".run_spec", "resolve_run_spec"),
    "validate_run_plan": (".run_spec", "validate_run_plan"),
    "RunSpecError": (".values", "RunSpecError"),
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
