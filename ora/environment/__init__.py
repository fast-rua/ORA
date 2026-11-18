"""Environment extension contracts and registered session construction.

A lazy façade: importing ``ora.environment`` loads nothing but this module, so
a caller that only wants one contract (``viewer`` reading the capture schemas,
a backend worker importing its own session) does not pay for the whole layer.
Every name in ``__all__`` resolves on first use to its owning module.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ActionValidationError": ("ora.contracts.action", "ActionValidationError"),
    "MOTION_ACTION_SCHEMA": ("ora.contracts.action", "MOTION_ACTION_SCHEMA"),
    "MOTION_ACTION_V3_SCHEMA": ("ora.contracts.action", "MOTION_ACTION_V3_SCHEMA"),
    "RUNTIME_ACTION_SCHEMA": ("ora.contracts.action", "RUNTIME_ACTION_SCHEMA"),
    "TARGET_STATE_ACTION_SCHEMA": ("ora.contracts.action", "TARGET_STATE_ACTION_SCHEMA"),
    "target_state_phases": ("ora.contracts.action", "target_state_phases"),
    "validate_runtime_action": ("ora.contracts.action", "validate_runtime_action"),
    "ENVIRONMENT_SESSION_SCHEMA": (".base", "ENVIRONMENT_SESSION_SCHEMA"),
    "ENVIRONMENT_STATUS_SCHEMA": (".base", "ENVIRONMENT_STATUS_SCHEMA"),
    "EnvironmentContractError": (".base", "EnvironmentContractError"),
    "RestartableEnvironmentSession": (".base", "RestartableEnvironmentSession"),
    "validate_restartable_environment_session": (".base", "validate_restartable_environment_session"),
    "validate_environment_session": (".base", "validate_environment_session"),
    "validate_environment_status": (".base", "validate_environment_status"),
    "EnvironmentSession": (".contracts", "EnvironmentSession"),
    "ENVIRONMENT_DESCRIPTOR_SCHEMA": (".definitions", "ENVIRONMENT_DESCRIPTOR_SCHEMA"),
    "ENVIRONMENT_DESCRIPTOR_V2_SCHEMA": (".definitions", "ENVIRONMENT_DESCRIPTOR_V2_SCHEMA"),
    "EnvironmentDefinition": (".definitions", "EnvironmentDefinition"),
    "EnvironmentDefinitionError": (".definitions", "EnvironmentDefinitionError"),
    "EnvironmentRuntimeDescriptor": (".definitions", "EnvironmentRuntimeDescriptor"),
    "environment_definition": (".definitions", "environment_definition"),
    "environment_definitions": (".definitions", "environment_definitions"),
    "resolve_runtime_descriptor": (".definitions", "resolve_runtime_descriptor"),
    "validate_runtime_extension": (".definitions", "validate_runtime_extension"),
    "EnvironmentFactoryError": (".factory", "EnvironmentFactoryError"),
    "build_environment_session": (".factory", "build_environment_session"),
    "EnvironmentRpcError": (".rpc", "EnvironmentRpcError"),
    "LocalRpcEnvironmentSession": (".rpc", "LocalRpcEnvironmentSession"),
    "RPC_SCHEMA": (".rpc", "RPC_SCHEMA"),
    "DescribedEnvironmentSession": (".session", "DescribedEnvironmentSession"),
    "SyntheticSession": (".synthetic", "SyntheticSession"),
    "synthetic_public_environment_context": (".synthetic", "synthetic_public_environment_context"),
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
