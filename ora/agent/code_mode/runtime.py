"""The episode's ToolRuntime with the code tools added (plan ``policy.code_mode``).

The wrapper advertises them after the base runtime's direct tools (as the plan's derived
layer does, so the Supervisor's definitions check still matches), validates ``bash``'s
arguments and classifies its call as world-changing, so it is the only call of
its reply. It never runs a command: the Pi policy session does
(``agent/pi/code_mode.py``), because only it records the programs' calls in the
agent trace; a call that reaches the runtime is refused.

``definitions`` and ``allowed_tools`` are what the model sees (``coding.advertised``)
and ``script_definitions`` the base runtime's whole list, which programs call and the
robot socket lists.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from ora.agent.error_classes import ToolError

from . import coding


def validate_command(arguments: Any, contract: Mapping[str, Any]) -> str:
    """``bash``'s command, or the argument error a refused call reports."""

    command = arguments.get("command") if isinstance(arguments, Mapping) else None
    if not isinstance(arguments, Mapping) or set(arguments) - {"command", "timeout"}:
        raise ToolError("bash takes command and timeout", code="invalid_arguments", retryable=True, field="command")
    if not isinstance(command, str) or not command.strip():
        raise ToolError("command must be a non-empty string", code="invalid_arguments", retryable=True,
                        field="command")
    if len(command) > int(contract["max_script_chars"]):
        raise ToolError(f"command is longer than {contract['max_script_chars']} characters", code="invalid_arguments",
                        retryable=True, field="command")
    return command


#: What is left of a script's allowance, in every script result of a ``library: extended`` contract.
ALLOWANCE_FIELD = "allowance_left"
ALLOWANCE_SCHEMA: dict[str, Any] = {"type": "object", "properties": {
    "actions": {"type": "integer", "minimum": 0}, "calls": {"type": "integer", "minimum": 0},
    "queries": {"type": "object", "properties": {
        "count": {"type": "integer", "minimum": 0}, "seconds": {"type": "number", "minimum": 0}}}}}
#: What one entry of a script result's ``calls`` says of a call in a ``library: extended`` contract: a failed call of a
#: tool that has an explanation carries it (``explain``), and a failed action its summary line (``summary``), so a
#: model that never asks for them still sees them.
CALLS_SCHEMA: dict[str, Any] = {"type": "array", "items": {"type": "object", "properties": {
    "call": {"type": "integer"}, "tool": {"type": "string"}, "ok": {"type": "boolean"}, "explain": {"type": "string"},
    "summary": {"type": "string"}, "dry_run": {"type": "boolean"}}}}


def extra_result_schemas(contract: Mapping[str, Any]) -> dict[str, Any]:
    """The result schemas code mode adds to the surface's; with ``library: extended`` ``bash``'s result
    declares ``allowance_left`` (the causal projection keeps only declared fields) and what a ``calls`` entry may
    say, including the ``explain`` of a failed call."""

    schemas = coding.result_schemas()
    if contract.get("library") == "extended":
        schemas[coding.BASH_TOOL] = {**schemas[coding.BASH_TOOL], "properties": {
            **schemas[coding.BASH_TOOL]["properties"], ALLOWANCE_FIELD: copy.deepcopy(ALLOWANCE_SCHEMA),
            "calls": copy.deepcopy(CALLS_SCHEMA)}}
    return schemas


#: The tools the policy session runs itself (never the base runtime).
SESSION_TOOLS = coding.CODING_TOOLS


@dataclass
class _CodeModeSurface:
    base: Any
    extra: Mapping[str, Any]

    @property
    def result_schemas(self) -> dict[str, Any]:
        return {**self.base.result_schemas, **copy.deepcopy(dict(self.extra))}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


@dataclass
class CodeModeToolRuntime:
    base: Any
    contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        lifecycle = tuple(getattr(self.base.surface, "lifecycle_tools", ()) or ())
        self._definitions = coding.advertised(self.base.definitions(), self.contract, lifecycle,
                                              getattr(self.base.surface, "result_schemas", None))
        self._surface = _CodeModeSurface(self.base.surface, extra_result_schemas(self.contract))

    @property
    def surface(self) -> Any:
        return self._surface

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        listed = [definition["name"] for definition in self._definitions]
        return (tuple(name for name in self.base.allowed_tools if name in listed)
                + tuple(name for name in listed if name in SESSION_TOOLS))

    def definitions(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._definitions)

    def script_definitions(self) -> list[dict[str, Any]]:
        """Every tool a program may name: the base runtime's list."""

        return list(self.base.definitions())

    @property
    def script_allowed_tools(self) -> tuple[str, ...]:
        return tuple(self.base.allowed_tools)

    def classify_call_effect(self, name: str, arguments: Mapping[str, Any]) -> str:
        if name == coding.BASH_TOOL:
            validate_command(arguments, self.contract)
            return "mutation"
        if name in coding.FILE_TOOLS:
            return "read_only"  # the workspace is not the world
        return str(self.base.classify_call_effect(name, arguments))

    def invoke_with_context(self, name: str, arguments: Mapping[str, Any], *,
                            call_context: Mapping[str, Any]) -> Mapping[str, Any]:
        if name in SESSION_TOOLS:
            raise ToolError(f"{name} runs in the policy session", code="code_mode_unavailable", retryable=False)
        invoke = getattr(self.base, "invoke_with_context", None)
        if callable(invoke):
            return invoke(name, arguments, call_context=call_context)
        return self.base.invoke(name, arguments)

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if name in SESSION_TOOLS:
            raise ToolError(f"{name} runs in the policy session", code="code_mode_unavailable", retryable=False)
        return self.base.invoke(name, arguments)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def build_code_mode_runtime(plan: Mapping[str, Any], *, base_runtime: Any) -> Any:
    """The runtime with the code tools when the plan's policy carries a code-mode contract."""

    contract = (plan.get("policy") or {}).get("code_mode")
    return CodeModeToolRuntime(base=base_runtime, contract=contract) if contract else base_runtime


__all__ = ["CodeModeToolRuntime", "build_code_mode_runtime", "validate_command"]
