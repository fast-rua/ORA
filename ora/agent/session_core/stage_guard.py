"""The Native turn instruction: the mutation and lifecycle tool protocol."""

from __future__ import annotations

from typing import Any


def native_turn_instruction(
    surface: Any, *, legacy_mutation_tool: str = "robot.submit_action",
) -> str:
    mutation_tools = list(getattr(surface, "mutation_tools", (legacy_mutation_tool,)))
    lifecycle_tools = list(getattr(surface, "lifecycle_tools", ()))
    if "toolkit_synchronous_execution_v1" in getattr(surface, "capabilities", ()):
        return (
            "Use read-only tools only when needed. A world-mutating tool "
            f"({', '.join(mutation_tools)}) must be the only tool call in its assistant response. "
            "Actions execute synchronously inside that tool; positive native-action spend ends the "
            "turn after it returns. Read the tool's receipts, then continue from the final published "
            "observation. Zero-spend refusals allow another call in the same turn. "
            "Policy lifecycle tools are: " + ", ".join(lifecycle_tools) + "."
        )
    instruction = (
        "Use read-only tools only when needed. A world-mutating tool "
        f"({', '.join(mutation_tools)}) must be the only tool call in its "
        "assistant response; after a successful action submission, wait for "
        "the receipt and fresh observation."
    )
    if lifecycle_tools:
        instruction += " Policy lifecycle tools are: " + ", ".join(lifecycle_tools) + "."
    return instruction
