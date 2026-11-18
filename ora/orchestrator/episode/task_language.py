"""Bind layout-dependent official language after reset, before the first turn."""

from typing import Any, Mapping

from ..faults import SupervisorError
from ..evaluation.context_manifest import build_context_manifest, context_placeholder
from .records import _append, _atomic_json


def bind_task_instruction(loop: Any, reset: Mapping[str, Any]) -> None:
    """Keep the sealed plan intact; the reset readback is separate run evidence."""
    loop.task_instruction = loop.plan.get("task_instruction", loop.plan["task"])
    runtime = loop.plan["runtime"]
    if runtime.get("kind") != "robodojo" or runtime.get("task_language_source") != "live_reset_readback":
        return
    instruction = reset.get("task_instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise SupervisorError("RoboDojo reset returned no official task instruction")
    loop.task_instruction = instruction
    record = {
        "schema": "ora.task-language-binding.v1", "task_instruction": instruction,
        "source": "live_reset_readback", "gate_verified": False,
        "task": loop.plan["task"], "layout_id": runtime["layout_id"],
        "warning": "Official language read at reset; no qualification gate verified this layout.",
    }
    _atomic_json(loop.native / "task-language-binding.json", record)
    _append(loop.events, {"type": "task_language_binding", "record": record})
    if hasattr(loop, "compiled_policy"):
        bound_plan = {**loop.plan, "task_instruction": instruction}
        manifest = build_context_manifest(
            bound_plan, compiled_policy=loop.compiled_policy,
            context=context_placeholder(bound_plan), run_id=loop.run_id,
            workspace_root=loop.run_root.parents[3],
        )
        manifest["task_instruction_binding"] = {
            "path": "native/task-language-binding.json", "source": "live_reset_readback",
        }
        _atomic_json(loop.run_root / "context-manifest.json", manifest)
