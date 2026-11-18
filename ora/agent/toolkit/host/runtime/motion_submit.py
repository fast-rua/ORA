"""Mutation availability for the motion tools.

``_ensure_mutation_available`` refuses a state-changing tool after the
trusted environment's terminal boundary. The toolkit serves the motion
tools (``dispatch.install``).

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.
"""

from __future__ import annotations

import json


from ora.agent.error_classes import ToolError


class MotionSubmissionRuntimeMixin:
    """Mutation availability for the motion tools."""

    def _ensure_mutation_available(self, tool: str) -> None:
        task_check_path = self.control_root / "policy-task-check.json"
        if not task_check_path.is_file():
            return
        task_check = json.loads(task_check_path.read_text(encoding="utf-8"))
        if task_check.get("terminal") is True:
            raise ToolError(
                f"{tool} is unavailable after the trusted environment terminal boundary",
                code="episode_terminal", retryable=False,
            )
