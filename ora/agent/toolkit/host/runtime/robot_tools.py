"""``_persist_action``, the one write of an action proposal, as a mixin of ``RobotToolRuntime``.

The toolkit serves the ``robot_runtime_v2`` tools (``dispatch.install``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ora.agent.toolkit.interface import dry_run
from ora.agent.error_classes import ToolError


class RobotToolsMixin:
    """``_persist_action``, the one write of an action proposal."""

    def _persist_action(
        self, target: Path, control: dict[str, Any], action: Mapping[str, Any],
    ) -> dict[str, Any]:
        if dry_run.requested():  # the action is compiled and checked; a dry run stops here (``action_preview.py``)
            raise dry_run.Proposed(action)
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise ToolError(
                "an action is already submitted for this observation",
                code="duplicate_action", retryable=False,
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(action, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        control["open"] = False
        temporary = (self.control_root / "current-observation.json").with_suffix(".json.tmp")
        temporary.write_text(json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.control_root / "current-observation.json")
        return {"accepted": True, "directive": "action_submitted", "action_id": action["action_id"]}
