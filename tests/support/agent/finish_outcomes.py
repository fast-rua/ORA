"""Test helpers for finish outcomes, shared by several test modules.

They were defined in ``tests/ora/agent/toolkit/host/finish/test_finish_outcomes.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import json


def _runtime(tmp_path, surface, *, task_success=False):
    """The fake Dojo plant's runtime on ``surface``, served the way a run serves it, at its third generation."""
    import tests.support.robodojo.closed_skills_default as parity
    from tests.support.agent.toolkit_run import served
    runtime = parity.tool_runtime(tmp_path, serve=False)
    runtime.surface = surface
    runtime.allowed_tools = tuple(item["name"] for item in surface.definitions)
    (runtime.control_root / "policy-task-check.json").write_text(
        json.dumps({"terminal": False, "task_success": task_success}), encoding="utf-8")
    served(runtime)
    pointer = runtime.control_root / "current-observation.json"
    pointer.write_text(json.dumps({**json.loads(pointer.read_text()), "generation": 3}), encoding="utf-8")
    return runtime


def _log(runtime, *events):
    with runtime.event_path.open("a", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")
