"""The tool manual is required only by a surface that advertises robot.read_manual."""

from __future__ import annotations

import pytest

from ora.agent.toolkit.host.runtime.base import RobotToolRuntime
from ora.agent.error_classes import ToolError

BOUNDS = {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]}
GRANTS = {"state_keys": ["privileged.task_check"], "tools": ["check_task"]}


def runtime(tmp_path, profile, **surface):
    return RobotToolRuntime(control_root=tmp_path / "control", manual_path=tmp_path / "missing-manual.md",
                            event_path=tmp_path / "events.jsonl", workspace_bounds_m=BOUNDS,
                            tool_surface_profile=profile, **surface)


def test_a_surface_with_read_manual_still_requires_the_manual(tmp_path):
    with pytest.raises(ToolError, match="manual is unavailable"):
        runtime(tmp_path, "robot_runtime_v2")


def test_a_surface_without_read_manual_runs_without_one(tmp_path):
    host = runtime(tmp_path, "robodojo_placement_toolkit_v5", toolkit_grants=GRANTS,
                   toolkit_receipt_delivery="tool_output")
    assert "robot.read_manual" not in {item["name"] for item in host.definitions()}
