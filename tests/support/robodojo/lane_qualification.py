"""Test helpers for robodojo lane qualification, shared by several test modules.

They were defined in
``tests/ora/orchestrator/backends/robodojo/test_robodojo_lane_qualification.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import json
import os


def _smoke_dir(workspace, name, *, runtime, layout_schedule=None, passed=True, gpu="0", mtime=1):
    root = workspace / "qualification" / name
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({"runtime": runtime}))
    report = root / "report.json"
    report.write_text(json.dumps({"schema": "ora.robodojo-b2-smoke.v1", "task": runtime["task_variant"],
                                  "gpu": gpu, "passed": passed, "layout_schedule": layout_schedule}))
    os.utime(report, (mtime, mtime))
    return report


def _configured(**changes):
    base = {"task_variant": "stack_bowls", "eval_seed": 0, "layout_id": 0, "layout_file": "f", "layout_sha256": "s",
            "num_envs": 1, "env_config": "arx_x5", "sensor_profile": "rgbd_metric_v1", "physics_dt": 0.004,
            "control_hz": 25, "native_action_limit": 800, "native_actions_per_command": 20,
            "physics_device": "cuda:0", "renderer_gpu": 0, "cuda_visible_devices": "0", "main_camera": "head",
            "policy_cameras": ["head", "left_wrist", "right_wrist"],
            "workspace_bounds_m": {"x": [-0.75, 0.75], "y": [-0.65, 0.55], "z": [0.72, 1.4]},
            "startup_timeout_s": 600, "request_timeout_s": 600, "close_timeout_s": 30}
    return {**base, **changes}
