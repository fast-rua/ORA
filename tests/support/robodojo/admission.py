"""Test helpers for robodojo admission, shared by several test modules.

They were defined in
``tests/ora/orchestrator/backends/robodojo/test_robodojo_admission.py``; test modules import
them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from ora.environment.benchmarks import robodojo


def _runtime() -> dict:
    return {
        "kind": "robodojo", "transport": "local_rpc", "root_env": "ROBODOJO_ROOT",
        "python_env": "ROBODOJO_PYTHON", "asset_root_env": "ROBODOJO_ASSET_ROOT",
        **{field: robodojo.REVISIONS[name] for name, field in robodojo.REVISION_FIELDS.items()},
        "task_variant": "arrange_largest_number", "eval_seed": 0, "layout_id": 0,
        "layout_file": "Eval_Layout/RoboDojo/arx_x5/0/arrange_largest_number_0.json",
        "layout_sha256": "a" * 64, "num_envs": 1, "env_config": "arx_x5",
        "sensor_profile": "rgbd_metric_v1", "physics_dt": 0.004, "control_hz": 25,
        "native_action_limit": 1050, "native_actions_per_command": 20,
        "main_camera": "head", "policy_cameras": list(robodojo.PUBLIC_CAMERAS),
        "required_host": "test-host", "require_gpu": True, "physics_device": "cuda:0",
        "renderer_gpu": 2, "cuda_visible_devices": "2", "startup_timeout_s": 600,
        "request_timeout_s": 180, "close_timeout_s": 30,
        "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
        "environment_signal_policy": "lifecycle_only", "expected_task_language": "Pick the item.",
        "infrastructure_qualification": {
            "path": "outputs/robodojo-compat/qualification/fixture.json", "sha256": "b" * 64,
        },
    }
