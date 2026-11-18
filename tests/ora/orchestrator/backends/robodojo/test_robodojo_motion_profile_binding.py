"""runtime.motion_profile on a RoboDojo plan: absent (profile v1) or an ARX X5 profile's {path, sha256}."""
from __future__ import annotations

import hashlib

import pytest

from ora.environment.benchmarks import robodojo
from ora.orchestrator.paths import PROJECT_ROOT

V2 = "ora/policy/embodiments/arx-x5/motion-profile-v2.json"


def _runtime(**extra) -> dict:
    task = "stack_blocks"
    return {
        "kind": "robodojo", "transport": "local_rpc", "root_env": "ROBODOJO_ROOT",
        "python_env": "ROBODOJO_PYTHON", "asset_root_env": "ROBODOJO_ASSET_ROOT",
        **{field: robodojo.REVISIONS[name] for name, field in robodojo.REVISION_FIELDS.items()},
        "task_variant": task, "eval_seed": 0, "layout_id": 0,
        "layout_file": f"Eval_Layout/RoboDojo/arx_x5/0/{task}_0.json", "layout_sha256": "a" * 64,
        "num_envs": 1, "env_config": "arx_x5", "sensor_profile": "rgbd_metric_v1", "physics_dt": 0.004,
        "control_hz": 25, "native_action_limit": robodojo.NATIVE_ACTION_LIMITS[task],
        "native_actions_per_command": 20, "main_camera": "head", "policy_cameras": list(robodojo.PUBLIC_CAMERAS),
        "required_host": "test-host", "require_gpu": True, "physics_device": "cuda:0", "renderer_gpu": 4,
        "cuda_visible_devices": "4", "startup_timeout_s": 600, "request_timeout_s": 180, "close_timeout_s": 30,
        "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}, **extra,
    }


def test_a_plan_binds_no_profile_or_an_arx_x5_one():
    robodojo.validate_runtime_extension(_runtime())
    record = {"path": V2, "sha256": hashlib.sha256((PROJECT_ROOT / V2).read_bytes()).hexdigest()}
    robodojo.validate_runtime_extension(_runtime(motion_profile=record))


@pytest.mark.parametrize("binding", [
    V2,  # a path the RunSpec binds is a record by the plan
    {"path": "ora/policy/embodiments/franka-panda/motion-profile-v4.json", "sha256": "a" * 64},
    {"path": V2, "sha256": "not-a-digest"},
    {"path": V2, "sha256": "a" * 64, "extra": 1},
])
def test_anything_else_is_refused(binding):
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="motion_profile"):
        robodojo.validate_runtime_extension(_runtime(motion_profile=binding))
