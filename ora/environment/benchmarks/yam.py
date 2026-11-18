"""Host-safe identity of the bimanual I2RT YAM station: tasks, rig modes and source pins.

A real cell has no task roster and no evaluator. ``operator_task`` takes its instruction
from the RunSpec and records a null official verdict; the operator judges the outcome.
``sim_pick_place`` is the fixed task of the simulated cell, used by qualification and
tests. Rig modes: ``sim`` (the station model), ``read`` (the real cameras and encoders,
motors never enabled, every motion answered with its plan) and ``on`` (the real arms move
after the operator approves each motion).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from ora.environment.controllers.yam_station import CONTROL_FREQ_HZ

I2RT_REVISION = "fe78bae67334b9af0b79833dd8ba141a61767f2a"   # v1.4.0
I2RT_PACKAGE = "i2rt"
STATION_MODEL_SHA256 = "9d398a8bcc65269b63d0f589b8dc939c81a9228b692f0b4f3cb90801e4731a45"
EMBODIMENT_ID = "yam_station_linear_4310_d405"
ROBOT_MODEL = "I2RT YAM v1 x2, linear_4310 grippers, wrist RealSense D405"
CONTROLLER = "YAM_IK_JOINT_SCHEDULE"
PUBLIC_CAMERAS = ("overhead", "front", "wrist_left", "wrist_right")
MAIN_CAMERA = "front"
RIG_MODES = ("sim", "read", "on")
ACTION_CODECS = ("motion_action_v3",)
#: ``runtime.motion_timing`` when set: ORA's shared timing worker (TOPP-RA + Ruckig) times the arm paths
#: under the interpreter the variable names. Absent, the station times them itself (the default).
MOTION_TIMING = {"implementation": "toppra_ruckig", "python_env": "ORA_TIMING_PYTHON"}
#: ``runtime.motion_planner`` when set: ORA's shared cuRobo V2 planner plans the arm paths in the public-depth world
#: (``backends/yam/shared_planner.py``). Absent, the station plans them itself (the default).
MOTION_PLANNER = {"implementation": "curobo_v2", "python_env": "CUROBO_V2_PYTHON", "root_env": "CUROBO_V2_ROOT",
                  "required_revision": "d17b54ce32cba095c0b000c4c58777075d11de0e", "request_timeout_s": 180,
                  "robot": "yam_linear_4310", "world_model": "public_depth_voxel_v1"}
#: The preflight check that starts the planner once; its detail binds the interpreter and source the worker reads.
MOTION_PLANNER_PREFLIGHT_CHECK = "shared_motion_planner"

TASKS: dict[str, dict[str, Any]] = {
    "operator_task": {
        "native_action_limit": CONTROL_FREQ_HZ * 900, "rig_modes": RIG_MODES,
        "instruction": None, "official_success": None,
    },
    "sim_pick_place": {
        "native_action_limit": CONTROL_FREQ_HZ * 300, "rig_modes": ("sim",),
        "instruction": "Put the red block on the table between the two arms",
        "official_success": None,
    },
}


class YamBenchmarkError(ValueError):
    """A YAM runtime field is not admitted."""


def public_context() -> dict[str, Any]:
    return {
        "schema": "ora.public-environment-context.v1",
        "environment_id": "yam",
        "embodiment_id": EMBODIMENT_ID,
        "robot_models": [ROBOT_MODEL],
        "controller": CONTROLLER,
        "public_cameras": list(PUBLIC_CAMERAS),
    }


def task_instruction(task: str, runtime: Mapping[str, Any]) -> str:
    fixed = TASKS[task]["instruction"]
    return fixed if fixed is not None else str(runtime.get("task_instruction") or "")


#: The station's free-play switches a runtime may carry (``ora play --env yam``; their flags, help and layers are
#: ``ora/orchestrator/backends/yam/switches.py``): each key is absent or holds exactly this value.
#: ``reach_query``: an unreachable target's refusal also names the nearest position that solves.
#: ``motion_timing``: the shared timing worker times the arm paths; absent, the station's own timing.
#: ``smooth_arm_commands``: a real cell streams its targets with a velocity feed-forward.
#: ``motion_planner``: the shared planner plans the arm paths; absent, the station's own planning.
RUNTIME_SWITCHES: Mapping[str, Any] = MappingProxyType({
    "reach_query": True, "motion_timing": MOTION_TIMING, "smooth_arm_commands": True, "motion_planner": MOTION_PLANNER,
})


def _runtime_switches(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Each switch as the plan keeps it: a flag False or True; a worker binding None or exactly its value."""

    out: dict[str, Any] = {}
    for key, value in RUNTIME_SWITCHES.items():
        if value is True:
            given = runtime.get(key, False)
            if not isinstance(given, bool):
                raise YamBenchmarkError(f"{key} is a boolean")
            out[key] = given
            continue
        given = runtime.get(key)
        if given is not None and (not isinstance(given, Mapping) or dict(given) != dict(value)):
            raise YamBenchmarkError(f"{key} is absent or exactly {dict(value)}")
        out[key] = None if given is None else dict(value)
    return out


def validate_runtime_extension(runtime: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(runtime, Mapping):
        raise YamBenchmarkError("YAM runtime must be an object")
    task = runtime.get("task")
    if task not in TASKS:
        raise YamBenchmarkError(f"YAM task is not admitted: {task!r} (admitted: {sorted(TASKS)})")
    seed = runtime.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise YamBenchmarkError("YAM seed must be a non-negative integer")
    mode = runtime.get("rig_mode", "sim")
    if mode not in TASKS[task]["rig_modes"]:
        raise YamBenchmarkError(f"YAM task {task!r} does not run on rig_mode {mode!r}")
    codec = runtime.get("action_codec", "motion_action_v3")
    if codec not in ACTION_CODECS:
        raise YamBenchmarkError(f"unsupported YAM action codec: {codec!r}")
    instruction = task_instruction(task, runtime)
    if not instruction.strip():
        raise YamBenchmarkError("operator_task needs a non-empty runtime.task_instruction")
    auto = runtime.get("auto_approve", False)
    if not isinstance(auto, bool) or (auto and mode != "on"):
        raise YamBenchmarkError("auto_approve is a boolean and applies to rig_mode 'on' only")
    # Motions only, through the cell's approval broker; enabling and releasing the arms are still asked.
    auto_motions = runtime.get("auto_approve_motions", False)
    if not isinstance(auto_motions, bool) or (auto_motions and mode != "on"):
        raise YamBenchmarkError("auto_approve_motions is a boolean and applies to rig_mode 'on' only")
    switches = _runtime_switches(runtime)
    # The real arms move only with a person beside the cell, declared in the RunSpec.
    if mode == "on" and runtime.get("operator_present") is not True:
        raise YamBenchmarkError("rig_mode 'on' moves the real arms: set runtime.operator_present to true "
                                "only with a person beside the cell and a hand on the stop")
    if runtime.get("required_revision", I2RT_REVISION) != I2RT_REVISION:
        raise YamBenchmarkError("i2rt source revision differs from the pin")
    return {"task": task, "seed": int(seed), "rig_mode": mode, "action_codec": codec,
            "task_instruction": instruction, "auto_approve": auto, "auto_approve_motions": auto_motions,
            **switches,
            "required_revision": I2RT_REVISION,
            "native_action_limit": TASKS[task]["native_action_limit"]}
