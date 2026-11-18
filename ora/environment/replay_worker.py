"""Isolated simulator worker: Viewer-derived controller replay videos and replay-program checks."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


JOB_SCHEMA = "ora.viewer-controller-replay-job.v2"
RESULT_SCHEMA = "ora.viewer-controller-replay.v2"
PROGRAM_JOB_SCHEMA = "ora.replay-program-job.v1"
FRAME_SCHEMA = "ora.viewer-controller-replay-frame.v2"
ROBOTWIN_DENSE_TARGET_CONTROLLERS = frozenset({
    "robotwin_dense_joint_targets",
    "robotwin_local_cartesian_dls",
    "robotwin_global_then_local_settle",
    "robotwin_global_precontact_then_local",
})
ROBOTWIN_HOLD_CONTROLLER = "robotwin_hold_current_drive_targets"
ROBOTWIN_STAGED_CONTROLLERS = frozenset({
    "robotwin_global_then_local_settle",
    "robotwin_global_precontact_then_local",
})
# Faults raised before a sample's physics step: a target was (or may have been)
# written without the step that followed it, so the record is not a prefix.
ROBOTWIN_PRE_STEP_FAULTS = frozenset({
    "backend_write_fault", "before_step_fault", "backend_step_fault",
})


class ReplayWorkerError(RuntimeError):
    """A derived simulator replay could not be compiled safely."""


def _uniform_capture_steps(step_count: int, frame_limit: int) -> frozenset[int]:
    """Choose a bounded, endpoint-preserving review sample for one action."""

    if step_count <= 0:
        return frozenset()
    if frame_limit <= 0:
        raise ReplayWorkerError("Viewer replay frame limit must be positive")
    if frame_limit == 1:
        return frozenset({step_count - 1})
    if step_count <= frame_limit:
        return frozenset(range(step_count))
    denominator = frame_limit - 1
    return frozenset(
        (index * (step_count - 1) + denominator // 2) // denominator
        for index in range(frame_limit)
    )


def _capture_steps(step_count: int, sampling: Mapping[str, Any]) -> frozenset[int]:
    """Resolve one closed replay sampling profile to global action-step indices."""

    mode = sampling.get("mode")
    if mode == "action_keyframes_v1":
        frame_limit = sampling.get("max_frames_per_action")
        hold_steps = sampling.get("hold_steps_per_sample")
        if (
            isinstance(frame_limit, bool)
            or not isinstance(frame_limit, int)
            or not 1 <= frame_limit <= 32
        ):
            raise ReplayWorkerError(
                "Viewer replay max_frames_per_action must be in [1, 32]"
            )
        if (
            isinstance(hold_steps, bool)
            or not isinstance(hold_steps, int)
            or not 1 <= hold_steps <= 64
        ):
            raise ReplayWorkerError(
                "Viewer replay hold_steps_per_sample must be in [1, 64]"
            )
        return _uniform_capture_steps(step_count, frame_limit)
    if mode == "physics_stride_v1":
        stride = sampling.get("capture_stride")
        if (
            isinstance(stride, bool)
            or not isinstance(stride, int)
            or not 1 <= stride <= 64
        ):
            raise ReplayWorkerError("Viewer replay capture stride must be in [1, 64]")
        return frozenset(range(0, step_count, stride))
    raise ReplayWorkerError(f"unsupported Viewer replay sampling mode: {mode}")


def _needs_final_capture(step_count: int, last_captured_step: int | None) -> bool:
    """Keep the exact end state when periodic capture did not include it."""

    return step_count > 0 and last_captured_step != step_count - 1


def _robotwin_replay_route(controller: Any) -> str:
    """Resolve one closed recorded-controller label to its replay executor."""

    if isinstance(controller, str) and controller in ROBOTWIN_DENSE_TARGET_CONTROLLERS:
        return "dense_targets"
    if controller == "robotwin_ee_action":
        return "ee_action"
    raise ReplayWorkerError(
        f"unsupported RoboTwin replay controller: {controller}"
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayWorkerError(f"expected JSON object: {path}")
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _required_environment_path(runtime: Mapping[str, Any], field: str) -> Path:
    name = runtime.get(field)
    if not isinstance(name, str) or not name:
        raise ReplayWorkerError(f"Runtime omits {field}")
    value = os.environ.get(name)
    if not value:
        raise ReplayWorkerError(f"required replay environment variable is unset: {name}")
    return Path(value).expanduser().resolve()


def _artifact_arrays(
    manifest_path: Path, entry: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    artifact = entry.get("artifact")
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
        return {}
    run_native_root = manifest_path.parent.parent.resolve()
    target = (run_native_root / str(artifact["path"])).resolve()
    if not target.is_relative_to(run_native_root) or not target.is_file():
        raise ReplayWorkerError("controller replay artifact escaped the run")
    if _hash(target) != artifact.get("sha256"):
        raise ReplayWorkerError("controller replay artifact hash drifted")
    with np.load(target, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _contact_sheet(images: Mapping[str, Any], cameras: list[str]) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps

    if cameras == [
        "observer_camera", "head_camera", "front_camera",
        "left_camera", "right_camera",
    ]:
        canvas = Image.new("RGB", (1280, 960), "#101413")
        layout = {
            "observer_camera": (0, 0, 427, 480),
            "head_camera": (427, 0, 853, 480),
            "front_camera": (853, 0, 1280, 480),
            "left_camera": (0, 480, 640, 960),
            "right_camera": (640, 480, 1280, 960),
        }
    else:
        canvas = Image.new("RGB", (1280, 720), "#101413")
        width = 1280 // max(1, len(cameras))
        layout = {
            name: (index * width, 0, 1280 if index == len(cameras) - 1 else (index + 1) * width, 720)
            for index, name in enumerate(cameras)
        }
    draw = ImageDraw.Draw(canvas)
    for name in cameras:
        left, top, right, bottom = layout[name]
        draw.rectangle((left, top, right - 1, bottom - 1), outline="#50605a")
        value = images.get(name)
        if value is not None:
            array = np.asarray(value)
            if array.ndim == 3 and array.shape[2] >= 3:
                source = Image.fromarray(
                    np.clip(array[..., :3], 0, 255).astype(np.uint8), "RGB",
                )
                rendered = ImageOps.contain(
                    source, (right - left, bottom - top), Image.Resampling.LANCZOS,
                )
                x = left + (right - left - rendered.width) // 2
                y = top + (bottom - top - rendered.height) // 2
                canvas.paste(rendered, (x, y))
        draw.rectangle((left, top, min(right, left + 230), top + 30), fill="#101413")
        draw.text((left + 8, top + 6), name.upper(), font=_font(18, bold=True), fill="#f2f0e8")
    return canvas


class _FrameWriter:
    def __init__(
        self, root: Path, *, cameras: list[str], sampling: Mapping[str, Any],
    ) -> None:
        self.root = root
        self.frames_root = root / "frames"
        self.frames_root.mkdir(parents=True, exist_ok=True)
        self.timeline = root / "timeline.jsonl"
        self.cameras = cameras
        self.sampling = dict(sampling)
        self.sequence = 0

    def append(
        self, images: Mapping[str, Any], *, entry: Mapping[str, Any] | None,
        phase: str, control_step: int | None,
    ) -> None:
        path = self.frames_root / f"frame-{self.sequence:06d}.jpg"
        _contact_sheet(images, self.cameras).save(
            path, format="JPEG", quality=88, subsampling=1,
        )
        record = {
            "schema": FRAME_SCHEMA,
            "frame_index": self.sequence,
            "path": path.relative_to(self.root).as_posix(),
            "action_sequence": (
                entry.get("action_sequence") if entry is not None else None
            ),
            "action_id": entry.get("action_id") if entry is not None else None,
            "action": entry.get("action") if entry is not None else "reset",
            "phase": phase,
            "control_step_index": control_step,
            "sampling": self.sampling,
            "cameras": self.cameras,
            "recorded_execution": (
                dict(entry["execution"])
                if entry is not None and isinstance(entry.get("execution"), Mapping)
                else None
            ),
            "capture_semantics": (
                "derived_controller_replay_action_keyframes"
                if self.sampling.get("mode") == "action_keyframes_v1"
                else "derived_controller_replay_after_simulator_step"
                if self.sampling.get("capture_stride") == 1
                else "derived_controller_replay_temporally_subsampled"
            ),
        }
        with self.timeline.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.sequence += 1


def _robotwin_images(environment: Any) -> dict[str, Any]:
    manager = getattr(environment, "cameras", None)
    update = getattr(manager, "update_picture", None)
    get_rgb = getattr(manager, "get_rgb", None)
    get_observer = getattr(manager, "get_observer_rgb", None)
    if not all(callable(item) for item in (update, get_rgb, get_observer)):
        raise ReplayWorkerError("RoboTwin replay cameras are unavailable")
    update()
    raw = get_rgb()
    if not isinstance(raw, Mapping):
        raise ReplayWorkerError("RoboTwin replay RGB result is malformed")
    result = {
        str(name): value.get("rgb")
        for name, value in raw.items()
        if isinstance(value, Mapping) and value.get("rgb") is not None
    }
    result["observer_camera"] = get_observer()
    return result


def _robotwin_control_sequence(
    entry: Mapping[str, Any], arrays: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = entry.get("metadata")
    structure = (
        metadata.get("control_sequence") if isinstance(metadata, Mapping) else None
    )
    if not isinstance(structure, Mapping):
        raise ReplayWorkerError("RoboTwin replay control structure is unavailable")
    result: dict[str, Any] = {}
    for channel, value in structure.items():
        if value is None:
            result[str(channel)] = None
            continue
        if not isinstance(value, Mapping):
            raise ReplayWorkerError("RoboTwin replay channel is malformed")
        compiled: dict[str, Any] = {}
        for field, item in value.items():
            if isinstance(item, Mapping) and isinstance(item.get("array"), str):
                key = str(item["array"])
                if key not in arrays:
                    raise ReplayWorkerError(f"RoboTwin replay array is missing: {key}")
                compiled[str(field)] = arrays[key]
            else:
                compiled[str(field)] = item
        result[str(channel)] = compiled
    return result


def _recorded_control_step_count(
    entry: Mapping[str, Any], arrays: Mapping[str, Any],
) -> int:
    """Recover the exact issued-step count across current and legacy traces."""

    metadata = entry.get("metadata")
    tracking = (
        metadata.get("controller_tracking")
        if isinstance(metadata, Mapping) else None
    )
    if isinstance(tracking, Mapping):
        value = tracking.get("physics_step_attempts")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    structure = (
        metadata.get("control_sequence")
        if isinstance(metadata, Mapping) else None
    )
    if not isinstance(structure, Mapping):
        return 0
    counts: list[int] = []
    for channel in structure.values():
        if not isinstance(channel, Mapping):
            continue
        num_step = channel.get("num_step")
        if isinstance(num_step, int) and not isinstance(num_step, bool):
            counts.append(max(0, num_step))
        for field in ("position", "result", "command"):
            binding = channel.get(field)
            key = binding.get("array") if isinstance(binding, Mapping) else None
            value = arrays.get(key) if isinstance(key, str) else None
            shape = getattr(value, "shape", ())
            if shape and isinstance(shape[0], int):
                counts.append(int(shape[0]))
    return max(counts, default=0)


@functools.lru_cache(maxsize=1)
def _robotwin_gripper_planner_steps() -> int:
    """The length of RoboTwin's planned gripper ramp, read from its own planner."""
    planner = importlib.import_module("envs.robot.planner")
    for name in ("MplibPlanner", "CuroboPlanner"):
        plan = getattr(getattr(planner, name, None), "plan_grippers", None)
        if callable(plan):
            return int(plan(None, 0.0, 1.0)["num_step"])
    raise ReplayWorkerError("RoboTwin's gripper planner is unavailable")


def _legacy_gripper_steps(
    sequence: Any, planner_steps: Callable[[], int] = _robotwin_gripper_planner_steps,
) -> Any:
    """Restore the per-step limit a legacy RoboTwin trace recorded as zero.

    The original passed its planned ramp's ``per_step`` to ``robot.set_gripper``,
    which moves the drive target by at most that much per step. A trace recorded
    before the recorder kept it says 0.0, so the replay jumped to each value and
    closed the jaws one step early; under contact the arm then drifted from the
    original (up to 1.3 mm at an action end in the runs checked). RoboTwin plans the ramp as
    ``linspace(now, target, n)`` with ``per_step = (target - now) / n``, so the
    recorded uniform increment ``d`` gives ``per_step = d * (n - 1) / n``, also for
    a ramp the action cut short. A moving path that is not such a ramp is refused.
    """
    import numpy as np

    if not isinstance(sequence, Mapping):
        return sequence
    result = dict(sequence)
    for channel in ("left_gripper", "right_gripper"):
        path = sequence.get(channel)
        if not isinstance(path, Mapping) or path.get("per_step"):
            continue
        values = np.asarray(path.get("result", []), dtype=np.float64)
        steps = np.diff(values)
        moving = steps[np.abs(steps) > 1e-12]
        if not moving.size:
            continue
        if not np.allclose(moving, moving[0], rtol=0.0, atol=1e-9):
            raise ReplayWorkerError(f"legacy RoboTwin {channel} path is not a planned ramp")
        n = planner_steps()
        result[channel] = {**path, "per_step": float(moving[0]) * (n - 1) / n}
    return result


def _extra_settle_steps(entry: Mapping[str, Any], arrays: Mapping[str, Any]) -> int:
    """Free physics steps the original ran after its recorded control sequence.

    The entry's ``physics_steps`` is the recorded total and decides. ``execution.settle_steps``
    names the length of the action's closing phase, and a ``global_precontact_then_local``
    action records that phase (its local contact steps) inside the control sequence as well.
    Stepping it again after the sequence left the arm that many steps without a new target,
    1-2 mm away from the original, and a same-seed replay of a successful RoboTwin run could
    end in failure. Only the steps the total has beyond the sequence are settling; a trace
    without a total keeps its recorded settle steps.
    """
    execution = entry.get("execution")
    settle = int(execution.get("settle_steps", 0)) if isinstance(execution, Mapping) else 0
    total = entry.get("physics_steps")
    if isinstance(total, bool) or not isinstance(total, int):
        return max(0, settle)
    return max(0, min(settle, total - _recorded_control_step_count(entry, arrays)))


def _robotwin_hold_steps(entry: Mapping[str, Any]) -> int:
    """Physics steps a RoboTwin ``wait`` ran while holding every drive target.

    A wait writes no target: SAPIEN keeps each joint and gripper drive target it
    was last given and the scene steps. Its record has no control sequence and
    is marked not replayable, but the steps ran, and skipping them starts every
    later action from a different state. Replaying them is the same number of
    plain scene steps. Any other non-replayable entry returns 0.
    """
    if entry.get("controller") != ROBOTWIN_HOLD_CONTROLLER or entry.get("action") != "wait":
        return 0
    steps = entry.get("physics_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ReplayWorkerError("RoboTwin wait entry records no physics step count")
    return steps


def _robotwin_fault_prefix_steps(entry: Mapping[str, Any], arrays: Mapping[str, Any]) -> int:
    """Physics steps a RoboTwin action ran before an execution guard stopped it.

    The measured-convergence loop writes a sample's targets, steps physics, then
    measures and applies its guards. A guard that fires after the step leaves a
    record holding every issued target, one row per step that ran, and the
    recorder marks it not replayable only because it faulted. Those steps ran,
    and skipping them shifts every later action, so replay runs them like any
    other dense entry. The record is accepted only when its counters prove it is
    such a prefix; otherwise the replay refuses rather than skip or guess.
    Returns 0 for an entry that is not an execution fault or ran nothing.
    """
    execution = entry.get("execution")
    if not isinstance(execution, Mapping) or execution.get("status") != "execution_fault":
        return 0
    label = f"RoboTwin action {entry.get('action_sequence')} execution fault"
    metadata = entry.get("metadata")
    tracking = metadata.get("controller_tracking") if isinstance(metadata, Mapping) else None
    abort = tracking.get("abort") if isinstance(tracking, Mapping) else None
    code = abort.get("code") if isinstance(abort, Mapping) else None
    if not isinstance(code, str):
        raise ReplayWorkerError(f"{label} records no controller abort")
    if code in ROBOTWIN_PRE_STEP_FAULTS:
        raise ReplayWorkerError(f"{label} {code} wrote a target without its physics step")
    counters = {name: tracking.get(name) for name in ("physics_step_attempts", "sampled_steps")}
    counters["physics_steps"] = entry.get("physics_steps")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counters.values()):
        raise ReplayWorkerError(f"{label} records no step counters")
    steps = counters["physics_steps"]
    if len(set(counters.values())) != 1:
        raise ReplayWorkerError(f"{label} step counters disagree: {counters}")
    if steps == 0:
        return 0
    controller = entry.get("controller")
    if controller not in ROBOTWIN_DENSE_TARGET_CONTROLLERS:
        raise ReplayWorkerError(f"{label} has no dense controller sequence: {controller}")
    if controller in ROBOTWIN_STAGED_CONTROLLERS and tracking.get("mode") != "global":
        raise ReplayWorkerError(f"{label} stopped after its global phase; the record keeps only the last phase")
    settle = execution.get("settle_steps", 0)
    if settle not in (0, None):
        raise ReplayWorkerError(f"{label} records settle steps after the fault")
    sequence = _robotwin_control_sequence(entry, arrays)
    arms = 0
    for channel, path in sequence.items():
        if path is None:
            continue
        rows = len(path["position"]) if channel.endswith("_arm") else len(path.get("result", []))
        if channel.endswith("_arm"):
            arms += 1
            if len(path.get("velocity", [])) != rows:
                raise ReplayWorkerError(f"{label} {channel} positions and velocities differ in length")
        if rows != steps:
            raise ReplayWorkerError(f"{label} {channel} holds {rows} rows for {steps} physics steps")
    if tracking.get("controller_write_attempts") != steps * arms:
        raise ReplayWorkerError(f"{label} target writes do not match its {steps} physics steps")
    return steps


def _robotwin_replay_needs_planner(manifest: Mapping[str, Any]) -> bool:
    """Return whether any replayable entry must reconstruct a legacy EE action."""

    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ReplayWorkerError("RoboTwin replay entries are malformed")
    return any(
        isinstance(entry, Mapping)
        and bool(entry.get("replayable"))
        and _robotwin_replay_route(entry.get("controller")) == "ee_action"
        for entry in entries
    )


def _disable_unused_robotwin_planners(robotwin_root: Path) -> tuple[Any, Any] | None:
    """Avoid constructing CuRobo when replay consumes recorded dense targets.

    RoboTwin constructs two planners during ``setup_demo`` even though dense
    replay never calls either planner.  The Viewer worker is isolated and
    source-bound, so replacing only that constructor hook is safe and removes
    the dominant cold-start cost without modifying the pinned source tree.
    """

    root_text = str(robotwin_root)
    curobo_text = str(robotwin_root / "envs/curobo/src")
    for path in (root_text, curobo_text):
        if path not in sys.path:
            sys.path.insert(0, path)
    previous_working_directory = Path.cwd().resolve()
    try:
        os.chdir(robotwin_root)
        module = importlib.import_module("envs.robot.robot")
    finally:
        os.chdir(previous_working_directory)
    robot_class = getattr(module, "Robot", None)
    original = getattr(robot_class, "set_planner", None)
    if robot_class is None or not callable(original):
        raise ReplayWorkerError("RoboTwin replay planner hook is unavailable")

    class SetupOnlyGripperPlanner:
        """Minimal deterministic helper required by RoboTwin reset."""

        @staticmethod
        def plan_grippers(now_val: float, target_val: float) -> dict[str, Any]:
            num_step = 200
            return {
                "num_step": num_step,
                "per_step": (target_val - now_val) / num_step,
                "result": module.np.linspace(now_val, target_val, num_step),
            }

    def recorded_target_only(self: Any, scene: Any = None) -> None:
        del scene
        self.communication_flag = False
        self.left_planner = SetupOnlyGripperPlanner()
        self.right_planner = SetupOnlyGripperPlanner()

    robot_class.set_planner = recorded_target_only
    return robot_class, original


def _apply_robotwin_control_target(
    environment: Any, control_sequence: Mapping[str, Any], control_index: int,
) -> None:
    """Apply one recorded dense target without invoking a planner."""

    def length(value: Any) -> int:
        shape = getattr(value, "shape", ())
        return int(shape[0]) if shape and isinstance(shape[0], int) else 0

    robot = environment.robot
    for arm in ("left", "right"):
        arm_channel = control_sequence.get(f"{arm}_arm")
        if isinstance(arm_channel, Mapping):
            position = arm_channel.get("position")
            velocity = arm_channel.get("velocity")
            if (
                length(position) > control_index
                and length(velocity) > control_index
            ):
                robot.set_arm_joints(
                    position[control_index], velocity[control_index], arm,
                )
        gripper = control_sequence.get(f"{arm}_gripper")
        if isinstance(gripper, Mapping):
            num_step = gripper.get("num_step")
            result = gripper.get("result")
            if (
                isinstance(num_step, int)
                and control_index < num_step
                and length(result) > control_index
            ):
                robot.set_gripper(
                    result[control_index], arm, gripper.get("per_step"),
                )


def _robotwin_replay_session(
    plan: Mapping[str, Any], project_root: Path, output_root: Path,
) -> Any:
    """The run's RoboTwin session, rebuilt from its RunPlan without capture."""
    from ora.environment.backends.robotwin.session import RoboTwinLiveSession
    from ora.environment.controllers.embodiment_control import (
        load_embodiment_control_profile,
    )

    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ReplayWorkerError("run omits Runtime configuration")
    binding = runtime.get("control_profile")
    control_profile = None
    if isinstance(binding, Mapping):
        relative = binding.get("path")
        digest = binding.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ReplayWorkerError("RoboTwin replay control profile is malformed")
        control_profile = load_embodiment_control_profile(
            project_root / relative, digest,
        )
    return RoboTwinLiveSession(
        robotwin_root=_required_environment_path(runtime, "root_env"),
        output_root=output_root / "runtime",
        task=str(plan["task"]),
        actual_seed=int(plan["seed"]),
        episode=0,
        task_config=str(runtime["task_config"]),
        required_revision=str(runtime["required_revision"]),
        workspace_bounds_m=runtime["workspace_bounds_m"],
        max_sim_steps=int(plan["budgets"]["max_sim_steps"]),
        operator_world_video=False,
        operator_execution_video=False,
        execution_capture={"enabled": False},
        environment_signal_policy=str(
            runtime.get("environment_signal_policy", "lifecycle_only")
        ),
        action_codec=str(plan.get("policy", {}).get("action_codec", "runtime_action_v3")),
        control_profile=control_profile,
    )


def _robotwin_scene_trace(directory: Path, run_id: str) -> Any:
    """The scene trace of a RoboTwin re-execution (``environment/scene_trace.py``)."""
    from ora.environment.backends.robotwin.scene_trace import RoboTwinSceneTrace

    return RoboTwinSceneTrace(directory, run_id=run_id)


def _libero_scene_trace(directory: Path, run_id: str) -> Any:
    """The scene trace of a LIBERO-Pro re-execution (``environment/scene_trace.py``)."""
    from ora.environment.backends.libero_pro.scene_trace import LiberoSceneTrace

    return LiberoSceneTrace(directory, run_id=run_id)


def _robocasa_replay_session(plan: Mapping[str, Any], output_root: Path, region_order: Mapping[int, Any]) -> Any:
    """The run's RoboCasa365 session, rebuilt from its RunPlan: no planner, recorded controls need none."""
    from ora.environment.backends.robocasa.session import RoboCasaSession

    return RoboCasaSession(output_root=output_root / "runtime", config={
        **plan["runtime"], "task": plan["task"], "seed": plan["seed"], "motion_planner": None,
        "action_codec": plan["policy"]["action_codec"], "region_order": region_order})


def _robocasa_scene_trace(directory: Path, run_id: str) -> Any:
    """The scene trace of a RoboCasa365 re-execution (``environment/scene_trace.py``)."""
    from ora.environment.backends.robocasa.scene_trace import RoboCasaSceneTrace

    return RoboCasaSceneTrace(directory, run_id=run_id)


def _replay_robotwin_entry(
    environment: Any, entry: Mapping[str, Any], arrays: Mapping[str, Any], *,
    sampling: Mapping[str, Any], capture: Callable[[Any, Any, str, int | None], None],
    hold_steps: int | None = None,
) -> str:
    """Run one recorded RoboTwin entry; returns what ran: ``controls``, ``hold`` or ``skipped``.

    ``capture(environment, entry, phase, control_step)`` sees the sampled steps;
    sampling selects captures only, never which steps run. ``hold_steps`` lets a
    replay program run a wait for another number of steps.
    """
    replay = bool(entry.get("replayable")) or _robotwin_fault_prefix_steps(entry, arrays) > 0
    hold = 0 if replay else _robotwin_hold_steps(entry)
    if hold and hold_steps is not None:
        hold = int(hold_steps)
    if not replay and not hold:
        ran = entry.get("physics_steps")
        if isinstance(ran, int) and not isinstance(ran, bool) and ran > 0 and hold_steps is None:
            # Skipping steps that ran starts every later action from
            # another state: that is no longer a replay of this run.
            raise ReplayWorkerError(
                f"RoboTwin action {entry.get('action_sequence')} ran {ran} physics steps "
                "that its record cannot replay"
            )
        capture(environment, entry, "not_replayed", None)
        return "skipped"
    controller = entry.get("controller")
    if bool(getattr(environment, "eval_success", False)):
        raise ReplayWorkerError("RoboTwin replay would write after a terminal boundary")
    if hold:
        # The wait's own steps, with no new target: run every one.
        selected_steps = _capture_steps(hold, sampling)
        original_update = getattr(environment, "_update_render")
        last_captured_global_step = None
        for index in range(hold):
            environment.scene.step()
            if index in selected_steps:
                original_update()
                capture(environment, entry, "hold", index)
                last_captured_global_step = index
        if _needs_final_capture(hold, last_captured_global_step):
            original_update()
            capture(environment, entry, "hold_final", hold - 1)
        return "hold"
    step = 0
    last_captured_global_step: int | None = None
    settle = _extra_settle_steps(entry, arrays)
    expected_control_steps = _recorded_control_step_count(entry, arrays)
    selected_steps = _capture_steps(
        expected_control_steps + settle, sampling,
    )
    original_update = getattr(environment, "_update_render")

    # Sampling controls camera capture only. Every recorded target and
    # settle step must run, and none twice: skipping or repeating steps
    # changes contact/grasp physics.
    def recorded_update() -> None:
        nonlocal step, last_captured_global_step
        if step in selected_steps:
            original_update()
            capture(environment, entry, "control", step)
            last_captured_global_step = step
        step += 1

    setattr(environment, "_update_render", recorded_update)
    try:
        replay_route = _robotwin_replay_route(controller)
        if replay_route == "dense_targets":
            original_picture = getattr(environment, "_take_picture", None)
            original_render_freq = getattr(environment, "render_freq", None)
            if callable(original_picture):
                setattr(environment, "_take_picture", lambda: None)
            if original_render_freq is not None:
                setattr(environment, "render_freq", 0)
            try:
                environment.take_dense_action(
                    _legacy_gripper_steps(_robotwin_control_sequence(entry, arrays)),
                    save_freq=1,
                )
            finally:
                if callable(original_picture):
                    setattr(environment, "_take_picture", original_picture)
                if original_render_freq is not None:
                    setattr(environment, "render_freq", original_render_freq)
        else:
            environment.take_action(arrays["ee_action"], action_type="ee")
    finally:
        setattr(environment, "_update_render", original_update)
    scene_step = getattr(getattr(environment, "scene", None), "step", None)
    for index in range(settle):
        if not callable(scene_step):
            break
        scene_step()
        global_step = step + index
        if global_step in selected_steps:
            original_update()
            capture(environment, entry, "settle", index)
            last_captured_global_step = global_step
    actual_total_steps = step + settle
    if _needs_final_capture(actual_total_steps, last_captured_global_step):
        original_update()
        capture(
            environment, entry, "settle_final" if settle else "control_final",
            (settle - 1) if settle else (step - 1),
        )
    return "controls"


def _replay_robotwin(
    *, plan: Mapping[str, Any], manifest: Mapping[str, Any],
    manifest_path: Path, project_root: Path, output_root: Path,
    sampling: Mapping[str, Any],
) -> tuple[int, list[str]]:
    session = _robotwin_replay_session(plan, project_root, output_root)
    cameras = list(manifest["camera_panels"])
    writer = _FrameWriter(
        output_root, cameras=cameras, sampling=sampling,
    )

    def capture(environment: Any, entry: Any, phase: str, control_step: int | None) -> None:
        writer.append(
            _robotwin_images(environment), entry=entry, phase=phase, control_step=control_step,
        )

    planner_patch = None
    if not _robotwin_replay_needs_planner(manifest):
        planner_patch = _disable_unused_robotwin_planners(session.robotwin_root)
    try:
        session.start()
        environment = session._require_environment()
        capture(environment, None, "reset", None)
        current_round = 0
        for entry in manifest["entries"]:
            if not isinstance(entry, Mapping):
                raise ReplayWorkerError("RoboTwin replay entry is malformed")
            # Replay must cross the same round boundaries the run did.
            # Without this the derived media shows every later round acting
            # on the previous round's end state, so the video and the
            # trajectory it claims to depict diverge.
            round_index = int(entry.get("round_index", 0))
            if round_index != current_round:
                session.restart_round(round_index)
                current_round = round_index
                environment = session._require_environment()
                capture(environment, None, "round_reset", None)
            arrays = _artifact_arrays(manifest_path, entry)
            _replay_robotwin_entry(
                environment, entry, arrays, sampling=sampling, capture=capture,
            )
    finally:
        try:
            session.close()
        finally:
            if planner_patch is not None:
                robot_class, original = planner_patch
                robot_class.set_planner = original
    return writer.sequence, cameras


def _libero_images(session: Any) -> dict[str, Any]:
    import numpy as np

    raw = session._raw_observation()
    return {
        "agentview": np.asarray(raw["agentview_image"])[::-1],
        "wrist": np.asarray(raw["robot0_eye_in_hand_image"])[::-1],
    }


def _refresh_libero_replay_observation(session: Any) -> None:
    """Transfer the actual step result; never label the reset camera cache fresh."""
    observation = session._worker_attr("harness_replay_observation")
    if not isinstance(observation, Mapping):
        raise ReplayWorkerError("LIBERO replay returned no step-bound observation")
    session._require_environment().current_raw_obs = [dict(observation)]


def _libero_qualified_controller_binding(manifest_path: Path) -> Mapping[str, Any]:
    """Load the exact launch-time qualification sealed by the source run."""

    reset = _load(manifest_path.parent.parent / "reset.json")
    if reset.get("schema") != "ora.libero-pro-reset.v1":
        raise ReplayWorkerError("run omits its sealed LIBERO reset evidence")
    binding = reset.get("qualified_controller_binding")
    if not isinstance(binding, Mapping):
        raise ReplayWorkerError(
            "run reset evidence omits its qualified LIBERO controller binding"
        )
    return binding


def _libero_replay_session(
    plan: Mapping[str, Any], manifest_path: Path, output_root: Path,
) -> Any:
    """The run's LIBERO-Pro session, rebuilt from its RunPlan and sealed reset evidence."""
    from ora.environment.backends.libero_pro.session import LiberoProSession

    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ReplayWorkerError("run omits Runtime configuration")
    qualified_controller_binding = _libero_qualified_controller_binding(manifest_path)
    return LiberoProSession(
        environment_driver=str(runtime.get("environment_driver", "direct_libero_v1")),
        libero_pro_root=_required_environment_path(runtime, "libero_pro_root_env"),
        data_root=_required_environment_path(runtime, "data_root_env"),
        output_root=output_root / "runtime",
        task=str(plan["task"]),
        seed=int(plan["seed"]),
        suite=str(runtime["suite"]),
        task_id=int(runtime["task_id"]),
        robustness_case=(
            str(runtime["robustness_case"])
            if runtime.get("robustness_case") is not None else None
        ),
        expected_task_language=str(runtime["expected_task_language"]),
        required_libero_pro_revision=str(runtime["required_libero_pro_revision"]),
        required_data_revision=str(runtime["required_data_revision"]),
        workspace_bounds_m=runtime["workspace_bounds_m"],
        workspace_profile=runtime.get("workspace_profile"),
        # Qualification is a launch-time result, so it is sealed in reset evidence
        # rather than the immutable preflight RunPlan.  Replay must revalidate that
        # exact binding instead of silently bypassing the normal Session gate.
        qualified_controller_binding=qualified_controller_binding,
        max_sim_steps=int(plan["budgets"]["max_sim_steps"]),
        servo_max_steps=int(runtime.get("servo_max_steps", 80)),
        action_codec="runtime_action_v3",
        execution_capture={"enabled": False},
    )


def _libero_command_setter(session: Any) -> Callable[[str, Any], None]:
    environment = session._require_environment()
    setter = getattr(getattr(environment, "env", None), "set_env_attr", None)
    if not callable(setter):
        raise ReplayWorkerError("LIBERO replay command transport is unavailable")
    return setter


def _replay_libero_entry(
    session: Any, entry: Mapping[str, Any], arrays: Mapping[str, Any], *,
    capture: Callable[[Any, str, int | None], None],
) -> str:
    """Run one recorded LIBERO entry; returns what ran: ``controls`` or ``skipped``.

    ``capture(entry, phase, control_step)`` sees every command step.
    """
    if not entry.get("replayable"):
        ran = entry.get("physics_steps")
        if isinstance(ran, int) and not isinstance(ran, bool) and ran > 0:
            raise ReplayWorkerError(
                f"LIBERO action {entry.get('action_sequence')} ran {ran} control steps "
                "that its record cannot replay"
            )
        capture(entry, "not_replayed", None)
        return "skipped"
    setter = _libero_command_setter(session)
    controller = entry.get("controller")
    if controller == "libero_joint_position":  # a v3 run's joint segment names its gains (libero_joint_controller)
        setter("harness_replay_controller_mode", (entry.get("metadata") or {}).get("joint_controller") or "JOINT_POSITION")
        commands = arrays.get("commands", [])
        for index, command in enumerate(commands):
            setter("harness_replay_command", command.tolist())
            _refresh_libero_replay_observation(session)
            capture(entry, "control", index)
        setter("harness_replay_controller_mode", "OSC_POSE")
        refresh = arrays.get("refresh_osc_command", [])
        for offset, command in enumerate(refresh):
            setter("harness_replay_command", command.tolist())
            _refresh_libero_replay_observation(session)
            capture(entry, "refresh", len(commands) + offset)
    elif controller == "libero_osc_pose":
        setter("harness_replay_controller_mode", "OSC_POSE")
        for index, command in enumerate(arrays.get("commands", [])):
            setter("harness_replay_command", command.tolist())
            _refresh_libero_replay_observation(session)
            capture(entry, "control", index)
    else:
        raise ReplayWorkerError(
            f"unsupported LIBERO replay controller: {controller}"
        )
    return "controls"


def _replay_libero(
    *, plan: Mapping[str, Any], manifest: Mapping[str, Any],
    manifest_path: Path, output_root: Path,
) -> tuple[int, list[str]]:
    session = _libero_replay_session(plan, manifest_path, output_root)
    cameras = list(manifest["camera_panels"])
    writer = _FrameWriter(
        output_root,
        cameras=cameras,
        sampling={"mode": "physics_stride_v1", "capture_stride": 1},
    )

    def capture(entry: Any, phase: str, control_step: int | None) -> None:
        writer.append(_libero_images(session), entry=entry, phase=phase, control_step=control_step)

    try:
        session.start()
        capture(None, "reset", None)
        current_round = 0
        _libero_command_setter(session)
        for entry in manifest["entries"]:
            if not isinstance(entry, Mapping):
                raise ReplayWorkerError("LIBERO replay entry is malformed")
            round_index = int(entry.get("round_index", 0))
            if round_index != current_round:
                session.restart_round(round_index)
                current_round = round_index
                capture(None, "round_reset", None)
            arrays = _artifact_arrays(manifest_path, entry)
            _replay_libero_entry(session, entry, arrays, capture=capture)
    finally:
        session.close()
    return writer.sequence, cameras


def run(job_path: Path) -> dict[str, Any]:
    from ora.environment.controller_replay import (
        load_controller_replay_manifest,
    )

    job = _load(job_path)
    if job.get("schema") == PROGRAM_JOB_SCHEMA:
        from .replay_program_worker import run_program  # replay programs: their own module

        return run_program(job)
    if job.get("schema") != JOB_SCHEMA:
        raise ReplayWorkerError("invalid Viewer replay job")
    run_root = Path(str(job["run_root"])).resolve()
    output_root = Path(str(job["output_root"])).resolve()
    project_root = Path(str(job["project_root"])).resolve()
    sampling = job.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ReplayWorkerError("Viewer replay job omits its sampling profile")
    # Validate the closed profile before allocating the simulator.
    _capture_steps(1, sampling)
    if output_root == run_root or output_root.is_relative_to(run_root):
        raise ReplayWorkerError("Viewer replay output must remain outside the run")
    plan = _load(run_root / "plan.json")
    manifest_path = run_root / "native/controller-replay/manifest.json"
    manifest = load_controller_replay_manifest(manifest_path)
    effective_sampling = (
        dict(sampling)
        if manifest["backend"] == "robotwin"
        else {"mode": "physics_stride_v1", "capture_stride": 1}
    )
    temporary = output_root.with_name(output_root.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        if manifest["backend"] == "robotwin":
            frame_count, cameras = _replay_robotwin(
                plan=plan, manifest=manifest, manifest_path=manifest_path,
                project_root=project_root, output_root=temporary,
                sampling=effective_sampling,
            )
        elif manifest["backend"] == "libero_pro":
            frame_count, cameras = _replay_libero(
                plan=plan, manifest=manifest, manifest_path=manifest_path,
                output_root=temporary,
            )
        else:
            raise ReplayWorkerError("unsupported controller replay backend")
        result = {
            "schema": RESULT_SCHEMA,
            "backend": manifest["backend"],
            "source_run_id": run_root.name,
            "source_manifest": {
                "path": "native/controller-replay/manifest.json",
                "sha256": _hash(manifest_path),
            },
            "frame_count": frame_count,
            "sampling": effective_sampling,
            "camera_panels": cameras,
            "timeline": {
                "path": "timeline.jsonl",
                "sha256": _hash(temporary / "timeline.jsonl"),
            },
            "capture_semantics": (
                "derived_same_source_seed_controller_replay_action_keyframes"
                if effective_sampling.get("mode") == "action_keyframes_v1"
                else "derived_same_source_seed_controller_replay"
                if effective_sampling.get("capture_stride") == 1
                else "derived_same_source_seed_controller_replay_temporally_subsampled"
            ),
            "evidence_boundary": (
                "viewer_derived_not_original_physics_or_official_verdict_evidence"
            ),
        }
        _atomic_json(temporary / "manifest.json", result)
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(temporary, output_root)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile one Viewer controller replay.")
    parser.add_argument("--job", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args.job.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
