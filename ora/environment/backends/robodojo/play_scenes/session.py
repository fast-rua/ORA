"""One isolated RoboDojo free-play session (worker backend ``robodojo_play``).

``RobodojoPlaySession`` is the benchmark ``RobodojoSession`` with only what a
play scene needs overridden; observation, every action route (runtime v3 and
sparse motion v3), motion probes, read-only alignment and IK, action
sequences and execution capture are the benchmark's own code:

- construction admits the scene into this process's native-action table at
  the play limit (``robodojo_play_scenes.PLAY_NATIVE_ACTION_LIMIT``; the
  benchmark table is not changed anywhere else), and binds the play seed;
- ``start`` boots through ``play_scenes/bootstrap.py`` and resets on the
  play seed's generated layout. Its reset record has no task instruction,
  infrastructure qualification or evaluator registration; it carries the
  scene, seed, generated-layout identity and the play scene registration;
- ``start`` binds the shared motion base as the benchmark's does
  (``shared_motion.start``): the timing worker the run's preflight bound, and
  the planner ``runtime.motion_planner`` names, on this worker's card. A
  planner that does not start, or whose forward kinematics differs from the
  simulator's, refuses the start; a plan that names no planner starts without
  one, and every planner request is answered ``no_planner``;
- nothing latches success or failure: the environment never reports an
  official terminal, ``persist_official_verdict`` is refused (a free-play run
  has no verdict), and a scene ends only at the Supervisor's boundary or the
  play step limit (``max_sim_steps``);
- ``restart_round(k, seed=s)`` always relaunches the simulation inside the
  same Kit process on seed ``s``'s layout (CPU and CUDA alike: an in-place
  CPU restart leaves respawned objects gravity-free), then proves the new
  scene is that layout, the robot is home and no channel carries the previous
  scene. The new simulation's physics clock starts again, so the relaunch also
  drops the last observation's world: the planner plans in this scene's only
  after its first observation (``shared_motion.world_input``);
- ``operator_view`` returns the operator's third-person camera
  (``operator_view.py``), which is never one of the policy's cameras; with
  the default mode the execution capture records it as a fourth camera, so
  the operator's page plays the motion through it;
- after each action the world keeps running until the scene's objects are
  at rest or ``lifecycle.PLAY_POST_ACTION_SETTLE`` caps it (2 s of simulated
  time), every robot target held (``_settle_after_action``): physics
  advances only inside an action, so an object the arm knocked over would
  otherwise hang in mid-fall until the next one. The settle runs inside the
  action's execution-capture window (its frames carry the action's id, its
  ``action_end`` frame comes after it), after the action's receipt, and is
  recorded per action in ``post-action-settle.jsonl``; it is not part of a
  receipt, an observation or the session's physics budget (``sim_steps``).
  Between the children of an action sequence it does not run;
- ``close`` closes the planner process (``shared_motion.close``) and tears
  down neither the simulation nor Kit in-process: it ends the worker process
  with ``os._exit`` once its RPC reply is written (at interpreter exit, or
  after ``CLOSE_EXIT_GRACE_S`` should Kit's non-daemon threads hold the exit).
"""

from __future__ import annotations

import atexit
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Mapping

from ora.environment.benchmarks import robodojo as benchmark
from ora.environment.benchmarks.robodojo_play_scenes import (
    PLAY_NATIVE_ACTION_LIMIT, PlaySceneSetError, layout_sha256, scene_definition, scene_language,
)
from ora.environment.controllers.robodojo_arx import controller_binding

from .. import shared_motion
from ..bridge import RobodojoError
from ..common import file_hash, pose_error, write_json
from ..reset_diagnostics import stack_timer
from ..round_lifecycle import (
    PER_ROUND_CLEARED_CHANNELS, ROUND_HOME_POSITION_TOLERANCE_M, ROUND_HOME_ROTATION_TOLERANCE_DEG,
)
from ..session import RobodojoSession
from . import operator_view
from .bootstrap import build_play_environment, configured_layouts, physics_device
from .lifecycle import PLAY_POST_ACTION_SETTLE, displaced

PLAY_RESET_SCHEMA = "ora.robodojo-play-reset.v1"
POST_ACTION_SETTLE_SCHEMA = "ora.robodojo-play-post-action-settle.v1"
POST_ACTION_SETTLE_RECORD = "post-action-settle.jsonl"
#: After ``close`` the worker process has this long to write its reply and exit on its own.
CLOSE_EXIT_GRACE_S = 10.0
PLAY_ROUND_RESET_SCHEMA = "ora.robodojo-play-round-reset.v1"
PLAY_ISOLATION_SCHEMA = "ora.robodojo-play-scene-isolation.v1"
SCENE_RESET_RELAUNCH = "simulation_relaunch"


def admit_play_scene(task: str) -> None:
    """Admit the play scene into this worker process's native-action table at the play limit.

    The benchmark session reads ``NATIVE_ACTION_LIMITS`` for its task. A play
    worker serves exactly one play scene, so the admission lives as long as
    the process; an official task is never touched.
    """
    limits = benchmark.NATIVE_ACTION_LIMITS
    if limits.get(task, PLAY_NATIVE_ACTION_LIMIT) != PLAY_NATIVE_ACTION_LIMIT:
        raise RobodojoError(f"play scene {task!r} collides with an official RoboDojo task")
    limits[task] = PLAY_NATIVE_ACTION_LIMIT


class RobodojoPlaySession(RobodojoSession):
    """A RoboDojo session on one of ORA's play scenes: no task, no verdict, relaunch-only scene resets."""

    def __init__(self, *, config: Mapping[str, Any], output_root: Path,
                 environment_factory: Callable[..., tuple[Any, Any, dict[str, Any]]] = build_play_environment) -> None:
        raw = dict(config)
        try:
            self.play_definition = scene_definition(str(raw.get("play_scene_set")), str(raw.get("task")))
        except PlaySceneSetError as exc:
            raise RobodojoError(str(exc)) from None
        self.play_layouts = configured_layouts(raw)
        seed = raw.get("play_seed")
        if type(seed) is not int or seed not in self.play_layouts:
            raise RobodojoError("the play seed must be one of runtime.play_layouts")
        if raw.get("physics_device") != physics_device(raw, self.play_definition):
            raise RobodojoError(f"play scene {self.play_definition.name} runs on "
                                f"{self.play_definition.physics_device} physics")
        if raw.get("layout_schedule") is not None:
            raise RobodojoError("a play scene changes layout only through restart_round(seed=...)")
        admit_play_scene(self.play_definition.name)
        super().__init__(config=raw, output_root=output_root, environment_factory=environment_factory)
        self.play_seed = seed
        self.scene_epoch = 0
        #: The RPC worker process ends with its session; an in-process owner (the audit) exits itself.
        self.exit_process_on_close = True

    # -- the scene ---------------------------------------------------------------------------

    def _task_instruction(self) -> str:
        return scene_language(self.play_definition)

    def _scene_registration(self) -> dict[str, Any]:
        value = getattr(self.environment, "scene_registration", None)
        if not isinstance(value, Mapping) or not isinstance(value.get("instances"), Mapping):
            raise RobodojoError("the play scene registration is unavailable")
        return deepcopy(dict(value))

    def _reset_environment(self, stack_log_name: str, *, relaunch_layout: int | None = None) -> Any:
        environment = self.environment
        environment.harness_layout_ids = tuple(sorted(self.play_layouts))
        if getattr(environment, "harness_simulation_time_offset_s", None) is None:
            # Every play scene may relaunch: epoch 0 starts the annotator clock with the context clock.
            environment.harness_simulation_time_offset_s = 0.0
        relaunch = None
        with (self.output_root / stack_log_name).open("w") as stack_log, stack_timer(120, stack_log):
            if relaunch_layout is None:
                environment.reset(seed=[self.play_seed])
            else:
                relaunch = environment.relaunch_for_layout(relaunch_layout)
                # A new simulation, whose physics clock starts again: the last observation's world is the
                # previous scene's, whatever the two clocks read (shared_motion.world_input).
                self.last_observation_root = self.last_observation_clock = None
        if environment.step_lim != self.native_action_limit or environment.reward_registered:
            raise RobodojoError("the play scene's lifecycle does not match its binding")
        return relaunch

    def _loaded_layout(self) -> dict[str, Any]:
        """The saved layout the live scene was built from, with its identity against the plan's."""
        loaded = self.environment.loaded_layout_state()
        digest = layout_sha256(loaded["layout"])
        return {"seed": int(loaded["env_seed"]), "layout_sha256": digest,
                "planned_layout_sha256": self.play_layouts.get(int(loaded["env_seed"])),
                "object_counts": loaded["object_counts"]}

    def _reset_record(self, schema: str, round_index: int) -> dict[str, Any]:
        layout = self._loaded_layout()
        return {
            "schema": schema, "task": self.task, "scene_set": self.play_definition.scene_set,
            "eval_seed": 0, "seed": self.play_seed, "layout_id": 0, "round_index": round_index,
            "play_layout": layout,
            "source_manifest": {"path": "source-manifest.json", "sha256": self.source_hash},
            "source_binding": {"source_manifest_sha256": self.source_hash,
                               "source_revisions": self.source["source_revisions"]},
            "control_binding": controller_binding(),
            "device_binding": self.source.get("device_binding"),
            "physics_steps_before_policy": self.reset_physics_steps,
            "physics_device": physics_device(self.config, self.play_definition),
            "reward_registered": False, "task_instruction": None,
            "public_environment_context": benchmark.public_context(),
            "scene_registration": self._scene_registration(),
            "action_codec": self.action_codec,
        }

    def start(self) -> dict[str, Any]:
        if self.started or self.closed:
            raise RobodojoError("a RoboDojo session starts exactly once")
        self.output_root.mkdir(parents=True, exist_ok=True)
        try:
            self.environment, self.app, self.source = self.environment_factory(self.config)
            self._reset_environment("reset-initialization-stack.txt")
            self.robots = {robot.arm_name.split("_")[0]: robot
                           for robot in self.environment.robot_manager.robot_list if robot.type == "target"}
            if set(self.robots) != {"left", "right"}:
                raise RobodojoError("ARX X5 must have exactly left and right target arms")
            from ora.environment.controllers.robodojo_arx import validate_robot_facts

            for robot in self.robots.values():
                validate_robot_facts({
                    "joint_dof": len(robot.arm_joints_name), "ee_link": robot.ee_link_name, "ee_type": robot.ee_type,
                    "gripper_scale": list(robot.gripper_scale), "gripper_sign": robot.gripper_move["sign"],
                    "gripper_mimic_multiplier": robot.gripper_move["mimic"][1],
                    "gripper_mimic_offset": robot.gripper_move["mimic"][2],
                })
            runtime_embodiment = self._runtime_embodiment()
            self.robot_geometry_verification = None
            if self.config.get("public_robot_geometry") is not None:
                from ora.contracts.robot_geometry import verify_loaded_robot_geometry
                from ..robot_asset_geometry import loaded_geometry
                self.robot_geometry_verification = verify_loaded_robot_geometry(
                    self.config["public_robot_geometry"], loaded_geometry(self))
            self._bind_reset_state()
            shared_motion.start(self)
            from ..layout_rounds import plant_facts

            self.session_plant_facts = plant_facts(self)
            manifest_path = self.output_root / "source-manifest.json"
            if manifest_path.exists():
                raise RobodojoError("refusing to overwrite a source manifest")
            write_json(manifest_path, self.source)
            self.source_hash = file_hash(manifest_path)
            self.started = True
            if self.config.get("qualification_binding") is not None:
                raise RobodojoError("a play scene has no infrastructure qualification binding")
            if self.config.get("expected_task_language", self._task_instruction()) != self._task_instruction():
                raise RobodojoError("the plan's scene language differs from the play scene")
            layout = self._loaded_layout()
            if layout["seed"] != self.play_seed or layout["layout_sha256"] != self.play_layouts[self.play_seed]:
                raise RobodojoError("the live scene is not the play seed's audited layout")
            reset = self._reset_record(PLAY_RESET_SCHEMA, 0)
            reset["runtime_embodiment"] = runtime_embodiment
            write_json(self.output_root / "reset.json", reset)
            self._start_execution_capture()
            return reset
        except BaseException:
            import traceback

            (self.output_root / "session-start-error.txt").write_text(traceback.format_exc())
            self._fault("robodojo_start_failed")
            raise

    def restart_round(self, round_index: int, *, seed: int | None = None) -> dict[str, Any]:
        """Relaunch the simulation on seed ``seed``'s layout (default: the current seed)."""
        from ..layout_rounds import plant_facts, relaunch_proven, simulation_time_offset

        if isinstance(round_index, bool) or not isinstance(round_index, int):
            raise RobodojoError("RoboDojo round index must be an integer")
        if round_index != self.round_index + 1:
            raise RobodojoError("RoboDojo round reset must advance by exactly one")
        self._require_started()
        target = self.play_seed if seed is None else seed
        if isinstance(target, bool) or type(target) is not int or target not in self.play_layouts:
            raise RobodojoError("a play scene resets only to one of its audited seeds")
        if (self.output_root / "official-verdict.json").exists():
            raise RobodojoError("a free-play session never persists an official verdict")
        self.finalize_operator_video_segment()
        self.environment.harness_round_index = round_index
        previous_seed, previous_offset = self.play_seed, simulation_time_offset(self)
        relaunch = self._reset_environment(f"reset-initialization-stack-round-{round_index:04d}.txt",
                                           relaunch_layout=target)
        self.play_seed, self.scene_epoch = target, self.scene_epoch + 1
        state = self._bind_reset_state()
        self.round_index = round_index
        layout = self._loaded_layout()
        # The relaunch proof is arithmetic; the cameras' clock was then measured (lifecycle.py).
        camera_clock = getattr(self.environment, "harness_epoch_clock", None)
        arithmetic = (camera_clock or {}).get("arithmetic_offset_s", simulation_time_offset(self))
        proof = {**dict(relaunch or {}), "simulation_time_offset_s": arithmetic}
        checks = {
            "simulation_relaunched": relaunch_proven(
                proof, layout_id=target, previous_layout_id=previous_seed, reset_clock=self.reset_physics_steps,
                previous_offset=previous_offset, offset=arithmetic),
            "camera_clock_measured": camera_clock is not None,
            "layout_is_the_seed": layout["seed"] == target and layout["layout_sha256"] == self.play_layouts[target],
            "plant_embodiment_unchanged": plant_facts(self) == self.session_plant_facts,
            "carried_channels_cleared": all(getattr(self, name) == empty
                                            for name, empty in PER_ROUND_CLEARED_CHANNELS.items()),
            "no_reward_or_terminal": (self.environment.reward_registered is False
                                      and self.environment._official_terminal is None
                                      and self.official_terminal is None and not self.status()["terminal"]),
        }
        position_error = rotation_error = 0.0
        for arm, value in state["end_effectors"].items():
            distance, angle = pose_error(self.session_home_poses[arm], value["pose_xyz_xyzw"])
            position_error, rotation_error = max(position_error, distance), max(rotation_error, math.degrees(angle))
        checks["arm_home_restored"] = (position_error <= ROUND_HOME_POSITION_TOLERANCE_M
                                       and rotation_error <= ROUND_HOME_ROTATION_TOLERANCE_DEG)
        isolation = {
            "schema": PLAY_ISOLATION_SCHEMA, "passed": all(checks.values()), "checks": dict(sorted(checks.items())),
            "measured": {"previous_seed": previous_seed, "seed": target, "layout": layout,
                         "max_home_position_error_m": round(position_error, 6),
                         "max_home_rotation_error_deg": round(rotation_error, 4),
                         "scene_epoch": self.scene_epoch},
            "failures": [f"{name} did not hold" for name, passed in sorted(checks.items()) if not passed],
        }
        reset = {**self._reset_record(PLAY_ROUND_RESET_SCHEMA, round_index),
                 "scene_reset_mode": SCENE_RESET_RELAUNCH, "relaunch": dict(relaunch or {}),
                 "simulation_time_offset_s": simulation_time_offset(self), "camera_clock": camera_clock,
                 "artifact_sequences_preserved": {"observation_sequence": self.observation_sequence,
                                                  "action_sequence": self.action_sequence,
                                                  "motion_probe_sequence": 0},
                 "round_isolation": isolation}
        write_json(self.output_root / "round-resets" / f"round-{round_index:04d}.json", reset)
        if not isolation["passed"]:
            self._fault("robodojo_play_scene_reset_failed")
            raise RobodojoError("the play scene reset did not load an isolated new scene: "
                                + "; ".join(isolation["failures"]))
        self._start_execution_capture()
        return reset

    def finalize_operator_video_segment(self) -> None:
        """Close the previous scene's capture boundary before its simulation is destroyed."""
        self._capture_execution_boundary("scene_end")

    # -- the world between actions ---------------------------------------------------------------

    def _execute_without_capture(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """The benchmark's action, then the post-action settle, both inside the action's capture window."""
        sequence = self.action_sequence
        receipt = super()._execute_without_capture(action)
        self._settle_after_action(str(action.get("action_id")), sequence)
        return receipt

    def _settle_after_action(self, action_id: str, action_sequence: int) -> dict[str, Any] | None:
        """Run the scene on, robot targets held, until its objects rest or the settle's cap; record it.

        At rest at once when no object moves faster than the rest speeds; otherwise one window of
        held physics steps after another until no object moved beyond the window bounds in one.

        Skipped while an action sequence has children left (its next child follows at once) and on a
        terminal session. The receipt is already written: the settle changes no receipt, no
        observation and no ``sim_steps``; the next observation sees the settled scene.
        """
        if self.is_terminal or any(frame["next"] < len(frame["children"]) for frame in self.action_sequences):
            return None
        bounds = PLAY_POST_ACTION_SETTLE
        environment = self.environment
        started, clock = time.monotonic(), self._clock()
        first = environment.object_motion()
        steps, windows, moving, ended = 0, 0, list(first["moving"]), "at_rest"
        try:
            if moving:
                before = environment.object_poses()
                while True:
                    if steps >= bounds["max_physics_steps"]:
                        ended = "step_limit"
                        break
                    chunk = min(bounds["window_physics_steps"], bounds["max_physics_steps"] - steps)
                    environment.hold_steps(chunk)
                    steps, windows = steps + chunk, windows + 1
                    after = environment.object_poses()
                    moving, before = displaced(before, after), after
                    if not moving:
                        break
        except BaseException:
            self._fault("robodojo_play_settle_failed")
            raise
        if self._clock() - clock != steps:
            self._fault("robodojo_play_settle_clock_mismatch")
            raise RobodojoError("the post-action settle did not advance exactly its physics steps")
        record = {
            "schema": POST_ACTION_SETTLE_SCHEMA, "action_id": action_id, "round_index": self.round_index,
            "action_sequence": action_sequence,
            "physics_step_start": clock - self.reset_physics_steps, "physics_steps": steps,
            "simulated_s": round(steps * benchmark.PHYSICS_DT_S, 6), "ended": ended,
            "windows": windows, "moving_at_start": first["moving"], "moving_at_end": moving,
            "max_linear_speed_m_s_at_start": round(first["max_linear_speed_m_s"], 6),
            "max_angular_speed_rad_s_at_start": round(first["max_angular_speed_rad_s"], 6),
            "max_joint_speed_at_start": round(first["max_joint_speed"], 6),
            "wall_s": round(time.monotonic() - started, 4), "visibility": "control_only",
        }
        with (self.output_root / POST_ACTION_SETTLE_RECORD).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    # -- the operator's view ---------------------------------------------------------------------

    def operator_view(self) -> dict[str, Any] | None:
        """The operator camera's latest frame (``ora.environment.base.validate_operator_view``), or None.

        Free play asks right after each published observation, whose ``get_obs``
        rendered this frame together with the policy's cameras. Reading the
        annotator renders nothing, steps no physics and changes no random
        state; the camera is never one of the policy's cameras (``CAMERAS``).
        """
        reader = None if self.environment is None else getattr(self.environment, "operator_rgb", None)
        if not callable(reader):
            return None
        rgb = reader()
        return None if rgb is None else operator_view.encode(rgb)

    def _operator_capture(self) -> bool:
        return operator_view.operator_view_mode(self.config) == "capture"

    def _start_execution_capture(self) -> None:
        """The benchmark's execution capture, with the operator camera as a fourth camera (``capture`` mode)."""
        if not self._operator_capture():
            return super()._start_execution_capture()
        if not self.execution_capture["enabled"]:
            return
        try:
            from ora.environment.execution_capture import ExecutionCapture

            from ..session import CAMERAS

            if self._execution_recorder is None:
                self._execution_recorder = ExecutionCapture(
                    self.output_root, backend="robodojo", cameras=[*CAMERAS, operator_view.OPERATOR_CAMERA],
                    identity={"run_id": self.output_root.parent.name, "task": self.task,
                              "source_manifest_sha256": self.source_hash}, config=self.execution_capture,
                )
            self._execution_action = {}
            self.environment.execution_capture_callback = self._after_capture_physics_step
            self._capture_execution_boundary("reset")
        except Exception as exc:
            self._capture_failure(exc)

    def _capture_execution_boundary(self, phase: str, *, force: bool = True) -> None:
        """The benchmark's capture boundary; in ``capture`` mode each frame adds the operator camera's latest."""
        if not self._operator_capture():
            return super()._capture_execution_boundary(phase, force=force)
        recorder = self._execution_recorder
        if recorder is None:
            return
        try:
            import numpy as np

            from ora.environment.benchmarks.robodojo import PHYSICS_DT_S

            from ..session import CAMERAS

            step = self._clock() - self.reset_physics_steps
            view = operator_view.OPERATOR_VIEW

            def pixels() -> dict[str, Any]:
                before = self._clock()
                # The benchmark's own readback: get_obs synchronizes the policy cameras without physics.
                raw = self.environment.get_obs()
                if self._clock() != before:
                    raise RuntimeError("execution capture readback advanced physics")
                frames = {name: raw["vision"][source]["color"] for name, source in CAMERAS.items()}
                operator = self.environment.operator_rgb()
                # A camera attached by this reset has not rendered yet: a black frame, not a stale one.
                frames[operator_view.OPERATOR_CAMERA] = (
                    operator if operator is not None else np.zeros((view["height"], view["width"], 3), np.uint8))
                return frames

            recorder.capture(
                pixels, round_index=self.round_index, sim_step=step,
                sim_time_s=step * PHYSICS_DT_S, step_unit="physx_step_since_round_reset",
                action_id=self._execution_action.get("action_id"),
                action=self._execution_action.get("action"),
                action_sequence=self._execution_action.get("action_sequence"), phase=phase, force=force,
            )
        except Exception as exc:
            self._capture_failure(exc)

    # -- no verdict ------------------------------------------------------------------------------

    def persist_official_verdict(self) -> dict[str, Any]:
        raise RobodojoError("a free-play scene has no official verdict")

    def _official_outcome(self) -> tuple[bool | None, bool, str]:
        return None, False, "free_play_no_verdict"

    # -- the end -----------------------------------------------------------------------------

    def close(self) -> None:
        """End the session: its evidence is on disk; neither the simulation nor Kit is torn down in-process.

        A crashing Kit teardown leaves the card's Vulkan state unusable, and the
        audit measured the simulation close itself outlasting a minute after
        the last scene. The planner process is closed first, while this process
        still runs (at most 15 s: ``MotionPlannerClient.close``). The RPC worker
        writes its close reply and returns; the process then ends with
        ``os._exit``, at interpreter exit or after ``CLOSE_EXIT_GRACE_S``, before
        the client's close deadline would signal it.
        """
        if self.closed:
            return
        self.finalize_operator_video()
        self.closed = True
        try:
            shared_motion.close(self)
        finally:
            if self.exit_process_on_close:
                atexit.register(_hard_exit)
                threading.Thread(target=_hard_exit, args=(CLOSE_EXIT_GRACE_S,), daemon=True).start()


def _hard_exit(delay_s: float = 0.0) -> None:
    if delay_s:
        time.sleep(delay_s)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def play_seed_layouts(config: Mapping[str, Any]) -> dict[str, str]:
    """``runtime.play_layouts`` as the worker reads it (for diagnostics)."""
    return {str(seed): digest for seed, digest in configured_layouts(config).items()}


__all__ = ["PLAY_ISOLATION_SCHEMA", "PLAY_RESET_SCHEMA", "PLAY_ROUND_RESET_SCHEMA", "POST_ACTION_SETTLE_RECORD",
           "POST_ACTION_SETTLE_SCHEMA", "RobodojoPlaySession", "admit_play_scene", "play_seed_layouts"]
