"""Privilege-clean RoboLab episode writer for the Droid Franka + Robotiq 2F-85.

The policy sees metric RGB-D from the fixed over-shoulder camera and the wrist
camera with per-observation calibration, TCP proprioception with the encoder
and the measured jaw separation, and receipts. Object poses, contacts, the task
class and its success terms stay in this worker; RoboLab's own termination
supplies the only official verdict, latched at the first native action it holds.
"""

from __future__ import annotations

import atexit
import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ora.contracts.action import validate_runtime_action
from ora.contracts.official_limit import limit_snapshot, official_limit_multiple
from ora.contracts.official_limit import verdict_fields as official_limit_verdict_fields
from ora.contracts.policy_receipt import policy_receipt
from ora.environment.base import ENVIRONMENT_STATUS_SCHEMA, validate_environment_status
from ora.environment.benchmarks.robolab import DECIMATION, MAIN_CAMERA, TASKS, validate_runtime_extension
from ora.environment.controllers import robolab_droid as D
from ora.environment.controllers.kept_plans import stamped

from . import shared_motion
from .common import CLOSE_EXIT_GRACE_S, hard_exit
from .common import (
    FORBIDDEN_MODALITIES, OBSERVATION_SCHEMA, RESET_SCHEMA, SCORE_SCHEMA, VERDICT_SCHEMA, RobolabError, file_hash,
    write_json,
)
from .control_record import ControlRecord
from .motion import execute_motion_v3
from .reach_check import RobolabReachCheckMixin


class RobolabSession(RobolabReachCheckMixin):
    """One trusted, isolated RoboLab episode."""

    def __init__(self, *, config: Mapping[str, Any], output_root: Path, exit_process_on_close: bool = False) -> None:
        #: A worker process that owns this session ends with it (``common.hard_exit``) instead of closing Kit.
        self.exit_process_on_close = exit_process_on_close
        runtime = {"task": config.get("task"), "seed": config.get("seed"),
                   "action_codec": config.get("action_codec", "motion_action_v3")}
        if config.get("required_revision") is not None:
            runtime["required_revision"] = config["required_revision"]
        extension = validate_runtime_extension(runtime)
        self.config = dict(config)
        self.task, self.seed = extension["task"], extension["seed"]
        self.native_action_limit = extension["native_action_limit"]
        self.official_limit_multiple = official_limit_multiple(config)
        #: Where the episode stops; ``native_action_limit`` stays the official limit.
        self.extended_native_action_limit = self.native_action_limit * self.official_limit_multiple
        self.official_limit_snapshot: dict[str, Any] | None = None
        self.success_after_official_limit: int | None = None
        #: RoboLab's own subtask progress (its Score), read after every native action; evaluation-only.
        self.subtask_progress: dict[str, Any] | None = None
        self.subtask_progress_at_official_limit: dict[str, Any] | None = None
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.bridge: Any = None
        self.source: dict[str, Any] | None = None
        self.physics_steps_per_tick = DECIMATION
        self.round_index = 0
        self.observation_sequence = 0
        self.action_sequence = 0
        self.native_actions = 0
        self.terminated = self.truncated = False
        self.first_success: dict[str, Any] | None = None
        self.abort_reason: str | None = None
        self.last_receipt: dict[str, Any] | None = None
        self.finger_action = D.GRIPPER_OPEN_ACTION
        self.gripper_target = "unknown"
        self.home_pose: list[float] | None = None
        self.last_safe_pose: list[float] | None = None
        self.robot_geometry_verification: dict[str, Any] | None = None
        self.motion_planner: Any = None
        self.last_observation_root: Path | None = None
        self.last_observation_id: str | None = None
        self.last_observation_clock: int | None = None
        self.world_input_cache: tuple[str, dict[str, str]] | None = None
        self.control_record: ControlRecord | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> dict[str, Any]:
        if self.bridge is not None:
            raise RobolabError("RoboLab episode already started")
        from .bridge import build_environment

        self.bridge, self.source = build_environment(self.config, self.output_root)
        self.control_record = ControlRecord(self.output_root, identity={
            "task": self.task, "seed": self.seed, "action_codec": self.config.get("action_codec", "motion_action_v3"),
            "source_commit": (self.source or {}).get("commit")})
        reset = self._reset(round_index=0, seed=self.seed)
        shared_motion.start(self)
        job = self.config.get("replay_program_job")
        if job is not None:  # a replay (environment/robolab_replay.py): the program runs here, then the worker closes
            from ora.environment.replay_program_worker import run_program

            run_program(json.loads(Path(job).read_text(encoding="utf-8")), session=self)
        witness = self.config.get("contract_witness_job")
        if witness is not None:  # a task-contract witness (orchestrator/backends/robolab/contract_witness.py)
            from .contract_witness import run_witness

            run_witness(json.loads(Path(witness).read_text(encoding="utf-8")), session=self)
        return reset

    def _reset(self, *, round_index: int, seed: int) -> dict[str, Any]:
        state = self.bridge.reset(seed)
        if state["frozen"] or state["episode_steps"] != 0:
            raise RobolabError("RoboLab reset did not produce a fresh episode")
        self.round_index, self.native_actions = round_index, 0
        self.terminated = self.truncated = False
        self.first_success = None
        self.official_limit_snapshot = self.success_after_official_limit = None
        self.subtask_progress = self.subtask_progress_at_official_limit = None
        self.finger_action, self.gripper_target = D.GRIPPER_OPEN_ACTION, "unknown"
        # Two held steps refresh the articulation buffers before any measurement.
        hold = self.bridge.tcp_pose()
        for _ in range(2):
            self.after_native_action(self.bridge.step(hold, self.finger_action))
        self.bridge.take_ticks()  # a replay runs the same reset (control_record.py)
        if self.control_record is not None:
            self.control_record.begin_round(round_index)
        self.home_pose = self.bridge.tcp_pose()
        self.last_safe_pose = None
        self._verify_loaded_geometry()
        reset = {"schema": RESET_SCHEMA, "task": self.task, "seed": seed, "round_index": round_index,
                 "task_language": (self.source or {}).get("instruction"),
                 "native_action_limit": self.native_action_limit, "control_frequency_hz": D.CONTROL_FREQ_HZ,
                 "tcp_pose_xyz_xyzw": self.home_pose, "official_success_at_reset": bool(self.terminated),
                 "created_unix_s": time.time()}
        if reset["official_success_at_reset"]:
            raise RobolabError("RoboLab scene is already solved at reset")
        write_json(self.output_root / "round-resets" / f"round-{round_index:04d}.json", reset)
        return reset

    def restart_round(self, round_index: int, *, seed: int | None = None) -> dict[str, Any]:
        if self.bridge is None:
            raise RobolabError("RoboLab episode is not started")
        return self._reset(round_index=round_index, seed=self.seed if seed is None else int(seed))

    def _verify_loaded_geometry(self) -> None:
        observed = {"right_gripper": self.bridge.contact_geometry()}
        write_json(self.output_root / f"robot-geometry/round-{self.round_index}-observed.json",
                   {"observed_geometry": observed, "finger_qpos_rad": self.bridge.finger_qpos(),
                    "jaw_separation_m": self.bridge.jaw_separation_m()})
        binding = self.config.get("public_robot_geometry")
        if binding is None:
            return
        from ora.contracts.robot_geometry import verify_loaded_robot_geometry

        try:
            result = verify_loaded_robot_geometry(binding, observed)
        except (ValueError, TypeError) as exc:
            write_json(self.output_root / f"robot-geometry/round-{self.round_index}-mismatch.json",
                       {"expected_binding": dict(binding), "observed_geometry": observed, "error": str(exc)})
            raise RobolabError("loaded Robotiq geometry differs from the public descriptor") from exc
        self.robot_geometry_verification = {**result, "round_index": self.round_index}
        write_json(self.output_root / f"robot-geometry/round-{self.round_index}.json",
                   {**self.robot_geometry_verification, "expected_binding": dict(binding), "observed_geometry": observed})

    # -------------------------------------------------------------- control
    def after_native_action(self, result: Mapping[str, Any]) -> None:
        self.native_actions += 1
        if result.get("subtask_progress") is not None:
            self.subtask_progress = {**result["subtask_progress"], "native_actions": self.native_actions}
        if self.native_actions == self.native_action_limit:
            self.subtask_progress_at_official_limit = self.subtask_progress
        past_official_limit = self.official_limit_multiple > 1 and self.native_actions > self.native_action_limit
        if result.get("official_result") and past_official_limit and not self.terminated:
            # The official result was settled at the limit and a later success never changes it.
            self.success_after_official_limit = self.native_actions
        elif result.get("official_result") and not self.terminated:
            self.terminated = True
            self.first_success = {"schema": "ora.robolab-first-success.v1", "source": "RoboLab termination success",
                                  "round_index": self.round_index, "action_sequence": self.action_sequence,
                                  "native_action": self.native_actions}
            write_json(self.output_root / "rounds" / f"round-{self.round_index:04d}" / "first-success.json",
                       self.first_success)
        if self.official_limit_snapshot is None and self.native_actions >= self.native_action_limit and not self.terminated:
            self.official_limit_snapshot = limit_snapshot(self.native_action_limit)
        if (result.get("truncated") or self.native_actions >= self.extended_native_action_limit
                or self.success_after_official_limit is not None):
            self.truncated = True

    def must_stop(self) -> bool:
        return self.terminated or self.truncated or self.abort_reason is not None

    def arm_state(self) -> dict[str, Any]:
        return {"right": {"pose_xyz_xyzw": self.bridge.tcp_pose(), "gripper": self._opening(),
                          "arm_joint_qpos": self.bridge.arm_qpos()}}

    def _opening(self) -> float:
        return float(min(1.0, max(0.0, self.bridge.jaw_separation_m() / D.MAXIMUM_APERTURE_M)))

    def workspace_refusal(self, pose: list[float]) -> str | None:
        """Why a servo move's target cannot be reached: the shared IK, or the reach radius when it has no answer."""
        reachable = shared_motion.reachable(self, pose)
        if reachable is None:
            reachable = float(np.linalg.norm(np.asarray(pose[:3]))) <= D.MAX_REACH_FROM_ROOT_M
        return None if reachable else "target_out_of_reach"

    def execute(self, action: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_runtime_action(action)
        if validated["schema"] != "ora-motion-action.v3":
            raise RobolabError("RoboLab accepts motion_action_v3 proposals only")
        if self.must_stop():
            raise RobolabError("RoboLab episode has ended; no further actions")
        receipt = stamped(self.motion_planner, execute_motion_v3(self, validated))
        path = self.output_root / "receipts" / f"action-{self.action_sequence:04d}.json"
        write_json(path, receipt)
        self.last_receipt = {"path": path.relative_to(self.output_root).as_posix(), "receipt": receipt,
                             "sha256": file_hash(path)}
        if self.control_record is not None:
            self.control_record.action(self.action_sequence, validated, self.bridge.take_ticks(), receipt)
        self.action_sequence += 1
        return receipt

    def scene_trace(self, directory: Path, run_id: str) -> Any:
        """A replay program's scene trace of this session (``scene_trace.py``)."""
        from .scene_trace import RobolabSceneTrace

        return RobolabSceneTrace(directory, run_id=run_id)

    # ----------------------------------------------------------- observation
    def observe(self, destination: Path) -> tuple[dict[str, Any], list[Path]]:
        from PIL import Image

        observation_id = (f"robolab:{self.task}:seed-{self.seed}:round-{self.round_index}:"
                          f"observation-{self.observation_sequence:04d}")
        target = Path(destination).resolve() / observation_id
        if target.exists():
            raise RobolabError(f"refusing to overwrite observation: {target}")
        target.mkdir(parents=True)
        clock = self.bridge.clock()
        cameras, images = {}, []
        for public, data in self.bridge.cameras().items():
            root = target / public
            root.mkdir()
            rgb = np.asarray(data["rgb"]).astype(np.uint8)
            depth = np.asarray(data["depth"], dtype=np.float32)
            depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
            rgb_path, depth_path, calibration_path = root / "rgb.png", root / "depth-m.npy", root / "calibration.json"
            Image.fromarray(rgb).save(rgb_path)
            np.save(depth_path, depth, allow_pickle=False)
            cam2world = np.asarray(data["cam2world_cv"])
            write_json(calibration_path, {
                "source_camera": data["source_camera"], "intrinsic_cv": data["intrinsic_cv"],
                "extrinsic_cv": np.linalg.inv(cam2world), "cam2world_cv": cam2world, "depth_unit": "metres",
                "depth_array_order": "opencv_top_left_row_major", "policy_quaternion_order": "xyzw",
                "mount": "end_effector" if public == "wrist" else "fixed_world",
            })
            files = {role: {"path": path.relative_to(target).as_posix(), "sha256": file_hash(path),
                            "bytes": path.stat().st_size}
                     for role, path in (("rgb_lossless_png", rgb_path), ("depth_numpy_m_f32", depth_path),
                                        ("calibration", calibration_path))}
            cameras[public] = {"source_camera": data["source_camera"], "width": int(rgb.shape[1]),
                               "height": int(rgb.shape[0]), "files": files}
            images.append(rgb_path)
        if self.bridge.clock() != clock:
            raise RobolabError("RoboLab observation advanced physics")
        proprioception = {"end_effectors": {"right": {
            "pose_xyz_xyzw": self.bridge.tcp_pose(), "pose_frame": "robotiq_2f85_tcp",
            "gripper": self._opening(), "jaw_separation_m": self.bridge.jaw_separation_m(),
            "gripper_finger_qpos": [self.bridge.finger_qpos()],
            "gripper_command": 1.0 - float(self.finger_action), "gripper_command_target": self.gripper_target,
        }}, "single_arm": True}
        write_json(target / "proprioception.json", proprioception)
        payload = {
            "schema": OBSERVATION_SCHEMA, "observation_id": observation_id, "task": self.task,
            "sequence": self.observation_sequence, "encoding": "metric_rgbd_v1", "captured_unix_s": time.time(),
            "main_camera": MAIN_CAMERA, "cameras": cameras, "proprioception": proprioception,
            **({"robot_geometry_verification": self.robot_geometry_verification}
               if self.robot_geometry_verification is not None else {}),
            "native_actions_used": self.native_actions, "native_action_limit": self.extended_native_action_limit,
            # The policy's copy: a paced move's timing evidence stays in the sealed receipt.
            "previous_execution_receipt": policy_receipt(self.last_receipt),
            "forbidden_modalities_absent": list(FORBIDDEN_MODALITIES),
        }
        write_json(target / "observation.json", payload)
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            path.chmod(0o444)
        self.observation_sequence += 1
        self.last_observation_root, self.last_observation_id = target, observation_id
        self.last_observation_clock = clock
        return payload, images

    # ------------------------------------------------------- status/verdict
    def status(self) -> dict[str, Any]:
        reason = (self.abort_reason or ("official_success" if self.terminated else None)
                  or ("robolab_native_timeout" if self.truncated else None))
        terminal = reason is not None
        return validate_environment_status({
            "schema": ENVIRONMENT_STATUS_SCHEMA, "phase": "terminal" if terminal else "running",
            "terminal": terminal, "reason": reason, "source": "robolab_session",
            "policy_visibility": "control_only", "signals": [],
        })

    @property
    def is_terminal(self) -> bool:
        return bool(self.status()["terminal"])

    def abort(self, reason: str) -> None:
        if not reason:
            raise RobolabError("abort reason must be non-empty")
        self.abort_reason = reason

    def persist_official_verdict(self) -> dict[str, Any]:
        verdict = {"schema": VERDICT_SCHEMA, "official_success": bool(self.terminated), "task": self.task,
                   "seed": self.seed, "round_index": self.round_index, "terminated": self.terminated,
                   "truncated": self.truncated, "abort_reason": self.abort_reason, "first_success": self.first_success,
                   "native_actions": self.native_actions, "native_action_limit": self.native_action_limit,
                   "official_predicate": TASKS[self.task]["official_success"],
                   "source": "RoboLab termination success", "created_unix_s": time.time(),
                   **official_limit_verdict_fields(
                       official_limit=self.native_action_limit, multiple=self.official_limit_multiple,
                       snapshot=self.official_limit_snapshot,
                       success_native_actions=self.success_after_official_limit),
                   "robolab_score": self.robolab_score()}
        write_json(self.output_root / "official-verdict.json", verdict)
        return verdict

    def robolab_score(self) -> dict[str, Any]:
        """RoboLab's graded Score of the episode, as its leaderboard computes it; never shown to the policy.

        RoboLab's SubtaskCompletionRecorder scores the subtasks done (0 to 1) and its results code counts an
        official success as 1.0. ``official_score`` is that rule within the official limit: 1.0 on official
        success, else the score read on the native action that reached the limit, or the last one read when
        the episode ended before it. A task RoboLab tracks no subtasks for has no score (null), as upstream.
        """

        within = self.subtask_progress_at_official_limit or self.subtask_progress
        official = 1.0 if self.terminated else (float(within["score"]) if within is not None else None)
        return {"schema": SCORE_SCHEMA, "source": "RoboLab SubtaskCompletionRecorder (get_current_subtask_info)",
                "official_score": official, "final": self.subtask_progress,
                "at_official_limit": self.subtask_progress_at_official_limit}

    def finalize_operator_video(self) -> dict[str, Any] | None:
        return None

    def close(self) -> None:
        """Finish the evidence; in a worker the process then ends with ``os._exit`` rather than closing Kit."""
        shared_motion.close(self)
        bridge, self.bridge = self.bridge, None
        if bridge is None:
            return
        if self.exit_process_on_close:
            atexit.register(hard_exit)
            threading.Thread(target=hard_exit, args=(CLOSE_EXIT_GRACE_S,), daemon=True).start()
            return
        bridge.close()
