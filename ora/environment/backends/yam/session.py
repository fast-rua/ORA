"""Privilege-clean episode writer for the bimanual I2RT YAM station.

The policy sees metric RGB-D from the station's cameras with per-observation
calibration, both TCPs with their measured jaw apertures and joint angles, and
receipts. A real cell has no automatic evaluator, so the official verdict is null:
whether the task was done is the operator's call, recorded beside the run.

The operator sees the cameras too: ``capture.py`` records their original pixels
(``native/execution-capture``) for the world view and ``ora web``'s camera panel,
never for the policy.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ora.contracts.action import validate_runtime_action
from ora.contracts.policy_receipt import policy_receipt
from ora.environment.base import ENVIRONMENT_STATUS_SCHEMA, validate_environment_status
from ora.environment.controllers import yam_station as Y
from ora.contracts.action_plan import is_request

from .common import (
    FORBIDDEN_MODALITIES, OBSERVATION_SCHEMA, RESET_SCHEMA, RIG_MODES, VERDICT_SCHEMA, YamError, file_hash,
    write_json,
)
from .kinematics import pose_xyz_xyzw
from . import shared_motion, shared_planner
from .action_plan import plan_action
from .motion import execute_motion_v3

MAIN_CAMERA = "front"
DEFAULT_CONTROL_TICK_LIMIT = Y.CONTROL_FREQ_HZ * 900      # 15 minutes of motion per episode
TERMINAL_GUARD_CODES = frozenset({"motor_overheat", "motor_fault", "bus_lost", "operator_stop"})


class YamSession:
    """One trusted YAM episode, simulated or on the real arms."""

    def __init__(self, *, config: Mapping[str, Any], output_root: Path) -> None:
        mode = config.get("rig_mode", "sim")
        if mode not in RIG_MODES:
            raise YamError(f"rig_mode must be one of {RIG_MODES}")
        instruction = config.get("task_instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise YamError("a YAM episode needs a non-empty task_instruction")
        self.config = dict(config)
        self.mode = mode
        self.task_instruction = instruction.strip()
        self.task = str(config.get("task", "operator_task"))
        self.seed = int(config.get("seed", 0))
        self.control_tick_limit = int(config.get("control_tick_limit", DEFAULT_CONTROL_TICK_LIMIT))
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.rig: Any = None
        self.kinematics: Any = None
        self.round_index = 0
        self.observation_sequence = 0
        self.action_sequence = 0
        self.control_ticks = 0
        self.truncated = False
        self.abort_reason: str | None = None
        self.guard_terminal: str | None = None
        self.last_receipt: dict[str, Any] | None = None
        self.commanded_q: dict[str, np.ndarray] = {}
        self.finger_targets: dict[str, float] = {}
        #: Each gripper's last commanded direction ("close" or "open"); its finger target may stop short of the end.
        self.gripper_goals: dict[str, str] = {}
        self.home_q: dict[str, np.ndarray] = {}
        self.last_safe_q: dict[str, np.ndarray] | None = None
        self.capture: Any = None
        self.kept_schedules: dict[str, Any] = {}
        self.ran_previewed_plan = False
        self.robot_geometry_verification: dict[str, Any] | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> dict[str, Any]:
        if self.rig is not None:
            raise YamError("YAM episode already started")
        from .rig import build_rig

        if self.mode != "sim":
            self._start_auto_approve()
        self.rig = build_rig(self.config)
        self.kinematics = self.rig.kinematics
        report = getattr(self.rig, "startup_report", None)
        if report is not None:
            write_json(self.output_root / "rig-startup.json", report)
        self._start_capture()
        # runtime.motion_timing: the timing worker this run's preflight started times the arm paths.
        shared_motion.start(self)
        # runtime.motion_planner: the shared cuRobo V2 planner plans the arm paths, checked against the model first.
        shared_planner.start(self)
        return self._reset(round_index=0)

    def _start_capture(self) -> None:
        """The world view's recorder (``runtime.execution_capture``); a recorder that cannot start is noted."""

        from .capture import YamCapture, rgb_images

        config = self.config.get("execution_capture")
        if config is None or not YamCapture.enabled(config):
            return
        rig = self.rig
        render = getattr(rig, "camera_rgb", None)
        read = render if callable(render) else (lambda: rgb_images(rig.cameras()))
        try:
            self.capture = YamCapture(self.output_root, task=self.task, rig_mode=self.mode, config=config,
                                      read_cameras=read, clock=lambda: self.control_ticks)
        except Exception as exc:  # noqa: BLE001 - the episode runs without the world view's frames
            write_json(self.output_root / "execution-capture-failure.json", {
                "schema": "ora.auxiliary-media-failure.v1", "component": "original_execution_capture",
                "error": f"{type(exc).__name__}: {exc}", "recorded_unix_s": time.time()})

    def _start_auto_approve(self) -> None:
        """The cell's auto-approve switch at this worker's start: on for its own motions when the run
        was started with it (``runtime.auto_approve_motions``, ``ora play --auto-approve``), else
        cleared, so no switch carries over from another run. Enabling and releasing are always asked."""

        from ora.cell.errors import CellError
        from ora.cell.operator import approval

        kinds = ("motion",) if self.config.get("auto_approve_motions") is True else ()
        try:
            approval.start_run(auto_kinds=kinds, by="ora play --auto-approve", run=str(self.output_root))
        except CellError as exc:
            raise YamError(str(exc)) from None

    def _reset(self, *, round_index: int) -> dict[str, Any]:
        self.round_index = round_index
        self.commanded_q = {arm: self.rig.arm_q(arm) for arm in Y.ARMS}
        self.finger_targets = {arm: self.rig.finger_m(arm) for arm in Y.ARMS}
        self.gripper_goals = {}
        self.home_q = {arm: q.copy() for arm, q in self.commanded_q.items()}
        self.last_safe_q = None
        self.kinematics.set_state(self.commanded_q, self.finger_state())
        self._verify_loaded_geometry()
        reset = {"schema": RESET_SCHEMA, "task": self.task, "task_language": self.task_instruction,
                 "rig_mode": self.mode, "seed": self.seed, "round_index": round_index,
                 "control_frequency_hz": Y.CONTROL_FREQ_HZ, "control_tick_limit": self.control_tick_limit,
                 "position_frame": Y.POSITION_FRAME,
                 "tcp_pose_xyz_xyzw": {arm: pose_xyz_xyzw(self.kinematics.tcp(arm)) for arm in Y.ARMS},
                 "table_top_z_m": self.rig.table_top_z_m, "official_success_at_reset": False,
                 "created_unix_s": time.time()}
        write_json(self.output_root / "round-resets" / f"round-{round_index:04d}.json", reset)
        if self.capture is not None:
            self.capture.reset(round_index)
        return reset

    def _verify_loaded_geometry(self) -> None:
        """Check the loaded station model's finger geometry against the bound public descriptor.

        Present only when the surface delivers public robot facts (``runtime.public_geometry_descriptor``).
        The real cell runs the same station model, so both rigs verify the geometry they plan with.
        """

        binding = self.config.get("public_robot_geometry")
        if binding is None:
            return
        from ora.contracts.robot_geometry import verify_loaded_robot_geometry

        observed = self.kinematics.contact_geometry()
        try:
            result = verify_loaded_robot_geometry(binding, observed)
        except (ValueError, TypeError) as exc:
            write_json(self.output_root / f"robot-geometry/round-{self.round_index}-mismatch.json",
                       {"expected_binding": dict(binding), "observed_geometry": observed, "error": str(exc)})
            raise YamError("loaded YAM finger geometry differs from the public descriptor") from exc
        self.robot_geometry_verification = {**result, "round_index": self.round_index}
        write_json(self.output_root / f"robot-geometry/round-{self.round_index}.json",
                   {**self.robot_geometry_verification, "expected_binding": dict(binding),
                    "observed_geometry": observed})

    def restart_round(self, round_index: int, *, seed: int | None = None) -> dict[str, Any]:
        if self.rig is None:
            raise YamError("YAM episode is not started")
        if seed is not None:
            self.seed = int(seed)
        if self.capture is not None:
            self.capture.terminal()
        home = getattr(self.rig, "home", None)
        if home is not None:            # a new round starts with the arms at rest
            home()
        return self._reset(round_index=round_index)

    # -------------------------------------------------------------- state
    def finger_state(self) -> dict[str, float]:
        return {arm: self.rig.finger_m(arm) for arm in Y.ARMS}

    def aperture_m(self, arm: str) -> float:
        return 2.0 * self.rig.finger_m(arm)

    def opening(self, arm: str) -> float:
        return float(min(1.0, max(0.0, self.rig.finger_m(arm) / Y.FINGER_OPEN_M)))

    def arm_state(self) -> dict[str, Any]:
        self.kinematics.set_state({arm: self.rig.arm_q(arm) for arm in Y.ARMS}, self.finger_state())
        return {arm: {"pose_xyz_xyzw": pose_xyz_xyzw(self.kinematics.tcp(arm)), "gripper": self.opening(arm),
                      "jaw_separation_m": self.aperture_m(arm), "arm_joint_qpos": self.rig.arm_q(arm).tolist()}
                for arm in Y.ARMS}

    def on_control_tick(self) -> None:
        """After each control tick of a motion (motion.py): the world view's frame when one is due."""

        if self.capture is not None:
            self.capture.control_tick()

    def on_guard_events(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            if event.get("code") in TERMINAL_GUARD_CODES:
                self.guard_terminal = str(event["code"])
        # A stopped rig holds where the arms are, not where the plan had them.
        self.commanded_q = {arm: self.rig.arm_q(arm) for arm in Y.ARMS}
        write_json(self.output_root / "guard-events" / f"action-{self.action_sequence:04d}.json", events)

    def must_stop(self) -> bool:
        if self.control_ticks >= self.control_tick_limit:
            self.truncated = True
        return self.truncated or self.abort_reason is not None or self.guard_terminal is not None

    # -------------------------------------------------------------- control
    def align_plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """The read-only bridge: a dry run's plan of one action (``action_plan.py``), and nothing else."""
        if not is_request(request):
            raise YamError("the YAM station serves only action plans on the read-only bridge")
        return plan_action(self, request["action"])

    def execute(self, action: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_runtime_action(action)
        if validated["schema"] != "ora-motion-action.v3":
            raise YamError("the YAM station accepts motion_action_v3 proposals only")
        if self.must_stop():
            raise YamError("YAM episode has ended; no further actions")
        if self.capture is not None:
            self.capture.action_start(validated, self.action_sequence)
        self.ran_previewed_plan = False
        try:
            receipt = execute_motion_v3(self, validated)
        except BaseException:
            if self.capture is not None:
                self.capture.action_failed()
            raise
        if self.ran_previewed_plan:
            receipt["ran_previewed_plan"] = True
        if self.capture is not None:
            self.capture.action_end()
        path = self.output_root / "receipts" / f"action-{self.action_sequence:04d}.json"
        write_json(path, receipt)
        self.last_receipt = {"path": path.relative_to(self.output_root).as_posix(), "receipt": receipt,
                             "sha256": file_hash(path)}
        self.action_sequence += 1
        return receipt

    # ----------------------------------------------------------- observation
    def observe(self, destination: Path) -> tuple[dict[str, Any], list[Path]]:
        from PIL import Image

        observation_id = (f"yam:{self.task}:{self.mode}:round-{self.round_index}:"
                          f"observation-{self.observation_sequence:04d}")
        target = Path(destination).resolve() / observation_id
        if target.exists():
            raise YamError(f"refusing to overwrite observation: {target}")
        target.mkdir(parents=True)
        cameras, images, pixels = {}, [], {}
        for public, data in self.rig.cameras().items():
            root = target / public
            root.mkdir()
            rgb = np.asarray(data["rgb"]).astype(np.uint8)
            depth = np.asarray(data["depth"], dtype=np.float32)
            depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
            rgb_path, depth_path, calibration_path = root / "rgb.png", root / "depth-m.npy", root / "calibration.json"
            Image.fromarray(rgb).save(rgb_path)
            np.save(depth_path, depth, allow_pickle=False)
            cam2world = data.get("cam2world_cv")
            calibration = {"source_camera": data["source_camera"], "intrinsic_cv": data["intrinsic_cv"],
                           "depth_unit": "metres", "depth_array_order": "opencv_top_left_row_major",
                           "policy_quaternion_order": "xyzw", "mount": data["mount"],
                           "calibration_source": data.get("calibration_source")}
            calibration.update({key: data[key] for key in ("depth_valid_fraction", "depth_pipeline_restarted",
                                                           "depth_below_floor", "distortion") if key in data})
            if cam2world is not None:
                cam2world = np.asarray(cam2world, dtype=float)
                calibration.update({"extrinsic_cv": np.linalg.inv(cam2world), "cam2world_cv": cam2world})
            write_json(calibration_path, calibration)
            files = {role: {"path": path.relative_to(target).as_posix(), "sha256": file_hash(path),
                            "bytes": path.stat().st_size}
                     for role, path in (("rgb_lossless_png", rgb_path), ("depth_numpy_m_f32", depth_path),
                                        ("calibration", calibration_path))}
            cameras[public] = {"source_camera": data["source_camera"], "width": int(rgb.shape[1]),
                               "height": int(rgb.shape[0]), "calibrated": cam2world is not None, "files": files}
            images.append(rgb_path)
            pixels[public] = rgb
        state = self.arm_state()
        proprioception = {"end_effectors": {arm: {
            "pose_xyz_xyzw": state[arm]["pose_xyz_xyzw"], "pose_frame": "yam_linear_4310_tcp",
            "gripper": state[arm]["gripper"], "jaw_separation_m": state[arm]["jaw_separation_m"],
            # The same measurement under the name the toolkit's jaw readers take (RoboTwin's).
            "gripper_opening_m": state[arm]["jaw_separation_m"],
            "gripper_finger_qpos": [self.rig.finger_m(arm)],
            "gripper_command_target": self.gripper_goals.get(arm) or (
                "close" if self.finger_targets.get(arm, 1.0) <= Y.FINGER_CLOSED_M else "open"),
            "arm_joint_qpos": state[arm]["arm_joint_qpos"],
        } for arm in Y.ARMS}, "single_arm": False, "position_frame": Y.POSITION_FRAME}
        write_json(target / "proprioception.json", proprioception)
        payload = {
            "schema": OBSERVATION_SCHEMA, "observation_id": observation_id, "task": self.task,
            "sequence": self.observation_sequence, "encoding": "metric_rgbd_v1", "captured_unix_s": time.time(),
            "main_camera": MAIN_CAMERA, "cameras": cameras, "proprioception": proprioception,
            "rig_mode": self.mode, "table_top_z_m": self.rig.table_top_z_m,
            "native_actions_used": self.control_ticks, "native_action_limit": self.control_tick_limit,
            # The policy's copy: a timed move's timing evidence stays in the sealed receipt.
            "previous_execution_receipt": policy_receipt(self.last_receipt),
            "forbidden_modalities_absent": list(FORBIDDEN_MODALITIES),
            **({"robot_geometry_verification": self.robot_geometry_verification}
               if self.robot_geometry_verification is not None else {}),
        }
        write_json(target / "observation.json", payload)
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            path.chmod(0o444)
        self.observation_sequence += 1
        if self.capture is not None:
            self.capture.observation(pixels)
        return payload, images

    # ------------------------------------------------------- status/verdict
    def status(self) -> dict[str, Any]:
        self.must_stop()
        reason = (self.abort_reason or self.guard_terminal
                  or ("yam_control_budget_exhausted" if self.truncated else None))
        terminal = reason is not None
        return validate_environment_status({
            "schema": ENVIRONMENT_STATUS_SCHEMA, "phase": "terminal" if terminal else "running",
            "terminal": terminal, "reason": reason, "source": "yam_session",
            "policy_visibility": "control_only", "signals": [],
        })

    @property
    def is_terminal(self) -> bool:
        return bool(self.status()["terminal"])

    def abort(self, reason: str) -> None:
        if not reason:
            raise YamError("abort reason must be non-empty")
        self.abort_reason = reason

    def persist_official_verdict(self) -> dict[str, Any]:
        verdict = {"schema": VERDICT_SCHEMA, "official_success": None, "task": self.task,
                   "task_language": self.task_instruction, "rig_mode": self.mode, "round_index": self.round_index,
                   "truncated": self.truncated, "abort_reason": self.abort_reason,
                   "guard_terminal": self.guard_terminal, "control_ticks": self.control_ticks,
                   "control_tick_limit": self.control_tick_limit,
                   "source": "no automatic evaluator on the YAM station; the operator judges the outcome",
                   "created_unix_s": time.time()}
        if self.capture is not None:
            self.capture.terminal()
        write_json(self.output_root / "official-verdict.json", verdict)
        return verdict

    def finalize_operator_video(self) -> dict[str, Any] | None:
        self._close_capture()
        return None

    def _close_capture(self) -> None:
        capture, self.capture = self.capture, None
        if capture is not None:
            capture.close()

    def close(self) -> None:
        self._close_capture()
        shared_motion.close(self)
        shared_planner.close(self)
        rig, self.rig = self.rig, None
        if rig is not None:
            rig.close()
