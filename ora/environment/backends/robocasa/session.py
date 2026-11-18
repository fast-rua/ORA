"""Privilege-clean RoboCasa365 episode writer for the PandaOmron.

The policy sees metric RGB-D from two public cameras with per-observation
calibration, grip-site and base proprioception, and receipts. Object ground
truth, fixture identities, the task class and its success predicate stay in
this worker; only ``_check_success`` (RoboCasa's own official predicate)
decides success, latched at the first control step it holds within the
task's horizon.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.contracts.action import validate_runtime_action
from ora.contracts.official_limit import limit_snapshot, official_limit_multiple
from ora.contracts.official_limit import verdict_fields as official_limit_verdict_fields
from ora.contracts.policy_receipt import policy_receipt
from ora.environment.base import ENVIRONMENT_STATUS_SCHEMA, validate_environment_status
from ora.environment.benchmarks.robocasa import (
    IMAGE_SIZE,
    MAIN_CAMERA,
    PUBLIC_CAMERAS,
    TASKS,
    validate_runtime_extension,
)
from ora.environment.controller_replay import ControllerReplayRecorder
from ora.environment.controllers import base_placement, jaw_flip_retry, motion_timing, robocasa_omron as C
from ora.contracts.action_plan import is_request
from ora.environment.controllers.kept_plans import stamped
from ora.environment.controllers.motion_pace import PaceRefusal

from . import shared_motion
from .action_plan import plan_action
from .common import (
    FORBIDDEN_MODALITIES,
    OBSERVATION_SCHEMA,
    REGION_ORDER_SCHEMA,
    RESET_SCHEMA,
    VERDICT_SCHEMA,
    RoboCasaError,
    _atomic_json,
    _file_hash,
    _quaternion_error_rotvec,
)
from .construction import build_environment, egl_device, region_order, source_manifest
from .motion import _paced_plan, execute_motion_v3
from .robot import OmronIO, StepResult

#: The controller replay's label: one full action vector per control step, sent to the composite controller as is.
REPLAY_CONTROLLER = "robocasa_composite_action"


#: What a refused lift height's record keeps of its refusal (``_lift_plan``): where and why the path failed.
_LIFT_REFUSAL_KEYS = ("path_kind", "cause", "joint", "jump_rad", "bound_rad", "path_fraction", "robot_part",
                      "penetration_m", "planner_reason")


class RoboCasaSession:
    """One trusted, isolated RoboCasa365 episode."""

    def __init__(self, *, config: Mapping[str, Any], output_root: Path) -> None:
        runtime = {
            "task": config.get("task"),
            "seed": config.get("seed"),
            "split": config.get("split", "target"),
            "action_codec": config.get("action_codec", "motion_action_v3"),
        }
        if config.get("required_revisions") is not None:
            runtime["required_revisions"] = config["required_revisions"]
        extension = validate_runtime_extension(runtime)
        self.task: str = extension["task"]
        self.seed: int = extension["seed"]
        self.action_codec: str = extension["action_codec"]
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.expected_task_language = str(config.get("expected_task_language") or "").strip()
        self.public_robot_geometry: Mapping[str, Any] | None = config.get("public_robot_geometry")
        self.robot_geometry_verification: dict[str, Any] | None = None
        self.environment_factory = config.get("environment_factory") or build_environment
        self.planner_config: Mapping[str, Any] | None = config.get("motion_planner")
        self.motion_planner: Any = None
        self.world_inputs: dict[tuple[int, int, bool], dict[str, str]] = {}
        #: This moment's cameras, public points, stand worlds and base searches (``shared_motion._moment``).
        self.stand_cache: dict[str, Any] = {}
        self.environment: Any = None
        self.io: OmronIO | None = None
        self.horizon: int = 0
        self.official_limit_multiple = official_limit_multiple(config)
        self.official_limit_snapshot: dict[str, Any] | None = None
        self.success_after_official_limit: int | None = None
        self.physics_steps_per_tick = int(round((1.0 / C.CONTROL_FREQ_HZ) / 0.002))
        self.round_index = 0
        self.observation_sequence = 0
        self.action_sequence = 0
        self.terminated = False
        self.truncated = False
        self.first_success: dict[str, Any] | None = None
        self.abort_reason: str | None = None
        self.last_receipt: dict[str, Any] | None = None
        self.gripper_target = "unknown"
        self.home_pose: list[float] | None = None
        self.home_arm_qpos: list[float] | None = None
        self.last_safe_arm_qpos: list[float] | None = None
        self.language: str | None = None
        self.source: dict[str, Any] | None = None
        self.world_mutation_sequence = 0
        #: A replay's region orders by round (``construction.region_order``); a run's are its own.
        self.recorded_regions: Mapping[int, Sequence[Sequence[str]]] = config.get("region_order") or {}
        self.replay_recorder = ControllerReplayRecorder(
            output_root=self.output_root, backend="robocasa",
            environment_identity={"task": self.task, "seed": self.seed, "split": extension["split"],
                                  "action_codec": self.action_codec},
            camera_panels=list(PUBLIC_CAMERAS),
        )

    # ------------------------------------------------------------ lifecycle
    def start(self) -> dict[str, Any]:
        if self.environment is not None:
            raise RoboCasaError("RoboCasa episode already started")
        if self.environment_factory is build_environment:
            self.source = source_manifest()
        motion_timing.bind_preflight(self.output_root)
        reset = self._reset(seed=self.seed, round_index=0)
        shared_motion.start(self)
        return reset

    def _reset(self, *, seed: int, round_index: int) -> dict[str, Any]:
        with region_order(self.recorded_regions.get(round_index)) as regions:
            environment = self.environment_factory(task=self.task, seed=seed)
            environment.reset()
        self.environment = environment
        self.io = OmronIO(environment)
        self.horizon = int(environment.horizon)
        self.official_limit_snapshot = self.success_after_official_limit = None
        language = environment.get_ep_meta().get("lang")
        if not isinstance(language, str) or not language.strip():
            raise RoboCasaError("RoboCasa episode has no task language")
        self.language = language.strip()
        if self.expected_task_language and self.expected_task_language != self.language:
            # Each seed samples its own scene; the language is bound per episode.
            raise RoboCasaError(
                "RoboCasa task language differs from the bound language: "
                f"{self.language!r} != {self.expected_task_language!r}"
            )
        self.round_index = round_index
        self.replay_recorder.begin_round(round_index)
        _atomic_json(self.replay_recorder.root / f"region-order-round-{round_index:04d}.json",
                     {"schema": REGION_ORDER_SCHEMA, "round_index": round_index, "regions": regions})
        self.terminated = self.truncated = False
        self.first_success = None
        self.gripper_target = "unknown"
        self.home_pose = self.io.grip_pose()
        self.home_arm_qpos = self.io.arm_qpos()
        self.last_safe_arm_qpos = None
        self._verify_loaded_geometry()
        reset = {
            "schema": RESET_SCHEMA,
            "task": self.task,
            "seed": seed,
            "round_index": round_index,
            "task_language": self.language,
            "horizon_control_steps": self.horizon,
            "control_frequency_hz": C.CONTROL_FREQ_HZ,
            "base_pose_xy_yaw": self.io.base_pose(),
            "source": self.source,
            "egl_device": egl_device(),
            "official_success_at_reset": bool(environment._check_success()),
            "created_unix_s": time.time(),
        }
        if reset["official_success_at_reset"]:
            raise RoboCasaError("RoboCasa scene is already solved at reset")
        _atomic_json(self.output_root / "round-resets" / f"round-{round_index:04d}.json", reset)
        return reset

    def _verify_loaded_geometry(self) -> None:
        """Compare the loaded Panda fingers with the public descriptor before any policy turn."""

        assert self.io is not None
        self.robot_geometry_verification = None
        observed = {"right_gripper": self.io.public_gripper_geometry()}
        _atomic_json(self.output_root / f"robot-geometry/round-{self.round_index}-observed.json",
                     {"observed_geometry": observed, "finger_qpos_m": self.io.finger_qpos()})
        if self.public_robot_geometry is None:
            return
        from ora.contracts.robot_geometry import verify_loaded_robot_geometry

        try:
            result = verify_loaded_robot_geometry(self.public_robot_geometry, observed)
        except (ValueError, TypeError) as exc:
            _atomic_json(self.output_root / f"robot-geometry/round-{self.round_index}-mismatch.json",
                         {"expected_binding": dict(self.public_robot_geometry), "observed_geometry": observed,
                          "error": str(exc)})
            raise RoboCasaError("loaded Panda geometry differs from the public descriptor") from exc
        self.robot_geometry_verification = {**result, "round_index": self.round_index}
        _atomic_json(self.output_root / f"robot-geometry/round-{self.round_index}.json",
                     {**self.robot_geometry_verification, "expected_binding": dict(self.public_robot_geometry),
                      "observed_geometry": observed})

    def restart_round(self, round_index: int, *, seed: int | None = None) -> dict[str, Any]:
        if self.environment is None:
            raise RoboCasaError("RoboCasa episode is not started")
        self._close_environment()
        return self._reset(seed=self.seed if seed is None else int(seed), round_index=round_index)

    def _require_io(self) -> OmronIO:
        if self.io is None:
            raise RoboCasaError("RoboCasa episode is not started")
        return self.io

    # -------------------------------------------------------------- control
    def after_control_step(self, result: StepResult) -> None:
        io = self._require_io()
        # A success the predicate reports after the horizon is never the official one.
        if result.official_success and not self.terminated and io.control_steps <= self.horizon:
            self.terminated = True
            self.first_success = {
                "schema": "ora.robocasa-first-success.v1",
                "source": "RoboCasa _check_success",
                "round_index": self.round_index,
                "action_sequence": self.action_sequence,
                "control_step": io.control_steps,
            }
            _atomic_json(
                self.output_root / "rounds" / f"round-{self.round_index:04d}" / "first-success.json",
                self.first_success,
            )
        elif result.official_success and self.official_limit_multiple > 1 and not self.terminated:
            self.success_after_official_limit = io.control_steps
        if self.official_limit_snapshot is None and io.control_steps >= self.horizon and not self.terminated:
            self.official_limit_snapshot = limit_snapshot(self.horizon)
        if io.control_steps >= self.extended_horizon or self.success_after_official_limit is not None:
            self.truncated = True

    @property
    def extended_horizon(self) -> int:
        """Where the episode stops; ``horizon`` stays the official limit."""
        return self.horizon * self.official_limit_multiple

    def must_stop(self) -> bool:
        return self.terminated or self.truncated or self.abort_reason is not None

    def arm_state(self) -> dict[str, Any]:
        io = self._require_io()
        return {"right": {
            "pose_xyz_xyzw": io.grip_pose(),
            "gripper": io.gripper_opening(),
            "arm_joint_qpos": io.arm_qpos(),
        }}

    def remember_safe_joints(self) -> None:
        """The arm joints at the start of an admitted move_pose (recover_motion previous)."""

        self.last_safe_arm_qpos = self._require_io().arm_qpos()

    def _reach(self, pose: list[float]) -> tuple[float, float]:
        """The target's distance from the arm mount at the nearest mount height the lift allows, and that height."""
        import numpy as np

        io = self._require_io()
        # The torso lift carries the arm mount vertically through its range, so
        # reach is measured from the nearest mount height the lift allows.
        mount = np.asarray(io.arm_base_position(), dtype=np.float64)
        torso = io.torso_height()
        low, high = mount[2] - (torso - C.TORSO_RANGE_M[0]), mount[2] + (C.TORSO_RANGE_M[1] - torso)
        nearest = np.array([mount[0], mount[1], min(max(float(pose[2]), low), high)])
        return float(np.linalg.norm(np.asarray(pose[:3]) - nearest)), float(nearest[2])

    def workspace_refusal(self, pose: list[float]) -> str | None:
        if self._reach(pose)[0] > C.ARM_MAX_REACH_FROM_SHOULDER_M:
            return "target_out_of_reach_from_base"
        if not 0.0 <= pose[2] <= 2.2:
            return "target_outside_kitchen_height"
        return None

    def reach_detail(self, pose: list[float], code: str, interaction: str,
                     speed: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """What an admission refusal of ``code`` (``workspace_refusal``) says: for a target out of reach, how far it
        is from the arm mount and the bound, and where the base could stand to reach it with this move
        (``_suggested``)."""
        if code != "target_out_of_reach_from_base":
            return {}
        distance, height = self._reach(pose)
        detail = {"path_kind": "reach", "distance_from_mount_m": round(distance, 4),
                  "reach_m": C.ARM_MAX_REACH_FROM_SHOULDER_M, "nearest_mount_height_m": round(height, 4)}
        return self._suggested({"refusal": detail, "code": code}, pose, self._kind(pose, interaction), interaction,
                               speed)["refusal"]

    def _suggested(self, plan: dict[str, Any], pose: list[float], kind: str, interaction: str,
                   speed: Mapping[str, Any] | None) -> dict[str, Any]:
        """``plan``, refused with a code a base move may answer, with the base poses from which this move is admitted
        (``shared_motion.base_search``, ``_admitted_from``). A line a base move does not mend gets none: one whose
        target the arm reaches from here fails on its path, and one whose gripper enters the world meets what lies
        around the target, which stays where it is."""
        refusal = plan.get("refusal")
        if (plan.get("code") not in base_placement.SEARCHED_CODES or not isinstance(refusal, Mapping)
                or refusal.get("endpoint") == shared_motion.ENDPOINT_REACHABLE
                or (refusal.get("path_kind") == "line" and refusal.get("robot_part") == "gripper")):
            return plan
        found = shared_motion.base_search(
            self, pose, (kind, interaction, speed),
            lambda stand: self._admitted_from(stand, pose, kind, interaction, speed))
        if found is None:
            return plan
        return {**plan, "refusal": {**refusal, "base_suggestions": found["suggestions"],
                                    "base_search": found["searched"]}}

    def _admitted_from(self, stand: Any, pose: list[float], kind: str, interaction: str,
                       speed: Mapping[str, Any] | None) -> bool:
        """Whether the move to ``pose`` is admitted from ``stand`` as its real call there would admit it: the same path
        (the line for a contact approach or a small precision turn, else the global route) in the world seen from
        there, then its pacing at ``speed`` with the wrist check and its jaw twin. The search's questions, so none of
        them is kept for a dry run."""
        plan = shared_motion.arm_path(self, pose, kind, interaction, stand, kept=False)
        return plan["refusal"] is None and not isinstance(
            self._paced_from(plan, pose, kind, interaction, speed, stand, kept=False), str)

    def base_path_refusal(self, target: Mapping[str, float]) -> dict[str, Any] | None:
        """``move_base``'s footprint check (``shared_motion.base_path_refusal``), or ``None`` when the move is clear
        or farther than the servo's travel bound, which ``motion.move_base`` refuses on its own."""
        start = self._require_io().base_pose()
        if math.hypot(float(target["x_m"]) - start[0], float(target["y_m"]) - start[1]) > C.BASE_MAX_TRAVEL_M:
            return None
        return shared_motion.base_path_refusal(self, target)

    def arm_path_plan(self, pose: list[float], interaction: str, *, allow_torso: bool = True,
                      speed: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Read-only admission of one arm motion, on the shared motion base (``shared_motion.py``).

        free_space, and a precision move that turns the hand more than
        ``C.JOINT_PATH_MIN_ROTATION_RAD``, take the planner's collision-free
        joint path; precision and contact_approach otherwise follow the straight
        grip-site line, checked in the public depth world. When the arm alone
        cannot reach the target at the measured torso height (and
        ``allow_torso``), the lift is admitted at the nearest height from which
        the whole move is (``_lift_plan``): the path from up there, in the world
        seen from there, and its pacing at ``speed``. The real call lifts, then
        plans again at the height it reached. A line whose target the arm
        reaches from here fails on its path, which a lift does not mend: it is
        refused. A refusal a base move may answer carries the base poses the arm
        reaches the target from (``_suggested``). Returns ``{"refusal": None,
        "path": [...], "poses": [...], "torso_height_m": None | height, "kind":
        ...}`` or the refusal with its code (``target_unreachable_at_base_pose``,
        ``target_in_collision_at_base_pose``, ``path_in_collision_at_base_pose``
        or ``motion_planner_unavailable``). ``allow_torso`` False is the arm
        alone at the measured height, with neither the lift nor a search. No
        live writes.
        """

        kind = self._kind(pose, interaction)
        first = shared_motion.arm_path(self, pose, kind, interaction, explain=allow_torso)
        if first["refusal"] is None or not allow_torso:
            return first
        if (first["code"] == "target_unreachable_at_base_pose"
                and first["refusal"].get("endpoint") != shared_motion.ENDPOINT_REACHABLE):
            lifted, tried = self._lift_plan(pose, kind, interaction, speed)
            if lifted is not None:
                return lifted
            # The receipt says what the lift's admission asked and why each height was refused (evidence only: the
            # policy's projection, refusal_detail.py, does not carry it).
            first = {**first, "refusal": {**first["refusal"], "torso_admission": tried}}
        return self._suggested(first, pose, kind, interaction, speed)

    def _kind(self, pose: list[float], interaction: str) -> str:
        """The path an arm move takes: the straight line for a contact approach and for a precision move that turns
        the hand at most ``C.JOINT_PATH_MIN_ROTATION_RAD``, else the planner's joint path."""
        turn = math.sqrt(sum(float(v) ** 2 for v in _quaternion_error_rotvec(
            pose[3:7], self._require_io().grip_pose()[3:7])))
        return ("line" if interaction == "contact_approach"
                or (interaction == "precision" and turn <= C.JOINT_PATH_MIN_ROTATION_RAD) else "joint")

    def _lift_plan(self, pose: list[float], kind: str, interaction: str,
                   speed: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """The move admitted at the nearest torso height from which all of it is, or ``None``, and what was tried.

        At most ``C.TORSO_ADMIT_HEIGHTS`` of the heights ``shared_motion.torso_heights`` offers are tried: from each the
        path is admitted as at the measured height, in the world seen from up there, and then paced at ``speed`` with
        the wrist check, its jaw twin included as the real call would run it (``_paced_from``). The record names the
        measured height, the heights the arm reaches the target from, and each tried height's refusal: its code and
        the detail that says where (``cause``, ``joint``, ``robot_part``, ``path_fraction``, ``penetration_m``...)."""
        measured = self._require_io().torso_height()
        heights = shared_motion.torso_heights(self, pose)
        tried: list[dict[str, Any]] = []
        for height in heights[:C.TORSO_ADMIT_HEIGHTS]:
            stand = shared_motion.lifted(self, height)
            plan = shared_motion.arm_path(self, pose, kind, interaction, stand)
            if plan["refusal"] is not None:
                tried.append({"torso_height_m": round(float(height), 4), "code": plan["code"],
                              **{key: plan["refusal"][key] for key in _LIFT_REFUSAL_KEYS if key in plan["refusal"]}})
                continue
            paced = self._paced_from(plan, pose, kind, interaction, speed, stand)
            if not isinstance(paced, str):
                return {**plan, "torso_height_m": height, "lift_m": height - measured, "lift_paced": paced}, {}
            tried.append({"torso_height_m": round(float(height), 4), "code": paced, "stage": "pacing"})
        return None, {"measured_m": round(float(measured), 4), "reaching_m": [round(float(h), 4) for h in heights],
                      "tried": tried}

    def _paced_from(self, plan: Mapping[str, Any], pose: list[float], kind: str, interaction: str,
                    speed: Mapping[str, Any] | None, stand: Any,
                    kept: bool = True) -> tuple[Any, dict[str, Any] | None] | str:
        """``plan`` (admitted from ``stand``) paced as the real call paces it, with its jaw twin where the wrist check
        refuses it and the jaws are open; ``(paced, jaw_flip_retry record or None)``, or the pacing's refusal code.
        ``kept`` False asks the twin past what a dry run keeps (a base search's question).

        Pacing reads the grip site's motion between joint rows, which the lift or the base moves as a whole: the
        measured kinematics time it as they would from there."""
        try:
            return _paced_plan(self, plan, interaction, speed), None
        except PaceRefusal as refusal:
            def admit(goal: list[float]) -> Any:
                again = shared_motion.arm_path(self, goal, kind, interaction, stand, kept=kept)
                return str(again["code"]) if again["refusal"] is not None else _paced_plan(self, again, interaction,
                                                                                            speed)

            retried = jaw_flip_retry.retry(pose, refusal, admit,
                                           jaws_open=lambda: self.io.gripper_directive == C.GRIPPER_OPEN_COMMAND)
            if retried is None:
                return str(refusal.code)
            _goal, paced, record = retried
            return jaw_flip_retry.recorded(paced, record), record

    def joint_target_plan(self, joints: list[float]) -> dict[str, Any]:
        """Read-only admission of a joint-space move to ``joints`` (recovery), torso held."""

        return shared_motion.joint_path(self, joints)

    def align_plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """The read-only bridge: a dry run's plan of one action (``action_plan.py``), and nothing else."""
        self._require_io()
        if not is_request(request):
            raise RoboCasaError("RoboCasa serves only action plans on the read-only bridge")
        return plan_action(self, request["action"])

    def execute(self, action: Mapping[str, Any]) -> dict[str, Any]:
        self._require_io()
        validated = validate_runtime_action(action)
        if validated["schema"] != "ora-motion-action.v3":
            raise RoboCasaError("RoboCasa accepts motion_action_v3 proposals only")
        if self.must_stop():
            raise RoboCasaError("RoboCasa episode has ended; no further actions")
        state_before = self._require_io().state_sha256()
        receipt = stamped(self.motion_planner, execute_motion_v3(self, validated))
        self.world_mutation_sequence += 1
        self._record_controls(validated, receipt, state_before)
        path = self.output_root / "receipts" / f"action-{self.action_sequence:04d}.json"
        _atomic_json(path, receipt)
        self.last_receipt = {"path": path.relative_to(self.output_root).as_posix(),
                             "receipt": receipt, "sha256": _file_hash(path)}
        self.action_sequence += 1
        return receipt

    def _record_controls(self, action: Mapping[str, Any], receipt: Mapping[str, Any], state_before: str) -> None:
        """Keep what the action sent, for its replay program; the state hashes prove a re-execution scene and all."""

        import numpy as np

        io = self._require_io()
        commands = io.take_commands()
        self.replay_recorder.record(
            action_sequence=self.action_sequence, action=action, controller=REPLAY_CONTROLLER,
            arrays={"commands": np.stack(commands)} if commands else {},
            physics_steps=len(commands), replayable=bool(commands),
            metadata={"state_sha256": {"before": state_before, "after": io.state_sha256()}},
        )
        self.replay_recorder.update(self.action_sequence, execution_status=str(receipt["execution_status"]))

    def probe_motion(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._require_io()
        return {
            "schema": "ora.motion-probe-result.v1",
            "observation_id": request.get("observation_id"),
            "world_mutation_sequence": self.world_mutation_sequence,
            "variant_policy": request.get("variant_policy"),
            "robot_decisions_consumed": 0,
            "world_state_changed": False,
            "results": shared_motion.probe(self, request),
        }

    # ----------------------------------------------------------- observation
    def _camera(self, source: str) -> tuple[Any, Any, dict[str, Any]]:
        import numpy as np
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
            get_real_depth_map,
        )

        sim = self.environment.sim
        rgb, depth = sim.render(width=IMAGE_SIZE, height=IMAGE_SIZE, camera_name=source, depth=True)
        rgb = np.ascontiguousarray(np.asarray(rgb)[::-1]).astype(np.uint8)
        depth_m = np.ascontiguousarray(
            np.asarray(get_real_depth_map(sim, depth), dtype=np.float32)[::-1]
        )
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
        extent = float(sim.model.stat.extent)
        meta = {
            "intrinsic": np.asarray(get_camera_intrinsic_matrix(sim, source, IMAGE_SIZE, IMAGE_SIZE)),
            "cam2world": np.asarray(get_camera_extrinsic_matrix(sim, source)),
            "near": float(sim.model.vis.map.znear) * extent,
            "far": float(sim.model.vis.map.zfar) * extent,
        }
        return rgb, depth_m, meta

    def observe(self, destination: Path) -> tuple[dict[str, Any], list[Path]]:
        import numpy as np
        from PIL import Image

        io = self._require_io()
        observation_id = (
            f"robocasa:{self.task}:seed-{self.seed}:round-{self.round_index}:"
            f"observation-{self.observation_sequence:04d}"
        )
        target = Path(destination).resolve() / observation_id
        if target.exists():
            raise RoboCasaError(f"refusing to overwrite observation: {target}")
        target.mkdir(parents=True)
        cameras: dict[str, Any] = {}
        images: list[Path] = []
        for public_name, source in PUBLIC_CAMERAS.items():
            rgb, depth, meta = self._camera(source)
            if depth.shape != rgb.shape[:2]:
                raise RoboCasaError(f"RGB-D shape mismatch: {source}")
            root = target / public_name
            root.mkdir()
            rgb_path, depth_path, calibration_path = root / "rgb.png", root / "depth-m.npy", root / "calibration.json"
            Image.fromarray(rgb).save(rgb_path)
            np.save(depth_path, depth, allow_pickle=False)
            _atomic_json(calibration_path, {
                "source_camera": source,
                "intrinsic_cv": meta["intrinsic"],
                "extrinsic_cv": np.linalg.inv(meta["cam2world"]),
                "cam2world_cv": meta["cam2world"],
                "depth_unit": "metres",
                "valid_depth_range_m": {"near": meta["near"], "far": meta["far"]},
                "depth_array_order": "opencv_top_left_row_major",
                "policy_quaternion_order": "xyzw",
                "mount": "eye_in_hand" if public_name == "wrist" else "torso_support_moves_with_base",
            })
            files = {}
            for role, path in (("rgb_lossless_png", rgb_path), ("depth_numpy_m_f32", depth_path),
                               ("calibration", calibration_path)):
                files[role] = {"path": path.relative_to(target).as_posix(),
                               "sha256": _file_hash(path), "bytes": path.stat().st_size}
            cameras[public_name] = {"source_camera": source, "width": int(rgb.shape[1]),
                                    "height": int(rgb.shape[0]), "files": files}
            images.append(rgb_path)
        command = io.gripper_directive
        proprioception = {
            "end_effectors": {"right": {
                "pose_xyz_xyzw": io.grip_pose(),
                "pose_frame": "robosuite_grip_site",
                "gripper": io.gripper_opening(),
                "gripper_finger_qpos": io.finger_qpos(),
                "gripper_command": (1.0 - float(command)) / 2.0,
                "gripper_command_target": self.gripper_target,
            }},
            "base": {"pose_xy_yaw": io.base_pose(), "frame": "robocasa_world",
                     "torso_height_m": io.torso_height()},
            "single_arm": True,
        }
        _atomic_json(target / "proprioception.json", proprioception)
        payload = {
            "schema": OBSERVATION_SCHEMA,
            "observation_id": observation_id,
            "task": self.task,
            "sequence": self.observation_sequence,
            "encoding": "metric_rgbd_v1",
            "captured_unix_s": time.time(),
            "main_camera": MAIN_CAMERA,
            "cameras": cameras,
            "proprioception": proprioception,
            **({"robot_geometry_verification": self.robot_geometry_verification}
               if self.robot_geometry_verification is not None else {}),
            "control_steps_used": io.control_steps,
            "control_steps_budget": self.extended_horizon,
            # The policy's copy: a paced move's timing evidence stays in the sealed receipt.
            "previous_execution_receipt": policy_receipt(self.last_receipt),
            "forbidden_modalities_absent": list(FORBIDDEN_MODALITIES),
        }
        _atomic_json(target / "observation.json", payload)
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            path.chmod(0o444)
        self.observation_sequence += 1
        return payload, images

    # ------------------------------------------------------- status/verdict
    def status(self) -> dict[str, Any]:
        reason = (
            self.abort_reason
            or ("official_success" if self.terminated else None)
            or ("robocasa_horizon_exhausted" if self.truncated else None)
        )
        terminal = reason is not None
        return validate_environment_status({
            "schema": ENVIRONMENT_STATUS_SCHEMA,
            "phase": "terminal" if terminal else "running",
            "terminal": terminal,
            "reason": reason,
            "source": "robocasa_session",
            "policy_visibility": "control_only",
            "signals": [],
        })

    @property
    def is_terminal(self) -> bool:
        return bool(self.status()["terminal"])

    def abort(self, reason: str) -> None:
        if not reason:
            raise RoboCasaError("abort reason must be non-empty")
        self.abort_reason = reason

    def persist_official_verdict(self) -> dict[str, Any]:
        verdict = {
            "schema": VERDICT_SCHEMA,
            "official_success": bool(self.terminated),
            "task": self.task,
            "seed": self.seed,
            "round_index": self.round_index,
            "task_language": self.language,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "abort_reason": self.abort_reason,
            "first_success": self.first_success,
            "control_steps": self.io.control_steps if self.io is not None else 0,
            "horizon_control_steps": self.horizon,
            "official_predicate": TASKS[self.task]["official_success"],
            "source": "RoboCasa _check_success",
            "created_unix_s": time.time(),
            **official_limit_verdict_fields(
                official_limit=self.horizon, multiple=self.official_limit_multiple,
                snapshot=self.official_limit_snapshot, success_native_actions=self.success_after_official_limit,
            ),
        }
        _atomic_json(self.output_root / "official-verdict.json", verdict)
        return verdict

    def finalize_operator_video(self) -> dict[str, Any] | None:
        return None

    def _close_environment(self) -> None:
        environment, self.environment, self.io = self.environment, None, None
        if environment is not None:
            close = getattr(environment, "close", None)
            if callable(close):
                close()

    def close(self) -> None:
        shared_motion.close(self)
        motion_timing.close()
        self._close_environment()
