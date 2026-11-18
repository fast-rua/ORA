"""The reward-free RoboDojo TaskEnv lifecycle of a play scene (worker only).

Upstream task classes pair ``TaskEnv`` with a ``RewardManager``, a step limit
and an instruction generator; the benchmark worker's ``EvaluationLifecycle``
(``backends/robodojo/bridge.py``) drives them and latches the official
success and failure. A play scene has none of those, so its class is
``PlaySceneLifecycle`` over upstream ``TaskEnv`` itself:

- ``__init__`` builds the observation manager as the benchmark does but no
  ``SeedManager`` (it reads a task's official ``Eval_Layout`` files);
- ``reset(seed=[s])`` loads the generated layout of seed ``s``
  (``harness_play_layouts``, set by the play bootstrap) through
  ``LayoutManager.set_saved_layout``, runs upstream's reset and the
  benchmark's camera/texture setup without the evaluator wiring, holds
  ``harness_settle_steps`` unrendered physics steps so the scene is at rest
  before the first observation, and only then runs upstream's layout
  stability check (the benchmark runs it before its 200 steps; a leaning
  object of a play scene may still turn then, not after the settle);
- ``advance_official_terminal`` is always ``None`` and ``native_tick`` runs the
  benchmark's fixed-cadence control without the reward step: nothing a task
  would count ends a play scene, only the step limit ``step_lim`` does;
  ``native_tick_sequence`` (a target per physics step: the rate-limited jaws, a
  smooth joint trajectory; the benchmark's ``tick_sequence.py``) likewise;
- ``relaunch_for_layout`` (inherited) closes the simulation and stage inside
  the same Kit process and resets on another seed's layout: the only way a
  play session changes scene, on CPU and CUDA physics alike. After it the
  reset measures the camera annotators' clock offset on the settled scene
  (``_calibrate_epoch_clock``): on CPU physics the closing context counts a
  few more steps than the arithmetic offset holds.

- the operator's third-person camera (``operator_view.py``; never a policy
  camera) is attached after each reset and released before a relaunch closes
  the stage; ``operator_rgb`` reads its latest frame without rendering.

- every dynamic body of the scene's objects (rigid objects and articulation
  links, never the robot or the table) gets ``PLAY_RIGID_BODY_LIMITS`` when it
  spawns (``_post_setup_scene``, and each later ``spawn_scene_objects``): an
  object the arm sweeps or the jaws squeeze tips over or slides away instead
  of being launched by the solver. The reset's scene registration records
  the limits and how many bodies carry them.

- ``hold_steps``, ``object_motion`` and ``object_poses`` serve the play
  session's post-action settle (``PLAY_POST_ACTION_SETTLE``, ``session.py``):
  physics steps with every robot target held (no control, no native action),
  each followed by the execution capture's callback as in ``native_tick``,
  and read-only readings of how fast the scene's objects move and where they
  are (``displaced`` compares two readings). Physics advances only
  inside an action, so without the settle an object the arm knocked over
  stood frozen in mid-fall until the next action. The reset's scene
  registration records the settle's bounds.

Every other inherited method (camera product release, layout identity
checks, ``get_obs``) is the benchmark's, unchanged. Isaac imports stay inside
methods, as in ``bridge.py``.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from ..bridge import EvaluationLifecycle, RobodojoError, _fixed_meta_control
from . import operator_view

PLAY_SCENE_REGISTRATION_SCHEMA = "ora.robodojo-play-scene-registration.v1"
EPOCH_CLOCK_SCHEMA = "ora.robodojo-play-camera-clock.v1"
#: PhysX limits on each dynamic body of a play scene's objects (``physxRigidBody:*``, stage metres). PhysX's own
#: defaults are 3 m/s depenetration, no linear limit and 5729.58 deg/s angular: an object the arm drove into or
#: the jaws squeezed was pushed out at up to 100 m/s and flew off the table. The linear limit stays above the
#: arm's fastest free-space motion (about 2.1 m/s measured) so a carried object keeps up with the jaws, and above a
#: short drop's speed; the angular one above an object toppling over.
PLAY_RIGID_BODY_LIMITS = {
    "max_depenetration_velocity_m_s": 0.5,
    "max_linear_velocity_m_s": 3.0,
    "max_angular_velocity_deg_s": 720.0,
}
RIGID_BODY_LIMITS_SCHEMA = "ora.robodojo-play-rigid-body-limits.v1"
#: After each action the world keeps running, every robot target held, until the scene's objects are at rest
#: or ``max_physics_steps`` (2.0 s of simulated time at the 4 ms step) have passed. Physics advances only inside
#: an action: measured on ``toys`` seed 3, four objects a sweep set moving (0.1 to 0.6 m/s) kept those speeds and
#: poses through 3 s of wall time and an observation, and came to rest 0.5 s of simulated time after the sweep
#: once stepped. The scene is at rest at once when no rigid object moves faster than ``rest_linear_speed_m_s``
#: or turns faster than ``rest_angular_speed_rad_s`` and no articulation joint moves faster than
#: ``rest_joint_speed``; otherwise it steps ``window_physics_steps`` at a time until, over one window, no object
#: moved more than ``window_position_change_m`` or turned more than ``window_rotation_change_deg`` and no joint
#: moved more than ``window_joint_change``. The window rule is what ends a settle: an object's velocity can
#: jitter at rest (``breakfast``'s grapes read 0.012 m/s and 0.33 rad/s while the play-scene audit holds them
#: within 5 mm and 2 degrees over 2 s), its pose does not. The robot's own motion does not hold the settle open.
PLAY_POST_ACTION_SETTLE = {
    "max_physics_steps": 500,
    "window_physics_steps": 10,
    "rest_linear_speed_m_s": 0.01,
    "rest_angular_speed_rad_s": 0.1,
    "rest_joint_speed": 0.1,
    "window_position_change_m": 0.001,
    "window_rotation_change_deg": 1.0,
    "window_joint_change": 0.005,
}
POST_ACTION_SETTLE_CONFIG_SCHEMA = "ora.robodojo-play-post-action-settle-config.v1"
#: How far the camera annotators' clock may sit from the arithmetic epoch offset after a relaunch.
#: On CPU physics the first relaunch of the audit measured 0.076 s (19 physics steps) more.
EPOCH_CLOCK_TOLERANCE_S = 0.25


class PlaySceneLifecycle(EvaluationLifecycle):
    """``EvaluationLifecycle`` without a task: no reward, no official terminal, generated layouts."""

    #: Set by the play bootstrap on the composed class.
    harness_play_layouts: Mapping[int, Mapping[str, Any]] = {}
    harness_settle_steps: int = 0
    step_lim: int = 0
    #: ``operator_view.OPERATOR_VIEW_MODES``; ``off`` attaches no operator camera.
    harness_operator_view_mode: str = operator_view.DEFAULT_OPERATOR_VIEW_MODE
    #: What ``limit_rigid_bodies`` last applied (the simulation's setup runs inside ``__init__``).
    harness_rigid_body_limits: dict[str, Any] | None = None

    def __init__(self, config: Any, app: Any):
        # TaskEnv.__init__ directly: EvaluationLifecycle.__init__ also builds the
        # SeedManager, which reads the official layout files of a task.
        super(EvaluationLifecycle, self).__init__(config, app)
        from env.observation_manager.obs_manager import ObsManager

        self.eval_cfg = config.eval_cfg
        self.task_name = self.eval_cfg.task_name
        self.eval_seed = 0
        self.success = [True] * self.num_envs
        self.end_flag = [False] * self.num_envs
        self.take_action_cnt = [0] * self.num_envs
        self.unstable_envs: set[int] = set()
        self.obs_manager = ObsManager(
            obs_config=deepcopy(self.eval_cfg.observation), num_envs=self.num_envs,
            dt=self.dt, task_name=self.task_name, description_cfg={"seen": 1},
            seeds_per_env=self.env_seed_list,
        )
        self.scene_manager.layout_manager.replay = True
        self._limit_rigid_bodies_on_spawn()
        #: A play scene registers no reward, now or ever.
        self.reward_registered = False
        self.scene_registration: dict[str, Any] | None = None
        self._official_terminal: dict[str, Any] | None = None

    def reset(self, seed: Any = None, options: Any = None) -> None:
        layouts = tuple(getattr(self, "harness_layout_ids", ()))
        if (not isinstance(seed, list) or len(seed) != 1 or type(seed[0]) is not int
                or seed[0] not in layouts or seed[0] not in self.harness_play_layouts or self.num_envs != 1):
            raise RobodojoError("only an audited seed of the play scene can be loaded")
        live = getattr(self, "sim", None) is not None
        if live and seed[0] != getattr(self, "harness_loaded_layout_id", seed[0]):
            raise RobodojoError("another play seed needs relaunch_for_layout, not an in-place reset")
        capture_manager = getattr(self, "capture_manager", None)
        if capture_manager is not None and capture_manager.tiled_cameras:
            self._release_capture_products()
        if live:
            self._keep_layout_identities()
        self.success, self.end_flag, self.take_action_cnt = [True], [False], [0]
        self.unstable_envs = set()
        self.scene_registration = None
        self._official_terminal = None
        self.harness_epoch_clock = None
        self.env_seeds = list(seed)
        self.harness_loaded_layout_id = seed[0]
        self.scene_manager.layout_manager.set_saved_layout(0, deepcopy(dict(self.harness_play_layouts[seed[0]])))
        # TaskEnv.reset: scene reload, robot reset, 300 unrendered steps, cameras.
        super(EvaluationLifecycle, self).reset(seed=seed, options=options)
        instances = self._verify_scene_identities()
        self.obs_manager.reset()
        self.setup_scene()
        self.robot_manager.set_origin_endpose()
        self.robot_manager.set_robot_init_state()
        for _ in range(int(self.harness_settle_steps)):
            self.sim_step(render=False)
        # Upstream's own stability check, on the settled scene: something leaning (a pen in its
        # holder) may still turn before the settle, never after it.
        stable, unstable = self.scene_manager.layout_manager.check_layout_stability(self)
        if not stable or 0 in unstable:
            raise RobodojoError("generated play layout did not settle stably")
        self.success, self.end_flag = [True], [False]
        self.harness_epoch_clock = self._calibrate_epoch_clock()
        self.scene_registration = {
            "schema": PLAY_SCENE_REGISTRATION_SCHEMA, "instances": instances,
            "native_action_limit": int(self.step_lim), "evaluator_registration": None,
            "settle_steps": int(self.harness_settle_steps),
            "rigid_body_limits": deepcopy(self.harness_rigid_body_limits),
            "post_action_settle": {"schema": POST_ACTION_SETTLE_CONFIG_SCHEMA, **PLAY_POST_ACTION_SETTLE},
        }
        # The operator's camera, on the settled scene: it renders with the next observation.
        if self.harness_operator_view_mode != "off" and getattr(self, "harness_operator_view", None) is None:
            self.harness_operator_view = operator_view.attach(self.scene_manager.env_origins[0])

    def _limit_rigid_bodies_on_spawn(self) -> None:
        """Objects a reset spawns later (CPU physics respawns rigid objects on every reset) get the limits too."""
        manager = self.scene_manager
        spawn = manager.spawn_scene_objects

        def spawn_and_limit(env_id: int, exclude_types: Any = None) -> Any:
            result = spawn(env_id=env_id, exclude_types=exclude_types)
            self.harness_rigid_body_limits = limit_rigid_bodies(manager)
            return result

        manager.spawn_scene_objects = spawn_and_limit

    def _post_setup_scene(self, sim: Any) -> None:
        # The simulation's setup spawns the scene's objects (all of them on CUDA physics, the articulations on
        # CPU), at the start and after every relaunch.
        super()._post_setup_scene(sim)
        self.harness_rigid_body_limits = limit_rigid_bodies(self.scene_manager)

    def operator_rgb(self) -> Any:
        """The operator camera's latest frame (None before it has rendered or when off); never renders."""
        return operator_view.read_rgb(getattr(self, "harness_operator_view", None))

    def hold_steps(self, count: int) -> None:
        """``count`` physics steps with every robot target held: no control, no native action, no reward.

        The articulation drives keep the targets the last native action set (``native_tick``), so the
        arms and grippers hold while the scene's objects fall, roll or slide on. Each step is an
        execution step for the capture, which samples it as it samples ``native_tick``'s.
        """
        for _ in range(int(count)):
            self.sim_step(render=False)
            capture = getattr(self, "execution_capture_callback", None)
            if capture is not None:
                capture()
        if self.unstable_envs:
            raise RobodojoError("the play scene became unstable")

    def object_motion(self) -> dict[str, Any]:
        """How fast the scene's objects move now (read-only, no render, no physics step).

        Every rigid object's linear (m/s) and angular (rad/s) speed and every articulation's
        fastest joint (rad/s or m/s), and the objects above ``PLAY_POST_ACTION_SETTLE``'s rest
        speeds; the robot, the table and an object without a velocity reading are not counted.
        """
        manager = self.scene_manager
        linear = angular = joint = 0.0
        moving: list[str] = []
        limits = PLAY_POST_ACTION_SETTLE
        for name, obj in manager._rigid_and_dynamic_objects[0].items():
            if not (callable(getattr(obj, "get_linear_velocity", None))
                    and callable(getattr(obj, "get_angular_velocity", None))):
                continue
            speed, spin = _norm(obj.get_linear_velocity()), _norm(obj.get_angular_velocity())
            linear, angular = max(linear, speed), max(angular, spin)
            if speed > limits["rest_linear_speed_m_s"] or spin > limits["rest_angular_speed_rad_s"]:
                moving.append(str(name))
        for name, obj in manager._articulation_objects[0].items():
            reader = getattr(obj, "get_joint_velocities", None)
            values = _values(reader()) if callable(reader) else []
            fastest = max((abs(value) for value in values), default=0.0)
            joint = max(joint, fastest)
            if fastest > limits["rest_joint_speed"]:
                moving.append(str(name))
        return {"max_linear_speed_m_s": linear, "max_angular_speed_rad_s": angular,
                "max_joint_speed": joint, "moving": sorted(moving)}

    def object_poses(self) -> dict[str, dict[str, list[float]]]:
        """Every scene object's pose (rigid: position and wxyz quaternion) or joint positions (read-only)."""
        manager = self.scene_manager
        poses: dict[str, dict[str, list[float]]] = {}
        for name, obj in manager._rigid_and_dynamic_objects[0].items():
            reader = getattr(obj, "_get_object_transform", None)
            if callable(reader):
                position, quaternion, _ = reader()
                poses[str(name)] = {"position": _values(position)[:3], "quaternion": _values(quaternion)[:4]}
        for name, obj in manager._articulation_objects[0].items():
            reader = getattr(obj, "get_joint_positions", None)
            if callable(reader):
                poses[str(name)] = {"joints": _values(reader())}
        return poses

    def relaunch_for_layout(self, layout_id: int) -> dict[str, Any]:
        # The operator camera's render product belongs to the stage the relaunch closes. It is the
        # operator's alone: a failed release is recorded and never stops the policy's scene change.
        handle, self.harness_operator_view = getattr(self, "harness_operator_view", None), None
        try:
            operator_view.release(handle)
            self.harness_operator_view_release_error = None
        except Exception as exc:  # noqa: BLE001
            self.harness_operator_view_release_error = f"{type(exc).__name__}: {exc}"
        return super().relaunch_for_layout(layout_id)

    def _calibrate_epoch_clock(self) -> dict[str, Any] | None:
        """After a relaunch: measure the camera annotators' clock offset on the settled scene.

        The annotators keep counting across a relaunch while the new context's
        clock restarts, so a frame is current when its annotator time minus the
        epoch offset equals the context clock (``camera_capture.
        synchronize_render_products``). ``relaunch_for_layout`` sets the offset
        to the earlier epochs' simulated time; on CPU physics the annotators
        also count steps the closing context took after that reading. With
        physics standing still, this renders until every camera reports one
        unchanged time and takes that offset, refusing a difference above
        ``EPOCH_CLOCK_TOLERANCE_S``. The first epoch (offset 0) keeps its own.
        """
        arithmetic = getattr(self, "harness_simulation_time_offset_s", None)
        if not arithmetic:
            return None
        context = float(self.sim.sim.current_time)
        clock = int(self.sim.sim.current_time_step_index)
        readings: list[list[float]] = []
        for _ in range(12):
            self.render()
            readings.append([float(view._harness_simulation_time.get_data()["simulationTime"])
                             for view in self.capture_manager.tiled_cameras])
            if len(readings) >= 2 and readings[-1] == readings[-2] and len(set(readings[-1])) == 1:
                break
        if int(self.sim.sim.current_time_step_index) != clock or float(self.sim.sim.current_time) != context:
            raise RobodojoError("camera clock calibration advanced physics")
        if len(readings) < 2 or readings[-1] != readings[-2] or len(set(readings[-1])) != 1:
            raise RobodojoError("the cameras' clocks did not settle after the relaunch")
        measured = readings[-1][0] - context
        record = {"schema": EPOCH_CLOCK_SCHEMA, "arithmetic_offset_s": float(arithmetic),
                  "measured_offset_s": measured, "difference_s": measured - float(arithmetic),
                  "context_time_s": context, "annotator_time_s": readings[-1][0], "renders": len(readings),
                  "tolerance_s": EPOCH_CLOCK_TOLERANCE_S}
        if not 0.0 <= measured - float(arithmetic) <= EPOCH_CLOCK_TOLERANCE_S:
            raise RobodojoError(f"the cameras' clock sits {measured - float(arithmetic):.4f} s from the relaunch "
                                f"offset (tolerance {EPOCH_CLOCK_TOLERANCE_S} s)")
        self.harness_simulation_time_offset_s = measured
        return record

    def setup_scene(self) -> None:
        """The benchmark's camera, texture and layout-stability setup, without the evaluator input."""
        from ..camera_capture import (
            attach_camera_parameters, install_single_environment_readback, wait_for_render_assets,
        )
        from isaacsim.core.simulation_manager import SimulationManager
        import omni.usd

        context = omni.usd.get_context()
        attach_camera_parameters(self.capture_manager)
        round_index = int(getattr(self, "harness_round_index", 0))
        wait_for_render_assets(
            manager=self.capture_manager, render=self.render,
            assets_loading=SimulationManager.assets_loading,
            stage_status=lambda: {"loading": list(context.get_stage_loading_status()),
                                  "streaming": bool(context.get_stage_streaming_status())},
            physics_clock=lambda: int(self.sim.sim.current_time_step_index),
            output_root=self.harness_output_root,
            record_name=("render-readiness.jsonl" if round_index == 0
                         else f"render-readiness-round-{round_index:04d}.jsonl"),
        )
        install_single_environment_readback(self.capture_manager, self.harness_output_root)
        self.scene_manager.apply_saved_poses(env_idx_list=[0])
        if not self.scene_manager.layout_manager.layout_valid[0]:
            raise RobodojoError("generated play layout is invalid")
        for _ in range(10):
            self.render()
        for index in range(200):
            self.sim_step()
            if index % 5 == 0:
                self.render()
                self.obs_manager.get_obs()

    def advance_official_terminal(self) -> None:
        """A play scene never has an official terminal transition."""
        return None

    def native_tick(self, control: Mapping[str, Any]) -> None:
        """The benchmark's fixed-cadence native action, without the reward step or a terminal."""
        if self.take_action_cnt[0] >= self.step_lim:
            raise RobodojoError("the play scene's native action limit was reached")
        self.take_action_cnt[0] += 1
        sequence = self.interpolate(control)
        manager = self.robot_manager.control_manager
        manager.push([0], [sequence])
        for requested in sequence:
            previous = deepcopy(manager.prev_control[0])
            commands = manager.pop([0])
            commands[0], resolved = _fixed_meta_control(commands[0], self.robot_manager, previous, requested)
            manager.prev_control[0] = resolved
            super(EvaluationLifecycle, self).step(meta_control_list=commands)
            self.sim_step(render=False)
            capture = getattr(self, "execution_capture_callback", None)
            if capture is not None:
                capture()
            callback = getattr(self, "controller_trace_callback", None)
            if callback is not None:
                callback(deepcopy(manager.prev_control[0]))
        if not manager.get_empty([0]):
            raise RobodojoError("native control queue did not drain at the fixed cadence")
        if self.unstable_envs:
            raise RobodojoError("the play scene became unstable")
        return None

    def native_tick_sequence(self, requested: Any) -> None:
        """``tick_sequence.run_native_tick_sequence``'s native action (the 10 per-step targets ``requested`` holds,
        or gives step by step), without the reward step, the support arms or a terminal, as ``native_tick``."""
        if self.take_action_cnt[0] >= self.step_lim:
            raise RobodojoError("the play scene's native action limit was reached")
        count = int(self.obs_manager.collect_interval)
        if count != 10:
            raise RobodojoError("native control cadence must be 10 physics steps")
        if not callable(requested) and len(requested) != count:
            raise RobodojoError("a recorded native tick must hold exactly 10 physics-step targets")
        self.take_action_cnt[0] += 1
        manager = self.robot_manager.control_manager
        for index in range(count):
            target = deepcopy(dict(requested(index) if callable(requested) else requested[index]))
            manager.push([0], [[target]])
            previous = deepcopy(manager.prev_control[0])
            commands = manager.pop([0])
            commands[0], resolved = _fixed_meta_control(commands[0], self.robot_manager, previous, target)
            manager.prev_control[0] = resolved
            super(EvaluationLifecycle, self).step(meta_control_list=commands)
            self.sim_step(render=False)
            capture = getattr(self, "execution_capture_callback", None)
            if capture is not None:
                capture()
            callback = getattr(self, "controller_trace_callback", None)
            if callback is not None:
                callback(deepcopy(manager.prev_control[0]))
        if not manager.get_empty([0]):
            raise RobodojoError("native control queue did not drain at the fixed cadence")
        if self.unstable_envs:
            raise RobodojoError("the play scene became unstable")
        return None


def _values(value: Any) -> list[float]:
    """A tensor, array or sequence of numbers as a flat list of floats (empty for None)."""
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    return [float(item) for item in value]


def _norm(value: Any) -> float:
    return math.sqrt(sum(item * item for item in _values(value)[:3]))


def displaced(before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]],
              bounds: Mapping[str, float] = PLAY_POST_ACTION_SETTLE) -> list[str]:
    """The objects that moved more than the settle's window bounds between two ``object_poses`` readings."""
    moved = []
    for name, now in after.items():
        then = before.get(name)
        if then is None:
            continue
        if "joints" in now:
            change = max((abs(a - b) for a, b in zip(now["joints"], then["joints"])), default=0.0)
            if change > bounds["window_joint_change"]:
                moved.append(name)
            continue
        shift = math.dist(now["position"], then["position"])
        dot = min(1.0, abs(sum(a * b for a, b in zip(now["quaternion"], then["quaternion"]))))
        turn = math.degrees(2.0 * math.acos(dot))
        if shift > bounds["window_position_change_m"] or turn > bounds["window_rotation_change_deg"]:
            moved.append(name)
    return sorted(moved)


def limit_rigid_bodies(scene_manager: Any, limits: Mapping[str, float] = PLAY_RIGID_BODY_LIMITS) -> dict[str, Any]:
    """Set ``limits`` on every non-kinematic rigid body under the scene's objects; returns what was applied."""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    bodies = 0
    for collection in (scene_manager._rigid_and_dynamic_objects[0], scene_manager._articulation_objects[0]):
        for obj in collection.values():
            root = stage.GetPrimAtPath(str(obj.prim_path))
            if not root or not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                if UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
                    continue
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                api.CreateMaxDepenetrationVelocityAttr().Set(float(limits["max_depenetration_velocity_m_s"]))
                api.CreateMaxLinearVelocityAttr().Set(float(limits["max_linear_velocity_m_s"]))
                api.CreateMaxAngularVelocityAttr().Set(float(limits["max_angular_velocity_deg_s"]))
                bodies += 1
    return {"schema": RIGID_BODY_LIMITS_SCHEMA, **{key: float(value) for key, value in limits.items()},
            "bodies": bodies, "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage))}


def play_scene_class(task_env: type, *, layouts: Mapping[int, Mapping[str, Any]], settle_steps: int,
                     step_limit: int, operator_view_mode: str = operator_view.DEFAULT_OPERATOR_VIEW_MODE) -> type:
    """The worker's play-scene class: ``PlaySceneLifecycle`` before upstream ``TaskEnv``."""
    if operator_view_mode not in operator_view.OPERATOR_VIEW_MODES:
        raise RobodojoError(f"operator view mode must be one of {list(operator_view.OPERATOR_VIEW_MODES)}")
    return type("HarnessRoboDojoPlayScene", (PlaySceneLifecycle, task_env), {
        "harness_play_layouts": {int(seed): deepcopy(dict(layout)) for seed, layout in layouts.items()},
        "harness_settle_steps": int(settle_steps), "step_lim": int(step_limit),
        "harness_operator_view_mode": operator_view_mode,
    })


__all__ = ["EPOCH_CLOCK_SCHEMA", "EPOCH_CLOCK_TOLERANCE_S", "PLAY_POST_ACTION_SETTLE", "PLAY_RIGID_BODY_LIMITS",
           "PLAY_SCENE_REGISTRATION_SCHEMA", "POST_ACTION_SETTLE_CONFIG_SCHEMA", "PlaySceneLifecycle",
           "RIGID_BODY_LIMITS_SCHEMA", "displaced", "limit_rigid_bodies", "play_scene_class"]
