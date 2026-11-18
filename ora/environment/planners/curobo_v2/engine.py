"""The cuRobo V2 calls behind ``worker.Worker``; imported only inside the V2 interpreter.

One ``MotionPlanner`` (IK, trajectory optimisation and graph search sharing
one scene collision checker) and one ``RobotCollisionChecker`` on that same
checker and the same robot model, so a start or goal check, an IK query and a
plan all see the same world and the same held object.

The robot is its V2 file with ORA's changes (``robots.py``: for the Panda the
fingers where LIBERO's are, for the X5 the fingers added and locked open, the
v0.7.8 LIBERO planner's 4 mm sphere buffer on both) and ``held_sphere_slots``
sphere slots on ``attached_object`` for a held load. The settings are the v0.7.8 LIBERO planner's: 32 IK and 4
trajectory seeds, up to four attempts with graph seeding from the third,
0.02 s interpolation.

Every IK attempt's seed population is kept (``detail.ik_attempts``): each seed's
joints, errors, convergence and feasibility, and the candidate and rejection
counts, copied from the solver's own result extraction without another solve.

The world is one voxel layer in a fixed-size cache (``depth_world.curobo_v2_grid``),
so the CUDA graphs see stable buffers; each goal's grid is copied into it.
Held spheres are written into every robot model the solvers hold. V2's
``SceneCollision.update_voxel_data`` is not used: its arguments do not match
the data method it forwards to.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from ora.environment.controllers.depth_world import curobo_v2_grid, world_parameters
from ora.environment.controllers.motion_planner import ROBOTS

from . import robots

#: The v0.7.8 LIBERO planner's settings, applied to V2.
SETTINGS = {
    "num_ik_seeds": 32, "num_trajopt_seeds": 4, "max_attempts": 4, "enable_graph_attempt": 2,
    "interpolation_dt_s": 0.02, "position_tolerance_m": 0.005, "orientation_tolerance": 0.05,
    "optimizer_collision_activation_distance_m": 0.01, "use_cuda_graph": True, "random_seed": 123,
}
HELD_LINK = "attached_object"
HELD_SLOTS = 32
VOXEL_LAYER = "public_depth"
#: IK attempts kept per plan (a plan makes at most ``max_attempts``).
MAX_IK_ATTEMPTS = 8


def _finite(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


class CuroboV2Engine:
    """The planner calls ``worker.Worker`` makes, on cuRobo V2."""

    def __init__(self, *, device: str, robot: str, world_model: str, robot_file: str | None = None) -> None:
        import curobo.runtime
        import torch
        from curobo._src.types.robot import RobotCfg
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        from curobo.content import get_robot_configs_path
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg

        started = time.time()
        # Pose and joint-space plans have different problem shapes: switching
        # between them re-captures the solvers' CUDA graphs, which V2 refuses
        # unless graph reset is enabled (as the RoboDojo SDK enables it).
        curobo.runtime.cuda_graph_reset = True
        self.torch = torch
        self.device_cfg = DeviceCfg(device=torch.device(device))
        self.tool_frame = ROBOTS[robot]["tool_frame"]
        self.robot = robot
        source = (Path(robot_file) if robot in robots.ROBOT_FILE_ROBOTS
                  else get_robot_configs_path() / robots.BUNDLED_FILES[robot])
        self.robot_file = str(source)
        data = robots.robot_config(robot, robots.robot_file(robot, source), held_link=HELD_LINK, held_slots=HELD_SLOTS)
        self.robot_sha256 = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        self.collision_links = list(data["robot_cfg"]["kinematics"]["collision_link_names"])
        self.spheres_sha256 = robots.sphere_set_sha256(data["robot_cfg"]["kinematics"])
        self.robot_cfg = RobotCfg.create(data, self.device_cfg)
        parameters = world_parameters(world_model, robot)
        cache = None
        if parameters is not None:
            layout = curobo_v2_grid(np.zeros(parameters.grid_shape, dtype=np.float32), parameters)
            cache = {"voxel": {"layers": 1, "dims": layout["dims_m"], "voxel_size": layout["voxel_size_m"]}}
        config = MotionPlannerCfg.create(
            robot=self.robot_cfg, collision_cache=cache, device_cfg=self.device_cfg,
            num_ik_seeds=SETTINGS["num_ik_seeds"], num_trajopt_seeds=SETTINGS["num_trajopt_seeds"],
            position_tolerance=SETTINGS["position_tolerance_m"],
            orientation_tolerance=SETTINGS["orientation_tolerance"],
            use_cuda_graph=SETTINGS["use_cuda_graph"], random_seed=SETTINGS["random_seed"],
            optimizer_collision_activation_distance=SETTINGS["optimizer_collision_activation_distance_m"],
            self_collision_check=True,
        )
        config.trajopt_solver_config.interpolation_dt = SETTINGS["interpolation_dt_s"]
        self.planner = MotionPlanner(config)
        self.joint_names = list(self.planner.joint_names)
        if self.joint_names != list(ROBOTS[robot]["joint_names"]):
            raise ValueError(f"V2 {source.name} plans other joints: {self.joint_names}")
        self.checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=self.robot_cfg, scene_collision_checker=self.planner.scene_collision_checker,
            collision_activation_distance=0.0, device_cfg=self.device_cfg,
        ))
        self._ik_successes: list[int] = []
        self._ik_attempts: list[dict[str, Any]] = []
        self._populations: list[dict[str, Any]] | None = None
        self.planner.warmup(enable_graph=True)
        self._observe_ik()
        # warmup() runs pose plans only; capture the joint-space graphs now too.
        home = self.planner.default_joint_state.position.reshape(-1).tolist()[:len(self.joint_names)]
        self.plan_joints(home, [home[0] + 0.2, *home[1:]])
        self.attach(None)
        self.reset_seeds()
        self.ready_s = time.time() - started

    # -- IK evidence -----------------------------------------------------------

    def _observe_ik(self) -> None:
        """Keep each IK attempt's success count and seed population; the solves themselves are untouched.

        ``_get_result`` is where the solver turns its whole seed population
        into the returned top seeds; an early exit on the starting seeds runs
        it once, an optimised solve again after the optimiser. The population
        an attempt's answer came from is the last one its ``solve_pose`` saw.
        A plan that never solved its IK is ``goal_unreachable``.
        """
        solver = self.planner.ik_solver
        get_result, solve_pose = solver._get_result, solver.solve_pose

        def observed_result(opt_result: Any, metrics: Any, num_seeds: int, return_seeds: int) -> Any:
            result = get_result(opt_result, metrics, num_seeds, return_seeds)
            if self._populations is not None:
                try:
                    self._populations.append(self._population(opt_result, metrics, num_seeds, return_seeds, result))
                except Exception as exc:  # evidence never turns a solve into a planner fault
                    self._populations.append({"status": "unavailable", "reason": "ik_metric_capture_failed",
                                              "error_type": type(exc).__name__, "error_message": str(exc)[:500]})
            return result

        def observed_solve(*arguments: Any, **keywords: Any) -> Any:
            self._populations = []
            try:
                result = solve_pose(*arguments, **keywords)
            finally:
                populations, self._populations = self._populations, None
            self._ik_successes.append(int(self.torch.count_nonzero(result.success).item()))
            if len(self._ik_attempts) < MAX_IK_ATTEMPTS:
                record = populations[-1] if populations else {"status": "unavailable",
                                                              "reason": "no_ik_population_observed"}
                self._ik_attempts.append({**record, "populations_evaluated": len(populations)})
            return result

        solver._get_result = observed_result
        solver.solve_pose = observed_solve

    def _population(self, opt_result: Any, metrics: Any, num_seeds: int, return_seeds: int,
                    result: Any) -> dict[str, Any]:
        """One attempt's seed population, the record ``detail.ik_attempts`` keeps."""
        torch = self.torch
        solver = self.planner.ik_solver
        dof = len(self.joint_names)
        if list(solver.joint_names) != self.joint_names:
            raise ValueError("unexpected_ik_joint_order")
        positions = opt_result.detach().reshape(-1, opt_result.shape[-1])
        rows = positions.shape[0]
        if positions.shape[-1] != dof or rows % num_seeds or not 1 <= num_seeds <= 512:
            raise ValueError("unexpected_ik_population_shape")

        def seeds(tensor: Any) -> Any:
            # The first problem of the (padded) batch is the request's.
            return tensor.detach().reshape(rows, -1)[:num_seeds]

        parts: dict[str, list[Any]] = {"position_tolerance": [], "orientation_tolerance": []}
        for name, values in zip(metrics.convergence.names, metrics.convergence.values):
            for key in parts:
                if key in name:
                    parts[key].append(seeds(values))
        if not all(parts.values()):
            raise ValueError("missing_ik_convergence_metrics")
        position = torch.cat(parts["position_tolerance"], dim=-1).max(dim=-1).values
        rotation = torch.cat(parts["orientation_tolerance"], dim=-1).max(dim=-1).values
        feasible = metrics.costs_and_constraints.get_feasible(include_all_hybrid=False, sum_horizon=True)
        if isinstance(feasible, bool):
            feasible = torch.full((num_seeds,), feasible, dtype=torch.bool, device=position.device)
        else:
            feasible = seeds(feasible).all(dim=-1)
        thresholds = [float(solver.config.position_tolerance), float(solver.config.orientation_tolerance)]
        converged = (position < thresholds[0]) & (rotation < thresholds[1])
        accepted = feasible & converged if solver.config.success_requires_convergence else feasible
        returned = result.success.detach().reshape(-1, return_seeds)[0]
        q = positions[:num_seeds]
        limits = solver.kinematics.get_joint_limits().position.detach()[:, :dof]
        outside = ((q < limits[0]) | (q > limits[1])).any(dim=-1)
        lists = {name: value.cpu().tolist() for name, value in (
            ("q", q), ("position", position), ("rotation", rotation), ("feasible", feasible),
            ("converged", converged), ("accepted", accepted), ("returned", returned), ("limits", limits),
            ("outside", outside))}
        if not all(math.isfinite(item) for row in lists["q"] for item in row):
            raise ValueError("non_finite_ik_seeds")
        count = lambda name, flag=True: sum(1 for item in lists[name] if bool(item) is flag)  # noqa: E731
        return {
            "status": "captured", "scope": "all_seeds_for_one_original_ik_attempt",
            "joint_names": list(self.joint_names), "joint_positions": lists["q"],
            "position_errors_m": lists["position"], "rotation_errors": lists["rotation"],
            "rotation_error_units": "curobo_v2_orientation_metric",
            "position_threshold_m": thresholds[0], "rotation_threshold": thresholds[1],
            "feasible": lists["feasible"], "converged": lists["converged"], "accepted": lists["accepted"],
            "returned_success": lists["returned"], "position_limits": lists["limits"],
            "outside_position_limits": lists["outside"],
            "candidate_counts": {
                "optimized": num_seeds, "returned": len(lists["returned"]),
                "returned_successful": count("returned"), "converged": count("converged"),
                "feasible": count("feasible"), "accepted": count("accepted"),
            },
            "rejection_counts": {
                "position_tolerance": sum(1 for item in lists["position"] if not item < thresholds[0]),
                "rotation_tolerance": sum(1 for item in lists["rotation"] if not item < thresholds[1]),
                "solver_infeasible": count("feasible", False), "outside_position_limits": count("outside"),
                "self_collision": None,
            },
            "unknown_reasons": ["per_term_constraint_costs_not_retained_by_sdk"],
            "claim_boundary": "finite_optimizer_samples;counts_overlap;not_global_unreachability_or_scene_clearance",
        }

    # -- identity --------------------------------------------------------------

    def _robot_models(self) -> list[Any]:
        """Every robot model the solvers and the checker evaluate spheres with."""
        planner = self.planner
        models = [planner.ik_solver.kinematics, planner.trajopt_solver.kinematics, self.checker.kinematics]
        for owner in (planner.ik_solver, planner.trajopt_solver, planner.graph_planner):
            if owner is None:
                continue
            optimizer = getattr(owner, "optimizer", None)
            rollouts = list(getattr(owner, "get_all_rollout_instances", lambda: [])())
            rollouts += list(getattr(optimizer, "get_all_rollout_instances", lambda: [])())
            core = getattr(owner, "core", None)
            rollouts += [getattr(core, name, None) for name in ("metrics_rollout", "auxiliary_rollout")]
            models += [rollout.transition_model.robot_model for rollout in rollouts
                       if getattr(getattr(rollout, "transition_model", None), "robot_model", None) is not None]
        return models

    def _sphere_parameters(self) -> list[Any]:
        """The distinct sphere tables behind those models (one when they share the robot config)."""
        seen: dict[int, Any] = {}
        for model in self._robot_models():
            parameters = model.config.kinematics_config
            seen.setdefault(id(parameters), parameters)
        return list(seen.values())

    def identity(self) -> dict[str, Any]:
        import curobo
        import warp

        models = self._robot_models()
        return {"curobo_file": curobo.__file__, "torch": self.torch.__version__, "warp": warp.__version__,
                "device": str(self.device_cfg.device), "gpu": self.torch.cuda.get_device_name(self.device_cfg.device),
                "robot_file": self.robot_file, "robot_overrides": robots.OVERRIDES[self.robot],
                **self.planner_identity(), "held_slots": HELD_SLOTS, "settings": SETTINGS,
                "robot_models": len(models), "sphere_tables": len(self._sphere_parameters()), "ready_s": self.ready_s}

    def planner_identity(self) -> dict[str, Any]:
        """What every answer names: the robot configuration and the sphere set it was planned with."""
        return {"robot_sha256": self.robot_sha256, "collision_spheres_sha256": self.spheres_sha256}

    def ik_identity(self) -> dict[str, Any]:
        return {"num_seeds": SETTINGS["num_ik_seeds"], "position_tolerance_m": SETTINGS["position_tolerance_m"],
                "orientation_tolerance": SETTINGS["orientation_tolerance"], "attempts_per_query": 1}

    # -- state -----------------------------------------------------------------

    def _tensor(self, rows: Any) -> Any:
        return self.torch.as_tensor(rows, dtype=self.torch.float32, device=self.device_cfg.device)

    def _joint_state(self, joints: Any) -> Any:
        from curobo.types import JointState

        return JointState.from_position(self._tensor([list(joints)]), joint_names=self.joint_names)

    def _goal(self, pose_xyzw: list[float]) -> Any:
        from curobo.types import GoalToolPose, Pose

        x, y, z, qx, qy, qz, qw = pose_xyzw
        pose = Pose(position=self._tensor([[x, y, z]]), quaternion=self._tensor([[qw, qx, qy, qz]]))
        return GoalToolPose.from_poses({self.tool_frame: pose}, ordered_tool_frames=list(self.planner.tool_frames),
                                       num_goalset=1)

    def _kinematics_state(self, joints: Any) -> Any:
        return self.planner.compute_kinematics(self._joint_state(joints))

    def _tool_pose(self, state: Any) -> list[float]:
        pose = state.tool_poses.get_link_pose(self.tool_frame)
        x, y, z = pose.position.reshape(-1)[:3].tolist()
        w, qx, qy, qz = pose.quaternion.reshape(-1)[:4].tolist()
        return [x, y, z, qx, qy, qz, w]

    def tool_pose(self, joints: Any) -> list[float]:
        """The tool frame's base-frame pose ``[x, y, z, qx, qy, qz, qw]`` at ``joints``."""
        return self._tool_pose(self._kinematics_state(joints))

    def tool_poses(self, rows: Any) -> list[list[float]]:
        """``tool_pose`` of every joint row in one kinematics call (a path's turns, ``Worker._turns``)."""
        from curobo.types import JointState

        joints = JointState.from_position(self._tensor(rows), joint_names=self.joint_names)
        pose = self.planner.compute_kinematics(joints).tool_poses.get_link_pose(self.tool_frame)
        return [[*position, x, y, z, w] for position, (w, x, y, z) in zip(
            pose.position.reshape(-1, 3).tolist(), pose.quaternion.reshape(-1, 4).tolist())]

    def kinematics(self, joints: Any) -> tuple[Any, Any]:
        """The tool frame's transform and the robot's collision spheres, base frame, at ``joints``."""
        from ora.environment.controllers.depth_world import pose_matrix

        state = self._kinematics_state(joints)
        spheres = np.asarray(state.robot_spheres.detach().cpu().tolist(), dtype=np.float64).reshape(-1, 4)
        return pose_matrix(self._tool_pose(state)), spheres

    def path_spheres(self, rows: Any) -> Any:
        """The robot's collision spheres at every row of a path, base frame: row, sphere, ``[x, y, z, radius]``."""
        from curobo.types import JointState

        state = self.planner.compute_kinematics(JointState.from_position(self._tensor(rows), joint_names=self.joint_names))
        return state.robot_spheres.detach().cpu().numpy().astype(np.float64).reshape(len(rows), -1, 4)

    def sphere_links(self) -> list[str]:
        """Each row of ``kinematics``'s spheres: the link it belongs to."""
        parameters = self.checker.kinematics.config.kinematics_config
        total = len(self.kinematics(self.planner.default_joint_state.position.reshape(-1).tolist()[
            :len(self.joint_names)])[1])
        links = [""] * total
        for link in self.collision_links:
            for index in parameters.get_sphere_index_from_link_name(link).tolist():
                links[int(index)] = link
        return links

    def _rows(self, state: Any, tensor: Any) -> Any:
        """``tensor`` (any leading shape) as rows in the planner's joint order, by the state's joint names."""
        if tensor is None:
            return None
        names = list(getattr(state, "joint_names", None) or self.joint_names)
        array = np.asarray(tensor.detach().cpu().tolist(), dtype=np.float64).reshape(-1, len(names))
        return array[:, [names.index(name) for name in self.joint_names]].tolist()

    def joint_limits(self) -> list[list[float]]:
        limits = self.planner.kinematics.get_joint_limits().position
        return [[float(item) for item in row] for row in limits.detach().cpu().tolist()]

    def check(self, joints: Any) -> dict[str, Any]:
        """Joint limits, self-collision and world collision costs at ``joints`` (zero is clear)."""
        low, high = self.joint_limits()
        within = all(lo <= float(q) <= hi for q, lo, hi in zip(joints, low, high))
        position = self._tensor([[list(joints)]])
        self.checker.setup_batch_tensors(1, 1)
        state = self.checker.get_kinematics(position)
        self_cost = float(self.checker.get_self_collision(state.robot_spheres).sum().item())
        world = None
        if self.checker.collision_constraint is not None:
            world = float(self.checker.get_collision_constraint(state).sum().item())
        return {"within_limits": within, "self_collision": self_cost, "world_collision": world}

    # -- world -----------------------------------------------------------------

    def load_world(self, grid: dict[str, Any]) -> None:
        from curobo.scene import Scene, VoxelGrid

        features = self.torch.as_tensor(grid["features"].reshape(-1), device=self.device_cfg.device)
        voxel = VoxelGrid(name=VOXEL_LAYER, pose=[*grid["center_base_m"], 1.0, 0.0, 0.0, 0.0],
                          dims=list(grid["dims_m"]), voxel_size=float(grid["voxel_size_m"]),
                          feature_tensor=features)
        self.planner.update_world(Scene(voxel=[voxel]))

    def attach(self, spheres_tool: Any) -> None:
        """Write the held spheres (tool frame) into every sphere table, or clear them with ``None``."""
        for parameters in self._sphere_parameters():
            if spheres_tool is None:
                parameters.reset_link_spheres(HELD_LINK)
                continue
            index = parameters.get_sphere_index_from_link_name(HELD_LINK)
            rows = [[0.0, 0.0, 0.0, -100.0]] * len(index)
            if len(spheres_tool) > len(rows):
                raise ValueError(f"{len(spheres_tool)} held spheres exceed {len(rows)} slots")
            rows[:len(spheres_tool)] = [[float(item) for item in sphere] for sphere in spheres_tool]
            parameters.link_spheres[:, index, :] = self._tensor(rows).to(parameters.link_spheres.dtype)

    def reset_seeds(self) -> None:
        self.planner.reset_seed()

    def close(self) -> None:
        """Release the solvers' CUDA graphs before the process exits."""
        self.planner.destroy()

    # -- solves ----------------------------------------------------------------

    def _trajectory(self, result: Any) -> dict[str, Any]:
        plan = result.get_interpolated_plan()
        return {"joint_names": list(self.joint_names), "positions": self._rows(plan, plan.position),
                "velocities": self._rows(plan, plan.velocity),
                "dt_s": float(self.planner.trajopt_solver.config.interpolation_dt)}

    def _outcome(self, result: Any) -> dict[str, Any]:
        success = result is not None and bool(result.success.any().item())
        return {"trajectory": self._trajectory(result) if success else None,
                "ik_solved": any(count > 0 for count in self._ik_successes),
                "ik_attempts": list(self._ik_attempts),
                "detail": {"ik_successes_per_attempt": list(self._ik_successes),
                           "trajopt_result": result is not None,
                           "solve_time_s": None if result is None else float(result.solve_time)}}

    def plan_pose(self, start: Any, pose_xyzw: list[float], *, max_attempts: int | None = None) -> dict[str, Any]:
        self._ik_successes, self._ik_attempts = [], []
        result = self.planner.plan_pose(self._goal(pose_xyzw), self._joint_state(start),
                                        max_attempts=max_attempts or SETTINGS["max_attempts"],
                                        enable_graph_attempt=SETTINGS["enable_graph_attempt"])
        return self._outcome(result)

    def plan_joints(self, start: Any, goal: Any) -> dict[str, Any]:
        self._ik_successes, self._ik_attempts = [], []
        result = self.planner.plan_cspace(self._joint_state(goal), self._joint_state(start),
                                          max_attempts=SETTINGS["max_attempts"],
                                          enable_graph_attempt=SETTINGS["enable_graph_attempt"])
        return self._outcome(result)

    def solve_ik(self, seed: Any, pose_xyzw: list[float]) -> dict[str, Any]:
        """One IK solve of the planner's own solver, seeded from ``seed`` (and its sampled seeds)."""
        result = self.planner.ik_solver.solve_pose(self._goal(pose_xyzw), current_state=self._joint_state(seed),
                                                   return_seeds=1)
        solved = bool(result.success.reshape(-1)[0].item())
        joints = self._rows(result.js_solution, result.js_solution.position)[0] if solved else None
        return {"solved": solved, "joint_positions": joints,
                "position_error_m": _finite(result.position_error.reshape(-1)[0]),
                "rotation_error": _finite(result.rotation_error.reshape(-1)[0])}


__all__ = ["CuroboV2Engine", "HELD_LINK", "HELD_SLOTS", "MAX_IK_ATTEMPTS", "SETTINGS"]
