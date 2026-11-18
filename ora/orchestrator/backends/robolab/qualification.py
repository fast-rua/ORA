"""Scripted RoboLab sparse-motion qualification through the real worker path.

The record this writes is what ``definitions.SPARSE_MOTION_QUALIFIED_ENVIRONMENTS``
cites for ``("motion_action_v3", "robolab")``. It drives one RoboLab task
through ``factory.build_robolab_worker_session`` (Isaac Sim 5.1 worker, local
RPC) with no policy and no provider: reset with the loaded Robotiq geometry
check, a public RGB-D observation, a short TCP move on the servo, a free move on
the shared planner with its timing and wrist check, the planner's refusal of a pose
out of reach, the gripper both ways, a shared wait, a planned home recovery, status,
the official verdict and a clean close.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.environment.controllers import robolab_droid as D
from ora.environment.controllers.depth_world import WORLD_MODEL
from ora.environment.factory import _public_geometry_config, build_robolab_worker_session
from ora.orchestrator.preflight.checks import motion_timing_check

SCHEMA = "ora.robolab-sparse-motion-qualification.v1"
#: The planner revision of the default baseline's ``runtime.motion_planner``.
PLANNER_REVISION = "d17b54ce32cba095c0b000c4c58777075d11de0e"
RECORD_NAME = "robolab-sparse-motion-qualification.json"
DESCRIPTOR = "ora/policy/embodiments/droid-franka-robotiq/descriptor-public-geometry-v1.json"


def _action(observation: Mapping[str, Any], name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "ora-motion-action.v3", "observation_id": observation["observation_id"],
            "action_id": "action-" + uuid.uuid4().hex[:16], "bound_observation_sha256": "0" * 64,
            "action": name, "arguments": dict(arguments)}


def qualify(*, task: str, seed: int, gpu: int, output_root: Path, descriptor: str | None) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    # The record's motion runs as an episode's does: planned, and timed by the worker this check binds.
    timing = motion_timing_check({"python_env": "ORA_TIMING_PYTHON"})
    if not timing["passed"]:
        raise RuntimeError(f"RoboLab qualification needs the timing worker: {timing['detail']}")
    (output_root / "preflight.json").write_text(json.dumps({"checks": [timing]}, indent=2) + "\n", encoding="utf-8")
    runtime: dict[str, Any] = {"root_env": "ROBOLAB_ROOT", "python_env": "ROBOLAB_PYTHON",
                               "graphics_runtime_env": "ROBOLAB_GRAPHICS_RUNTIME", "renderer_gpu": gpu,
                               "startup_timeout_s": 900, "request_timeout_s": 1200, "close_timeout_s": 120,
                               "motion_planner": {
                                   "implementation": "curobo_v2", "required_revision": PLANNER_REVISION,
                                   "request_timeout_s": 300, "robot": "franka_robotiq", "world_model": WORLD_MODEL,
                                   "python_executable": os.path.abspath(os.path.expanduser(
                                       os.environ["CUROBO_V2_PYTHON"])),
                                   "root": str(Path(os.environ["CUROBO_V2_ROOT"]).expanduser().resolve())}}
    if descriptor is not None:
        from ora.agent.composition.embodiment.geometry import bind_public_geometry_descriptor
        from ora.orchestrator.paths import PROJECT_ROOT
        runtime["public_geometry_descriptor"] = bind_public_geometry_descriptor(descriptor, project_root=PROJECT_ROOT)
    session = build_robolab_worker_session(runtime=runtime, output_root=output_root / "native", task=task,
                                           seed=seed, action_codec="motion_action_v3")
    checks: list[dict[str, Any]] = []
    started = time.time()

    def check(name: str, passed: bool, **detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), **detail})

    def observe() -> dict[str, Any]:
        observation, _images = session.observe(output_root / "native" / "observations")
        return observation

    def run(name: str, observation: Mapping[str, Any], action: str, arguments: Mapping[str, Any],
            accept: Callable[[Mapping[str, Any]], bool]) -> Mapping[str, Any]:
        receipt = session.execute(_action(observation, action, arguments))
        arm = receipt.get("arms", {}).get("right", {})
        check(name, accept(receipt), status=receipt.get("execution_status"), failure_code=receipt.get("failure_code"),
              steps=receipt.get("steps_used"), pose_error=arm.get("pose_error"), gripper=arm.get("gripper_outcome"))
        return receipt

    try:
        reset = session.start()
        check("reset", reset.get("official_success_at_reset") is False, language=reset.get("task_language"),
              limit=reset.get("native_action_limit"), tcp=reset.get("tcp_pose_xyz_xyzw"))
        observation = observe()
        cameras = observation.get("cameras", {})
        check("observation_rgbd", set(cameras) == {"agentview", "wrist"}, cameras=sorted(cameras),
              geometry=observation.get("robot_geometry_verification", {}).get("passed"))
        pose = observation["proprioception"]["end_effectors"]["right"]["pose_xyz_xyzw"]
        effector = observation["proprioception"]["end_effectors"]["right"]
        check("proprioception", len(pose) == 7 and effector.get("jaw_separation_m", 0) > 0.07,
              tcp=pose[:3], jaw_separation_m=effector.get("jaw_separation_m"))
        within = lambda r: (r.get("execution_status") == "reached"  # noqa: E731
                            and (r["arms"]["right"]["pose_error"] or {}).get("translation_m", 1.0)
                            <= D.POSITION_TOLERANCE_M + 0.002)
        strategy = lambda name: lambda r: r.get("motion_strategy") == name  # noqa: E731
        down = {"position_m": [pose[0], pose[1], pose[2] - 0.04], "orientation_xyzw": pose[3:7]}
        run("move_pose_precision", observation, "move_pose",
            {"targets": {"right": down}, "interactions": {"right": "precision"}},
            lambda r: within(r) and strategy("cartesian_servo_direct")(r))
        observation = observe()
        far = {"position_m": [pose[0] + 0.10, pose[1] + 0.12, pose[2] + 0.02], "orientation_xyzw": pose[3:7]}
        planned = run("move_free_space", observation, "move_free_space",
                      {"targets": {"right": far}, "interactions": {"right": "free_space"}},
                      lambda r: within(r) and strategy("planned_joint_path")(r))
        motion = planned.get("motion") or {}
        check("planned_path_timed", motion.get("timing", {}).get("implementation") == "toppra_ruckig"
              and bool(motion.get("wrist")), timing=motion.get("timing"), wrist=motion.get("wrist"))
        observation = observe()
        out = {"position_m": [pose[0] + 1.5, pose[1], pose[2]], "orientation_xyzw": pose[3:7]}
        refused = session.execute(_action(observation, "move_free_space",
                                          {"targets": {"right": out}, "interactions": {"right": "free_space"}}))
        check("planner_refuses_unreachable", refused.get("execution_status") == "plan_rejected"
              and refused.get("failure_stage") == "planner" and refused.get("steps_used") == 0,
              failure_code=refused.get("failure_code"), status=refused.get("execution_status"))
        run("set_gripper_close", observation, "set_gripper", {"targets": {"right": "close"}},
            lambda r: r.get("execution_status") == "reached"
            and (r["arms"]["right"].get("gripper_outcome") or {}).get("state") == "reached_target")
        run("set_gripper_open", observation, "set_gripper", {"targets": {"right": "open"}},
            lambda r: r.get("execution_status") == "reached"
            and (r["arms"]["right"].get("gripper_outcome") or {}).get("state") == "reached_target")
        run("wait", observation, "wait", {"ticks": 5},
            lambda r: r.get("execution_status") == "reached" and (r.get("wait") or {}).get("ticks_used") == 5)
        observation = observe()
        run("recover_home", observation, "recover_motion", {"targets": {"right": "home"}},
            lambda r: within(r) and strategy("planned_joint_path")(r))
        status = session.status()
        check("status_running", status.get("terminal") is False, status=status.get("reason"))
        verdict = session.persist_official_verdict()
        check("official_verdict", verdict.get("official_success") is False, native_actions=verdict.get("native_actions"))
    finally:
        try:
            session.close()
            check("close", True)
        except Exception as exc:  # noqa: BLE001
            check("close", False, error=f"{type(exc).__name__}: {exc}")
    record = {"schema": SCHEMA, "task": task, "seed": seed, "renderer_gpu": gpu, "action_codec": "motion_action_v3",
              "passed": all(item["passed"] for item in checks), "checks": checks, "started_unix_s": started,
              "finished_unix_s": time.time(), "controller_constants": D.published_limits()}
    path = output_root / RECORD_NAME
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["path"], record["sha256"] = str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="BananaInBowlTask")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptor", default=None)
    args = parser.parse_args(argv)
    record = qualify(task=args.task, seed=args.seed, gpu=args.gpu, output_root=args.output, descriptor=args.descriptor)
    for item in record["checks"]:
        print(("PASS " if item["passed"] else "FAIL ") + json.dumps(item, default=str)[:400])
    print(json.dumps({"passed": record["passed"], "path": record["path"], "sha256": record["sha256"]}))
    return 0 if record["passed"] else 1
