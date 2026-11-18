"""Scripted RoboCasa365 sparse-motion qualification through the real worker path.

The record this writes is what ``definitions.SPARSE_MOTION_QUALIFIED_ENVIRONMENTS``
cites for ``("motion_action_v3", "robocasa")``. It drives one kitchen through
``factory.build_robocasa_worker_session`` (isolated interpreter, local RPC)
with no policy and no provider: reset, a public RGB-D observation, arm
precision and free-space moves, the gripper both ways, a shared wait, base
translation and rotation through ``move_base``, home recovery, an IK probe,
status, the official verdict and a clean close. The motion runs on the shared
motion base as an episode's does, with the default ``runtime.motion_planner``.
Each check records its measured values; the record passes only when every check
passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.environment.controllers import robocasa_omron as C
from ora.environment.factory import build_robocasa_worker_session
from ora.orchestrator.backends.robocasa.preflight import motion_planner_check
from ora.orchestrator.paths import PROJECT_ROOT
from ora.orchestrator.preflight.checks import motion_timing_check
from ora.orchestrator.spec.inputs import default_instance

SCHEMA = "ora.robocasa-sparse-motion-qualification.v1"
RECORD_NAME = "robocasa-sparse-motion-qualification.json"


def _turn_about_local_z(quaternion_xyzw: list[float], angle: float) -> list[float]:
    """The orientation turned by ``angle`` about its own local z (the approach axis)."""

    x, y, z, w = quaternion_xyzw
    hz, hw = math.sin(angle / 2.0), math.cos(angle / 2.0)
    # q * (0, 0, sin, cos)
    return [w * 0.0 + x * hw + y * hz - z * 0.0,
            w * 0.0 - x * hz + y * hw + z * 0.0,
            w * hz + x * 0.0 - y * 0.0 + z * hw,
            w * hw - x * 0.0 - y * 0.0 - z * hz]


def _action(observation: Mapping[str, Any], name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "ora-motion-action.v3", "observation_id": observation["observation_id"],
            "action_id": "action-" + uuid.uuid4().hex[:16], "bound_observation_sha256": "0" * 64,
            "action": name, "arguments": dict(arguments)}


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def qualify(*, task: str, seed: int, output_root: Path, python_env: str = "ROBOCASA_PYTHON") -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    # The record's motion runs as an episode's does: timed by the worker this check binds.
    timing = motion_timing_check({"python_env": "ORA_TIMING_PYTHON"})
    if not timing["passed"]:
        raise RuntimeError(f"RoboCasa qualification needs the timing worker: {timing['detail']}")
    motion_planner = default_instance("robocasa", PROJECT_ROOT)[1]["runtime"]["motion_planner"]
    planner = motion_planner_check(motion_planner)
    if not planner["passed"]:
        raise RuntimeError(f"RoboCasa qualification needs the shared planner: {planner['detail']}")
    (output_root / "preflight.json").write_text(json.dumps({"checks": [timing, planner]}, indent=2) + "\n",
                                                encoding="utf-8")
    session = build_robocasa_worker_session(
        runtime={"python_env": python_env, "startup_timeout_s": 300, "request_timeout_s": 600,
                 "close_timeout_s": 60, "motion_planner": motion_planner},
        output_root=output_root / "native", task=task, seed=seed, action_codec="motion_action_v3",
    )
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
        check(name, accept(receipt), status=receipt.get("execution_status"),
              failure_code=receipt.get("failure_code"), steps=receipt.get("steps_used"),
              pose_error=arm.get("pose_error"), gripper=arm.get("gripper_outcome"),
              base=(receipt.get("base") or {}).get("base_error"),
              # why a move was refused or stopped (its admission's reach check, the lift's record) and the lift it made
              **{key: receipt[key] for key in ("reach_check", "torso") if receipt.get(key) is not None})
        return receipt

    try:
        reset = session.start()
        check("reset", bool(reset.get("task_language")) and reset.get("official_success_at_reset") is False,
              language=reset.get("task_language"), horizon=reset.get("horizon_control_steps"))
        observation = observe()
        cameras = observation.get("cameras", {})
        check("observation_rgbd", set(cameras) == {"agentview", "wrist"} and all(
            {"rgb_lossless_png", "depth_numpy_m_f32", "calibration"} <= set(cameras[name]["files"])
            for name in cameras), cameras=sorted(cameras))
        base = observation["proprioception"].get("base", {}).get("pose_xy_yaw")
        pose = observation["proprioception"]["end_effectors"]["right"]["pose_xyz_xyzw"]
        check("proprioception", isinstance(base, list) and len(base) == 3 and len(pose) == 7,
              base=base, grip=pose[:3])
        within = lambda r: (r.get("execution_status") == "reached"  # noqa: E731
                            and (r["arms"]["right"]["pose_error"] or {}).get("translation_m", 1.0)
                            <= C.ARM_POSITION_TOLERANCE_M + 0.002)
        # The seed-0 OpenDrawer robot faces a cabinet front 0.4 m ahead of its hand, in view of
        # the public cameras: admission refuses the straight line into it before any motion.
        heading = base[2]
        into_fixture = {"position_m": [pose[0] + 0.40 * math.cos(heading), pose[1] + 0.40 * math.sin(heading), pose[2]],
                        "orientation_xyzw": pose[3:7]}
        run("move_pose_into_fixture_refused", observation, "move_pose",
            {"targets": {"right": into_fixture}, "interactions": {"right": "precision"}},
            lambda r: r.get("execution_status") == "plan_rejected"
            and r.get("failure_code") in {"target_in_collision_at_base_pose", "path_in_collision_at_base_pose"}
            and r.get("native_actions") == 0)
        run("set_gripper_open", observation, "set_gripper", {"targets": {"right": "open"}},
            lambda r: r.get("execution_status") == "reached")
        observation = observe()
        down = {"position_m": [pose[0], pose[1], pose[2] - 0.04], "orientation_xyzw": pose[3:7]}
        run("move_pose_precision", observation, "move_pose",
            {"targets": {"right": down}, "interactions": {"right": "precision"}}, within)
        observation = observe()
        far = {"position_m": [pose[0] + 0.12, pose[1] + 0.08, pose[2] + 0.04], "orientation_xyzw": pose[3:7]}
        run("move_free_space", observation, "move_free_space",
            {"targets": {"right": far}, "interactions": {"right": "free_space"}}, within)
        observation = observe()
        current = observation["proprioception"]["end_effectors"]["right"]["pose_xyz_xyzw"]
        turned = {"position_m": current[:3], "orientation_xyzw": _turn_about_local_z(current[3:7], math.radians(60.0))}
        run("move_pose_reorient_60deg", observation, "move_pose",
            {"targets": {"right": turned}, "interactions": {"right": "precision"}},
            lambda r: r.get("execution_status") == "reached"
            and (r["arms"]["right"]["pose_error"] or {}).get("orientation_rad", 1.0) <= C.ARM_ORIENTATION_TOLERANCE_RAD + 0.01
            and r.get("motion_strategy") == "joint_path")
        run("set_gripper_close", observation, "set_gripper", {"targets": {"right": "close"}},
            lambda r: r.get("execution_status") == "reached"
            and (r["arms"]["right"].get("gripper_outcome") or {}).get("state") == "reached_target")
        run("wait", observation, "wait", {"ticks": 5},
            lambda r: r.get("execution_status") == "reached" and (r.get("wait") or {}).get("ticks_used") == 5)
        observation = observe()
        x, y, yaw = observation["proprioception"]["base"]["pose_xy_yaw"]
        back = {"x_m": x - 0.3 * math.cos(yaw), "y_m": y - 0.3 * math.sin(yaw), "yaw_rad": yaw}
        run("move_base_translate", observation, "move_base", {"target": back},
            lambda r: r.get("execution_status") == "reached")
        turn = {**back, "yaw_rad": _wrap(yaw + 0.4)}
        run("move_base_rotate", observation, "move_base", {"target": turn},
            lambda r: r.get("execution_status") == "reached")
        run("move_base_return", observation, "move_base", {"target": back},
            lambda r: r.get("execution_status") == "reached")
        # 0.30 m above the hand is beyond the arm alone at the start torso height
        # (2026-10-18 probe): admission names a height, the torso lifts first.
        observation = observe()
        current = observation["proprioception"]["end_effectors"]["right"]["pose_xyz_xyzw"]
        high = {"position_m": [current[0], current[1], current[2] + 0.30], "orientation_xyzw": current[3:7]}
        raised = run("move_pose_torso_raise", observation, "move_pose",
                     {"targets": {"right": high}, "interactions": {"right": "precision"}},
                     lambda r: within(r) and (r.get("torso") or {}).get("stop_reason") == "reached"
                     and r["torso"]["to_m"] - r["torso"]["from_m"] >= 0.1)
        checks[-1].setdefault("torso", raised.get("torso"))
        run("recover_home", observation, "recover_motion", {"targets": {"right": "home"}},
            lambda r: r.get("execution_status") == "reached")
        observation = observe()
        current = observation["proprioception"]["end_effectors"]["right"]["pose_xyz_xyzw"]
        probe = session.probe_motion({
            "observation_id": observation["observation_id"], "variant_policy": "parallel_jaw_equivalents",
            "queries": [{"query_id": "q0", "arm": "right",
                         "pose": {"position_m": current[:3], "orientation_xyzw": current[3:7]}}],
        })
        result = (probe.get("results") or [{}])[0]
        check("probe_motion", probe.get("schema") == "ora.motion-probe-result.v1" and result.get("reachable") is True,
              variants=[variant.get("reachable") for variant in result.get("variants", [])])
        status = session.status()
        check("status_running", status.get("terminal") is False, status=status.get("reason"))
        verdict = session.persist_official_verdict()
        check("official_verdict", verdict.get("official_success") is False
              and verdict.get("source") == "RoboCasa _check_success",
              control_steps=verdict.get("control_steps"))
    finally:
        try:
            session.close()
            check("close", True)
        except Exception as exc:  # noqa: BLE001  the record keeps the close failure
            check("close", False, error=f"{type(exc).__name__}: {exc}")
    record = {
        "schema": SCHEMA, "task": task, "seed": seed, "action_codec": "motion_action_v3",
        "passed": all(item["passed"] for item in checks), "checks": checks,
        "started_unix_s": started, "finished_unix_s": time.time(),
        "controller_constants": C.published_limits(),
    }
    path = output_root / RECORD_NAME
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["path"] = str(path)
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="OpenDrawer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = qualify(task=args.task, seed=args.seed, output_root=args.output)
    for item in record["checks"]:
        print(("PASS " if item["passed"] else "FAIL ") + json.dumps(item, default=str)[:300])
    print(json.dumps({"passed": record["passed"], "path": record["path"], "sha256": record["sha256"]}))
    return 0 if record["passed"] else 1
