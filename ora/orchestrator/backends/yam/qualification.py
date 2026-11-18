"""Scripted YAM sparse-motion qualification through the real worker path.

The record this writes is what ``definitions.SPARSE_MOTION_QUALIFIED_ENVIRONMENTS``
cites for ``("motion_action_v3", "yam")``. It drives the simulated station through
``factory.build_yam_worker_session`` (isolated worker, local RPC) with no policy and no
provider: reset, a calibrated four-camera RGB-D observation, both TCPs, precision and
free-space moves of one arm, a simultaneous two-arm move, both grippers both ways, a
shared wait, the refusals of an unreachable target and of two arms brought together,
home recovery, status, the null verdict and a clean close.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.environment.controllers import yam_station as Y
from ora.environment.factory import build_yam_worker_session

SCHEMA = "ora.yam-sparse-motion-qualification.v1"
RECORD_NAME = "yam-sparse-motion-qualification.json"
CAMERAS = {"overhead", "front", "wrist_left", "wrist_right"}


def _action(observation: Mapping[str, Any], name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "ora-motion-action.v3", "observation_id": observation["observation_id"],
            "action_id": "action-" + uuid.uuid4().hex[:16], "bound_observation_sha256": "0" * 64,
            "action": name, "arguments": dict(arguments)}


def qualify(*, output_root: Path, python_env: str) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    runtime: dict[str, Any] = {"kind": "yam", "transport": "local_rpc", "python_env": python_env,
                               "rig_mode": "sim", "task": "sim_pick_place", "seed": 0}
    session = build_yam_worker_session(runtime=runtime, output_root=output_root / "native",
                                       task="sim_pick_place", seed=0, action_codec="motion_action_v3")
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
        arms = {arm: {"pose_error": record.get("pose_error"),
                      "gripper": (record.get("gripper_outcome") or {}).get("state")}
                for arm, record in receipt.get("arms", {}).items() if record.get("targeted")}
        check(name, accept(receipt), status=receipt.get("execution_status"), failure_code=receipt.get("failure_code"),
              steps=receipt.get("steps_used"), arms=arms)
        return receipt

    def within(*arms: str) -> Callable[[Mapping[str, Any]], bool]:
        return lambda r: (r.get("execution_status") == "reached" and all(
            (r["arms"][arm]["pose_error"] or {}).get("translation_m", 1.0) <= Y.POSITION_TOLERANCE_M + 0.002
            for arm in arms))

    def gripper(*arms: str) -> Callable[[Mapping[str, Any]], bool]:
        return lambda r: (r.get("execution_status") == "reached" and all(
            (r["arms"][arm].get("gripper_outcome") or {}).get("state") == "reached_target" for arm in arms))

    def refused(code: str) -> Callable[[Mapping[str, Any]], bool]:
        return lambda r: r.get("execution_status") == "plan_rejected" and r.get("failure_code") == code \
            and r.get("steps_used") == 0

    try:
        reset = session.start()
        check("reset", reset.get("official_success_at_reset") is False and set(reset["tcp_pose_xyz_xyzw"]) == set(Y.ARMS),
              language=reset.get("task_language"), limit=reset.get("control_tick_limit"))
        observation = observe()
        cameras = observation.get("cameras", {})
        check("observation_rgbd", set(cameras) == CAMERAS and all(c.get("calibrated") for c in cameras.values()),
              cameras=sorted(cameras))
        effectors = observation["proprioception"]["end_effectors"]
        check("proprioception", set(effectors) == set(Y.ARMS) and all(
            len(e["pose_xyz_xyzw"]) == 7 and e["jaw_separation_m"] > 0.09 for e in effectors.values()),
            tcp={arm: e["pose_xyz_xyzw"][:3] for arm, e in effectors.items()},
            jaw_separation_m={arm: e["jaw_separation_m"] for arm, e in effectors.items()})
        left, right = effectors["left"]["pose_xyz_xyzw"], effectors["right"]["pose_xyz_xyzw"]
        pose = lambda p, dx=0.0, dy=0.0, dz=0.0: {"position_m": [p[0] + dx, p[1] + dy, p[2] + dz],  # noqa: E731
                                                  "orientation_xyzw": p[3:7]}
        run("move_pose_precision", observation, "move_pose",
            {"targets": {"left": pose(left, dz=-0.04)}, "interactions": {"left": "precision"}}, within("left"))
        observation = observe()
        run("move_free_space", observation, "move_free_space",
            {"targets": {"left": pose(left, dx=0.06, dy=-0.10, dz=-0.04)}, "interactions": {"left": "free_space"}},
            within("left"))
        run("move_free_space_two_arms", observation, "move_free_space",
            {"targets": {"left": pose(left, dx=0.04, dy=0.04), "right": pose(right, dx=0.04, dy=-0.04)},
             "interactions": {"left": "free_space", "right": "free_space"}}, within("left", "right"))
        run("set_gripper_close", observation, "set_gripper", {"targets": {"left": "close", "right": "close"}},
            gripper("left", "right"))
        run("set_gripper_open", observation, "set_gripper", {"targets": {"left": "open", "right": "open"}},
            gripper("left", "right"))
        run("wait", observation, "wait", {"ticks": 5},
            lambda r: r.get("execution_status") == "reached" and (r.get("wait") or {}).get("ticks_used") == 5)
        run("refuse_unreachable", observation, "move_pose",
            {"targets": {"left": pose(left, dx=0.30)}, "interactions": {"left": "precision"}},
            lambda r: r.get("execution_status") == "plan_rejected" and r.get("steps_used") == 0
            and r.get("failure_code") in {"target_out_of_reach", "target_ik_unreachable"})
        run("refuse_inter_arm", observation, "move_free_space",
            # Both poses solve by IK; their TCPs are 0.16 m apart but the wrist cameras face
            # each other 1.2 cm apart (station model, 2026-10-28).
            {"targets": {"left": {"position_m": [0.34, 0.02, 0.08], "orientation_xyzw": left[3:7]},
                         "right": {"position_m": [0.34, -0.14, 0.08], "orientation_xyzw": right[3:7]}},
             "interactions": {"left": "free_space", "right": "free_space"}}, refused("inter_arm_clearance"))
        run("recover_home", observation, "recover_motion", {"targets": {"left": "home", "right": "home"}},
            within("left", "right"))
        status = session.status()
        check("status_running", status.get("terminal") is False, status=status.get("reason"))
        verdict = session.persist_official_verdict()
        check("official_verdict_null", verdict.get("official_success") is None,
              control_ticks=verdict.get("control_ticks"))
    finally:
        try:
            session.close()
            check("close", True)
        except Exception as exc:  # noqa: BLE001
            check("close", False, error=f"{type(exc).__name__}: {exc}")
    record = {"schema": SCHEMA, "task": "sim_pick_place", "seed": 0, "rig_mode": "sim",
              "action_codec": "motion_action_v3", "passed": all(item["passed"] for item in checks),
              "checks": checks, "started_unix_s": started, "finished_unix_s": time.time(),
              "controller_constants": Y.published_limits()}
    path = output_root / RECORD_NAME
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["path"], record["sha256"] = str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-env", default="YAM_PYTHON",
                        help="name of the environment variable holding the worker interpreter")
    args = parser.parse_args(argv)
    record = qualify(output_root=args.output, python_env=args.python_env)
    for item in record["checks"]:
        print(("PASS " if item["passed"] else "FAIL ") + json.dumps(item, default=str)[:400])
    print(json.dumps({"passed": record["passed"], "path": record["path"], "sha256": record["sha256"]}))
    return 0 if record["passed"] else 1
