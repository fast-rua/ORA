"""Host-owned motion qualification using the production worker bootstrap.

The numerical suite imports no simulator runtime. Execution holds and recovery
anchors belong to ExecutionStateOwner, just as in an ordinary episode. Every
arm move runs as the route every episode runs: ``MatrixPlanner`` plans it here
with the toolkit lib over the worker's read-only kinematics query, and the
worker executes it as one ``follow_joint_trajectory`` with the shared planner,
the timing worker and the ARX X5 motion profile the default baseline binds (a
runtime configuration without them gets the default baseline's, and the record
says which).
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from ora.contracts.joint_trajectory import native_action_cap
from ora.environment.backends.robodojo.motion_qualification import run_cases, build_qualification_record
from ora.environment.workspace_profiles import ROBODOJO_ARX_X5_TABLETOP_V1, resolve_workspace_profile
from ora.orchestrator.episode.execution_state import bind_execution_state
from ora.orchestrator.paths import PROJECT_ROOT
from ora.orchestrator.preflight.checks import motion_timing_check

#: The runtime fields that make a move the one an episode runs: the shared planner, its timing worker and the
#: ARX X5 motion profile the toolkit plans and actuates with.
MOTION_STACK_FIELDS = ("motion_planner", "motion_timing", "motion_profile")


def _free_memory_mib(device: str) -> int:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    rows = [line.split(",") for line in output.splitlines()]
    selected = [int(row[1]) for row in rows if row[0].strip() == device]
    if len(selected) != 1:
        raise RuntimeError(f"GPU {device} is not visible to nvidia-smi")
    return selected[0]


def motion_stack(runtime: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """``runtime`` with the planner, timing and motion profile every episode binds (its own, else the default
    baseline's), and where each came from. A field present but not an object is refused: the matrix qualifies the
    planned route."""
    from ora.agent.composition.baseline import default_baseline_id, load_native_baseline

    baseline = default_baseline_id("robodojo")
    default = load_native_baseline(PROJECT_ROOT, baseline)["instances"]["robodojo"]["runtime"]
    bound, source = dict(runtime), {}
    for field in MOTION_STACK_FIELDS:
        if field not in bound:
            bound[field], source[field] = copy.deepcopy(default[field]), f"baseline:{baseline}"
        elif isinstance(bound[field], Mapping):
            source[field] = "runtime_config"
        else:
            raise SystemExit(f"runtime.{field} must be an object: the motion qualification runs the planned route "
                             "every episode runs. No worker was started.")
    return bound, source


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item)))


class _Kinematics:
    """The one interface method the motion lib plans with: the worker's read-only kinematics query."""

    def __init__(self, query: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> None:
        self._query = query

    def kinematics(self, *, request: Mapping[str, Any]) -> Any:
        from ora.agent.toolkit.interface.projections import kinematics
        from ora.agent.toolkit.interface.types import Response, canonical

        data = json.loads(kinematics(self._query(dict(request))).data)
        return Response(canonical({"schema": "ora.toolkit.response.v1", "method": "kinematics", "status": "ok",
                                   "data": data, "error": None}))


class MatrixPlanner:
    """Plans a qualification request as the default RoboDojo tools plan theirs and binds its action; runs nothing.

    A ``move`` is planned as move_pose plans a compiled request at its default speed: the interaction label picks
    the route (the default surfaces' staged contact approach included) and the orientation is used exactly.
    ``home`` is planned as return_home plans it and ``retrace`` as recover_pose does (their own option builder and
    paces). ``kinematics`` answers ``ora.kinematics-request.v1`` (the worker's ``align_plan`` bridge); ``state``
    returns the execution-state values (home anchors, undo stacks). A toolkit refusal comes back as
    ``{"refusal": {"code", "message", "constraints"}}`` with no action bound.
    """

    def __init__(self, *, kinematics: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                 state: Callable[[], Mapping[str, Any]], global_planner: bool,
                 workspace_bounds_m: Mapping[str, Any], motion_profile: Mapping[str, Any] | None = None) -> None:
        self.api = SimpleNamespace(planner=_Kinematics(kinematics))
        self.state, self.global_planner = state, bool(global_planner)
        self.workspace_bounds_m = dict(workspace_bounds_m)
        # The tool context's motion part as far as planning reads it: the profile's fast (and v1's without one).
        self.config = {"motion": {"global_planner": self.global_planner,
                                  **({} if motion_profile is None else {"profile": dict(motion_profile)})}}

    @property
    def free_route(self) -> str:
        """The route a free-space move takes here (``motion_checks.FREE_ROUTES``)."""
        return "global" if self.global_planner else "joint"

    def __call__(self, observation: Mapping[str, Any], request: Mapping[str, Any], action_id: str) -> dict[str, Any]:
        from ora.agent.toolkit.lib.motion_plan import checks, execute

        effectors = observation["proprioception"]["end_effectors"]
        try:
            planned, route, recovery = {"move": self._move, "home": self._home,
                                        "retrace": self._retrace}[request["kind"]](effectors, request)
        except checks.PlanRefusal as refusal:
            return {"refusal": {"code": refusal.code, "message": str(refusal),
                                "constraints": _plain(refusal.constraints)}}
        part = ({"fraction": planned.fraction, "goals_xyz_xyzw": dict(planned.goals)} if planned.fraction < 1.0
                else None)
        action = execute.trajectory_action(action_id=action_id, observation={
            "observation_id": str(observation["observation_id"]),
            "observation_sha256": str(observation["observation_sha256"])}, knots=planned.knots["joints"],
            expected=planned.expected, settle=planned.settle, part_of_move=part)
        if recovery is not None:
            action["arguments"]["recovery"] = _plain(recovery)
        return {"action": action, "route": route, "planned_native_actions": planned.knots["native_actions"]}

    @staticmethod
    def _joints(effectors: Mapping[str, Any], arms: Any) -> dict[str, list[float]]:
        return {arm: [float(v) for v in effectors[arm]["arm_joint_qpos"]] for arm in arms}

    def _move(self, effectors, request):
        from ora.agent.toolkit.lib.motion_plan import move, route
        from ora.agent.toolkit.tools.budgets import native_action_budget
        from ora.agent.toolkit.tools.move_pose._smooth import DEFAULT_SPEED, speeds

        targets = request["targets"]
        pace, settle = speeds(self.config)[DEFAULT_SPEED]
        planned = move.plan(
            self.api, starts={arm: [float(v) for v in effectors[arm]["pose_xyz_xyzw"]] for arm in targets},
            goals={arm: [float(v) for v in pose] for arm, pose in targets.items()},
            start_joints=self._joints(effectors, targets), pace=pace, settle=settle,
            native_action_budget=native_action_budget("move_pose"), route_kind="interaction",
            interactions=dict(request["interactions"]), capabilities=(route.STAGED_CAPABILITY,),
            global_planner=self.global_planner)
        return planned, planned.path.route, None

    def _home(self, effectors, request):
        from ora.agent.toolkit.lib.motion_plan import recovery
        from ora.agent.toolkit.tools.budgets import native_action_budget
        from ora.agent.toolkit.tools.recover_pose._trajectory import SETTLE, _home_options, paces

        values, anchors, options, warnings = self.state(), {}, {}, []
        for arm in request["arms"]:
            stored = values[f"exec.recovery.{arm}.home"]
            anchors[arm] = {"strategy": "home", "arm_joint_qpos": list(stored["arm_joint_qpos"]),
                            "pose_xyz_xyzw": list(stored["pose_xyz_xyzw"])}
            options[arm] = _home_options(self.api, self.config, arm,
                                         anchors[arm], effectors[arm], self.workspace_bounds_m, warnings)
        planned = recovery.plan(self.api, options=options, start_joints=self._joints(effectors, request["arms"]),
                                goals={arm: anchor["pose_xyz_xyzw"] for arm, anchor in anchors.items()},
                                paces=paces(self.config), settle=SETTLE,
                                native_action_budget=native_action_budget("return_home"))
        return planned, "retract_then_unwind", {"targets": {arm: "home" for arm in anchors},
                                                "recovery_targets": anchors}

    def _retrace(self, effectors, request):
        from ora.agent.toolkit.lib.motion_plan import checks, recovery
        from ora.agent.toolkit.tools.budgets import native_action_budget
        from ora.agent.toolkit.tools.recover_pose._trajectory import SETTLE, paces

        values, start = self.state(), self._joints(effectors, request["arms"])
        options, goals, stated = {}, {}, {}
        for arm in request["arms"]:
            moves = (values.get(f"exec.recovery.{arm}.undo") or {}).get("moves") or []
            if not moves:
                raise checks.PlanRefusal(f"the {arm} arm has no move to retrace", code="retrace_nothing_to_undo")
            top = moves[-1]
            nodes, info = recovery.retrace_nodes(top["arm_joint_qpos"], top["measured_path_rad"], start[arm])
            if nodes is None:
                raise checks.PlanRefusal(f"the {arm} arm is off its last move's path", code="retrace_arm_off_path",
                                         constraints=dict(info))
            options[arm] = [recovery.Option(recovery.RETRACE, (nodes,), {"retrace": info})]
            goals[arm] = top["pose_xyz_xyzw"]
            stated[arm] = {"moves": len(moves), "arm_joint_qpos": list(top["arm_joint_qpos"])}
        planned = recovery.plan(self.api, options=options, start_joints=start, goals=goals, paces=paces(self.config),
                                settle=SETTLE, native_action_budget=native_action_budget("recover_pose"))
        return planned, recovery.RETRACE, {"targets": {arm: "retrace" for arm in stated}, "retrace": stated}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ora qualify robodojo motion",
        description=(
            "Run the scripted RoboDojo motion v3 qualification on the qualified host. "
            "Requires the explicit runtime configuration JSON (the compatibility package's "
            "runtime_configuration) and writes one device-keyed record."
        ),
    )
    parser.add_argument("--runtime-config", type=Path, required=True, help="RoboDojo runtime configuration JSON")
    parser.add_argument("--output", type=Path, required=True, help="evidence directory for the worker and record")
    parser.add_argument("--gate", type=Path, default=None,
                        help="optional device-keyed gate path receiving the identical record bytes")
    arguments = parser.parse_args(argv)
    runtime = json.loads(arguments.runtime_config.read_text(encoding="utf-8"))
    if os.environ.get("ORA_EXECUTION_HOST") != runtime.get("required_host"):
        raise SystemExit("the host marker must name the Runtime's RoboDojo host")
    runtime, stack_source = motion_stack(runtime)
    from ora.agent.toolkit.host.motion import profile_entry

    motion_profile = profile_entry(runtime.get("motion_profile"), root=PROJECT_ROOT)
    device = str(runtime["cuda_visible_devices"])
    free_mib = _free_memory_mib(device)
    if free_mib < 20000:
        raise SystemExit(f"GPU {device} has {free_mib} MiB free; the RoboDojo worker needs 20000")
    git = lambda *args: subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    source_commit = git("rev-parse", "HEAD")
    source_dirty = bool(git("status", "--porcelain"))
    # The worker times its moves as a run's preflight bound them (motion_timing.bind_preflight reads the
    # preflight.json beside its output root); without the record it would fall back to motion_pace's timing.
    resource = runtime.get("resource_binding")
    timing = motion_timing_check(runtime["motion_timing"], environment_overrides=(
        resource.get("worker_environment") if isinstance(resource, Mapping) else None))
    if not timing["passed"]:
        raise SystemExit(f"the motion timing worker did not start: {timing['detail']}. No worker was started.")
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "preflight.json").write_text(json.dumps({"checks": [timing]}, indent=2) + "\n",
                                                     encoding="utf-8")

    from ora.environment.benchmarks.robodojo import NATIVE_ACTION_LIMITS
    from ora.environment.factory import build_robodojo_worker_session

    task = str(runtime["task_variant"])
    native_action_limit = NATIVE_ACTION_LIMITS[task]
    environment_root = arguments.output / "environment"
    session = build_robodojo_worker_session(
        runtime=runtime, output_root=environment_root, task=task, seed=0,
        max_sim_steps=native_action_limit * 10, action_codec="motion_action_v3",
    )
    observation_root = arguments.output / "observations"
    started = time.time()

    def observe() -> Mapping[str, Any]:
        observation, _images = session.observe(observation_root)
        # observation_sha256 is Runtime-layer identity: the hash of the
        # persisted observation.json, exactly as the context projector
        # computes it for live runs.
        manifest = observation_root / str(observation["observation_id"]) / "observation.json"
        return {**observation, "observation_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}

    def current_poses(observation: Mapping[str, Any]) -> dict[str, list[float]]:
        return {arm: list(state["pose_xyz_xyzw"])
                for arm, state in observation["proprioception"]["end_effectors"].items()}

    def restart_round(index: int) -> Mapping[str, Any]:
        # The Supervisor rotates the previous round's verdict before a reset;
        # the gate keeps it as round evidence instead of discarding it.
        verdict = environment_root / "official-verdict.json"
        if verdict.exists():
            archive = environment_root / "rounds" / f"round-{index - 1:04d}" / "official-verdict.json"
            archive.parent.mkdir(parents=True, exist_ok=True)
            verdict.replace(archive)
        return session.restart_round(index, seed=0)

    # The live action envelope the default surfaces bind; return_home's lift stays under its ceiling.
    profile = runtime.get("workspace_profile", ROBODOJO_ARX_X5_TABLETOP_V1)
    workspace = resolve_workspace_profile(profile)
    close_record: dict[str, Any] = {}
    try:
        owner = bind_execution_state(session)

        def execute(action: Mapping[str, Any]) -> Mapping[str, Any]:
            # As a tool lends it: the most native actions one such action may use.
            with owner.limited_action(native_action_cap(action)):
                return session.execute(action)

        reset = dict(session.start())
        planner = MatrixPlanner(kinematics=session.align_plan, state=lambda: owner.writer.values(),
                                global_planner=True, workspace_bounds_m=workspace["bounds_m"],
                                motion_profile=motion_profile)
        results = run_cases(
            observe=observe, execute=execute, plan=planner, probe=session.probe_motion,
            restart_round=restart_round,
            status=session.status, persist_verdict=session.persist_official_verdict,
            current_poses=current_poses, native_action_limit=native_action_limit,
            free_route=planner.free_route, measure_jaw_state=session.measure_jaw_state,
        )
    finally:
        close_started = time.monotonic()
        session.close()
        close_record["close_seconds"] = round(time.monotonic() - close_started, 3)
        try:
            process = json.loads((environment_root / "environment-session.json").read_text(encoding="utf-8"))
            close_record["worker_returncode"] = process.get("worker_returncode")
            close_record["close_error"] = process.get("close_error")
        except (OSError, ValueError):
            close_record["worker_returncode"] = None
    timing_detail = timing["detail"] if isinstance(timing["detail"], Mapping) else {}
    record = build_qualification_record(
        source_commit=source_commit,
        host={
            "execution_host": os.environ.get("ORA_EXECUTION_HOST", "unknown"),
            "cuda_visible_devices": device, "free_memory_mib_at_start": free_mib,
            "device_binding": {key: value for key, value in (reset.get("device_binding") or {}).items()
                               if key != "startup_log_prefix"},
            "runtime_lease": None,
        },
        session_identity={
            "task": task, "eval_seed": 0, "layout_id": 0, "layout_file": runtime.get("layout_file"),
            "layout_sha256": runtime.get("layout_sha256"), "action_codec": "motion_action_v3",
            "native_action_limit": native_action_limit, "rounds": 2,
            "source_revisions": (reset.get("source_binding") or {}).get("source_revisions"),
            "source_manifest_sha256": (reset.get("source_manifest") or {}).get("sha256"),
            "source_dirty": source_dirty,
            "motion_planner": {key: runtime["motion_planner"].get(key)
                               for key in ("implementation", "required_revision", "robot", "world_model")},
            "motion_timing": {"implementation": runtime["motion_timing"].get("implementation"),
                              "python_sha256": timing_detail.get("python_sha256")},
            "motion_profile": (None if motion_profile is None else
                               {"profile_id": motion_profile["profile_id"], "path": motion_profile["path"],
                                "sha256": motion_profile["sha256"]}),
            "motion_stack_source": stack_source, "free_route": planner.free_route,
            "workspace_profile": {"id": profile, "sha256": workspace["profile_sha256"]},
        },
        case_results=results,
        evidence={
            "environment_root": str(environment_root), "observation_root": str(observation_root),
            "wall_seconds": round(time.time() - started, 3), **close_record,
        },
    )
    body = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    target = arguments.output / "robodojo-sparse-motion-qualification.json"
    target.write_text(body, encoding="utf-8")
    written = {"record": str(target)}
    # A gate path carries only passing admission evidence; failing runs stay
    # in their qualification directory so an old passing gate is never
    # overwritten by a later regression.
    if arguments.gate is not None and record["passed"]:
        arguments.gate.parent.mkdir(parents=True, exist_ok=True)
        arguments.gate.write_text(body, encoding="utf-8")
        written["gate"] = str(arguments.gate)
    print(json.dumps({
        "passed": record["passed"], "cases": len(record["cases"]),
        "failed": [case["case_id"] for case in record["cases"] if not case["passed"]],
        "missing": record["missing_cases"],
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), **written,
    }, indent=2))
    return 0 if record["passed"] else 1
