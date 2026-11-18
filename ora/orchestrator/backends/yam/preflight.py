"""Fresh admission checks for a YAM runtime: the worker interpreter, the pinned i2rt
source and station model, the shared timing and planner workers a runtime names, and for
a real rig the CAN buses and the RealSense cameras. Nothing here enables a motor or opens
a camera stream."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.environment.benchmarks.yam import I2RT_REVISION, STATION_MODEL_SHA256

from ora.orchestrator.backends.registry import PreflightContext
from ora.orchestrator.preflight.checks import check as _check, execution_host_check, motion_timing_check

_PROBE = """
import hashlib, importlib.metadata as m, json, pathlib
import i2rt
root = pathlib.Path(i2rt.__file__).resolve().parent
model = root / "robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.xml"
out = {"source": str(root.parent), "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest() if model.is_file() else None,
       "station_urdf": str(model.with_suffix(".urdf")) if model.with_suffix(".urdf").is_file() else None}
for p in ("mujoco", "mink", "pyrealsense2", "python-can"):
    try: out[p] = m.version(p)
    except m.PackageNotFoundError: out[p] = None
try:
    import pyrealsense2 as rs
    out["realsense_serials"] = sorted(d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices())
except Exception as exc:
    out["realsense_serials"] = None; out["realsense_error"] = str(exc)
print(json.dumps(out))
"""


def preflight_runtime(plan: Mapping[str, Any], context: PreflightContext) -> list[dict[str, Any]]:
    return yam_runtime(plan, runner=context.runner)


def motion_planner_check(planner: Mapping[str, Any], robot_file: str | None) -> dict[str, Any]:
    """Start the shared planner once for the station's arm; the passing detail is what the worker binds
    (``environment/backends/yam/shared_planner.py``: ``python`` and ``root``), the robot file the i2rt station
    URDF."""

    from ora.environment.benchmarks.yam import MOTION_PLANNER_PREFLIGHT_CHECK as PREFLIGHT_CHECK
    from ora.environment.controllers.motion_planner import planner_client

    python, root = os.environ.get(str(planner["python_env"])), os.environ.get(str(planner["root_env"]))
    if not python or not root or not robot_file:
        return _check(PREFLIGHT_CHECK, False, {"missing": [name for name, value in (
            (planner["python_env"], python), (planner["root_env"], root), ("station_urdf", robot_file)) if not value]})
    python = os.path.abspath(os.path.expanduser(python))
    try:
        with tempfile.TemporaryDirectory(prefix="ora-planner-preflight-") as directory:
            client = planner_client(planner, python_executable=Path(python), root=Path(root),
                                    output_root=Path(directory), robot_file=Path(robot_file))
            try:
                binding = client.start()
            finally:
                client.close()
    except Exception as exc:  # noqa: BLE001 - any failure to start is this check's detail
        return _check(PREFLIGHT_CHECK, False, str(exc))
    return _check(PREFLIGHT_CHECK, True, {
        "python": python, "root": str(Path(root).expanduser().resolve()), "revision": binding["revision"],
        "python_sha256": binding["python_sha256"], "device": binding["device"], "robot": binding["robot"],
        "world_model": binding["world_model"], "robot_file": binding["robot_file"],
        "robot_file_sha256": binding["robot_file_sha256"]})


def _can_interface(name: str) -> dict[str, Any]:
    root = Path("/sys/class/net") / name
    if not root.exists():
        return {"present": False}
    state = (root / "operstate").read_text().strip() if (root / "operstate").exists() else "?"
    return {"present": True, "operstate": state, "up": state in {"up", "unknown"}}


def yam_runtime(
    plan: Mapping[str, Any], *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[dict[str, Any]]:
    runtime = plan["runtime"]
    mode = runtime.get("rig_mode", "sim")
    checks = [execution_host_check(runtime, os.environ.get("ORA_EXECUTION_HOST"))] if mode != "sim" else []
    raw_python = os.environ.get(str(runtime.get("python_env")))
    python = Path(os.path.abspath(os.path.expanduser(raw_python))) if raw_python else None
    executable = bool(python and python.is_file() and os.access(python, os.X_OK))
    checks.append(_check("environment_python", executable, {"env": runtime.get("python_env"),
                                                            "executable": str(python) if python else None}))
    timing = runtime.get("motion_timing")
    if isinstance(timing, Mapping):
        # The worker reads this check's binding from the run's record (motion_timing.bind_preflight).
        checks.append(motion_timing_check(timing))
    if not executable:
        return checks
    result = runner([str(python), "-c", _PROBE], capture_output=True, text=True, timeout=120, check=False)
    try:
        probe = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        probe = {"error": result.stderr.strip()[-600:]}
    checks.append(_check("yam_packages", all(probe.get(p) for p in ("mujoco", "mink")), {
        p: probe.get(p) for p in ("mujoco", "mink", "pyrealsense2", "python-can")}))
    source = probe.get("source")
    if source and (Path(source) / ".git").exists():
        head = runner(["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        dirty = runner(["git", "-C", source, "status", "--porcelain"], capture_output=True, text=True, check=False)
        checks.append(_check("i2rt_source", head.stdout.strip() == I2RT_REVISION and not dirty.stdout.strip(),
                             {"expected": I2RT_REVISION, "observed": head.stdout.strip(),
                              "dirty": bool(dirty.stdout.strip())}))
    else:
        checks.append(_check("i2rt_source", False, {"source": source, "reason": "not a git checkout"}))
    checks.append(_check("station_model", probe.get("model_sha256") == STATION_MODEL_SHA256,
                         {"expected": STATION_MODEL_SHA256, "observed": probe.get("model_sha256")}))
    planner = runtime.get("motion_planner")
    if isinstance(planner, Mapping):
        # The worker reads this check's binding from the run's record (shared_planner.located).
        checks.append(motion_planner_check(planner, probe.get("station_urdf")))
    if mode in {"read", "on"}:
        from ora.cell import config as cell_config
        from ora.cell.errors import CellError

        try:
            cell = cell_config.load(runtime.get("cell_config"))
        except CellError as exc:
            checks.append(_check("cell_config", False, {"error": str(exc)}))
            return checks
        checks.append(_check("cell_config", cell.sim_state is None, {"cell": cell.cell_id, "path": str(cell.path)} | (
            {} if cell.sim_state is None else {"error": "a simulated cell (robot.sim) has no arms to drive: "
                                                        "rig_mode read and on need a real cell"})))
        buses = {arm: _can_interface(channel) for arm, channel in cell.buses.items()}
        checks.append(_check("can_buses", all(b.get("up") for b in buses.values()), buses))
        wanted = sorted(cell.serials.values())
        seen = probe.get("realsense_serials") or []
        checks.append(_check("realsense_cameras", bool(wanted) and set(wanted) <= set(seen),
                             {"configured": wanted, "connected": seen}))
    if mode == "on":
        checks.append(_check("operator_present", runtime.get("operator_present") is True,
                             {"declared": runtime.get("operator_present")}))
    return checks
