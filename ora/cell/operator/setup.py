"""``ora cell setup``: a new cell from its config to its first play, one step at a time.

``status`` says, for each step, whether the cell's files and a passive look at the cell
show it ``done``, ``missing`` or ``stale`` (done once, but no longer true: a gripper whose
reading left its recorded limits after a power cycle, a table measured before the
calibration that moved the cameras), and which step comes next. It moves nothing; it reads
the arms only when no program holds the cell.

``run`` does the steps that are not done, in order, each asking the person through an
asker (``asking.py``) before anything moves, and stops at the first that fails or is
declined; run again, it goes on from there. ``redo`` repeats steps that are done.

1. ``devices``: the buses are up and every camera of the config is connected.
2. ``rest``: the arms' rest pose, recorded with the arms folded at rest.
3. ``grippers``: each gripper's [closed, open] limits, measured, and which way it opens,
   as the person sees it. Redo after a power cycle when the reading has left the limits.
4. ``fixed_cameras``: a first pose of every fixed camera, from the checkerboard seen by the
   wrist cameras at a few poses near rest.
5. ``handeye``: the guided hand-eye calibration (``handeye.py``): fixed cameras, wrist
   mounts, arm bases and joint zeros from one fit.
6. ``table``: the table top's height, from the fixed cameras' depth with that calibration.
7. ``verify``: the table each wrist camera sees, placed through its arm, is level and at
   the measured height.

The checkerboard (square size and inner corners) is asked once and kept in the
calibration (``board``). Each question says which step asks it (``details.for``: ``devices``,
``rest``, ``fixed_cameras``; the hand-eye placements carry their ``placement`` number).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np

from .. import bundle
from ..calibration import gripper_readings_off
from ..errors import CellError
from ..geometry import table_plane
from ..lock import CellLock, holder
from . import handeye, motion, tools
from .asking import Asker, arm_words
from .context import Cell

STEPS = (
    ("devices", "设备连接", "CAN 总线和每台相机都接好了"),
    ("rest", "记录静止位", "两只手臂折叠在静止位时的关节读数"),
    ("grippers", "夹爪限位和方向", "每只夹爪的合、开两端，以及你看到的开合方向"),
    ("fixed_cameras", "固定相机粗定位", "固定相机大概在哪里，手眼标定从这里出发"),
    ("handeye", "手眼标定", "固定相机、腕部相机安装、手臂底座和关节零点，一次求解"),
    ("table", "桌面高度", "用标定后的固定相机重新测桌面"),
    ("verify", "验证", "腕部相机经过手臂看到的桌面是平的，高度和测量一致"),
)
REQUIRED_FOR_PLAY = ("devices", "rest", "grippers", "fixed_cameras", "handeye", "table")
TABLE_RANGE_M = (0.3, 2.0)
TASK_ANSWER_S = 1800.0
MOTION_ANSWER_S = 600.0
TABLE_EXTENT = {"center_xy_m": [0.4, 0.0], "size_xy_m": [1.2, 1.2]}


def _step(step_id: str, state: str, detail: str) -> dict[str, Any]:
    title = next(title for i, title, _ in STEPS if i == step_id)
    return {"id": step_id, "title": title, "state": state, "detail": detail}


# ---------------------------------------------------------------- status
def status(cell: Cell, *, read: bool = True) -> dict[str, Any]:
    """Each step's state, the next one and whether the cell is ready for play. Moves nothing."""

    calibration = cell.calibration()
    steps = [_devices(cell), _rest(calibration), _grippers(cell, calibration, read=read),
             _fixed(cell, calibration), _handeye(calibration), _table(calibration), _verify(calibration)]
    following = next((s["id"] for s in steps if s["state"] != "done"), None)
    ready = all(s["state"] == "done" for s in steps if s["id"] in REQUIRED_FOR_PLAY)
    return {"steps": steps, "next": following, "ready_for_play": ready}


def _devices(cell: Cell) -> dict[str, Any]:
    problems = []
    for arm, channel in cell.adapter.buses.items():
        state = cell.bus.interface(channel)
        if not state.get("present") or not state.get("up"):
            problems.append(f"{arm_words([arm])}的总线 {channel} {'没有启动' if state.get('present') else '不存在'}")
    try:
        connected = cell.camera_driver.connected()
        problems += [f"相机 {role}（{c.serial}）没有连接" for role, c in cell.config.cameras.items()
                     if c.serial not in connected]
    except CellError as exc:
        problems.append(f"查不了相机（{exc}）")
    if problems:
        return _step("devices", "missing", "；".join(problems))
    return _step("devices", "done", f"{len(cell.adapter.buses)} 条总线、{len(cell.config.cameras)} 台相机都在")


def _rest(calibration: Mapping[str, Any]) -> dict[str, Any]:
    if not calibration.get("rest_reference"):
        return _step("rest", "missing", "还没有记录")
    at = calibration.get("rest_reference_unix_s")
    return _step("rest", "done", "已记录" + (f"（{time.strftime('%m-%d %H:%M', time.localtime(at))}）" if at else ""))


def _grippers(cell: Cell, calibration: Mapping[str, Any], *, read: bool) -> dict[str, Any]:
    limits = calibration.get("gripper_limits") or {}
    if not limits:
        return _step("grippers", "missing", "还没有测过")
    if not read or holder() is not None:
        return _step("grippers", "done", "已测过；机器正被占用，这次没有核对读数")
    try:
        with CellLock("cell operator (setup status: gripper readings)"):
            readings = cell.read_arms()
    except CellError as exc:
        return _step("grippers", "done", f"已测过；没能核对读数（{exc}）")
    off = [f"{arm_words([arm])}读数 {item['reading_rad']:.2f} 不在记录的 [{item['low_rad']:.2f}, {item['high_rad']:.2f}] 里"
           for arm, item in gripper_readings_off(
               limits, {arm: reading.gripper_raw for arm, reading in readings.items()}).items()]
    if off:
        return _step("grippers", "stale", "；".join(off) + "：上电后读数变了，要重测")
    return _step("grippers", "done", "读数都在记录的限位里")


def _fixed(cell: Cell, calibration: Mapping[str, Any]) -> dict[str, Any]:
    cameras = calibration.get("cameras") or {}
    missing = [role for role in cell.fixed_cameras() if role not in cameras]
    if missing:
        return _step("fixed_cameras", "missing", "还没有位置：" + "、".join(missing))
    return _step("fixed_cameras", "done", "、".join(cell.fixed_cameras()) + " 都有位置")


def _handeye(calibration: Mapping[str, Any]) -> dict[str, Any]:
    fit = calibration.get("bundle_fit")
    if not fit:
        return _step("handeye", "missing", "还没有做")
    held = [v["mm"] for placement in (fit.get("held_out") or {}).values() for views in placement.values() for v in views]
    detail = f"留一验证误差中位 {np.median(held):.1f} mm" if held else "已做（没有留一验证）"
    return _step("handeye", "done", detail)


def _table(calibration: Mapping[str, Any]) -> dict[str, Any]:
    table = calibration.get("table")
    if not table:
        return _step("table", "missing", "还没有测")
    fitted = (calibration.get("bundle_fit") or {}).get("fitted_unix_s")
    measured = table.get("measured_unix_s")
    if calibration.get("bundle_fit") and not (measured and (fitted is None or measured >= fitted)):
        return _step("table", "stale", f"桌面高度 {table['top_z_m']:.3f} m 是在手眼标定之前测的")
    return _step("table", "done", f"桌面高度 {table['top_z_m']:.3f} m")


def _verify(calibration: Mapping[str, Any]) -> dict[str, Any]:
    verified = calibration.get("verified")
    table = calibration.get("table") or {}
    if not verified:
        return _step("verify", "missing", "还没有验证")
    if (table.get("measured_unix_s") or 0) > verified.get("at", 0):
        return _step("verify", "stale", "上次验证之后桌面又测过")
    return _step("verify", "done" if verified.get("passed") else "stale",
                 "通过" if verified.get("passed") else "上次没通过：" + verified.get("why", ""))


# ---------------------------------------------------------------- run
def run(cell: Cell, asker: Asker, *, redo: tuple[str, ...] = (), only: tuple[str, ...] = (),
        progress: Callable[[str], None] = lambda _m: None,
        event: Callable[[Mapping[str, Any]], None] = lambda _e: None) -> dict[str, Any]:
    """Do the steps that are not done (or named in ``redo``), in order; stop at the first that fails."""

    known = [i for i, _, _ in STEPS]
    for name in (*redo, *only):
        if name not in known:
            raise CellError(f"unknown setup step {name!r}: one of {', '.join(known)}")
    done: list[str] = []
    for step in status(cell)["steps"]:
        name = step["id"]
        if only and name not in only:
            continue
        if step["state"] == "done" and name not in redo:
            continue
        event({"step": name, "state": "running"})
        progress(f"{step['title']}：{step['detail']}")
        try:
            outcome = DO[name](cell, asker, progress)
        except CellError as exc:
            outcome = {"failed": str(exc)}
        if outcome.get("failed"):
            event({"step": name, "state": "failed", "why": outcome["failed"]})
            return {"done": done, "failed": name, "why": outcome["failed"], "status": status(cell, read=False)}
        event({"step": name, "state": "done"})
        done.append(name)
    return {"done": done, "failed": None, "status": status(cell)}


def _confirm(asker: Asker, kind: str, title: str, text: str, details: Mapping[str, Any] | None = None) -> str | None:
    """None when approved, else why not."""

    ok, reason = asker.confirm(kind, title, text, dict(details or {}),
                               MOTION_ANSWER_S if kind == "motion" else TASK_ANSWER_S)
    return None if ok else f"没有批准：{reason}"


def _do_devices(cell: Cell, asker: Asker, progress: Callable[[str], None]) -> dict[str, Any]:
    while True:
        found = _devices(cell)
        if found["state"] == "done":
            return {}
        why = _confirm(asker, "task", "接好设备", found["detail"] + "。接好后批准，我再检查一次；拒绝则停在这一步。",
                       {"for": "devices"})
        if why:
            return {"failed": why}


def _do_rest(cell: Cell, asker: Asker, progress: Callable[[str], None]) -> dict[str, Any]:
    why = _confirm(asker, "task", "把手臂放到静止位",
                   "把两只手臂都折叠放到静止位（断电时可以用手扶过去），放好后批准。我只读关节，不会动手臂。",
                   {"for": "rest"})
    if why:
        return {"failed": why}
    tools.record_rest(cell)
    return {}


def _do_grippers(cell: Cell, asker: Asker, progress: Callable[[str], None]) -> dict[str, Any]:
    why = _confirm(asker, "motion", "测夹爪限位",
                   "两只夹爪会依次张开、闭合一次（用力很小，每个方向最多推 2 秒），手臂在静止位保持不动。手离开夹爪。")
    if why:
        return {"failed": why}
    found = motion.grippers(cell, save=True)
    if found["stopped"] or not found["gripper_limits"]:
        return {"failed": "测夹爪被停止了"}
    progress("夹爪限位：" + ", ".join(f"{a} {np.round(v, 3).tolist()}" for a, v in found["gripper_limits"].items()))
    tools.orient_grippers(cell, asker)
    return {}


def _board(cell: Cell, asker: Asker) -> bundle.Board:
    stored = cell.calibration().get("board") or {}
    pattern = tuple(stored["pattern"]) if stored.get("pattern") else None
    board = handeye.ask_board(asker, stored.get("square_mm"), pattern)
    if stored.get("square_mm") != board.square_m * 1000 or pattern != board.pattern:
        calibration = cell.calibration()
        calibration["board"] = {"square_mm": board.square_m * 1000, "pattern": list(board.pattern)}
        cell.save_calibration(calibration)
    return board


def _do_fixed(cell: Cell, asker: Asker, progress: Callable[[str], None]) -> dict[str, Any]:
    board = _board(cell, asker)
    why = _confirm(asker, "task", "摆放标定板",
                   "把标定板平放在两臂前方中间，固定住；两只手臂抬起时腕部相机要能看到它，固定相机也要能看到。摆好后批准。",
                   {"for": "fixed_cameras", "arms": list(cell.adapter.arms)})
    if why:
        return {"failed": why}
    why = _confirm(asker, "motion", "两臂抬起拍标定板",
                   "两只手臂会一起在静止位附近抬起几次（肩、肘各转约 15–25°），用腕部相机拍标定板，夹爪开合一次，然后回到起始位并放松。")
    if why:
        return {"failed": why}
    report = motion.calibrate_with_board(cell, square_mm=board.square_m * 1000, pattern=board.pattern, save=True)
    if report.get("stopped"):
        return {"failed": f"动作被停止：{report['stopped']}"}
    if not report.get("placed"):
        return {"failed": report.get("why") or "没有一台固定相机定上位置"}
    progress("固定相机位置：" + ", ".join(f"{r} {v}" for r, v in report["placed"].items()))
    return {}


def _do_handeye(cell: Cell, asker: Asker, progress: Callable[[str], None]) -> dict[str, Any]:
    board = _board(cell, asker)
    report = handeye.calibrate(cell, asker=asker, square_mm=board.square_m * 1000, pattern=board.pattern,
                               save=True, progress=progress)
    if not report.get("saved"):
        return {"failed": report.get("not_saved") or "没有写入"}
    return {}


def measure_table(cell: Cell) -> dict[str, Any]:
    """The table top's height from the fixed cameras' depth (the median of their planes)."""

    cameras = cell.calibration().get("cameras") or {}
    roles = [r for r in cell.fixed_cameras() if r in cameras]
    with CellLock("cell operator (measuring the table)"):
        shots = motion.capture(cell, roles, frames=5)
    planes = {}
    for role, data in shots.items():
        plane = table_plane(data["depth"], data["intrinsic_cv"], np.asarray(cameras[role]["cam2world_cv"], float),
                            near=TABLE_RANGE_M[0], far=TABLE_RANGE_M[1])
        if plane is not None and plane["tilt_deg"] < 5.0:
            planes[role] = plane
    if not planes:
        raise CellError("no fixed camera sees a level table")
    return {"top_z_m": float(np.median([p["height_m"] for p in planes.values()])),
            "planes": {r: {"height_m": round(p["height_m"], 4), "tilt_deg": round(p["tilt_deg"], 2)}
                       for r, p in planes.items()}}


def _do_table(cell: Cell, asker: Asker, progress: Callable[[str], None]) -> dict[str, Any]:
    found = measure_table(cell)
    calibration = cell.calibration()
    extent = {k: v for k, v in (calibration.get("table") or TABLE_EXTENT).items() if k in TABLE_EXTENT}
    calibration["table"] = {**TABLE_EXTENT, **extent, "top_z_m": found["top_z_m"], "measured_unix_s": time.time(),
                            "source": "ora cell setup: fixed cameras' depth " + ", ".join(
                                f"{r} {p['height_m']:.3f}" for r, p in found["planes"].items())}
    cell.save_calibration(calibration)
    progress(f"桌面高度 {found['top_z_m']:.3f} m（{found['planes']}）")
    return {}


def _do_verify(cell: Cell, asker: Asker, progress: Callable[[str], None]) -> dict[str, Any]:
    found = tools.wrist_check(cell)
    failed = {arm: "；".join(r.get("problems") or [r.get("why") or "没通过"])
              for arm, r in found["arms"].items() if r.get("passed") is False}
    calibration = cell.calibration()
    calibration["verified"] = {"at": time.time(), "passed": not failed, "why": "；".join(f"{a}: {w}" for a, w in failed.items()),
                               "arms": found["arms"]}
    cell.save_calibration(calibration)
    if failed:
        return {"failed": "腕部相机看到的桌面和标定不一致：" + calibration["verified"]["why"]}
    table = (calibration.get("table") or {}).get("top_z_m")
    progress("验证通过：" + "；".join(
        f"{arm_words([a])}看到的桌面倾斜 {r['tilt_deg']:.1f}°、高度 {r['height_m']:.3f} m"
        + (f"（差 {abs(r['height_m'] - table) * 1000:.0f} mm）" if table is not None else "")
        for a, r in found["arms"].items() if r.get("height_m") is not None))
    return {}


DO: dict[str, Callable[[Cell, Asker, Callable[[str], None]], dict[str, Any]]] = {
    "devices": _do_devices, "rest": _do_rest, "grippers": _do_grippers, "fixed_cameras": _do_fixed,
    "handeye": _do_handeye, "table": _do_table, "verify": _do_verify}
