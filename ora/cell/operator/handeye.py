"""Guided hand-eye calibration: the person places a checkerboard, the arms look at it, one fit measures the cell.

``calibrate`` runs the whole procedure (``ora cell calibrate --method hand-eye``):

1. The board: its square, measured, and its inner corners (asked when not given).
2. For each placement, the person lays the board flat where the named arms reach and
   both fixed cameras see it, and approves (``task``). The fixed cameras locate it; each
   named arm gets up to ``POSES_PER_ARM`` wrist views planned around it in the robot
   model, every pose and every segment between them checked against the table (the
   board's own plane when the calibration has no table), the other arm and the arm's own
   links. The plan is asked as a ``motion``; one arm moves at a time, the other at rest,
   and the arms return to where they started and are released. A board no fixed camera
   sees, too few safe views, a declined motion or a stopped one ask for the placement
   again, with the reason; declining a placement ends the captures.
3. One fit of everything (``bundle.solve``) and its leave-one-placement-out check
   (``bundle.validate``), then ``write``: approved, the old calibration is copied beside
   itself and the new one saved.

The first placement is shared by every arm (it ties the arms into one world); the others
alternate between arms, so the default five give each of two arms three. Every question
goes through an asker (``asking.py``): the approval broker (the board, ``ora cell approve``
and the play terminal answer it) or the terminal. The cell lock is held from the first placement to
the last capture; between motions the arms are released at rest and the label says so,
so ``ora cell home`` moves nothing then. During a motion the operator's ``home`` stops it
and the arms go back to where they started.
"""

from __future__ import annotations

import json
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .. import bundle
from ..cameras.charuco import chessboard_pose
from ..errors import CellError
from ..lock import AT_REST, HOMES_ON_SIGNAL, CellLock
from .asking import Asker, arm_words
from .context import Cell
from .motion import HOLD_S, RECORD_ROOT, Mover, Stop, capture

PLACEMENTS = 5
POSES_PER_ARM = 8
MIN_POSES = 5
MIN_TABLE_M = 0.04                  # every moving geom above the table (or the board's plane)
MIN_INTER_ARM_M = 0.15
MIN_SELF_M = 0.02
MARGIN_PX = 45
DISTANCES_M = (0.27, 0.34)          # camera to board centre
TILTS_DEG = (0, 22, 38)             # view from straight above, then leaning
ROLLS_DEG = (-35, 0, 35)
BOARD_TABLE_SIZE_M = (0.8, 0.8)     # the table modelled from the board when none is calibrated
TASK_ANSWER_S = 1800.0
MOTION_ANSWER_S = 600.0
WRITE_ANSWER_S = 3600.0
HELD_OUT_MM_FLAG = 10.0             # a placement's median held-out error
FIT_PX_FLAG = 4.0                   # a camera's median fit error
OUTLIER_PX = 10.0


# Where to put the board, in turn for each arm: near and turned one way, far and turned the
# other, then out to the side. Views of one placement constrain a wrist's mount; only
# placements that differ in position and angle tell its joint zeros and base apart.
BOTH_HINT = "两臂之间正前方，离两个底座约 35 cm，板子正放"
ARM_HINTS = ("{arm}前方近处，离它的底座约 25–30 cm，板子转约 30°",
             "{arm}前方远处，离它的底座约 40–45 cm，板子往另一边转约 20°",
             "{arm}外侧（远离另一只手臂），离它的底座约 35 cm，板子正放")
EXTRA_PLACEMENTS = 2


def schedule(arms: Sequence[str], count: int) -> list[tuple[str, ...]]:
    """Which arms look at each placement: all of them first, then one at a time in turn."""

    arms = tuple(arms)
    if len(arms) == 1:
        return [arms] * count
    return [arms] + [(arms[i % len(arms)],) for i in range(count - 1)]


def placement_hint(turns: Sequence[tuple[str, ...]], index: int) -> str:
    """Where the board goes for placement ``index`` of ``turns``: a different spot for each turn of an arm."""

    shown = turns[index]
    if len(shown) > 1:
        return BOTH_HINT
    earlier = sum(1 for t in turns[:index] if t == shown)
    return ARM_HINTS[earlier % len(ARM_HINTS)].format(arm=arm_words(shown))


# ---------------------------------------------------------------- the board
def ask_board(asker: Asker, square_mm: float | None, pattern: tuple[int, int] | None) -> bundle.Board:
    while square_mm is None:
        value, reason = asker.value(
            "标定板：一个格子的边长",
            "用尺子量标定板上 10 个格子的总长度，除以 10，输入一个格子的边长（毫米），例如 20。"
            "这个数决定标定里所有距离的比例：打印缩放差 2%，结果就差 2%。",
            {"unit": "mm", "example": "20"}, TASK_ANSWER_S)
        if value is None:
            raise CellError(f"no square size: {reason}")
        try:
            square_mm = float(value.replace("mm", "").strip())
        except ValueError:
            square_mm = None
            continue
        if not 3.0 <= square_mm <= 150.0:
            square_mm = None
    while pattern is None:
        value, reason = asker.value(
            "标定板：内角点数",
            "数黑白格相交的内角点：横向几个、纵向几个（不算最外圈的边），输入 列x行，例如 11x8。",
            {"unit": "columns x rows", "example": "11x8"}, TASK_ANSWER_S)
        if value is None:
            raise CellError(f"no board pattern: {reason}")
        pattern = parse_pattern(value)
    return bundle.Board(square_mm / 1000.0, pattern)


def parse_pattern(text: str) -> tuple[int, int] | None:
    try:
        columns, rows = (int(v) for v in text.lower().replace("×", "x").replace("*", "x").split("x"))
    except ValueError:
        return None
    return (columns, rows) if columns >= 3 and rows >= 3 and columns != rows else None


def locate(cell: Cell, board: bundle.Board) -> dict[str, Any]:
    """Where each placed fixed camera puts the board (board2world), from a few frames each."""

    cameras = cell.calibration().get("cameras") or {}
    roles = [r for r in cell.fixed_cameras() if r in cameras]
    out: dict[str, Any] = {}
    for role, data in capture(cell, roles, frames=3).items():
        pose = chessboard_pose(data["rgb"], data["intrinsic_cv"], data["distortion"]["coeffs"], board.square_m,
                               board.pattern, depth=data["depth"])
        if pose is None:
            out[role] = None
            out.setdefault("_why", {})[role] = _partial(data["rgb"], board)
            continue
        board2world = np.asarray(cameras[role]["cam2world_cv"], float) @ pose["board2cam"]
        out[role] = {"board2world": board2world, "reprojection_px": pose["reprojection_px"],
                     "depth_scale_ratio": pose.get("depth_scale_ratio"),
                     "center": bundle._transform(board.points(), board2world).mean(0)}
    return out


def _partial(rgb: np.ndarray, board: bundle.Board) -> str:
    """Why a camera did not find the whole board: it sees part of it, or none."""

    import cv2

    columns, rows = board.pattern
    gray = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2GRAY)
    smaller = (max(3, columns - 4), max(3, rows - 3))
    found, _ = cv2.findChessboardCornersSB(gray, smaller, flags=cv2.CALIB_CB_EXHAUSTIVE)
    if found:
        return "只看到板子的一部分：可能被夹爪或手臂挡住了一角，或者有一边出了画面"
    return "完全没看到板子"


def located_why(located: Mapping[str, Any]) -> str:
    """What the fixed cameras saw of a board none of them found whole, in plain words."""

    parts = [f"{role} {why}" for role, why in (located.get("_why") or {}).items()]
    return "没有一台固定相机看到整块标定板（" + "；".join(parts) + "）。看板的实时画面里可以对照着挪"


# ---------------------------------------------------------------- planning
def _look_at(eye: np.ndarray, target: np.ndarray, toward_arm: np.ndarray, roll: float) -> np.ndarray:
    """An OpenCV camera at ``eye`` looking at ``target``, image down toward the arm, then turned by ``roll``."""

    z = target - eye
    z /= np.linalg.norm(z)
    y = toward_arm - z * np.dot(toward_arm, z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    c, s = np.cos(roll), np.sin(roll)
    x, y = c * x + s * y, -s * x + c * y
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, y, z])
    pose[:3, 3] = eye
    return pose


def sees_board(cam2world: np.ndarray, intrinsic: np.ndarray, size: tuple[int, int], corners_world: np.ndarray,
               hidden: Sequence[Sequence[float]]) -> bool:
    """Every corner in front of the camera, inside the image with a margin, and clear of what the jaws hide."""

    width, height = size
    cam = bundle._transform(corners_world, np.linalg.inv(cam2world))
    if np.any(cam[:, 2] < 0.15) or np.any(cam[:, 2] > 0.5):
        return False
    uv = (intrinsic @ (cam / cam[:, 2:3]).T)[:2].T
    if np.any(uv[:, 0] < MARGIN_PX) or np.any(uv[:, 0] > width - MARGIN_PX) \
            or np.any(uv[:, 1] < MARGIN_PX) or np.any(uv[:, 1] > height - MARGIN_PX):
        return False
    for x0, y0, x1, y1 in hidden:
        if np.any((uv[:, 0] > x0 - 15) & (uv[:, 0] < x1 + 15) & (uv[:, 1] > y0 - 15) & (uv[:, 1] < y1 + 15)):
            return False
    return True


class Clearance:
    """The checks every planned pose and segment passes: table, the other arm, the arm's own links."""

    def __init__(self, kinematics: Any, rest: Mapping[str, np.ndarray]) -> None:
        self.kinematics = kinematics
        # The folded rest pose may already bring the wrist near the lower links: a pair may
        # come no closer than it is at rest (or MIN_SELF_M, whichever is smaller).
        kinematics.set_state(rest)
        self.self_floor = {arm: np.minimum(MIN_SELF_M, kinematics.self_distances(arm) - 0.003) for arm in rest}

    def state(self, q: Mapping[str, np.ndarray], arm: str) -> str | None:
        k = self.kinematics
        k.set_state(q)
        table = k.table_distance(arm)
        if table is not None and table < MIN_TABLE_M:
            return f"{table * 100:.1f} cm from the table"
        if len(q) > 1 and k.inter_arm_distance() < MIN_INTER_ARM_M:
            return "near the other arm"
        if np.any(k.self_distances(arm) < self.self_floor[arm]):
            return "near its own links"
        return None

    def path(self, a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], arm: str) -> str | None:
        for t in np.linspace(0.0, 1.0, 25):
            why = self.state({k: a[k] + t * (b[k] - a[k]) for k in a}, arm)
            if why:
                return why
        return None


def board_table(cell: Cell, board2world: np.ndarray, board: bundle.Board) -> dict[str, Any]:
    """The calibrated table, else the board's plane: the floor no planned pose comes near."""

    table = cell.table()
    if table is not None:
        return table
    centre = bundle._transform(board.points(), board2world).mean(0)
    return {"center_xy_m": centre[:2].tolist(), "size_xy_m": list(BOARD_TABLE_SIZE_M), "top_z_m": float(centre[2])}


def plan(cell: Cell, arms: Sequence[str], board2world: np.ndarray, board: bundle.Board,
         wrist: Mapping[str, tuple[np.ndarray, tuple[int, int]]]) -> dict[str, Any]:
    """Wrist views of the board for each arm, ordered into one safe route from rest and back.

    ``wrist`` gives each arm's wrist camera intrinsics and image size."""

    adapter = cell.adapter
    rest = {a: np.asarray(v, float) for a, v in cell.rest().items()}
    kinematics = cell.adapter.kinematics(board_table(cell, board2world, board))
    from .. import calibration as cell_calibration

    cell_calibration.apply(kinematics, cell.calibration())
    clear = Clearance(kinematics, rest)
    limits = kinematics.arm_limits
    corners = bundle._transform(board.points(), board2world)
    centre = corners.mean(0)
    hidden = list(adapter.wrist_image_occluded)
    reach = np.asarray(adapter.reach_seed_offset, float)
    bound = np.asarray(adapter.reach_offset_limit_rad, float)
    out: dict[str, Any] = {"board2world": board2world.tolist(), "square_m": board.square_m, "pattern": list(board.pattern),
                           "rest": {a: rest[a].tolist() for a in rest}, "arms": {}, "skipped": []}
    for arm in arms:
        intrinsic, size = wrist[arm]
        kinematics.set_state(rest)
        cam_in_tcp = np.linalg.inv(kinematics.tcp(arm)) @ kinematics.wrist_camera(arm)
        base = np.array(kinematics.chain(arm)[0])
        toward = np.arctan2(*(base[:2] - centre[:2])[::-1])        # azimuth from the board to the arm's base
        seed = rest[arm] + reach
        seed[0] = np.clip(np.arctan2(centre[1] - base[1], centre[0] - base[0]), *limits[arm][0])
        candidates = []
        for distance in DISTANCES_M:
            for tilt in TILTS_DEG:
                for side in ((0,) if tilt == 0 else (-40, 0, 40)):
                    azimuth = toward + np.radians(side)
                    direction = np.array([np.sin(np.radians(tilt)) * np.cos(azimuth),
                                          np.sin(np.radians(tilt)) * np.sin(azimuth), np.cos(np.radians(tilt))])
                    for roll in ROLLS_DEG:
                        camera = _look_at(centre + distance * direction, centre,
                                          np.array([np.cos(toward), np.sin(toward), 0.0]), np.radians(roll))
                        result = kinematics.ik(arm, camera @ np.linalg.inv(cam_in_tcp), seed)
                        if result.position_error_m > 0.01 or result.orientation_error_rad > 0.05:
                            continue
                        q = np.asarray(result.q, float)
                        if np.any(np.abs(q - rest[arm]) > bound) or not kinematics.within_limits(arm, q):
                            continue
                        state = {**rest, arm: q}
                        kinematics.set_state(state)
                        if not sees_board(kinematics.wrist_camera(arm), intrinsic, size, corners, hidden):
                            continue
                        if clear.state(state, arm):
                            continue
                        view = centre - kinematics.wrist_camera(arm)[:3, 3]
                        candidates.append({"q": q, "view": view / np.linalg.norm(view), "roll": roll, "distance": distance})
        chosen: list[dict[str, Any]] = []
        while candidates and len(chosen) < POSES_PER_ARM:
            if not chosen:
                best = min(candidates, key=lambda c: np.sum(np.abs(c["q"] - rest[arm])))
            else:
                def spread(c: dict[str, Any]) -> float:
                    return min(np.linalg.norm(c["view"] - o["view"]) + 0.01 * abs(c["roll"] - o["roll"])
                               + abs(c["distance"] - o["distance"]) for o in chosen)
                best = max(candidates, key=spread)
            chosen.append(best)
            candidates = [c for c in candidates if c is not best]
        route, here, left = [], rest[arm], list(chosen)
        while left:                                   # nearest first; through rest when the straight way is not clear
            nxt = left.pop(int(np.argmin([np.max(np.abs(c["q"] - here)) for c in left])))
            if clear.path({**rest, arm: here}, {**rest, arm: nxt["q"]}, arm):
                why = clear.path({**rest, arm: here}, rest, arm) or clear.path(rest, {**rest, arm: nxt["q"]}, arm)
                if why:
                    out["skipped"].append(f"{arm}: a view whose way is not clear ({why})")
                    continue
                route.append({"q": rest[arm].tolist(), "via_rest": True})
            route.append({"q": nxt["q"].tolist(), "roll": nxt["roll"], "distance": nxt["distance"]})
            here = nxt["q"]
        while route and clear.path({**rest, arm: np.asarray(route[-1]["q"])}, rest, arm):
            out["skipped"].append(f"{arm}: the last view, whose way back to rest is not clear")
            route.pop()
        views = [r for r in route if not r.get("via_rest")]
        out["arms"][arm] = {"route": route, "views": len(views), "max_offset_deg": (
            np.degrees(np.max([np.abs(np.asarray(r["q"]) - rest[arm]) for r in route], 0)).round(1).tolist()
            if route else None)}
    return out


# ---------------------------------------------------------------- one placement
def run_placement(cell: Cell, planned: Mapping[str, Any], record: Path, lock: CellLock, label: str) -> str | None:
    """Move through the plan and record every capture; returns why it stopped, or None."""

    record.mkdir(parents=True, exist_ok=True)
    (record / "plan.json").write_text(json.dumps(planned, indent=1, default=lambda v: np.asarray(v).tolist()))
    roles = list(cell.config.cameras)
    wrist_role = {arm: role for role, arm in cell.adapter.wrist_cameras.items()}

    def save(name: str, shots: Mapping[str, Mapping[str, Any]], q: Mapping[str, np.ndarray]) -> None:
        for role, data in shots.items():
            np.savez_compressed(record / f"{name}-{role}.npz", rgb=data["rgb"], depth=data["depth"],
                                intrinsic=np.asarray(data["intrinsic_cv"]),
                                distortion=np.asarray(data["distortion"]["coeffs"], float),
                                **{f"q_{a}": np.asarray(v, float) for a, v in q.items()})

    handlers = {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGUSR2, signal.SIGINT)}
    stopped = None
    cell.quiet()
    lock.relabel(f"cell operator ({label}: moving) {HOMES_ON_SIGNAL}")
    try:
        mover = Mover(cell)
        start = mover.held.joints()
        try:
            time.sleep(HOLD_S)
            save("rest", capture(cell, roles, frames=5), mover.held.joints())
            for arm, spec in planned["arms"].items():
                role = wrist_role[arm]
                camera = cell.open_camera(role)
                try:
                    for i, step in enumerate(spec["route"]):
                        goal = {a: np.asarray(q, float) for a, q in start.items()}
                        goal[arm] = np.asarray(step["q"], float)
                        mover.move(goal)
                        if step.get("via_rest"):
                            continue
                        time.sleep(HOLD_S)
                        shots = [camera.capture() for _ in range(3)]
                        save(f"{arm}-{i:02d}", {role: shots[-1]}, mover.held.joints())
                finally:
                    camera.close()
                mover.move({a: np.asarray(q, float) for a, q in start.items()})
            save("end", capture(cell, roles, frames=5), mover.held.joints())
        except Stop as stop:
            stopped = str(stop)
            mover.hold_here()
        finally:
            mover.back_and_release(start)
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)
        lock.relabel(_idle_label(label))
    (record / "result.json").write_text(json.dumps({"stopped": stopped}))
    return stopped


def _idle_label(label: str) -> str:
    return f"cell operator ({label}: waiting for the person, arms released at rest) {HOMES_ON_SIGNAL} {AT_REST}"


def _motion_words(arms: Sequence[str], planned: Mapping[str, Any]) -> tuple[str, str]:
    views = sum(planned["arms"][a]["views"] for a in arms)
    offsets = [planned["arms"][a]["max_offset_deg"] for a in arms if planned["arms"][a]["max_offset_deg"]]
    widest = float(np.max(offsets)) if offsets else 0.0
    title = f"{arm_words(arms)}拍标定板（{views} 个姿势）"
    text = (f"{arm_words(arms)}会一只一只地移动到 {views} 个姿势，用腕部相机拍标定板；另一只手臂保持静止位。"
            f"每个关节最多偏离静止位 {widest:.0f}°，所有部件离桌面至少 {MIN_TABLE_M * 100:.0f} cm，"
            f"两臂之间至少 {MIN_INTER_ARM_M * 100:.0f} cm。拍完回到起始位并放松。期间不要碰标定板；"
            "要停就按回到起始位（板子页面、ora cell home 或 /home）。")
    return title, text


def _placement_words(index: int, count: int, arms: Sequence[str], why: str | None,
                     hint: str | None = None) -> tuple[str, str]:
    where = hint or ("两臂之间、两只手臂都够得着的地方" if len(arms) > 1 else f"{arm_words(arms)}前方、它够得着的地方")
    text = (f"把标定板平放在{where}，用胶带或重物固定住；整块板要在固定相机的画面里看得到，"
            "也别离夹爪太近（会被挡住一角）。摆好后批准；要结束拍摄就拒绝（已拍的摆位照样用来求解）。")
    if why:
        text = f"上一次没有拍成：{why}。\n" + text
    return f"摆放标定板（第 {index + 1}/{count} 次）", text


# ---------------------------------------------------------------- the whole procedure
def calibrate(cell: Cell, *, asker: Asker, arms: Sequence[str] | None = None, placements: int = PLACEMENTS,
              square_mm: float | None = None, pattern: tuple[int, int] | None = None,
              records: Sequence[Path] = (), save: bool = True,
              progress: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    """Capture (unless ``records`` are given), fit, check and, approved, write the cell's calibration."""

    wrist_arm = dict(cell.adapter.wrist_cameras)
    arms = list(arms or [a for a in cell.adapter.arms if a in wrist_arm.values()])
    label = "hand-eye calibration"
    report: dict[str, Any] = {"records": [], "placements": []}
    if records:
        records = [Path(r).expanduser() for r in records]
        found = {bundle.record_board(r) for r in records} - {None}
        if square_mm is None and pattern is None and len(found) == 1:
            board = found.pop()
        else:
            board = ask_board(asker, square_mm, pattern)
        report["records"] = [str(r) for r in records]
        turns, lock, quiet = [], None, {}
    else:
        cell.rest()                                    # refuses without a recorded rest pose
        placed = [r for r in cell.fixed_cameras() if r in (cell.calibration().get("cameras") or {})]
        if not placed:
            raise CellError("no fixed camera has a pose yet: place them roughly first "
                            "(`ora cell calibrate --method charuco|wrists|board-poses`)")
        board = ask_board(asker, square_mm, pattern)
        session = RECORD_ROOT.expanduser() / time.strftime("handeye-%Y%m%d-%H%M%S")
        report["session"] = str(session)
        turns = schedule(arms, placements)
        lock = CellLock(_idle_label(label)).acquire()
        quiet = {s: signal.signal(s, lambda *_: None) for s in (signal.SIGUSR1, signal.SIGUSR2)}

    def capture_placement(index: int, count: int, shown: tuple[str, ...], hint: str, number: int) -> bool:
        """One placement asked, located, planned, approved and captured; False when the person ends it."""

        why = None
        while True:
            title, text = _placement_words(index, count, shown, why, hint)
            ok, reason = asker.confirm("task", title, text, {"placement": number, "of": count, "arms": list(shown),
                                                             "hint": hint}, TASK_ANSWER_S)
            if not ok:
                progress(f"captures ended at placement {number}: {reason}")
                return False
            progress(f"placement {number}: locating the board")
            try:
                located = locate(cell, board)
            except CellError as exc:
                why = f"固定相机拍不了（{exc}）"
                continue
            seen = {r: v for r, v in located.items() if v and not r.startswith("_")}
            if not seen:
                why = located_why(located)
                continue
            source = next(iter(seen))
            progress(f"placement {number}: planning wrist views ({source} located the board)")
            planned = plan(cell, shown, seen[source]["board2world"], board, wrist)
            short = [a for a in shown if planned["arms"][a]["views"] < MIN_POSES]
            if short:
                why = (f"{arm_words(short)}只找到 {min(planned['arms'][a]['views'] for a in short)} 个安全的拍摄姿势"
                       f"（至少要 {MIN_POSES} 个）：把板子往{arm_words(short)}挪近一些，或放平")
                continue
            planned["located"] = {r: v and {"center": v["center"], "reprojection_px": v["reprojection_px"]}
                                  for r, v in located.items() if not r.startswith("_")}
            title, text = _motion_words(shown, planned)
            ok, reason = asker.confirm("motion", title, text, {"arms": list(shown), "plan": planned["arms"]},
                                       MOTION_ANSWER_S)
            if not ok:
                why = f"动作被拒绝（{reason}）"
                continue
            record = Path(report["session"]) / f"placement-{number}"
            progress(f"placement {number}: moving ({title})")
            try:
                stopped = run_placement(cell, planned, record, lock, label)
            except CellError as exc:
                stopped = f"没能接管手臂：{exc}"
            if stopped:
                shutil.rmtree(record, ignore_errors=True)
                why = f"动作中途停止（{stopped}）"
                continue
            report["records"].append(str(record))
            report["placements"].append({"arms": list(shown), "views": {a: planned["arms"][a]["views"] for a in shown},
                                         "skipped": planned["skipped"]})
            return True

    try:
        if lock is not None:
            wrist = {}
            for role, arm in wrist_arm.items():
                if arm in arms:
                    shot = capture(cell, [role], frames=1)[role]
                    wrist[arm] = (np.asarray(shot["intrinsic_cv"], float), (shot["rgb"].shape[1], shot["rgb"].shape[0]))
            for index, shown in enumerate(turns):
                if not capture_placement(index, len(turns), shown, placement_hint(turns, index), index + 1):
                    break
            records = [Path(r) for r in report["records"]]
            if not records:
                raise CellError("no placement was captured: nothing to fit")
        fitted = _fit(cell, records, board, wrist_arm, report, progress)
        extra = 0
        while lock is not None and save and extra < EXTRA_PLACEMENTS:
            weak = weak_arms(report, wrist_arm)
            if not weak:
                break
            arm = weak[0]
            turns.append((arm,))
            extra += 1
            progress(f"the {arm} arm is predicted poorly: asking for one more placement")
            hint = placement_hint(turns, len(turns) - 1)
            ask = (f"建议给{arm_words([arm])}加一个摆位（{report['weak'][arm]}）：{hint}。批准就摆板继续拍；"
                   "拒绝则直接看现在的结果。")
            ok, _ = asker.confirm("task", f"再加一个摆位？（{arm_words([arm])}）", ask, {"arm": arm}, TASK_ANSWER_S)
            if not ok or not capture_placement(len(turns) - 1, len(turns), (arm,), hint, len(turns)):
                break
            records = [Path(r) for r in report["records"]]
            fitted = _fit(cell, records, board, wrist_arm, report, progress)
    finally:
        for s_, handler in quiet.items():
            signal.signal(s_, handler)
        if lock is not None:
            lock.release()
    initial, fit, validation = fitted
    updated = bundle.apply_fit(initial, fit, records=records, board=board, validation=validation)
    report["saved"] = False
    if not save:
        return report
    title, text = _write_words(report)
    ok, reason = asker.confirm("write", title, text, {"changes": report["changes"], "held_out": validation,
                                                      "concerns": report["concerns"]}, WRITE_ANSWER_S)
    if not ok:
        report["not_saved"] = reason
        return report
    path = cell.config.calibration_path
    if path is not None and Path(path).expanduser().is_file():
        backup = Path(path).expanduser().with_name(Path(path).stem + time.strftime(".before-handeye-%Y%m%d-%H%M%S.json"))
        shutil.copy2(Path(path).expanduser(), backup)
        report["backup"] = str(backup)
    cell.save_calibration(updated)
    report["saved"] = True
    return report


def _fit(cell: Cell, records: Sequence[Path], board: bundle.Board, wrist_arm: Mapping[str, str],
         report: dict[str, Any], progress: Callable[[str], None]) -> tuple[dict[str, Any], bundle.Fit, dict[str, Any]]:
    """One fit of every capture and its leave-one-placement-out check, into ``report``."""

    progress("finding the board in every capture")
    roles = list(cell.config.cameras)
    views, missed = bundle.load_views(records, roles, wrist_arm, board, list(cell.adapter.arms))
    report["views"] = {r: sum(1 for v in views if v.role == r) for r in roles}
    report["board_not_found"] = len(missed)
    initial = cell.calibration()
    kinematics = cell.adapter.kinematics(None)
    fit = bundle.solve(views, kinematics, board, wrist_arm=wrist_arm, initial=initial, progress=progress)
    validation = bundle.validate(views, kinematics, board, wrist_arm=wrist_arm, initial=initial, progress=progress)
    report["fit"] = fit.report
    report["held_out"] = validation
    report["changes"] = bundle.summary(fit, initial)
    report["placements_per_arm"] = {a: sum(1 for p in {v.placement for v in views if wrist_arm.get(v.role) == a})
                                    for a in fit.arms}
    report["outliers"] = [v["capture"] for views in fit.report.get("views", {}).values() for v in views
                          if v["px"] > OUTLIER_PX]
    report["concerns"] = concerns(report)
    return initial, fit, validation


def weak_arms(report: dict[str, Any], wrist_arm: Mapping[str, str]) -> list[str]:
    """Arms whose held-out views are predicted worse than HELD_OUT_MM_FLAG, or seen in fewer than
    three placements, worst first; ``report["weak"]`` says why for each."""

    held: dict[str, list[float]] = {}
    for roles in report.get("held_out", {}).values():
        for role, views in roles.items():
            if role in wrist_arm:
                held.setdefault(wrist_arm[role], []).extend(v["mm"] for v in views)
    why, score = {}, {}
    for arm, count in report.get("placements_per_arm", {}).items():
        typical = float(np.median(held[arm])) if held.get(arm) else None
        if typical is not None and typical > HELD_OUT_MM_FLAG:
            why[arm], score[arm] = f"留一验证误差中位 {typical:.0f} mm", typical
        elif count < 3:
            why[arm], score[arm] = f"只拍了 {count} 个摆位", 100.0 - count
    report["weak"] = why
    return sorted(why, key=lambda arm: -score[arm])


def concerns(report: Mapping[str, Any]) -> list[str]:
    """What should make the person hesitate before writing, in plain words."""

    out = []
    for role, views in report["fit"].get("views", {}).items():
        typical = float(np.median([v["px"] for v in views]))
        if typical > FIT_PX_FLAG:
            out.append(f"{role} 的画面拟合误差中位 {typical:.1f} 像素（正常在 {FIT_PX_FLAG:.0f} 以内）：板子可能在拍摄中动过，"
                       "或者机器人模型还有这次没修正的误差")
    for placement, roles in report["held_out"].items():
        typical = float(np.median([v["mm"] for views in roles.values() for v in views]))
        if typical > HELD_OUT_MM_FLAG:
            out.append(f"第 {placement} 个摆位的留一验证误差中位 {typical:.0f} mm（超过 {HELD_OUT_MM_FLAG:.0f} mm）")
    if not report["held_out"]:
        out.append("摆位太少，做不了留一验证：精度没有被独立检验")
    for arm, count in report["placements_per_arm"].items():
        if count < 3:
            out.append(f"{arm_words([arm])}只在 {count} 个摆位里拍到板子：它的结果是外推的，建议至少 3 个")
    return out


def _write_words(report: Mapping[str, Any]) -> tuple[str, str]:
    changes = report["changes"]
    lines = []
    for arm, base in changes["base"].items():
        lines.append(f"{arm_words([arm])}底座相对{arm_words([changes['reference_arm']])}修正 "
                     f"{np.linalg.norm(base['translation_mm']):.0f} mm")
    for arm, offsets in changes["offsets_deg"].items():
        worst = max(abs(v) for v in offsets["fitted"]) if offsets["fitted"] else 0.0
        lines.append(f"{arm_words([arm])}关节零点最大修正 {worst:.1f}°")
    for arm, wrist in changes["wrist"].items():
        lines.append(f"{arm_words([arm])}腕部相机安装：离图纸 {wrist['from_model_deg']:.1f}°、{wrist['from_model_mm']:.0f} mm")
    for role, moved in changes["fixed"].items():
        lines.append(f"{role} 相机移动 {moved['moved_mm']:.0f} mm、{moved['turned_deg']:.1f}°")
    held = [v["mm"] for placement in report["held_out"].values() for views in placement.values() for v in views]
    if held:
        lines.append(f"留一验证：没参与拟合的画面，预测误差中位 {np.median(held):.1f} mm、最大 {max(held):.1f} mm")
    if report.get("outliers"):
        lines.append(f"{len(report['outliers'])} 张画面误差超过 {OUTLIER_PX:.0f} 像素，按离群处理，没有拉偏结果")
    lines += ["注意：" + c for c in report["concerns"]]
    lines.append("批准就写入（旧文件会先备份）；拒绝则什么都不改。")
    return "写入新的标定？", "\n".join(lines)
