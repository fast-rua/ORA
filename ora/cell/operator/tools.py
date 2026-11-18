"""Setting a cell up: reading the arms, recording rest, checking and calibrating cameras.

None of these enable a motor (arm-moving calibration is ``motion.py``). Each takes the
cell lock while it touches buses or cameras, and returns what it found; ``ora cell``
prints it and the cell board shows it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import CellError
from ..geometry import joint_sanity, register_fixed_camera, table_plane, transform, wrist_table_check
from ..lock import CellLock
from .asking import Asker, arm_words
from .context import Cell
from .motion import capture

WRIST_TABLE_RANGE_M = (0.07, 0.6)
DEFAULT_TABLE_EXTENT = {"center_xy_m": [0.4, 0.0], "size_xy_m": [1.2, 1.2]}
FIXED_DEPTH_FRAMES = 10


def _posed(cell: Cell, readings: dict[str, Any]):
    kinematics = cell.kinematics()
    open_m = cell.adapter.finger_open_m
    kinematics.set_state({arm: r.q for arm, r in readings.items()},
                         {arm: (r.gripper_opening if r.gripper_opening is not None else 1.0) * open_m
                          for arm, r in readings.items()})
    return kinematics


def read_state(cell: Cell) -> dict[str, Any]:
    with CellLock("cell operator (read)"):
        readings = cell.read_arms()
    kinematics = _posed(cell, readings)
    joints = joint_sanity({arm: r.q for arm, r in readings.items()}, kinematics.arm_limits,
                          cell.calibration().get("rest_reference"))
    return {"arms": {arm: {**r.__dict__, "q": r.q.tolist(), "tcp_xyz_m": kinematics.tcp(arm)[:3, 3].tolist()}
                     for arm, r in readings.items()}, "joints": joints}


def record_rest(cell: Cell) -> dict[str, Any]:
    with CellLock("cell operator (record rest)"):
        readings = cell.read_arms()
    kinematics = cell.kinematics()
    joints = joint_sanity({arm: r.q for arm, r in readings.items()}, kinematics.arm_limits, None)
    if not joints["passed"]:
        raise CellError(f"joints outside the model's limits {joints['outside_limits']}: "
                        "fix the encoder zero before recording a rest pose")
    calibration = cell.calibration()
    calibration["rest_reference"] = {arm: r.q.tolist() for arm, r in readings.items()}
    calibration["rest_reference_unix_s"] = time.time()
    cell.save_calibration(calibration)
    return calibration["rest_reference"]


def wrist_check(cell: Cell) -> dict[str, Any]:
    """The table each wrist camera sees, placed by the encoders: level, at the calibrated height."""

    table = cell.table()
    expected = None if not table else float(table["top_z_m"])
    with CellLock("cell operator (wrist check)"):
        readings = cell.read_arms()
        kinematics = _posed(cell, readings)
        out = {}
        for role, arm in cell.adapter.wrist_cameras.items():
            if role not in cell.config.cameras:
                continue
            data = capture(cell, [role], frames=1)[role]
            plane = table_plane(data["depth"], data["intrinsic_cv"], kinematics.wrist_camera(arm),
                                near=WRIST_TABLE_RANGE_M[0], far=WRIST_TABLE_RANGE_M[1])
            out[arm] = {**wrist_table_check(plane, expected), "depth_valid": data["depth_valid_fraction"]}
    return {"at": time.time(), "expected_table_z_m": expected, "arms": out}


def fixed_camera_on_arms(cell: Cell, role: str, *, save: bool, table_top_z_m: float = 0.0) -> dict[str, Any]:
    """One fixed camera registered on the arms as they stand now (needs them in its view)."""

    with CellLock("cell operator (fixed camera)"):
        readings = cell.read_arms()
        data = capture(cell, [role], frames=FIXED_DEPTH_FRAMES)[role]
    calibration = cell.calibration()
    table_z = float((calibration.get("table") or {}).get("top_z_m", table_top_z_m))
    result = register_fixed_camera(data["depth"], data["intrinsic_cv"], _posed(cell, readings), table_top_z_m=table_z)
    if save and result["accepted"]:
        calibration.setdefault("cameras", {})[role] = {
            "cam2world_cv": np.asarray(result["cam2world_cv"]).tolist(), "method": "arm_registration",
            "provisional": False, "median_residual_m": result["median_residual_m"],
            "inlier_fraction": result["inlier_fraction"], "arm_q": {a: r.q.tolist() for a, r in readings.items()},
            "serial": cell.config.cameras[role].serial, "intrinsic_cv": data["intrinsic_cv"],
            "captured_unix_s": time.time()}
        if not calibration.get("table"):
            calibration["table"] = {**DEFAULT_TABLE_EXTENT, "top_z_m": result["table_height_m"],
                                    "source": f"fixed camera {role}"}
        cell.save_calibration(calibration)
    return {k: v for k, v in result.items() if k != "cam2world_cv"}


def fixed_camera_from_wrists(cell: Cell, role: str, *, save: bool, force: bool = False) -> dict[str, Any]:
    """Provisional: a fixed camera matched to what the wrist cameras (and fixed cameras already
    placed) see of the same table (``cameras/wrist_registration.py``)."""

    from ..cameras.wrist_registration import coloured_points, register_to_wrists

    calibration = cell.calibration()
    placed = [r for r in cell.fixed_cameras() if r != role and r in (calibration.get("cameras") or {})]
    with CellLock("cell operator (fixed camera from wrists)"):
        readings = cell.read_arms()
        kinematics = _posed(cell, readings)
        reference_xyz, reference_rgb = [], []
        wrists = [r for r in cell.adapter.wrist_cameras if r in cell.config.cameras]
        frames = capture(cell, [*wrists, *placed, role], frames=FIXED_DEPTH_FRAMES)
    for wrist in wrists:
        data = frames[wrist]
        points, rgb = coloured_points(data["depth"], data["rgb"], data["intrinsic_cv"], near=0.07, far=1.5)
        reference_xyz.append(transform(points, kinematics.wrist_camera(cell.adapter.wrist_cameras[wrist])))
        reference_rgb.append(rgb)
    for other in placed:
        data = frames[other]
        points, rgb = coloured_points(data["depth"], data["rgb"], data["intrinsic_cv"], near=0.25, far=3.0)
        reference_xyz.append(transform(points, np.asarray(calibration["cameras"][other]["cam2world_cv"])))
        reference_rgb.append(rgb)
    data = frames[role]
    points, rgb = coloured_points(data["depth"], data["rgb"], data["intrinsic_cv"], near=0.25, far=3.0)
    rough_table = float(np.median(np.concatenate(reference_xyz)[:, 2]))
    result = register_to_wrists(points, rgb, np.concatenate(reference_xyz), np.concatenate(reference_rgb),
                                prior=lambda pose: pose[2, 3] > rough_table + 0.3)
    if save and (result["accepted"] or force):
        calibration.setdefault("cameras", {})[role] = {
            "cam2world_cv": np.asarray(result["cam2world_cv"]).tolist(), "method": "wrist_cloud_registration",
            "provisional": True, "reference": [*wrists, *placed],
            "median_residual_m": result["median_distance_to_wrist_points_m"], "correlation": result["score"],
            "rival_ratio": result["rival_ratio"], "serial": cell.config.cameras[role].serial,
            "intrinsic_cv": data["intrinsic_cv"], "captured_unix_s": time.time()}
        cell.save_calibration(calibration)
    return {k: v for k, v in result.items() if k != "cam2world_cv"}


def charuco_image(out: Path, width: int, height: int) -> str:
    import cv2

    from ..cameras.charuco import image

    cv2.imwrite(str(out), image(width, height))
    return f"wrote {out} ({width}x{height})"


def _pose_gap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    turn = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1.0, 1.0)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(np.degrees(np.arccos(turn)))


def charuco(cell: Cell, *, square_mm: float, save: bool, frames: int = 5, force: bool = False) -> dict[str, Any]:
    """Every camera from one ChArUco board on the table, placed in the world through the wrist cameras."""

    from ..cameras.charuco import board_pose

    with CellLock("cell operator (charuco)"):
        readings = cell.read_arms()
        shots = capture(cell, list(cell.config.cameras), frames=frames)
    kinematics = _posed(cell, readings)
    seen, report = {}, {"cameras": {}}
    for role, data in shots.items():
        pose = board_pose(data["rgb"], data["intrinsic_cv"], data["distortion"]["coeffs"], square_mm / 1000,
                          depth=data["depth"])
        report["cameras"][role] = None if pose is None else {
            "corners": pose["corners"], "reprojection_px": pose["reprojection_px"],
            "depth_scale_ratio": pose.get("depth_scale_ratio")}
        if pose is not None:
            seen[role] = {**pose, "intrinsic_cv": data["intrinsic_cv"]}
    anchors = {role: kinematics.wrist_camera(arm) @ seen[role]["board2cam"]
               for role, arm in cell.adapter.wrist_cameras.items() if role in seen}
    if not anchors:
        raise CellError("no wrist camera sees the board: it cannot be placed in the world")
    if len(anchors) > 1:
        shift, turn = _pose_gap(*list(anchors.values())[:2])
        report["wrist_chains_disagree"] = {"mm": round(shift * 1000, 1), "deg": round(turn, 2)}
    anchor_role = next(iter(anchors))
    board2world = anchors[anchor_role]
    ratios = [s["depth_scale_ratio"] for s in seen.values() if s.get("depth_scale_ratio")]
    if ratios and abs(np.median(ratios) - 1.0) > 0.03 and not force:
        raise CellError(f"depth says the board is {np.median(ratios):.3f}x as far as PnP with {square_mm} mm squares: "
                        "re-measure one square")
    placed = {role: board2world @ np.linalg.inv(seen[role]["board2cam"]) for role in cell.fixed_cameras() if role in seen}
    report["placed"] = {role: np.round(pose[:3, 3], 3).tolist() for role, pose in placed.items()}
    report["anchor"] = anchor_role
    if save and placed:
        calibration = cell.calibration()
        for role, cam2world in placed.items():
            calibration.setdefault("cameras", {})[role] = {
                "cam2world_cv": cam2world.tolist(), "method": "charuco", "provisional": False,
                "anchor": f"{anchor_role} (encoders + CAD)", "square_mm": square_mm, "median_residual_m": None,
                "reprojection_px": seen[role]["reprojection_px"], "corners": seen[role]["corners"],
                "serial": cell.config.cameras[role].serial, "intrinsic_cv": seen[role]["intrinsic_cv"],
                "captured_unix_s": time.time()}
        cell.save_calibration(calibration)
    return report


def board_relative(cell: Cell, role: str, reference: str, *, square_mm: float, pattern: tuple[int, int],
                   save: bool, frames: int = 5) -> dict[str, Any]:
    """A fixed camera from a placed one: both see the same checkerboard, nothing moves.

    ``cam2world(role) = cam2world(reference) @ board2cam(reference) @ inv(board2cam(role))``;
    the board's half turn is resolved by the rough pose of ``role`` when there is one,
    else by keeping the camera above the board."""

    from ..cameras.charuco import chessboard_pose, flip_variants

    calibration = cell.calibration()
    placed = (calibration.get("cameras") or {}).get(reference)
    if placed is None:
        raise CellError(f"{reference} is not placed yet")
    square = square_mm / 1000.0
    with CellLock("cell operator (board relative)"):
        shots = capture(cell, [reference, role], frames=frames)
    poses = {name: chessboard_pose(data["rgb"], data["intrinsic_cv"], data["distortion"]["coeffs"], square, pattern,
                                   depth=data["depth"]) for name, data in shots.items()}
    report = {name: None if p is None else {"corners": p["corners"], "reprojection_px": round(p["reprojection_px"], 3),
                                            "depth_scale_ratio": p.get("depth_scale_ratio")} for name, p in poses.items()}
    if poses[reference] is None or poses[role] is None:
        raise CellError(f"both cameras must see the whole board: {report}")
    board2world = np.asarray(placed["cam2world_cv"], float) @ poses[reference]["board2cam"]
    candidates = [board2world @ np.linalg.inv(v) for v in flip_variants(poses[role]["board2cam"], square, pattern)]
    rough = [np.asarray(c["cam2world_cv"], float) for c in [(calibration.get("cameras") or {}).get(role)] +
             [r for r in calibration.get("rejected_cameras", []) if r.get("serial") == cell.config.cameras[role].serial] if c]
    if rough:
        candidates.sort(key=lambda c: min(_pose_gap(c, r)[1] for r in rough))
    elif candidates[0][2, 3] < candidates[1][2, 3]:
        candidates.reverse()
    cam2world = candidates[0]
    result = {"views": report, "camera_at": np.round(cam2world[:3, 3], 3).tolist(),
              "half_turn_by": "rough pose" if rough else "camera above the board"}
    if save:
        calibration.setdefault("cameras", {})[role] = {
            "cam2world_cv": cam2world.tolist(), "method": f"checkerboard_relative_to_{reference}", "provisional": False,
            "square_mm": square_mm, "pattern": list(pattern), "median_residual_m": None,
            "reprojection_px": poses[role]["reprojection_px"], "serial": cell.config.cameras[role].serial,
            "intrinsic_cv": shots[role]["intrinsic_cv"], "captured_unix_s": time.time()}
        cell.save_calibration(calibration)
    return result


# ---------------------------------------------------------------- which way the grippers open
ORIENT_ANSWER_S = 1800.0
OPEN_WORDS = {"开", "张开", "打开", "open", "o"}
CLOSED_WORDS = {"合", "闭", "闭合", "关", "关闭", "合上", "closed", "close", "c"}


def orient_grippers(cell: Cell, asker: Asker) -> dict[str, Any]:
    """Check that each gripper's recorded [closed, open] limits match what the person sees.

    The driver's own calibration finds both stops but names them by the motor's nominal
    direction, which a gripper may not follow: then "open" closes it, and holding it "open"
    squeezes the jaws shut until the motor overheats. Nothing moves here: the grippers are
    read passively, the person says for each whether it is open or closed, and a pair the
    reading contradicts is swapped and saved. A gripper stopped between its limits cannot
    tell the direction and is left as it is."""

    limits = cell.gripper_limits()
    if not limits:
        raise CellError("no gripper limits yet: `ora cell grippers` measures them")
    with CellLock("cell operator (checking which way the grippers open)"):
        readings = cell.read_arms()
    report: dict[str, Any] = {}
    swapped = {}
    for arm, (closed, opened) in limits.items():
        raw = readings[arm].gripper_raw if arm in readings else None
        if raw is None:
            report[arm] = "no gripper reading"
            continue
        fraction = (float(raw) - closed) / (opened - closed)
        if 0.25 < fraction < 0.75:
            report[arm] = (f"stopped between its limits (reading {raw:.3f}, {fraction:.0%} of the way to open): "
                           "open or close it fully, then check again")
            continue
        says_open = None
        while says_open is None:
            answer, reason = asker.value(
                f"{arm_words([arm])}夹爪现在是张开还是闭合？",
                f"看一眼{arm_words([arm])}的夹爪：张开就回答「开」，闭合就回答「合」。手臂和夹爪都不会动。",
                {"arm": arm, "example": "开", "reading": float(raw)}, ORIENT_ANSWER_S)
            if answer is None:
                raise CellError(f"no answer for the {arm} gripper: {reason}")
            word = answer.strip().lower()
            says_open = True if word in OPEN_WORDS else False if word in CLOSED_WORDS else None
        reads_open = fraction >= 0.5
        if says_open == reads_open:
            report[arm] = f"matches: {'open' if says_open else 'closed'} at reading {raw:.3f}"
        else:
            swapped[arm] = [opened, closed]
            report[arm] = (f"swapped: the reading {raw:.3f} is the {'open' if says_open else 'closed'} end; "
                           f"[closed, open] is now {[round(v, 4) for v in swapped[arm]]}")
    if swapped:
        calibration = cell.calibration()
        calibration.setdefault("gripper_limits", {}).update(swapped)
        calibration.setdefault("notes", []).append(
            f"{time.strftime('%Y-%m-%d %H:%M')}: gripper limits swapped after the person saw them: {', '.join(swapped)}")
        cell.save_calibration(calibration)
    return {"grippers": report, "saved": bool(swapped)}

