"""One joint fit of the robot's geometry and its cameras, from a checkerboard seen many ways.

The board lies still on the table for one placement while the arms carry their wrist
cameras around it and the fixed cameras watch; it is then moved and the views repeat
(``operator/handeye.py`` guides this). Every detected corner is one residual: its pixel
against the corner projected through

- a fixed camera: its pose ``T`` in the world;
- a wrist camera: ``W @ model(q + D) @ C``, the arm's base correction ``W`` (in the world,
  one per arm but the reference arm, whose model base defines the world), the robot model
  at the encoder readings ``q`` plus each joint's zero error ``D`` (the inner joints only,
  with a prior of ``OFFSET_PRIOR_RAD``: the first joint's error is the base's or the
  world's turn, the last one's is the camera mount's), and the camera's mount correction
  ``C`` (camera frame);

with one board pose ``Z`` per placement. A wrist camera sees the board through its arm, so
a wrong mount, a wrong zero and a wrong base all bend its view; only views from many poses
and several placements tell them apart, and only one fit of everything keeps an error of
one from being absorbed by another. The fit is a two-stage least squares (plain, then
Huber at 1 px). A plain checkerboard's half turn is resolved per view against the
starting guess.

``validate`` refits without one placement's wrist views and predicts them: the held-out
error, in pixels and in millimetres at the board, is what the calibration is worth where
it was not fitted. Nothing here touches hardware; records are the ``.npz`` captures
``operator/handeye.py`` writes (``<label>-<role>.npz`` with ``rgb``, ``intrinsic``,
``distortion`` and ``q_<arm>``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

OFFSET_PRIOR_RAD = 0.05
METHOD = "handeye_board_bundle"


@dataclass(frozen=True)
class Board:
    square_m: float
    pattern: tuple[int, int]                     # inner corners: columns, rows

    def points(self) -> np.ndarray:
        columns, rows = self.pattern
        return np.array([[c * self.square_m, r * self.square_m, 0.0] for r in range(rows) for c in range(columns)])


@dataclass
class View:
    placement: int
    role: str
    q: dict[str, np.ndarray]
    corners: np.ndarray
    intrinsic: np.ndarray
    distortion: np.ndarray
    source: str = ""


@dataclass
class Fit:
    arms: list[str]                              # arms with wrist views; arms[0] is the reference
    wrist: dict[str, np.ndarray]                 # C per arm
    bases: dict[str, np.ndarray]                 # W per arm but the reference
    offsets: dict[str, np.ndarray]               # D per arm, every joint (zeros where not fitted)
    fixed: dict[str, np.ndarray]                 # T per fixed camera
    boards: list[np.ndarray]                     # Z per placement
    report: dict[str, Any] = field(default_factory=dict)


def _matrix(v: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    m = np.eye(4)
    m[:3, :3] = Rotation.from_rotvec(v[:3]).as_matrix()
    m[:3, 3] = v[3:6]
    return m


def _vector(m: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return np.r_[Rotation.from_matrix(m[:3, :3]).as_rotvec(), m[:3, 3]]


def _turn_deg(m: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(m[:3, :3]) - 1) / 2, -1.0, 1.0))))


def detect(rgb: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """The board's inner corners in pixels (row by row), or None when the whole board is not seen."""

    import cv2

    gray = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    return corners.reshape(-1, 2).astype(np.float64) if found else None


def project(cam2world: np.ndarray, points_world: np.ndarray, intrinsic: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    import cv2

    world2cam = np.linalg.inv(cam2world)
    uv, _ = cv2.projectPoints(points_world, cv2.Rodrigues(world2cam[:3, :3])[0], world2cam[:3, 3], intrinsic, distortion)
    return uv.reshape(-1, 2)


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def record_board(record: Path) -> Board | None:
    """The board a record was taken with (its ``plan.json``), when it says."""

    try:
        plan = json.loads((record / "plan.json").read_text())
    except (OSError, ValueError):
        return None
    if not plan.get("square_m") or not plan.get("pattern"):
        return None
    return Board(float(plan["square_m"]), tuple(int(v) for v in plan["pattern"]))


def load_views(records: Sequence[Path], roles: Sequence[str], wrist_arm: Mapping[str, str], board: Board,
               arms: Sequence[str]) -> tuple[list[View], list[str]]:
    """Every capture of the records in which the whole board is found, and the ones in which it is not.

    A wrist camera's view counts only when its own arm moved for it (a capture labelled with
    that arm): at rest the jaws hide the board, and a partial find is no measurement."""

    views, missed = [], []
    for index, record in enumerate(records):
        for path in sorted(Path(record).glob("*.npz")):
            role = next((r for r in roles if path.stem.endswith("-" + r)), None)
            if role is None:
                continue
            label = path.stem[: -len(role) - 1]
            if role in wrist_arm and not label.startswith(wrist_arm[role]):
                continue
            data = np.load(path)
            corners = detect(data["rgb"], board.pattern)
            if corners is None:
                missed.append(str(path))
                continue
            q = {a: np.asarray(data[f"q_{a}"], float) for a in arms if f"q_{a}" in data}
            views.append(View(index, role, q, corners, np.asarray(data["intrinsic"], float),
                              np.asarray(data["distortion"], float) if "distortion" in data else np.zeros(5), str(path)))
    return views, missed


def solve(views: Sequence[View], kinematics: Any, board: Board, *, wrist_arm: Mapping[str, str],
          initial: Mapping[str, Any], fit_bases: bool = True, fit_offsets: bool = True,
          holdout: int | None = None, progress: Callable[[str], None] = lambda _m: None) -> Fit:
    """The joint fit; ``kinematics`` is the robot model with no installation correction applied.

    ``initial`` is the current calibration: its fixed cameras (a fixed camera without a pose
    is left out), wrist corrections and joint offsets start the fit. ``holdout`` leaves that
    placement's wrist views out of the fit and reports how well it predicts them."""

    from scipy.optimize import least_squares

    from . import calibration as cell_calibration

    kinematics.camera_corrections = {}
    kinematics.joint_offsets = {}
    if hasattr(kinematics, "set_base_corrections"):
        kinematics.set_base_corrections({})
    cameras = initial.get("cameras") or {}
    fixed = sorted({v.role for v in views if v.role not in wrist_arm and v.role in cameras})
    views = [v for v in views if v.role in wrist_arm or v.role in fixed]
    arms = [a for a in kinematics_arms(kinematics, wrist_arm) if any(wrist_arm.get(v.role) == a for v in views)]
    if not arms:
        raise ValueError("no wrist camera saw the board: nothing to fit")
    placements = sorted({v.placement for v in views})
    based = arms[1:] if fit_bases else []
    joints = len(kinematics.arm_limits[arms[0]])
    inner = list(range(1, joints - 1)) if fit_offsets else []
    points = board.points()
    fitted = [v for v in views if not (holdout is not None and v.placement == holdout and v.role in wrist_arm)]

    corrections = cell_calibration.camera_corrections(initial)
    offsets0 = cell_calibration.joint_offsets(initial)
    C0 = {a: corrections.get(a, np.eye(4)) for a in arms}
    T0 = {r: np.asarray(cameras[r]["cam2world_cv"], float) for r in fixed}

    def wrist_pose(q: Mapping[str, np.ndarray], arm: str, C: Mapping, W: Mapping, D: Mapping) -> np.ndarray:
        state = {a: np.asarray(v, float).copy() for a, v in q.items()}
        for a, d in D.items():
            if a in state:
                state[a][inner] += d
        kinematics.set_state(state)
        return W.get(arm, np.eye(4)) @ kinematics.wrist_camera(arm) @ C[arm]

    def camera(view: View, C: Mapping, T: Mapping, W: Mapping, D: Mapping) -> np.ndarray:
        if view.role in T:
            return T[view.role]
        return wrist_pose(view.q, wrist_arm[view.role], C, W, D)

    def pnp(view: View) -> np.ndarray:
        import cv2

        _, rvec, tvec = cv2.solvePnP(points, view.corners, view.intrinsic, view.distortion)
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
        return m

    D0 = {a: np.asarray(offsets0.get(a, np.zeros(joints)), float)[inner] for a in arms} if inner else {}
    Z0 = []
    for p in placements:
        seen = [v for v in fitted if v.placement == p]
        anchor = next((v for v in seen if v.role in T0), None) or next((v for v in seen), None)
        if anchor is None:
            raise ValueError(f"placement {p} has no view left to fit")
        Z0.append(camera(anchor, C0, T0, {}, D0) @ pnp(anchor))

    def unpack(x: np.ndarray):
        i = 0
        C, W, T, Z, D = {}, {}, {}, [], {}
        for a in arms:
            C[a] = _matrix(x[i:i + 6]); i += 6
        for a in based:
            W[a] = _matrix(x[i:i + 6]); i += 6
        for r in fixed:
            T[r] = _matrix(x[i:i + 6]); i += 6
        for _ in placements:
            Z.append(_matrix(x[i:i + 6])); i += 6
        for a in arms if inner else []:
            D[a] = x[i:i + len(inner)]; i += len(inner)
        return C, W, T, Z, D

    x0 = np.concatenate([_vector(C0[a]) for a in arms] + [np.zeros(6) for _ in based] + [_vector(T0[r]) for r in fixed]
                        + [_vector(z) for z in Z0] + ([D0[a] for a in arms] if inner else []))
    slot = {p: i for i, p in enumerate(placements)}

    order: dict[int, np.ndarray] = {}

    def resolve(x: np.ndarray) -> int:
        """Each view's corners in the order the current guess projects them; how many turned."""

        C, W, T, Z, D = unpack(x)
        turned = 0
        for view in views:
            uv = project(camera(view, C, T, W, D), _transform(points, Z[slot[view.placement]]), view.intrinsic,
                         view.distortion)
            flip = np.linalg.norm(uv - view.corners[::-1], axis=1).mean() < np.linalg.norm(uv - view.corners, axis=1).mean()
            corners = view.corners[::-1] if flip else view.corners
            turned += int(id(view) in order and not np.array_equal(order[id(view)], corners))
            order[id(view)] = corners
        return turned

    def residual(x: np.ndarray) -> np.ndarray:
        C, W, T, Z, D = unpack(x)
        out = [(project(camera(v, C, T, W, D), _transform(points, Z[slot[v.placement]]), v.intrinsic, v.distortion)
                - order[id(v)]).ravel() for v in fitted]
        if inner:
            out.append(np.concatenate(list(D.values())) / OFFSET_PRIOR_RAD)
        return np.concatenate(out)

    progress(f"fitting {len(x0)} unknowns to {len(fitted)} views")
    resolve(x0)
    result = least_squares(residual, x0, loss="linear", diff_step=1e-6, max_nfev=200)
    turned = resolve(result.x)                   # a view the starting guess read the wrong way round
    result = least_squares(residual, result.x, loss="huber", f_scale=1.0, diff_step=1e-6, max_nfev=200)
    C, W, T, Z, D = unpack(result.x)
    full = {}
    for a, d in D.items():
        full[a] = np.zeros(joints)
        full[a][inner] = d
    fit = Fit(arms, C, W, full, T, Z)

    report: dict[str, Any] = {"views": {}, "cost": round(float(result.cost), 2), "placements": len(placements),
                              "turned_after_first_fit": turned,
                              "unknowns": int(len(x0)), "reference_arm": arms[0]}
    for view in views:
        cam = camera(view, C, T, W, D)
        world = _transform(points, Z[slot[view.placement]])
        error = np.linalg.norm(project(cam, world, view.intrinsic, view.distortion) - order[id(view)], axis=1)
        rms = float(np.sqrt((error ** 2).mean()))
        depth = float(np.median(_transform(world, np.linalg.inv(cam))[:, 2]))
        entry = {"px": round(rms, 2), "mm": round(rms * depth / float(view.intrinsic[0, 0]) * 1000, 2),
                 "capture": "/".join(Path(view.source).parts[-2:])}
        key = "held_out" if holdout is not None and view.placement == holdout and view.role in wrist_arm else "views"
        report.setdefault(key, {}).setdefault(view.role, []).append(entry)
    fit.report = report
    return fit


def kinematics_arms(kinematics: Any, wrist_arm: Mapping[str, str]) -> list[str]:
    """The arms in the robot model's order (the first one with wrist views becomes the reference)."""

    return [a for a in kinematics.arm_limits if a in set(wrist_arm.values())]


def validate(views: Sequence[View], kinematics: Any, board: Board, *, wrist_arm: Mapping[str, str],
             initial: Mapping[str, Any], progress: Callable[[str], None] = lambda _m: None,
             **options: Any) -> dict[str, Any]:
    """Leave each placement's wrist views out in turn: how well the rest predicts them.

    A placement is left out only when the arms it shows are still seen in another one."""

    out: dict[str, Any] = {}
    placements = sorted({v.placement for v in views})
    for p in placements:
        shown = {wrist_arm[v.role] for v in views if v.placement == p and v.role in wrist_arm}
        elsewhere = {wrist_arm[v.role] for v in views if v.placement != p and v.role in wrist_arm}
        if not shown or not shown <= elsewhere:
            continue
        progress(f"predicting placement {p + 1} from the others")
        try:
            fit = solve(views, kinematics, board, wrist_arm=wrist_arm, initial=initial, holdout=p, **options)
        except ValueError:
            continue                             # nothing but its wrist views placed the board
        out[str(p + 1)] = fit.report.get("held_out", {})
    return out


def summary(fit: Fit, initial: Mapping[str, Any]) -> dict[str, Any]:
    """What the fit changes, in degrees and millimetres, against the current calibration."""

    from scipy.spatial.transform import Rotation

    from . import calibration as cell_calibration

    now_wrist = cell_calibration.camera_corrections(initial)
    now_base = cell_calibration.base_corrections(initial)
    now_offsets = cell_calibration.joint_offsets(initial)
    out: dict[str, Any] = {"reference_arm": fit.arms[0], "wrist": {}, "base": {}, "offsets_deg": {}, "fixed": {}}
    for a, c in fit.wrist.items():
        change = np.linalg.inv(now_wrist.get(a, np.eye(4))) @ c
        out["wrist"][a] = {"from_model_deg": round(_turn_deg(c), 2), "from_model_mm": round(float(np.linalg.norm(c[:3, 3])) * 1000, 1),
                           "change_deg": round(_turn_deg(change), 2),
                           "change_mm": round(float(np.linalg.norm(change[:3, 3])) * 1000, 1)}
    for a, w in fit.bases.items():
        out["base"][a] = {"translation_mm": (w[:3, 3] * 1000).round(1).tolist(),
                          "rotvec_deg": np.degrees(Rotation.from_matrix(w[:3, :3]).as_rotvec()).round(2).tolist(),
                          "change_mm": round(float(np.linalg.norm(w[:3, 3] - now_base.get(a, np.eye(4))[:3, 3])) * 1000, 1)}
    for a, d in fit.offsets.items():
        out["offsets_deg"][a] = {"fitted": np.degrees(d).round(2).tolist(),
                                 "now": np.degrees(now_offsets.get(a, np.zeros_like(d))).round(2).tolist()}
    for r, t in fit.fixed.items():
        before = np.asarray(initial["cameras"][r]["cam2world_cv"], float)
        out["fixed"][r] = {"moved_mm": round(float(np.linalg.norm(t[:3, 3] - before[:3, 3])) * 1000, 1),
                           "turned_deg": round(_turn_deg(np.linalg.inv(before) @ t), 2)}
    return out


def apply_fit(calibration: Mapping[str, Any], fit: Fit, *, records: Sequence[Path], board: Board,
              validation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The calibration with the fit's values in place of the ones it measured; the rest unchanged."""

    from scipy.spatial.transform import Rotation

    out = json.loads(json.dumps(dict(calibration)))
    source = {"method": METHOD, "records": [str(r) for r in records], "square_mm": board.square_m * 1000,
              "pattern": list(board.pattern)}

    def rigid(m: np.ndarray) -> dict[str, Any]:
        return {"rotvec_rad": Rotation.from_matrix(m[:3, :3]).as_rotvec().tolist(), "translation_m": m[:3, 3].tolist()}

    for a, c in fit.wrist.items():
        out.setdefault("wrist_camera_corrections", {})[a] = {**rigid(c), **source}
    for a in fit.arms[1:]:
        if a in fit.bases:
            out.setdefault("arm_base_corrections", {})[a] = {**rigid(fit.bases[a]), **source, "relative_to": fit.arms[0]}
    for a, d in fit.offsets.items():
        out.setdefault("joint_zero_offsets", {})[a] = {"rad": d.tolist(), **source}
    views = fit.report.get("views", {})
    for r, t in fit.fixed.items():
        out["cameras"][r] = {**out["cameras"][r], "cam2world_cv": t.tolist(), "method": METHOD, "provisional": False,
                             "records": source["records"]}
        if views.get(r):
            out["cameras"][r]["reprojection_px"] = float(np.median([v["px"] for v in views[r]]))
    out["bundle_fit"] = {**source, "fit": fit.report, "held_out": dict(validation or {}), "fitted_unix_s": time.time()}
    return out
