"""The joint fit finds a cell's measured errors again: a base moved, joint zeros off, a camera mount
turned, a fixed camera placed roughly, all from board corners alone; a placement left out is
predicted; and the result is a calibration file the kit loads."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ora.cell import bundle
from ora.cell import calibration as cell_calibration
from ora.cell.operator.handeye import _look_at

BOARD = bundle.Board(0.02, (11, 8))
K = np.array([[430.0, 0.0, 424.0], [0.0, 430.0, 240.0], [0.0, 0.0, 1.0]])
BASES = {"left": np.array([0.0, 0.3, 0.0]), "right": np.array([0.0, -0.3, 0.0])}


def rigid(rotvec_deg=(0, 0, 0), translation=(0, 0, 0)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    m[:3, 3] = translation
    return m


def _r(axis: str, angle: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler(axis, angle).as_matrix()
    return m


def _t(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    return rigid(translation=(x, y, z))


class ToyArms:
    """Two six-joint arms with links between their joints, a wrist camera looking along the tool."""

    def __init__(self) -> None:
        self.arm_limits = {arm: np.tile([-3.0, 3.0], (6, 1)) for arm in BASES}
        self.camera_corrections: dict = {}
        self.joint_offsets: dict = {}
        self.q = {arm: np.zeros(6) for arm in BASES}

    def set_base_corrections(self, corrections) -> None:
        assert not corrections                   # the fit applies its own

    def set_state(self, arm_q, finger_m=None) -> None:
        for arm, q in arm_q.items():
            self.q[arm] = np.asarray(q, float) + self.joint_offsets.get(arm, 0.0)

    @staticmethod
    def chain(arm: str, q: np.ndarray) -> np.ndarray:
        return (_t(*BASES[arm]) @ _r("z", q[0]) @ _t(z=0.1) @ _r("y", q[1]) @ _t(x=0.35) @ _r("y", q[2])
                @ _t(x=0.3) @ _r("y", q[3]) @ _t(x=0.06) @ _r("x", q[4]) @ _r("z", q[5]) @ _r("y", np.pi / 2))

    def wrist_camera(self, arm: str) -> np.ndarray:
        return self.chain(arm, self.q[arm])


TRUTH = {
    "base": {"right": rigid((0.4, -0.6, 1.0), (0.006, 0.08, 0.02))},
    "offsets": {"left": np.array([0.0, 0.03, -0.02, 0.015, 0.01, 0.0]),
                "right": np.array([0.0, -0.01, 0.012, 0.02, -0.015, 0.0])},
    "wrist": {"left": rigid((1.5, -0.8, 0.5), (0.004, -0.002, 0.003)),
              "right": rigid((-0.7, 1.2, -0.4), (-0.003, 0.002, -0.004))},
    "front": _look_at(np.array([1.0, 0.05, 0.55]), np.array([0.45, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.0),
}
PLACEMENTS = [((0.45, 0.12, 0.2), ("left", "right")), ((0.42, 0.25, -0.3), ("left",)),
              ((0.42, -0.25, 0.25), ("right",)), ((0.5, -0.05, -0.1), ("left", "right"))]


def _board_pose(x: float, y: float, yaw: float) -> np.ndarray:
    centre = BOARD.points().mean(0)
    return _t(x, y, 0.0) @ _r("z", yaw) @ _t(*-centre)


def _ik(arm: str, target: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
    def error(q):
        pose = ToyArms.chain(arm, q) @ TRUTH["wrist"].get(arm, np.eye(4))
        pose = TRUTH["base"].get(arm, np.eye(4)) @ pose
        delta = np.linalg.inv(target) @ pose
        return np.r_[delta[:3, 3] * 10, Rotation.from_matrix(delta[:3, :3]).as_rotvec()]

    fit = least_squares(error, seed, xtol=1e-12, ftol=1e-12)
    return fit.x if np.linalg.norm(fit.fun) < 1e-7 else None


def _views() -> list[bundle.View]:
    views = []
    rest = {arm: np.array([0.0, -0.6, 1.4, 0.6, 0.0, 0.0]) for arm in BASES}
    for index, ((x, y, yaw), arms) in enumerate(PLACEMENTS):
        board2world = _board_pose(x, y, yaw)
        corners = bundle._transform(BOARD.points(), board2world)
        centre = corners.mean(0)
        views.append(bundle.View(index, "front", {}, bundle.project(TRUTH["front"], corners, K, np.zeros(5)), K,
                                 np.zeros(5)))
        for arm in arms:
            toward = np.arctan2(*(BASES[arm][:2] - centre[:2])[::-1])
            seed = rest[arm].copy()
            seed[0] = np.arctan2(centre[1] - BASES[arm][1], centre[0] - BASES[arm][0])
            found = 0
            for tilt, side, roll in [(0, 0, 0), (25, -40, 30), (25, 40, -30), (35, 0, 0), (20, 0, 45), (35, -60, -20)]:
                azimuth = toward + np.radians(side)
                direction = np.array([np.sin(np.radians(tilt)) * np.cos(azimuth),
                                      np.sin(np.radians(tilt)) * np.sin(azimuth), np.cos(np.radians(tilt))])
                camera = _look_at(centre + 0.3 * direction, centre, np.array([np.cos(toward), np.sin(toward), 0.0]),
                                  np.radians(roll))
                q_model = _ik(arm, camera, seed)
                if q_model is None:
                    continue
                reading = q_model - TRUTH["offsets"][arm]
                views.append(bundle.View(index, f"wrist_{arm}", {arm: reading},
                                         bundle.project(camera, corners, K, np.zeros(5)), K, np.zeros(5)))
                found += 1
            assert found >= 5, (index, arm, found)
    return views


WRIST_ARM = {"wrist_left": "left", "wrist_right": "right"}
ROUGH = {"cameras": {"front": {"cam2world_cv": (TRUTH["front"] @ rigid((0.8, -0.5, 1.0), (0.012, -0.008, 0.01))).tolist()}}}


@pytest.fixture(scope="module")
def views() -> list[bundle.View]:
    return _views()


@pytest.fixture(scope="module")
def fit(views) -> bundle.Fit:
    return bundle.solve(views, ToyArms(), BOARD, wrist_arm=WRIST_ARM, initial=ROUGH)


def test_the_fit_finds_the_base_the_zeros_the_mounts_and_the_fixed_camera(fit) -> None:
    assert fit.arms == ["left", "right"] and set(fit.bases) == {"right"}
    assert np.linalg.norm(fit.bases["right"][:3, 3] - TRUTH["base"]["right"][:3, 3]) < 0.002
    for arm in BASES:
        assert np.max(np.abs(fit.offsets[arm] - TRUTH["offsets"][arm])) < 0.004
        assert np.linalg.norm(fit.wrist[arm][:3, 3] - TRUTH["wrist"][arm][:3, 3]) < 0.002
    assert np.linalg.norm(fit.fixed["front"][:3, 3] - TRUTH["front"][:3, 3]) < 0.002
    assert max(v["px"] for views in fit.report["views"].values() for v in views) < 1.0


def test_a_placement_left_out_is_predicted_by_the_others(views) -> None:
    held = bundle.validate(views, ToyArms(), BOARD, wrist_arm=WRIST_ARM, initial=ROUGH)
    assert set(held) == {"1", "2", "3", "4"}       # every arm is seen in another placement too
    assert all(v["mm"] < 2.0 for roles in held.values() for views in roles.values() for v in views)


def test_a_placement_whose_arm_is_seen_nowhere_else_is_not_left_out(views) -> None:
    only_first_right = [v for v in views if not (v.role == "wrist_right" and v.placement != 0)]
    held = bundle.validate(only_first_right, ToyArms(), BOARD, wrist_arm=WRIST_ARM, initial=ROUGH)
    assert "1" not in held and "2" in held


def test_the_fit_becomes_a_calibration_the_kit_loads(tmp_path, fit) -> None:
    written = bundle.apply_fit({**cell_calibration.empty(), **ROUGH}, fit, records=[tmp_path / "placement-1"], board=BOARD)
    path = tmp_path / "cell.json"
    cell_calibration.save(path, written)
    loaded = cell_calibration.load(path)
    assert np.allclose(cell_calibration.base_corrections(loaded)["right"], fit.bases["right"])
    assert np.allclose(cell_calibration.joint_offsets(loaded)["left"], fit.offsets["left"])
    assert np.allclose(cell_calibration.camera_corrections(loaded)["right"], fit.wrist["right"])
    assert np.allclose(loaded["cameras"]["front"]["cam2world_cv"], fit.fixed["front"])
    assert loaded["arm_base_corrections"]["right"]["relative_to"] == "left"
    assert "left" not in loaded["arm_base_corrections"]
    assert json.loads(json.dumps(loaded["bundle_fit"]))["square_mm"] == pytest.approx(20.0)
