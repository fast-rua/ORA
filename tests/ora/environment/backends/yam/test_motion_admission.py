"""The reach check measures from where this cell's arm base is (the calibrated model), not the CAD
drawing: a right base 9.5 cm nearer the left one brings the far right corner out of reach."""

from __future__ import annotations

import types

import numpy as np
import pytest

from ora.environment.backends.yam import motion


def _session(right_base):
    kinematics = types.SimpleNamespace(chain=lambda arm: [list(right_base)])
    return types.SimpleNamespace(kinematics=kinematics)


def test_reach_is_measured_from_the_calibrated_base() -> None:
    target = np.eye(4)
    target[:3, 3] = [0.55, -0.55, 0.1]                  # 0.61 m from the CAD base (0, -0.305, 0)
    with pytest.raises(motion.Refusal) as refused:
        motion.admit_target(_session((0.0, -0.21, 0.0)), "right", target, "free_space")
    assert refused.value.code == "target_out_of_reach"
