"""The YAM linear_4310 compiler: Contact-GraspNet's fingertip point lands on the TCP, jaw x and approach z."""

from __future__ import annotations

import numpy as np
import pytest

from ora.agent.toolkit.host.grasp.compiler_registry import YAM_TCP_COMPILER_IDENTITIES
from ora.agent.toolkit.lib._grasp_frames import _compiled_grasp_geometry


@pytest.mark.parametrize("arm", ["left", "right"])
def test_the_tcp_is_the_proposals_fingertip_point_on_the_jaw_x_approach_z_frame(arm):
    identity = YAM_TCP_COMPILER_IDENTITIES[arm]
    origin, approach, jaw = np.array([0.4, 0.1, 0.30]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])
    candidate = {"pose_convention": identity["source_pose_convention"],
                 "source_gripper_origin_xyz_m": origin.tolist(),
                 "source_pregrasp_origin_xyz_m": (origin - 0.05 * approach).tolist(),
                 "approach_world": approach.tolist(), "jaw_world": jaw.tolist(), "gripper_opening_m": 0.04}
    geometry = _compiled_grasp_geometry(candidate, compiler_identity=identity)
    np.testing.assert_allclose(geometry["grip_site"], origin + 0.1034 * approach)
    np.testing.assert_allclose(geometry["contact_baseline"], geometry["grip_site"])
    np.testing.assert_allclose(geometry["pregrasp_grip_site"], geometry["grip_site"] - 0.05 * approach)
    # Public TCP axes (controllers/yam_station.py): local x the closing line, local z the approach.
    np.testing.assert_allclose(geometry["x_axis"], jaw)
    np.testing.assert_allclose(geometry["z_axis"], approach)
