"""The Panda's public-RGB-D top-down family, behind the learned one.

29 of 222 recorded LIBERO ``grasp_plan`` calls got no candidate from
Contact-GraspNet. The family ALOHA and the X5 already have now answers those
calls on LIBERO, and a call the network answers is unchanged.

All geometry is synthetic and offline: identity camera-to-world, a table plane
at z = 0 and box-shaped targets.
"""
from __future__ import annotations

import numpy as np
import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.toolkit.host.grasp import panda_topdown
from ora.agent.toolkit.host.grasp.compiler import PANDA_GRIP_SITE_COMPILER_IDENTITY, compile_grasp_candidate
from ora.agent.toolkit.host.grasp.proposers import (
    CONTACT_FIRST, TOPDOWN_FIRST, AlohaHybridGraspProposer, PcaTopDownGraspProposer,
)
from ora.agent.toolkit.host.shared.artifact_errors import TargetInsufficient
from tests.support.robodojo.x5_topdown import _NoLearned

BOUNDS = {"x": [-0.3, 0.3], "y": [-0.3, 0.3], "z": [-0.05, 0.5]}
LIBERO = "libero_placement_direct_overhead_v1"
FINGERTIP_M = panda_topdown.PROFILE["fingertip_beyond_contact_centre_m"]


def _box(height, *, size=(0.03, 0.06), step=0.004):
    xs = np.arange(-size[0] / 2, size[0] / 2 + 1e-9, step)
    ys = np.arange(-size[1] / 2, size[1] / 2 + 1e-9, step)
    return np.asarray([(x, y, height) for x in xs for y in ys] + [(x, y, z) for x in xs for y in (ys[0], ys[-1])
                       for z in np.arange(0.0, height, step)], dtype=np.float64)


def _table(*, half=0.12, step=0.005):
    values = np.arange(-half, half, step)
    return np.asarray([(x, y, 0.0) for x in values for y in values], dtype=np.float64)


class _Learned(_NoLearned):
    def __init__(self, count: int) -> None:
        self.count = count

    def propose(self, **kwargs):
        candidates, diagnostics = super().propose(**kwargs)
        return [{"candidate_id": f"grasp-net-{index}", "rank": index} for index in range(self.count)], diagnostics


def _proposer(learned=None) -> AlohaHybridGraspProposer:
    surface = resolve_tool_surface(LIBERO)
    return AlohaHybridGraspProposer(contact_backend=learned or _NoLearned(),
                                    **panda_topdown.hybrid_backend_arguments(surface))


def _propose(proposer, target, *, limit=3):
    return proposer.propose(
        scene_points_camera_m=np.vstack([target, _table()]), target_points_camera_m=target, camera_to_world=np.eye(4),
        target_ref="target-1", workspace_bounds_m=BOUNDS, max_candidates=limit,
        compiler_identity=dict(PANDA_GRIP_SITE_COMPILER_IDENTITY))


def test_a_target_the_network_missed_gets_top_down_candidates():
    candidates, diagnostics = _propose(_proposer(), _box(0.04))
    assert [item["pose_convention"] for item in candidates] == ["pca_topdown_parallel_jaw_center_v1"] * 2
    assert [item["rank"] for item in candidates] == [0, 1]
    assert diagnostics["candidate_family_counts"] == {
        "learned_available": 0, "public_rgbd_topdown_available": 2, "selected": 2}
    assert all(compile_grasp_candidate(item)["approach_delta"]["translation_m"][2] == pytest.approx(0.04)
               for item in candidates)


def test_top_down_is_shown_only_when_the_network_returned_nothing():
    candidates, diagnostics = _propose(_proposer(_Learned(2)), _box(0.04), limit=3)
    assert [item["candidate_id"] for item in candidates] == ["grasp-net-0", "grasp-net-1"]
    assert [item["rank"] for item in candidates] == [0, 1]
    assert diagnostics["candidate_family_counts"]["public_rgbd_topdown_available"] > 0
    full, _diagnostics = _propose(_proposer(_Learned(3)), _box(0.04), limit=3)
    assert [item["candidate_id"] for item in full] == ["grasp-net-0", "grasp-net-1", "grasp-net-2"]


def test_the_contact_centre_is_the_grip_site():
    low, _diagnostics = _propose(_proposer(), _box(0.02))
    height = low[0]["topdown_height"]
    assert low[0]["grasp_center_xyz_m"][2] == pytest.approx(height["contact_centre_z_m"])
    assert height["support_z_m"] == pytest.approx(0.0, abs=1e-3)
    fingertip = low[0]["grasp_center_xyz_m"][2] - FINGERTIP_M
    assert fingertip == pytest.approx(panda_topdown.PROFILE["table_clearance_m"], abs=1e-3)
    tall, _diagnostics = _propose(_proposer(), _box(0.12))
    assert tall[0]["grasp_center_xyz_m"][2] == pytest.approx(0.12 - panda_topdown.PROFILE["top_clearance_m"], abs=2e-3)


def test_a_wide_target_is_rejected_with_its_reason():
    candidates, diagnostics = _propose(_proposer(), _box(0.04, size=(0.2, 0.2)))
    assert candidates == []
    assert all("aperture_exceeds_panda_limit" in item["reasons"] for item in diagnostics["rejected"])


def test_a_sparse_target_costs_only_the_top_down_family():
    target = _box(0.04)[:4]
    candidates, diagnostics = _propose(_proposer(_Learned(1)), target)
    assert [item["candidate_id"] for item in candidates] == ["grasp-net-0"]
    assert diagnostics["candidate_family_counts"]["public_rgbd_topdown_available"] == 0
    aloha = AlohaHybridGraspProposer(contact_backend=_Learned(1))
    with pytest.raises(TargetInsufficient):
        _propose(aloha, target)


def test_only_the_libero_default_selects_the_family():
    assert resolve_tool_surface(LIBERO).supports(panda_topdown.CAPABILITY)
    for name in ("robocasa_mobile_direct_v1", "robolab_direct_v1", "robotwin_toolchain_b2_closed_noservo_v2"):
        assert panda_topdown.hybrid_backend_arguments(resolve_tool_surface(name)) is None


def test_the_released_hybrid_keeps_its_configuration():
    configuration = AlohaHybridGraspProposer(contact_backend=_NoLearned()).configuration()
    assert configuration["selection"] == TOPDOWN_FIRST
    assert "topdown_profile" not in configuration["topdown_backend"]["configuration"]
    assert "topdown_profile" not in PcaTopDownGraspProposer().configuration()
    fallback = _proposer().configuration()
    assert fallback["selection"] == CONTACT_FIRST
    assert fallback["topdown_backend"]["configuration"]["topdown_profile"] == panda_topdown.PROFILE
