"""Prepared grasp regression: source binding, routing, staged effects and evidence."""


import numpy as np
import pytest

from ora.agent.toolkit.host.skills import lift_evidence as skill_retention


@pytest.mark.parametrize("profile", [
    "robotwin_toolchain_b2_closed_noservo_v2", "robodojo_placement_toolkit_v5",
])
def test_maintained_defaults_bind_prepared_grasp_and_release_disclosure(profile):
    from tests.support.agent.surface_catalog import surface_of
    surface = surface_of(profile)
    assert surface.supports(skill_retention.CAPABILITY)
    help_text = next(d["description"] for d in surface.definitions if d["name"] == "grasp_execute")
    assert "releases anything currently held" in help_text
    assert "five stage actions" in help_text
    assert "completed sequence only records execution" in help_text


@pytest.mark.parametrize("case,expected", [
    ("moving", "rigid_motion_supported"), ("stationary", "stationary_surface_supported"),
    ("both", "unknown"), ("occluded", "unknown"), ("baseline_moved", "unknown"),
    ("tiny_lift", "unknown"),
])
def test_surface_hypotheses_preserve_motion_and_ambiguity(case, expected):
    x, y = np.meshgrid(np.linspace(-.02, .02, 9), np.linspace(-.02, .02, 9))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    delta = np.array([0., 0., .003 if case == "tiny_lift" else .03])
    after = points + delta
    before = points.copy()
    if case == "stationary":
        after = points.copy()
    elif case == "both":
        after = np.vstack((points, after))
    elif case == "occluded":
        after = after[:4]
    elif case == "baseline_moved":
        before += [.1, 0., 0.]
    evidence = skill_retention.compare_surfaces(
        points=points, before_scene=before, after_scene=after,
        before_position=np.zeros(3), after_position=delta,
        before_rotation=np.eye(3), after_rotation=np.eye(3),
    )
    assert evidence["status"] == expected
    assert "semantic_identity_unverified" in evidence["scope"]
