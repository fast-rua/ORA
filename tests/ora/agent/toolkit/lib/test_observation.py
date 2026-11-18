"""Typed robot state keeps what a backend publishes and guesses no frame."""

from ora.agent.toolkit.lib import observation
from ora.agent.toolkit.lib.values import JsonValue, PoseM


def test_a_pose_without_a_frame_stays_untyped_and_public():
    state = observation.robot_state(JsonValue.of(dict(end_effectors=dict(
        left=dict(pose_xyz_xyzw=[0., 0., 1., 0., 0., 0., 1.]),
        right=dict(pose_xyz_xyzw=[0., 0., 1., 0., 0., 0., 1.], pose_frame="world")))))
    left, right = state.arms
    assert left.pose is None and left.public.json()["pose_xyz_xyzw"][2] == 1.
    assert isinstance(right.pose, PoseM)
