"""Propose one close of the one gripper; the pose target holds."""

from ..move_pose import _atomic


def run(api, arguments):
    """Replay: recorded observation and action; the receipt comes with the next observation, opening normalized."""
    return _atomic.set_jaws(api, arguments, name="close_gripper", opening=0.0)
