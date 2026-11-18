"""Constants and helpers shared by ``core`` and its mixins: the policy cameras and the policy-state forwarders."""

from __future__ import annotations

from typing import Any


# RoboDojo publishes three metric RGB-D cameras under its own names; the
# capability gates them exactly like robotwin_rgbd_v1 gates the RoboTwin set.
ROBODOJO_POLICY_CAMERAS = frozenset({"head", "left_wrist", "right_wrist"})
# The YAM station's two fixed and two wrist RGB-D cameras, gated by yam_rgbd_v1.
YAM_POLICY_CAMERAS = frozenset({"overhead", "front", "wrist_left", "wrist_right"})


def load_policy_state(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Call ``core.load_policy_state``, looked up when called.

    Tests replace ``load_policy_state`` and ``verify_policy_state`` on the
    ``core`` module, some of them with a stand-in host object that has none
    of the runtime's methods, so the mixins that read policy state import these
    two forwarders instead of the evidence functions.
    """

    from ora.agent.toolkit.host.runtime import core

    return core.load_policy_state(*args, **kwargs)


def verify_policy_state(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Call ``core.verify_policy_state``, looked up when called."""

    from ora.agent.toolkit.host.runtime import core

    return core.verify_policy_state(*args, **kwargs)
