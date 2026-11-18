"""The toolkit's speed presets are the shared ones, so a move_pose speed means one thing everywhere.

``tools/move_pose/_smooth.py`` names ``Pace`` objects a tool can pass on and
``ora/contracts/motion_speed.py`` states the same numbers for every backend;
agent and environment may not import each other, so this test holds them together.
"""

from __future__ import annotations

import pytest

from ora.agent.toolkit.lib.motion_plan import timing
from ora.agent.toolkit.tools.move_pose import _smooth
from ora.contracts import motion_speed


def test_the_presets_are_the_toolkits():
    for name, pace in (("fast", _smooth.FULL), ("normal", _smooth.NORMAL), ("careful", _smooth.CAREFUL)):
        assert motion_speed.PRESET_CAPS[name] == pace.caps()
    assert timing.SLOW_APPROACH_M == motion_speed.SLOW_APPROACH_M
    assert motion_speed.SLOW_PRESET == "careful" and motion_speed.DEFAULT_SPEED == _smooth.DEFAULT_SPEED


@pytest.mark.parametrize("speed", [0.05, 0.1, 0.25, 1.0])
def test_a_numeric_speed_is_the_toolkits(speed):
    assert motion_speed.custom_caps(speed) == _smooth.custom_pace(speed).caps()


def test_a_pace_is_an_explicit_speed_request():
    request = {"speed": motion_speed.EXPLICIT_SPEED, "caps": _smooth.NORMAL.caps(), "slow_caps": _smooth.CAREFUL.caps(),
               "slow_approaching": True}
    assert motion_speed.speed_caps(request) == (_smooth.NORMAL.caps(), _smooth.CAREFUL.caps())
    with pytest.raises(ValueError):
        motion_speed.validate_speed({**request, "slow_caps": None})
