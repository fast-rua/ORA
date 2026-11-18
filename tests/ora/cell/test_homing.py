"""``homing.drive``: the outcomes its callers' tests (operator/test_actions.py, the YAM worker's test_real_rig.py)
do not reach on their fakes: a control loop that stops on the way, an arm that settles short of rest, and time."""

from __future__ import annotations

import numpy as np

from ora.cell import homing

GOAL = {"left": np.zeros(6), "right": np.zeros(6)}
AWAY = {"left": np.full(6, 0.2), "right": np.full(6, -0.2)}


class Arms:
    def __init__(self, sag: float = 0.0) -> None:
        self.q = {arm: q.copy() for arm, q in AWAY.items()}
        self.running = {"left": True, "right": True}
        self.sag, self.ticks = sag, 0

    def read(self):
        return {arm: q + self.sag for arm, q in self.q.items()}

    def command(self, targets):
        self.ticks += 1
        self.q.update({arm: np.asarray(q, float) for arm, q in targets.items()})


def settings(**changes) -> homing.HomeSettings:
    values = dict(speed_rad_s=10.0, period_s=0.01, tracking_stop_rad=0.1, tolerance_rad=0.02, limit_s=5.0)
    return homing.HomeSettings(**{**values, **changes})


def test_a_control_loop_that_stops_on_the_way_ends_the_drive_and_names_the_arm() -> None:
    arms = Arms()

    def wait() -> None:
        if arms.ticks == 2:
            arms.running["right"] = False

    outcome = homing.drive(GOAL, read=arms.read, command=arms.command, alive=lambda: dict(arms.running),
                           wait=wait, settings=settings(speed_rad_s=1.0))
    assert (outcome.state, outcome.arms) == ("dead", ["right"])


def test_an_arm_that_settles_short_of_rest_ends_not_at_rest_after_the_settle_time() -> None:
    arms = Arms(sag=0.05)
    outcome = homing.drive(GOAL, read=arms.read, command=arms.command, alive=lambda: dict(arms.running),
                           wait=lambda: None, settings=settings(settle_s=0.0))
    assert outcome.state == "not_at_rest" and abs(outcome.off_rad - 0.05) < 1e-9
    # without a settle time it waits for rest until the time runs out
    outcome = homing.drive(GOAL, read=arms.read, command=arms.command, alive=lambda: dict(arms.running),
                           wait=lambda: None, settings=settings(limit_s=0.05))
    assert outcome.state == "timeout"


def test_arms_that_follow_reach_rest_and_nothing_moves_on_a_refused_line() -> None:
    from tests.support.cell.fake import FakeKinematics

    arms = Arms()
    outcome = homing.drive(GOAL, read=arms.read, command=arms.command, alive=lambda: dict(arms.running),
                           wait=lambda: None, settings=settings(), kinematics=FakeKinematics(inter_arm_m=0.01))
    assert outcome.state == "path_blocked" and arms.ticks == 0
    outcome = homing.drive(GOAL, read=arms.read, command=arms.command, alive=lambda: dict(arms.running),
                           wait=lambda: None, settings=settings(), kinematics=FakeKinematics())
    assert outcome.state == "at_rest" and arms.ticks > 0
