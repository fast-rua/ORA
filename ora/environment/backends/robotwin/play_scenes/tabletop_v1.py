"""``tabletop-v1``: three tables with many objects, rebuilt from the lost long-horizon prototypes.

Worker only (imports RoboTwin's ``envs``). The prototypes' task source and
renderer were lost; these layouts are rebuilt from their asset lists and camera
images (``robotwin-long-horizon-v1``) and are like them, not identical. Frame:
RoboTwin's world, the table centred at the origin, ``x`` to the robot's right,
``y`` away from the robot, the table top at z 0.74 m. Every seed jitters the
layout; nothing here is a task.
"""

from __future__ import annotations

import numpy as np

from .base import IDENTITY, PlaySceneTask, jitter

#: A knife lying on its side: the blade's height (the model's z) turned into the table plane.
KNIFE_FLAT = (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)
#: RoboTwin's 034_knife is calibrated at 0.45 m long; at this scale it is 0.22 m, the fork's length.
KNIFE_SCALE = 0.15
#: RoboTwin's 027_table-tennis carries no calibrated scale (its mesh is 1.87 units across):
#: this makes a 40 mm ball.
BALL_SCALE = 0.04 / 1.8650
#: 027_table-tennis/base1 is the orange ball (base0 is white, hard to see on the white table).
BALL_MODEL_ID = 1


class table_setting_sequence(PlaySceneTask):
    """Plate, bowl and cup in a row near the robot; fork, knife and coaster beyond them."""

    def build_scene(self) -> None:
        near, far = -0.12, 0.13
        self.add_model("plate", "003_plate", 0, (-0.25 + jitter(0.012), near + jitter(0.012)))
        self.add_model("bowl", "002_bowl", 3, (0.0 + jitter(0.012), near + jitter(0.012)),
                       yaw=jitter(np.pi))
        self.add_model("cup", "021_cup", 0, (0.19 + jitter(0.012), near + jitter(0.012)), yaw=jitter(np.pi))
        # The fork's long axis is its model y, which already lies along the table's y.
        self.add_model("fork", "033_fork", 0, (-0.07 + jitter(0.01), far + jitter(0.012)),
                       yaw=jitter(0.25), upright=IDENTITY)
        self.add_model("knife", "034_knife", 0, (0.07 + jitter(0.01), far + jitter(0.012)),
                       yaw=jitter(0.25), upright=KNIFE_FLAT, scale=KNIFE_SCALE)
        self.add_model("coaster", "019_coaster", 0, (0.24 + jitter(0.01), far + jitter(0.012)))


class shell_game_memory_3(PlaySceneTask):
    """Three upright cups in a row and an orange ball in front of one of them."""

    def build_scene(self) -> None:
        row = 0.04 + jitter(0.01)
        cups = []
        for name, x in (("cup_left", -0.20), ("cup_middle", 0.0), ("cup_right", 0.20)):
            cups.append(x + jitter(0.01))
            self.add_model(name, "021_cup", 0, (cups[-1], row + jitter(0.006)), yaw=jitter(np.pi))
        front = cups[int(np.random.randint(3))]
        self.add_model("ball", "027_table-tennis", BALL_MODEL_ID, (front + jitter(0.01), -0.12 + jitter(0.01)),
                       scale=BALL_SCALE)


#: The prototype's colours, left to right: the blocks' and, lighter, their pads'.
BLOCK_COLOURS = {
    "red": (0.86, 0.18, 0.18), "blue": (0.16, 0.56, 0.95), "green": (0.20, 0.74, 0.32),
    "yellow": (0.96, 0.80, 0.16), "purple": (0.64, 0.36, 0.86), "pink": (0.98, 0.56, 0.76),
}
COLUMNS_X = (-0.275, -0.165, -0.055, 0.055, 0.165, 0.275)
BLOCK_HALF = (0.02, 0.02, 0.02)
PAD_HALF = (0.04, 0.04, 0.001)
WALL_HALF = (0.135, 0.01, 0.03)
DIVIDER_HALF = (0.01, 0.065, 0.03)
WALL_GREY = (0.62, 0.62, 0.64)


def _lighter(colour: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(round(0.55 * value + 0.45, 4) for value in colour)  # type: ignore[return-value]


class obstacle_rearrangement_6(PlaySceneTask):
    """Six blocks near the robot, two walls with a gap and a divider, six pads beyond the walls.

    The pads keep the prototype's colour order; the seed shuffles the blocks'
    colours along their row. Walls, divider and pads are fixed to the table.
    """

    def build_scene(self) -> None:
        names = list(BLOCK_COLOURS)
        order = [names[index] for index in np.random.permutation(len(names))]
        for name, x in zip(order, COLUMNS_X):
            self.add_box(f"block_{name}", (x + jitter(0.008), -0.15 + jitter(0.008)), BLOCK_HALF,
                         BLOCK_COLOURS[name], yaw=jitter(0.3))
        self.add_box("divider", (0.0, -0.095), DIVIDER_HALF, WALL_GREY, static=True)
        self.add_box("wall_left", (-0.165, 0.0), WALL_HALF, WALL_GREY, static=True)
        self.add_box("wall_right", (0.165, 0.0), WALL_HALF, WALL_GREY, static=True)
        for name, x in zip(names, COLUMNS_X):
            self.add_box(f"pad_{name}", (x, 0.12), PAD_HALF, _lighter(BLOCK_COLOURS[name]), static=True)


__all__ = ["obstacle_rearrangement_6", "shell_game_memory_3", "table_setting_sequence"]
