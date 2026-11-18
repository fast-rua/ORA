"""Original execution capture for the Synthetic world: a drawn top view of its end effectors.

Synthetic has no renderer; this draws what its state is, the two end effector
targets on the workspace plane, with height and gripper, so the operator's world
view (``ora.orchestrator.world_view``) and the recorded process video show the episode.
The frames are Control-only, like every execution capture: the policy keeps
the Synthetic observation it always received.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .execution_capture import ExecutionCapture

WIDTH, HEIGHT = 480, 320
#: Synthetic time: each action is this many capture steps of 0.1 s.
STEPS_PER_ACTION = 5
STEP_S = 0.1
DEFAULT_BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}
ARM_COLOURS = {"left": (86, 156, 255), "right": (255, 150, 70)}


def _bounds(value: Any) -> dict[str, tuple[float, float]]:
    result = {}
    for axis, default in DEFAULT_BOUNDS.items():
        pair = value.get(axis) if isinstance(value, Mapping) else None
        low, high = pair if isinstance(pair, (list, tuple)) and len(pair) == 2 else default
        result[axis] = (float(low), float(high)) if float(high) > float(low) else tuple(default)
    return result


def draw_top_view(end_effectors: Mapping[str, Mapping[str, Any]], *, bounds: Any = None,
                  label: str = "") -> Any:
    """An RGB uint8 array: the workspace from above, each arm's target, its height and gripper."""

    import numpy as np
    from PIL import Image, ImageDraw

    limits = _bounds(bounds)
    (x0, x1), (y0, y1), (z0, z1) = limits["x"], limits["y"], limits["z"]
    image = Image.new("RGB", (WIDTH, HEIGHT), (24, 26, 30))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 16, 28, WIDTH - 72, HEIGHT - 16
    # World +x points up the picture, +y to the left (a camera above, looking down).
    def pixel(x: float, y: float) -> tuple[float, float]:
        u = left + (y1 - y) / (y1 - y0) * (right - left)
        v = top + (x1 - x) / (x1 - x0) * (bottom - top)
        return u, v

    for step in range(int(round((x1 - x0) / 0.1)) + 1):
        _, v = pixel(x0 + step * 0.1, y0)
        draw.line([(left, v), (right, v)], fill=(44, 48, 56))
    for step in range(int(round((y1 - y0) / 0.1)) + 1):
        u, _ = pixel(x0, y0 + step * 0.1)
        draw.line([(u, top), (u, bottom)], fill=(44, 48, 56))
    draw.rectangle([left, top, right, bottom], outline=(110, 116, 128))
    origin = pixel(0.0, 0.0)
    draw.line([origin, pixel(0.1, 0.0)], fill=(200, 80, 80), width=2)
    draw.line([origin, pixel(0.0, 0.1)], fill=(80, 200, 80), width=2)
    draw.text((left, 8), f"Synthetic world, top view{(' · ' + label) if label else ''}", fill=(200, 204, 212))
    # Height gauge on the right: one marker per arm.
    gauge = (right + 24, top, right + 40, bottom)
    draw.rectangle(gauge, outline=(110, 116, 128))
    draw.text((right + 16, bottom + 2), "z", fill=(150, 156, 168))
    for arm in sorted(end_effectors):
        state = end_effectors[arm]
        pose = list(state.get("pose_xyz_xyzw") or [0, 0, 0])
        x, y, z = (float(pose[index]) for index in range(3))
        gripper = float(state.get("gripper", 1.0))
        colour = ARM_COLOURS.get(arm, (220, 220, 220))
        u, v = pixel(min(max(x, x0), x1), min(max(y, y0), y1))
        radius = 6 + 10 * min(max((z - z0) / (z1 - z0), 0.0), 1.0)
        box = [u - radius, v - radius, u + radius, v + radius]
        # An open gripper is a ring, a closed one a disc.
        if gripper >= 0.5:
            draw.ellipse(box, outline=colour, width=3)
        else:
            draw.ellipse(box, fill=colour)
        # Labels of arms at one place stay apart: left above, right below.
        draw.text((u + radius + 4, v - (16 if arm == "left" else -2)), f"{arm} z={z:.2f} gripper={gripper:.1f}",
                  fill=colour)
        level = gauge[3] - min(max((z - z0) / (z1 - z0), 0.0), 1.0) * (gauge[3] - gauge[1])
        draw.line([(gauge[0] - 4, level), (gauge[2] + 4, level)], fill=colour, width=3)
    return np.asarray(image, dtype=np.uint8)


#: The Synthetic operator view: the robot from the front, for the human only (never an observation).
OPERATOR_VIEW_CAMERA = "operator_front_view"
OPERATOR_WIDTH, OPERATOR_HEIGHT = 640, 400


def draw_operator_view(end_effectors: Mapping[str, Mapping[str, Any]], *, bounds: Any = None) -> bytes:
    """A PNG of the two arms seen from the front (y across, z up): what a third-person camera would show."""

    import io

    from PIL import Image, ImageDraw

    limits = _bounds(bounds)
    (_x0, _x1), (y0, y1), (z0, z1) = limits["x"], limits["y"], limits["z"]
    image = Image.new("RGB", (OPERATOR_WIDTH, OPERATOR_HEIGHT), (232, 236, 240))
    draw = ImageDraw.Draw(image)
    left, right, top, table = 40, OPERATOR_WIDTH - 40, 40, OPERATOR_HEIGHT - 70

    def pixel(y: float, z: float) -> tuple[float, float]:
        # The camera faces the robot: +y (the robot's left) is on the picture's right.
        u = left + (y - y0) / (y1 - y0) * (right - left)
        v = table - (z - z0) / (z1 - z0) * (table - top)
        return u, v

    draw.rectangle([0, table, OPERATOR_WIDTH, OPERATOR_HEIGHT], fill=(196, 178, 150))
    draw.line([(0, table), (OPERATOR_WIDTH, table)], fill=(150, 130, 104), width=2)
    draw.text((left, 12), "Synthetic world, operator's front view", fill=(70, 76, 86))
    for arm in sorted(end_effectors):
        state = end_effectors[arm]
        pose = list(state.get("pose_xyz_xyzw") or [0, 0, 0])
        y, z = float(pose[1]), float(pose[2])
        gripper = float(state.get("gripper", 1.0))
        colour = ARM_COLOURS.get(arm, (120, 120, 120))
        base = pixel(0.35 * (y1 if arm == "left" else y0), z0)
        hand = pixel(min(max(y, y0), y1), min(max(z, z0), z1))
        # Two hands at one point stay apart on the picture, each on its own arm's side.
        hand = (hand[0] + (18 if arm == "left" else -18), hand[1])
        elbow = ((base[0] + hand[0]) / 2 + (-30 if arm == "left" else 30), min(base[1], hand[1]) - 40)
        draw.rectangle([base[0] - 26, table - 6, base[0] + 26, table + 18], fill=(90, 96, 108))
        draw.line([base, elbow, hand], fill=colour, width=12, joint="curve")
        for point in (base, elbow):
            draw.ellipse([point[0] - 9, point[1] - 9, point[0] + 9, point[1] + 9], fill=(60, 64, 72))
        # Two jaws below the hand: apart when open, together when closed.
        gap = 4 + 14 * min(max(gripper, 0.0), 1.0)
        for side in (-1, 1):
            u = hand[0] + side * gap
            draw.rectangle([u - 3, hand[1], u + 3, hand[1] + 22], fill=(50, 54, 60))
        draw.text((hand[0] + (12 if arm == "left" else -92), hand[1] + 26), f"{arm} z={z:.2f}", fill=colour)
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=1)
    return output.getvalue()


class SyntheticCapture:
    """One recorder for a Synthetic session: reset, action, control, terminal and close frames."""

    def __init__(self, output_root: Path, *, task: str, cameras: list[str], config: Mapping[str, Any],
                 bounds: Any = None) -> None:
        self.recorder = ExecutionCapture(
            output_root, backend="synthetic",
            identity={"run_id": output_root.parent.name, "task": task},
            cameras=list(cameras), config=config,
        )
        self.bounds = bounds
        self.round_index = 0
        self.step = 0

    def _frame(self, end_effectors: Mapping[str, Any], phase: str, *, action: Mapping[str, Any] | None = None,
               sequence: int | None = None, force: bool = True) -> None:
        recorder = self.recorder
        label = f"round {self.round_index} · {phase}"

        def images() -> dict[str, Any]:
            picture = draw_top_view(end_effectors, bounds=self.bounds, label=label)
            return {camera: picture for camera in recorder.cameras}

        recorder.capture(
            images, round_index=self.round_index,
            action_id=(action or {}).get("action_id"), action=(action or {}).get("action"),
            action_sequence=sequence, sim_step=self.step, sim_time_s=self.step * STEP_S,
            step_unit="synthetic_capture_step", phase=phase, force=force,
        )

    def reset(self, round_index: int, end_effectors: Mapping[str, Any]) -> None:
        self.round_index, self.step = round_index, 0
        self._frame(end_effectors, "reset")

    def action(self, action: Mapping[str, Any], sequence: int, before: Mapping[str, Any],
               after: Mapping[str, Any], *, wall_s: float = 0.0) -> None:
        """The action's start, the motion between its two states, and its end.

        With ``wall_s`` (a demo's ACTION_WALL_ENV) the motion takes that long:
        one capture step per STEP_S of wall time, so its frames arrive as a real
        robot's would.
        """

        import time

        steps = max(STEPS_PER_ACTION, round(wall_s / STEP_S)) if wall_s > 0 else STEPS_PER_ACTION
        self._frame(before, "action_start", action=action, sequence=sequence)
        for index in range(1, steps):
            if wall_s > 0:
                time.sleep(wall_s / steps)
            self.step += 1
            share = index / steps
            between = {}
            for arm, state in after.items():
                start = before.get(arm, state)
                pose = [a + (b - a) * share for a, b in zip(start["pose_xyz_xyzw"], state["pose_xyz_xyzw"])]
                gripper = start["gripper"] + (state["gripper"] - start["gripper"]) * share
                between[arm] = {"pose_xyz_xyzw": pose, "gripper": gripper}
            self._frame(between, "control", action=action, sequence=sequence, force=False)
        if wall_s > 0:
            time.sleep(wall_s / steps)
        self.step += 1
        self._frame(after, "action_end", action=action, sequence=sequence)

    def terminal(self, end_effectors: Mapping[str, Any]) -> None:
        self._frame(end_effectors, "terminal")

    def close(self, end_effectors: Mapping[str, Any]) -> None:
        self._frame(end_effectors, "close")
        self.recorder.close()


__all__ = ["OPERATOR_VIEW_CAMERA", "SyntheticCapture", "draw_operator_view", "draw_top_view"]
