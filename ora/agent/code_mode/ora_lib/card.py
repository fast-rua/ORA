"""The frames card: how this scene's frames relate, written from the current published observation.

``frames_card()`` returns a short text a script prints (``from ora_lib import frames_card``): the conventions
(units, axes, quaternion order), then for each camera where it is and how the world's axes show in its image, and
for each arm where its controlled point is and which way its local axes point. Everything in it is computed from
what the observation carries -- each camera's published calibration and each arm's measured pose -- with the
arithmetic of ``frames``, so it states nothing a script could not compute itself, and it is the same text for the
same observation. The calibration of a scene is per observation, not known when the prompt is composed
(``docs/robots/frames-and-units.md``), which is why the card is written here and not in the prompt.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import observation as _observation

#: A component of a unit vector smaller than this (about 14 degrees off the axis) is not worth a word.
SIGNIFICANT = 0.25
WORLD_AXES = (("+X", (1.0, 0.0, 0.0)), ("+Y", (0.0, 1.0, 0.0)), ("+Z", (0.0, 0.0, 1.0)))

CONVENTIONS = ("Metres and radians (a name says when not, e.g. angle_deg); quaternions [x, y, z, w]; world +Z is up; "
               "camera:<name> is OpenCV (x right, y down, z forward); pixel:<name> has u right and v down from the "
               "top-left of the image.")


def _vector(v: Any) -> str:
    return "(" + ", ".join(f"{round(float(x), 2) + 0.0:.2f}" for x in v) + ")"  # + 0.0: no "-0.00"


def _words(components: np.ndarray) -> str:
    """Where a camera-frame vector ``(right, down, forward)`` points in that camera's view, in words."""

    right, down, forward = (float(x) for x in components)
    parts = sorted(((abs(right), "right" if right > 0 else "left"), (abs(down), "down" if down > 0 else "up"),
                    (abs(forward), "away from the camera" if forward > 0 else "toward the camera")), reverse=True)
    named = [name for size, name in parts if size >= SIGNIFICANT]
    return " and ".join(named) if named else "toward the camera"


def _camera(camera: _observation.Camera) -> list[str]:
    to_world = camera.camera_to_world
    rotation, position = to_world[:3, :3], to_world[:3, 3]
    right, down, forward = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    fixed = camera.mount in (None, "fixed_world")
    pitch = float(np.degrees(np.arcsin(np.clip(-forward[2], -1.0, 1.0))))
    lines = [f"{camera.name} ({'fixed' if fixed else 'mount ' + str(camera.mount) + ', moves with the robot'}, "
             f"{camera.width}x{camera.height}): at world {_vector(position)}, looking along world {_vector(forward)}, "
             f"{abs(pitch):.0f} deg {'below' if pitch > 0 else 'above'} horizontal."]
    flat = np.array([-down[0], -down[1], 0.0])
    up = f"; image up along the table = world {_vector(flat / np.linalg.norm(flat))}" if np.linalg.norm(flat) > 1e-6 else ""
    lines.append(f"  Image right = world {_vector(right)}{up}.")  # up laid flat: away from the camera on the table
    shows = "; ".join(f"{name} shows {_words(rotation.T @ np.array(axis))}" for name, axis in WORLD_AXES)
    lines.append(f"  World {shows}.")
    if not fixed:
        lines.append("  This view moves: these numbers are the current ones; read them again after a move.")
    return lines


def _arm(observation: _observation.Observation, arm: str) -> str:
    pose = observation.pose(arm)
    axes = ", ".join(f"{name} {_vector(pose.axis(name).v)}" for name in ("x", "y", "z"))
    return f"tcp:{arm}: at world {_vector(pose.xyz)}; its local axes in world: {axes}."


def frames_card(observation: _observation.Observation | None = None) -> str:
    """The frames card of ``observation`` (default: the current published one), a text to print."""

    observation = observation if observation is not None else _observation.current()
    lines = [f"Frames card, observation {observation.observation_id}"
             + (f" (step {observation.step})" if observation.step is not None else "") + ".", CONVENTIONS]
    cameras = [observation.cameras[name] for name in sorted(observation.cameras)]
    main = observation.main_camera
    cameras.sort(key=lambda camera: (camera.name != main, camera.mount not in (None, "fixed_world")))
    for camera in cameras:
        lines += _camera(camera)
    for name, error in sorted(observation.camera_errors.items()):
        lines.append(f"{name}: not available in this observation ({error.get('message') or error}).")
    for arm in observation.arms():
        lines.append(_arm(observation, arm))
    if observation.arms():
        lines.append("Which local axis is the approach axis (the fingers' direction) and which the jaw line is in the "
                     "robot's manual; ora_lib's Pose.axis('x' | 'y' | 'z') gives any of them in world.")
    return "\n".join(lines)


__all__ = ["frames_card"]
