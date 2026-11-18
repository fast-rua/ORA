"""Jaw symmetry: a parallel jaw turned a half-turn about its approach axis is the same jaw.

Pure computation. Every orientation of the ARX X5 jaws has a twin (the same
orientation turned 180 degrees about the controlled point's approach axis,
link6 local +X); an empty jaw at either one grips the same way, but the wrist
travels very differently to reach them. The default motion tools apply one
rule on every interaction and orientation form (``nearest_jaw_twin``):

* an empty jaw takes whichever of the requested orientation and its twin is
  nearer the current measured wrist orientation (``requested`` or
  ``twin_nearer_current``);
* a jaw holding something never takes the twin of a request with a sign, since
  the held object would turn a half-turn with it (``held_no_flip``); a jaw line
  has no sign, so a request that names one takes the nearer of the two here too:
  the hand then keeps the object as it holds it;
* an explicit exact request is used as given (``exact``).

A tool of its own may call the same function, or deviate from it.
"""
from __future__ import annotations

import math
from typing import Sequence

#: How a target's orientation was chosen between the requested one and its jaw twin.
JAW_SYMMETRY_LABELS = ("requested", "twin_nearer_current", "held_no_flip", "exact")
#: The controlled point's approach axis in its own frame (link6 local +X on the ARX X5).
APPROACH_LOCAL = (1.0, 0.0, 0.0)


def _unit(quaternion: Sequence[float]) -> list[float]:
    values = [float(v) for v in quaternion]
    norm = math.sqrt(sum(v * v for v in values))
    if len(values) != 4 or not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("an orientation is four finite numbers (x, y, z, w) with a non-zero norm")
    return [v / norm for v in values]


def _multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw, lw * rw - lx * rx - ly * ry - lz * rz]


def turn_deg(first: Sequence[float], second: Sequence[float]) -> float:
    """Pure computation: the rotation (degrees) between two xyzw orientations.

    Example: turn_deg([0, 0, 0, 1], [1, 0, 0, 0]) == 180.0."""
    a, b = _unit(first), _unit(second)
    return math.degrees(2.0 * math.acos(min(1.0, abs(sum(x * y for x, y in zip(a, b))))))


def jaw_twin(orientation_xyzw: Sequence[float], approach_local: Sequence[float] = APPROACH_LOCAL) -> list[float]:
    """Pure computation: the orientation turned a half-turn about its own approach axis (xyzw, w >= 0).

    Example: jaw_twin([0, 0, 0, 1]) == [1.0, 0.0, 0.0, 0.0]."""
    axis = [float(v) for v in approach_local]
    norm = math.sqrt(sum(v * v for v in axis))
    if len(axis) != 3 or norm < 1e-9:
        raise ValueError("the approach axis is three numbers with a non-zero norm")
    turned = _multiply(_unit(orientation_xyzw), [v / norm for v in axis] + [0.0])
    sign = -1.0 if turned[3] < 0 or (turned[3] == 0 and next((v for v in turned if v), 0.0) < 0) else 1.0
    return [sign * v + 0.0 for v in turned]


def nearest_jaw_twin(current_xyzw: Sequence[float], requested_xyzw: Sequence[float], *, held: bool = False,
                     exact: bool = False, signless: bool = False,
                     approach_local: Sequence[float] = APPROACH_LOCAL) -> dict:
    """Pure computation: the orientation to command for ``requested_xyzw`` under the jaw-symmetry rule.

    Returns ``{"orientation_xyzw": [...], "jaw_symmetry": label, "turn_deg": {"requested": a, "twin": b}}``:
    ``exact`` keeps the request (``exact``); ``held`` keeps it (``held_no_flip``) unless ``signless``, a request
    written as a jaw line, which has no sign; otherwise the twin is taken
    only when it is strictly nearer the current orientation (``twin_nearer_current``), else the request
    (``requested``). ``turn_deg`` is the rotation from the current orientation to each.
    Example: nearest_jaw_twin([0, 0, 0, 1], [1, 0, 0, 0])['jaw_symmetry'] == 'twin_nearer_current'."""
    requested = _unit(requested_xyzw)
    twin = jaw_twin(requested, approach_local)
    turns = {"requested": round(turn_deg(current_xyzw, requested), 3), "twin": round(turn_deg(current_xyzw, twin), 3)}
    if exact:
        return {"orientation_xyzw": requested, "jaw_symmetry": "exact", "turn_deg": turns}
    if held and not signless:
        return {"orientation_xyzw": requested, "jaw_symmetry": "held_no_flip", "turn_deg": turns}
    if turn_deg(current_xyzw, twin) < turn_deg(current_xyzw, requested) - 1e-6:
        return {"orientation_xyzw": twin, "jaw_symmetry": "twin_nearer_current", "turn_deg": turns}
    return {"orientation_xyzw": requested, "jaw_symmetry": "requested", "turn_deg": turns}


__all__ = ["APPROACH_LOCAL", "JAW_SYMMETRY_LABELS", "jaw_twin", "nearest_jaw_twin", "turn_deg"]
