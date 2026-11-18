"""The preview of one action: its planned path over time, what it costs and what stands near it.

Every moving action on every backend answers a dry run with the segments of ``contracts/action_plan.py``; this turns them
into the one result: per arm the native actions, how far and how much the hand turns, how high its path rises above
its ends and the least clearance against the public depth, and one image from ``_preview_render``. Everything here is
planner output, forward kinematics, public calibration and public depth, so nothing depends on a simulator and no physics
is rolled out.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import clearance
from ._clearance import rotation
from ._preview_render import Camera, render
from .values import JsonValue

#: Says what the preview does not predict, once in every answer.
CLAIM = ("A preview of the planned motion: planner output and forward kinematics from the measured start, drawn with the "
         "published camera calibration on the latest image. It runs no physics, so it predicts no contact, no weight or "
         "slip of what the hand holds and no response of the world, and says nothing of what follows the first contact. A dashed line is commanded, "
         "not planned: a servo follows it and only its run says how. ")
#: What ``image`` says when the run renders no preview picture (``tools/_latency.py``).
PICTURE_OFF = {"status": "off", "reason": "this run renders no preview pictures (latency profile)"}
#: Where a hand's public jaw geometry is missing: the jaw centre is the controlled point.
NO_HAND = {"contact_center_local": [0.0, 0.0, 0.0], "opening_axis_local": None, "maximum_aperture_m": None}
#: How long a drawn finger is, back from the jaw centre along the approach axis. A sketch of the jaws' place and
#: direction, not their collision geometry.
FINGER_DRAWN_M = 0.07


def cameras_from(manifest: Mapping[str, Any], read: Callable[[str, str], Any]) -> list[Camera]:
    """Every camera the manifest lists, in its order, that has an image and a calibration. It names no camera.

    ``read(camera, role)`` returns the file of a ``files`` role, decoded: an RGB array, a depth array or a mapping;
    None or an exception means that camera has none."""
    found = []
    for name, record in (manifest.get("cameras") or {}).items():
        if not isinstance(record, Mapping):
            continue
        try:
            rgb, calibration = read(name, "rgb_lossless_png"), read(name, "calibration")
            depth = read(name, "depth_numpy_m_f32")
        except Exception:  # a camera this observation does not publish is simply not drawn on
            continue
        if rgb is None or calibration is None:
            continue
        extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
        found.append(Camera(name, np.asarray(rgb), None if depth is None else np.asarray(depth),
                            np.asarray(calibration["intrinsic_cv"], dtype=np.float64), extrinsic))
    return found


def jaw_point(pose: Sequence[float], local: Sequence[float]) -> list[float]:
    """Pure computation: a point of the hand frame in the world. Example: jaw_point(pose, [0, 0, 0.1])."""
    return [float(value) for value in np.asarray(pose[:3], float) + rotation(pose[3:]) @ np.asarray(local, float)]


def _turn_deg(first: Sequence[float], second: Sequence[float]) -> float:
    a, b = np.asarray(first, float), np.asarray(second, float)
    dot = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def _moving(segments: Sequence[Mapping[str, Any]], arm: Any) -> list[Mapping[str, Any]]:
    return [item for item in segments if item["arm"] == arm and item["kind"] != "gripper" and not item["not_predicted"]]


def arm_numbers(segments: Sequence[Mapping[str, Any]], arm: Any,
                hand: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One arm's cost up to the first contact: native actions where they are known, travel and turn of the hand, and
    how high the path rises: ``rise_above_ends_m``, its highest point above the higher of its two ends (0 where an end
    is the highest), and ``rise_at_fraction``, where that point is (the first such pose of ``path_of``, 0 at the start
    and 1 at the end, as clearance's ``at_fraction``); both None for an arm with no path. The height is the jaw contact
    centre's where ``hand`` gives the public jaw geometry (``contact_center_local``), else the controlled point's."""
    parts = _moving(segments, arm)
    known = [item["native_actions"] for item in parts if item["native_actions"] is not None]
    travel = turn = 0.0
    for item in parts:
        for first, second in zip(item["poses"], item["poses"][1:]):
            travel += math.dist(first[:3], second[:3])
            turn += _turn_deg(first[3:], second[3:])
    native = sum(known) if known else None
    rise, at = _rise(path_of(segments, arm), (hand or NO_HAND)["contact_center_local"])
    return {"native_actions": native, "servo_not_counted": len(known) < len(parts),
            "hand_travel_m": round(travel, 4), "hand_turn_deg": round(turn, 2),
            "rise_above_ends_m": rise, "rise_at_fraction": at}


def _rise(path: Sequence[Sequence[float]], local: Sequence[float]) -> tuple[float | None, float | None]:
    if len(path) < 2:
        return None, None
    heights = [jaw_point(pose, local)[2] for pose in path]
    peak = max(range(len(heights)), key=heights.__getitem__)
    return round(heights[peak] - max(heights[0], heights[-1]), 4), round(peak / (len(heights) - 1), 3)


def path_of(segments: Sequence[Mapping[str, Any]], arm: Any) -> list[list[float]]:
    """One arm's whole path up to the first contact, controlled-point poses in order."""
    path: list[list[float]] = []
    for item in _moving(segments, arm):
        path.extend(item["poses"])
    return path


def _approach(hand: Mapping[str, Any]) -> np.ndarray | None:
    """The hand's approach axis in its own frame: declared, else along its jaw centre's offset; None where neither."""
    declared = hand.get("approach_axis_local")
    axis = np.asarray(declared if declared is not None else hand["contact_center_local"], float)
    length = float(np.linalg.norm(axis))
    return axis / length if length > 1e-9 else None


def jaws_at(pose: Sequence[float], hand: Mapping[str, Any], width_m: float | None) -> list[list[float]] | None:
    """The jaws at ``pose`` as four world points, one finger's base and tip, then the other's tip and base: a sketch of
    where they are and which way they point. None where the hand has no opening or approach axis.
    Example: jaws_at(pose, hands["left"], 0.06)."""
    axis, approach = hand.get("opening_axis_local"), _approach(hand)
    width = width_m if width_m is not None else hand.get("maximum_aperture_m")
    if axis is None or approach is None or width is None:
        return None
    centre, side = np.asarray(hand["contact_center_local"], float), 0.5 * float(width) * np.asarray(axis, float)
    tips = (centre + side, centre - side)
    return [jaw_point(pose, tips[0] - FINGER_DRAWN_M * approach), jaw_point(pose, tips[0]),
            jaw_point(pose, tips[1]), jaw_point(pose, tips[1] - FINGER_DRAWN_M * approach)]


def openings_m(manifest: Mapping[str, Any], hands: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """Each arm's measured jaw opening in metres, from the published normalized opening and the widest aperture."""
    found = {}
    for arm, effector in ((manifest.get("proprioception") or {}).get("end_effectors") or {}).items():
        opening, aperture = (effector or {}).get("gripper"), (hands.get(arm) or {}).get("maximum_aperture_m")
        if type(opening) in (int, float) and aperture is not None:
            found[arm] = max(0.0, min(1.0, float(opening))) * float(aperture)
    return found


def drawn(segments: Sequence[Mapping[str, Any]], hands: Mapping[str, Mapping[str, Any]],
          openings: Mapping[str, float] | None = None,
          held_boxes: Mapping[Any, Sequence[Sequence[float]]] | None = None) -> list[dict[str, Any]]:
    """The segments with the world points a drawing needs: the jaw centre's path, the jaws where an arm's path starts
    and ends (at the measured opening, else the widest), the box an arm holds there (``held_boxes``, eight corners in
    its controlled-point frame) and the fingers of a jaw motion."""
    result = []
    for item in segments:
        hand = hands.get(item["arm"]) or NO_HAND
        points = [jaw_point(pose, hand["contact_center_local"]) for pose in item["poses"]]
        entry = {**item, "points": points}
        if item["kind"] == "arm" and item["poses"]:
            width = (openings or {}).get(item["arm"])
            entry["jaws_start"] = jaws_at(item["poses"][0], hand, width)
            entry["jaws_end"] = jaws_at(item["poses"][-1], hand, width)
            box = (held_boxes or {}).get(item["arm"])
            if box is not None:
                entry["held_start"] = [jaw_point(item["poses"][0], corner) for corner in box]
                entry["held_end"] = [jaw_point(item["poses"][-1], corner) for corner in box]
        axis = hand["opening_axis_local"]
        if item["kind"] == "gripper" and axis is not None:
            centre, pose = np.asarray(hand["contact_center_local"], float), item["poses"][-1]
            for key, width in (("fingers_from", item.get("jaw_width_from_m")), ("fingers_to", item.get("jaw_width_to_m"))):
                if width is not None:
                    entry[key] = [jaw_point(pose, centre + sign * 0.5 * float(width) * np.asarray(axis, float))
                                  for sign in (1.0, -1.0)]
        result.append(entry)
    return result


#: How a summary line says a clearance's ``part`` reaches into what the depth shows.
_REACH = {"fingers": "the jaws reach", "held_object": "the held object's box reaches"}


def _holds(holding: Mapping[str, Any]) -> str:
    """How a summary line says what an arm holds (``clearance.held``)."""
    size = holding.get("box_size_m")
    box = "x".join(f"{value * 1000:.0f}" for value in size) + " mm" if size else ""
    if holding.get("extent") == "localized":
        return f"holds {holding.get('target_ref') or 'its target'} (its localized box, {box})"
    gap = holding.get("jaw_gap_m")
    apart = "" if gap is None else f", {gap * 1000:.0f} mm apart"
    if holding.get("extent") == "between_jaws":
        return f"holds what its jaws closed on ({box} between the jaws{apart}; what reaches past them is not measured)"
    return f"holds what its jaws closed on (its size is not measured{apart})"


def _line(arm: Any, numbers: Mapping[str, Any], seen: Mapping[str, Any],
          holding: Mapping[str, Any] | None = None) -> str:
    who = ("base" if arm is None else str(arm)) + (f": {_holds(holding)}," if holding else ":")
    cost = ("a servo, whose cost only its run says" if numbers["native_actions"] is None else
            f"{numbers['native_actions']} native actions" + (" and a servo" if numbers["servo_not_counted"] else ""))
    near = (f"{_REACH.get(seen.get('part'), _REACH['fingers'])} {seen['inside_m'] * 1000:.0f} mm into what the depth "
            f"shows at {seen['at_fraction'] * 100:.0f}% of the path"
            if seen.get("status") == "measured" and seen.get("inside_m") else
            f"least clearance {seen['clearance_m'] * 1000:.0f} mm at {seen['at_fraction'] * 100:.0f}% of the path"
            if seen.get("status") == "measured" else
            f"nothing within {seen['clearance_at_least_m'] * 1000:.0f} mm of the jaws" if seen.get("status") == "beyond"
            else "too short for a clearance beyond its ends" if seen.get("status") == "ends_only"
            else "clearance unknown")
    entered = seen.get("enters_cleared_space")
    near += f", enters what the planner clears at {entered['at_fraction'] * 100:.0f}%" if entered else ""
    rise, at = numbers["rise_above_ends_m"], numbers["rise_at_fraction"]
    risen = ("" if rise is None else "never rises above its higher end, " if rise == 0 and at in (0.0, 1.0) else
             f"rises {rise * 1000:.0f} mm above its higher end at {at * 100:.0f}% of the path, ")
    return (f"{who} {cost}, hand travels {numbers['hand_travel_m']:.3f} m and turns {numbers['hand_turn_deg']:.0f} deg, "
            f"{risen}{near}")


def preview(segments: Sequence[Mapping[str, Any]], *, manifest: Mapping[str, Any], cameras: Sequence[Camera],
            hands: Mapping[str, Mapping[str, Any]], jaws: Any, held: Any,
            body: Mapping[str, Sequence[Sequence[float]]] | None = None, headline: str | None = None,
            refused: str | None = None, picture: bool = True) -> dict[str, Any]:
    """Everything a dry run adds to its record, for one plan's segments.

    ``body`` is each moving arm's chain of joints (``action_plan.body_points``); ``headline`` replaces the picture's
    legend, for a request that has no plan to cost; ``refused`` is the refusal's code, which the picture marks at what
    was asked; ``picture`` False (a run whose latency profile renders no previews) leaves the picture out and says so in
    ``image``. Returns ``arms`` (each arm's numbers,
    ``held_object`` where its jaws hold something (``clearance.held``), and ``clearance``), ``summary`` (one line per arm), ``segments`` (without their poses), ``image`` (what the picture
    shows) and ``png``, the picture itself, where a camera sees it."""
    arms, lines, nearest, cleared, boxes = {}, [], [], [], {}
    for arm in dict.fromkeys(item["arm"] for item in segments):
        numbers = arm_numbers(segments, arm, hands.get(arm))
        path = path_of(segments, arm)
        holding = (clearance.held(manifest, arm=arm, jaws=JsonValue.of(jaws), held=JsonValue.of(held or {})).json()
                   if arm is not None and jaws else None)
        if holding:
            corners = holding.pop("local_corners", None)
            if corners is not None:
                boxes[arm] = corners
            numbers = {**numbers, "held_object": holding}
        seen: Mapping[str, Any] = {"status": "unknown", "reason": "geometry_unavailable"}
        if arm is not None and path and jaws and cameras:
            seen = clearance.along_path(manifest, cameras, arm=arm, poses=path, jaws=JsonValue.of(jaws),
                                        held=JsonValue.of(held or {}), body=(body or {}).get(arm, ())).json()
        if seen.get("status") == "measured":
            nearest.append(seen)
        if seen.get("enters_cleared_space"):
            cleared.append(seen["enters_cleared_space"])
        lines.append(_line(arm, numbers, seen, holding))
        arms[arm] = {**numbers, "clearance": dict(seen)} if arm is not None else numbers
    result: dict[str, Any] = {
        "arms": {arm: value for arm, value in arms.items() if arm is not None},
        "segments": [{key: value for key, value in item.items() if key != "poses"} for item in segments],
        "summary": "; ".join(lines)}
    if None in arms:
        result["base"] = arms[None]
    if not picture:
        result["image"] = dict(PICTURE_OFF)
        return result
    if not cameras:
        result["image"] = {"status": "unavailable", "reason": "no published camera has an image and a calibration"}
        return result
    png, shown = render(cameras, drawn(segments, hands, openings_m(manifest, hands), boxes),
                        legend=[headline] if headline else lines, nearest=nearest, cleared=cleared, refused=refused)
    return {**result, "image": {"status": "attached", **shown}, "png": png}


__all__ = ["CLAIM", "PICTURE_OFF", "Camera", "FINGER_DRAWN_M", "arm_numbers", "cameras_from", "drawn", "jaw_point", "jaws_at",
           "openings_m", "path_of", "preview"]
