"""Public depth/robot footprint measurement; explicit inputs, no files or host cache."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math
import numpy as np
from . import _clearance as oc

RECENTER_MARGIN_M = 0.02
HOLD_ENCODER = "normalized_opening_above_empty_close_reference"
HOLD_SEPARATION = "measured_jaw_separation_above_near_closed"
OPENING_ENCODER = "finger_qpos_minus_reference"
OPENING_SIGNED = "signed_finger_qpos"
OPENING_SYMMETRIC = "half_measured_jaw_separation"
OPENING_SYMMETRIC_FROM_REFERENCE = "half_measured_jaw_separation_minus_reference"


@dataclass(frozen=True)
class JawGeometry:
    fingers: tuple[oc.FingerBounds, ...]
    reference_finger_qpos: tuple[float, ...]
    empty_close_reference: float | None
    contact_center_local: tuple[float, float, float]
    wrist_cameras: tuple[str, ...]
    finger_opening: str = OPENING_ENCODER
    finger_signs: tuple[float, ...] = ()
    hold_rule: str = HOLD_ENCODER
    near_closed_m: float | None = None
    root_face: tuple[int, ...] = oc.ROOT_FACE


#: Where a held box comes from: the policy's localization of the target it named, or the gap the closed jaws measure.
LOCALIZED = "localized"
BETWEEN_JAWS = "between_jaws"


@dataclass(frozen=True)
class HeldExtent:
    target_ref: str | None
    local_corners: Any = None
    recentered: bool = False
    source: str = LOCALIZED


def jaws_hold(
    effector: Mapping[str, Any], geometry: JawGeometry, observation: Mapping[str, Any]
) -> bool:
    if effector.get("gripper_command_target") != "close":
        return False
    if geometry.hold_rule == HOLD_SEPARATION:

        separation = measured_jaw_separation_m(effector)
        return separation is not None and separation > float(geometry.near_closed_m)
    feedback = (effector.get("gripper_feedback") or {}).get("arrival_check") or {}
    tolerance = feedback.get(
        "tolerance_normalized",
        (observation.get("controller_feedback") or {}).get(
            "gripper_tolerance_normalized"
        ),
    )
    opening = effector.get("gripper")
    if not isinstance(tolerance, (int, float)) or not isinstance(opening, (int, float)):
        return False
    return float(opening) - geometry.empty_close_reference > float(tolerance)


def _qpos(effector: Mapping[str, Any], geometry: JawGeometry) -> list[float]:
    """Each finger's displacement along its published opening direction from the reference opening."""
    if geometry.finger_opening == OPENING_SYMMETRIC:

        separation = measured_jaw_separation_m(effector)
        if separation is None:
            raise ValueError("measured jaw separation is unavailable")
        return [separation / len(geometry.fingers)] * len(geometry.fingers)
    if geometry.finger_opening == OPENING_SYMMETRIC_FROM_REFERENCE:

        separation = measured_jaw_separation_m(effector)
        if separation is None:
            raise ValueError("measured jaw separation is unavailable")
        return [separation / len(geometry.fingers) - reference for reference in geometry.reference_finger_qpos]
    values = effector.get("gripper_finger_qpos")
    if geometry.finger_opening == OPENING_SIGNED:
        if not isinstance(values, Sequence) or len(values) != len(
            geometry.finger_signs
        ):
            raise ValueError("measured finger positions are unavailable")
        return [
            sign * (float(value) - reference)
            for sign, value, reference in zip(
                geometry.finger_signs,
                values,
                geometry.reference_finger_qpos,
                strict=True,
            )
        ]
    if not isinstance(values, Sequence) or len(values) != len(geometry.fingers):
        raise ValueError("measured finger positions are unavailable")
    return [
        float(value) - reference
        for value, reference in zip(values, geometry.reference_finger_qpos)
    ]


def _measured_robot(
    observation: Mapping[str, Any],
    effectors: Mapping[str, Any],
    geometry: Mapping[str, JawGeometry],
    held: Mapping[str, HeldExtent],
    camera_positions: Mapping[str, Any],
) -> tuple[list[oc.RobotVolume], dict[str, bool]]:
    """Every arm's measured hand as a robot volume, and which of them hold something."""
    robot, holding = [], {}
    for name, effector in sorted(effectors.items()):
        if name not in geometry:
            continue
        jaws = geometry[name]
        holding[name] = jaws_hold(effector, jaws, observation)
        extent = held.get(name) if holding[name] else None
        pose = [float(value) for value in effector["pose_xyz_xyzw"]]
        measured = oc.footprint_at(
            pose,
            finger_qpos=_qpos(effector, jaws),
            fingers=jaws.fingers,
            held_local_corners=None if extent is None else extent.local_corners,
        )
        robot.append(
            oc.robot_volume(
                measured.solids,
                controlled_point=pose[:3],
                extra_points=[
                    camera_positions[camera]
                    for camera in jaws.wrist_cameras
                    if camera in camera_positions
                ],
                root_indices=jaws.root_face,
            )
        )
    return robot, holding


def between_jaws_corners(geometry: JawGeometry, finger_qpos: Sequence[float]) -> Any:
    """The box between two fingers at their measured opening, in the controlled-point frame, or ``None``.

    Along the opening axis it spans the gap between the fingers' inner faces; across it, what the two finger boxes
    share. Closed jaws that stopped apart hold at least this much of something: a measured lower bound of a load."""
    found = _between_jaws(geometry, finger_qpos)
    return None if found is None else oc.box_corners(found[0], found[1])


def _between_jaws(geometry: JawGeometry, finger_qpos: Sequence[float]) -> tuple[Any, Any, int] | None:
    """``between_jaws_corners``' box as its low and high corners and the opening axis."""
    if len(geometry.fingers) != 2:
        return None
    lows, highs, signs = [], [], []
    for finger, opening in zip(geometry.fingers, finger_qpos, strict=True):
        direction = np.asarray(finger.opening_direction, float)
        shift = float(opening) * direction
        lows.append(np.asarray(finger.minimum, float) + shift)
        highs.append(np.asarray(finger.maximum, float) + shift)
        signs.append(direction)
    axis = int(np.argmax(np.abs(signs[0])))
    if signs[0][axis] * signs[1][axis] >= 0:
        return None
    outward, inward = (0, 1) if signs[0][axis] > 0 else (1, 0)
    low, high = np.maximum(lows[0], lows[1]), np.minimum(highs[0], highs[1])
    low[axis], high[axis] = highs[inward][axis], lows[outward][axis]
    if (high <= low).any():
        return None
    return low, high, axis


def measured_held(
    observation: Mapping[str, Any], geometry: Mapping[str, JawGeometry], held: Mapping[str, HeldExtent]
) -> dict[str, HeldExtent]:
    """``held`` with every arm whose jaws hold (``jaws_hold``) but whose load has no measured box given the box between
    its jaws (``between_jaws_corners``): what the jaws measure, whatever closed them."""
    effectors = (observation.get("proprioception") or {}).get("end_effectors") or {}
    result = dict(held)
    for name, jaws in geometry.items():
        effector = effectors.get(name)
        if effector is None or not jaws_hold(effector, jaws, observation):
            continue
        extent = result.get(name)
        if extent is not None and extent.local_corners is not None:
            continue
        try:
            corners = between_jaws_corners(jaws, _qpos(effector, jaws))
        except ValueError:
            corners = None
        if corners is not None:
            result[name] = HeldExtent(None if extent is None else extent.target_ref, corners, False, BETWEEN_JAWS)
    return result


def held_object(
    observation: Mapping[str, Any], *, arm: str, geometry: Mapping[str, JawGeometry], held: Mapping[str, HeldExtent]
) -> dict[str, Any] | None:
    """What one arm's jaws hold, as a preview reports it, or ``None`` when they hold nothing (``jaws_hold``).

    ``extent`` says where its box comes from: ``localized`` (the policy's localization of the target it named when the
    jaws closed), ``between_jaws`` (only the gap the closed jaws measure; what reaches past them is not measured) or
    ``unknown``; ``box_size_m`` is that box's size in the controlled-point frame, ``local_corners`` its corners there
    and ``jaw_gap_m`` the measured gap."""
    effectors = (observation.get("proprioception") or {}).get("end_effectors") or {}
    if arm not in geometry or arm not in effectors or not jaws_hold(effectors[arm], geometry[arm], observation):
        return None
    extent = measured_held(observation, {arm: geometry[arm]}, held).get(arm)
    corners = None if extent is None or extent.local_corners is None else np.asarray(extent.local_corners, float)
    try:
        gap = _between_jaws(geometry[arm], _qpos(effectors[arm], geometry[arm]))
    except ValueError:
        gap = None
    return {
        "target_ref": None if extent is None else extent.target_ref,
        "extent": "unknown" if corners is None else extent.source,
        "box_size_m": None if corners is None else [round(float(v), 4) for v in np.ptp(corners, axis=0)],
        "jaw_gap_m": None if gap is None else round(float(gap[1][gap[2]] - gap[0][gap[2]]), 4),
        "local_corners": None if corners is None else corners.tolist(),
    }


def path_clearance(
    observation: Mapping[str, Any],
    *,
    arm: str,
    poses: Sequence[Sequence[float]],
    geometry: Mapping[str, JawGeometry],
    held: Mapping[str, HeldExtent],
    views: tuple[list[oc.DepthView], dict[str, Any]],
    body: Sequence[Sequence[float]] = (),
) -> dict[str, Any]:
    """The ``clearance`` of one arm's planned path (``_clearance.along_path``); a load the jaws hold with no measured box
    sweeps the box between the jaws (``measured_held``)."""
    effectors = (observation.get("proprioception") or {}).get("end_effectors") or {}
    if arm not in geometry or arm not in effectors:
        return {"status": "unknown", "reason": "geometry_unavailable", "claim_boundary": oc.PATH_CLAIM}
    depth_views, camera_positions = views
    held = measured_held(observation, geometry, held)
    robot, holding = _measured_robot(observation, effectors, geometry, held, camera_positions)
    extent = held.get(arm) if holding.get(arm) else None
    return oc.along_path(
        depth_views,
        poses=poses,
        finger_qpos=_qpos(effectors[arm], geometry[arm]),
        fingers=geometry[arm].fingers,
        held_local_corners=None if extent is None else extent.local_corners,
        robot=robot,
        body=body,
    )


def annotate_observation(
    observation: Mapping[str, Any],
    root: object,
    *,
    arm: str,
    target_pose: Sequence[float],
    geometry: Mapping[str, JawGeometry],
    held: Mapping[str, HeldExtent],
    views: tuple[list[oc.DepthView], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The ``observed_clearance`` value for one arm's commanded target."""
    effectors = (observation.get("proprioception") or {}).get("end_effectors") or {}
    if arm not in geometry or arm not in effectors:
        return _unknown("geometry_unavailable")
    views, camera_positions = views
    robot, holding = _measured_robot(observation, effectors, geometry, held, camera_positions)
    extent = held.get(arm) if holding.get(arm) else None
    footprint = oc.footprint_at(
        [float(value) for value in target_pose],
        finger_qpos=_qpos(effectors[arm], geometry[arm]),
        fingers=geometry[arm].fingers,
        held_local_corners=None if extent is None else extent.local_corners,
    )
    result = oc.measure(
        views,
        observation_id=str(observation.get("observation_id", "")),
        footprint=footprint,
        robot=robot,
    )
    result["held_object"] = (
        None
        if not holding.get(arm)
        else {
            "target_ref": None if extent is None else extent.target_ref,
            "extent_included": extent is not None and extent.local_corners is not None,
            "recentered_under_jaws": bool(extent is not None and extent.recentered),
        }
    )
    return result


def _unknown(reason: str) -> dict[str, Any]:
    return {
        "status": "unknown",
        "reason": reason,
        "radius_m": oc.RADIUS_M,
        "claim_boundary": oc.CLAIM_BOUNDARY,
    }


def held_local_corners(
    localization: Mapping[str, Any],
    attach_pose: Sequence[float],
    contact_center_local: Sequence[float],
) -> tuple[Any, bool] | None:
    """A localization's world box fixed to link6 at the close pose, or None when unmeasured."""
    low, high = localization.get("aabb_min_xyz_m"), localization.get("aabb_max_xyz_m")
    if (
        localization.get("status", "localized") != "localized"
        or low is None
        or high is None
    ):
        return None
    low, high = np.asarray(low, float), np.asarray(high, float)
    support = localization.get("support_z_m")
    if isinstance(support, (int, float)) and math.isfinite(float(support)):
        low[2] = min(low[2], float(support))
    turn, origin = oc.rotation(attach_pose[3:]), np.asarray(attach_pose[:3], float)
    jaw = origin + turn @ np.asarray(contact_center_local, float)
    centre, half = (low + high) / 2.0, (high - low) / 2.0
    recentered = bool(
        (np.abs(jaw[:2] - centre[:2]) > half[:2] + RECENTER_MARGIN_M).any()
    )
    if recentered:
        low[:2], high[:2] = low[:2] + (jaw[:2] - centre[:2]), high[:2] + (
            jaw[:2] - centre[:2]
        )
    return (oc.box_corners(low, high) - origin) @ turn, recentered


def measured_jaw_separation_m(record: Any) -> float | None:
    """The measured jaw separation of one end effector, or ``None``.

    Two backends publish the same physical quantity in different shapes.
    RoboTwin publishes ``gripper_opening_m`` directly. LIBERO publishes
    ``gripper_finger_qpos``: two prismatic finger displacements whose summed
    magnitude is the separation, which is the conversion
    ``agent/context/observation/public_facts.py`` already applies when it reports
    ``measured_aperture_mm`` to the policy. Reading only the first key left the
    absorbed verify stage reporting ``unavailable`` on every LIBERO episode
    while the measurement sat one field away in the same record.

    The normalized ``gripper`` value in that record is a command, not a length,
    and is never a substitute for either.
    """

    if not isinstance(record, Mapping):
        return None
    published = record.get("gripper_opening_m")
    if type(published) in (int, float) and math.isfinite(float(published)):
        return float(published)
    if record.get("pose_frame") == "robodojo_episode_local_ee_link":
        return None  # ARX q=0 has a source mesh gap; encoder sum is not aperture.
    fingers = record.get("gripper_finger_qpos")
    if not isinstance(fingers, list) or len(fingers) != 2:
        return None
    if not all(type(q) in (int, float) and math.isfinite(float(q)) for q in fingers):
        return None
    return sum(abs(float(q)) for q in fingers)
