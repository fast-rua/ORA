"""One enlargement card per grounding or point-segmentation candidate.

The card was two 480 px panels -- the exact source crop and the same crop with
the candidate mask tinted -- and so it showed only what the camera's 2-D pixels
show. A candidate's shape in depth was nowhere in an image: a recorded
put_bottles_into_dustbin run ranked a floor-standing dustbin first in
``vision_ground`` (centre (-0.81, 0.07, 0.59), visible z 0.39-0.73) and the
policy still did not recognise it, because a tall open container seen from
above is a dark ring in a crop.

Under :data:`CAPABILITY` the card grows from 1024x576 to 1024x800 and a strip
below the crops draws the candidate's own depth points, the inlier set the
localizer measures (``mask_localization._mask_depth_inlier_selection``), in the
candidate's colour, over grey context points from every depth-valid pixel of
the same crop, in two metric views at one scale:

* elevation -- horizontal is the camera-right axis projected onto the floor
  plane, as an offset from the candidate centre; vertical is world z;
* top -- world x horizontal, world y vertical (up is +y, as in the grasp
  review's TOP panel).

Both windows are centred on the candidate's localized centre. The vertical
span is ``max(0.10 m, 2.5 x largest extent)``; the horizontal span follows
from the plot's aspect at the same metres-per-pixel. A dashed line marks the
localizer's ``support_z_m`` in the elevation view when the surface also
declares ``target_support_height_v1`` (so the line never shows a number the
surface's own results withhold) and the estimate exists. Every value drawn is
the public RGB-D measurement already behind the candidate's localization;
nothing is task-specific or simulator-derived.

The crops keep their size, position and resampling, so the published
``resampling`` string and the policy-visible preview record keep their keys;
only ``render_dimensions`` changes value. Without the capability the card is
byte-identical to the released renderer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import mask_localization
from .grounding_palette import grounding_candidate_colour
from ora.agent.toolkit.lib._vision_card import (
    CAPABILITY, CARD_WIDTH, EXTENT_SPAN_FACTOR, GROUNDING_VISUALIZATION_ID, LEGACY_CARD_HEIGHT, MINIMUM_SPAN_M,
    PROFILE_CARD_HEIGHT, PROFILE_RENDERER, SUPPORT_CAPABILITY, VALID_DEPTH_M, _DATA_HEIGHT, _DATA_WIDTH, _PANEL_LEFTS,
    _draw_view, _strip_unavailable, _world_points, crop_box,
)


def options(surface: Any) -> dict[str, Any] | None:
    """What a resolved surface asks the card to draw, or ``None`` for the released card."""

    if surface is None or not surface.supports(CAPABILITY):
        return None
    return {"renderer": PROFILE_RENDERER,
            "support_line": bool(surface.supports(SUPPORT_CAPABILITY))}


def identity(profile_options: Mapping[str, Any] | None) -> dict[str, Any]:
    """The query-identity extension; empty without the capability so released refs keep their hashes."""

    return {} if profile_options is None else {"depth_profile": dict(profile_options)}


def profile_input(
    profile_options: Mapping[str, Any] | None, *, depth_m: Any,
    calibration_path: Path, localization: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Bundle what one card's strip reads; ``None`` keeps the released card."""

    if profile_options is None:
        return None
    return {"options": dict(profile_options), "depth_m": depth_m,
            "calibration_path": Path(calibration_path),
            "localization": dict(localization or {})}


def render_candidate_preview(
    *, rgb_u8: Any, candidate: Mapping[str, Any], text: str,
    points_uv: Sequence[Sequence[int]] = (),
    depth_profile: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Render raw and mask-highlighted enlargements for one semantic candidate."""

    import cv2
    import numpy as np
    from PIL import Image

    rgb = np.asarray(rgb_u8, dtype=np.uint8)
    height, width = rgb.shape[:2]
    crop_xyxy = crop_box(
        bbox_xyxy=[int(item) for item in candidate["bbox_xyxy"]],
        image_width=width,
        image_height=height,
    )
    x0, y0, x1, y1 = crop_xyxy
    raw_crop = rgb[y0:y1, x0:x1]
    mask_crop = np.asarray(candidate["mask"], dtype=bool)[y0:y1, x0:x1]
    if raw_crop.size == 0 or mask_crop.shape != raw_crop.shape[:2]:
        raise ValueError("grounding candidate crop is empty or unaligned")

    rank = int(candidate["rank"])
    colour = np.asarray(grounding_candidate_colour(rank), dtype=np.float64)
    selected = raw_crop.copy()
    selected[mask_crop] = np.clip(
        0.62 * selected[mask_crop].astype(np.float64) + 0.38 * colour,
        0, 255,
    ).astype(np.uint8)

    panel_size = 480
    raw_panel = cv2.resize(
        raw_crop, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC,
    )
    selected_panel = cv2.resize(
        selected, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC,
    )
    card_height = LEGACY_CARD_HEIGHT if depth_profile is None else PROFILE_CARD_HEIGHT
    canvas = np.full((card_height, CARD_WIDTH, 3), 20, dtype=np.uint8)
    canvas[80:560, 16:496] = raw_panel
    canvas[80:560, 528:1008] = selected_panel
    for index, (u, v) in enumerate(points_uv):
        if x0 <= u < x1 and y0 <= v < y1:
            px = min(panel_size - 1, int((u - x0) * panel_size / (x1 - x0)))
            py = min(panel_size - 1, int((v - y0) * panel_size / (y1 - y0)))
            for left in (16, 528):
                location = (left + px, 80 + py)
                cv2.drawMarker(canvas, location, (255, 64, 128), cv2.MARKER_CROSS, 13, 2)
                cv2.putText(canvas, f"Q{index}", (location[0] + 5, location[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 64, 128), 1, cv2.LINE_AA)
    display_colour = tuple(int(item) for item in colour)
    title = (
        f"T{rank}  score={float(candidate['backend_score']):.4f}  "
        f"query={text[:72]}"
    )
    cv2.putText(
        canvas, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        display_colour, 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "SOURCE CROP + QUERY POINTS" if points_uv else "RAW SOURCE CROP", (26, 108), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "MASK HIGHLIGHT", (538, 108), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (16, 80), (495, 559), display_colour, 2)
    cv2.rectangle(canvas, (528, 80), (1007, 559), display_colour, 2)
    metadata = {
        "renderer": GROUNDING_VISUALIZATION_ID,
        "rank": rank,
        "target_ref": candidate["target_ref"],
        "backend_score": float(candidate["backend_score"]),
        "source_dimensions": [width, height],
        "source_bbox_xyxy": [int(item) for item in candidate["bbox_xyxy"]],
        "crop_xyxy": crop_xyxy,
        "render_dimensions": [CARD_WIDTH, card_height],
        "resampling": "bicubic_visual_enlargement_no_new_source_detail",
        "claim_boundary": (
            "policy-visible enlargement of exact public RGB pixels and mask; "
            "not higher-resolution sensing or semantic ground truth"
        ),
    }
    if depth_profile is not None:
        metadata["depth_profile"] = _draw_depth_profile(
            canvas, mask=np.asarray(candidate["mask"], dtype=bool),
            crop_xyxy=crop_xyxy, colour=display_colour, profile=depth_profile,
        )
    return Image.fromarray(canvas, mode="RGB"), metadata


def _calibration(path: Path) -> tuple[Any, Any]:
    import numpy as np
    from ora.agent.toolkit.lib import _rigid

    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
    if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise ValueError("depth profile calibration shape mismatch")
    return intrinsic, _rigid.invert(extrinsic)


def _draw_depth_profile(
    canvas: Any, *, mask: Any, crop_xyxy: list[int], colour: tuple[int, int, int],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Draw the strip under the crops; return the private record of what it shows.

    The strip is evidence about the card, never a reason to lose the card: an
    unusable input leaves a one-line ``unavailable`` strip and its reason.
    """

    import numpy as np

    record: dict[str, Any] = {"renderer": PROFILE_RENDERER, "options": dict(profile["options"])}
    localization = profile["localization"]
    depth = np.asarray(profile["depth_m"], dtype=np.float64)
    if (localization.get("status") != "localized" or localization.get("center_xyz_m") is None
            or depth.shape != mask.shape):
        return _strip_unavailable(canvas, record, "candidate_not_localized")
    try:
        geometry = _profile_geometry(mask=mask, depth=depth, crop_xyxy=crop_xyxy,
                                     profile=profile)
    except (ValueError, KeyError, OSError, np.linalg.LinAlgError):
        return _strip_unavailable(canvas, record, "calibration_unusable")
    views = (
        ("ELEVATION", "camera-right offset / world z (m)",
         lambda points: (points - geometry["center"]) @ geometry["right"],
         lambda points: points[:, 2], 0.0, float(geometry["center"][2]), True),
        ("TOP", "world x / world y (m)", lambda points: points[:, 0],
         lambda points: points[:, 1], float(geometry["center"][0]),
         float(geometry["center"][1]), False),
    )
    view_records = [
        _draw_view(canvas, left=left, spec=spec, geometry=geometry, colour=colour)
        for left, spec in zip(_PANEL_LEFTS, views, strict=True)
    ]
    return {**record, "status": "rendered",
            "window_center_xyz_m": [float(value) for value in geometry["center"]],
            "vertical_span_m": geometry["vertical_span"],
            "horizontal_span_m": geometry["horizontal_span"],
            "metres_per_pixel": geometry["metres_per_pixel"],
            "elevation_horizontal_axis_world": [float(value) for value in geometry["right"]],
            "candidate_depth_points": int(len(geometry["target"])),
            "context_depth_points": int(len(geometry["context"])),
            "support_z_m": geometry["support"],
            "views": view_records}


def _profile_geometry(*, mask: Any, depth: Any, crop_xyxy: list[int],
                      profile: Mapping[str, Any]) -> dict[str, Any]:
    """World points, window and axes the two views share."""

    import numpy as np

    localization = profile["localization"]
    intrinsic, camera_to_world = _calibration(profile["calibration_path"])
    inliers, _selection = mask_localization._mask_depth_inlier_selection(mask=mask, depth_m=depth)
    rows, columns = np.nonzero(inliers)
    x0, y0, x1, y1 = crop_xyxy
    window = depth[y0:y1, x0:x1]
    valid = np.isfinite(window) & (window > VALID_DEPTH_M[0]) & (window <= VALID_DEPTH_M[1])
    context_rows, context_columns = np.nonzero(valid)
    extent = localization.get("extent_xyz_m") or [0.0, 0.0, 0.0]
    largest = max([float(localization.get("largest_extent_m") or 0.0),
                   *[float(value) for value in extent]])
    vertical_span = max(MINIMUM_SPAN_M, EXTENT_SPAN_FACTOR * largest)
    right = camera_to_world[:3, 0].copy()
    right[2] = 0.0
    if float(np.linalg.norm(right)) < 1e-3:
        right = np.asarray([1.0, 0.0, 0.0])
    support = localization.get("support_z_m") if profile["options"].get("support_line") else None
    return {
        "target": _world_points(depth, rows, columns, intrinsic, camera_to_world),
        "context": _world_points(depth, context_rows + y0, context_columns + x0, intrinsic,
                                 camera_to_world),
        "center": np.asarray(localization["center_xyz_m"], dtype=np.float64),
        "right": right / np.linalg.norm(right),
        "vertical_span": float(vertical_span),
        "horizontal_span": float(vertical_span / _DATA_HEIGHT * _DATA_WIDTH),
        "metres_per_pixel": float(vertical_span / _DATA_HEIGHT),
        "support": None if support is None else float(support),
    }
