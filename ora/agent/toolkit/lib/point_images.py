"""Pure inspection drawings; callers choose points, palette and candidate allocation."""
from .values import Array
from .perception import MaskCandidate


def point_overview(rgb: Array, *, points_uv: tuple[tuple[int, int], ...],
                   candidates: tuple[MaskCandidate | None, ...],
                   colours: tuple[tuple[int, int, int], ...],
                   unanswered_colour: tuple[int, int, int]) -> Array:
    """Pure computation: tint each point's supplied leading mask, box and labelled cross.

    Example: point_overview(rgb, points_uv=((12, 34),), candidates=(candidate,),
    colours=((48, 220, 96),), unanswered_colour=(168, 168, 168)). Coordinates
    are OpenCV pixels, RGB uint8; no selection or request is performed.
    """
    if len(points_uv) != len(candidates) or not colours:
        raise ValueError("one explicit candidate slot is required per point")
    best = [None if item is None else dict(item.fields.json(), mask=item.mask.numpy())
            for item in candidates]
    import cv2
    import numpy as np

    canvas = rgb.numpy().astype(np.float64)
    for index, item in enumerate(best):
        if item is None:
            continue
        mask = np.asarray(item["mask"], dtype=bool)
        colour = np.asarray(colours[index % len(colours)], dtype=np.float64)
        canvas[mask] = np.clip(.60 * canvas[mask] + .40 * colour, 0, 255)
    rendered = canvas.astype(np.uint8)
    for index, item in enumerate(best):
        if item is None:
            continue
        colour = tuple(int(value) for value in colours[index % len(colours)])
        contours, _hierarchy = cv2.findContours(
            np.asarray(item["mask"], dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rendered, contours, -1, colour, 1)
        x0, y0, x1, y1 = item["bbox_xyxy"]
        cv2.rectangle(rendered, (x0, y0), (x1 - 1, y1 - 1), colour, 1)
    for index, (u, v) in enumerate(points_uv):
        item = best[index]
        colour = (tuple(int(value) for value in colours[index % len(colours)])
                  if item is not None else unanswered_colour)
        label = f"Q{index}:{item['candidate_id']}" if item is not None else f"Q{index}:none"
        cv2.drawMarker(rendered, (u, v), colour, cv2.MARKER_CROSS, 9, 1)
        cv2.putText(rendered, label, (u + 4, max(10, v - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, colour, 1, cv2.LINE_AA)
    return Array.of(rendered)



def point_mask_overlay(rgb: Array, mask: Array, *, point_uv: tuple[int, int],
                       bbox_xyxy: tuple[int, int, int, int]) -> Array:
    """Pure computation: released single-mask tint, boundary, extent and prompt cross.

    Example: point_mask_overlay(rgb, mask, point_uv=(12, 34), bbox_xyxy=(0, 0, 20, 40)).
    Inputs are aligned pixels, output is lossless RGB uint8; no semantic selection.
    """
    import cv2
    import numpy as np

    rgb = rgb.numpy()
    selected = mask.numpy().astype(bool)
    overlay = rgb.copy()
    tint = np.asarray([48, 220, 96], dtype=np.float64)
    overlay[selected] = np.clip(
        0.55 * overlay[selected].astype(np.float64) + 0.45 * tint, 0, 255,
    ).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(
        selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    x0, y0, x1, y1 = bbox_xyxy
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 220, 64), 1)
    cv2.drawMarker(
        overlay, point_uv, (255, 64, 64), cv2.MARKER_CROSS,
        markerSize=max(7, min(rgb.shape[:2]) // 20), thickness=1,
    )
    return Array.of(overlay)

