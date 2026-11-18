"""Pure RGB-D measurements and lossless inspection images, with immutable values.

These functions accept public pixels and calibration, never paths or service
clients. Candidate selection, publication and task preferences belong to tools.
"""
from dataclasses import dataclass
from io import BytesIO
import numpy as np

from .values import Array, JsonValue
from .perception import MaskCandidate
from . import _vision_measurement, _vision_card, _vision_palette


@dataclass(frozen=True)
class InspectionImage:
    """RGB uint8 HxWx3 inspection card and complete rendering metadata; pixels, not new sensing."""
    rgb: Array
    metadata: JsonValue


def mask_summary(mask: Array, depth_m: Array) -> JsonValue:
    """Pure computation: pixel extent and valid-depth statistics (metres); empty masks fail.

    Example: mask_summary(candidate.mask, camera.depth_m). Preserves the existing
    valid-depth window (1e-6, 2.5] m and reports all measured counts/fractions.
    """
    return JsonValue.of(_vision_measurement._mask_summary(mask.numpy(), depth_m.numpy()))


def localize_mask(mask: Array, depth_m: Array, calibration: JsonValue, *, extended_frame: bool = True) -> JsonValue:
    """Pure computation: full visible-mask PCA, support height and footprint in world metres.

    Example: localize_mask(candidate.mask, camera.depth_m, camera.calibration).
    OpenCV extrinsic is world→camera. Insufficient depth returns explicit nulls
    and counts; malformed calibration fails. No completed object shape or truth.
    This version preserves the existing primary-component/MAD measurement recipe.
    """
    return JsonValue.of(_vision_measurement._mask_depth_localization(
        mask=mask.numpy(), depth_m=depth_m.numpy(), calibration=calibration.json(), extended_frame=extended_frame))


def mask_overview(rgb: Array, candidates: tuple[MaskCandidate, ...]) -> Array:
    """Pure computation: tint all supplied masks and draw their pixel boxes/rank labels.

    Example: mask_overview(camera.rgb, segmentation.candidates). Input order
    selects palette colours, never reorders/filter candidates. Returns RGB uint8.
    """
    import cv2
    source = rgb.numpy()
    overlay = source.astype(np.float64)
    colours = tuple(np.asarray(_vision_palette.grounding_candidate_colour(i), dtype=np.float64)
                    for i in range(len(candidates)))
    for candidate, colour in zip(candidates, colours, strict=True):
        selected = candidate.mask.numpy().astype(bool)
        overlay[selected] = np.clip(.60 * overlay[selected] + .40 * colour, 0, 255)
    rendered = overlay.astype(np.uint8)
    for candidate, colour in zip(candidates, colours, strict=True):
        selected = candidate.mask.numpy().astype(bool)
        fields = candidate.fields.json()
        display_colour = tuple(int(item) for item in colour)
        contours, _ = cv2.findContours(selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rendered, contours, -1, display_colour, 1)
        x0, y0, x1, y1 = fields['bbox_xyxy']
        cv2.rectangle(rendered, (x0, y0), (x1 - 1, y1 - 1), display_colour, 2)
        cv2.putText(rendered, f"T{fields['rank']}", (x0 + 2, max(12, y0 + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, display_colour, 1, cv2.LINE_AA)
    return Array.of(rendered)


def candidate_inspection(rgb: Array, candidate: MaskCandidate, *, text: str,
                         depth_m: Array | None = None, calibration: JsonValue | None = None,
                         localization: JsonValue | None = None, support_line: bool = False,
                         points_uv: tuple[tuple[int, int], ...] = ()) -> InspectionImage:
    """Pure computation: source/mask crop card, optionally with metric depth-profile plots.

    Example: candidate_inspection(camera.rgb, candidate, text='cup', depth_m=camera.depth_m,
    calibration=camera.calibration, localization=measured, support_line=True).
    Requires caller-supplied target_ref/rank/score/bbox fields; does not choose a
    target. All depth-profile inputs must be supplied together. Image enlargement
    adds no source detail; an unmeasurable profile is marked unavailable.
    """
    provided = (depth_m is not None, calibration is not None, localization is not None)
    if any(provided) and not all(provided):
        raise ValueError('depth profile requires depth, calibration and localization')
    profile = None if not all(provided) else dict(
        options=dict(renderer=_vision_card.PROFILE_RENDERER, support_line=support_line),
        depth_m=depth_m.numpy(), calibration=calibration.json(), localization=localization.json())
    image, metadata = _vision_card.render_candidate_preview(
        rgb_u8=rgb.numpy(), candidate=dict(candidate.fields.json(), mask=candidate.mask.numpy()),
        text=text, points_uv=points_uv, depth_profile=profile)
    return InspectionImage(Array.of(image), JsonValue.of(metadata))


def png_bytes(rgb: Array) -> bytes:
    """Pure computation: lossless RGB PNG bytes, preserving the existing PIL encoding.

    Example: png_bytes(mask_overview(camera.rgb, candidates)). No file writes.
    """
    from PIL import Image
    stream = BytesIO()
    Image.fromarray(rgb.numpy(), mode='RGB').save(stream, format='PNG')
    return stream.getvalue()


def npy_bytes(value: Array) -> bytes:
    """Pure computation: pickle-free NPY encoding with dtype/shape/order preserved.

    Example: npy_bytes(candidate.mask). No file writes or simulator access.
    """
    stream = BytesIO()
    np.save(stream, value.numpy(), allow_pickle=False)
    return stream.getvalue()


def measured_depth_selection(mask: Array, depth_m: Array) -> tuple[Array, JsonValue]:
    """Pure computation: released component/MAD pixel gate and complete measured fractions.

    Example: selected, quality = measured_depth_selection(mask, camera.depth_m).
    Pixels are OpenCV row-major; optical depth is metres. No quality acceptance
    policy: the caller compares reported fractions to its own thresholds.
    """
    selected, quality = _vision_measurement._mask_depth_inlier_selection(
        mask=mask.numpy().astype(bool), depth_m=depth_m.numpy().astype(np.float64))
    return Array.of(selected), JsonValue.of(quality)


def npz_bytes(arrays: tuple[tuple[str, Array], ...]) -> bytes:
    """Pure computation: compressed numeric arrays with explicit unique archive names.

    Example: npz_bytes((("mask", candidate.mask),)). Preserves dtype/shape;
    emits only in-memory bytes, never a caller-selected file or pickle payload.
    """
    names = [name for name, value in arrays]
    if len(set(names)) != len(names) or any(not name or not name.replace("_", "").isalnum() for name in names):
        raise ValueError("array archive names must be unique alphanumeric identifiers")
    stream = BytesIO()
    np.savez_compressed(stream, **{name: value.numpy() for name, value in arrays})
    return stream.getvalue()
