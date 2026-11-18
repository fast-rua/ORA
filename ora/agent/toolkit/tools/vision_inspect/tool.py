"""``vision.inspect``: one current or saved camera image, or a crop of it, in RGB or a depth colour map."""

import hashlib

from ...lib import images, observation
from ...lib.values import JsonValue
from ..errors import ToolRefusal
from ..observe import _annotated

FIELDS = {"camera", "modality", "crop", "resolution", "step"}
THUMBNAIL = (512, 512)


def _crop(crop, width, height):
    """``[x0, y0, x1, y1]`` inside the image, or the refusal; None is the whole image."""
    if crop is None:
        return None
    if not isinstance(crop, list) or len(crop) != 4 or not all(isinstance(v, int) and not isinstance(v, bool) for v in crop):
        raise ToolRefusal("crop must be integer [x0,y0,x1,y1]")
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ToolRefusal("crop is outside the source image")
    return (x0, y0, x1, y1)


def run(api, arguments):
    """Replay: recorded publication bytes; pure crop and colour map. Pixels are zero-based OpenCV (u, v) of the source."""
    if set(arguments) - FIELDS or "camera" not in arguments:
        raise ToolRefusal("vision.inspect fields are camera, optional modality, crop, resolution and step")
    selection = api.artifacts.tool_context()["vision_inspect"]
    if "error" in selection:
        raise ToolRefusal.from_public(selection["error"])
    obs = observation.observe(api, identity=JsonValue.of(selection["identity"]))
    camera = arguments["camera"]
    if not isinstance(camera, str) or camera not in obs.manifest.json().get("cameras", {}):
        raise ToolRefusal("camera is not policy-visible")
    failure = obs.result.data.json().get("camera_errors", {}).get(camera)
    if failure:
        raise ToolRefusal.from_public(failure)
    refs = obs.camera(camera)
    record = refs.declaration.json()
    modality = arguments.get("modality", "rgb")
    if modality not in {"rgb", "depth"}:
        raise ToolRefusal("modality must be rgb or depth", field="modality", received=modality,
                          constraints={"enum": ["rgb", "depth"]})
    if modality == "rgb":
        source, depth_stats = api.artifacts.read_blob(refs.rgb.json()), None
        width, height = images.dimensions(source)
    else:
        source, depth_stats = images.depth_colours(observation.load_pixels(api, refs).depth_m)
        height, width = source.shape[:2]
    crop = _crop(arguments.get("crop"), width, height)
    resolution = arguments.get("resolution", "thumbnail")
    if resolution not in {"thumbnail", "full"}:
        raise ToolRefusal("resolution must be thumbnail or full")
    view = images.inspected(source, crop=crop, fit=THUMBNAIL if resolution == "thumbnail" else None)
    return {
        "step": selection["step"],
        "source_state_sha256": selection["source_state_sha256"],
        "camera": camera,
        "modality": modality,
        "source_size": [int(record["width"]), int(record["height"])],
        "returned_size": [view.width, view.height],
        "source_roi_xyxy": arguments.get("crop") or [0, 0, int(record["width"]), int(record["height"])],
        "coordinate_convention": "opencv_top_left_uv",
        "source_image_sha256": (refs.rgb if modality == "rgb" else refs.depth_m).json()["sha256"],
        "source_artifacts": {"rgb": record.get("files", {}).get("rgb_lossless_png"),
                             "depth": record.get("files", {}).get("depth_numpy_m_f32"),
                             "calibration": record.get("files", {}).get("calibration")},
        "calibration_sha256": refs.calibration.json()["sha256"],
        "depth_stats": depth_stats,
        "image_bytes": len(view.png),
        "image_sha256": hashlib.sha256(view.png).hexdigest(),
        "image_base64": _annotated.encoded(view.png),
    }
