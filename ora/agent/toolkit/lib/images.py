"""Pure calibrated annotation of supplied PNG bytes; no image paths or hidden state."""

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import numpy as np
from .values import Array, JsonValue


@dataclass(frozen=True)
class RenderedView:
    """Lossless RGB PNG, output pixel dimensions, and world-metre annotation identity."""

    png: bytes
    width: int
    height: int
    identity: JsonValue


def dimensions(png: bytes) -> tuple[int, int]:
    """Pure computation: width/height in pixels. Example: dimensions(recorded_png)."""
    from PIL import Image

    with Image.open(BytesIO(png)) as image:
        return image.size


def inspected(source, *, crop: tuple[int, int, int, int] | None = None, fit: tuple[int, int] | None = None) -> RenderedView:
    """Pure computation: one image as RGB PNG, cropped to source pixels ``[x0, y0, x1, y1]``, then shrunk to fit ``fit``.

    ``source`` is PNG bytes or an RGB uint8 array. Example: inspected(png, fit=(512, 512)). Without ``fit`` the crop keeps
    its pixels; an image already inside ``fit`` is not enlarged. The identity is empty: nothing is drawn.
    """
    from PIL import Image

    if isinstance(source, bytes):
        with Image.open(BytesIO(source)) as opened:
            image = opened.convert("RGB")
    else:
        image = Image.fromarray(source.numpy(), mode="RGB")
    if crop is not None:
        image = image.crop(crop)
    if fit is not None:
        image.thumbnail(fit)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return RenderedView(stream.getvalue(), image.width, image.height, JsonValue.of({}))


def depth_colours(depth: Array) -> tuple[Array, dict]:
    """Pure computation: metric depth as an RGB uint8 array, and the statistics of its valid pixels.

    Colours span the 2nd to 98th percentile of the valid depths, in metres; a pixel with no depth is the low colour. Not a
    measurement: read depth from the array. Example: colours, stats = depth_colours(camera.depth_m).
    """
    values = depth.numpy().astype(np.float32)
    valid = values[np.isfinite(values) & (values > 1e-6)]
    if not valid.size:
        return Array.of(np.zeros((*values.shape, 3), dtype=np.uint8)), dict(valid_count=0, valid_fraction=0.0)
    low, high = np.percentile(valid, [2, 98])
    high = max(float(high), float(low) + 1e-6)
    normalized = np.clip((values - low) / (high - low), 0, 1)
    normalized[~np.isfinite(values) | (values <= 1e-6)] = 0
    colour = np.stack([normalized, 1.0 - np.abs(2.0 * normalized - 1.0), 1.0 - normalized], axis=-1)
    return Array.of((colour * 255).astype(np.uint8)), dict(
        valid_count=int(valid.size), valid_fraction=float(valid.size / values.size), min_m=float(np.min(valid)),
        median_m=float(np.median(valid)), max_m=float(np.max(valid)), visualization_percentile_m=[float(low), float(high)])


def _draw_marks(image, projected) -> bytes:
    """Draw (x, y, label, colour) pixel marks on a PIL image; return its PNG bytes."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for x, y, label, colour in projected:
        colour = tuple(colour)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=colour, width=1)
        draw.point((x, y), fill=colour)
        if label:
            draw.text((x + 4, y + 2), label, fill=colour)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def annotate(
    png: bytes,
    calibration: bytes,
    *,
    camera: str,
    points: JsonValue,
    resolution: tuple[int, int] | None = None,
) -> RenderedView:
    """Pure computation: project labelled world-metre points and draw at output pixel scale.

    Replay: recompute from supplied recorded bytes. Example: annotate(png, calibration,
    camera='head', points=JsonValue.of([['L', [0, 0, 1], [255, 0, 255]]])).
    OpenCV world-to-camera extrinsic, +x right/+y down/+z forward; no depth query.
    """
    from PIL import Image

    with Image.open(BytesIO(png)) as source:
        image = source.convert("RGB")
    width, height = image.size
    output_width, output_height = resolution or image.size
    spec = json.loads(calibration)
    intrinsic = np.asarray(spec["intrinsic_cv"], dtype=float)
    extrinsic = np.asarray(spec["extrinsic_cv"], dtype=float)
    projected, identity_points = [], []
    for label, xyz, colour in points.json():
        camera_point = extrinsic @ np.asarray([*xyz, 1.0], dtype=float)
        if camera_point[2] <= 1e-6:
            continue
        pixel = intrinsic @ camera_point[:3]
        x, y = int(round(float(pixel[0] / pixel[2]))), int(
            round(float(pixel[1] / pixel[2]))
        )
        identity_points.append(dict(label=label, xyz_m=xyz, colour=colour))
        if 0 <= x < width and 0 <= y < height:
            projected.append(
                (
                    int(round(x * (output_width - 1) / max(width - 1, 1))),
                    int(round(y * (output_height - 1) / max(height - 1, 1))),
                    label,
                    colour,
                )
            )
    if image.size != (output_width, output_height):
        image = image.resize((output_width, output_height), Image.Resampling.BILINEAR)
    encoded = _draw_marks(image, projected)
    identity = dict(
        kind="observe",
        camera=camera,
        source_sha256=hashlib.sha256(png).hexdigest(),
        calibration_sha256=hashlib.sha256(calibration).hexdigest(),
        source_resolution=[width, height],
        output_resolution=[output_width, output_height],
        points=identity_points,
    )
    return RenderedView(encoded, output_width, output_height, JsonValue.of(identity))


def contact_sheet(panels: tuple[tuple[str, bytes], ...], *, points: JsonValue,
                  resolution: tuple[int, int]) -> bytes:
    """Pure computation: bounded labelled PNG panels with a point-colour legend.

    Inputs are in-memory PNGs and explicit labels/world-metre points/RGB colours.
    Example: contact_sheet((("head", png),), points=labels, resolution=(256, 256)).
    """
    import math
    from PIL import Image, ImageDraw

    labelled_points = points.json()
    if not panels or not labelled_points or any(type(value) is not int or value <= 0 for value in resolution):
        raise ValueError("contact sheet requires panels, labels and positive pixel dimensions")
    width, height = resolution
    count = len(panels)
    columns = 1 if count == 1 else math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    cell_width = width // columns
    cell_height = height // rows
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for index, (view, path) in enumerate(panels):
        panel = Image.open(BytesIO(path)).convert("RGB")
        panel.thumbnail(
            (max(1, cell_width), max(1, cell_height)),
            Image.Resampling.LANCZOS,
        )
        column = index % columns
        row = index // columns
        left = column * cell_width + (cell_width - panel.width) // 2
        top = row * cell_height + (cell_height - panel.height) // 2
        canvas.paste(panel, (left, top))
        draw.rectangle(
            (column * cell_width, row * cell_height,
             column * cell_width + min(cell_width, 8 + 7 * len(view)),
             row * cell_height + 14),
            fill=(0, 0, 0),
        )
        draw.text(
            (column * cell_width + 3, row * cell_height + 2),
            view,
            fill=(255, 255, 255),
        )

    legend_columns = min(4, len(labelled_points))
    legend_rows = math.ceil(len(labelled_points) / legend_columns)
    legend_height = 16 * legend_rows
    legend_top = height - legend_height
    legend_width = width // legend_columns
    draw.rectangle((0, legend_top, width, height), fill=(0, 0, 0))
    for index, (point_id, _xyz, colour) in enumerate(labelled_points):
        colour = tuple(colour)
        column = index % legend_columns
        row = index // legend_columns
        left = column * legend_width
        top = legend_top + row * 16
        draw.rectangle((left + 3, top + 4, left + 10, top + 11), fill=colour)
        draw.text((left + 14, top + 2), point_id, fill=colour)

    stream = BytesIO()
    canvas.save(stream, format="PNG")
    return stream.getvalue()
