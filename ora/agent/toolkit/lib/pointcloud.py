"""Orthographic RGB views of a fused public cloud, with marks drawn at world coordinates. Pure; no files."""
from dataclasses import dataclass
from io import BytesIO
import hashlib
import numpy as np

from .fusion import FusedCloud
from .images import RenderedView, _draw_marks
from .values import JsonValue

#: Each view's name and the two world axes it shows, horizontal then vertical.
VIEW_AXES = {"pointcloud_top": ("x", "y"), "pointcloud_front": ("x", "z"), "pointcloud_side": ("y", "z")}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class CloudView:
    """One view's lossless PNG and the world bounds its pixels span, metres, top-left OpenCV pixels."""
    name: str
    png: bytes
    width: int
    height: int
    horizontal_axis: str
    vertical_axis: str
    horizontal_bounds_m: tuple[float, float]
    vertical_bounds_m: tuple[float, float]


def render(cloud: FusedCloud, workspace_bounds_m: JsonValue, *, resolution: tuple[int, int]) -> dict[str, CloudView]:
    """Pure computation: one orthographic RGB view of the cloud per axis pair, in VIEW_AXES order.

    Example: render(fusion.fuse(sources, bounds), bounds, resolution=(256, 256)). The view spans the
    workspace, +vertical up; later retained voxels overwrite earlier ones. Not a rendering of any camera.
    """
    from PIL import Image

    bounds = workspace_bounds_m.json()
    width, height = resolution
    points, colours = cloud.points.xyz_m.numpy(), cloud.rgb.numpy()
    views = {}
    for name, (horizontal, vertical) in VIEW_AXES.items():
        h_index, v_index = _AXIS_INDEX[horizontal], _AXIS_INDEX[vertical]
        h_low, h_high = [float(value) for value in bounds[horizontal]]
        v_low, v_high = [float(value) for value in bounds[vertical]]
        x = np.rint((points[:, h_index] - h_low) / (h_high - h_low) * (width - 1)).astype(int)
        y = np.rint((v_high - points[:, v_index]) / (v_high - v_low) * (height - 1)).astype(int)
        in_frame = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        canvas[y[in_frame], x[in_frame]] = colours[in_frame]
        stream = BytesIO()
        Image.fromarray(canvas, mode="RGB").save(stream, format="PNG")
        views[name] = CloudView(name, stream.getvalue(), width, height, horizontal, vertical,
                                (h_low, h_high), (v_low, v_high))
    return views


def click_constraint(view: CloudView, x: int, y: int) -> dict[str, float]:
    """Pure computation: the two world coordinates, metres, one pixel of a view shows; the third axis stays unknown.

    Example: click_constraint(views['pointcloud_top'], 12, 40) gives {'x': ..., 'y': ...}. Outside the view is a ValueError.
    """
    if not (0 <= x < view.width and 0 <= y < view.height):
        raise ValueError("pointcloud click is outside the selected view")
    h_low, h_high = view.horizontal_bounds_m
    v_low, v_high = view.vertical_bounds_m
    return {view.horizontal_axis: float(h_low + x / (view.width - 1) * (h_high - h_low)),
            view.vertical_axis: float(v_high - y / (view.height - 1) * (v_high - v_low))}


def annotate(view: CloudView, *, points: JsonValue) -> RenderedView:
    """Pure computation: draw labelled world-metre points on one view, those it frames, at its own pixels.

    Example: annotate(views['pointcloud_top'], points=JsonValue.of([['rim', [0, 0, 1], [0, 180, 255]]])).
    """
    from PIL import Image

    h_low, h_high = view.horizontal_bounds_m
    v_low, v_high = view.vertical_bounds_m
    projected, identity_points = [], []
    for label, xyz, colour in points.json():
        x = int(round((xyz[_AXIS_INDEX[view.horizontal_axis]] - h_low) / (h_high - h_low) * (view.width - 1)))
        y = int(round((v_high - xyz[_AXIS_INDEX[view.vertical_axis]]) / (v_high - v_low) * (view.height - 1)))
        identity_points.append(dict(label=label, xyz_m=xyz, colour=colour))
        if 0 <= x < view.width and 0 <= y < view.height:
            projected.append((x, y, label, colour))
    with Image.open(BytesIO(view.png)) as source:
        image = source.convert("RGB")
    identity = dict(kind="observe", view=view.name, source_sha256=hashlib.sha256(view.png).hexdigest(),
                    points=identity_points)
    return RenderedView(_draw_marks(image, projected), view.width, view.height, JsonValue.of(identity))
