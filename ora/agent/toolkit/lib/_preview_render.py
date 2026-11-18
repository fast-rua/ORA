"""The one drawing of a planned action: its segments projected on the public cameras that see them.

Its inputs are the path over time, each camera's published calibration and latest image. It names no task, tool or
backend; a camera is whatever the observation manifest lists, chosen by how much of the path it sees, so it draws the
same on a simulator and on a real robot. It only projects and draws.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from ._grasp_overlay import _overlay_fonts, _project_world

#: At most this many panels, each at most this big; the legend strip adds this much height.
MAX_PANELS = 2
PANEL_WIDTH, PANEL_HEIGHT = 640, 480
LEGEND_LINE = 18
#: The legend's small font is about this wide per character; a line wraps to the picture's width.
LEGEND_CHAR_PX = 7
#: One colour per arm, in the order the plan names them; what is not planned is drawn dashed.
ARM_COLOURS = ((255, 176, 64), (64, 200, 255), (160, 255, 120))
AFTER_CONTACT = (190, 190, 190)
NEAREST = (255, 64, 64)
CLEARED = (255, 235, 60)
REFUSED = (255, 40, 40)
#: Every mark has a dark edge so it reads on any image; a path is this wide, the edge this much wider.
HALO = (0, 0, 0)
PATH_WIDTH, HALO_EXTRA = 4, 3
#: Time marks along a planned path: this many, evenly in native actions.
TIME_MARKS = 3


class Camera(NamedTuple):
    """One published camera: RGB uint8 HxWx3, depth metres HxW or None, OpenCV K, 4x4 world-to-camera."""

    name: str
    rgb: Any
    depth_m: Any
    intrinsic: Any
    extrinsic: Any


def _in_frame(pixel: tuple[int, int] | None, camera: Camera) -> bool:
    height, width = np.asarray(camera.rgb).shape[:2]
    return pixel is not None and 0 <= pixel[0] < width and 0 <= pixel[1] < height


def seen_cameras(cameras: Sequence[Camera], paths: Sequence[Sequence[Sequence[float]]]) -> list[Camera]:
    """The cameras that see most of ``paths``, in manifest order among equals; at most ``MAX_PANELS``.

    A path in no camera's view still gets the first camera: the model reads that as out of view."""
    points = [point for path in paths for point in path]
    scored = [(-sum(_in_frame(_project_world(point, intrinsic=camera.intrinsic, extrinsic=camera.extrinsic), camera)
                    for point in points), index) for index, camera in enumerate(cameras)]
    chosen = [cameras[index] for score, index in sorted(scored) if score < 0][:MAX_PANELS]
    return chosen or list(cameras[:1])


def _dashed(draw: Any, start: tuple[int, int], end: tuple[int, int], colour: tuple, width: int) -> None:
    length = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    pieces = max(1, length // 8)
    for index in range(0, pieces, 2):
        first = (start[0] + (end[0] - start[0]) * index / pieces, start[1] + (end[1] - start[1]) * index / pieces)
        second = (start[0] + (end[0] - start[0]) * (index + 1) / pieces,
                  start[1] + (end[1] - start[1]) * (index + 1) / pieces)
        draw.line((first, second), fill=colour, width=width)


def _path(draw: Any, pixels: Sequence[tuple[int, int] | None], colour: tuple, *, dashed: bool, width: int) -> None:
    """A polyline with a dark edge under it, dashed where it is not planned."""
    for fill, size in ((HALO, width + HALO_EXTRA), (colour, width)):
        for first, second in zip(pixels, pixels[1:]):
            if first is None or second is None:
                continue
            if dashed:
                _dashed(draw, first, second, fill, size)
            else:
                draw.line((first, second), fill=fill, width=size)


def _label(draw: Any, at: tuple[int, int], text: str, colour: tuple, font: Any,
           point: tuple[int, int] | None = None) -> None:
    """Text on a dark box, so it reads on any image; moved inside the image where it would leave it. The label of a
    ``point`` outside the image is not drawn: moved in, it would name the wrong place."""
    width, height = draw.im.size
    if not text or (point is not None and not (0 <= point[0] < width and 0 <= point[1] < height)):
        return
    box = draw.textbbox(at, text, font=font)
    shift_x = min(0, width - 2 - box[2]) or max(0, 2 - box[0])
    shift_y = min(0, height - 2 - box[3]) or max(0, 2 - box[1])
    if shift_x or shift_y:
        at = (at[0] + shift_x, at[1] + shift_y)
        box = draw.textbbox(at, text, font=font)
    draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=HALO)
    draw.text(at, text, fill=colour, font=font)


def _mark(draw: Any, pixel: tuple[int, int] | None, colour: tuple, *, filled: bool, label: str, font: Any) -> None:
    if pixel is None:
        return
    x, y = pixel
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=HALO, width=5)
    draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=colour, width=3, fill=colour if filled else None)
    _label(draw, (x + 10, y - 8), label, colour, font, pixel)


#: A jaw sketch spanning more than this share of a panel is a camera on the hand itself, and is left out.
JAWS_MAX_SPAN = 0.3


def _jaws(draw: Any, pixels: Sequence[tuple[int, int] | None] | None, colour: tuple, width: int,
          size: tuple[int, int]) -> None:
    """The jaws sketched as a U: one finger from base to tip, across the opening, the other from tip to base."""
    if not pixels or any(pixel is None for pixel in pixels):
        return
    xs, ys = [pixel[0] for pixel in pixels], [pixel[1] for pixel in pixels]
    if max(xs) - min(xs) > JAWS_MAX_SPAN * size[0] or max(ys) - min(ys) > JAWS_MAX_SPAN * size[1]:
        return
    _path(draw, list(pixels), colour, dashed=False, width=width)


#: The twelve edges of a box given as ``_clearance.box_corners`` orders its eight corners (index 4x + 2y + z).
BOX_EDGES = tuple((first, first | bit) for first in range(8) for bit in (1, 2, 4) if not first & bit)


def _box(draw: Any, pixels: Sequence[tuple[int, int] | None] | None, colour: tuple, width: int) -> None:
    """A held box's twelve edges, with a dark edge under them; not drawn where a corner leaves the camera."""
    if not pixels or any(pixel is None for pixel in pixels):
        return
    for fill, size in ((HALO, width + HALO_EXTRA), (colour, width)):
        for first, second in BOX_EDGES:
            draw.line((pixels[first], pixels[second]), fill=fill, width=size)


def _cross(draw: Any, pixel: tuple[int, int] | None, colour: tuple, size: int) -> None:
    if pixel is None:
        return
    x, y = pixel
    for fill, width in ((HALO, 7), (colour, 4)):
        draw.line((x - size, y - size, x + size, y + size), fill=fill, width=width)
        draw.line((x - size, y + size, x + size, y - size), fill=fill, width=width)


def _panel(camera: Camera, segments: Sequence[Mapping[str, Any]], colours: Mapping[Any, tuple],
           nearest: Sequence[Mapping[str, Any]], cleared: Sequence[Mapping[str, Any]], fonts: Any,
           refused: str | None = None) -> Any:
    """One camera's image with every segment drawn on it, at most ``PANEL_WIDTH`` by ``PANEL_HEIGHT``."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(camera.rgb, dtype=np.uint8), mode="RGB")
    scale = min(PANEL_WIDTH / image.width, PANEL_HEIGHT / image.height)
    grow = max(scale, 1.0) if image.width * scale < 240 else scale
    size = (max(1, round(image.width * grow)), max(1, round(image.height * grow)))
    image = image.resize(size, Image.Resampling.BILINEAR) if size != (image.width, image.height) else image.copy()
    draw = ImageDraw.Draw(image)

    def pixel(point: Sequence[float]) -> tuple[int, int] | None:
        projected = _project_world(point, intrinsic=camera.intrinsic, extrinsic=camera.extrinsic)
        return None if projected is None else (round(projected[0] * grow), round(projected[1] * grow))

    for item in segments:
        colour = AFTER_CONTACT if item["not_predicted"] else colours[item["arm"]]
        pixels = [pixel(point) for point in item["points"]]
        if item["kind"] == "gripper":
            for side in item.get("fingers_to") or ():
                _mark(draw, pixel(side), colour, filled=False, label="", font=fonts.small)
            for side in item.get("fingers_from") or ():
                _mark(draw, pixel(side), colour, filled=True, label="", font=fonts.small)
            continue
        _path(draw, pixels, colour, dashed=not item["planned"] or item["not_predicted"], width=PATH_WIDTH)
        if item["planned"] and len(pixels) > TIME_MARKS:
            for mark in range(1, TIME_MARKS + 1):
                spot = pixels[round(mark * (len(pixels) - 1) / (TIME_MARKS + 1))]
                if spot is not None:
                    draw.ellipse((spot[0] - 4, spot[1] - 4, spot[0] + 4, spot[1] + 4), fill=(255, 255, 255),
                                 outline=HALO, width=1)
    arms = {}
    for item in segments:
        if item["kind"] != "gripper" and not item["not_predicted"]:
            arms.setdefault(item["arm"], []).append(item)
    for arm, parts in arms.items():
        # The jaws where the hand is now, thin, and where it ends, bold; a refused request ends in red.
        end_colour = REFUSED if refused else colours[arm]
        _jaws(draw, [pixel(point) for point in parts[0].get("jaws_start") or ()], colours[arm], 2, image.size)
        _jaws(draw, [pixel(point) for point in parts[-1].get("jaws_end") or ()], end_colour, PATH_WIDTH, image.size)
        # What the jaws hold: its box where the hand is now, thin, and where it ends, bold and labelled.
        _box(draw, [pixel(point) for point in parts[0].get("held_start") or ()], colours[arm], 1)
        held_end = [pixel(point) for point in parts[-1].get("held_end") or ()]
        _box(draw, held_end, end_colour, 2)
        if held_end and all(spot is not None for spot in held_end):
            corner = min(held_end, key=lambda spot: (spot[1], spot[0]))
            _label(draw, (corner[0] + 6, corner[1] - 16), "held", end_colour, fonts.small, corner)
        _mark(draw, pixel(parts[0]["points"][0]), colours[arm], filled=False, label="now", font=fonts.small)
        goal = pixel(parts[-1]["points"][-1])
        if refused:
            _cross(draw, goal, REFUSED, 10)
            if goal is not None:
                _label(draw, (goal[0] + 14, goal[1] - 10), f"asked: {refused}", REFUSED, fonts.body, goal)
        else:
            _mark(draw, goal, colours[arm], filled=True, label="end", font=fonts.small)
    for item in nearest:
        spot = pixel(item["position_m"])
        if spot is not None:
            _cross(draw, spot, NEAREST, 7)
            words = (f"{float(item['inside_m']) * 1000:.0f} mm inside" if item.get("inside_m")
                     else f"closest {float(item['clearance_m']) * 1000:.0f} mm")
            _label(draw, (spot[0] + 11, spot[1] + 2), words, NEAREST, fonts.small, spot)
    for item in cleared:
        spot = pixel(item["position_m"])
        if spot is not None:
            x, y = spot
            draw.polygon(((x, y - 9), (x + 9, y), (x, y + 9), (x - 9, y)), outline=HALO, width=5)
            draw.polygon(((x, y - 8), (x + 8, y), (x, y + 8), (x - 8, y)), outline=CLEARED, width=3)
            _label(draw, (x + 12, y - 7), "planner clears this", CLEARED, fonts.small, spot)
    _label(draw, (4, 2), camera.name, (255, 255, 255), fonts.small)
    return image


def _wrapped(lines: Sequence[str], limit: int) -> list[str]:
    """Each line broken at spaces to at most ``limit`` characters."""
    result = []
    for line in lines:
        current = ""
        for word in line.split(" "):
            if current and len(current) + 1 + len(word) > limit:
                result.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        result.append(current)
    return result


def render(cameras: Sequence[Camera], segments: Sequence[Mapping[str, Any]], *, legend: Sequence[str],
           nearest: Sequence[Mapping[str, Any]] = (), cleared: Sequence[Mapping[str, Any]] = (),
           refused: str | None = None) -> tuple[bytes, dict[str, Any]]:
    """A PNG of the action on the cameras that see it, and what it shows.

    Each of ``segments`` is a plan segment whose ``points`` are the jaw centre's world path, with ``jaws_start`` and
    ``jaws_end`` on an arm's path (and ``held_start`` and ``held_end``, the eight corners of the box its jaws hold) and
    ``fingers_from`` and ``fingers_to`` on a jaw motion. ``nearest`` marks the least clearance (how far inside, where
    the jaws reach into what the depth shows), ``cleared`` where the jaws enter what a planner cleared at an end of the
    path, and ``refused`` (the refusal's code) marks what was asked as refused.
    Example: render(cameras, drawn, legend=['left: 40 native actions'])."""
    from PIL import Image, ImageDraw

    chosen = seen_cameras(cameras, [item["points"] for item in segments])
    arms = list(dict.fromkeys(item["arm"] for item in segments))
    colours = {arm: ARM_COLOURS[index % len(ARM_COLOURS)] for index, arm in enumerate(arms)}
    fonts = _overlay_fonts()
    panels = [_panel(camera, segments, colours, nearest, cleared, fonts, refused) for camera in chosen]
    width = sum(panel.width for panel in panels)
    legend = _wrapped(legend, max(20, (width - 12) // LEGEND_CHAR_PX))
    height = max(panel.height for panel in panels) + LEGEND_LINE * max(1, len(legend)) + 4
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, 0))
        left += panel.width
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(legend):
        draw.text((6, max(panel.height for panel in panels) + 2 + index * LEGEND_LINE), line,
                  fill=REFUSED if refused else (235, 235, 235), font=fonts.small)
    stream = BytesIO()
    canvas.save(stream, format="PNG", optimize=True)
    return stream.getvalue(), {"cameras": [camera.name for camera in chosen], "size": [canvas.width, canvas.height]}


__all__ = ["ARM_COLOURS", "Camera", "MAX_PANELS", "render", "seen_cameras"]
