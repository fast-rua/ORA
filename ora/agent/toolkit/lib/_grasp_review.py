"""Pure grasp contact-sheet and world-cloud renderers; arrays supplied by callers."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from . import _grasp_extent as scene_extent_review
from ._grasp_frames import _compiled_grasp_geometry
from ._grasp_overlay import (
    _draw_candidate_card, _draw_contact_sheet_legend, _draw_target_panel,
    _gripper_wireframe, _new_contact_sheet, _overlay_camera_matrices,
    _overlay_fonts, _overlay_target, _project_world, _tinted_target_source,
)
GRASP_BACKEND_SINGLE_VIEW_SCOPE = "target_source_camera_rgbd_only"
GRASP_PLAN_VISUALIZATION_ID = "grasp-plan-target-local-contact-sheet.v2"
GRASP_PLAN_REVIEW_ID = "grasp-plan-global-pointcloud-review.v3"
#: One colour per drawable grasp candidate, shared by the camera overlay and
#: the global review so the two renderings of one candidate list cannot
#: disagree. The number of colours is the number of candidates that can be
#: attributed, so extending this table is what raises that bound.
GRASP_CANDIDATE_COLOURS = ((255, 82, 82), (80, 210, 255), (255, 205, 64), (96, 255, 96))
GRASP_MAX_DRAWABLE_CANDIDATES = len(GRASP_CANDIDATE_COLOURS)


def _render_grasp_overlay(
    *, rgb, mask, calibration, target_world, mask_sha256,
    observation_id: str, target_ref: str,
    candidates: list[Mapping[str, Any]],
    compiler_identity: Mapping[str, Any],
    execution_tool: str = "move_free_space",
) -> tuple[Any, dict[str, Any]]:
    """Render the target-local contact sheet; the steps are in overlay.py."""

    source, mask, source_array = _tinted_target_source(rgb, mask)
    intrinsic, extrinsic = _overlay_camera_matrices(calibration)
    colours = GRASP_CANDIDATE_COLOURS
    fonts = _overlay_fonts()
    target = _overlay_target(source, mask, source_array)
    compiled_records, projection_records = _grasp_overlay_candidate_records(
        candidates, colours, compiler_identity=compiler_identity,
        intrinsic=intrinsic, extrinsic=extrinsic,
    )
    image, draw = _new_contact_sheet(
        observation_id=observation_id, target_ref=target_ref, fonts=fonts,
    )
    target_crop = _draw_target_panel(image, draw, target, fonts)
    _draw_contact_sheet_legend(
        draw, compiler_identity=compiler_identity,
        execution_tool=execution_tool, fonts=fonts,
    )
    for index in range(3):
        _draw_candidate_card(
            image, draw, index=index, compiled_records=compiled_records,
            target=target, intrinsic=intrinsic, extrinsic=extrinsic,
            target_world=target_world, fonts=fonts,
        )

    return image, {
        "renderer": GRASP_PLAN_VISUALIZATION_ID,
        "source_resolution": [source.width, source.height],
        "target_crop_xyxy": list(target_crop),
        "render_resolution": [image.width, image.height],
        "target_mask_sha256": mask_sha256,
        "candidate_projections": projection_records,
        "candidate_panel_count": len(compiled_records),
        "policy_visible": True,
        "claim_boundary": (
            "target_local_geometry_preview_only; deterministic public RGB-D "
            "derivation; IK, reachability, contact, retention and task success "
            "are not established"
        ),
    }


def _grasp_overlay_candidate_records(
    candidates: list[Mapping[str, Any]], colours: Any, *,
    compiler_identity: Mapping[str, Any], intrinsic: Any, extrinsic: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compile each drawable candidate and project its wireframe points."""

    projection_records: list[dict[str, Any]] = []
    compiled_records: list[dict[str, Any]] = []
    for candidate, colour in zip(candidates, colours, strict=False):
        geometry = _compiled_grasp_geometry(
            candidate, compiler_identity=compiler_identity,
        )
        shape = _gripper_wireframe(geometry, float(candidate["gripper_opening_m"]))
        world_points = list(shape["points"])
        source_pixels = [
            _project_world(point, intrinsic=intrinsic, extrinsic=extrinsic)
            for point in world_points
        ]
        compiled_records.append({
            "candidate": candidate,
            "colour": colour,
            "geometry": geometry,
            "wireframe": shape,
            "source_pixels": source_pixels,
        })
        projection_records.append({
            "candidate_id": candidate["candidate_id"],
            "rank": int(candidate["rank"]),
            "raw_origin_uv": list(source_pixels[4]) if source_pixels[4] else None,
            "contact_baseline_uv": list(source_pixels[5]) if source_pixels[5] else None,
            "grip_site_uv": list(source_pixels[6]) if source_pixels[6] else None,
            "pregrasp_uv": list(source_pixels[7]) if source_pixels[7] else None,
            "compiler_status": "returned",
        })
    return compiled_records, projection_records


def _render_grasp_global_review(
    *, camera_overlay: Any, camera_panel_resolution: list[int],
    scene_world: Any, target_world: Any, scene_source_index: Any,
    scene_sources: list[Mapping[str, Any]],
    workspace_bounds_m: Mapping[str, list[float]],
    observation_id: str, target_ref: str, camera: str,
    candidates: list[Mapping[str, Any]],
    compiler_identity: Mapping[str, Any],
    backend_input_scope: str = GRASP_BACKEND_SINGLE_VIEW_SCOPE,
    backend_cameras: Sequence[str] | None = None,
    backend_points_by_camera: Mapping[str, int] | None = None,
    backend_excluded_cameras: Sequence[Mapping[str, Any]] | None = None,
    observed_extent: Mapping[str, Any] | None = None,  # scene_extent_review.cell(...)
    policy_visible: bool = False,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from PIL import Image, ImageDraw

    scene = np.asarray(scene_world, dtype=np.float64)
    target = np.asarray(target_world, dtype=np.float64)
    source_index = np.asarray(scene_source_index, dtype=np.int16)
    if source_index.shape != (len(scene),):
        raise ValueError("grasp review source attribution is malformed")
    axes = {"x": 0, "y": 1, "z": 2}
    for axis in axes:
        bounds = workspace_bounds_m.get(axis)
        if (
            not isinstance(bounds, list) or len(bounds) != 2
            or not all(isinstance(value, (int, float)) for value in bounds)
            or float(bounds[0]) >= float(bounds[1])
        ):
            raise ValueError("grasp review requires finite xyz workspace bounds")

    colours = GRASP_CANDIDATE_COLOURS
    panel_size = 360
    panel_margin = 22
    view_specs = (
        ("TOP", "x", "y"),
        ("FRONT", "x", "z"),
        ("SIDE", "y", "z"),
    )
    view_records: list[dict[str, Any]] = []

    def make_panel(title: str, horizontal: str, vertical: str) -> Any:
        horizontal_index = axes[horizontal]
        vertical_index = axes[vertical]
        horizontal_bounds = [float(value) for value in workspace_bounds_m[horizontal]]
        vertical_bounds = [float(value) for value in workspace_bounds_m[vertical]]
        lower_h, upper_h = horizontal_bounds
        lower_v, upper_v = vertical_bounds
        raster = np.full((panel_size, panel_size, 3), [28, 32, 40], dtype=np.uint8)

        def pixels(points: Any) -> tuple[Any, Any, Any]:
            values = np.asarray(points, dtype=np.float64)
            valid = (
                np.isfinite(values).all(axis=1)
                & (values[:, horizontal_index] >= lower_h)
                & (values[:, horizontal_index] <= upper_h)
                & (values[:, vertical_index] >= lower_v)
                & (values[:, vertical_index] <= upper_v)
            )
            selected = values[valid]
            if not len(selected):
                return np.empty(0, dtype=int), np.empty(0, dtype=int), valid
            x = np.rint(
                (selected[:, horizontal_index] - lower_h)
                / (upper_h - lower_h) * (panel_size - 1)
            ).astype(int)
            y = np.rint(
                (upper_v - selected[:, vertical_index])
                / (upper_v - lower_v) * (panel_size - 1)
            ).astype(int)
            return x, y, valid

        scene_x, scene_y, scene_valid = pixels(scene)
        retained_sources = source_index[scene_valid]
        source_colours = ((102, 110, 124), (151, 122, 232))
        for index, source_colour in enumerate(source_colours):
            selected = retained_sources == index
            raster[scene_y[selected], scene_x[selected]] = source_colour
        target_x, target_y, target_valid = pixels(target)
        for delta_x, delta_y in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            x = np.clip(target_x + delta_x, 0, panel_size - 1)
            y = np.clip(target_y + delta_y, 0, panel_size - 1)
            raster[y, x] = [32, 210, 190]
        image = Image.fromarray(raster, mode="RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, panel_size - 1, panel_size - 1), outline=(82, 90, 104))
        draw.text((10, 9), scene_extent_review.panel_title(observed_extent is not None, f"{title}  {horizontal.upper()}–{vertical.upper()}"), fill=(242, 244, 248))
        draw.text(
            (10, panel_size - 20),
            f"{horizontal} [{lower_h:+.2f},{upper_h:+.2f}]  "
            f"{vertical} [{lower_v:+.2f},{upper_v:+.2f}] m",
            fill=(178, 187, 200),
        )

        def one_point(point: Any) -> tuple[int, int] | None:
            x, y, valid = pixels(np.asarray(point, dtype=np.float64).reshape(1, 3))
            return (int(x[0]), int(y[0])) if bool(valid[0]) else None

        for candidate, colour in zip(candidates, colours, strict=False):
            geometry = _compiled_grasp_geometry(
                candidate, compiler_identity=compiler_identity,
            )
            center = np.asarray(geometry["grip_site"], dtype=np.float64)
            pregrasp = np.asarray(
                geometry["pregrasp_grip_site"], dtype=np.float64,
            )
            jaw = np.asarray(geometry["jaw_axis"], dtype=np.float64)
            # Draw the jaw at the compiled opening, capped by *this*
            # embodiment's admission limit. The literal 0.04 that used to sit
            # here is half the Panda maximum and was a wrong-embodiment
            # constant on every non-Panda panel.
            half = min(
                float(compiler_identity.get("maximum_opening_m", 0.08)),
                float(candidate["gripper_opening_m"]),
            ) / 2.0
            center_pixel = one_point(center)
            pregrasp_pixel = one_point(pregrasp)
            jaw_pixels = (one_point(center - jaw * half), one_point(center + jaw * half))
            if jaw_pixels[0] is not None and jaw_pixels[1] is not None:
                draw.line(jaw_pixels, fill=colour, width=4)
            if center_pixel is not None:
                x, y = center_pixel
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=colour, width=2)
                draw.text((x + 7, y + 2), f"G{candidate['rank']}", fill=colour)
            if pregrasp_pixel is not None:
                x, y = pregrasp_pixel
                draw.rectangle((x - 4, y - 4, x + 4, y + 4), outline=colour, width=2)
            if center_pixel is not None and pregrasp_pixel is not None:
                draw.line((pregrasp_pixel, center_pixel), fill=colour, width=2)

        view_records.append({
            "view": title.lower(),
            "horizontal_axis": horizontal,
            "vertical_axis": vertical,
            "horizontal_bounds_m": horizontal_bounds,
            "vertical_bounds_m": vertical_bounds,
            "scene_points_in_bounds": int(np.count_nonzero(scene_valid)),
            "target_points_in_bounds": int(np.count_nonzero(target_valid)),
            "scene_points_by_source": {
                str(source["camera"]): int(np.count_nonzero(
                    retained_sources == int(source["source_index"])
                ))
                for source in scene_sources
            },
        })
        return image

    cell_size = 384
    left_width = cell_size * 2
    legend_width = 400
    canvas = Image.new("RGB", (left_width + legend_width, cell_size * 2), (16, 19, 25))
    draw = ImageDraw.Draw(canvas)
    source_width, source_height = camera_panel_resolution
    source_panel = camera_overlay.crop((0, 0, source_width, source_height)).convert("RGB")
    source_panel.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)
    camera_cell = Image.new("RGB", (panel_size, panel_size), (28, 32, 40))
    camera_cell.paste(
        source_panel,
        ((panel_size - source_panel.width) // 2, (panel_size - source_panel.height) // 2),
    )
    camera_draw = ImageDraw.Draw(camera_cell)
    camera_draw.rectangle((0, 0, panel_size - 1, panel_size - 1), outline=(82, 90, 104))
    camera_draw.text(
        (10, 9), f"POLICY CONTACT SHEET  {camera}", fill=(242, 244, 248),
    )
    panels = [camera_cell if observed_extent is None else observed_extent["image"]] + [
        make_panel(*spec) for spec in view_specs]
    for index, panel in enumerate(panels):
        column = index % 2
        row = index // 2
        canvas.paste(
            panel,
            (column * cell_size + panel_margin // 2,
             row * cell_size + panel_margin // 2),
        )

    left = left_width + 18
    draw.text((left, 18), "GRASP BACKEND + GLOBAL REVIEW", fill=(242, 244, 248))
    draw.text((left, 44), f"observation  {observation_id[-34:]}", fill=(165, 174, 190))
    draw.text((left, 62), f"target       {target_ref}", fill=(165, 174, 190))
    draw.text((left, 92), "REVIEW SCOPE", fill=(255, 176, 64))
    review_caption, backend_caption = scene_extent_review.captions(
        observed_extent is not None, scene_sources=scene_sources,
        backend_cameras=backend_cameras, camera=camera)
    draw.text((left, 112), review_caption, fill=(225, 229, 236))
    draw.text((left, 132), backend_caption, fill=(255, 176, 64))
    draw.text((left, 158), f"scene points   {len(scene):,}", fill=(192, 200, 212))
    draw.text((left, 178), f"target points  {len(target):,}", fill=(32, 210, 190))
    source_top = 198
    source_colours = ((102, 110, 124), (151, 122, 232))
    for source, source_colour in zip(scene_sources, source_colours, strict=False):
        retained_count = int(source.get("retained_voxel_points", 0))
        draw.text(
            (left, source_top),
            f"{source['camera']:<10} {retained_count:,} retained",
            fill=source_colour,
        )
        source_top += 18
    draw.text((left, source_top + 12), "CANDIDATES", fill=(242, 244, 248))
    top = source_top + 38
    for candidate, colour in zip(candidates, colours, strict=False):
        draw.rectangle((left, top + 2, left + 10, top + 12), fill=colour)
        draw.text(
            (left + 18, top),
            f"G{candidate['rank']}  score {float(candidate['backend_score']):.3f}  "
            f"opening {float(candidate['gripper_opening_m']) * 1000:.1f} mm",
            fill=(235, 238, 243),
        )
        top += 28
    warning_top = canvas.height - 122
    draw.line((left, warning_top, canvas.width - 18, warning_top), fill=(68, 75, 89))
    draw.text((left, warning_top + 12), "REVIEW-ONLY DERIVATION", fill=(80, 210, 255))
    legend_names = [str(source["camera"]) for source in scene_sources]
    draw.text(
        (left, warning_top + 34),
        " · ".join(
            f"{shade} {name} review input"
            for shade, name in zip(("gray", "purple"), legend_names, strict=False)
        ) or "no review scene input",
        fill=(192, 200, 212),
    )
    draw.text((left, warning_top + 54), "cyan: target-mask input points", fill=(32, 210, 190))
    draw.text((left, warning_top + 78), "missing surfaces remain missing", fill=(255, 176, 64))
    draw.text((left, warning_top + 98), "not IK, collision, contact or retention proof", fill=(255, 176, 64))
    resolved_backend_cameras = (
        [str(source["camera"]) for source in scene_sources]
        if backend_cameras is None else [str(name) for name in backend_cameras]
    )
    return canvas, {
        "renderer": GRASP_PLAN_REVIEW_ID,
        "review_scope": "synchronized_agentview_wrist_rgbd_fusion",
        # Derived, never asserted: this field named a single view even after
        # the caller began fusing, so it now restates what the proposer got.
        "backend_input_scope": backend_input_scope,
        "backend_cameras": resolved_backend_cameras,
        "backend_points_by_camera": (
            {} if backend_points_by_camera is None
            else {str(key): int(value) for key, value in backend_points_by_camera.items()}
        ),
        "backend_excluded_cameras": [
            dict(item) for item in (backend_excluded_cameras or ())
        ],
        "target_source_camera": camera,
        "scene_cameras": [str(source["camera"]) for source in scene_sources],
        "scene_points_by_source": {
            str(source["camera"]): int(source.get("retained_voxel_points", 0))
            for source in scene_sources
        },
        "scene_point_count": int(len(scene)),
        "target_point_count": int(len(target)),
        "views": view_records,
        **({} if observed_extent is None else {"observed_extent": observed_extent["record"]}),
        "render_resolution": [canvas.width, canvas.height],
        # True where perception_feedback appends this image to the result.
        "policy_visible": policy_visible,
        "claim_boundary": (
            "review_only fused public-camera projection; the proposer received "
            + ", ".join(resolved_backend_cameras)
            + "; the target-mask subset stays single-view from "
            + camera
            + "; surfaces missing from every contributing camera remain missing; "
            "not IK, collision, contact or retention evidence"
        ),
    }


