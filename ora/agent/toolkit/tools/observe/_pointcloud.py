"""The point-cloud views ``observe`` adds on a surface that declares them (``config["pointcloud"]``)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...lib import fusion, observation, pointcloud
from ...lib.images import RenderedView
from ...lib.values import JsonValue
from ..errors import ToolRefusal


def views(config: Mapping[str, Any]) -> tuple[str, ...]:
    """The derived view names this surface admits; none without its section."""

    return tuple(pointcloud.VIEW_AXES) if "pointcloud" in config else ()


def render(
    api: Any, obs: observation.Observation, section: Mapping[str, Any]
) -> dict[str, pointcloud.CloudView]:
    """Fuse the section's cameras from ``obs`` and render every orthographic view."""

    sources = []
    for name in section["cameras"]:
        if name not in obs.manifest.json().get("cameras", {}):
            raise ValueError(f"pointcloud fusion requires camera {name}")
        refs = obs.camera(name)
        sources.append(
            fusion.FusionSource(
                observation.load_camera(api, refs),
                JsonValue.of(refs.declaration.json()["files"]),
            )
        )
    bounds = JsonValue.of(section["workspace_bounds_m"])
    return pointcloud.render(
        fusion.fuse(tuple(sources), bounds),
        bounds,
        resolution=tuple(section["resolution"]),
    )


def annotate(view: pointcloud.CloudView, rows: Sequence[Any]) -> RenderedView:
    """Draw ``observe``'s history marks on one view; arm marks are for camera views."""

    points = [[label, xyz, list(colour)] for label, xyz, colour in rows]
    return pointcloud.annotate(view, points=JsonValue.of(points))


def robot_facts(by_view: Mapping[str, Any]) -> dict[str, Any]:
    """Public robot facts with no camera, for a call that named only derived views."""

    entries = list(by_view.values())
    entry = next((item for item in entries if "error" not in item), entries[0])
    if "error" in entry:
        raise ToolRefusal.from_public(entry["error"])
    return {**entry["value"], "cameras": {}}
