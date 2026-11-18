"""One view of the current observation with world marks on it, published under the name of what drew them."""

import base64

from ...lib import images, pointcloud
from ...lib.images import RenderedView
from ...lib.values import JsonValue
from ..errors import ToolRefusal
from .tool import publish_overlay


def published(api, rendered, view, manifest, kind):
    """Publish ``rendered`` as ``kind``, the overlay's identity; its PNG as base64."""
    identity = JsonValue.of({**rendered.identity.json(), "kind": kind})
    return publish_overlay(api, RenderedView(rendered.png, rendered.width, rendered.height, identity), view, manifest)


def camera_view(api, obs, atomic, camera, points):
    """``camera`` at the surface's RGB resolution with ``points``, each ``[label, xyz_m, colour]``, projected on it."""
    manifest = obs.manifest.json()
    if camera not in manifest.get("cameras", {}):
        raise ToolRefusal("camera is not policy-visible")
    failure = obs.result.data.json().get("camera_errors", {}).get(camera)
    if failure:
        raise ToolRefusal.from_public(failure)
    refs = obs.camera(camera)
    size = atomic["rgb_resolution"]
    return images.annotate(api.artifacts.read_blob(refs.rgb.json()), api.artifacts.read_blob(refs.calibration.json()),
                           camera=camera, points=JsonValue.of(points), resolution=None if size == "native" else tuple(size))


def encoded(png):
    return base64.b64encode(png).decode("ascii")


def cloud_view(clouds, view, points):
    """One derived view with ``points`` drawn at their world coordinates."""
    return pointcloud.annotate(clouds[view], points=JsonValue.of(points))
