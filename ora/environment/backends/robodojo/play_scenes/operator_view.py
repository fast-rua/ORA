"""The operator's third-person view of a RoboDojo play scene (worker only, never policy input).

One USD camera, ``operator_camera``, looks at the table from its far side at
three quarters (both arms, the whole tabletop: the presentation view of the
scene previews), with one Replicator render product and RGB annotator. It is
not one of the policy's cameras (``backends/robodojo/session.CAMERAS``): it is
never registered with RoboDojo's camera or capture manager, so no observation,
perception input or policy image can contain it. Its pixels go only to

- ``RobodojoPlaySession.operator_view`` (``ora.environment.base`` contract):
  the latest frame, read after the observation's own render, which Free play
  stores under ``native/operator-view/`` as operator evidence; and
- the execution capture (``execution-capture/``, Control-only), as a fourth
  camera beside the policy's three, so the operator's page plays the robot's
  motion through it.

Reading it never renders, steps physics or touches the scene's random state.
The render product exists while a scene is loaded: ``lifecycle.py`` attaches
it after each reset and releases it before a relaunch closes the stage, as
the bridge releases its capture annotators. Isaac imports stay inside the
functions. ``OPERATOR_VIEW_FIELD`` (a worker-config field no plan can carry)
selects ``capture`` (default: view and capture), ``view`` (view only) or
``off``, so its rendering cost can be measured.
"""

from __future__ import annotations

import io
import math
from typing import Any, Mapping

OPERATOR_CAMERA = "operator_camera"
OPERATOR_CAMERA_PRIM = "/World/OraOperatorView/operator_camera"
#: Episode-frame metres (RoboDojo's table top at z 0.765, the arm bases at y -0.45): from beyond the
#: table's far edge, up and to the right, at the table's centre just in front of the arms. The scene
#: previews' presentation pose and lens, 15 mm at 16:9: the same vertical field as the earlier
#: 640 x 480 view at 20 mm, which the play area and both arms fill, and a wider one across. 1280 x 720
#: because the execution capture records this camera in every frame beside the policy's three
#: 640 x 480 cameras: measured at 30 fps on toys seed 3 (2026-10-20), 720p cost 89 ms per render and
#: 2.7 MB per captured frame against 68 ms and 1.8 MB at 640 x 480; 1920 x 1080 cost 121 ms and
#: 4.35 MB, outran the capture writer (390 ms per frame) and exceeds the capture's 8 MiB per-frame bound.
OPERATOR_VIEW = {"eye": (0.92, 0.50, 1.36), "target": (0.0, -0.17, 0.80), "up": (0.0, 0.0, 1.0),
                 "focal_mm": 15.0, "horizontal_aperture_mm": 20.955, "width": 1280, "height": 720}
OPERATOR_VIEW_FIELD = "operator_view_mode"
OPERATOR_VIEW_MODES = ("capture", "view", "off")
DEFAULT_OPERATOR_VIEW_MODE = "capture"
#: A frame this flat is an annotator that has not rendered yet, not a picture.
MIN_FRAME_STD = 1.0


def operator_view_mode(config: Mapping[str, Any]) -> str:
    """The worker config's operator-view mode (``capture`` when absent)."""
    mode = config.get(OPERATOR_VIEW_FIELD, DEFAULT_OPERATOR_VIEW_MODE)
    if mode not in OPERATOR_VIEW_MODES:
        raise ValueError(f"{OPERATOR_VIEW_FIELD} must be one of {list(OPERATOR_VIEW_MODES)}")
    return str(mode)


def look_at(eye: Any, target: Any, up: Any = (0.0, 0.0, 1.0)) -> dict[str, list[float]]:
    """The camera's forward, right and up axes (a USD camera looks along its -z): pure math."""
    def unit(vector: Any) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    forward = unit([t - e for e, t in zip(eye, target)])
    right = unit([forward[1] * up[2] - forward[2] * up[1], forward[2] * up[0] - forward[0] * up[2],
                  forward[0] * up[1] - forward[1] * up[0]])
    true_up = [right[1] * forward[2] - right[2] * forward[1], right[2] * forward[0] - right[0] * forward[2],
               right[0] * forward[1] - right[1] * forward[0]]
    return {"forward": forward, "right": right, "up": true_up}


def attach(origin: Any) -> dict[str, Any]:
    """Define the operator camera on the current stage with its render product and RGB annotator."""
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, UsdGeom

    view = OPERATOR_VIEW
    stage = omni.usd.get_context().get_stage()
    camera = UsdGeom.Camera.Define(stage, OPERATOR_CAMERA_PRIM)
    camera.GetFocalLengthAttr().Set(view["focal_mm"])
    camera.GetHorizontalApertureAttr().Set(view["horizontal_aperture_mm"])
    camera.GetVerticalApertureAttr().Set(view["horizontal_aperture_mm"] * view["height"] / view["width"])
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 50.0))
    if hasattr(origin, "detach"):
        origin = origin.detach().cpu().numpy()
    offset = np.asarray(origin, dtype=float).reshape(-1)[:3]
    eye = Gf.Vec3d(*(np.asarray(view["eye"]) + offset))
    target = Gf.Vec3d(*(np.asarray(view["target"]) + offset))
    matrix = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(*view["up"])).GetInverse()
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(matrix)
    product = rep.create.render_product(OPERATOR_CAMERA_PRIM, (view["width"], view["height"]))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])
    return {"camera": OPERATOR_CAMERA, "prim": OPERATOR_CAMERA_PRIM, "product": product, "annotator": annotator}


def release(handle: Mapping[str, Any] | None) -> None:
    """Detach the annotator and destroy the render product (before the stage they render closes).

    ``detach`` takes render-product paths, as the bridge's own capture release
    does (``_release_capture_products``); the product object is a HydraTexture.
    """
    if not handle:
        return
    product = handle["product"]
    try:
        handle["annotator"].detach([getattr(product, "path", product)])
    finally:
        destroy = getattr(product, "destroy", None)
        if callable(destroy):
            destroy()


def read_rgb(handle: Mapping[str, Any] | None) -> Any:
    """The annotator's latest RGB frame (H x W x 3 uint8), or None before it has rendered one. No render."""
    if not handle:
        return None
    import numpy as np

    raw = handle["annotator"].get_data()
    raw = raw.get("data") if isinstance(raw, Mapping) else raw
    if raw is None or np.size(raw) == 0:
        return None
    array = np.asarray(raw)
    if array.ndim != 3 or array.shape[2] < 3:
        return None
    rgb = np.ascontiguousarray(array[:, :, :3]).astype(np.uint8)
    return rgb if float(rgb.std()) >= MIN_FRAME_STD else None


def encode(rgb: Any) -> dict[str, Any]:
    """``base.validate_operator_view``'s answer for one RGB frame."""
    import numpy as np
    from PIL import Image

    array = np.asarray(rgb)
    output = io.BytesIO()
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(output, format="PNG", compress_level=1)
    return {"png": output.getvalue(), "camera": OPERATOR_CAMERA, "width": int(array.shape[1]),
            "height": int(array.shape[0])}


__all__ = ["DEFAULT_OPERATOR_VIEW_MODE", "OPERATOR_CAMERA", "OPERATOR_CAMERA_PRIM", "OPERATOR_VIEW",
           "OPERATOR_VIEW_FIELD", "OPERATOR_VIEW_MODES", "attach", "encode", "look_at", "operator_view_mode",
           "read_rgb", "release"]
