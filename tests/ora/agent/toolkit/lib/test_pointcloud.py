"""Orthographic cloud views are byte for byte what the two-view materialization writes."""

import json

import numpy as np
from PIL import Image

from ora.agent.toolkit.lib import fusion, observation, pointcloud
from ora.agent.toolkit.lib.values import Array, JsonValue, TransformM
from ora.agent.toolkit.host.shared.rgbd_fusion import VIEW_AXES, materialize_two_view_pointcloud
from tests.support.agent.toolkit_perception import BOUNDS, camera_pose, rendered_camera


def _published(root, name, tilt_deg):
    pose = camera_pose(tilt_deg)
    rgb, depth, intrinsic = rendered_camera(pose)
    (root / name).mkdir(parents=True)
    Image.fromarray(rgb, mode="RGB").save(root / name / "rgb.png")
    np.save(root / name / "depth-m.npy", depth, allow_pickle=False)
    calibration = dict(intrinsic_cv=intrinsic, extrinsic_cv=np.linalg.inv(pose).tolist())
    (root / name / "calibration.json").write_text(json.dumps(calibration))
    import hashlib

    files = {}
    for role, filename in (("rgb_lossless_png", "rgb.png"), ("depth_numpy_m_f32", "depth-m.npy"),
                           ("calibration", "calibration.json")):
        data = (root / name / filename).read_bytes()
        files[role] = dict(path=f"{name}/{filename}", sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    camera = observation.CameraRGBD(
        name, Array.of(rgb), Array.of(depth), Array.of(np.asarray(intrinsic, dtype=float)),
        TransformM(f"camera/{name}", "world", Array.of(pose)), JsonValue.of(calibration))
    return dict(files=files), fusion.FusionSource(camera, JsonValue.of(files))


def test_views_equal_the_materialized_ones(tmp_path):
    source = tmp_path / "observation"
    agentview, first = _published(source, "agentview", 40.0)
    wrist, second = _published(source, "wrist", 0.0)
    materialized = materialize_two_view_pointcloud(
        observation_root=source, observation=dict(observation_id="o", cameras=dict(agentview=agentview, wrist=wrist)),
        output_root=tmp_path / "old", workspace_bounds_m=BOUNDS, resolution=(96, 64))
    bounds = JsonValue.of(BOUNDS)
    views = pointcloud.render(fusion.fuse((first, second), bounds), bounds, resolution=(96, 64))
    assert tuple(views) == tuple(VIEW_AXES) == tuple(pointcloud.VIEW_AXES)
    for name, view in views.items():
        record = materialized["views"][name]
        assert view.png == (tmp_path / "old" / f"{name}.png").read_bytes()
        assert [view.width, view.height] == record["resolution"]
        assert list(view.horizontal_bounds_m) == record["horizontal_bounds_m"]
        assert list(view.vertical_bounds_m) == record["vertical_bounds_m"]
        assert (view.horizontal_axis, view.vertical_axis) == (record["horizontal_axis"], record["vertical_axis"])


def test_a_mark_outside_a_view_keeps_its_identity_and_draws_nothing(tmp_path):
    bounds = JsonValue.of(BOUNDS)
    _, source = _published(tmp_path, "agentview", 40.0)
    view = pointcloud.render(fusion.fuse((source,), bounds), bounds, resolution=(32, 32))["pointcloud_top"]
    drawn = pointcloud.annotate(view, points=JsonValue.of([["far", [9.0, 9.0, 1.0], [0, 180, 255]]]))
    assert drawn.png == view.png
    assert drawn.identity.json()["points"][0]["label"] == "far"
    assert drawn.identity.json()["view"] == "pointcloud_top"
