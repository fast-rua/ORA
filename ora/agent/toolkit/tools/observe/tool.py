"""Current annotated cameras or retained raw cameras, using granted public evidence."""

import base64
import hashlib
import json
from ...interface.api import DataInterface
from ...interface.types import NotRecorded, canonical
from ...lib import images
from ...lib.observation import observe
from ...lib.values import JsonValue
from ..move_pose import _atomic
from ..move_pose._context import BOTH_ARMS
from ..move_pose.tool import measured_states
from ..errors import ToolRefusal
from . import _pointcloud
from ._arguments import solved_history, refuse_historical_arguments, refuse_views


def publish_overlay(api: DataInterface, rendered, view: str, manifest: dict) -> str:
    """Publish one annotated view under its identity; return its PNG as base64."""
    identity = rendered.identity.json()
    key = (
        "overlay-"
        + hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()[:24]
    )
    meta = dict(observation_id=manifest["observation_id"], camera=view)
    record = dict(
        schema="ora.toolkit.observation-overlay.v1",
        key=key,
        identity=identity,
        **meta,
        image_sha256=hashlib.sha256(rendered.png).hexdigest(),
    )
    api.artifacts.publish_tool_artifact(
        kind="observation_overlay",
        key=key,
        files={"manifest.json": canonical(record), "image.png": rendered.png},
        metadata=meta,
    )
    return base64.b64encode(rendered.png).decode("ascii")


def run(api: DataInterface, args: dict) -> dict:
    """Replay: recorded publication only; overlays are pure recomputation.

    Historical step selection uses the boundary-frozen public-state timeline;
    unpublished/future observations cannot be selected. Missing evidence fails.
    """
    tool_config = api.artifacts.tool_context()
    config = tool_config["observe"]
    if tool_config.get("synchronous_execution"):
        config = {**config, "points": api.state.read("tool.marks")}
    step = args.get("step")
    historical = (
        "step" in args
        and step is not None
        and config["retention"]
        and not (type(step) is int and step == config["current_step"])
    )
    selected = None
    if historical:
        refuse_historical_arguments(args)
        if config.get("history_error"):
            raise ToolRefusal.from_public(config["history_error"])
        lo, hi = config["step_range"]
        if type(step) is not int or not lo <= step <= hi:
            raise ToolRefusal(
                f"step {step!r} is not a saved observation step of the current round [{lo}, {hi}]",
                code="invalid_arguments",
                field="step",
                received=step,
                constraints={"minimum": lo, "maximum": hi, "scope": "current_round"},
            )
        selected = config["history"].get(str(step))
        if selected is None and config.get("parent_boundaries_only"):
            raise ToolRefusal(
                f"step {step!r} is an internal publication, not a completed tool-call observation",
                code="invalid_arguments", field="step", received=step,
                constraints={"scope": "completed_tool_calls"})
        if selected is None:
            raise NotRecorded("未记录: saved observation publication")
    else:
        if "step" in args and config["retention"]:
            args = {k: v for k, v in args.items() if k != "step"}
        if set(args) - {"views", "history_point_ids"}:
            raise ToolRefusal("observe accepts views and history_point_ids")
    views = args.get("views") or list(config["default_views"])
    clouds = _pointcloud.views(config)
    refuse_views(
        views,
        set(config["cameras"]) | (set() if historical else set(clouds)),
        historical=historical,
        config=config,
    )
    obs = observe(api, identity=None if selected is None else JsonValue.of(selected))
    manifest = obs.manifest.json()
    atomic = config.get("atomic")
    arms = tuple(config.get("arms") or BOTH_ARMS)
    if atomic:
        grip, opening = _atomic.grip_state(manifest)
        states = {arms[0]: grip}
    else:
        states = measured_states(manifest, arms)
    point_rows = (
        solved_history(config["points"], args.get("history_point_ids") or [])
        if not historical
        else []
    )
    arm_points = ([("", states[arms[0]][:3], (255, 0, 255))] if len(arms) == 1 else [
        ("L", states["left"][:3], (255, 0, 255)),
        ("R", states["right"][:3], (255, 180, 0)),
    ])
    encoded, dimensions = [], {}
    rendered_clouds = None
    for view in views:
        if view in clouds:
            rendered_clouds = rendered_clouds or _pointcloud.render(api, obs, config["pointcloud"])
            rendered = _pointcloud.annotate(rendered_clouds[view], point_rows)
            encoded.append(publish_overlay(api, rendered, view, manifest))
            dimensions[view] = dict(width=rendered.width, height=rendered.height)
            continue
        if view not in manifest["cameras"]:
            raise ToolRefusal("camera is not policy-visible")
        camera_error = obs.result.data.json().get("camera_errors", {}).get(view)
        if camera_error:
            raise ToolRefusal.from_public(camera_error)
        camera = obs.camera(view)
        if camera.rgb is None:
            raise NotRecorded("未记录: observation RGB")
        png = api.artifacts.read_blob(camera.rgb.json())
        if historical:
            width, height = images.dimensions(png)
        else:
            if camera.calibration is None:
                raise NotRecorded("未记录: observation calibration")
            size = config["rgb_resolution"]
            rendered = images.annotate(
                png,
                api.artifacts.read_blob(camera.calibration.json()),
                camera=view,
                points=JsonValue.of(
                    [
                        [label, xyz, list(colour)]
                        for label, xyz, colour in [*arm_points, *point_rows]
                    ]
                ),
                resolution=None if size == "native" else tuple(size),
            )
            png, width, height = rendered.png, rendered.width, rendered.height
            publish_overlay(api, rendered, view, manifest)
        encoded.append(base64.b64encode(png).decode("ascii"))
        dimensions[view] = dict(width=width, height=height)
    if atomic:
        result = dict(success=True, observation_id=manifest["observation_id"], views=views, grip_xyz_m=grip[:3],
                      aperture_mm=opening * 80.0, aperture_limits_mm=dict(near_closed=3.0, open=80.0),
                      images_base64=encoded)
        return {**result, "view_dimensions": dimensions} if atomic["view_dimensions"] else result
    result = dict(
        success=True,
        observation_id=manifest["observation_id"],
        views=views,
        view_dimensions=dimensions,
        images_base64=encoded,
    )
    effectors = manifest["proprioception"]["end_effectors"]
    if historical:
        effector = effectors.get("right") or effectors.get("left")
        result.update(
            status="actual_historical_observation",
            observation_step=step,
            grip_xyz_m=effector["pose_xyz_xyzw"][:3],
            aperture_mm=float(effector.get("gripper", 0)) * 80.0,
            aperture_limits_mm=dict(near_closed=3.0, open=80.0),
        )
    else:
        result["end_effectors"] = {
            arm: dict(
                pose_xyz_xyzw=states[arm],
                gripper_command_target=effectors[arm].get(
                    "gripper_command_target", "unknown"
                ),
                **(
                    {"gripper_opening_m": effectors[arm].get("gripper_opening_m")}
                    if config["measured_jaw"]
                    else {}
                ),
            )
            for arm in arms
        }
    if config["public_facts"]:
        facts = None
        by_view = obs.result.data.json().get("public_robot_facts_by_view")
        if by_view is None:
            raise NotRecorded("未记录: public robot fact projection")
        for view in views:
            if view in clouds:  # a derived view, no camera: its facts are the robot's alone
                continue
            entry = by_view[view]
            if "error" in entry:
                raise ToolRefusal.from_public(entry["error"])
            value = entry["value"]
            if facts is None:
                facts = {**value, "cameras": {}}
            facts["cameras"].update(value["cameras"])
        if facts is None:
            facts = _pointcloud.robot_facts(by_view)
        facts["snapshot_role"] = "historical" if historical else "current"
        for key in ("grip_xyz_m", "aperture_mm", "aperture_limits_mm", "end_effectors"):
            result.pop(key, None)
        result["public_robot_facts"] = facts
    if config["perception_feedback"]:
        result["measurement_contract"] = (
            "RGB pixels: (u,v), top-left origin; world geometry: metres; each image is the named saved view, not a new capture. Missing/occluded pixels provide no evidence."
        )
        result["image_mapping"] = [
            dict(
                image_index=index,
                role="saved_public_view",
                camera=view,
                candidate_id="",
                target_ref="",
                observation_step=step,
            )
            for index, view in enumerate(views)
        ]
    return result
