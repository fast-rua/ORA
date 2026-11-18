"""Trusted publication grants; portable observe performs argument policy and rendering."""

import hashlib
import json
import os
import time
from ora.agent.toolkit.host.shared import saved_steps
from ora.agent.error_classes import ToolError
from ora.agent.toolkit.host.shared.artifact_io import _append, _artifact
from ora.agent.toolkit.interface.types import PublicData, AccessDenied, NotRecorded

from .publication import published_manifest
from .state import ATOMIC_CAPABILITY


def atomic_configuration(host):
    """What the atomic tools (``move_pose/_atomic.py``) read of this host. Only their surfaces have this section, so a
    context without it, every sealed one among them, reads as before."""
    return dict(
        rgb_resolution=host.view_config["rgb_resolution"],
        view_dimensions=host.surface.view_config is not None,
        feedback_batch=host._active_mark_point_batch is not None,
        controlled_end_effector=host.controlled_end_effector,
        pointcloud=dict(cameras=["agentview", "wrist"], workspace_bounds_m=dict(host.workspace_bounds_m),
                        resolution=list(host._derived_resolution())),
    )


def configuration(host):
    control, _, _ = host._current()
    result = dict(
        perception_feedback=host.surface.supports("public_rgbd_perception_feedback_v2"),
        cameras=sorted(host._physical_policy_cameras()),
        default_views=list(host.view_config["default_observe_views"]),
        rgb_resolution=host.view_config["rgb_resolution"],
        points=host.toolkit_state_access.read("tool.marks"),
        retention=host.surface.supports("public_rgbd_grasp_retention_recovery_v5"),
        current_step=saved_steps.current_step(host),
        history={},
        public_facts=host.surface.supports("current_public_robot_facts_v1"),
        measured_jaw=host.surface.supports("measured_jaw_separation_v1"),
    )
    from .motion import environment_plans

    if environment_plans(host.surface) and not host.surface.supports("dual_arm_sparse_targets_v1"):
        result["arms"] = ["right"]
    if host.surface.supports(ATOMIC_CAPABILITY):
        result["atomic"] = atomic_configuration(host)
    publications = {}
    try:
        states, start = saved_steps.round_timeline(host, control)
        result["step_range"] = [states[start]["step"], states[-1]["step"]]
        from .visibility import visible_steps
        visible = visible_steps(host, states[start:])
        if host.surface.supports("toolkit_legacy_projection_v1"):
            result["parent_boundaries_only"] = True
        for state in states[start:]:
            if state["step"] not in visible:
                continue
            identity = dict(
                observation_id=state["observation"]["id"],
                observation_sha256=state["observation"]["manifest"]["sha256"],
            )
            result["history"][str(state["step"])] = identity
            publications[identity["observation_id"]] = (state["step"], identity)
    except ToolError as error:
        result["history_error"] = error.as_dict()
    if host.surface.supports("two_view_pointcloud_v1"):
        result["pointcloud"] = dict(
            cameras=["agentview", "wrist"],
            workspace_bounds_m=dict(host.workspace_bounds_m),
            resolution=list(host._derived_resolution()),
        )
    for capability, name in (("robotwin_rgbd_v1", "RoboTwin"), ("yam_rgbd_v1", "station")):
        if host.surface.supports(capability):
            result["camera_list"] = name
    return result, publications


def publication(host, store, identity, root, manifest):
    """Manifest-declared public bytes and the existing public-facts projection only."""
    root, manifest, ref = published_manifest(store, identity)
    value = dict(observation=identity, manifest=ref, files=[], camera_errors={})
    paths = [(root / "proprioception.json", None)]
    for name, camera in manifest.get("cameras", {}).items():
        try:
            host._camera(root, manifest, name)
        except ToolError as error:
            value["camera_errors"][name] = error.as_dict()
            continue
        paths.extend(
            (root / item["path"], item) for item in camera.get("files", {}).values()
        )
    for path, declared in paths:
        if not path.is_file():
            continue
        media = (
            "image/png"
            if path.suffix == ".png"
            else "application/x-npy" if path.suffix == ".npy" else "application/json"
        )
        item = store.reference(
            path.relative_to(store.root).as_posix(), media_type=media
        )
        if declared is not None and any(
            item[key] != declared[key] for key in ("sha256", "bytes")
        ):
            raise AccessDenied("published camera artifact hash drift")
        value["files"].append(item)
    if host.surface.supports("current_public_robot_facts_v1"):
        facts = {}
        for view in host._physical_policy_cameras() & manifest["cameras"].keys():
            try:
                projected = host._attach_observe_public_facts(
                    dict(views=[view]), root, manifest
                )
                facts[view] = dict(value=projected["public_robot_facts"])
            except ToolError as error:
                facts[view] = dict(error=error.as_dict())
        value["public_robot_facts_by_view"] = facts
    return PublicData.from_json(value, sources=[ref])


def adapter(host, store, publications):
    def read(parameters, inputs, target_store):
        identity = parameters["observation"]
        from .state import current as current_state
        current, root, manifest = current_state(host)
        if identity != current:
            row = publications.get(identity["observation_id"])
            if row is None or row[1] != identity:
                raise NotRecorded("unbound historical observation")
            _, _, root, manifest = host._select_observation(row[0])
        return publication(host, store, identity, root, manifest)

    return read


def selected_adapter(host, store, identity):
    """The one publication this call selected: unlike ``adapter`` it asks for no open observation, so a saved step stays
    readable while an action is pending, as it always was on ``robot_runtime_v2``."""

    def read(parameters, inputs, target_store):
        if parameters["observation"] != identity:
            raise NotRecorded("unbound historical observation")
        return publication(host, store, identity, None, None)

    return read


def publish(host, key, files, metadata):
    """Fixed compatibility overlay destination, preserving old artifact/cache checks."""
    record = json.loads(files["manifest.json"])
    observation_id = metadata["observation_id"]
    target = (
        host.artifact_root
        / hashlib.sha256(observation_id.encode()).hexdigest()[:20]
        / "overlays"
        / (key.removeprefix("overlay-") + ".png")
    )
    expected = files["image.png"]
    if target.is_file():
        if target.read_bytes() != expected:
            raise ToolError(
                "derived view artifact hash drift",
                code="artifact_hash_mismatch",
                retryable=False,
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.png")
        temporary.write_bytes(expected)
        os.replace(temporary, target)
        target.chmod(0o444)
    _append(
        host.event_path,
        dict(
            type="openeta_view_derived",
            observation_id=observation_id,
            identity=record["identity"],
            artifact=_artifact(target, host.control_root.parent),
            time=time.time(),
        ),
    )
