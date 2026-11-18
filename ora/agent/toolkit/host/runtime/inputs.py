"""Freeze only public tool artifact stores before a parent call; replay never scans them."""

import json
from pathlib import Path
from ora.agent.toolkit.interface.tool_data import verify_artifact, artifact_key
from ora.agent.toolkit.interface.types import canonical
from ora.agent.toolkit.host.perception.target_store import load_segmented_target
from ora.agent.toolkit.host.shared.artifact_io import _verify_artifact
from ora.agent.toolkit.host.shared import service_refusals, target_refusals
from ora.agent.error_classes import ToolError
from .tool_data import configuration
from ora.agent.toolkit.host.execution import configuration as execution_configuration
from ora.agent.toolkit.host.state import ATOMIC_CAPABILITY


def read_artifact(store, kind, key, root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_bytes())
    refs = [] if kind == "segmentation_query" else [manifest["overlay"]]
    if kind == "segmentation_query":
        refs.extend([manifest["raw_masks"], *manifest["images"]])
    elif kind == "target":
        refs.append(manifest["private_mask"])
    elif kind == "grounding_query":
        refs.extend(
            row["artifact"] for row in manifest["visualization"]["candidate_previews"]
        )
    else:
        refs.append(manifest["input_pointcloud"]["artifact"])
        refs.extend(
            manifest[role]["artifact"]
            for role in ("review_visualization", "review_pointcloud", "backend_output")
            if isinstance(manifest.get(role), dict) and "artifact" in manifest[role]
        )
    for ref in refs:
        _verify_artifact(root, ref)
    names = {"manifest.json", *(ref["path"] for ref in refs)}
    if any(Path(name).name != name for name in names):
        raise ValueError("public artifact member path is not a filename")
    files = {name: (root / name).read_bytes() for name in names}
    meta = {k: manifest["identity"][k] for k in ("observation_id", "camera")}
    verify_artifact(kind, key, files, meta)

    def media(name):
        return (
            "image/png"
            if name.endswith(".png")
            else (
                "application/x-npy"
                if name.endswith(".npy")
                else (
                    "application/x-npz" if name.endswith(".npz") else "application/json"
                )
            )
        )

    return dict(
        files={
            name: store.put(value, media_type=media(name))
            for name, value in files.items()
        },
        metadata=meta,
    )


def freeze(host, store):
    """Live-only source allowlist: indexed masks and immutable grounding/grasp query stores."""
    config = configuration(host)
    config["execution"] = execution_configuration(host)
    from ora.agent.toolkit.host.motion import configuration as motion_configuration

    config["motion"] = motion_configuration(host)
    control = host._current()[0]
    config["mark_point"] = {"resolution": list(host._derived_resolution()),
                             "step": (control.get("policy_state") or {}).get("step")}
    if host.surface.supports(ATOMIC_CAPABILITY):
        from ora.agent.toolkit.host.observe import atomic_configuration

        config["atomic"] = atomic_configuration(host)
    current = control["observation_id"]
    from ora.agent.toolkit.host.visibility import artifact_observations
    visible = artifact_observations(host, control)
    config.update(current_target_refs=[], grounding_query_refs=[], grasp_query_refs=[], segmentation_query_refs=[])
    inputs = {}
    root = host.artifact_root
    for path in sorted((root / "segmented-target-index").glob("target-*.json")):
        key = path.stem
        index = json.loads(path.read_bytes())
        if visible is not None and index.get("observation_id") not in visible:
            continue
        if index.get("observation_id") == current:
            config["current_target_refs"].append(key)
        try:
            target, mask, _ = load_segmented_target(artifact_root=root, target_ref=key)
            value = read_artifact(store, "target", key, mask.parent)
        except ValueError as error:
            value = dict(
                error=target_refusals.unavailable_target(
                    error, target_ref=key, artifact_root=root, observation_id=current
                ).as_dict()
            )
        inputs[artifact_key("target", key)] = value
    for kind, directory, field in [
        ("grounding_query", "grounding-queries", "grounding_query_refs"),
        ("grasp_query", "grasp-queries", "grasp_query_refs"),
        ("segmentation_query", "segmentation-batches", "segmentation_query_refs"),
    ]:
        for path in sorted((root / directory).glob("*/manifest.json")):
            key = path.parent.name
            if visible is not None:
                identity = json.loads(path.read_bytes()).get("identity", {})
                if identity.get("observation_id") not in visible:
                    continue
            try:
                value = read_artifact(store, kind, key, path.parent)
            except ValueError as error:
                value = dict(
                    error=service_refusals.refusal(
                        error,
                        code=(
                            "grounding_failed"
                            if kind == "grounding_query"
                            else "segmentation_failed" if kind == "segmentation_query"
                            else "grasp_proposal_failed"
                        ),
                    ).as_dict()
                )
            inputs[artifact_key(kind, key)] = value
            config[field].append(key)
    return config, inputs


def index_record(context, config, inputs):
    """Append-only boundary input index; all source members have already been copied to CAS."""
    return dict(
        schema="ora.toolkit.tool-inputs.v1",
        tool_call_id=context.json()["tool_call_id"],
        context=context.json(),
        configuration=config,
        inputs=inputs,
    )
