"""Trusted assembly for the opt-in toolkit interface; old dispatch is unchanged.

This module sits above toolkit and the existing tool families. It wraps their
clients and control boundary without moving any worker or executor code.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.journal import Journal
from ora.agent.toolkit.interface.projections import alignment, control_result, kinematics, motion_probe, reach
from ora.agent.toolkit.interface.runtime import Runtime
from ora.agent.toolkit.interface.services import grasp_adapter, sam3_adapters
from ora.agent.toolkit.interface.types import AccessDenied, CallContext, NotRecorded, PublicData

from .publication import published_manifest


def bind_runtime(*, runtime, context: CallContext, store: BlobStore, journal: Journal,
                 bindings: Mapping[str, dict], sam3=None, grasp=None,
                 completed_receipts: Mapping[str, PublicData] | None = None,
                 submit=None, runtime_type=Runtime) -> Runtime:
    """Bind one parent call to the existing runtime's validated observation.

    ``submit`` must be the existing validated proposal/skill-sequence entry,
    supplied by the tool adapter. No unchecked write-to-control shortcut is
    provided. Completed receipts are public projections from the selected
    boundary only, supplied by trusted assembly, never discovered by scanning.
    """
    from .state import current
    from .lifecycle import check_adapter

    def publication():
        observed, _, _ = current(runtime)
        root, manifest, manifest_ref = published_manifest(store, observed)
        published = dict(observation=observed, manifest=manifest_ref, files=[])
        # Only manifest-declared camera files plus the publication's proprioception.
        paths = [(root / "proprioception.json", None)]
        for camera in manifest.get("cameras", {}).values():
            for source in camera.get("files", {}).values():
                paths.append((root / source["path"], source))
        for path, declared in paths:
            if not path.is_file():
                continue  # Missing optional camera pixels stay absent, never synthesized.
            mime = "image/png" if path.suffix == ".png" else "application/x-npy" if path.suffix == ".npy" else "application/json"
            ref = store.reference(path.relative_to(store.root).as_posix(), media_type=mime)
            if declared is not None and any(ref[key] != declared[key] for key in ("sha256", "bytes")):
                raise AccessDenied("published camera artifact hash drift")
            published["files"].append(ref)
        return observed, published, manifest_ref

    initial, _, _ = current(runtime)
    if initial != context.json()["state"].get("observation"):
        raise AccessDenied("context is not the currently published observation")
    receipts = completed_receipts or {}

    def read_observation(parameters, inputs, target_store):
        observed, published, manifest_ref = publication()
        if parameters["observation"] != observed:
            raise NotRecorded("unbound observation; select its published boundary explicitly")
        return PublicData.from_json(published, sources=[manifest_ref])

    def read_proprioception(parameters, inputs, target_store):
        published = read_observation(parameters, inputs, target_store)
        import json
        published = json.loads(published.data)
        refs = [ref for ref in published["files"] if ref["path"].endswith("/proprioception.json")]
        if len(refs) != 1:
            raise NotRecorded("no unique published proprioception")
        return PublicData.from_json(dict(proprioception=refs[0]), sources=refs)

    def read_receipt(parameters, inputs, target_store):
        from ora.agent.toolkit.interface.projections import receipt
        value = runtime.toolkit_state_access.values().get("exec.last_receipt")
        if isinstance(value, dict) and value.get("action_id") == parameters["action_id"]:
            return receipt(value)
        if parameters["action_id"] in receipts:
            return receipts[parameters["action_id"]]
        raise NotRecorded("receipt was not completed at this boundary")

    def callback(name, projector):
        def invoke(parameters, inputs, target_store):
            observed, _, _ = current(runtime)  # Closed observations cannot be probed.
            request = parameters["request"]
            if name == "motion_probe_callback" and (
                request.get("observation_id") != observed["observation_id"]
                or request.get("observation_sha256") != observed["observation_sha256"]
            ):
                raise AccessDenied("motion probe observation binding differs")
            function = getattr(runtime, name)
            if function is None:
                raise NotRecorded("backend does not support this callback")
            return projector(function(request))
        return invoke

    def submit_action(parameters, inputs, target_store):
        if submit is None:
            raise AccessDenied("no validated action submission capability bound")
        runtime._current()
        return control_result(submit(parameters["action"]))

    method_bindings = dict(bindings)
    from ora.agent.model_services.clients import SAM3_MODEL_IDENTITY, CONTACT_GRASPNET_MODEL_IDENTITY
    for client, names, model in (
        (sam3, ("segment", "segment_points", "ground"), SAM3_MODEL_IDENTITY),
        (grasp, ("grasp_candidates",), CONTACT_GRASPNET_MODEL_IDENTITY),
    ):
        if client is not None:
            config = dict(client.configuration()) if hasattr(client, "configuration") else {}
            config.pop("endpoint", None)
            identity = dict(backend=getattr(client, "backend_id", names[0]),
                            model_identity=dict(model), service_identity=getattr(client, "expected_binding", None),
                            configuration_identity=config)
            for name in names:
                method_bindings.setdefault(name, identity)

    adapters = dict(
        observation=read_observation, proprioception=read_proprioception, receipt=read_receipt,
        probe_motion=callback("motion_probe_callback", motion_probe),
        align_plan=callback("align_plan_callback", alignment), reach_check=callback("align_plan_callback", reach),
        kinematics=callback("align_plan_callback", kinematics),
        check_task=check_adapter(runtime),
        submit_action=submit_action,
        report_issue=lambda p, i, s: control_result(runtime._report_issue(p["issue"])),
        finish_episode=lambda p, i, s: control_result(runtime._finish(p["request"])),
    )
    if sam3 is not None:
        adapters.update(sam3_adapters(sam3))
    if grasp is not None:
        def propose(parameters, scene, target):
            same_settings = (grasp.raw_candidate_limit == parameters["max_candidates"]
                             and grasp.forward_passes == parameters["forward_passes"])
            client = grasp if same_settings else replace(
                grasp, raw_candidate_limit=parameters["max_candidates"], forward_passes=parameters["forward_passes"])
            try:
                return client.propose(scene_points_camera_m=scene, target_points_camera_m=target,
                                      camera_to_world=parameters["camera_to_world"], target_ref="toolkit-raw-pool",
                                      workspace_bounds_m=parameters["workspace_bounds_m"], max_candidates=parameters["max_candidates"])
            finally:
                if client is not grasp:
                    client._client.close()
        adapters["grasp_candidates"] = grasp_adapter(propose)
    def current_context():
        observed, _, _ = current(runtime)
        return CallContext.create(**dict(context.json(), state={**context.json()["state"], "observation": observed}))

    return runtime_type(context=context, store=store, journal=journal, adapters=adapters,
                        bindings=method_bindings, context_provider=current_context)
