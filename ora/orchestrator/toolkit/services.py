"""Bind existing perception clients to an explicitly supplied readiness report."""
from contextlib import contextmanager, ExitStack
from dataclasses import replace

from ora.agent.model_services.clients import (
    Sam3ApiSegmenter, SAM3_MODEL_IDENTITY, CONTACT_GRASPNET_MODEL_IDENTITY,
)
from ora.agent.toolkit.interface.services import sam3_adapters, grasp_adapter
from ora.agent.toolkit.host.grasp.contact_graspnet_client import ContactGraspNetApiProposer
from ora.orchestrator.spec.inputs import service_prerequisite


@contextmanager
def service_session(report):
    """No report means no service capability. Never discover/start/allocate a service.

    Report validation and endpoint configuration follow `ora spec robodojo
    compose --service-report`. Clients verify live identity when actually called.
    HTTP handles are closed after the script, including failed experiments.
    """
    if report is None:
        yield {}, {}
        return
    prerequisite = service_prerequisite(report, "robodojo")
    children = prerequisite["services"]
    with ExitStack() as stack:
        sam = Sam3ApiSegmenter(endpoint=children["segmentation"]["endpoint"],
                              expected_binding=children["segmentation"])
        stack.callback(sam._client.close)
        grasp = ContactGraspNetApiProposer(endpoint=children["grasp_proposals"]["endpoint"],
                                           expected_binding=children["grasp_proposals"],
                                           hover_offset_m=0.04)  # Default RoboDojo grasp_plan client configuration.
        stack.callback(grasp._client.close)
        bindings = {}
        for client, methods, model in ((sam, ("segment", "segment_points", "ground"), SAM3_MODEL_IDENTITY),
                                       (grasp, ("grasp_candidates",), CONTACT_GRASPNET_MODEL_IDENTITY)):
            configuration = dict(client.configuration())
            configuration.pop("endpoint", None)
            identity = dict(backend=client.backend_id, service_identity=client.expected_binding,
                            model_identity=dict(model), configuration_identity=configuration)
            bindings.update({name: identity for name in methods})

        def propose(parameters, scene, target):
            # The legacy client also compiles poses; the interface projector
            # exposes only its complete canonical raw pool, not that selection.
            with ExitStack() as request_stack:
                client = grasp
                if (grasp.raw_candidate_limit != parameters["max_candidates"]
                        or grasp.forward_passes != parameters["forward_passes"]):
                    client = replace(grasp, raw_candidate_limit=parameters["max_candidates"],
                                     forward_passes=parameters["forward_passes"])
                    request_stack.callback(client._client.close)
                return client.propose(scene_points_camera_m=scene, target_points_camera_m=target,
                                      camera_to_world=parameters["camera_to_world"], target_ref="toolkit-raw-pool",
                                      workspace_bounds_m=parameters["workspace_bounds_m"], max_candidates=parameters["max_candidates"])

        yield dict(sam3_adapters(sam), grasp_candidates=grasp_adapter(propose)), bindings
