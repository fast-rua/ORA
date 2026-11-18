"""Observe → re-ground → back-project → PCA. No simulator queries or actions.

Run through `ora toolkit replay RUN --at 0 --script PATH --service-report REPORT`.
Args default to the historical tool inputs; camera/text/max_candidates are used.
All candidate geometries are returned. Picking a target is deliberately left to
the experiment. Missing pixels/service or an unrecorded observation fails.
"""
from ora.agent.toolkit.lib.observation import observe, load_camera
from ora.agent.toolkit.lib.perception import ground
from ora.agent.toolkit.lib.geometry import mask_to_cloud, principal_geometry, transform_points
from ora.agent.toolkit.lib.blobs import save_array


def run(api, args):
    observation = observe(api)
    refs = observation.camera(args.get("camera", "head"))
    camera = load_camera(api, refs)
    # Legacy tool recording stores decoded RGB arrays as inputs, not PNG bytes.
    # Use that same public input representation for a comparable request hash.
    segmentation = ground(api, rgb=save_array(api, camera.rgb), text=args["text"],
                          max_candidates=args.get("max_candidates", 3))
    geometries = []
    for candidate in segmentation.candidates:
        cloud = transform_points(mask_to_cloud(camera, candidate.mask), camera.camera_to_world)
        count = len(cloud.xyz_m.numpy())
        geometry = principal_geometry(cloud) if count else None
        geometries.append(dict(candidate=candidate.fields.json(), points=count, frame=cloud.frame,
                               center_m=None if geometry is None else geometry.center_m.numpy().tolist(),
                               extents_m=None if geometry is None else geometry.extents_m.numpy().tolist(),
                               axes=None if geometry is None else geometry.axes.numpy().tolist(),
                               ambiguous_axes=None if geometry is None else geometry.ambiguous_axes))
    return dict(observation=observation.identity.json(), geometries=geometries,
                comparison=None if segmentation.result.comparison is None else segmentation.result.comparison.json())
