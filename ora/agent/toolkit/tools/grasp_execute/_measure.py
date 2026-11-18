"""Freeze selected public mask geometry and phrase; no grasp or identity verdict."""
import json
import numpy as np
from ...lib import observation, geometry, vision
from ...lib.blobs import read_array
from ...lib.values import Array, JsonValue, PointCloudM
from ..grasp_plan.tool import target_input
from ..errors import ToolRefusal
from ...interface.types import InterfaceError, PublicToolFailure
from ._localization import localize


def selected_surface(api, obs, target_ref, config):
    try:
        target, record=target_input(api,target_ref,obs,config)
        camera=observation.load_camera(api,obs.camera(target['identity']['camera']))
        camera=observation.CameraRGBD(camera.name,camera.rgb,Array.of(camera.depth_m.numpy().astype(np.float64)),
            camera.intrinsic_px,camera.camera_to_world,camera.calibration)
        mask=read_array(api,JsonValue.of(record['files']['mask.npy']))
        status={}
        if config['execution']['depth_inliers']:
            mask,quality=vision.measured_depth_selection(mask,camera.depth_m)
            fields=quality.json()
            failed=[key for key,minimum in [('depth_valid_fraction',.7),('depth_inlier_fraction',.7),('primary_component_fraction',.6)]
                    if fields[key]<minimum]
            if failed:return None,None
            status=dict(status='measured',quality_failures=[])
        cloud=geometry.transform_points(geometry.mask_to_cloud(camera,mask),camera.camera_to_world)
        value=geometry.visible_surface_geometry(cloud).json()
        value.update(observation_id=obs.identity.json()['observation_id'],target_ref=target_ref,source_camera=camera.name,**status)
        localization=localize(value)
        points=np.asarray(value['voxel_points_world_m'],dtype=float)
        if len(points)<16:return localization,None
        if len(points)>512:points=points[np.linspace(0,len(points)-1,512,dtype=int)]
        return localization,dict(camera=camera.name,points=points.tolist())
    except PublicToolFailure:
        return None,None
    except InterfaceError:
        raise
    except (ToolRefusal,KeyError,ValueError,OSError):
        return None,None


def semantics(api, config, target_ref):
    for key in sorted(config['grounding_query_refs']):
        record=api.artifacts.tool_artifact(kind='grounding_query',key=key)
        if record is None:continue
        manifest=json.loads(api.artifacts.read_blob(record['files']['manifest.json']))
        identity=manifest.get('identity',{})
        for row in manifest.get('candidates',()):
            if str(row.get('target_ref'))!=target_ref:continue
            score=row.get('backend_score')
            return dict(phrase=str(identity.get('text') or '') or None,
                score=float(score) if isinstance(score,(int,float)) else None,rank=row.get('rank'),
                camera=str(identity.get('camera') or '') or None,observation_id=str(identity.get('observation_id') or '') or None,
                selection='grounded_from_a_phrase')
    return dict(phrase=None,score=None,rank=None,camera=None,observation_id=None,selection='not_grounded_from_a_phrase')
