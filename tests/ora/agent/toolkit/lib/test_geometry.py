"""Known-coordinate checks, immutable ownership, and explicit failures for ambiguous data."""
from dataclasses import FrozenInstanceError
import numpy as np
import pytest

from ora.agent.toolkit.lib.geometry import (
    depth_inliers, mask_to_cloud, offset_pose, principal_geometry, transform_points, transform_pose,
)
from ora.agent.toolkit.lib.grasp import candidate_pose, controlled_pose, candidates
from ora.agent.toolkit.lib.observation import observe, load_camera
from ora.agent.toolkit.lib.values import Array, JsonValue, PointCloudM, PoseM, TransformM
from tests.support.agent.toolkit_lib import make_runtime


def test_json_array_and_pose_own_immutable_copies():
    original={'nested':{'values':[1,2]}}
    frozen=JsonValue.of(original)
    original['nested']['values'].append(3)
    frozen.json()['nested']['values'].append(4)
    assert frozen.json()=={'nested':{'values':[1,2]}}
    value=np.asfortranarray(np.arange(12,dtype='>f4').reshape(3,4))
    frozen_array=Array.of(value)
    value[:]=0
    assert frozen_array.order=='F' and frozen_array.dtype=='>f4'
    np.testing.assert_array_equal(frozen_array.numpy(),np.arange(12).reshape(3,4))
    with pytest.raises(ValueError):frozen_array.numpy().setflags(write=True)
    with pytest.raises(ValueError):frozen_array.numpy()[0,0]=3
    position=[1.,2.,3.]
    pose=PoseM('world',position,[0.,0.,0.,1.]);position[0]=5
    assert pose.position_m==(1.,2.,3.)
    with pytest.raises(FrozenInstanceError):pose.frame='camera/head'


@pytest.mark.parametrize('value',[np.array([object()]),np.array([(1,)],dtype=[('x','i4')])])
def test_object_and_structured_arrays_rejected(value):
    with pytest.raises(ValueError):Array.of(value)


@pytest.mark.parametrize('kind',['reflection','scale','bad_bottom','nan','complex'])
def test_nonrigid_or_nonreal_transform_rejected(kind):
    value=np.eye(4,dtype=complex if kind=='complex' else float)
    if kind=='reflection':value[0,0]=-1
    if kind=='scale':value[0,0]=2
    if kind=='bad_bottom':value[3,0]=1
    if kind=='nan':value[0,0]=np.nan
    with pytest.raises(ValueError):TransformM('camera/head','world',Array.of(value))


def test_metric_backprojection_and_frame_composition(tmp_path):
    api,_,_=make_runtime(tmp_path)
    camera=load_camera(api,observe(api).camera('head'))
    mask=np.zeros((4,4),dtype=bool);mask[1,1]=True;mask[1,3]=True
    cloud=mask_to_cloud(camera,Array.of(mask))
    np.testing.assert_allclose(cloud.xyz_m.numpy(),[[0,0,1],[1,0,1]])
    world=transform_points(cloud,camera.camera_to_world)
    np.testing.assert_allclose(world.xyz_m.numpy(),[[2,0,1],[3,0,1]])
    pose=PoseM('camera/head',(0,0,1),(0,0,0,1))
    transformed=transform_pose(pose,camera.camera_to_world)
    assert transformed.position_m==(2.,0.,1.)
    assert offset_pose(transformed,delta_m=(0,0,.2),frame='world').position_m==(2.,0.,1.2)
    with pytest.raises(ValueError):transform_points(world,camera.camera_to_world)
    with pytest.raises(ValueError):transform_pose(transformed,camera.camera_to_world)
    with pytest.raises(ValueError):offset_pose(transformed,delta_m=(0,0,1),frame='camera/head')
    assert mask_to_cloud(camera,Array.of(np.zeros((4,4),dtype=bool))).xyz_m.shape==(0,3)


def test_depth_gate_retains_diagnostics_and_no_invented_points():
    mask=Array.of(np.ones((5,5),dtype=bool))
    depth=np.ones((5,5));depth[0,0]=np.nan;depth[0,1]=0;depth[0,2]=2
    result=depth_inliers(mask,Array.of(depth))
    assert result.mask.numpy().sum()==22
    assert result.diagnostics.json()['valid_pixels']==23
    assert result.diagnostics.json()['half_width_m']==.03
    assert not depth_inliers(mask,Array.of(depth),minimum_points=30).mask.numpy().any()
    assert not depth_inliers(mask,Array.of(np.zeros((5,5)))).mask.numpy().any()
    with pytest.raises(ValueError):depth_inliers(mask,Array.of(np.ones((4,4))))


def test_principal_axes_known_shape_and_degeneracy():
    points=np.array([[x,y,z] for x in (-3,3) for y in (-2,2) for z in (-1,1)],dtype=float)
    result=principal_geometry(PointCloudM('world',Array.of(points)))
    np.testing.assert_allclose(result.center_m.numpy(),[0,0,0])
    np.testing.assert_allclose(result.axes.numpy(),np.eye(3))
    np.testing.assert_allclose(result.extents_m.numpy(),[6,4,2])
    assert not result.ambiguous_axes
    cube=PointCloudM('world',Array.of(points/np.array([3,2,1])))
    assert principal_geometry(cube).ambiguous_axes
    with pytest.raises(ValueError):principal_geometry(PointCloudM('world',Array.of(np.empty((0,3)))))
    with pytest.raises(ValueError):PointCloudM('world',Array.of([[np.nan,0,0]]))


def test_full_pool_controlled_point_offset_has_no_selection_policy(tmp_path):
    api,_,_=make_runtime(tmp_path)
    camera=load_camera(api,observe(api).camera('head'))
    cloud=mask_to_cloud(camera,Array.of(np.ones((4,4),dtype=bool)))
    pool=candidates(api,scene=cloud,target=cloud,camera_to_world=camera.camera_to_world,
                    workspace_bounds_m=JsonValue.of({'x':[-5,5],'y':[-5,5],'z':[0,5]}),max_candidates=8)
    assert candidate_pose(pool,2).position_m==(.02,0.,0.)
    offset=np.eye(4);offset[2,3]=.05
    tcp=controlled_pose(pool,2,controlled_to_grasp=TransformM('tcp/left','grasp',Array.of(offset)),
                        camera_to_world=camera.camera_to_world)
    assert tcp.position_m==(2.02,0.,.05) and tcp.frame=='world'
    assert len(pool.scores.numpy())==8
    with pytest.raises(IndexError):candidate_pose(pool,8)


def test_approach_orientation_maps_declared_axes_and_rejects_degenerate_inputs():
    from ora.agent.toolkit.lib.pose import approach_orientation
    from ora.agent.toolkit.lib.values import JsonValue
    from scipy.spatial.transform import Rotation
    import pytest
    axes = JsonValue.of({"approach": [1., 0., 0.], "closing": [0., 1., 0.]})
    for direction in ((0., 0., -1.), (1., 0., 0.), (0., 1., 0.)):
        for roll in (0., .3, 1.2):
            quaternion = approach_orientation(direction, roll, local_axes=axes)
            np.testing.assert_allclose(Rotation.from_quat(quaternion).apply([1., 0., 0.]), direction, atol=1e-14)
    with pytest.raises(ValueError, match="finite nonzero"):
        approach_orientation((0., 0., 0.), 0., local_axes=axes)
    with pytest.raises(ValueError, match="orthogonal"):
        approach_orientation((1., 0., 0.), 0., local_axes=JsonValue.of({"approach": [1., 0., 0.], "closing": [1., 0., 0.]}))
