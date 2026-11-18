"""Pure, deterministic metric geometry; no I/O, implicit randomness or task policy."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from . import _rigid
from ._principal_axes import principal_axes
from .observation import CameraRGBD
from .values import Array, JsonValue, PointCloudM, PoseM, TransformM


def transform_points(cloud: PointCloudM, transform: TransformM) -> PointCloudM:
    """Replay: pure computation. Apply source→target SE(3), metres; frame mismatch raises ValueError.

    Example: transform_points(camera_cloud, camera.camera_to_world).
    """
    if cloud.frame!=transform.source_frame:
        raise ValueError('point cloud and transform source frames differ')
    matrix=transform.matrix.numpy()
    return PointCloudM(transform.target_frame,Array.of(_rigid.apply(cloud.xyz_m.numpy(),matrix)))


def mask_to_cloud(camera: CameraRGBD, mask: Array, *, min_depth_m: float = 1e-6,
                  max_depth_m: float = 2.5) -> PointCloudM:
    """Replay: pure computation. Back-project visible mask pixels in row-major order, camera metres.

    Example: mask_to_cloud(camera, segmentation.candidates[0].mask). Drop nonfinite,
    nonpositive/out-of-range depth; empty input gives (0,3), not invented points.
    Shape/range mismatch raises ValueError. No downsample or hidden candidate preference.
    """
    depth=camera.depth_m.numpy()
    selected=mask.numpy()
    if selected.shape!=depth.shape or not 0<=min_depth_m<max_depth_m or not np.isfinite(max_depth_m):
        raise ValueError('mask dimensions/depth limits invalid')
    valid=selected.astype(bool)&np.isfinite(depth)&(depth>min_depth_m)&(depth<=max_depth_m)
    rows,columns=np.nonzero(valid)
    z=depth[rows,columns]
    k=camera.intrinsic_px.numpy()
    xyz=np.stack(((columns-k[0,2])*z/k[0,0],(rows-k[1,2])*z/k[1,1],z),axis=1)
    return PointCloudM(camera.camera_to_world.source_frame,Array.of(xyz))


@dataclass(frozen=True)
class DepthSelection:
    """Selected HxW pixels and all counts/thresholds; geometry quality is not object truth."""
    mask: Array
    diagnostics: JsonValue


def depth_inliers(mask: Array, depth_m: Array, *, min_depth_m: float = 1e-6,
                  max_depth_m: float = 2.5, minimum_points: int = 16,
                  minimum_half_width_m: float = .03, mad_scale: float = 4.*1.4826) -> DepthSelection:
    """Replay: pure computation. Largest 8-connected valid component, then median/MAD depth gate.

    Example: depth_inliers(mask, camera.depth_m). Parameters expose the old localization
    algorithm without task selection. Too few pixels returns empty mask plus counts;
    malformed dimensions or thresholds raise ValueError. Ties follow row-major labels.
    """
    import cv2
    selected=mask.numpy().astype(bool)
    depth=depth_m.numpy()
    if (selected.ndim!=2 or selected.shape!=depth.shape or not selected.size
            or not 0<=min_depth_m<max_depth_m or minimum_points<1 or minimum_half_width_m<0 or mad_scale<0
            or not np.isfinite([max_depth_m,minimum_half_width_m,mad_scale]).all()):
        raise ValueError('invalid mask/depth selection parameters')
    valid=selected&np.isfinite(depth)&(depth>min_depth_m)&(depth<=max_depth_m)
    count,labels=cv2.connectedComponents(valid.astype(np.uint8),connectivity=8)
    sizes=np.bincount(labels.ravel(),minlength=count)
    sizes[0]=0
    primary=labels==int(np.argmax(sizes)) if count>1 else np.zeros_like(valid)
    result=np.zeros_like(valid)
    median=mad=half=None
    if primary.sum()>=minimum_points:
        median=float(np.median(depth[primary]))
        mad=float(np.median(np.abs(depth[primary]-median)))
        half=max(minimum_half_width_m,mad_scale*mad)
        result=primary&(np.abs(depth-median)<=half)
    return DepthSelection(Array.of(result),JsonValue.of(dict(mask_pixels=int(selected.sum()),
        valid_pixels=int(valid.sum()),components=count-1,primary_pixels=int(primary.sum()),
        inlier_pixels=int(result.sum()),median_depth_m=median,mad_depth_m=mad,half_width_m=half,
        min_depth_m=min_depth_m,max_depth_m=max_depth_m,minimum_points=minimum_points,
        minimum_half_width_m=minimum_half_width_m,mad_scale=mad_scale,connectivity=8)))


@dataclass(frozen=True)
class PrincipalGeometry:
    """Visible cloud median/AABB/PCA; axes are rows in frame, extents metres, variance m²."""
    frame: str
    center_m: Array
    lower_m: Array
    upper_m: Array
    axes: Array
    extents_m: Array
    variance_m2: Array
    ambiguous_axes: bool


def principal_geometry(cloud: PointCloudM, *, lower_percentile: float = 2.5,
                       upper_percentile: float = 97.5, degeneracy_atol_m2: float = 1e-12) -> PrincipalGeometry:
    """Replay: pure computation. Median-centered PCA ordered by variance; signs by dominant component.

    Example: principal_geometry(world_cloud). Empty clouds/invalid percentiles fail.
    Repeated eigenvalues flag ambiguous_axes: do not treat their axes as stable object
    orientation. Percentile extents remain paired with variance-ordered axes, not sorted by size.
    """
    points=cloud.xyz_m.numpy()
    if not len(points) or not 0<=lower_percentile<upper_percentile<=100 or not np.isfinite(degeneracy_atol_m2) or degeneracy_atol_m2<0:
        raise ValueError('nonempty cloud and valid percentile/degeneracy limits required')
    center=np.median(points,axis=0)
    centered=points-center
    values,vectors=principal_axes(centered.T@centered/max(1,len(points)-1))
    axes=vectors.T.copy()
    for axis in axes:
        if axis[np.argmax(np.abs(axis))]<0: axis*=-1
    lo,hi=np.percentile(points,[lower_percentile,upper_percentile],axis=0)
    a,b=np.percentile(centered@axes.T,[lower_percentile,upper_percentile],axis=0)
    return PrincipalGeometry(cloud.frame,Array.of(center),Array.of(lo),Array.of(hi),Array.of(axes),
        Array.of(b-a),Array.of(values),bool(np.any(np.abs(np.diff(values))<=degeneracy_atol_m2)))


def transform_pose(pose: PoseM, transform: TransformM) -> PoseM:
    """Replay: pure computation. Compose parent transforms; metres and xyzw. Mismatched frames fail.

    Example: transform_pose(camera_pose, camera.camera_to_world). Quaternion sign is w>=0.
    """
    if pose.frame!=transform.source_frame:
        raise ValueError('pose and transform source frames differ')
    matrix=transform.matrix.numpy()
    p=matrix[:3,:3]@pose.position_m+matrix[:3,3]
    q=(Rotation.from_matrix(matrix[:3,:3])*Rotation.from_quat(pose.orientation_xyzw)).as_quat()
    if q[3]<0:q=-q
    return PoseM(transform.target_frame,tuple(p),tuple(q))


def offset_pose(pose: PoseM, *, delta_m: tuple[float,float,float], frame: str) -> PoseM:
    """Replay: pure computation. Translate in the explicit parent frame; mismatch/nonfinite fails.

    Example: offset_pose(world_pose, delta_m=(0,0,.1), frame='world'). The caller chooses
    direction/distance; this is not a lift strategy or a sequence/action submission.
    """
    if frame!=pose.frame or np.asarray(delta_m).shape!=(3,):
        raise ValueError('translation requires matching frame and three coordinates')
    return PoseM(frame,tuple(np.asarray(pose.position_m)+delta_m),pose.orientation_xyzw)


def visible_surface_geometry(cloud: PointCloudM, *, voxel_size_m: float = .01) -> JsonValue:
    """Pure computation: mean-centered PCA and stable rounded-grid samples, world metres.

    Example: visible_surface_geometry(mask_cloud_in_world). Extents are full
    measured ranges in variance order; no hidden shape completion. At least 16
    finite points are required. Voxel sampling keeps the first source point.
    """
    points=cloud.xyz_m.numpy().astype(np.float64)
    if cloud.frame != 'world' or len(points)<16 or not np.isfinite(points).all() or not voxel_size_m>0:
        raise ValueError('segmented target world geometry is invalid')
    centroid=np.mean(points,axis=0)
    covariance=np.cov((points-centroid).T)
    eigenvalues,eigenvectors=principal_axes(covariance)
    axes=eigenvectors.T
    for axis in axes:
        if axis[int(np.argmax(np.abs(axis)))]<0:axis*=-1.
    projections=(points-centroid)@axes.T
    extents=np.max(projections,axis=0)-np.min(projections,axis=0)
    quantized=np.round(points/float(voxel_size_m)).astype(np.int64)
    _,indices=np.unique(quantized,axis=0,return_index=True)
    return JsonValue.of(dict(centroid_world_m=centroid.tolist(),principal_axes_world=axes.tolist(),
        principal_variances_m2=eigenvalues.tolist(),principal_extents_m=extents.tolist(),
        visible_point_count=len(points),voxel_size_m=float(voxel_size_m),voxel_points_world_m=points[np.sort(indices)].tolist()))
