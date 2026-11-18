"""Complete CGN candidate pools and parameterized pose conversion, without selection policy."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from ..interface.api import DataInterface
from .blobs import read_array, save_array
from .values import Array, JsonValue, PointCloudM, PoseM, PublicResult, TransformM


@dataclass(frozen=True)
class GraspCandidates:
    """All N aligned candidates: camera poses Nx4x4 (m), scores N, contacts Nx3 (m), openings N (m)."""
    frame: str
    poses_m: Array
    scores: Array
    contact_points_m: Array
    openings_m: Array
    metadata: JsonValue
    result: PublicResult

    def __post_init__(self):
        n=len(self.scores.numpy())
        if (not self.frame or self.poses_m.shape!=(n,4,4) or self.scores.shape!=(n,)
                or self.contact_points_m.shape!=(n,3) or self.openings_m.shape!=(n,)
                or not all(np.isfinite(a.numpy()).all() for a in
                           (self.poses_m,self.scores,self.contact_points_m,self.openings_m))
                or np.any(self.openings_m.numpy()<0)):
            raise ValueError('CGN pool arrays must align and contain finite metric values')


def candidates(api: DataInterface, *, scene: PointCloudM, target: PointCloudM,
               camera_to_world: TransformM, workspace_bounds_m: JsonValue,
               max_candidates: int = 32, forward_passes: int = 1) -> GraspCandidates:
    """Replay: re-request CGN; new point clouds work, history only supplies comparison.

    Example: candidates(api, scene=scene, target=target, camera_to_world=T,
    workspace_bounds_m=JsonValue.of({'x':[-1,1],'y':[-1,1],'z':[0,2]})). Inputs are
    explicitly cast to float32 C-order, matching the client. Expose all 1–32 requested
    backend candidates, source order, no compiler/ranking/filter. Frame mismatch fails.
    Service failure propagates; one forward pass is the current service contract.
    """
    if (scene.frame!=target.frame or scene.frame!=camera_to_world.source_frame
            or camera_to_world.target_frame!='world'):
        raise ValueError('scene, target and transform must share one camera frame')
    scene_ref=save_array(api,Array.of(np.ascontiguousarray(scene.xyz_m.numpy(),dtype=np.float32)))
    target_ref=save_array(api,Array.of(np.ascontiguousarray(target.xyz_m.numpy(),dtype=np.float32)))
    result=PublicResult.of(api.perception.grasp_candidates(scene=scene_ref.json(),target=target_ref.json(),
        camera_to_world=camera_to_world.matrix.numpy().tolist(),workspace_bounds_m=workspace_bounds_m.json(),
        max_candidates=max_candidates,forward_passes=forward_passes))
    data=result.data.json()
    arrays=data['arrays']
    return GraspCandidates(scene.frame,*(read_array(api,JsonValue.of(arrays[k])) for k in
        ('poses_camera','scores','contact_points_camera','openings_m')),JsonValue.of(data['metadata']),result)


def candidate_pose(pool: GraspCandidates, index: int) -> PoseM:
    """Replay: pure computation. Convert one raw CGN grasp frame to xyzw; index/SE(3) errors fail.

    Example: candidate_pose(pool,0). This is not a robot TCP pose; use an explicit
    source-bound grasp→controlled-point transform before submitting any robot target.
    """
    matrix=pool.poses_m.numpy()[index]
    TransformM('grasp',pool.frame,Array.of(matrix))
    q=Rotation.from_matrix(matrix[:3,:3]).as_quat()
    if q[3]<0:q=-q
    return PoseM(pool.frame,tuple(matrix[:3,3]),tuple(q))


def controlled_pose(pool: GraspCandidates, index: int, *, controlled_to_grasp: TransformM,
                    camera_to_world: TransformM) -> PoseM:
    """Replay: pure computation. Compose world←camera←grasp←controlled point, metres/xyzw.

    Example: controlled_pose(pool,0,controlled_to_grasp=descriptor_T,camera_to_world=T).
    Caller supplies the embodiment-bound offset/axes; no gripper guess, candidate selection,
    reach claim, or approach/lift policy. Frame mismatch or non-rigid input raises ValueError.
    """
    if controlled_to_grasp.target_frame!='grasp' or camera_to_world.source_frame!=pool.frame:
        raise ValueError('controlled/grasp/camera frames do not compose')
    raw=TransformM('grasp',pool.frame,Array.of(pool.poses_m.numpy()[index]))
    m=camera_to_world.matrix.numpy()@raw.matrix.numpy()@controlled_to_grasp.matrix.numpy()
    q=Rotation.from_matrix(m[:3,:3]).as_quat()
    if q[3]<0:q=-q
    return PoseM(camera_to_world.target_frame,tuple(m[:3,3]),tuple(q))
