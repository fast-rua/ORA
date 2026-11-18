"""Published RGB-D and measured robot state. Replay never samples a new simulator state."""
from dataclasses import dataclass
import numpy as np

from ..interface.api import DataInterface
from ..interface.types import CorruptRecord, NotRecorded
from . import _rigid
from .blobs import read_array, read_json
from .values import Array, JsonValue, PoseM, PublicResult, TransformM


@dataclass(frozen=True)
class CameraRefs:
    """One published camera's public files; missing files stay None, never guessed."""
    name: str
    rgb: JsonValue | None
    depth_m: JsonValue | None
    calibration: JsonValue | None
    declaration: JsonValue


@dataclass(frozen=True)
class CameraPixels:
    """One camera's recorded RGB, depth and calibration file, with no pinhole or rigid-pose claim."""
    name: str
    rgb: Array
    depth_m: Array
    calibration: JsonValue


@dataclass(frozen=True)
class CameraRGBD:
    """RGB uint8 HxWx3, optical-axis depth metres HxW, OpenCV K in pixels; +x right,+y down,+z forward."""
    name: str
    rgb: Array
    depth_m: Array
    intrinsic_px: Array
    camera_to_world: TransformM
    calibration: JsonValue

    def __post_init__(self):
        rgb, depth, k = self.rgb.numpy(), self.depth_m.numpy(), self.intrinsic_px.numpy()
        if (rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or depth.shape!=rgb.shape[:2]
                or k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0
                or not np.allclose(k[2],[0,0,1]) or k[0,1]!=0 or k[1,0]!=0):
            raise ValueError('RGB-D dimensions or zero-skew pinhole intrinsic invalid')


@dataclass(frozen=True)
class ArmState:
    """Measured pose in declared pose_frame; arm joints radians; gripper_normalized is not metres."""
    name: str
    pose: PoseM | None
    arm_joint_qpos: tuple[float, ...]
    gripper_normalized: float | None
    public: JsonValue

    def __post_init__(self):
        object.__setattr__(self,'arm_joint_qpos',tuple(self.arm_joint_qpos))

    @property
    def joint_positions_rad(self) -> tuple[float, ...]:
        """Pure computation: unit-explicit view of canonical measured arm_joint_qpos."""
        return self.arm_joint_qpos


@dataclass(frozen=True)
class RobotState:
    """All public proprioception including finger encoders, feedback and command provenance."""
    arms: tuple[ArmState, ...]
    public: JsonValue

    def __post_init__(self):
        object.__setattr__(self,'arms',tuple(self.arms))


def robot_state(value: JsonValue) -> RobotState:
    """Pure computation: type measured fields, retaining all input fields. No guessed pose frame.

    Example: robot_state(observation.manifest.json()['proprioception'] wrapped in JsonValue).
    A missing pose, or one that names no pose_frame (RoboTwin publishes none), stays None and in public.
    """
    rows = []
    for name, arm in value.json().get('end_effectors',{}).items():
        pose, frame = arm.get('pose_xyz_xyzw'), arm.get('pose_frame')
        typed = None if pose is None or frame is None else PoseM(frame,tuple(pose[:3]),tuple(pose[3:]))
        rows.append(ArmState(name,typed,tuple(arm.get('arm_joint_qpos',())),arm.get('gripper'),JsonValue.of(arm)))
    return RobotState(tuple(rows),value)


@dataclass(frozen=True)
class Observation:
    """A published observation, all camera declarations and measured state, with original evidence."""
    identity: JsonValue
    manifest_ref: JsonValue
    manifest: JsonValue
    cameras: tuple[CameraRefs, ...]
    robot: RobotState
    result: PublicResult

    def __post_init__(self):
        object.__setattr__(self,'cameras',tuple(self.cameras))

    def camera(self, name: str) -> CameraRefs:
        """Pure computation; select an explicitly named published camera, or KeyError."""
        return next((c for c in self.cameras if c.name==name), None) or _missing_camera(name)


def _missing_camera(name):
    raise KeyError(name)


def observe(api: DataInterface, *, identity: JsonValue | None = None) -> Observation:
    """Replay: consume recorded observation at cursor; unrecorded request fails, never resamples.

    Example: obs = observe(api); refs = obs.camera('head'). All declarations survive
    cropped evidence; missing pixels fail only when requested with load_camera.
    """
    result = PublicResult.of(api.state.observation(observation=None if identity is None else identity.json()))
    data = result.data.json()
    manifest_ref = JsonValue.of(data['manifest'])
    manifest = read_json(api,manifest_ref)
    doc = manifest.json()
    if doc.get('observation_id') != data['observation']['observation_id']:
        raise CorruptRecord('published observation identity mismatch')
    def find(ref):
        matches = [] if ref is None else [r for r in data['files'] if all(r[k]==ref[k] for k in ('sha256','bytes'))]
        return JsonValue.of(matches[0]) if matches else None
    cameras=[]
    for name, camera in doc.get('cameras',{}).items():
        files=camera.get('files',{})
        cameras.append(CameraRefs(name,find(files.get('rgb_lossless_png')),find(files.get('depth_numpy_m_f32')),
                                  find(files.get('calibration')),JsonValue.of(camera)))
    return Observation(JsonValue.of(data['observation']),manifest_ref,manifest,tuple(cameras),
                       robot_state(JsonValue.of(doc.get('proprioception',{}))),result)


def proprioception(api: DataInterface, *, identity: JsonValue | None = None) -> RobotState:
    """Replay: consume recorded proprioception request, missing fails; radians/metres as ArmState.

    Example: proprioception(api).arms. Does not replace missing data with final robot state.
    """
    result=PublicResult.of(api.state.proprioception(observation=None if identity is None else identity.json()))
    return robot_state(read_json(api,JsonValue.of(result.data.json()['proprioception'])))


def load_pixels(api: DataInterface, refs: CameraRefs) -> CameraPixels:
    """Replay: read granted recorded pixels/calibration, missing fails; no extra cursor call.

    Example: load_pixels(api, obs.camera('front')). For a measurement that checks the
    calibration fields it uses: an uncalibrated station camera has no extrinsic_cv and
    still answers a depth query, and nothing here assumes zero skew or a rigid extrinsic.
    """
    if any(ref is None for ref in (refs.rgb,refs.depth_m,refs.calibration)):
        raise NotRecorded('未记录: camera RGB/depth/calibration was not exported')
    calibration=read_json(api,refs.calibration)
    return CameraPixels(refs.name,read_array(api,refs.rgb),read_array(api,refs.depth_m),calibration)


def load_camera(api: DataInterface, refs: CameraRefs, *, world_frame: str = 'world') -> CameraRGBD:
    """Replay: read granted recorded pixels/calibration, missing fails; no extra cursor call.

    Example: load_camera(api, obs.camera('head')). extrinsic_cv maps world→camera;
    invert it explicitly. Supports 3x4/4x4 rigid extrinsics, zero-skew OpenCV K only.
    """
    pixels=load_pixels(api,refs)
    spec=pixels.calibration.json()
    if spec.get('extrinsic_cv') is None:
        # A real cell's fixed camera before its calibration (the YAM station's): pixels and depth, no pose.
        raise ValueError(f'camera {refs.name} has no extrinsic calibration in this observation; '
                         'use a calibrated camera')
    extrinsic=np.asarray(spec['extrinsic_cv'],dtype=float)
    if extrinsic.shape==(3,4):
        extrinsic=np.vstack([extrinsic,[0,0,0,1]])
    transform=TransformM(world_frame,f'camera/{refs.name}',Array.of(extrinsic))
    return CameraRGBD(refs.name,pixels.rgb,pixels.depth_m,Array.of(np.asarray(spec['intrinsic_cv'],dtype=float)),
                      TransformM(transform.target_frame,world_frame,Array.of(_rigid.invert(extrinsic))),pixels.calibration)
