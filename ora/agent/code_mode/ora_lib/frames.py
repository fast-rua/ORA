"""Values that carry their frame, one conversion (``x.to(frame)``), and helpers for the common spatial steps.

Frames (``docs/robots/frames-and-units.md``):

- ``world``: the robot's world frame, metres;
- ``tcp:<arm>``: the arm's measured end-effector frame (the point a pose command moves, with its local axes:
  the approach axis is local ``+Z`` on Franka Panda, ``+X`` on ALOHA and ARX X5);
- ``camera:<name>``: a camera's OpenCV frame, ``x`` right, ``y`` down, ``z`` forward (the optical axis);
- ``pixel:<name>``: that camera's image, ``u`` right and ``v`` down from the top-left, native pixels.

Values: ``Point(xyz, frame=...)`` (one ``(3,)`` or a batch ``(N, 3)``), ``Pixel(uv, camera=...)``,
``Pose(xyz, xyzw, frame=...)`` and ``Direction(v, frame=...)`` (a free vector: a direction or a
displacement). Lengths are metres and angles radians; a name says when it is otherwise (``angle_deg``).

Conversions use the current published observation (``observation.current()``): its calibration for cameras,
its measured end-effector poses for ``tcp:<arm>``, its depth to lift a pixel. A ``tcp:<arm>`` or wrist-camera
value is relative to where that arm or camera is in the current observation, so convert it to ``world``
before an action moves them. Combining values of different frames raises ``FrameError``, which names the
conversion; so does adding a bare list, whose frame would be a guess. NumPy reads a ``Point``'s or a ``Direction``'s
numbers (``np.asarray(p)``, ``np.round(d, 3)``, the frame stays behind), but a ufunc over one (``np.add(p, q)``) raises.

    from ora_lib import Point, Pixel, Direction, offset, direction, pixels_to_world, project
    cup_world = pixels_to_world("agentview", [[128, 140]])[0]        # depth median over a 7x7 window
    above = cup_world + Direction([0, 0, 0.10], frame="world")
    above.to("pixel:agentview")                                      # where it shows in the image
    left = direction("left", view="agentview", horizontal=True)      # image-left, along the table
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from . import geometry
from . import observation as _observation

#: Unit image directions in the camera frame: left/right along ``x``, up/down along ``y`` (OpenCV).
IMAGE_DIRECTIONS = {"left": (-1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0), "up": (0.0, -1.0, 0.0),
                    "down": (0.0, 1.0, 0.0)}


class FrameError(ValueError):
    """Values of different frames combined, or a conversion that does not exist."""


def _space_frame(frame: Any, kind: str) -> str:
    """``frame`` checked for a value that lives in 3-D space (not ``pixel:``)."""

    if not isinstance(frame, str) or not frame:
        raise FrameError(f"a {kind} needs a frame: world, tcp:<arm> or camera:<name>")
    name = frame.partition(":")[2]
    if frame == "world" or (frame.startswith(("tcp:", "camera:")) and name):
        return frame
    if frame.startswith("pixel:"):
        raise FrameError(f"a {kind} lives in world, tcp:<arm> or camera:<name>; a pixel is Pixel(uv, camera=...)")
    raise FrameError(f"unknown frame {frame!r}: frames are world, tcp:<arm>, camera:<name> and pixel:<name>")


def _camera_name(camera: Any) -> str:
    if isinstance(camera, _observation.Camera):
        return camera.name
    name = str(camera)
    return name.split(":", 1)[1] if name.startswith(("camera:", "pixel:")) else name


def _scene(observation: _observation.Observation | None) -> _observation.Observation:
    return observation if observation is not None else _observation.current()


def _transform(target: str, source: str, observation: _observation.Observation | None) -> np.ndarray:
    if target == source:
        return np.eye(4)
    return _scene(observation).transform(target, source)


def _mismatch(a: Any, b: Any, a_name: str, b_name: str) -> FrameError:
    return FrameError(
        f"cannot combine a {type(a).__name__} in {a.frame} with a {type(b).__name__} in {b.frame}: convert one "
        f"first, e.g. {b_name}.to({a.frame!r}) or {a_name}.to({b.frame!r})")


def _bare(value: Any, kind: str, frame: str) -> TypeError:
    return TypeError(f"a bare {type(value).__name__} has no frame; say which, e.g. Direction(v, frame={frame!r}) "
                     f"to move a {kind}")


def _in_frame(value: Any, frame: str, label: str) -> None:
    """A constructor's ``value`` (or a list of them) is bare numbers or a Point or Direction in ``frame``: reading
    another frame's numbers (``__array__``) would drop that frame."""

    for item in value if isinstance(value, (list, tuple)) else (value,):
        if isinstance(item, (Point, Direction)) and item.frame != frame:
            raise FrameError(f"{label} is in {item.frame}, not {frame}: convert it first, e.g. value.to({frame!r})")


def _vectors(value: Any, label: str, frame: str) -> np.ndarray:
    _in_frame(value, frame, label)
    a = np.array(value, dtype=np.float64)
    if a.ndim not in (1, 2) or a.shape[-1] != 3:
        raise ValueError(f"{label} must be 3 numbers or an (N, 3) array; got shape {a.shape}")
    a.setflags(write=False)
    return a


def _summary(a: np.ndarray, width: int) -> str:
    if a.ndim == 1:
        return repr([round(float(x), 6) for x in a])
    return f"<{len(a)} x {width}>"


class Point(_observation.ListsAttributes):
    """A point (``(3,)``) or points (``(N, 3)``) in metres in one frame; NaN marks a point with no value."""

    __slots__ = ("xyz", "frame")
    __array_ufunc__ = None  # numpy defers to our operators, which name the frame

    def __init__(self, xyz: Any, frame: str) -> None:
        self.frame = _space_frame(frame, "Point")
        self.xyz = _vectors(xyz, "a Point's xyz", self.frame)

    def to(self, frame: str, *, observation: _observation.Observation | None = None) -> "Point | Pixel":
        """The same point(s) in ``frame``; ``pixel:<camera>`` projects them (a ``Pixel``, NaN behind it)."""

        if isinstance(frame, str) and frame.startswith("pixel:"):
            camera = _scene(observation).camera(frame)
            in_camera = geometry.transform_points(self.xyz, _transform(f"camera:{camera.name}", self.frame,
                                                                       observation))
            return Pixel(geometry.project(in_camera, camera.K), camera=camera.name)
        frame = _space_frame(frame, "Point")
        return Point(geometry.transform_points(self.xyz, _transform(frame, self.frame, observation)), frame)

    @property
    def valid(self) -> Any:
        """Whether each point has a value (a bool, or a bool array for a batch)."""

        return np.isfinite(self.xyz).all(axis=-1)

    @property
    def x(self) -> Any:
        return self.xyz[..., 0]

    @property
    def y(self) -> Any:
        return self.xyz[..., 1]

    @property
    def z(self) -> Any:
        return self.xyz[..., 2]

    def mean(self) -> "Point":
        """The centroid of a batch, over the points with a value."""

        return Point(np.nanmean(np.atleast_2d(self.xyz), axis=0), self.frame)

    def distance(self, other: "Point") -> Any:
        """The distance in metres to ``other`` (same frame)."""

        return (self - other).norm()

    def __add__(self, other: Any) -> "Point":
        if isinstance(other, Direction):
            if other.frame != self.frame:
                raise _mismatch(self, other, "point", "direction")
            return Point(self.xyz + other.v, self.frame)
        if isinstance(other, Point):
            raise TypeError("adding two Points has no meaning; p - q is a Direction, Point.mean() a centroid")
        raise _bare(other, "Point", self.frame)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "Point | Direction":
        if isinstance(other, Point):
            if other.frame != self.frame:
                raise _mismatch(self, other, "point", "other")
            return Direction(self.xyz - other.xyz, self.frame)
        if isinstance(other, Direction):
            if other.frame != self.frame:
                raise _mismatch(self, other, "point", "direction")
            return Point(self.xyz - other.v, self.frame)
        raise _bare(other, "Point", self.frame)

    def __rsub__(self, other: Any) -> Any:
        raise _bare(other, "Point", self.frame)

    def __len__(self) -> int:
        if self.xyz.ndim == 1:
            raise TypeError("a single Point has no len(); a batch of N points has")
        return len(self.xyz)

    def __getitem__(self, index: Any) -> "Point":
        if self.xyz.ndim == 1:
            raise TypeError("a single Point cannot be indexed; use .xyz, .x, .y, .z")
        return Point(self.xyz[index], self.frame)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """Its ``xyz`` as a new array (``np.asarray(p)``); ``copy`` is NumPy 2's argument, and this always copies."""

        return np.array(self.xyz, dtype=dtype)

    def to_json(self) -> dict[str, Any]:
        return {"frame": self.frame, "xyz": self.xyz.tolist()}

    tolist = to_json  # how the sandbox's JSON encoding (robot.py) sends a value it holds

    def __repr__(self) -> str:
        return f"Point({_summary(self.xyz, 3)}, frame={self.frame!r})"


class Direction(_observation.ListsAttributes):
    """A free vector (``(3,)`` or ``(N, 3)``) in one frame: a direction, or a displacement in metres. ``v`` holds it
    (``xyz`` is the same array, as on a ``Point``)."""

    __slots__ = ("v", "frame")
    __array_ufunc__ = None

    def __init__(self, v: Any, frame: str) -> None:
        self.frame = _space_frame(frame, "Direction")
        self.v = _vectors(v, "a Direction's v", self.frame)

    def to(self, frame: str, *, observation: _observation.Observation | None = None) -> "Direction":
        """The same vector in ``frame`` (rotated only; a direction has no position)."""

        if isinstance(frame, str) and frame.startswith("pixel:"):
            raise FrameError("a Direction has no pixel form; project two points instead: "
                             "project(p, camera) and project(p + d, camera)")
        frame = _space_frame(frame, "Direction")
        return Direction(geometry.rotate_vectors(self.v, _transform(frame, self.frame, observation)), frame)

    @property
    def xyz(self) -> np.ndarray:
        """The same array as ``v``, as a ``Point`` has it."""

        return self.v

    def norm(self) -> Any:
        """The length (metres for a displacement)."""

        return np.linalg.norm(self.v, axis=-1)

    def unit(self) -> "Direction":
        norm = np.linalg.norm(self.v, axis=-1, keepdims=True)
        if np.any(norm < 1e-12):
            raise ValueError("a zero vector has no direction")
        return Direction(self.v / norm, self.frame)

    def _same(self, other: Any) -> "Direction":
        if not isinstance(other, Direction):
            raise _bare(other, "Direction", self.frame) if not isinstance(other, (Point, Pose, Pixel)) else \
                TypeError(f"cannot combine a Direction with a {type(other).__name__} this way")
        if other.frame != self.frame:
            raise _mismatch(self, other, "direction", "other")
        return other

    def __add__(self, other: Any) -> Any:
        if isinstance(other, (Point, Pose)):
            return other + self
        return Direction(self.v + self._same(other).v, self.frame)

    def __sub__(self, other: Any) -> "Direction":
        return Direction(self.v - self._same(other).v, self.frame)

    def __neg__(self) -> "Direction":
        return Direction(-self.v, self.frame)

    def __mul__(self, scale: Any) -> "Direction":
        if isinstance(scale, (Point, Direction, Pose, Pixel)):
            raise TypeError("a Direction scales by a number; use dot() or cross() between Directions")
        return Direction(self.v * np.asarray(scale, dtype=np.float64)[..., None]
                         if np.ndim(scale) else self.v * float(scale), self.frame)

    __rmul__ = __mul__

    def __truediv__(self, scale: Any) -> "Direction":
        return self * (1.0 / np.asarray(scale, dtype=np.float64))

    def dot(self, other: "Direction") -> Any:
        return np.sum(self.v * self._same(other).v, axis=-1)

    def cross(self, other: "Direction") -> "Direction":
        return Direction(np.cross(self.v, self._same(other).v), self.frame)

    def angle_rad(self, other: "Direction") -> Any:
        """The angle to ``other`` (same frame), radians."""

        cosine = self.unit().dot(other.unit())
        return np.arccos(np.clip(cosine, -1.0, 1.0))

    def angle_deg(self, other: "Direction") -> Any:
        """The angle to ``other`` (same frame), degrees."""

        return np.degrees(self.angle_rad(other))

    def __len__(self) -> int:
        if self.v.ndim == 1:
            raise TypeError("a single Direction has no len(); a batch has")
        return len(self.v)

    def __getitem__(self, index: Any) -> "Direction":
        if self.v.ndim == 1:
            raise TypeError("a single Direction cannot be indexed; use .v")
        return Direction(self.v[index], self.frame)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """Its ``v`` as a new array (``np.round(d, 3)``); ``copy`` is NumPy 2's argument, and this always copies."""

        return np.array(self.v, dtype=dtype)

    def to_json(self) -> dict[str, Any]:
        return {"frame": self.frame, "v": self.v.tolist()}

    tolist = to_json

    def __repr__(self) -> str:
        return f"Direction({_summary(self.v, 3)}, frame={self.frame!r})"


class Pose(_observation.ListsAttributes):
    """A position (metres) and an orientation (quaternion ``[x, y, z, w]``, normalized) in one frame."""

    __slots__ = ("xyz", "xyzw", "frame")
    __array_ufunc__ = None

    def __init__(self, xyz: Any, xyzw: Any, frame: str) -> None:
        self.frame = _space_frame(frame, "Pose")
        _in_frame(xyz, self.frame, "a Pose's xyz")
        xyz = np.array(xyz, dtype=np.float64)
        if xyz.shape != (3,):
            raise ValueError(f"a Pose's xyz must be 3 numbers; got shape {xyz.shape}")
        xyzw = geometry.unit_quaternion(xyzw)
        xyz.setflags(write=False)
        xyzw.setflags(write=False)
        self.xyz, self.xyzw = xyz, xyzw

    def to(self, frame: str, *, observation: _observation.Observation | None = None) -> "Pose":
        """The same pose in ``frame`` (its quaternion with ``w >= 0``)."""

        if isinstance(frame, str) and frame.startswith("pixel:"):
            raise FrameError("a Pose has no pixel form; project its position: project(pose.position, camera)")
        frame = _space_frame(frame, "Pose")
        if frame == self.frame:
            return self
        xyz, xyzw = geometry.transform_pose(self.xyz, self.xyzw, _transform(frame, self.frame, observation))
        return Pose(xyz, xyzw, frame)

    @property
    def position(self) -> Point:
        return Point(self.xyz, self.frame)

    @property
    def rotation(self) -> Rotation:
        return Rotation.from_quat(self.xyzw)

    @property
    def matrix(self) -> np.ndarray:
        """The 4x4 ``frame <- pose`` transform."""

        return geometry.pose_matrix(self.xyz, self.xyzw)

    def axis(self, name: str) -> Direction:
        """The pose's own ``x``, ``y`` or ``z`` axis as a unit Direction in the pose's frame."""

        if name not in ("x", "y", "z"):
            raise ValueError("axis is 'x', 'y' or 'z'")
        return Direction(self.rotation.as_matrix()[:, "xyz".index(name)], self.frame)

    def rotated(self, axis: Any, *, angle_deg: float | None = None, angle_rad: float | None = None,
                observation: _observation.Observation | None = None) -> "Pose":
        """This pose turned about ``axis`` through its position, by ``angle_deg`` or ``angle_rad``.

        ``axis`` is ``"x"``, ``"y"`` or ``"z"`` (the pose's own axes) or a Direction in any frame."""

        if (angle_deg is None) == (angle_rad is None):
            raise ValueError("give exactly one of angle_deg and angle_rad")
        angle = np.radians(angle_deg) if angle_deg is not None else float(angle_rad)
        if isinstance(axis, Direction):
            turn = Rotation.from_rotvec(axis.to(self.frame, observation=observation).unit().v * angle)
            rotation = turn * self.rotation
        elif axis in ("x", "y", "z"):
            rotation = self.rotation * Rotation.from_rotvec(np.eye(3)["xyz".index(axis)] * angle)
        else:
            raise TypeError("axis is 'x', 'y', 'z' or a Direction")
        return Pose(self.xyz, rotation.as_quat(), self.frame)

    def __add__(self, other: Any) -> "Pose":
        if isinstance(other, Direction):
            if other.frame != self.frame:
                raise _mismatch(self, other, "pose", "direction")
            if other.v.shape != (3,):
                raise ValueError("a Pose moves by one Direction, not a batch")
            return Pose(self.xyz + other.v, self.xyzw, self.frame)
        raise _bare(other, "Pose", self.frame) if not isinstance(other, (Point, Pose, Pixel)) else \
            TypeError(f"a Pose moves by a Direction, not a {type(other).__name__}")

    __radd__ = __add__

    def __sub__(self, other: Any) -> "Pose":
        if isinstance(other, Direction):
            return self + (-other)
        raise _bare(other, "Pose", self.frame) if not isinstance(other, (Point, Pose, Pixel)) else \
            TypeError(f"a Pose moves by a Direction, not a {type(other).__name__}")

    def to_json(self) -> dict[str, Any]:
        return {"frame": self.frame, "xyz": self.xyz.tolist(), "xyzw": self.xyzw.tolist()}

    tolist = to_json

    def __repr__(self) -> str:
        return (f"Pose({[round(float(x), 6) for x in self.xyz]}, {[round(float(x), 6) for x in self.xyzw]}, "
                f"frame={self.frame!r})")


class Pixel(_observation.ListsAttributes):
    """A pixel (``(2,)``) or pixels (``(N, 2)``) ``[u, v]`` of one camera's image; NaN marks one with no value."""

    __slots__ = ("uv", "camera")
    __array_ufunc__ = None

    def __init__(self, uv: Any, camera: Any) -> None:
        self.camera = _camera_name(camera)
        if not self.camera:
            raise FrameError("a Pixel needs its camera: Pixel(uv, camera='agentview')")
        uv = np.array(uv, dtype=np.float64)
        if uv.ndim not in (1, 2) or uv.shape[-1] != 2:
            raise ValueError(f"a Pixel's uv must be 2 numbers or an (N, 2) array; got shape {uv.shape}")
        uv.setflags(write=False)
        self.uv = uv

    @property
    def frame(self) -> str:
        return f"pixel:{self.camera}"

    @property
    def u(self) -> Any:
        return self.uv[..., 0]

    @property
    def v(self) -> Any:
        return self.uv[..., 1]

    @property
    def valid(self) -> Any:
        return np.isfinite(self.uv).all(axis=-1)

    def rounded(self) -> np.ndarray:
        """The nearest integer pixels (``int64``), as ``geometry_query`` rounds a projection."""

        if not np.all(self.valid):
            raise ValueError("some pixels have no value (NaN); select them first, e.g. px[px.valid]")
        return np.rint(self.uv).astype(np.int64)

    def in_frame(self, *, observation: _observation.Observation | None = None) -> Any:
        """Whether each pixel rounds to one inside its camera's image."""

        camera = _scene(observation).camera(self.camera)
        rounded = np.rint(np.where(np.isfinite(self.uv), self.uv, -1.0))
        return self.valid & (rounded[..., 0] >= 0) & (rounded[..., 0] < camera.width) \
            & (rounded[..., 1] >= 0) & (rounded[..., 1] < camera.height)

    def depth(self, *, window_px: int = 3, observation: _observation.Observation | None = None) -> Any:
        """The camera's published depth (metres) at each pixel: the median of the valid values in a window of
        ``window_px`` pixels on each side, as ``geometry_query`` takes it; NaN where there is none."""

        camera = _scene(observation).camera(self.camera)
        median, _ = geometry.window_depth(camera.depth_m, self._nearest(), window_px)
        return median[0] if self.uv.ndim == 1 else median

    def _nearest(self) -> np.ndarray:
        rounded = np.rint(np.atleast_2d(self.uv))
        rounded[~np.isfinite(rounded).all(axis=1)] = -1  # outside every image: no depth
        return rounded.astype(np.int64)

    def to(self, frame: str, *, window_px: int = 3,
           observation: _observation.Observation | None = None) -> "Point | Pixel":
        """The pixel(s) lifted into ``frame`` over the camera's published depth (``depth()``), or, for another
        ``pixel:<camera>``, lifted and projected into that camera. A pixel with no depth gives NaN."""

        scene = _scene(observation)
        camera = scene.camera(self.camera)
        if frame == self.frame:
            return self
        median, _ = geometry.window_depth(camera.depth_m, self._nearest(), window_px)
        in_camera = Point(geometry.back_project(np.atleast_2d(self.uv), median, camera.K)
                          .reshape(self.uv.shape[:-1] + (3,)), f"camera:{camera.name}")
        return in_camera.to(frame, observation=scene)

    def __add__(self, other: Any) -> "Pixel":
        if isinstance(other, (Point, Direction, Pose, Pixel)):
            raise TypeError(f"a Pixel moves by a pixel offset [du, dv], not a {type(other).__name__}")
        return Pixel(self.uv + np.asarray(other, dtype=np.float64), self.camera)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, Pixel):
            if other.camera != self.camera:
                raise FrameError(
                    f"cannot combine a Pixel in {self.frame} with a Pixel in {other.frame}: they are different "
                    f"images; convert one first, e.g. other.to({self.frame!r}) (lifted over its depth)")
            return self.uv - other.uv
        if isinstance(other, (Point, Direction, Pose)):
            raise TypeError(f"a Pixel moves by a pixel offset [du, dv], not a {type(other).__name__}")
        return Pixel(self.uv - np.asarray(other, dtype=np.float64), self.camera)

    def __len__(self) -> int:
        if self.uv.ndim == 1:
            raise TypeError("a single Pixel has no len(); a batch has")
        return len(self.uv)

    def __getitem__(self, index: Any) -> "Pixel":
        if self.uv.ndim == 1:
            raise TypeError("a single Pixel cannot be indexed; use .uv, .u, .v")
        return Pixel(self.uv[index], self.camera)

    def to_json(self) -> dict[str, Any]:
        return {"frame": self.frame, "uv": self.uv.tolist()}

    tolist = to_json

    def __repr__(self) -> str:
        return f"Pixel({_summary(self.uv, 2)}, camera={self.camera!r})"


def offset(value: Any, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, *, frame: str,
           observation: _observation.Observation | None = None) -> Any:
    """``value`` (a Pose or a Point) moved by ``(dx, dy, dz)`` metres along the axes of ``frame``.

    ``frame`` is any frame -- ``"tcp:right"`` moves along the arm's measured tool axes, as ``move_pose``'s
    ``end_effector`` delta does -- or ``"local"``: the Pose's own axes, for an approach offset from a target
    pose (``offset(grasp, dz=-0.05, frame="local")`` backs off 5 cm along a Panda grasp's approach axis, local
    ``+Z``; on ALOHA or ARX X5 it is ``dx``). The result keeps ``value``'s frame and orientation.
    """

    if not isinstance(value, (Pose, Point)):
        raise TypeError("offset moves a Pose or a Point")
    delta = np.array([dx, dy, dz], dtype=np.float64)
    if frame == "local":
        if not isinstance(value, Pose):
            raise FrameError("a Point has no axes of its own; give a frame (world, tcp:<arm>, camera:<name>)")
        return value + Direction(value.rotation.apply(delta), value.frame)
    return value + Direction(delta, frame).to(value.frame, observation=observation)


def direction(name: str, *, view: Any, horizontal: bool = False,
              observation: _observation.Observation | None = None) -> Direction:
    """The unit ``world`` vector that points ``"left"``, ``"right"``, ``"up"`` or ``"down"`` in a camera's image.

    With ``horizontal`` the vector is laid flat on the world's horizontal plane (``z`` dropped, ``+Z`` up) and
    normalized: "up in the image" then means away from the camera along the table. ``view`` is a camera name.
    """

    if name not in IMAGE_DIRECTIONS:
        raise ValueError(f"direction is one of {', '.join(IMAGE_DIRECTIONS)}")
    camera = _scene(observation).camera(_camera_name(view))
    world = Direction(IMAGE_DIRECTIONS[name], f"camera:{camera.name}").to("world", observation=observation)
    if horizontal:
        flat = np.array([world.v[0], world.v[1], 0.0])
        if np.linalg.norm(flat) < 1e-6:
            raise ValueError(f"{name!r} in {camera.name}'s image is vertical in the world; it has no horizontal part")
        world = Direction(flat, "world")
    return world.unit()


def pixels_to_world(camera: Any, uv: Any, *, window_px: int = 3,
                    observation: _observation.Observation | None = None) -> Point:
    """Pixels ``[u, v]`` (or ``(N, 2)``, any number) of ``camera`` lifted to ``world`` points over its published
    depth: the median of the valid depth in a window of ``window_px`` pixels on each side, then the pinhole
    model, as ``geometry_query``'s ``points``. A pixel outside the image or with no depth gives NaN
    (``Point.valid``)."""

    return Pixel(uv, camera).to("world", window_px=window_px, observation=observation)


def project(points: Any, camera: Any, *, observation: _observation.Observation | None = None) -> Pixel:
    """Points (a Point in any frame, or a Pose's position) to ``camera``'s pixels (floats); a point behind the
    camera gives NaN. ``Pixel.in_frame()`` says which land inside the image."""

    if isinstance(points, Pose):
        points = points.position
    if not isinstance(points, Point):
        raise TypeError("project takes a Point (or a Pose); a bare array has no frame: Point(xyz, frame='world')")
    return points.to(f"pixel:{_camera_name(camera)}", observation=observation)


__all__ = ["Direction", "FrameError", "IMAGE_DIRECTIONS", "Pixel", "Point", "Pose", "direction", "offset",
           "pixels_to_world", "project"]
