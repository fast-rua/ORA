"""Immutable values. Frames are explicit strings, lengths metres, angles radians.

Replay: these constructors are pure computation. JSON and arrays own immutable
bytes; exporting a dict gives a copy, exporting an array gives a read-only view.
They provide value semantics, not an in-process security sandbox.
"""
from dataclasses import dataclass
import math

import numpy as np

from ..interface.types import Response, InterfaceError, canonical, decode


@dataclass(frozen=True)
class JsonValue:
    """All public fields, including unknown versioned diagnostics, as immutable JSON bytes."""
    encoded: bytes

    def __post_init__(self):
        object.__setattr__(self, 'encoded', bytes(self.encoded))
        decode(self.encoded)

    @classmethod
    def of(cls, value):
        """Pure computation; copy finite JSON, e.g. JsonValue.of({'arm':'left'})."""
        return cls(canonical(value))

    def json(self):
        """Pure computation; return an independent mutable JSON tree, never shared state."""
        return decode(self.encoded)


@dataclass(frozen=True)
class Array:
    """Owned immutable numeric array; explicit dtype/shape/order, no object arrays."""
    data: bytes
    dtype: str
    shape: tuple[int, ...]
    order: str = 'C'

    def __post_init__(self):
        object.__setattr__(self, 'data', bytes(self.data))
        object.__setattr__(self, 'shape', tuple(self.shape))
        dtype = np.dtype(self.dtype)
        if (dtype.hasobject or dtype.fields or dtype.kind not in 'biufc' or self.order not in ('C','F')
                or any(type(n) is not int or n < 0 for n in self.shape)
                or math.prod(self.shape)*dtype.itemsize != len(self.data)):
            raise ValueError('invalid immutable array layout')

    @classmethod
    def of(cls, value):
        """Pure computation; copy a numeric array, preserving dtype and C/F layout."""
        a = np.asarray(value)
        order = 'F' if a.flags.f_contiguous and not a.flags.c_contiguous else 'C'
        return cls(a.tobytes(order=order), a.dtype.str, a.shape, order)

    def numpy(self):
        """Pure computation; bytes-backed view cannot be made writable; copy to edit."""
        return np.frombuffer(self.data, dtype=self.dtype).reshape(self.shape, order=self.order)


@dataclass(frozen=True)
class PoseM:
    """Position in metres and quaternion xyzw in the named parent frame; no implicit frame conversion."""
    frame: str
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    def __post_init__(self):
        p, q = tuple(self.position_m), tuple(self.orientation_xyzw)
        if (not self.frame or len(p)!=3 or len(q)!=4 or not all(math.isfinite(x) for x in (*p,*q))
                or not math.isclose(sum(x*x for x in q),1.,abs_tol=1e-3)):
            raise ValueError('pose requires a named frame, finite metres and unit xyzw quaternion')
        object.__setattr__(self,'position_m',p)
        object.__setattr__(self,'orientation_xyzw',q)


@dataclass(frozen=True)
class TransformM:
    """Rigid 4x4 source→target transform, translation in metres; column-vector convention."""
    source_frame: str
    target_frame: str
    matrix: Array

    def __post_init__(self):
        a = self.matrix.numpy()
        if (not self.source_frame or not self.target_frame or a.dtype.kind not in 'fiu' or a.shape!=(4,4) or not np.isfinite(a).all()
                or not np.allclose(a[3],[0,0,0,1],atol=1e-8,rtol=0)
                or not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-5,rtol=0)
                or not np.isclose(np.linalg.det(a[:3,:3]),1.,atol=1e-5,rtol=0)):
            raise ValueError('transform must be finite rigid SE(3), with named source/target frames')


@dataclass(frozen=True)
class PointCloudM:
    """Nx3 observed/derived points in the named frame, metres; never completed object shape."""
    frame: str
    xyz_m: Array

    def __post_init__(self):
        a = self.xyz_m.numpy()
        if not self.frame or a.dtype.kind not in 'fiu' or a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all():
            raise ValueError('point cloud must be finite Nx3 with a named frame')


@dataclass(frozen=True)
class PublicResult:
    """Full immutable public envelope plus separate comparison; retains original response bytes."""
    envelope: JsonValue
    comparison: JsonValue | None = None

    @classmethod
    def of(cls, response: Response):
        """Pure computation; wrap original bytes without dropping provenance/error/diagnostics."""
        return cls(JsonValue(response.payload), None if response.comparison is None else JsonValue.of(response.comparison))

    @property
    def data(self):
        """Pure computation; complete public data, or InterfaceError on error/timeout status."""
        value = self.envelope.json()
        if value['status'] != 'ok':
            if isinstance(value.get('error'), dict) and 'tool_error' in value['error']:
                from ..interface.types import PublicToolFailure
                raise PublicToolFailure(value['error']['tool_error'])
            raise InterfaceError(f"{value['method']}: {value['status']} {value['error']}")
        return JsonValue.of(value['data'])
