"""TOPP-RA + Ruckig timing of one joint path under a speed request: the core of the ``retime`` command.

A caller (``motion_pace.pace_path``, or a planner client answering a plan v2
with ``timing``) sends a joint path already on a grid at most 0.02 rad apart,
the controlled point's pose at every row (its own forward kinematics: nothing
here knows a robot), the speed request (``ora.contracts.motion_speed``) and the
joints' limits. The answer is the path sampled at the backend's control tick,
with the same semantics as ``motion_pace``:

1. The path is cut at its corners (``motion_pace.CORNER_DEG``) into pieces
   that each start and end at rest, and each piece is a cubic spline in joint
   space through the rows, parameterized by its joint arc length.
2. TOPP-RA (Pham & Pham, IEEE T-RO 2018) finds the fastest progress along each
   piece under every cap at once, with ``MARGIN`` on every cap: each joint's
   speed and acceleration (the preset's, lowered to the backend's ceiling), the
   controlled point's speed and turn rate (``x <= (v / sigma'(s))**2``) and its
   ramp, the tangential acceleration ``sigma'(s) u + sigma''(s) x``, where
   ``sigma`` and ``theta`` are the controlled point's arc length and turn
   along the path, fitted through the caller's poses. ``slow_approaching``
   adds the careful caps, joints included, from the gridpoint at or before
   ``SLOW_APPROACH_M`` from the end. ``ParametrizeConstAccel`` keeps the
   motion on the path. TOPP-RA has no jerk, and a joint at rest cannot take an
   acceleration step, so both ends of a piece also keep below the speed a
   constant jerk reaches from rest (``rest_speed_squared``), on a grid dense
   enough there to resolve it.
3. The profile is sampled at whole ticks, stretched uniformly so the last
   tick ends on the goal (no cap can grow).
4. Ruckig (Community edition, Berscheid & Kroeger, RSS 2021) certifies the
   rows an executor will follow. Each row's state comes from the rows
   themselves (``tick_states``, central differences, at rest at both ends):
   TOPP-RA's own acceleration is bang-bang and is only ever executed through
   the rows, so sampled at a tick it says nothing a servo can follow. From
   each tick's state Ruckig must reach the next one in exactly one tick
   (``DurationDiscretization.Discrete``) within the joints' speed, acceleration
   and hardware jerk limits, so a jerk-limited interpolant between the rows
   exists. A tick it cannot is timed again without the discretization: an
   executor taking Ruckig's shortest duration instead runs ahead of or behind
   the rows by that difference times the controlled point's speed there, and
   the sum over the move must stay within ``DEVIATION_TOLERANCE_M`` (no such
   tick off by more than ``MAX_TICK_OVERRUN``). The rows themselves stay on the
   path. A move past that is re-solved with every acceleration cap lowered by
   ``BACKOFF``, up to ``MAX_ATTEMPTS`` times, and is ``timing_infeasible`` after
   that. Ruckig's ``intermediate_positions`` (waypoints) is never set: in the
   Community wheel it calls a remote API.
5. The wrist check, the budget (refused, or cut to the longest first part,
   never inside a large wrist turn) and the wrist report are
   ``motion_pace``'s, on the timed rows.

Runs only under the timing interpreter (``ORA_TIMING_PYTHON``), which holds
toppra, ruckig, numpy and scipy (``requirements.py``).
"""

from __future__ import annotations

import importlib.metadata
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import ruckig
import toppra
from scipy.interpolate import CubicSpline
from toppra.constraint import DiscretizationType, JointAccelerationConstraint, LinearConstraint
from toppra.constraint.linear_constraint import canlinear_colloc_to_interpolate

from ora.contracts.motion_speed import SLOW_APPROACH_M
from ora.environment.controllers import motion_pace as mp
from ora.environment.controllers.motion_planner import error_answer
from ora.environment.controllers.motion_retime import (
    RETIME_ANSWER_SCHEMA, TIMING_IMPLEMENTATION, validate_retime_request,
)

#: Every cap is enforced at this share: TOPP-RA holds a cap only at its gridpoints.
MARGIN = 0.97
#: Gridpoint spacing along a piece's joint arc length, and the fewest gridpoints of a piece.
GRID_STEP_RAD = 0.01
MIN_GRIDPOINTS = 100
#: Share of the joints' jerk limit a piece's start and stop are planned at (the rest is the certifier's slack).
REST_JERK_SHARE = 0.5
#: The extra gridpoints at each end of a piece: this many, cubically spaced (a constant jerk's s ~ t^3) ...
REST_GRIDPOINTS = 12
#: ... over this much of the piece's joint arc length, at most a quarter of the piece.
REST_GRID_RAD = 0.05
#: A move Ruckig cannot certify is re-solved with every acceleration cap times this, at most this often.
BACKOFF = 0.8
MAX_ATTEMPTS = 8
#: Stands in for an absent cap in the solver's bounds (it takes no infinity).
UNBOUNDED = 1e8
TINY = 1e-9
PARAMETRIZER = "ParametrizeConstAccel"
SOLVER = "seidel"
#: The most the controlled point may run off the rows' timing along a certified move (metres), and the
#: largest share of a tick one row's Ruckig duration may differ from the tick.
DEVIATION_TOLERANCE_M = 0.001
MAX_TICK_OVERRUN = 0.25
LIBRARIES = ("toppra", "ruckig", "numpy", "scipy")


def versions() -> dict[str, str]:
    """The installed versions of the timing libraries."""

    return {name: importlib.metadata.version(name) for name in LIBRARIES}


def settings() -> dict[str, Any]:
    return {"margin": MARGIN, "grid_step_radians": GRID_STEP_RAD, "min_gridpoints": MIN_GRIDPOINTS,
            "rest_jerk_share": REST_JERK_SHARE, "rest_gridpoints": REST_GRIDPOINTS, "rest_grid_radians": REST_GRID_RAD,
            "parametrizer": PARAMETRIZER, "solver": SOLVER, "acceleration_backoff": BACKOFF,
            "max_attempts": MAX_ATTEMPTS, "ruckig_duration_discretization": "discrete",
            "deviation_tolerance_m": DEVIATION_TOLERANCE_M, "max_tick_overrun": MAX_TICK_OVERRUN}


class TimingInfeasible(Exception):
    """No timing within the limits: TOPP-RA found none, or Ruckig certified none."""

    def __init__(self, message: str, constraints: Mapping[str, Any]):
        super().__init__(message)
        self.constraints = dict(constraints)


# --------------------------------------------------------------- caps


@dataclass(frozen=True)
class Caps:
    """One region's caps in SI units; ``inf`` is no cap. Joint caps are per joint."""

    linear: float
    angular: float
    linear_accel: float
    angular_accel: float
    joint_speed: np.ndarray
    joint_accel: np.ndarray

    def scaled(self, factor: float) -> "Caps":
        """These caps with every acceleration times ``factor``."""

        return replace(self, linear_accel=self.linear_accel * factor, angular_accel=self.angular_accel * factor,
                       joint_accel=self.joint_accel * factor)

    def tighter(self, other: "Caps") -> "Caps":
        return Caps(min(self.linear, other.linear), min(self.angular, other.angular),
                    min(self.linear_accel, other.linear_accel), min(self.angular_accel, other.angular_accel),
                    np.minimum(self.joint_speed, other.joint_speed), np.minimum(self.joint_accel, other.joint_accel))


def _value(cap: float | None, convert: Any = float) -> float:
    return math.inf if cap is None else float(convert(cap))


def caps_of(pace: mp.Pace, velocity_ceiling: Sequence[float] | None,
            accel_ceiling: Sequence[float] | None, dof: int) -> Caps:
    """A pace's caps with its joint caps lowered to the backend's ceiling per joint (never raised)."""

    speed = np.full(dof, float(pace.joint_speed_rad_s))
    accel = np.full(dof, float(pace.joint_accel_rad_s2))
    if velocity_ceiling is not None:
        speed = np.minimum(speed, np.asarray(velocity_ceiling, dtype=float))
    if accel_ceiling is not None:
        accel = np.minimum(accel, np.asarray(accel_ceiling, dtype=float))
    return Caps(_value(pace.linear_m_s), _value(pace.angular_deg_s, math.radians),
                _value(pace.linear_accel_m_s2), _value(pace.angular_accel_deg_s2, math.radians), speed, accel)


# --------------------------------------------------------------- one piece


class PathCaps(LinearConstraint):
    """The controlled point's speed, turn rate and ramps, and each joint's speed, per gridpoint.

    ``x`` bounds carry every speed cap; the rows ``a u + b x`` carry the tangential ramps (``sigma'``,
    ``sigma''``; ``theta'``, ``theta''``) and, where the slow zone lowers them, the joints' accelerations.
    Caps vary along the path, so ``F`` and ``g`` are per gridpoint. The rows hold at both ends of each
    grid segment (interpolation), as ``JointAccelerationConstraint`` does: the acceleration of a segment
    is one number, and held at its first gridpoint alone a turn of the path overshot the ramp by 8%."""

    def __init__(self, gridpoints: np.ndarray, parameters: tuple[Any, ...]):
        super().__init__()
        self._gridpoints = gridpoints
        self._parameters = parameters
        self.set_discretization_type(DiscretizationType.Interpolation)

    def compute_constraint_params(self, path: Any, gridpoints: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        if not np.array_equal(gridpoints, self._gridpoints):
            raise ValueError("the constraint was built for other gridpoints")
        return self._parameters


@dataclass
class Piece:
    """One corner-to-corner piece of the path timed rest to rest."""

    nodes: np.ndarray
    s_nodes: np.ndarray
    offset: float
    path: Any
    sigma: CubicSpline
    theta: CubicSpline
    grid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    us: np.ndarray = field(default_factory=lambda: np.zeros(0))
    zone_from: int | None = None

    @property
    def length(self) -> float:
        return float(self.s_nodes[-1])

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def at(self, t: float) -> float:
        """Path position at time ``t`` into the piece (constant path acceleration per grid segment)."""

        t = min(max(t, 0.0), self.duration)
        index = min(int(np.searchsorted(self.times, t, side="right")) - 1, len(self.us) - 1)
        tau = t - self.times[index]
        s = self.grid[index] + self.sd[index] * tau + 0.5 * self.us[index] * tau * tau
        return float(min(max(s, self.grid[index]), self.grid[index + 1]))


def _spline(s: np.ndarray, values: np.ndarray) -> CubicSpline:
    return CubicSpline(s, values) if len(s) > 2 else CubicSpline(s, values, bc_type="natural")


def _piece(nodes: np.ndarray, sigma_nodes: np.ndarray, theta_nodes: np.ndarray, offset: float) -> Piece:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))])
    return Piece(nodes=nodes, s_nodes=s, offset=offset, path=toppra.SplineInterpolator(s, nodes),
                 sigma=_spline(s, sigma_nodes), theta=_spline(s, theta_nodes))


def gridpoints(length: float) -> np.ndarray:
    """A piece's gridpoints: even along it, and cubically spaced at each end, where it leaves and reaches rest."""

    even = np.linspace(0.0, length, max(MIN_GRIDPOINTS, int(math.ceil(length / GRID_STEP_RAD)) + 1))
    reach = min(REST_GRID_RAD, length / 4.0)
    ends = reach * (np.arange(1, REST_GRIDPOINTS + 1) / REST_GRIDPOINTS) ** 3
    return np.unique(np.concatenate([even, ends, length - ends]))


def rest_speed_squared(distance: np.ndarray, slopes: np.ndarray, jerk: np.ndarray) -> np.ndarray:
    """The largest path speed squared a constant jerk reaches from rest in ``distance`` along the path.

    ``slopes`` are the joints' ``q'(s)`` per gridpoint. The path's jerk ``J`` is the largest that keeps every
    joint within ``REST_JERK_SHARE`` of its limit; from rest ``s = J t^3 / 6`` and ``sd = J t^2 / 2``, so
    ``sd^2 = 6^(4/3) / 4 * J^(2/3) * s^(4/3)``."""

    scale = np.max(np.abs(slopes) / jerk, axis=1)
    path_jerk = REST_JERK_SHARE / np.maximum(scale, TINY)
    return 6.0 ** (4.0 / 3.0) / 4.0 * path_jerk ** (2.0 / 3.0) * distance ** (4.0 / 3.0)


def _parameters(piece: Piece, grid: np.ndarray, move: Caps, zone: Caps | None, zone_from_sigma: float,
                jerk: np.ndarray) -> tuple[tuple[Any, ...], int | None]:
    """``PathCaps``' parameters on ``grid`` and the gridpoint the slow zone starts at (``None``: no zone)."""

    count, dof = len(grid), piece.nodes.shape[1]
    q1, q2 = np.asarray(piece.path(grid, 1)), np.asarray(piece.path(grid, 2))
    sigma = piece.sigma(grid)
    s1, s2 = piece.sigma(grid, 1), piece.sigma(grid, 2)
    t1, t2 = piece.theta(grid, 1), piece.theta(grid, 2)
    zone_from = None
    if zone is not None and sigma[-1] > zone_from_sigma + 1e-12:
        # The cap holds at gridpoints only: start at the gridpoint at or before the boundary.
        before = np.nonzero(sigma <= zone_from_sigma + 1e-12)[0]
        zone_from = int(before[-1]) if before.size else 0
    region = [zone if zone_from is not None and index >= zone_from else move for index in range(count)]
    upper = np.full(count, UNBOUNDED)
    for index, caps in enumerate(region):
        bounds = [UNBOUNDED]
        moving = np.abs(q1[index]) > TINY
        if moving.any():
            bounds.append(float(np.min((MARGIN * caps.joint_speed[moving] / np.abs(q1[index][moving])) ** 2)))
        for cap, slope in ((caps.linear, abs(s1[index])), (caps.angular, abs(t1[index]))):
            if math.isfinite(cap) and slope > TINY:
                bounds.append((MARGIN * cap / slope) ** 2)
        upper[index] = min(bounds)
    upper = np.minimum(upper, rest_speed_squared(np.minimum(grid, grid[-1] - grid), q1, jerk))
    xbound = np.column_stack([np.zeros(count), upper])
    rows_a, rows_b, limits = [], [], []
    if any(math.isfinite(caps.linear_accel) or math.isfinite(caps.angular_accel) for caps in region):
        rows_a += [s1, t1]
        rows_b += [s2, t2]
        limits += [[MARGIN * min(caps.linear_accel, UNBOUNDED) for caps in region],
                   [MARGIN * min(caps.angular_accel, UNBOUNDED) for caps in region]]
    if zone_from is not None and zone is not None and np.any(zone.joint_accel < move.joint_accel):
        for joint in range(dof):
            rows_a.append(q1[:, joint])
            rows_b.append(q2[:, joint])
            limits.append([MARGIN * caps.joint_accel[joint] if caps is zone else UNBOUNDED for caps in region])
    if not rows_a:
        return (None, None, None, None, None, None, xbound), zone_from
    a, b = np.column_stack(rows_a), np.column_stack(rows_b)
    width = a.shape[1]
    single = np.vstack([np.eye(width), -np.eye(width)])
    F = np.repeat(single[None, :, :], count, axis=0)
    g = np.hstack([np.asarray(limits).T, np.asarray(limits).T])
    a, b, c, F, g, _, _ = canlinear_colloc_to_interpolate(a, b, np.zeros_like(a), F, g, None, None, grid)
    return (a, b, c, F, g, None, xbound), zone_from


def _solve(piece: Piece, move: Caps, zone: Caps | None, zone_from_sigma: float, jerk: np.ndarray) -> Piece:
    """``piece`` with its rest-to-rest TOPP-RA profile; raises ``TimingInfeasible``."""

    grid = gridpoints(piece.length)
    parameters, zone_from = _parameters(piece, grid, move, zone, zone_from_sigma, jerk)
    constraints = [JointAccelerationConstraint(MARGIN * move.joint_accel), PathCaps(grid, parameters)]
    instance = toppra.algorithm.TOPPRA(constraints, piece.path, gridpoints=grid, solver_wrapper=SOLVER,
                                       parametrizer=PARAMETRIZER)
    if instance.compute_trajectory(0.0, 0.0) is None:
        raise TimingInfeasible("TOPP-RA found no timing of the path within its caps",
                               {"toppra_return_code": str(instance.problem_data.return_code)})
    sd = np.maximum(np.asarray(instance.problem_data.sd_vec, dtype=float), 0.0)
    grid = np.asarray(instance.problem_data.gridpoints, dtype=float)
    steps = np.diff(grid)
    if np.any(sd[:-1] + sd[1:] <= TINY):
        raise TimingInfeasible("TOPP-RA stopped the path inside a piece", {"toppra_return_code": "Ok"})
    times = np.concatenate([[0.0], np.cumsum(2.0 * steps / (sd[:-1] + sd[1:]))])
    us = (sd[1:] ** 2 - sd[:-1] ** 2) / (2.0 * steps)
    return replace(piece, grid=grid, sd=sd, times=times, us=us, zone_from=zone_from)


# --------------------------------------------------------------- the move


@dataclass
class Timed:
    """A timed move: one row per tick from the start, with velocities, accelerations and where each row is."""

    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    arc: np.ndarray
    sigma: np.ndarray
    theta: np.ndarray
    stretch: float
    raw_duration_s: float
    pieces: list[Piece]
    zone_arc: float | None
    attempts: int
    scale: float
    extended: int
    deviation_m: float
    solve_ms: float
    ruckig_ms: float

    @property
    def ticks(self) -> int:
        return len(self.positions) - 1


def _sample(pieces: list[Piece], tick_s: float) -> tuple[np.ndarray, ...]:
    total = sum(piece.duration for piece in pieces)
    count = max(1, math.ceil(total / tick_s - 1e-9))
    stretch = total / (count * tick_s)
    starts = np.concatenate([[0.0], np.cumsum([piece.duration for piece in pieces])])
    dof = pieces[0].nodes.shape[1]
    positions = np.zeros((count + 1, dof))
    arc, sigma, theta = (np.zeros(count + 1) for _ in range(3))
    for row in range(count + 1):
        t = total * row / count
        index = min(int(np.searchsorted(starts, t, side="right")) - 1, len(pieces) - 1)
        piece = pieces[index]
        s = piece.at(t - starts[index])
        positions[row] = piece.path(s)
        arc[row], sigma[row], theta[row] = piece.offset + s, float(piece.sigma(s)), float(piece.theta(s))
    positions[0], positions[-1] = pieces[0].nodes[0], pieces[-1].nodes[-1]
    return positions, arc, sigma, theta, stretch, total


def tick_states(positions: np.ndarray, tick_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Each row's velocity and acceleration as the rows themselves give them, at rest at both ends.

    Central differences over the neighbouring rows: what an executor following the rows goes through. They
    average the rows' own steps, so unlike a spline through the rows they never exceed the speed the rows keep."""

    velocities, accelerations = np.zeros_like(positions), np.zeros_like(positions)
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * tick_s)
    accelerations[1:-1] = (positions[2:] - 2.0 * positions[1:-1] + positions[:-2]) / (tick_s * tick_s)
    return velocities, accelerations


def ruckig_input(dof: int) -> Any:
    """A Ruckig input of ``dof`` joints. Only state-to-state fields are ever set on it: never
    ``intermediate_positions``, whose waypoints the Community wheel sends to a remote API."""

    return ruckig.InputParameter(dof)


def _target_acceleration(velocity: np.ndarray, acceleration: np.ndarray, vmax: np.ndarray, amax: np.ndarray,
                         jmax: np.ndarray) -> np.ndarray:
    """An acceleration Ruckig accepts with this velocity: |a| <= sqrt(2 j (vmax - |v|)) (its README)."""

    bound = np.minimum(amax, np.sqrt(np.maximum(0.0, 2.0 * jmax * (vmax - np.abs(velocity)))) * 0.999)
    return np.clip(acceleration, -bound, bound)


@dataclass(frozen=True)
class Certificate:
    """What Ruckig says of a move's rows: ``deviation_m`` is ``inf`` when a tick is past any tolerance."""

    extended: int
    deviation_m: float
    accelerations: np.ndarray

    @property
    def holds(self) -> bool:
        return self.deviation_m <= DEVIATION_TOLERANCE_M


def _duration(otg: Any, given: Any, trajectory: Any) -> float | None:
    try:
        result = otg.calculate(given, trajectory)
    except ruckig.RuckigError:  # an input it refuses, such as a state past a limit
        return None
    return float(trajectory.duration) if result in (ruckig.Result.Working, ruckig.Result.Finished) else None


def certify(positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray, *, tick_s: float,
            vmax: np.ndarray, amax: np.ndarray, jmax: np.ndarray, speeds_m_s: np.ndarray) -> Certificate:
    """Ruckig's verdict on every tick-to-tick segment, whose controlled point moves at ``speeds_m_s``.

    A segment Ruckig runs in exactly one tick adds nothing. Any other adds its shortest duration's
    difference from the tick times the speed there; accelerations are clamped to what it accepts."""

    dof = positions.shape[1]
    exact, shortest = ruckig.Ruckig(dof, tick_s), ruckig.Ruckig(dof)
    trajectory = ruckig.Trajectory(dof)
    given = ruckig_input(dof)
    given.max_velocity, given.max_acceleration, given.max_jerk = list(vmax), list(amax), list(jmax)
    clamped = np.asarray([_target_acceleration(velocities[row], accelerations[row], vmax, amax, jmax)
                          for row in range(len(positions))])
    extended, deviation = 0, 0.0
    for row in range(len(positions) - 1):
        given.current_position, given.current_velocity = list(positions[row]), list(velocities[row])
        given.current_acceleration = list(clamped[row])
        given.target_position, given.target_velocity = list(positions[row + 1]), list(velocities[row + 1])
        given.target_acceleration = list(clamped[row + 1])
        given.duration_discretization = ruckig.DurationDiscretization.Discrete
        duration = _duration(exact, given, trajectory)
        if duration is not None and duration <= tick_s * (1.0 + 1e-6):
            continue
        extended += 1
        given.duration_discretization = ruckig.DurationDiscretization.Continuous
        duration = _duration(shortest, given, trajectory)
        if duration is None or abs(duration - tick_s) > MAX_TICK_OVERRUN * tick_s:
            deviation = math.inf
            continue
        deviation += abs(duration - tick_s) * float(speeds_m_s[row])
    return Certificate(extended, deviation, clamped)


def time_path(nodes: np.ndarray, sigma_nodes: np.ndarray, theta_nodes: np.ndarray, *, move: Caps,
              zone: Caps | None, zone_from_sigma: float, tick_s: float, jerk: np.ndarray) -> Timed:
    """The path through ``nodes`` timed at ``tick_s`` and certified by Ruckig; raises ``TimingInfeasible``."""

    corners = mp.corners(nodes)
    scale, attempts, certificate = 1.0, 0, None
    solve_ms = ruckig_ms = 0.0
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        mark = time.perf_counter()
        pieces, offset = [], 0.0
        for first, last in zip(corners, corners[1:]):
            piece = _piece(nodes[first:last + 1], sigma_nodes[first:last + 1], theta_nodes[first:last + 1], offset)
            pieces.append(_solve(piece, move.scaled(scale), zone.scaled(scale) if zone else None, zone_from_sigma,
                                 jerk))
            offset += piece.length
        positions, arc, sigma, theta, stretch, total = _sample(pieces, tick_s)
        solve_ms += (time.perf_counter() - mark) * 1e3
        mark = time.perf_counter()
        velocities, accelerations = tick_states(positions, tick_s)
        certificate = certify(positions, velocities, accelerations, tick_s=tick_s, vmax=move.joint_speed,
                              amax=move.joint_accel * scale, jmax=jerk, speeds_m_s=np.diff(sigma) / tick_s)
        ruckig_ms += (time.perf_counter() - mark) * 1e3
        if certificate.holds:
            zone_arc = next((piece.offset + float(piece.grid[piece.zone_from]) for piece in pieces
                             if piece.zone_from is not None), None)
            return Timed(positions, velocities, certificate.accelerations, arc, sigma, theta, stretch, total, pieces,
                         zone_arc, attempts, scale, certificate.extended, certificate.deviation_m, solve_ms,
                         ruckig_ms)
        scale *= BACKOFF
    raise TimingInfeasible("Ruckig cannot follow the timed path within the joints' jerk limits",
                           {"attempts": attempts, "extended_segments": certificate.extended,
                            "acceleration_scale": round(scale / BACKOFF, 6)})


# --------------------------------------------------------------- checks and the answer


def _distinct(rows: np.ndarray, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = [0]
    for index in range(1, len(rows)):
        if float(np.max(np.abs(rows[index] - rows[keep[-1]]))) > mp.DUPLICATE_ROW_RAD:
            keep.append(index)
    nodes, kept = rows[keep].copy(), poses[keep].copy()
    if len(nodes) > 1:
        nodes[-1], kept[-1] = rows[-1], poses[-1]
    return nodes, kept


def _fractions(nodes: np.ndarray) -> np.ndarray:
    """Each node's share of the path's largest-joint length (``motion_pace``'s grid fractions)."""

    steps = np.max(np.abs(np.diff(nodes, axis=0)), axis=1)
    total = float(np.sum(steps))
    if total <= 0:
        return np.linspace(0.0, 1.0, len(nodes))
    fractions = np.concatenate([[0.0], np.cumsum(steps) / total])
    fractions[-1] = 1.0
    return fractions


def _verification(timed: Timed, move: Caps, zone: Caps | None, zone_from_sigma: float,
                  tick_s: float) -> dict[str, float | None]:
    """Each cap's largest ratio over the ticks: the controlled point from the fitted arc length and turn, the
    joints from the rows. A tick is held to the slow caps when it starts within the last ``SLOW_APPROACH_M``."""

    linear, angular = np.diff(timed.sigma) / tick_s, np.diff(timed.theta) / tick_s
    slow = (timed.sigma[:-1] >= zone_from_sigma - 1e-9) if zone is not None else np.zeros(len(linear), bool)
    caps = [zone if inside else move for inside in slow]

    def ratio(values: np.ndarray, limits: list[float]) -> float | None:
        pairs = [(abs(v), c) for v, c in zip(values, limits) if math.isfinite(c)]
        return round(max(v / c for v, c in pairs), 4) if pairs else None

    ramp = np.diff(linear) / tick_s if len(linear) > 1 else np.zeros(0)
    joints_v = np.diff(timed.positions, axis=0) / tick_s
    joints_a = np.diff(joints_v, axis=0) / tick_s if len(joints_v) > 1 else np.zeros((0, move.joint_speed.size))
    return {"max_linear_over_cap": ratio(linear, [c.linear for c in caps]),
            "max_angular_over_cap": ratio(angular, [c.angular for c in caps]),
            "max_ramp_over_cap": ratio(ramp, [c.linear_accel for c in caps[1:]]),
            "max_joint_speed_over_cap": round(float(np.max(np.abs(joints_v) / move.joint_speed)), 4),
            "max_joint_accel_over_cap": (round(float(np.max(np.abs(joints_a) / move.joint_accel)), 4)
                                         if len(joints_a) else 0.0)}


def _peaks(timed: Timed) -> dict[str, float]:
    """The top speed and turn rate of the controlled point the timed profile reaches (at its gridpoints)."""

    linear = angular = 0.0
    for piece in timed.pieces:
        linear = max(linear, float(np.max(np.abs(piece.sigma(piece.grid, 1)) * piece.sd)))
        angular = max(angular, float(np.max(np.abs(piece.theta(piece.grid, 1)) * piece.sd)))
    return {"planned_peak_linear_m_s": round(linear * timed.stretch, 4),
            "planned_peak_angular_deg_s": round(math.degrees(angular * timed.stretch), 2)}


def _spline_deviation(timed: Timed) -> float:
    """The largest joint distance between a piece's spline and the straight line between its rows (midpoints)."""

    worst = 0.0
    for piece in timed.pieces:
        middle = (piece.s_nodes[:-1] + piece.s_nodes[1:]) / 2.0
        straight = (piece.nodes[:-1] + piece.nodes[1:]) / 2.0
        worst = max(worst, float(np.max(np.abs(np.asarray(piece.path(middle)) - straight))))
    return worst


def _refused(request: Mapping[str, Any], refusal: Exception, code: str, constraints: Mapping[str, Any],
             started: float, detail: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": RETIME_ANSWER_SCHEMA, "status": "refused", "reason": code, "trajectory": None,
            "report": None, "planner": _planner(), "timing_s": {"total": time.time() - started},
            "request": dict(request),
            "detail": {**dict(detail), "refusal": {"message": str(refusal), "constraints": dict(constraints)}}}


def _planner() -> dict[str, Any]:
    return {"implementation": TIMING_IMPLEMENTATION, "versions": versions(), "settings": settings()}


def _trivial(request: Mapping[str, Any], node: np.ndarray, pose: np.ndarray, started: float) -> dict[str, Any]:
    """A path that does not move: one tick at rest (as ``motion_pace.knots`` gives it)."""

    wrist = request["wrist"]
    rows = [node.tolist(), node.tolist()]
    zeros = [[0.0] * len(node)] * 2
    report = {"ticks": 1, "duration_s": round(float(request["tick_s"]), 6), "fractions": [0.0, 1.0],
              "slow_from_tick": None, "fraction_of_move": 1.0, "needed_ticks": 1, "part_segments": 1,
              "peaks": {"planned_peak_linear_m_s": 0.0, "planned_peak_angular_deg_s": 0.0},
              "wrist": {**mp.wrist_report(node, rows[1:], joint_index=wrist["joint_index"],
                                          limits_rad=tuple(wrist["limits_rad"])),
                        **mp.path_turns([pose.tolist()])}}
    return {"schema": RETIME_ANSWER_SCHEMA, "status": "timed", "reason": None,
            "trajectory": {"joint_names": list(request["joint_names"]), "positions": rows, "velocities": zeros,
                           "accelerations": zeros, "dt_s": float(request["tick_s"])},
            "report": report, "planner": _planner(), "timing_s": {"total": time.time() - started},
            "request": dict(request), "detail": {"attempts": 0, "pieces": 0, "moving": False}}


def retime(raw: Any) -> dict[str, Any]:
    """Answer one retime request (``ora.motion-retime-answer.v1``); errors are answers too."""

    started = time.time()
    request = None
    try:
        request = validate_retime_request(raw)
        return _retime(request, started)
    except Exception as exc:  # an answer, never a crash: the worker keeps serving
        import sys
        import traceback

        traceback.print_exc(file=sys.stderr)
        return error_answer(RETIME_ANSWER_SCHEMA, request, "planner_error", exc,
                            implementation=TIMING_IMPLEMENTATION, started=started)


def _retime(request: Mapping[str, Any], started: float) -> dict[str, Any]:
    rows = np.asarray(request["path"]["joint_positions"], dtype=float)
    poses = np.asarray(request["path"]["tool_poses_xyzw"], dtype=float)
    nodes, poses = _distinct(rows, poses)
    if len(nodes) == 1:
        return _trivial(request, nodes[0], poses[0], started)
    dof, tick_s = nodes.shape[1], float(request["tick_s"])
    limits, wrist = request["limits"], request["wrist"]
    pace, slow_pace = mp.paces(request["speed"])
    move = caps_of(pace, limits["joint_velocity_rad_s"], limits["joint_acceleration_rad_s2"], dof)
    zone = None
    if request["slow_approach"] and slow_pace is not None:
        zone = move.tighter(caps_of(slow_pace, limits["joint_velocity_rad_s"], limits["joint_acceleration_rad_s2"],
                                    dof))
    jerk = np.asarray(limits["joint_jerk_rad_s3"], dtype=float)
    linear, angular = mp.segment_lengths(poses.tolist())
    sigma_nodes = np.concatenate([[0.0], np.cumsum(linear)])
    theta_nodes = np.concatenate([[0.0], np.cumsum(angular)])
    zone_from_sigma = float(sigma_nodes[-1]) - SLOW_APPROACH_M
    fractions = _fractions(nodes)
    budget = int(request["budget_ticks"])
    joint, wrist_limits = int(wrist["joint_index"]), (float(wrist["limits_rad"][0]), float(wrist["limits_rad"][1]))
    detail: dict[str, Any] = {"jerk_source": limits["jerk_source"], "nodes": len(nodes)}

    def timed(count: int) -> Timed:
        return time_path(nodes[:count + 1], sigma_nodes[:count + 1], theta_nodes[:count + 1], move=move, zone=zone,
                         zone_from_sigma=zone_from_sigma, tick_s=tick_s, jerk=jerk)

    segments = len(nodes) - 1
    try:
        whole = timed(segments)
    except TimingInfeasible as exc:
        return _refused(request, exc, "timing_infeasible", exc.constraints, started, detail)
    start = nodes[0].tolist()
    try:
        turns = mp.path_turns(poses.tolist())
        if request["wrist"].get("unwind_check", True):
            turns = mp.check_unwind(poses.tolist(), evidence={"wrist": mp.wrist_report(
                start, whole.positions[1:].tolist(), joint_index=joint, limits_rad=wrist_limits)})
    except mp.PaceRefusal as refusal:
        return _refused(request, refusal, refusal.code, refusal.constraints, started, detail)
    part, part_segments = whole, segments
    try:
        if whole.ticks > budget and not request["cut_to_budget"]:
            raise mp.budget_refusal(budget, whole.ticks)
        if whole.ticks > budget:
            part, part_segments = _fit_budget(timed, segments, budget, whole.ticks,
                                              mp.wrist_turn_spans(nodes[:, joint]), fractions)
    except mp.PaceRefusal as refusal:
        return _refused(request, refusal, refusal.code, refusal.constraints, started, detail)
    except TimingInfeasible as exc:
        return _refused(request, exc, "timing_infeasible", exc.constraints, started, detail)
    node_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))])
    row_fractions = np.interp(part.arc, node_arc, fractions)
    row_fractions[0], row_fractions[-1] = 0.0, float(fractions[part_segments])
    slow_from = None
    if part.zone_arc is not None:
        slow_from = next((row for row in range(1, part.ticks + 1) if part.arc[row] > part.zone_arc + 1e-12), None)
    report = {
        "ticks": part.ticks, "duration_s": round(part.ticks * tick_s, 6),
        "fractions": [float(value) for value in row_fractions], "slow_from_tick": slow_from,
        "fraction_of_move": float(fractions[part_segments]), "needed_ticks": whole.ticks,
        "part_segments": part_segments, "peaks": _peaks(part),
        "wrist": {**mp.wrist_report(start, part.positions[1:].tolist(), joint_index=joint, limits_rad=wrist_limits),
                  **turns},
    }
    detail.update({
        "attempts": part.attempts, "acceleration_scale": round(part.scale, 6),
        "ruckig_extended_segments": part.extended, "max_deviation_m": round(part.deviation_m, 7),
        "pieces": len(part.pieces),
        "gridpoints": [len(piece.grid) for piece in part.pieces], "raw_duration_s": round(part.raw_duration_s, 6),
        "stretch": round(part.stretch, 6), "solve_ms": round(part.solve_ms, 3), "ruckig_ms": round(part.ruckig_ms, 3),
        "spline_deviation_radians": round(_spline_deviation(part), 9),
        "verification": _verification(part, move, zone, zone_from_sigma, tick_s),
    })
    return {"schema": RETIME_ANSWER_SCHEMA, "status": "timed", "reason": None,
            "trajectory": {"joint_names": list(request["joint_names"]), "positions": part.positions.tolist(),
                           "velocities": part.velocities.tolist(), "accelerations": part.accelerations.tolist(),
                           "dt_s": tick_s},
            "report": report, "planner": _planner(), "timing_s": {"total": time.time() - started},
            "request": dict(request), "detail": detail}


def _fit_budget(timed: Any, segments: int, budget: int, needed: int, spans: Sequence[tuple[int, int]],
                fractions: np.ndarray) -> tuple[Timed, int]:
    """The longest first part whose ticks fit ``budget`` (``motion_pace.fit_budget`` on node prefixes)."""

    low, high, best = 1, segments, None
    while low <= high:
        middle = (low + high) // 2
        candidate = timed(middle)
        if candidate.ticks <= budget:
            best, low = (middle, candidate), middle + 1
        else:
            high = middle - 1
    if best is None:
        raise mp.cover_refusal(budget)
    nodes, span = mp.atomic_start(spans, best[0])
    if nodes == 0:
        raise mp.unsplit_refusal(budget, needed, fractions[span[0]], fractions[span[1]])
    if nodes != best[0]:
        best = (nodes, timed(nodes))
    return best[1], best[0]


__all__ = ["BACKOFF", "Caps", "Certificate", "DEVIATION_TOLERANCE_M", "MARGIN", "MAX_ATTEMPTS", "MAX_TICK_OVERRUN",
           "TimingInfeasible", "caps_of",
           "certify", "gridpoints", "rest_speed_squared", "retime", "ruckig_input", "settings", "tick_states",
           "time_path", "versions"]
