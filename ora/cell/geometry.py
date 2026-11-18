"""Geometry checks of a real cell against its robot model (any ``Kinematics``, ``ora.cell.adapter``).

* ``table_plane`` fits the table in a calibrated camera's depth. Seen from a wrist D405,
  whose pose comes from the encoders through the CAD chain, the table must come out
  level and at the table height: a tilted or shifted plane means an encoder zero or the
  camera mount has moved.
* ``register_fixed_camera`` finds a fixed camera's pose from what it sees of the arms.
  The table plane fixes the camera's tilt; the arms, placed by forward kinematics of the
  measured joints, fix its heading and position. No marker or target is needed.
* ``joint_sanity`` compares measured joints with the model limits and the cell's
  recorded rest pose.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# Planes and registration.
PLANE_RANSAC_ITERATIONS = 300
PLANE_INLIER_M = 0.006
WRIST_TILT_LIMIT_DEG = 3.0
WRIST_HEIGHT_LIMIT_M = 0.015
REGISTRATION_YAW_STEP_DEG = 5.0
REGISTRATION_SHIFTS_M = ((0.0, 0.0), (0.1, 0.0), (-0.1, 0.0), (0.0, 0.1), (0.0, -0.1))
REGISTRATION_COARSE_POINTS = 1500
REGISTRATION_REFINE = 8
REGISTRATION_RIVAL_RATIO = 1.15
REGISTRATION_ACCEPT_MEDIAN_M = 0.006
REGISTRATION_ACCEPT_INLIERS = 0.5
REGISTRATION_INLIER_M = 0.01
MODEL_VOXEL_M = 0.004
# A measured joint this far outside the model range is an encoder zero, not a pose
# (i2rt itself pads the range by 0.15 rad).
JOINT_LIMIT_PAD_RAD = 0.15
REST_DEVIATION_RAD = 0.1


# ---------------------------------------------------------------- depth to points
def depth_points(depth: np.ndarray, intrinsic: Sequence[Sequence[float]], *, near: float, far: float,
                 stride: int = 4) -> np.ndarray:
    """Camera-frame (OpenCV) points of every ``stride``-th pixel with depth in [near, far]."""

    k = np.asarray(intrinsic, dtype=float)
    sampled = np.asarray(depth, dtype=float)[::stride, ::stride]
    v, u = np.nonzero((sampled > near) & (sampled < far))
    z = sampled[v, u]
    u, v = u * stride, v * stride
    return np.c_[(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z]


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def fit_plane(points: np.ndarray, *, seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """Largest plane by RANSAC, refined by least squares: (unit normal, offset, inlier mask)."""

    rng = np.random.default_rng(seed)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(PLANE_RANSAC_ITERATIONS):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-9:
            continue
        normal /= np.linalg.norm(normal)
        inliers = np.abs(points @ normal - a @ normal) < PLANE_INLIER_M
        if inliers.sum() > best.sum():
            best = inliers
    centroid = points[best].mean(axis=0)
    centred = points[best] - centroid
    normal = np.linalg.eigh(centred.T @ centred)[1][:, 0]     # smallest-variance direction
    return normal, float(normal @ centroid), best


def table_plane(depth: np.ndarray, intrinsic: Sequence[Sequence[float]], cam2world: np.ndarray, *,
                near: float, far: float) -> dict[str, Any] | None:
    """The dominant plane of a calibrated view, in world: tilt from level and height."""

    points = depth_points(depth, intrinsic, near=near, far=far)
    if len(points) < 500:
        return None
    world = transform(points, cam2world)
    normal, _, inliers = fit_plane(world)
    if normal[2] < 0:
        normal = -normal
    return {"tilt_deg": float(np.degrees(np.arccos(min(1.0, normal[2])))),
            "height_m": float(np.median(world[inliers, 2])), "inlier_fraction": float(inliers.mean()),
            "points": int(len(points))}


def wrist_table_check(plane: Mapping[str, Any] | None, table_top_z_m: float | None, *,
                      max_tilt_deg: float = WRIST_TILT_LIMIT_DEG,
                      max_height_gap_m: float = WRIST_HEIGHT_LIMIT_M) -> dict[str, Any]:
    if plane is None:
        return {"passed": None, "why": "too little depth to fit the table"}
    problems = []
    if plane["tilt_deg"] > max_tilt_deg:
        problems.append(f"table tilts {plane['tilt_deg']:.1f} deg in the wrist view")
    if table_top_z_m is not None and abs(plane["height_m"] - table_top_z_m) > max_height_gap_m:
        problems.append(f"table at {plane['height_m']:.3f} m in the wrist view, {table_top_z_m:.3f} m calibrated")
    return {"passed": not problems, "problems": problems, **plane}


# ---------------------------------------------------------------- joints
def joint_sanity(arm_q: Mapping[str, np.ndarray], limits: Mapping[str, np.ndarray],
                 rest: Mapping[str, Sequence[float]] | None) -> dict[str, Any]:
    report: dict[str, Any] = {"outside_limits": {}, "rest_deviation_rad": None, "at_rest": None}
    for arm, q in arm_q.items():
        low, high = limits[arm][:, 0] - JOINT_LIMIT_PAD_RAD, limits[arm][:, 1] + JOINT_LIMIT_PAD_RAD
        bad = {f"joint{i + 1}": float(value) for i, value in enumerate(q) if not low[i] <= value <= high[i]}
        if bad:
            report["outside_limits"][arm] = bad
    if rest:
        deviation = {arm: (np.asarray(arm_q[arm]) - np.asarray(rest[arm])).tolist() for arm in arm_q}
        report["rest_deviation_rad"] = deviation
        report["at_rest"] = all(max(abs(v) for v in values) <= REST_DEVIATION_RAD for values in deviation.values())
    report["passed"] = not report["outside_limits"]
    return report


# ---------------------------------------------------------------- registration
def arm_surface_points(kinematics: Any, *, per_face: int = 2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """World points sampled on every mesh of the station model at its current state, with
    the unit normals of the faces they lie on."""

    mujoco, model, data = kinematics.mujoco, kinematics.model, kinematics.data
    mujoco.mj_kinematics(model, data)
    rng = np.random.default_rng(seed)
    offset = np.asarray(kinematics.world_offset, dtype=float)
    points, normals = [], []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[g]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        faces = model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        face_normal = np.cross(b - a, c - a)
        length = np.linalg.norm(face_normal, axis=1, keepdims=True)
        keep = length[:, 0] > 1e-12
        a, b, c, face_normal = a[keep], b[keep], c[keep], face_normal[keep] / length[keep]
        r1, r2 = rng.random((2, len(a), per_face, 1))
        s = np.sqrt(r1)
        local = ((1 - s) * a[:, None] + s * (1 - r2) * b[:, None] + s * r2 * c[:, None]).reshape(-1, 3)
        rotation = data.geom_xmat[g].reshape(3, 3)
        points.append(local @ rotation.T + data.geom_xpos[g] + offset)
        normals.append(np.repeat(face_normal, per_face, axis=0) @ rotation.T)
    return np.vstack(points), np.vstack(normals)


def voxel_downsample(points: np.ndarray, size: float) -> np.ndarray:
    """Indices of one point (the first) per ``size`` cube."""

    _, first = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    return np.sort(first)


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _level(normal: np.ndarray) -> np.ndarray:
    """A rotation taking ``normal`` to +z (Rodrigues)."""

    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z)
    sin, cos = np.linalg.norm(axis), float(normal @ z)
    if sin < 1e-9:
        return np.eye(3) if cos > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / sin
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + sin * skew + (1 - cos) * skew @ skew


def _plane_step(scene: np.ndarray, model: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """One damped Gauss-Newton step of point-to-plane alignment in heading and position.

    The turn is about the vertical through the scene's centroid; the tilt stays as it is."""

    centre = scene.mean(0)
    arm = scene - centre
    residual = np.einsum("ij,ij->i", normals, scene - model)
    jacobian = np.c_[normals[:, 0] * -arm[:, 1] + normals[:, 1] * arm[:, 0], normals]
    hessian = jacobian.T @ jacobian
    step = np.linalg.solve(hessian + 1e-6 * np.trace(hessian) * np.eye(4), -jacobian.T @ residual)
    turn = np.eye(4)
    turn[:3, :3] = _rz(step[0])
    there, back = np.eye(4), np.eye(4)
    there[:3, 3], back[:3, 3] = centre + step[1:], -centre
    return there @ turn @ back


def _icp(tree: Any, model: np.ndarray, normals: np.ndarray, scene: np.ndarray, start: np.ndarray, *,
         iterations: int, radius: float, keep: float = 0.7) -> tuple[np.ndarray, float, float]:
    """Trimmed point-to-plane ICP of ``scene`` onto ``model`` in heading and position only.

    The camera's tilt comes from the table plane, measured on thousands of points; left
    free, a top-down view of the arms (mostly flat upper faces) lets it drift by degrees.
    Point-to-plane, unlike point-to-point, slides along those faces in a few steps.
    Returns (pose, median residual, inlier fraction)."""

    pose = start.copy()
    for _ in range(iterations):
        moved = transform(scene, pose)
        distance, index = tree.query(moved, distance_upper_bound=radius)
        found = np.isfinite(distance)
        if found.sum() < 50:
            return pose, float("inf"), 0.0
        use = found & (distance <= np.quantile(distance[found], keep))
        update = _plane_step(moved[use], model[index[use]], normals[index[use]])
        pose = update @ pose
        if np.linalg.norm(update[:3, 3]) < 1e-5 and abs(update[1, 0]) < 1e-6:
            break
    distance, _ = tree.query(transform(scene, pose), distance_upper_bound=radius)
    found = np.isfinite(distance)
    median = float(np.median(distance[found])) if found.any() else float("inf")
    return pose, median, float((distance < REGISTRATION_INLIER_M).mean())


def register_fixed_camera(depth: np.ndarray, intrinsic: Sequence[Sequence[float]], kinematics: Any, *,
                          table_top_z_m: float = 0.0, near: float = 0.25, far: float = 2.0,
                          max_points: int = 6000, seed: int = 0,
                          extra_views: Sequence[tuple[np.ndarray, Mapping[str, Sequence[float]]]] = ()) -> dict[str, Any]:
    """cam2world of a fixed camera from its depth of the table and the two arms.

    ``extra_views`` adds (depth, arm joints) pairs taken by the same, unmoved camera with the
    arms elsewhere: their points and the arms at those joints join the fit, so a few poses
    pin down what one pose leaves loose."""

    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    points = np.concatenate([depth_points(d, intrinsic, near=near, far=far, stride=2)
                             for d in (depth, *(view[0] for view in extra_views))])
    if len(points) < 2000:
        return {"accepted": False, "why": f"only {len(points)} depth points between {near} and {far} m"}
    normal, offset, on_table = fit_plane(points[rng.permutation(len(points))[:20000]])
    on_table = np.abs(points @ normal - offset) < PLANE_INLIER_M
    if offset > 0:                       # orient the normal from the table towards the camera
        normal, offset = -normal, -offset
    height = -offset                     # the camera's distance above the table plane
    level = _level(normal)
    above = points[(points @ normal - offset > 0.015) & (points @ normal - offset < 0.8)]
    if len(above) < 500:
        return {"accepted": False, "why": "almost nothing stands on the table in this view"}
    scene = above[rng.permutation(len(above))[:max_points]]
    surfaces = [arm_surface_points(kinematics)]
    for _, arm_q in extra_views:
        kinematics.set_state(arm_q)
        surfaces.append(arm_surface_points(kinematics))
    model, normals = np.concatenate([m for m, _ in surfaces]), np.concatenate([n for _, n in surfaces])
    kept = voxel_downsample(model, MODEL_VOXEL_M)
    kept = kept[model[kept, 2] > table_top_z_m + 0.015]
    model, normals = model[kept], normals[kept]
    tree = cKDTree(model)
    coarse_scene = scene[:REGISTRATION_COARSE_POINTS]
    candidates = []
    for yaw in np.radians(np.arange(0.0, 360.0, REGISTRATION_YAW_STEP_DEG)):
        start = np.eye(4)
        start[:3, :3] = _rz(yaw) @ level
        start[:3, 3] = [0.0, 0.0, table_top_z_m + height]
        start[:2, 3] += model[:, :2].mean(0) - transform(coarse_scene, start)[:, :2].mean(0)
        for dx, dy in REGISTRATION_SHIFTS_M:
            shifted = start.copy()
            shifted[:2, 3] += (dx, dy)
            pose, median, inliers = _icp(tree, model, normals, coarse_scene, shifted, iterations=20, radius=0.15)
            candidates.append((inliers, -median, pose))
    candidates.sort(key=lambda item: item[:2], reverse=True)
    refined = []
    for _, _, pose in candidates[:REGISTRATION_REFINE]:
        for radius, iterations in ((0.05, 30), (0.02, 30), (0.01, 30)):
            pose, median, inliers = _icp(tree, model, normals, scene, pose, iterations=iterations, radius=radius)
        refined.append((median, inliers, pose))
    refined.sort(key=lambda item: item[0])
    median, inliers, pose = refined[0]
    # A different pose that fits almost as well means the view does not pin the camera down.
    rival = next((item for item in refined[1:] if _distinct(item[2], pose)
                  and item[0] < median * REGISTRATION_RIVAL_RATIO), None)
    table_world = transform(points[on_table][:5000], pose)
    accepted = (median <= REGISTRATION_ACCEPT_MEDIAN_M and inliers >= REGISTRATION_ACCEPT_INLIERS
                and rival is None)
    why = None
    if rival is not None:
        why = (f"ambiguous: another pose {_pose_gap(rival[2], pose)} (m, deg) away fits with median "
               f"{rival[0] * 1000:.1f} mm")
    elif not accepted:
        why = f"median residual {median * 1000:.1f} mm, inliers {inliers:.0%}: not a clean fit of the arms"
    return {"accepted": bool(accepted), "cam2world_cv": pose, "median_residual_m": median,
            "inlier_fraction": inliers, "rival_median_m": None if rival is None else rival[0],
            "scene_points": int(len(scene)), "model_points": int(len(model)),
            "table_height_m": float(np.median(table_world[:, 2])), "why": why}


def _distinct(a: np.ndarray, b: np.ndarray) -> bool:
    shift, turn = _pose_gap(a, b)
    return shift > 0.02 or turn > 2.0


def _pose_gap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    turn = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1.0, 1.0)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(np.degrees(np.arccos(turn)))


# ---------------------------------------------------------------- registration on moving arms
def _moving_surfaces(kinematics: Any, arm_qs: Sequence[Mapping[str, Sequence[float]]], *, per_face: int = 2,
                     seed: int = 0) -> list[np.ndarray]:
    """Per pose, world points on the meshes that move between the poses (the arms' links)."""

    mujoco, model, data = kinematics.mujoco, kinematics.model, kinematics.data
    placements = []
    for q in arm_qs:
        kinematics.set_state(q)
        mujoco.mj_kinematics(model, data)
        placements.append((data.geom_xpos.copy(), data.geom_xmat.copy()))
    meshes = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    moving = [g for g in meshes if max(np.linalg.norm(p[0][g] - placements[0][0][g]) for p in placements) > 0.01]
    offset = np.asarray(kinematics.world_offset, dtype=float)
    rng = np.random.default_rng(seed)
    local = {}
    for g in moving:
        mesh = model.geom_dataid[g]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        faces = model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        r1, r2 = rng.random((2, len(faces), per_face, 1))
        s = np.sqrt(r1)
        local[g] = ((1 - s) * a[:, None] + s * (1 - r2) * b[:, None] + s * r2 * c[:, None]).reshape(-1, 3)
    out = []
    for xpos, xmat in placements:
        points = [local[g] @ xmat[g].reshape(3, 3).T + xpos[g] + offset for g in moving]
        out.append(voxel_downsample_points(np.vstack(points), MODEL_VOXEL_M) if points else np.zeros((0, 3)))
    return out


def voxel_downsample_points(points: np.ndarray, size: float) -> np.ndarray:
    return points[voxel_downsample(points, size)]


def _pixels_points(depth: np.ndarray, mask: np.ndarray, intrinsic: Sequence[Sequence[float]]) -> np.ndarray:
    k = np.asarray(intrinsic, float)
    v, u = np.nonzero(mask)
    z = depth[v, u]
    return np.c_[(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z]


def register_on_moving_arms(views: Sequence[tuple[np.ndarray, Mapping[str, Sequence[float]]]],
                            intrinsic: Sequence[Sequence[float]], kinematics: Any,
                            rough_cam2world: Sequence[np.ndarray], *, change_m: float = 0.03,
                            near_model_m: float = 0.15, max_points: int = 5000, seed: int = 0) -> dict[str, Any]:
    """cam2world of a fixed camera from views of the arms at rest (``views[0]``) and moved.

    The scene is static, so a pixel nearer in a moved view than at rest shows an arm there.
    Only those pixels, and only near where a rough pose puts the arms, are fitted to the
    moving links at the measured joints; no plane is assumed to be the table."""

    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    rest_depth = views[0][0]
    models = _moving_surfaces(kinematics, [q for _, q in views])
    sets = []                                   # (camera-frame points, model tree, model points)
    for i, (depth, _) in enumerate(views):
        if i == 0:
            nearer = np.zeros(depth.shape, bool)
            for other, _ in views[1:]:
                nearer |= (rest_depth > 0) & (other > 0) & (rest_depth < other - change_m)
            points = _pixels_points(rest_depth, nearer, intrinsic)
        else:
            mask = (depth > 0) & (rest_depth > 0) & (depth < rest_depth - change_m)
            points = _pixels_points(depth, mask, intrinsic)
        if len(points) and len(models[i]):
            sets.append((points[rng.permutation(len(points))[:max_points]], cKDTree(models[i]), models[i]))
    if not sets or sum(len(p) for p, _, _ in sets) < 300:
        return {"accepted": False, "why": "the arms barely changed this camera's view"}

    def fit(start: np.ndarray) -> tuple[np.ndarray, float, float]:
        pose = start.copy()
        for radius in (near_model_m, 0.08, 0.04, 0.02, 0.012):
            for _ in range(12):
                source, target = [], []
                for points, tree, model in sets:
                    moved = transform(points, pose)
                    distance, index = tree.query(moved, distance_upper_bound=radius)
                    found = np.isfinite(distance)
                    if found.sum() > 10:
                        keep = found & (distance <= np.quantile(distance[found], 0.8))
                        source.append(moved[keep])
                        target.append(model[index[keep]])
                if not source or sum(len(x) for x in source) < 50:
                    return pose, float("inf"), 0.0
                a, b = np.vstack(source), np.vstack(target)
                ca, cb = a.mean(0), b.mean(0)
                u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
                rotation = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
                step = np.eye(4)
                step[:3, :3], step[:3, 3] = rotation, cb - rotation @ ca
                pose = step @ pose
        distances = np.concatenate([tree.query(transform(points, pose), distance_upper_bound=0.05)[0]
                                    for points, tree, _ in sets])
        found = np.isfinite(distances)
        median = float(np.median(distances[found])) if found.any() else float("inf")
        return pose, median, float((distances < REGISTRATION_INLIER_M).mean())

    results = []
    for rough in rough_cam2world:
        for yaw in np.radians(np.arange(-20.0, 20.1, 5.0)):
            for dx in (-0.08, 0.0, 0.08):
                for dy in (-0.08, 0.0, 0.08):
                    start = np.asarray(rough, float).copy()
                    start[:3, :3] = _rz(yaw) @ start[:3, :3]
                    start[:3, 3] = start[:3, 3] + np.array([dx, dy, 0.0])
                    results.append(fit(start))
    results.sort(key=lambda item: (-item[2], item[1]))
    pose, median, inliers = results[0]
    rival = next((r for r in results[1:] if _distinct(r[0], pose) and r[2] > inliers * 0.9), None)
    accepted = median <= REGISTRATION_ACCEPT_MEDIAN_M and inliers >= REGISTRATION_ACCEPT_INLIERS and rival is None
    why = None
    if rival is not None:
        why = f"ambiguous: another pose {_pose_gap(rival[0], pose)} (m, deg) away keeps {rival[2]:.0%} inliers"
    elif not accepted:
        why = f"median {median * 1000:.1f} mm, inliers {inliers:.0%}: not a clean fit of the moving arms"
    return {"accepted": bool(accepted), "cam2world_cv": pose, "median_residual_m": median, "inlier_fraction": inliers,
            "arm_points": int(sum(len(p) for p, _, _ in sets)), "why": why}
