"""Deterministic saved layouts of ORA's RoboDojo play scenes (pure Python, no simulator).

Public RoboDojo only replays saved ``Eval_Layout`` files; its layout sampler
is not in the released tree. ``generate`` is a small stand-in: it reads one
scene YAML (RoboDojo task-config format,
``backends/robodojo/play_scenes/scenes/<set>/<scene>.yml``) and a
seed and writes a layout in the exact ``Eval_Layout`` record format, so the
unmodified ``SceneManager``/``LayoutManager`` replay path loads it
(``LayoutManager.set_saved_layout``). The same YAML, seed and pinned RoboDojo
assets give the same bytes on every host.

Rules:

- Table-level objects: origin x/y uniform in ``xlim``/``ylim`` (episode
  frame), yaw uniform in ``rotate_deg``, orientation = yaw about world z times
  the asset's ``place`` orientation (identity with ``place_tag: none``), z =
  table top + the rotated aligned box's lowest point + 2 mm (0.5 mm for an
  articulation, as RoboDojo's own layouts sit them). A ``zlim`` puts the
  bottom that high above the table instead. Footprints (convex hull of the
  rotated box) keep ``FOOTPRINT_GAP_M`` from each other and
  ``TABLE_EDGE_MARGIN_M`` from the table edge, and, with
  ``gripper_keep_out``, ``clearance_m`` from each zone (the home-pose gripper
  footprints of the two arms).
- Child objects (``relative_plane`` names another label): x/y offset in the
  parent's yawed frame, yaw relative to the parent, bottom 3 mm above the
  parent's box top, or at the parent's base + ``zlim`` when given.
- Fixtures (``Room``, ``Table``, ``Ground``, ``Background``) and the
  ``camera_stand`` geometry are copied from an official layout
  (``FIXTURE_LAYOUT``), so room, table, robot and cameras are RoboDojo's.

The placement report beside each layout records every object's spawn pose,
footprint and gripper clearance; it is qualification evidence, never policy
input.
"""

from __future__ import annotations

import copy
import json
import math
import random
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ora.environment.benchmarks.robodojo_play_scenes import TABLE_TOP_Z_M, TABLE_XY_M, X5_BASES_XY_M

GENERATOR = "ora.robodojo-play-layout-generator.v1"
#: The official layout whose fixtures (room, table, ground, background, camera stand) every scene reuses.
FIXTURE_LAYOUT = "Eval_Layout/RoboDojo/arx_x5/0/make_toast_0.json"
FIXTURE_SECTIONS = ("Room", "Table", "Ground", "Background")
OBJECT_ROOT = "Object/RoboDojo"
#: YAML block sections in the order they are placed; Clutter is written as cluttered Rigid records.
SECTIONS = ("Articulation", "Geometry", "Clutter", "Rigid")
TABLE_EDGE_MARGIN_M = 0.04
FOOTPRINT_GAP_M = 0.015
TABLE_DROP_M = 0.002
ARTICULATION_DROP_M = 0.0005
CHILD_DROP_M = 0.003
MAX_ATTEMPTS = 2000
#: The home-pose gripper footprints of the two ARX X5 arms projected on the table (x/y ranges): the
#: world bounding boxes of link6 (the gripper body), link7/link8 (the fingers) and the wrist camera
#: read from the stage at the home pose (fingertips reach y = -0.1947, inner edges x = +-0.2125),
#: extended back over the arm to its base (y -0.48). A scene's ``gripper_keep_out.zones`` may
#: restate them.
DEFAULT_GRIPPER_ZONES = ({"xlim": [-0.387, -0.212], "ylim": [-0.48, -0.194]},
                         {"xlim": [0.212, 0.388], "ylim": [-0.48, -0.194]})


class LayoutGenerationError(ValueError):
    """A scene YAML that cannot be placed, or names an asset the pinned Assets lack."""


# --- geometry ------------------------------------------------------------------------------------

def quaternion_multiply(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """The wxyz product ``a * b``."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


def normalized(q: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in q))
    if norm <= 0.0:
        raise LayoutGenerationError("a zero quaternion")
    return [value / norm for value in q]


def yaw_quaternion(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def rotation_matrix(q_wxyz: Sequence[float]) -> list[list[float]]:
    w, x, y, z = normalized(q_wxyz)
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def placed_box(vertices: Sequence[Sequence[float]], q_wxyz: Sequence[float]) -> dict[str, Any]:
    """An asset's aligned box turned by ``q_wxyz`` about its origin: lowest/highest z and x/y corners."""
    rotation = rotation_matrix(q_wxyz)
    world = [[sum(rotation[row][k] * vertex[k] for k in range(3)) for row in range(3)] for vertex in vertices]
    return {"zmin": min(point[2] for point in world), "zmax": max(point[2] for point in world),
            "xy": [(point[0], point[1]) for point in world]}


def hull(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """The convex hull (counter-clockwise) of 2-D points."""
    unique = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(unique) <= 2:
        return unique

    def cross(o: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def separation(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """The largest separating-axis gap between two convex polygons (negative when they overlap)."""
    best = -math.inf
    for polygon in (a, b):
        for index in range(len(polygon)):
            x1, y1 = polygon[index]
            x2, y2 = polygon[(index + 1) % len(polygon)]
            nx, ny = y2 - y1, x1 - x2
            norm = math.hypot(nx, ny)
            if norm < 1e-12:
                continue
            nx, ny = nx / norm, ny / norm
            pa = [px * nx + py * ny for px, py in a]
            pb = [px * nx + py * ny for px, py in b]
            best = max(best, max(min(pb) - max(pa), min(pa) - max(pb)))
    return best


def zone_polygon(zone: Mapping[str, Sequence[float]]) -> list[tuple[float, float]]:
    (x0, x1), (y0, y1) = zone["xlim"], zone["ylim"]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def footprint(vertices: Sequence[Sequence[float]], position: Sequence[float],
              q_wxyz: Sequence[float]) -> list[tuple[float, float]]:
    """The horizontal footprint of an asset box placed at ``position`` with ``q_wxyz``."""
    box = placed_box(vertices, q_wxyz)
    return hull([(position[0] + x, position[1] + y) for x, y in box["xy"]])


# --- the scene YAML -------------------------------------------------------------------------------

def load_scene(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LayoutGenerationError(f"{path} is not a scene mapping")
    return value


def gripper_keep_out(scene: Mapping[str, Any]) -> dict[str, Any]:
    """``{zones, clearance_m}``: the scene's keep-out (clearance None when the scene sets none)."""
    keep_out = scene.get("gripper_keep_out")
    if keep_out is None:
        return {"zones": [dict(zone) for zone in DEFAULT_GRIPPER_ZONES], "clearance_m": None}
    if not isinstance(keep_out, Mapping) or not isinstance(keep_out.get("clearance_m"), (int, float)):
        raise LayoutGenerationError("gripper_keep_out needs clearance_m (and optional zones)")
    zones = keep_out.get("zones") or [dict(zone) for zone in DEFAULT_GRIPPER_ZONES]
    for zone in zones:
        if (not isinstance(zone, Mapping) or len(zone.get("xlim") or ()) != 2 or len(zone.get("ylim") or ()) != 2):
            raise LayoutGenerationError("a gripper_keep_out zone needs xlim and ylim")
    return {"zones": [{"xlim": [float(v) for v in zone["xlim"]], "ylim": [float(v) for v in zone["ylim"]]}
                      for zone in zones],
            "clearance_m": float(keep_out["clearance_m"])}


def blocks(scene: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Every object block in placement order: a parent before the blocks placed on it."""
    pending = [(section, block) for section in SECTIONS for block in scene.get(section) or []]
    for section, block in pending:
        if (not isinstance(block, Mapping) or not isinstance(block.get("common"), Mapping)
                or not block.get("category") or not (block.get("select_mode") or {}).get("label")):
            raise LayoutGenerationError(f"malformed {section} block {block!r}")
    labels = [block["select_mode"]["label"][0] for _, block in pending]
    if len(set(labels)) != len(labels):
        raise LayoutGenerationError("scene labels must be unique")
    ordered: list[tuple[str, Mapping[str, Any]]] = []
    resolved: set[str] = set()
    while pending:
        progress = False
        for item in list(pending):
            parent = item[1]["common"].get("relative_plane", "Table")
            if parent == "Table" or parent in resolved:
                ordered.append(item)
                resolved.add(item[1]["select_mode"]["label"][0])
                pending.remove(item)
                progress = True
        if not progress:
            raise LayoutGenerationError("unresolved relative_plane: "
                                        + ", ".join(block["select_mode"]["label"][0] for _, block in pending))
    return ordered


def metadata_path(section: str, category: str, index: int) -> str:
    """Assets-relative metadata of one library model (Clutter models live under ``Clutter``)."""
    if not isinstance(category, str) or not category or "/" in category or category in {".", ".."}:
        raise LayoutGenerationError(f"invalid asset category {category!r}")
    if type(index) is not int or index < 0:
        raise LayoutGenerationError(f"invalid asset index {index!r}")
    return f"{OBJECT_ROOT}/{section}/{category}/{index:05d}/metadata.json"


def _yaw_range(common: Mapping[str, Any]) -> tuple[float, float]:
    if not common.get("rotate_rand", False):
        return (0.0, 0.0)
    degrees = common.get("rotate_deg", 0)
    if isinstance(degrees, (int, float)):
        return (-float(degrees), float(degrees))
    return (float(degrees[0]), float(degrees[1]))


def seeded_rng(scene: str, seed: int) -> random.Random:
    """The scene's own random stream for ``seed``: the same on every host and Python build."""
    return random.Random(zlib.crc32(f"{scene}:{seed}".encode()))


def generate(scene_spec: Mapping[str, Any], *, scene: str, seed: int, asset_root: Path) -> dict[str, Any]:
    """``{"layout", "placements", "keep_out", "metadata"}`` of one scene at one seed."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise LayoutGenerationError("a play-scene seed is a non-negative integer")
    assets = Path(asset_root)
    rng = seeded_rng(scene, seed)
    keep_out = gripper_keep_out(scene_spec)
    zones = [zone_polygon(zone) for zone in keep_out["zones"]]
    (table_x0, table_x1), (table_y0, table_y1) = TABLE_XY_M["x"], TABLE_XY_M["y"]
    placed: dict[str, dict[str, Any]] = {}
    layout: dict[str, Any] = {"Rigid": {}, "Articulation": {}, "Geometry": {}}
    placements: list[dict[str, Any]] = []
    metadata_files: dict[str, bytes] = {}
    for section, block in blocks(scene_spec):
        common = block["common"]
        category = block["category"][0]
        name = category["name"]
        index = rng.choice(list(category.get("index", [0])))
        label = block["select_mode"]["label"][0]
        relative = metadata_path(section, name, index)
        try:
            raw = (assets / relative).read_bytes()
        except OSError as exc:
            raise LayoutGenerationError(f"asset metadata {relative} is missing") from exc
        metadata_files[relative] = raw
        meta = json.loads(raw)
        vertices = ((meta.get("geometry") or {}).get("aligned_bbox") or {}).get("vertices")
        if not isinstance(vertices, list) or len(vertices) != 8:
            raise LayoutGenerationError(f"{relative} has no aligned bounding box")
        places = (meta.get("active") or {}).get("place") or {}
        tag = common.get("place_tag", "up")
        base_q = places[tag]["orientation"] if tag in places else [1.0, 0.0, 0.0, 0.0]
        parent_label = common.get("relative_plane", "Table")
        parent = placed.get(parent_label)
        low, high = _yaw_range(common)
        attempts = 0
        while True:
            attempts += 1
            if attempts > MAX_ATTEMPTS:
                raise LayoutGenerationError(f"{scene} seed {seed}: could not place {label}")
            yaw = rng.uniform(low, high) + (parent["yaw"] if parent else 0.0)
            q = normalized(quaternion_multiply(yaw_quaternion(yaw), base_q))
            box = placed_box(vertices, q)
            dx = rng.uniform(*common["xlim"])
            dy = rng.uniform(*common["ylim"])
            if parent is None:
                x, y = dx, dy
                drop = ARTICULATION_DROP_M if section == "Articulation" else TABLE_DROP_M
                z = TABLE_TOP_Z_M - box["zmin"] + drop
                if common.get("zlim") is not None:
                    z = TABLE_TOP_Z_M + float(common["zlim"][0]) - box["zmin"]
            else:
                c, s = math.cos(math.radians(parent["yaw"])), math.sin(math.radians(parent["yaw"]))
                x = parent["pos"][0] + c * dx - s * dy
                y = parent["pos"][1] + s * dx + c * dy
                if common.get("zlim") is not None:
                    z = parent["pos"][2] + parent["box"]["zmin"] + float(common["zlim"][0]) - box["zmin"]
                else:
                    z = parent["pos"][2] + parent["box"]["zmax"] - box["zmin"] + CHILD_DROP_M
            outline = hull([(x + px, y + py) for px, py in box["xy"]])
            if parent is None:
                xs, ys = [point[0] for point in outline], [point[1] for point in outline]
                if (min(xs) < table_x0 + TABLE_EDGE_MARGIN_M or max(xs) > table_x1 - TABLE_EDGE_MARGIN_M
                        or min(ys) < table_y0 + TABLE_EDGE_MARGIN_M or max(ys) > table_y1 - TABLE_EDGE_MARGIN_M):
                    continue
                if any(record["parent"] == "Table" and separation(outline, record["footprint"]) < FOOTPRINT_GAP_M
                       for record in placed.values()):
                    continue
                if (keep_out["clearance_m"] is not None
                        and min(separation(outline, zone) for zone in zones) < keep_out["clearance_m"]):
                    continue
            break
        placed[label] = {"pos": [x, y, z], "yaw": yaw, "box": box, "footprint": outline, "parent": parent_label}
        physics = dict(meta.get("physics") or {})
        physics.update(block.get("physics") or {})
        physics["type"] = ("articulation" if section == "Articulation"
                           else "geometry" if section == "Geometry" else "rigid")
        record = {
            "category": name, "category_idx": index, "group": None,
            "xlim": list(common["xlim"]), "ylim": list(common["ylim"]), "zlim": common.get("zlim"),
            "qpos": [1, 0, 0, 0], "rotate_deg": common.get("rotate_deg", 0),
            "rotate_rand": bool(common.get("rotate_rand", False)),
            "relative_plane": parent_label, "place_tag": tag if tag in places else None,
            "margin": 0.01, "check_mode": "bbox", "need_check_stable": True, "label": label,
            "default_pos": [x, y, z], "default_ori": q, "scale": [1.0, 1.0, 1.0],
            "physics": physics, "visual": {},
        }
        out_section = section
        if section == "Clutter":
            record["type"] = "cluttered"
            out_section = "Rigid"
        layout[out_section].setdefault(name, []).append(record)
        placements.append({
            "label": label, "section": section, "category": name, "category_idx": index,
            "metadata": relative, "relative_plane": parent_label,
            "spawn_pos": [round(value, 6) for value in (x, y, z)],
            "spawn_ori_wxyz": [round(value, 6) for value in q], "yaw_deg": round(yaw, 4),
            "sample_attempts": attempts,
            "footprint_xy": [[round(px, 6), round(py, 6)] for px, py in outline],
            "nearer_base_distance_m": round(min(math.dist((x, y), base) for base in X5_BASES_XY_M.values()), 6),
            "gripper_zone_clearance_m": (round(min(separation(outline, zone) for zone in zones), 6)
                                         if parent is None else None),
        })
    try:
        fixtures = json.loads((assets / FIXTURE_LAYOUT).read_text(encoding="utf-8"))
    except OSError as exc:
        raise LayoutGenerationError(f"fixture layout {FIXTURE_LAYOUT} is missing") from exc
    layout["Geometry"]["camera_stand"] = copy.deepcopy(fixtures["Geometry"]["camera_stand"])
    for key in FIXTURE_SECTIONS:
        layout[key] = copy.deepcopy(fixtures[key])
    layout = {key: value for key, value in layout.items() if value}
    return {"generator": GENERATOR, "scene": scene, "seed": seed, "layout": layout, "placements": placements,
            "keep_out": keep_out, "metadata": metadata_files}


def object_kinds(placements: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """label -> ``movable`` (a free rigid body a policy can pick up) or ``fixed`` (an articulation)."""
    return {item["label"]: "fixed" if item["section"] == "Articulation" else "movable" for item in placements}


__all__ = [
    "DEFAULT_GRIPPER_ZONES", "FIXTURE_LAYOUT", "GENERATOR", "LayoutGenerationError", "blocks", "footprint",
    "generate", "gripper_keep_out", "hull", "load_scene", "metadata_path", "object_kinds", "placed_box",
    "quaternion_multiply", "rotation_matrix", "seeded_rng", "separation", "zone_polygon",
]
