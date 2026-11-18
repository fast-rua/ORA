"""The scene trace: every rigid body's mesh once and every body's world pose over a re-executed run.

``ora replay RUN --scene`` (``replay_program.export_scene_trace``) re-executes the run's replay program
as ``--check`` does and keeps the trace only when that check passes, so a sealed run gets its simulated
objects after the fact. The trace lives outside the run, the run's evidence and seal never change, and
nothing in it reaches a policy. The run dashboard draws it (``viewer/dashboard/scene_trace.py``)::

    <workspace>/scene-trace/<run_id>/     <workspace>: the directory holding runs/, like ground-truth/
      manifest.json      ora.scene-trace.v1: run id, source, backend, clock, the bodies, each action's span and,
                         when the exporter knows it, each arm's TCP as a body plus an offset (``tcp``)
      meshes/<id>.glb    one mesh per body: visual geometry in metres in the body's frame, one colour per
                         vertex (linear), at most ``MAX_FACES`` faces (vertex clustering)
      poses.bin          little-endian float32 [row, body, 7]: xyz, quaternion xyzw, in the manifest's body
                         order; NaN while a body is not in the round's scene
      steps.jsonl        one line per row: index, round, action_sequence, action_id, native_action (the
                         action's own steps so far), physics_step (since the round reset), sim_time_s
      replay-program-check.json   the check the re-execution passed

Rows are sampled: the round's first state, every ``clock.stride``-th physics step since the round reset
(about 50 Hz, the dashboard's rate: every 5th of RoboTwin's 250 Hz, every 10th of LIBERO-Pro's 500 Hz; RoboDojo's
every 10th of 250 Hz is one row per native action, 25 Hz, RoboCasa365's every 25th of 500 Hz one per control step, 20 Hz,
and RoboLab's every 8th of 120 Hz, 15 Hz) and each action's last step. ``physics_step`` is the execution capture's clock
(``clock.unit``: ``sapien_step_since_round_reset``, ``mujoco_step_since_round_reset``, ``physx_step_since_round_reset``),
or the receipts' on RoboLab, which has no capture (``physx_step_since_round_start``), and a row at step ``n`` is the
state after ``n`` steps as that simulator reports it. RoboCasa365 has no capture either; its steps are the control
steps a replay feeds, at 25 MuJoCo steps each.

This module is simulator-neutral and runs inside the isolated simulator worker with numpy: an exporter
hands the writer its round's bodies (``backends/robotwin/scene_trace.py`` reads them from SAPIEN,
``mujoco_scene_trace.py`` from MuJoCo, ``backends/robodojo/isaac_scene.py`` from any Isaac stage) and calls it
around every action and after every physics step.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Sequence

SCHEMA = "ora.scene-trace.v1"
DIRECTORY = "scene-trace"
MANIFEST = "manifest.json"
POSES = "poses.bin"
STEPS = "steps.jsonl"
CHECK_REPORT = "replay-program-check.json"
SOURCE = "replay_program"
#: Every Nth physics step becomes a row: RoboTwin's 250 Hz to the 50 Hz the dashboard's robot frames use.
STRIDE = 5
#: The most faces of one body's mesh, as the dashboard's other object meshes.
MAX_FACES = 20000
#: The grid vertex clustering starts from and the coarsest it goes to, cells along the longest side.
CLUSTER_CELLS = 256
CLUSTER_CELLS_MIN = 16
#: A static body longer than this is the world's backdrop (a wall, the ground), not a thing on the table.
BACKDROP_EXTENT_M = 2.0
#: In how many directions a static body must pass that length: a kitchen's long counter is one, its walls two.
BACKDROP_DIRECTIONS = 1
#: A textured part with fewer vertices shows its texture's mean: a few corners do not sample a picture.
MIN_SAMPLED_VERTICES = 256
#: ``Body.kind``; the writer turns a very large ``static`` body into ``backdrop``.
KINDS = ("object", "robot", "static", "backdrop")


class SceneTraceError(RuntimeError):
    """A scene trace cannot be written."""


class Mesh(NamedTuple):
    positions: Any  # (V, 3) metres, in the body's frame
    colors: Any     # (V, 4) uint8 linear RGBA
    faces: Any      # (F, 3) vertex indices


@dataclass(frozen=True)
class Body:
    """One rigid body of a round's scene. ``key`` names it across rounds (an actor, or articulation/link)."""

    key: str
    name: str
    kind: str
    read_pose: Callable[[], Sequence[float]]  # xyz + quaternion xyzw, world frame
    mesh: Mesh | None


def rotation(wxyz: Any) -> Any:
    """The 3x3 matrix of a unit quaternion."""
    import numpy as np

    w, x, y, z = (float(value) for value in wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def linear(srgb: Any) -> Any:
    """sRGB in 0..1 to linear, which is what a glTF vertex colour holds."""
    import numpy as np

    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def merge(meshes: Sequence[Mesh]) -> Mesh | None:
    """One mesh from a body's parts; None when they hold no face."""
    import numpy as np

    meshes = [mesh for mesh in meshes if len(mesh.faces)]
    if not meshes:
        return None
    offsets = np.cumsum([0, *(len(mesh.positions) for mesh in meshes[:-1])])
    return Mesh(np.concatenate([mesh.positions for mesh in meshes]), np.concatenate([mesh.colors for mesh in meshes]),
                np.concatenate([np.asarray(mesh.faces) + offset for mesh, offset in zip(meshes, offsets)]))


def simplify(mesh: Mesh, max_faces: int = MAX_FACES) -> Mesh:
    """Vertex clustering on a coarser and coarser grid until at most ``max_faces`` faces stay; winding kept."""
    import numpy as np

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) <= max_faces:
        return mesh
    positions = np.asarray(mesh.positions, dtype=np.float64)
    low = positions.min(axis=0)
    extent = float(np.ptp(positions, axis=0).max()) or 1.0
    cells = CLUSTER_CELLS
    while True:
        index = np.minimum(((positions - low) / extent * cells).astype(np.int64), cells - 1)
        keys = (index[:, 0] * cells + index[:, 1]) * cells + index[:, 2]
        _, cluster = np.unique(keys, return_inverse=True)
        cluster = cluster.ravel()
        corners = cluster[faces]
        corners = corners[(corners[:, 0] != corners[:, 1]) & (corners[:, 1] != corners[:, 2])
                          & (corners[:, 0] != corners[:, 2])]
        _, first = np.unique(np.sort(corners, axis=1), axis=0, return_index=True)
        corners = corners[np.sort(first)]
        if len(corners) <= max_faces or cells <= CLUSTER_CELLS_MIN:
            break
        cells = max(CLUSTER_CELLS_MIN, int(cells * 0.8))
    count = int(cluster.max()) + 1
    weight = np.maximum(np.bincount(cluster, minlength=count), 1).astype(np.float64)
    merged_positions = np.stack([np.bincount(cluster, positions[:, axis], count) / weight for axis in range(3)], axis=1)
    colors = np.asarray(mesh.colors, dtype=np.float64)
    merged_colors = np.stack([np.bincount(cluster, colors[:, channel], count) / weight for channel in range(4)], axis=1)
    return Mesh(merged_positions, np.clip(np.rint(merged_colors), 0, 255).astype(np.uint8), corners)


def encode_glb(mesh: Mesh) -> bytes:
    """A binary glTF 2.0 file: one primitive with positions, vertex colours and indices, double-sided."""
    import numpy as np

    positions = np.asarray(mesh.positions, dtype="<f4")
    chunks = ((positions.tobytes(), 34962), (np.asarray(mesh.colors, dtype=np.uint8).tobytes(), 34962),
              (np.asarray(mesh.faces, dtype="<u4").reshape(-1).tobytes(), 34963))
    binary, views = b"", []
    for data, target in chunks:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data), "target": target})
        binary += data + b"\0" * (-len(data) % 4)
    document = {
        "asset": {"version": "2.0", "generator": "ora.scene-trace"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"doubleSided": True, "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.8}}],
        "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "min": [float(value) for value in positions.min(axis=0)],
             "max": [float(value) for value in positions.max(axis=0)]},
            {"bufferView": 1, "componentType": 5121, "normalized": True, "count": len(positions), "type": "VEC4"},
            {"bufferView": 2, "componentType": 5125, "count": int(np.asarray(mesh.faces).size), "type": "SCALAR"},
        ],
    }
    text = json.dumps(document, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    return (struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(binary))
            + struct.pack("<II", len(text), 0x4E4F534A) + text
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


class SceneTraceWriter:
    """One run's trace in one directory: meshes as each round starts, everything else at ``finish``."""

    def __init__(self, directory: Path, *, run_id: str, backend: str, clock_unit: str, stride: int = STRIDE,
                 backdrop_directions: int = BACKDROP_DIRECTIONS) -> None:
        self.directory = Path(directory)
        self.meshes = self.directory / "meshes"
        self.meshes.mkdir(parents=True, exist_ok=True)
        self.run_id, self.backend, self.clock_unit, self.stride = run_id, backend, clock_unit, stride
        self.backdrop_directions = backdrop_directions
        self.records: list[dict[str, Any]] = []
        self.known: dict[tuple[str, str | None], int] = {}
        self.ids: set[str] = set()
        self.rows: list[tuple[list[int], Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.spans: list[dict[str, Any]] = []
        self.rounds: list[int] = []
        self.bodies: list[Body] = []
        self.columns: list[int] = []
        self.tcp: dict[str, Any] = {}  # arm -> {"body": key, "offset_xyz_xyzw": the TCP in that body's frame}
        self.timestep_s = 0.0
        self.round_index = 0
        self.physics_step = 0
        self.action: Mapping[str, Any] | None = None
        self.action_start = 0

    def _identifier(self, key: str) -> str:
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("._-") or "body"
        identifier, suffix = base, 1
        while identifier in self.ids:
            suffix += 1
            identifier = f"{base}-{suffix}"
        self.ids.add(identifier)
        return identifier

    def _column(self, body: Body) -> int:
        """The body's place in the manifest: one per key and mesh, so a body kept across rounds keeps its column."""
        import numpy as np

        if body.kind not in KINDS:
            raise SceneTraceError(f"unknown body kind {body.kind!r}")
        mesh = simplify(body.mesh) if body.mesh is not None else None
        data = encode_glb(mesh) if mesh is not None else None
        digest = hashlib.sha256(data).hexdigest() if data is not None else None
        known = self.known.get((body.key, digest))
        if known is not None:
            return known
        identifier = self._identifier(body.key)
        kind = body.kind
        if mesh is not None and kind == "static" and int(
                (np.ptp(mesh.positions, axis=0) > BACKDROP_EXTENT_M).sum()) >= self.backdrop_directions:
            kind = "backdrop"
        record = {"id": identifier, "name": body.name, "kind": kind, "mesh": None, "faces": 0}
        if data is not None:
            (self.meshes / f"{identifier}.glb").write_bytes(data)
            record.update(mesh=f"meshes/{identifier}.glb", faces=len(mesh.faces))
        self.known[(body.key, digest)] = len(self.records)
        self.records.append(record)
        return len(self.records) - 1

    def _row(self) -> None:
        import numpy as np

        action = self.action or {}
        self.steps.append({
            "index": len(self.steps), "round": self.round_index, "action_sequence": action.get("action_sequence"),
            "action_id": action.get("action_id"),
            "native_action": self.physics_step - self.action_start if self.action is not None else None,
            "physics_step": self.physics_step, "sim_time_s": self.physics_step * self.timestep_s,
        })
        self.rows.append((self.columns, np.asarray([body.read_pose() for body in self.bodies], dtype="<f4")))

    def begin_round(self, round_index: int, bodies: Sequence[Body], *, timestep_s: float) -> None:
        """A round's scene is ready: record its bodies and its first state."""
        self.round_index, self.timestep_s, self.physics_step = round_index, float(timestep_s), 0
        self.bodies = list(bodies)
        self.columns = [self._column(body) for body in self.bodies]
        self.rounds.append(round_index)
        self._row()

    def begin_action(self, entry: Mapping[str, Any]) -> None:
        self.action, self.action_start = entry, self.physics_step

    def after_step(self) -> None:
        """One physics step of the round ran."""
        self.physics_step += 1
        if self.physics_step % self.stride == 0:
            self._row()

    def end_action(self) -> None:
        """The action's last step always gets a row, and its span goes into the manifest."""
        if self.action is None:
            raise SceneTraceError("end_action without begin_action")
        if self.steps[-1]["physics_step"] != self.physics_step:
            self._row()
        self.spans.append({"round": self.round_index, "action_sequence": self.action.get("action_sequence"),
                           "action_id": self.action.get("action_id"), "first_step": self.action_start,
                           "last_step": self.physics_step})
        self.action = None

    def finish(self) -> dict[str, Any]:
        """Write the rows and the manifest; returns the manifest."""
        import numpy as np

        poses = np.full((len(self.rows), len(self.records), 7), np.nan, dtype="<f4")
        for index, (columns, values) in enumerate(self.rows):
            poses[index, columns] = values
        (self.directory / POSES).write_bytes(poses.tobytes())
        (self.directory / STEPS).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in self.steps),
                                            encoding="utf-8")
        manifest = {"schema": SCHEMA, "run_id": self.run_id, "source": SOURCE, "backend": self.backend,
                    "clock": {"unit": self.clock_unit, "timestep_s": self.timestep_s,
                              "hz": round(1.0 / self.timestep_s, 3) if self.timestep_s else None,
                              "stride": self.stride},
                    "shape": list(poses.shape), "rounds": self.rounds, "bodies": self.records, "spans": self.spans,
                    **({"tcp": self.tcp} if self.tcp else {})}
        (self.directory / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return manifest


class StepTrace:
    """Drives a ``SceneTraceWriter`` over a replay: the same step tap as the run's execution capture counts the
    steps from each round's reset, so the trace's clock is the capture's."""

    def __init__(self, writer: SceneTraceWriter) -> None:
        self.writer = writer
        self.tap: Any = None

    def watch(self, scene: Any) -> None:
        """Count the steps of this round's native scene."""
        from ora.environment.execution_capture import PhysicsStepTap

        self.tap = PhysicsStepTap(scene, self.writer.after_step)

    def begin_action(self, entry: Mapping[str, Any]) -> None:
        self.writer.begin_action(entry)

    def end_action(self) -> None:
        self.writer.end_action()

    def close(self) -> None:
        """Take the tap off; a step it could not read fails the trace (a missing row must not look like a pause)."""
        if self.tap is None:
            return
        tap, self.tap = self.tap, None
        tap.close()
        if tap.error is not None:
            raise SceneTraceError(f"reading the scene after a physics step failed: {tap.error}")

    def finish(self) -> dict[str, Any]:
        self.close()
        return self.writer.finish()


def traces_root(run_root: Path, traces: Path | None = None) -> Path:
    """The directory holding runs' traces: ``traces``, else ``scene-trace/`` beside the run's ``runs/``."""
    return Path(traces).resolve() if traces is not None else Path(run_root).resolve().parent.parent / DIRECTORY


__all__ = [
    "BACKDROP_DIRECTIONS", "BACKDROP_EXTENT_M", "CHECK_REPORT", "DIRECTORY", "KINDS", "MANIFEST", "MAX_FACES", "MIN_SAMPLED_VERTICES",
    "POSES", "SCHEMA", "SOURCE", "STEPS", "STRIDE", "Body", "Mesh", "SceneTraceError", "SceneTraceWriter", "StepTrace",
    "encode_glb", "linear", "merge", "rotation", "simplify", "traces_root",
]
