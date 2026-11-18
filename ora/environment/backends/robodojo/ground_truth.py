"""Optional private ground-truth capture of a RoboDojo episode, for the developer side only.

``runtime.ground_truth_capture: true`` turns it on; absent or ``false`` leaves
the worker exactly as before. Nothing here is policy-visible: the recorder
returns nothing to the session's RPC methods, adds nothing to receipts,
observations or status, and writes its original capture outside the run.
The Supervisor may copy the latest already-written tick into the privileged
namespace of a state snapshot; public observation/context bytes stay unchanged.

Location: ``<workspace>/ground-truth/<run_id>/`` beside ``<workspace>/runs/``,
derived from the worker's ``shared_root`` (the run directory). A run directory
that is not ``.../runs/<run_id>`` is refused rather than guessed.

Contents:

* ``manifest.json``: schema, run id, frames and units, row and error counts;
  rewritten at every round start and at close. ``errors`` counts reads that
  failed as a whole; ``field_errors`` counts single fields (one object's pose
  or velocity, one material value) recorded as ``{"error": ...}`` inside an
  otherwise good row or round file, so ``errors == 0`` alone is not "every
  getter succeeded".
* ``physics/round-NNNN.json`` (C1): after each round reset, every layout
  instance's asset (USD path, model name, scale), configured physics block and
  the values read back from the simulator (mass, static and dynamic friction,
  restitution), the world pose of static props (geometry), the kinds that have
  no scene object at all (ground, room, table), and each articulation's joint
  names.
* ``ticks.jsonl`` (C2): one ``round_start`` row after each round reset, before
  any native action, then after every native action (10 physics steps), each
  rigid instance's env-relative pose as the official evaluator reads it and
  its linear and angular velocity, and articulation joint positions and
  velocities, keyed by round, action sequence, native action and physics clock.
  Static props and scene parts do not move and are not repeated per tick.

Every read is a getter; none writes simulator state. A reader failure is
recorded in the row and never raised: the episode must not depend on capture.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

CAPTURE_SCHEMA = "ora.robodojo-ground-truth-capture.v1"
TICK_SCHEMA = "ora.robodojo-ground-truth-tick.v1"
PHYSICS_SCHEMA = "ora.robodojo-ground-truth-physics.v1"
VISIBILITY = "developer_only_never_policy_visible"
POSITION_FRAME = "env_relative, as the official evaluator reads it (LayoutManager.get_instance_pose relative=True)"
ORIENTATION = "as returned by the upstream layout manager (wxyz), not converted"
VELOCITY_SOURCE = "isaacsim SingleRigidPrim.get_linear_velocity / get_angular_velocity, world frame, units as returned"
STATIC_POSE = "static props: isaacsim get_world_pose (world frame, wxyz), read once per round"
#: Layout kinds whose pose changes during an episode; the rest are read once per round.
TICK_KINDS = frozenset({"rigid"})
#: The manifest's row count is refreshed this often, so a worker that never closes still says how far it got.
MANIFEST_EVERY_ROWS = 50


class GroundTruthError(RuntimeError):
    """Ground-truth capture was requested but cannot be placed safely."""


def enabled(config: Mapping[str, Any]) -> bool:
    value = config.get("ground_truth_capture", False)
    if not isinstance(value, bool):
        raise GroundTruthError("runtime.ground_truth_capture must be true or false")
    return value


def capture_root(config: Mapping[str, Any]) -> Path:
    """``<workspace>/ground-truth/<run_id>``, outside the run directory by construction."""
    run_root = Path(str(config["shared_root"])).resolve()
    if run_root.parent.name != "runs":
        raise GroundTruthError(f"ground-truth capture needs a .../runs/<run_id> run directory, not {run_root}")
    root = run_root.parent.parent / "ground-truth" / run_root.name
    if root == run_root or run_root in root.parents:
        raise GroundTruthError("ground-truth capture must stay outside the run directory")
    return root


def _floats(value: Any) -> list[float]:
    value = _plain(value) if type(value).__name__ in ("ListConfig", "DictConfig") else value
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    flat: list[float] = []

    def walk(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
        else:
            flat.append(float(item))

    walk(value)
    return flat


def _field_errors(value: Any) -> list[str]:
    """The ``{"error": ...}`` entries nested anywhere in a reader's result."""
    found: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            if isinstance(item.get("error"), str):
                found.append(item["error"])
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)

    walk(value)
    return found


def _safe(read: Callable[[], Any]) -> Any:
    try:
        return read()
    except Exception as exc:  # noqa: BLE001 - a reader failure is evidence, never an episode failure
        return {"error": f"{type(exc).__name__}: {exc}"}


def _plain(value: Any) -> Any:
    """OmegaConf / numpy values as JSON."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    return json.loads(json.dumps(value, default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item)))


class SceneReaders:
    """Getter-only reads of the live RoboDojo scene (RoboDojo interpreter only)."""

    def __init__(self, session: Any) -> None:
        self.session = session

    @property
    def _scene(self) -> Any:
        return self.session.environment.scene_manager

    def instances(self) -> dict[str, str]:
        return {str(name): str(kind) for name, kind in sorted(self._scene.layout_manager.instance_type_by_env[0].items())}

    def _object(self, name: str) -> Any:
        return self._scene.layout_manager.get_scene_object(0, name)

    def _articulations(self) -> Mapping[str, Any]:
        return (getattr(self._scene, "_articulation_objects", None) or [{}])[0]

    def physics(self) -> dict[str, Any]:
        instances: dict[str, Any] = {}
        for name, kind in self.instances().items():
            obj = self._object(name)
            entry: dict[str, Any] = {"kind": kind}
            if obj is None:
                entry["scene_object"] = None
                instances[name] = entry
                continue
            entry["asset"] = {
                "usd_path": getattr(obj, "usd_path", None),
                "model_name": getattr(obj, "model_name", None), "model_id": getattr(obj, "model_id", None),
                "category": getattr(obj, "category_name", None),
                "scale": _safe(lambda: _floats(obj.scale)) if getattr(obj, "scale", None) is not None else None,
            }
            if getattr(obj, "physics_config", None) is not None:
                entry["configured_physics"] = _safe(lambda: _plain(obj.physics_config))
            readback: dict[str, Any] = {}
            if hasattr(obj, "get_mass"):
                readback["mass_kg"] = _safe(lambda: _floats(obj.get_mass())[0])
            material = getattr(obj, "physics_material", None)
            if material is not None:
                readback["static_friction"] = _safe(lambda: float(material.get_static_friction()))
                readback["dynamic_friction"] = _safe(lambda: float(material.get_dynamic_friction()))
                readback["restitution"] = _safe(lambda: float(material.get_restitution()))
            entry["simulator_readback"] = readback
            if kind not in TICK_KINDS and hasattr(obj, "get_world_pose"):
                def world_pose(obj: Any = obj) -> dict[str, Any]:
                    position, orientation = obj.get_world_pose()
                    return {"position_m": _floats(position)[:3], "orientation_wxyz": _floats(orientation)[:4]}
                entry["static_world_pose"] = _safe(world_pose)
            instances[name] = entry
        articulations = {
            str(name): {"dof_names": [str(item) for item in getattr(obj, "dof_names", None) or []]}
            for name, obj in self._articulations().items()
        }
        return {"instances": instances, "articulations": articulations}

    def tick(self) -> dict[str, Any]:
        layout = self._scene.layout_manager
        objects: dict[str, Any] = {}
        for name, kind in self.instances().items():
            if kind not in TICK_KINDS:
                continue
            entry: dict[str, Any] = {"kind": kind}
            try:
                position, orientation = layout.get_instance_pose(env_idx=0, inst_name=name, relative=True)
                entry["position_m"] = _floats(position)[:3]
                entry["orientation_wxyz"] = _floats(orientation)[:4]
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"{type(exc).__name__}: {exc}"
            obj = self._object(name)
            if obj is not None and hasattr(obj, "get_linear_velocity"):
                entry["linear_velocity"] = _safe(lambda: _floats(obj.get_linear_velocity())[:3])
                entry["angular_velocity"] = _safe(lambda: _floats(obj.get_angular_velocity())[:3])
            objects[name] = entry
        articulations: dict[str, Any] = {}
        for name, obj in self._articulations().items():
            articulations[str(name)] = {
                "joint_positions": _safe(lambda: _floats(obj.get_joint_positions())),
                "joint_velocities": _safe(lambda: _floats(obj.get_joint_velocities())),
            }
        return {"objects": objects, "articulations": articulations}


class GroundTruthRecorder:
    """Appends private ground truth beside the run; never raises into the episode."""

    def __init__(self, root: Path, *, run_id: str, readers: Any) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.readers = readers
        self.rows = 0
        self.errors = 0
        self.first_error: str | None = None
        self.field_errors = 0
        self.first_field_error: str | None = None
        self.rounds: list[dict[str, Any]] = []
        self.disabled_reason: str | None = None
        if self.root.exists() and any(self.root.iterdir()):
            raise GroundTruthError(f"refusing to write into a non-empty ground-truth directory: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _note_error(self, message: str) -> None:
        self.errors += 1
        self.first_error = self.first_error or message

    def _note_field_errors(self, value: Any) -> None:
        for message in _field_errors(value):
            self.field_errors += 1
            self.first_field_error = self.first_field_error or message

    def _write_manifest(self) -> None:
        manifest = {
            "schema": CAPTURE_SCHEMA, "run_id": self.run_id, "visibility": VISIBILITY,
            "ticks": {"path": "ticks.jsonl", "schema": TICK_SCHEMA, "rows": self.rows,
                      "cadence": "a round_start row after each round reset, then after every native action "
                                 "(10 physics steps)"},
            "physics": {"schema": PHYSICS_SCHEMA, "rounds": self.rounds},
            "frames": {"position": POSITION_FRAME, "orientation": ORIENTATION, "velocity": VELOCITY_SOURCE,
                       "static_props": STATIC_POSE},
            "errors": self.errors, "first_error": self.first_error,
            "field_errors": self.field_errors, "first_field_error": self.first_field_error,
            "disabled_reason": self.disabled_reason,
            "updated_unix_s": time.time(),
        }
        path = self.root / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    def round_started(self, *, round_index: int, layout_id: Any, physics_clock: int) -> None:
        if self.disabled_reason is not None:
            return
        try:
            body = self.readers.physics()
        except Exception as exc:  # noqa: BLE001
            body = {"error": f"{type(exc).__name__}: {exc}"}
            self._note_error(body["error"])
        else:
            self._note_field_errors(body)
        record = {"schema": PHYSICS_SCHEMA, "round_index": round_index, "layout_id": layout_id,
                  "physics_clock": physics_clock, **body}
        relative = f"physics/round-{round_index:04d}.json"
        try:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            self.rounds.append({"round_index": round_index, "layout_id": layout_id, "path": relative})
            self._write_manifest()
        except Exception as exc:  # noqa: BLE001 - capture stops; the episode goes on
            self.disabled_reason = f"{type(exc).__name__}: {exc}"

    def tick(self, *, round_index: int, action_sequence: int, action_id: str | None, segment: str | None,
             native_action: int, physics_clock: int) -> None:
        if self.disabled_reason is not None:
            return
        row: dict[str, Any] = {"round_index": round_index, "action_sequence": action_sequence,
                               "action_id": action_id, "segment": segment,
                               "native_action": native_action, "physics_clock": physics_clock}
        try:
            body = self.readers.tick()
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            self._note_error(row["error"])
        else:
            row.update(body)
            self._note_field_errors(body)
        try:
            with (self.root / "ticks.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            self.rows += 1
            if self.rows % MANIFEST_EVERY_ROWS == 0:
                self._write_manifest()
        except Exception as exc:  # noqa: BLE001
            self.disabled_reason = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        try:
            self._write_manifest()
        except Exception:  # noqa: BLE001 - nothing left to protect
            pass


def round_started(session: Any) -> None:
    """The session's round-start hook: C1, then the ``round_start`` C2 row; nothing when capture is off."""
    if session.ground_truth is not None:
        session.ground_truth.round_started(round_index=session.round_index, layout_id=session.layout_id,
                                           physics_clock=session._clock())
        tick(session, None, "round_start")


def tick(session: Any, action_id: str | None, segment: str | None) -> None:
    """The session's after-native-action hook; nothing when capture is off."""
    if session.ground_truth is not None:
        session.ground_truth.tick(round_index=session.round_index, action_sequence=session.action_sequence,
                                  action_id=action_id, segment=segment,
                                  native_action=int(session.environment.take_action_cnt[0]),
                                  physics_clock=session._clock())


def recorder_for(session: Any) -> GroundTruthRecorder | None:
    """The session's recorder when its config asks for one, else ``None``."""
    if not enabled(session.config):
        return None
    root = capture_root(session.config)
    return GroundTruthRecorder(root, run_id=root.name, readers=SceneReaders(session))


__all__ = [
    "CAPTURE_SCHEMA", "GroundTruthError", "GroundTruthRecorder", "PHYSICS_SCHEMA", "SceneReaders",
    "TICK_SCHEMA", "VISIBILITY", "capture_root", "enabled", "recorder_for", "round_started", "tick",
]
