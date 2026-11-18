"""Code mode's extended library (``policy.code_mode.library: extended``): what its sandbox gets besides ``robot``.

When ``contract.get("library") == "extended"``, every script's and command's sandbox also has:

- ``ora_lib`` (``ora_lib/``: frames, projections and geometry; NumPy and SciPy only). Its files are bound
  read-only one by one at ``/ora/lib/ora_lib`` (``library_binds``). ``shell.py`` puts ``/ora/lib`` on a command's
  ``PYTHONPATH``.
- The current published observation, read-only at ``/ora/observation`` (``ObservationMount``). The control
  pointer's identity is resolved as the toolkit's ``geometry_query`` resolves it
  (``toolkit.host.publication.published_manifest``: ``<run>/policy-observations/<id>/observation.json``, which
  must have the pointer's hash). The public artifacts that tool reads are copied into a private directory:
  each camera's depth, calibration and RGB (decoded to an array), each file checked against the manifest's
  hash and size first, and the measured robot state. ``current.json`` (``ora.code-mode-observation.v1``)
  names them, and ``ora_lib.observation`` reads them. After each call of the script the mount is brought up
  to date before the answer returns. So when a call published a new observation (an action did), the script
  reads that one next.
- The public robot facts the policy's observation message carries for that observation, on a surface that delivers
  them (``current_public_robot_facts_v1``): the delivered projection of the same proprioception, read from the
  policy's current context (``inputs/current-context.json``), only when it names the mounted observation.
- The masks of the segmented targets recorded on that observation (``vision_ground`` and ``vision_segment``,
  the model's own calls included), each checked against its manifest and hash as the tools load a target
  (``load_segmented_target``): ``object_geometry`` measures them. A target belongs to the observation it was
  segmented on, so the next observation starts with none.

Any other contract gets neither, and its sandbox is exactly what it was.

A contract with ``source`` (``script_reads`` names it) also binds ORA's own code read-only under ``/ora/src``, file by
file from this checkout at its repository path (``source_files``), so what a model reads is what runs and nothing is kept
up by hand: the toolkit's tools and lib, the motion planner, every ``ora`` module they import, and each robot's public
geometry. The tools the model must not reason about from inside (``SOURCE_EXCLUDED_TOOLS``) stay out, and so does
everything nothing mounted imports: the evaluators, the simulator backends, the Supervisor. It is for reading:
``/ora/src`` is not on ``PYTHONPATH``, and a script still acts through ``robot``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

LIBRARY_ROOT = Path(__file__).with_name("ora_lib")
LIBRARY_MOUNT = "/ora/lib"
#: The directory that holds the ``ora`` package: a source file is mounted at its path under it.
SOURCE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_MOUNT = "/ora/src"
#: The directories whose every file (``SOURCE_SUFFIXES``) is mounted: the toolkit's tools and what they are built from,
#: and the motion planner that plans their moves and words their refusals.
SOURCE_DIRECTORIES = ("ora/agent/toolkit/lib", "ora/agent/toolkit/tools", "ora/environment/planners")
#: Each robot's public geometry, by pattern, so that a new robot needs no entry.
SOURCE_GEOMETRY = ("ora/policy/embodiments/*/manual-public-geometry-*.md",
                   "ora/policy/embodiments/*/descriptor-public-geometry-*.json")
#: Tools whose code stays out: they end the episode or report on it, so a model has nothing to learn there but how it
#: is judged.
SOURCE_EXCLUDED_TOOLS = frozenset({"check_task", "finish_episode", "report_issue"})
SOURCE_SUFFIXES = frozenset({".json", ".md", ".py"})
OBSERVATION_MOUNT = "/ora/observation"
SCHEMA = "ora.code-mode-observation.v1"
#: The tool runtime's artifact root in the run (``ManipulationToolRuntime.artifact_root``), where the tools record
#: a segmented target under its observation and camera (``target_store``).
TOOL_ARTIFACTS = "openeta-tool-artifacts"
TARGET_DIRECTORY = "segmentation-queries"
#: The policy's current context in the run (``context/observation/projector.py`` writes it before it opens the
#: observation), with the public robot facts the observation message carries.
CURRENT_CONTEXT = "sandbox/workspace/inputs/current-context.json"
#: The camera files ``geometry_query`` reads, by manifest role, and their names in the mount.
CAMERA_FILES = {"calibration": "calibration.json", "depth_numpy_m_f32": "depth-m.npy", "rgb_lossless_png": "rgb.npy"}
_NAME = re.compile(r"[^/.\\][^/\\]*")


def library_binds() -> list[str]:
    """``bwrap`` arguments binding ``ora_lib``'s modules read-only at ``/ora/lib/ora_lib``."""

    command: list[str] = []
    for path in sorted(LIBRARY_ROOT.glob("*.py")):
        command += ["--ro-bind", str(path), f"{LIBRARY_MOUNT}/ora_lib/{path.name}"]
    return command


def source_files() -> list[Path]:
    """The files a script may read: those of ``SOURCE_DIRECTORIES`` and ``SOURCE_GEOMETRY``, and the ``ora`` modules
    their Python imports, followed to the end (``_imports``), all but the excluded tools'. They are looked up on every
    call, so a new file is mounted with no list to update."""

    def kept(path: Path) -> bool:
        return not SOURCE_EXCLUDED_TOOLS & set(path.relative_to(SOURCE_ROOT).parts)

    found = {path for top in SOURCE_DIRECTORIES for path in (SOURCE_ROOT / top).rglob("*")
             if path.is_file() and path.suffix in SOURCE_SUFFIXES and "__pycache__" not in path.parts and kept(path)}
    found |= {path for pattern in SOURCE_GEOMETRY for path in SOURCE_ROOT.glob(pattern) if path.is_file()}
    pending = [path for path in found if path.suffix == ".py"]
    while pending:
        path = pending.pop()
        stat = path.stat()
        for module in _imports(path, SOURCE_ROOT, stat.st_mtime_ns, stat.st_size):
            if module not in found and kept(module):
                found.add(module)
                pending.append(module)
    return sorted(found)


@lru_cache(maxsize=None)
def _imports(path: Path, root: Path, _mtime_ns: int, _size: int) -> frozenset[Path]:
    """The files under ``root`` of the ``ora`` modules ``path`` imports: ``import ora.a.b`` names ``ora.a.b``; ``from
    ora.a import b``, or its relative form, names ``ora.a`` and, where it is a module, ``ora.a.b``. A name with no file
    (a class, a dynamic import) is skipped. Cached by the file's modification time and size."""

    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (SyntaxError, ValueError):
        return frozenset()
    package = path.relative_to(root).parts[:-1]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level <= len(package) + 1:
            base = package[:len(package) - node.level + 1] if node.level else ()
            module = ".".join([*base, *([node.module] if node.module else [])])
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names if alias.name != "*")
    found: set[Path] = set()
    for name in names:
        if name != "ora" and not name.startswith("ora."):
            continue
        stem = root.joinpath(*name.split("."))
        file = next((file for file in (stem.with_suffix(".py"), stem / "__init__.py") if file.is_file()), None)
        if file is not None:
            found.add(file)
    return frozenset(found)


def source_binds() -> list[str]:
    """``bwrap`` arguments binding ``source_files`` read-only under ``/ora/src``, each at its path under ``SOURCE_ROOT``."""

    command: list[str] = []
    for path in source_files():
        command += ["--ro-bind", str(path), f"{SOURCE_MOUNT}/{path.relative_to(SOURCE_ROOT).as_posix()}"]
    return command


class MountError(ValueError):
    """The current observation, or one of its cameras, cannot be mounted; ``code`` is the tools' error code."""

    def __init__(self, message: str, code: str = "artifact_integrity_fault") -> None:
        super().__init__(message)
        self.code = code


class ObservationMount:
    """A private, read-only copy of the current published observation for one script or command, at ``root``.

    ``refresh()`` follows the control pointer. When the observation changed, it copies the new one into a
    fresh numbered directory and replaces ``current.json`` atomically. Earlier copies stay until the mount
    closes, so a script still holding one can read it. When there is no current observation (none
    published, or closed since an action was submitted), ``current.json`` says why.
    """

    def __init__(self, control_root: Path, root: Path) -> None:
        self.control_root, self.root = Path(control_root), Path(root)
        self._identity: tuple[Any, ...] | None = None
        self._copies = 0
        self._index: dict[str, Any] = {}
        self._directory = ""

    def served(self, handle_call: Callable[[str, Mapping[str, Any]], Mapping[str, Any]]) -> Callable[..., Mapping[str, Any]]:
        """``handle_call``, with the mount brought up to date after each call, before its answer returns."""

        def call(name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
            try:
                return handle_call(name, arguments)
            finally:
                try:
                    self.refresh()
                except OSError:  # no copy could be written: the script finds none rather than a stale one
                    (self.root / "current.json").unlink(missing_ok=True)
        return call

    def refresh(self) -> None:
        try:
            pointer = json.loads((self.control_root / "current-observation.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pointer = {}
        identity = (pointer.get("observation_id"), pointer.get("observation_sha256"), pointer.get("open"))
        if identity != self._identity:
            try:
                index = self._copy(pointer)
            except Exception as exc:  # noqa: BLE001 - the script reads why instead of a stale or missing observation
                index = {"schema": SCHEMA, "observation_id": None,
                         "error": {"code": getattr(exc, "code", "artifact_integrity_fault"), "message": str(exc)}}
            self._index = index
            self._write(index)
            self._identity = identity
        self._mount_targets()

    def _write(self, index: Mapping[str, Any]) -> None:
        temporary = self.root / ".current.json.tmp"
        temporary.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.root / "current.json")

    def _mount_targets(self) -> None:
        """Add the masks of the targets recorded on the mounted observation since the last look to its index.

        A target the tool published but whose index record is not written yet is left for the next look; one that
        does not verify is listed under ``target_errors``, as a camera is under ``camera_errors``."""

        from ora.agent.toolkit.host.perception.target_store import load_segmented_target
        from ora.agent.toolkit.host.shared import artifact_errors

        index = self._index
        observation_id = index.get("observation_id")
        if not observation_id or index.get("error"):
            return
        artifacts = self.control_root.parent / TOOL_ARTIFACTS
        root = artifacts / "policy-observations" / observation_id
        mounted, errors = index.setdefault("targets", {}), index.setdefault("target_errors", {})
        added = False
        for path in sorted(root.glob(f"*/{TARGET_DIRECTORY}/target-*")):
            reference = path.name
            if reference in mounted or reference in errors:
                continue
            try:
                manifest, mask, _ = load_segmented_target(artifact_root=artifacts, target_ref=reference)
                identity = manifest.get("identity") or {}
                if identity.get("observation_id") != observation_id or not isinstance(identity.get("camera"), str):
                    raise MountError("the target is bound to another observation or camera")
                folder = self.root / self._directory / "targets"
                folder.mkdir(exist_ok=True)
                shutil.copyfile(mask, folder / f"{reference}.npy")
                mounted[reference] = {"camera": identity["camera"],
                                      "mask": f"{self._directory}/targets/{reference}.npy"}
            except artifact_errors.TargetRefUnknown:
                continue
            except Exception as exc:  # noqa: BLE001 - the script reads why this target has no mask
                errors[reference] = {"code": getattr(exc, "code", "artifact_integrity_fault"), "message": str(exc)}
            added = True
        if added:
            self._write(index)

    def _copy(self, pointer: Mapping[str, Any]) -> dict[str, Any]:
        from ora.agent.toolkit.interface.blobs import BlobStore
        from ora.agent.toolkit.host.publication import published_manifest

        if not pointer:
            raise MountError("no current observation", "environment_unavailable")
        if pointer.get("open") is not True:
            raise MountError("the current observation is closed: an action was submitted on it or its round ended, "
                             "and no fresh observation is published yet", "action_pending_fresh_observation")
        identity = {"observation_id": pointer.get("observation_id"),
                    "observation_sha256": pointer.get("observation_sha256")}
        source, manifest, _ = published_manifest(BlobStore(self.control_root.parent), identity)
        self._copies += 1
        name = self._directory = f"{self._copies:04d}"
        target = self.root / name
        target.mkdir()
        cameras: dict[str, Any] = {}
        errors: dict[str, Any] = {}
        for camera, record in (manifest.get("cameras") or {}).items():
            try:
                cameras[camera] = _copy_camera(source, target, name, camera, record)
            except MountError as exc:
                errors[camera] = {"code": exc.code, "message": str(exc)}
        (target / "proprioception.json").write_text(json.dumps(manifest.get("proprioception") or {}), encoding="utf-8")
        facts = _robot_facts(self.control_root.parent, identity["observation_id"])
        if facts is not None:
            (target / "robot-facts.json").write_text(json.dumps(facts), encoding="utf-8")
        step = (pointer.get("policy_state") or {}).get("step")
        return {"schema": SCHEMA, "observation_id": identity["observation_id"],
                "observation_sha256": identity["observation_sha256"],
                "step": step if isinstance(step, int) and not isinstance(step, bool) else None,
                "main_camera": manifest.get("main_camera"), "cameras": cameras, "camera_errors": errors,
                "proprioception": f"{name}/proprioception.json",
                **({"robot_facts": f"{name}/robot-facts.json"} if facts is not None else {})}


def _robot_facts(run_root: Path, observation_id: str) -> Mapping[str, Any] | None:
    """The public robot facts the policy's context carries for this observation; None on a surface that delivers
    none, or when the context has moved on to another observation."""

    try:
        facts = json.loads((run_root / CURRENT_CONTEXT).read_text(encoding="utf-8")).get("public_robot_facts")
    except (OSError, ValueError):
        return None
    return facts if isinstance(facts, Mapping) and facts.get("observation_id") == observation_id else None


def _copy_camera(source: Path, target: Path, name: str, camera: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """One camera's verified files into ``target/<camera>``; ``MountError`` when one is missing or drifted."""

    if not _NAME.fullmatch(camera) or not isinstance(record, Mapping):
        raise MountError("camera record is invalid")
    data = {role: _verified(source, record, role) for role in CAMERA_FILES}
    try:
        width, height = int(record["width"]), int(record["height"])
    except (KeyError, TypeError, ValueError):
        raise MountError("camera lacks its width and height") from None
    import numpy as np
    from PIL import Image

    try:
        with Image.open(BytesIO(data["rgb_lossless_png"])) as image:
            rgb = np.asarray(image.convert("RGB"))
    except Exception as exc:  # noqa: BLE001 - an unreadable image is that camera's error
        raise MountError(f"camera RGB does not decode: {exc}") from None
    folder = target / camera
    folder.mkdir()
    (folder / CAMERA_FILES["calibration"]).write_bytes(data["calibration"])
    (folder / CAMERA_FILES["depth_numpy_m_f32"]).write_bytes(data["depth_numpy_m_f32"])
    np.save(folder / CAMERA_FILES["rgb_lossless_png"], rgb, allow_pickle=False)
    return {"width": width, "height": height, "mount": record.get("mount"), "files": {
        "calibration": f"{name}/{camera}/{CAMERA_FILES['calibration']}",
        "depth_m": f"{name}/{camera}/{CAMERA_FILES['depth_numpy_m_f32']}",
        "rgb": f"{name}/{camera}/{CAMERA_FILES['rgb_lossless_png']}"}}


def _verified(source: Path, record: Mapping[str, Any], role: str) -> bytes:
    """A camera file's bytes, inside the observation root and matching the manifest's hash and size."""

    ref = (record.get("files") or {}).get(role)
    if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
        raise MountError(f"camera lacks {role}")
    root = source.resolve()
    path = (root / ref["path"]).resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise MountError(f"camera {role} is missing or escapes the observation root")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref.get("sha256") or ("bytes" in ref and len(data) != ref["bytes"]):
        raise MountError(f"camera {role} hash drift", "artifact_hash_mismatch")
    return data


@contextmanager
def observation_mount(contract: Mapping[str, Any], control_root: Path) -> Iterator[ObservationMount | None]:
    """The current observation's mount for one script or command; None unless the contract's library is
    ``extended``. The copy lives in a private temporary directory, removed when the block ends."""

    if contract.get("library") != "extended":
        yield None
        return
    root = Path(tempfile.mkdtemp(prefix="ora-code-observation-"))
    try:
        mount = ObservationMount(control_root, root)
        mount.refresh()
        yield mount
    finally:
        shutil.rmtree(root, ignore_errors=True)


__all__ = ["LIBRARY_MOUNT", "OBSERVATION_MOUNT", "ObservationMount", "SCHEMA", "SOURCE_EXCLUDED_TOOLS", "SOURCE_MOUNT",
           "library_binds", "observation_mount", "source_binds", "source_files"]
