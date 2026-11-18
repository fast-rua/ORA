"""Source overlays: a run on a base commit with some tool and tool-surface files replaced.

A checkout normally runs only as a clean commit. An overlay lets an outside
author (an RSI agent's evaluation) replace files under ``OVERLAY_ROOTS`` on top
of one committed base without a new commit and without re-qualifying anything:
the roots hold no file a qualification record hashes (the RoboDojo worker and
play kernels, ``action_executor.py``, ``pyproject.toml``, ``uv.lock``, the
RoboTwin seed-scan inputs).

The checkout states its overlay in ``.ora-source-overlay.json`` at its root
(ignored by git; ``ora.source-overlay.v1``)::

    {"schema": "ora.source-overlay.v1", "base_commit": "<40 hex>",
     "files": {"<path>": "<sha256>" | null, ...},   # null: removed from the base
     "provenance": {...}}                             # who authored it, as given

``source_state`` accepts such a checkout only when HEAD is the base commit and
every difference from HEAD (tracked, staged or untracked) is a listed file
whose bytes match. The plan keeps ``source`` as ``{commit, dirty}`` (the base
commit, clean) and records the overlay whole, with its sha256, in a separate
``source_overlay`` block that the plan hash covers; launch, the Supervisor and
sealing require the same overlay again.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, NamedTuple

SCHEMA = "ora.source-overlay.v1"
OVERLAY_FILE = ".ora-source-overlay.json"
#: Where an overlay may replace files: the toolkit's tools, the tool surfaces and the task files
#: (policy.task_file), plus the lock over them.
OVERLAY_ROOTS = ("ora/agent/toolkit/tools/", "ora/policy/tool_surfaces/", "ora/policy/task_files/")
OVERLAY_FILES = ("policy.lock",)
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class SourceOverlayError(ValueError):
    """The checkout's overlay is malformed or does not match the checkout."""


class SourceState(NamedTuple):
    commit: str | None
    dirty: bool
    overlay: dict[str, Any] | None
    error: str | None = None


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60, check=False)


def allowed(path: str) -> bool:
    return path in OVERLAY_FILES or (path.startswith(OVERLAY_ROOTS) and "/../" not in path and not path.endswith("/"))


def validate_overlay(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise SourceOverlayError(f"{OVERLAY_FILE} is not {SCHEMA}")
    if set(value) - {"schema", "base_commit", "files", "provenance"}:
        raise SourceOverlayError(f"{OVERLAY_FILE} has unknown keys")
    if not isinstance(value.get("base_commit"), str) or not _HEX40.fullmatch(value["base_commit"]):
        raise SourceOverlayError("source overlay base_commit must be a full commit hash")
    files = value.get("files")
    if not isinstance(files, Mapping) or not files:
        raise SourceOverlayError("source overlay files must list at least one path")
    for path, digest in files.items():
        if not isinstance(path, str) or not allowed(path):
            raise SourceOverlayError(f"source overlay may not replace {path!r}; roots: {OVERLAY_ROOTS + OVERLAY_FILES}")
        if digest is not None and not (isinstance(digest, str) and _HEX64.fullmatch(digest)):
            raise SourceOverlayError(f"source overlay sha256 of {path} is invalid")
    if not isinstance(value.get("provenance", {}), Mapping):
        raise SourceOverlayError("source overlay provenance must be an object")
    return {"schema": SCHEMA, "base_commit": value["base_commit"],
            "files": {str(path): files[path] for path in sorted(files)},
            "provenance": dict(value.get("provenance", {}))}


def record(overlay: Mapping[str, Any]) -> dict[str, Any]:
    """The plan's ``source_overlay`` block: the overlay and the sha256 of its canonical bytes."""
    canonical = json.dumps(overlay, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {**overlay, "sha256": hashlib.sha256(canonical).hexdigest()}


def _changed(root: Path) -> list[str]:
    out = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames")
    if out.returncode:
        raise SourceOverlayError(out.stderr.strip() or "git status failed")
    return sorted({entry[3:] for entry in out.stdout.split("\0") if entry})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_state(root: Path) -> SourceState:
    """HEAD, whether the checkout differs from it beyond its overlay, and the overlay's plan record."""

    head = _git(root, "rev-parse", "HEAD")
    if head.returncode:
        return SourceState(None, True, None, head.stderr.strip() or "git rev-parse failed")
    commit = head.stdout.strip()
    path = root / OVERLAY_FILE
    try:
        changed = _changed(root)
        if not path.is_file():
            return SourceState(commit, bool(changed), None)
        overlay = validate_overlay(json.loads(path.read_text(encoding="utf-8")))
        if overlay["base_commit"] != commit:
            raise SourceOverlayError(f"source overlay base {overlay['base_commit'][:12]} is not HEAD {commit[:12]}")
        unlisted = [name for name in changed if name not in overlay["files"]]
        if unlisted:
            raise SourceOverlayError(f"checkout differs outside its source overlay: {unlisted[:5]}")
        for name, digest in overlay["files"].items():
            target = root / name
            if (digest is None) != (not target.is_file()) or (digest is not None and _sha256(target) != digest):
                raise SourceOverlayError(f"source overlay file {name} does not match the checkout")
    except (OSError, ValueError) as exc:
        return SourceState(commit, True, None, str(exc))
    return SourceState(commit, False, record(overlay))


def matches_plan(state: SourceState, plan: Mapping[str, Any]) -> bool:
    """The checkout is the plan's clean base commit with exactly the plan's overlay (or none)."""
    source = plan.get("source") or {}
    return (state.error is None and state.commit == source.get("commit") and not state.dirty
            and state.overlay == plan.get("source_overlay"))


def build_overlay(root: Path, *, base_commit: str, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The overlay of a checkout as it now differs from HEAD (for the author that prepared it)."""
    files = {}
    for name in _changed(root):
        if name == OVERLAY_FILE:
            continue
        target = root / name
        files[name] = _sha256(target) if target.is_file() else None
    return validate_overlay({"schema": SCHEMA, "base_commit": base_commit, "files": files,
                             "provenance": dict(provenance or {})})
