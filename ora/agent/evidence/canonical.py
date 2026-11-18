"""Canonical JSON bytes and hashes, the trace error, and the content-addressed
artifact store every evidence writer shares.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class TrajectoryError(RuntimeError):
    """Canonical agent evidence is missing, corrupt or unsafe to export."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class ArtifactStore:
    """Store generated native-agent payloads by content hash."""

    run_root: Path

    def __post_init__(self) -> None:
        self.run_root = self.run_root.resolve()
        self.root = self.run_root / "native/agent-artifacts"
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(
        self,
        data: bytes,
        *,
        kind: str,
        suffix: str = ".bin",
        media_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        digest = sha256_bytes(data)
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        target = self.root / f"{digest}{safe_suffix}"
        if target.exists():
            if target.read_bytes() != data:
                raise TrajectoryError(f"artifact hash collision: {digest}")
        else:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=self.root,
            )
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    if target.read_bytes() != data:
                        raise TrajectoryError(f"artifact hash collision: {digest}")
            finally:
                temporary.unlink(missing_ok=True)
        return {
            "kind": kind,
            "path": target.relative_to(self.run_root).as_posix(),
            "sha256": digest,
            "bytes": len(data),
            "media_type": media_type,
        }

    def put_json(self, value: Any, *, kind: str) -> dict[str, Any]:
        return self.put_bytes(
            canonical_bytes(value), kind=kind, suffix=".json", media_type="application/json"
        )

    def reference(self, path: Path, *, kind: str, media_type: str) -> dict[str, Any]:
        resolved = path.resolve()
        if not resolved.is_file() or not resolved.is_relative_to(self.run_root):
            raise TrajectoryError(f"artifact reference escapes run: {path}")
        return {
            "kind": kind,
            "path": resolved.relative_to(self.run_root).as_posix(),
            "sha256": file_sha256(resolved),
            "bytes": resolved.stat().st_size,
            "media_type": media_type,
        }


def image_data_url(path: Path) -> str:
    """Return one deterministic PNG/JPEG provider content URL."""

    suffix = path.suffix.lower()
    media = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{media};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
