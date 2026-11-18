"""The session's Context Artifact catalog: stable logical refs bound to
content-addressed storage, and the index persisted beside them.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ora.agent.evidence.canonical import ArtifactStore, canonical_bytes
from .index_persistence import BatchedIndexPersistence, atomic_json
from .artifact_openers import ARTIFACT_READ_NAMES, public_artifact_ref, recoverable_artifact_open
from .artifact_reader import (
    RefHashError,
    RefScopeError,
    RefStorageError,
    RefSyntaxError,
    RefUnknownError,
)
from .markers import CONTEXT_ARTIFACT_REF_SCHEMA, _REF_PATTERN


CONTEXT_ARTIFACT_INDEX_SCHEMA = "ora.context-artifact-index.v1"


_atomic_json = atomic_json  # historical private name


@dataclass
class ContextArtifactCatalog(BatchedIndexPersistence):
    """Bind storage refs to stable logical refs within one PolicySession."""

    run_root: Path
    policy_session_id: str
    artifacts: ArtifactStore
    reader_name: str = "artifact.read"

    def __post_init__(self) -> None:
        self.run_root = self.run_root.resolve()
        if self.reader_name not in ARTIFACT_READ_NAMES:
            raise ValueError("unsupported Context Artifact reader name")
        if _REF_PATTERN.fullmatch(
            f"artifact://{self.policy_session_id}/" + "0" * 64
        ) is None:
            raise ValueError("policy_session_id cannot form an artifact scope")
        self.index_path = self.run_root / "native/context-artifacts/index.json"
        self._entries: dict[str, dict[str, Any]] = {}
        self._persist()

    def _write_index(self) -> None:
        # Serialization never mutates the entries and sort_keys fixes their
        # order, so no defensive copy is needed.
        atomic_json(self.index_path, {
            "schema": CONTEXT_ARTIFACT_INDEX_SCHEMA,
            "policy_session_id": self.policy_session_id,
            "artifacts": self._entries,
        })

    def register_storage_ref(
        self,
        storage_ref: Mapping[str, Any],
        *,
        kind: str,
        round_index: int | None,
        action_step: int | None,
    ) -> dict[str, Any]:
        for label, value in (
            ("round_index", round_index), ("action_step", action_step),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"artifact {label} is invalid")
        digest = storage_ref.get("sha256")
        relative = storage_ref.get("path")
        media_type = storage_ref.get("media_type")
        byte_count = storage_ref.get("bytes")
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(relative, str)
            or not isinstance(media_type, str)
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError("storage artifact reference is invalid")
        path = (self.run_root / relative).resolve()
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.is_relative_to(self.run_root)
            or not path.is_file()
            or path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            or path.stat().st_size != byte_count
        ):
            raise ValueError("storage artifact failed path/hash validation")
        logical = f"artifact://{self.policy_session_id}/{digest}"
        existing = self._entries.get(logical)
        entry = {
            "schema": CONTEXT_ARTIFACT_REF_SCHEMA,
            "ref": logical,
            "sha256": digest,
            "kind": kind,
            "media_type": media_type,
            "bytes": byte_count,
            "policy_scope": self.policy_session_id,
            "sources": [{"round": round_index, "action_step": action_step}],
            "storage_ref": copy.deepcopy(dict(storage_ref)),
        }
        if existing is not None:
            immutable = {
                key: existing.get(key) for key in (
                    "schema", "ref", "sha256", "kind", "media_type", "bytes",
                    "policy_scope", "storage_ref",
                )
            }
            candidate = {key: entry.get(key) for key in immutable}
            # Storage is content-addressed: the same bytes stored as a tool image and as an
            # observation frame are one file; only the storage kind label differs.
            for value in (immutable, candidate):
                if isinstance(value.get("storage_ref"), Mapping):
                    value["storage_ref"] = {k: v for k, v in value["storage_ref"].items() if k != "kind"}
            if canonical_bytes(immutable) != canonical_bytes(candidate):
                raise ValueError("logical artifact identity drift")
            source = {"round": round_index, "action_step": action_step}
            sources = existing.setdefault("sources", [])
            if source not in sources:
                sources.append(source)
                sources.sort(key=lambda item: (
                    -1 if item.get("round") is None else int(item["round"]),
                    -1 if item.get("action_step") is None else int(item["action_step"]),
                ))
                self._persist()
            return self.public_ref(
                existing, round_index=round_index, action_step=action_step,
            )
        self._entries[logical] = entry
        self._persist()
        return self.public_ref(
            entry, round_index=round_index, action_step=action_step,
        )

    def put_bytes(
        self,
        data: bytes,
        *,
        kind: str,
        suffix: str,
        media_type: str,
        round_index: int | None,
        action_step: int | None,
    ) -> dict[str, Any]:
        storage = self.artifacts.put_bytes(
            data, kind=kind, suffix=suffix, media_type=media_type,
        )
        return self.register_storage_ref(
            storage, kind=kind, round_index=round_index, action_step=action_step,
        )

    def public_ref(
        self, entry: Mapping[str, Any], *, round_index: int | None = None,
        action_step: int | None = None,
    ) -> dict[str, Any]:
        return public_artifact_ref(entry, reader_name=self.reader_name,
                                   round_index=round_index, action_step=action_step)

    def resolve(self, logical_ref: str) -> tuple[dict[str, Any], bytes]:
        match = _REF_PATTERN.fullmatch(logical_ref)
        if match is None:
            raise RefSyntaxError("artifact reference syntax is invalid")
        if match.group("scope") != self.policy_session_id:
            raise RefScopeError("artifact reference is outside this PolicySession")
        entry = self._entries.get(logical_ref)
        if entry is None:
            raise RefUnknownError("artifact reference is unavailable")
        storage = entry.get("storage_ref")
        relative = storage.get("path") if isinstance(storage, Mapping) else None
        path = (self.run_root / str(relative)).resolve() if isinstance(relative, str) else None
        if (
            path is None
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.is_relative_to(self.run_root)
            or not path.is_file()
            or path.is_symlink()
        ):
            raise RefStorageError("artifact storage is unavailable")
        data = path.read_bytes()
        if (
            hashlib.sha256(data).hexdigest() != entry.get("sha256")
            or len(data) != entry.get("bytes")
            or match.group("sha256") != entry.get("sha256")
        ):
            raise RefHashError("artifact content hash does not match its reference")
        return copy.deepcopy(entry), data

    def public_refs_for_round(self, round_index: int | None) -> list[dict[str, Any]]:
        """List policy-readable refs first observed in one physical Episode."""

        if round_index is not None and (
            isinstance(round_index, bool) or not isinstance(round_index, int)
            or round_index < 0
        ):
            raise ValueError("Artifact round index is invalid")
        values: list[dict[str, Any]] = []
        for entry in self._entries.values():
            sources = entry.get("sources")
            if not isinstance(sources, list):
                continue
            matching = [
                source for source in sources
                if isinstance(source, Mapping) and source.get("round") == round_index
            ]
            if not matching:
                continue
            action_steps = [
                source.get("action_step") for source in matching
                if isinstance(source.get("action_step"), int)
                and not isinstance(source.get("action_step"), bool)
            ]
            reference = self.public_ref(
                entry,
                round_index=round_index,
                action_step=min(action_steps) if action_steps else None,
            )
            reference["open"] = recoverable_artifact_open(reference)
            values.append(reference)
        return sorted(values, key=lambda item: (
            -1 if item.get("action_step") is None else int(item["action_step"]),
            str(item.get("ref")),
        ))
