"""Context Artifact index persistence: atomic and batched.

The Session Artifact index is one JSON file rewritten whenever a reference is
registered. A lossless projection can register many references, so the
rewrite is coalesced to once per projection (``batched_persistence``); each
rewrite is fsynced before it replaces the index.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace ``path``, fsyncing the new content first."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class BatchedIndexPersistence:
    """Mixin: ``_persist`` writes now, or once when the outermost batch exits."""

    _batch_depth = 0
    _batch_dirty = False

    def _write_index(self) -> None:  # pragma: no cover - implemented by owner
        raise NotImplementedError

    def _persist(self) -> None:
        if self._batch_depth:
            self._batch_dirty = True
            return
        self._write_index()

    @contextlib.contextmanager
    def batched_persistence(self) -> Iterator[None]:
        """Coalesce index rewrites inside one projection into a single write.

        In-memory resolution is unchanged. The file is rewritten once on exit
        (also on error) when anything was registered, before the caller can
        publish any reference minted inside the block.
        """

        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._batch_dirty:
                self._batch_dirty = False
                self._write_index()
