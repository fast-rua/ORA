"""Digests of what the toolkit answers, recorded while the legacy tool handlers were its oracle.

A test module that compared a toolkit call with the handler it replaced keeps one file under
``tests/data/toolkit-goldens/``, at the module's path below ``tests/``. It maps a key the test names to the sha256 of
the value's canonical JSON, with every float rounded to 10 significant digits (BLAS kernels differ in the last bit)
and the test's own directories replaced by ``<root>``. Each digest was recorded with the handler's answer still
compared and equal, in the change that removed those comparisons.

A deliberate change to what a tool answers re-records its module: ``ORA_RECORD_TOOLKIT_GOLDENS=1`` writes the digests
of the calls that run, instead of checking them.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "tests/data/toolkit-goldens"
SCHEMA = "ora.toolkit-goldens.v1"
RECORDING = os.environ.get("ORA_RECORD_TOOLKIT_GOLDENS") == "1"


def rounded(value: Any) -> Any:
    """``value`` with each float at 10 significant digits."""
    if isinstance(value, float):
        return float(f"{value:.10g}")
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    return value


def plain(value: Any) -> Any:
    """``value`` as JSON: bytes holding JSON as that value, other bytes (an image) as their sha256."""
    if isinstance(value, (bytes, bytearray)):
        try:
            return plain(json.loads(value))
        except ValueError:
            return {"sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def digest(value: Any, roots: Iterable[Path | str] = ()) -> str:
    text = json.dumps(rounded(plain(value)), sort_keys=True, separators=(",", ":"), allow_nan=False)
    for root in sorted({str(root) for root in roots}, key=len, reverse=True):
        text = text.replace(root, "<root>")
    return hashlib.sha256(text.encode()).hexdigest()


class Goldens:
    """The recorded digests of one test module."""

    def __init__(self, test_file: str) -> None:
        relative = Path(test_file).resolve().relative_to(ROOT / "tests")
        self.path = DIRECTORY / relative.with_suffix(".json")

    def _recorded(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))["digests"]

    def check(self, key: str, value: Any, roots: Iterable[Path | str] = ()) -> None:
        """``value`` is what the toolkit answered for ``key`` when its handler was the oracle."""
        found = digest(value, roots)
        if RECORDING:
            self._record(key, found)
            return
        recorded = self._recorded()
        assert key in recorded, f"{self.path.name} records no {key!r}"
        assert found == recorded[key], f"{key}: the toolkit answers otherwise than it did beside its handler"

    def checker(self, request: Any) -> Any:
        """``check(label, value, *roots)`` for one test: its key is the test's name, ``label`` and its parameter ids."""
        callspec = getattr(request.node, "callspec", None)
        suffix = f"[{callspec.id}]" if callspec is not None else ""
        prefix = f"{request.node.originalname}/"
        return lambda label, value, *roots: self.check(prefix + label + suffix, value, roots)

    def _record(self, key: str, found: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(str(self.path).encode()).hexdigest()[:16]
        with open(Path(tempfile.gettempdir()) / f"ora-toolkit-goldens-{name}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            digests = {**self._recorded(), key: found}
            self.path.write_text(json.dumps({"schema": SCHEMA, "digests": dict(sorted(digests.items()))},
                                            indent=1) + "\n", encoding="utf-8")
