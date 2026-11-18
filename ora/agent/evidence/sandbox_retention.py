"""Whether a run keeps its ``sandbox/`` files after sealing, or only their hashes.

``sandbox/workspace`` is the execution model's working directory: its instructions, its contract and, above all,
the images handed to the model (``inputs/action-observations/``, ``inputs/current-visual/``), derived again after
every action. In a long RoboDojo episode it is over a gigabyte, almost all of it those images, which are derived
from the run's own ``policy-observations/``.

The run option ``save_sandbox`` (RunSpec and task file; default ``true``) chooses:

- ``true``: today's behaviour, every file stays.
- ``false``: the run is evaluated and sealed with its sandbox exactly as before (the seal inventory hashes every
  file), then :func:`omit_after_seal` writes :data:`MANIFEST` with each ``sandbox/`` file's path, size and sha256
  and removes the files. A plan records the option only when it is ``false``.

Readers that need the files say so: :func:`not_saved` names what is missing and where its hashes are. Export and
SFT export accept a sealed inventory record whose file is absent when :data:`MANIFEST` lists it with the same
sha256 and size (:func:`omitted_record`); re-evaluation refuses (the evaluation recorded at seal time stands).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

#: The RunSpec, task file and plan key.
OPTION = "save_sandbox"
#: The run-relative directory the option covers.
PREFIX = "sandbox/"
#: Written in place of the files: their paths, sizes and sha256.
MANIFEST = "sandbox/hashes.json"
SCHEMA = "ora.sandbox-hashes.v1"
#: A failed omission, recorded beside the run; the files stay.
REFUSAL = "native/sandbox-hashes.refusal.json"


class SandboxRetentionError(RuntimeError):
    """The sandbox cannot be reduced to its hashes, or a reader needs files that were not saved."""


def saves_sandbox(plan: Mapping[str, Any]) -> bool:
    """True unless the plan says ``save_sandbox: false``."""
    return plan.get(OPTION, True) is not False


def manifest(run_root: Path) -> dict[str, Any] | None:
    path = Path(run_root) / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def omitted_record(run_root: Path, record: Mapping[str, Any]) -> bool:
    """A sealed inventory record whose file was removed under ``save_sandbox: false`` (same sha256 and size)."""
    if not str(record.get("path", "")).startswith(PREFIX):
        return False
    body = manifest(run_root)
    if body is None:
        return False
    return any(item.get("path") == record["path"] and item.get("sha256") == record.get("sha256")
               and item.get("bytes") == record.get("bytes") for item in body.get("files", []))


def not_saved(run_root: Path, needed_by: str) -> str:
    """The message of a reader that needs sandbox files this run did not keep."""
    return (f"{needed_by} needs the sandbox files of run {Path(run_root).name}, which were not saved "
            f"(save_sandbox: false): {MANIFEST} keeps only their paths, sizes and sha256. "
            "Run with save_sandbox: true to keep them.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def omit(run_root: Path) -> dict[str, Any]:
    """Replace the sealed ``sandbox/`` files of a ``save_sandbox: false`` run by :data:`MANIFEST`."""
    run_root = Path(run_root).resolve()
    seal_path, plan_path = run_root / "seal.json", run_root / "plan.json"
    if not seal_path.is_file():
        return {"status": "unsealed"}
    if not plan_path.is_file() or saves_sandbox(json.loads(plan_path.read_text(encoding="utf-8"))):
        return {"status": "kept"}
    if (run_root / MANIFEST).exists():
        return {"status": "already_omitted"}
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    records = seal.get("inventory")
    if not isinstance(records, list):
        raise SandboxRetentionError("seal.json has no inventory")
    if any(item.get("path") == MANIFEST for item in records):
        raise SandboxRetentionError(f"{MANIFEST} is itself sealed evidence")
    files = sorted((dict(path=item["path"], bytes=item["bytes"], sha256=item["sha256"])
                    for item in records if str(item.get("path", "")).startswith(PREFIX)), key=lambda x: x["path"])
    for item in files:
        path = run_root / item["path"]
        if not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"] \
                or _sha256(path) != item["sha256"]:
            raise SandboxRetentionError(f"sealed sandbox file drifted before omission: {item['path']}")
    body = {
        "schema": SCHEMA, "run_id": run_root.name, "created_unix_s": time.time(), "option": {OPTION: False},
        "seal_inventory_sha256": seal.get("inventory_sha256"),
        "claim_boundary": "these files were sealed with these sha256 and sizes, then removed; their bytes are not "
                          "kept. The images among them derive from the run's policy-observations/.",
        "file_count": len(files), "bytes": sum(item["bytes"] for item in files), "files": files,
    }
    target = run_root / MANIFEST
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(body, indent=1) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    sandbox = run_root / PREFIX.rstrip("/")
    for directory, names, children in os.walk(sandbox, topdown=False):
        base = Path(directory)
        for name in children:
            candidate = base / name
            if candidate != target:
                candidate.unlink()
        for name in names:
            candidate = base / name
            if candidate.is_symlink():  # the scratch aliases (../AGENTS.md, ../inputs, ...)
                candidate.unlink()
            else:
                candidate.rmdir()
    return {"status": "omitted", "files": len(files), "bytes": body["bytes"]}


def omit_after_seal(run_root: Path) -> dict[str, Any]:
    """Landing hook: :func:`omit`, silent and fail-open (a refusal is recorded in :data:`REFUSAL`, files stay)."""
    try:
        return omit(run_root)
    except Exception as exc:  # noqa: BLE001 - recorded beside the run, never raised into landing
        try:
            refusal = Path(run_root) / REFUSAL
            refusal.parent.mkdir(parents=True, exist_ok=True)
            refusal.write_text(json.dumps({"schema": "ora.sandbox-hashes-refusal.v1", "run_id": Path(run_root).name,
                                           "reason": str(exc)}) + "\n", encoding="utf-8")
        except OSError:
            pass
        return {"status": "refused", "reason": str(exc)}


__all__ = ["MANIFEST", "OPTION", "PREFIX", "REFUSAL", "SCHEMA", "SandboxRetentionError", "manifest",
           "not_saved", "omit", "omit_after_seal", "omitted_record", "saves_sandbox"]
