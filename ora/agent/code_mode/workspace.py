"""The persistent code-mode workspace (plan ``policy.code_mode``): edits, reads and snapshots.

One directory per policy session, ``<run>/native/code-workspace``, bound read-write
at ``/work`` in every command's sandbox. ``write`` and ``edit`` apply their edits
here (all or none); ``read`` reads from it. After
each command the policy session records a snapshot: each file's path, size and
sha256, with its bytes stored once under ``<run>/native/code-workspace-objects``,
so the program the model wrote at every step stays in the run's evidence.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ora.agent.error_classes import ToolError

from .contract import MAX_EDITS, WORKSPACE_MAX_BYTES, WORKSPACE_MAX_FILES

DEFAULT_READ_LINES = 400


def workspace_root(run_root: Path) -> Path:
    root = Path(run_root) / "native" / "code-workspace"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _target(root: Path, path: Any) -> Path:
    """A workspace path, refused when it is absolute, climbs out, or names a link."""

    if not isinstance(path, str) or not path.strip():
        raise ToolError("a workspace path must be a non-empty string", code="invalid_arguments", retryable=True,
                        field="path")
    relative = PurePosixPath(path.strip().removeprefix("/work/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ToolError(f"{path} is outside /work", code="invalid_arguments", retryable=True, field="path")
    target = root.joinpath(*relative.parts)
    for parent in [target, *target.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ToolError(f"{path} goes through a link", code="invalid_arguments", retryable=True, field="path")
    return target


def apply_edits(root: Path, edits: Any) -> list[str]:
    """Apply ``edits`` in order, all or none; returns the paths changed."""

    if edits is None:
        return []
    if not isinstance(edits, Sequence) or isinstance(edits, (str, bytes)) or len(edits) > MAX_EDITS:
        raise ToolError(f"edits must be a list of at most {MAX_EDITS} edits", code="invalid_arguments",
                        retryable=True, field="edits")
    staged: dict[Path, str] = {}
    for index, edit in enumerate(edits):
        if not isinstance(edit, Mapping) or set(edit) - {"path", "content", "old_string", "new_string"}:
            raise ToolError(f"edit {index}: path with content, or path with old_string and new_string",
                            code="invalid_arguments", retryable=True, field="edits")
        target = _target(root, edit.get("path"))
        if "content" in edit:
            if not isinstance(edit["content"], str) or "old_string" in edit:
                raise ToolError(f"edit {index}: content must be a string, alone", code="invalid_arguments",
                                retryable=True, field="edits")
            staged[target] = edit["content"]
            continue
        old, new = edit.get("old_string"), edit.get("new_string", "")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ToolError(f"edit {index}: old_string must be non-empty text, new_string text",
                            code="invalid_arguments", retryable=True, field="edits")
        current = staged.get(target)
        if current is None:
            if not target.is_file():
                raise ToolError(f"edit {index}: {edit['path']} does not exist", code="edit_failed", retryable=True,
                                field="edits")
            current = target.read_text(encoding="utf-8")
        count = current.count(old)
        if count != 1:
            raise ToolError(f"edit {index}: old_string occurs {count} times in {edit['path']}, not once",
                            code="edit_failed", retryable=True, field="edits")
        staged[target] = current.replace(old, new, 1)
    for target, text in staged.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return sorted(str(target.relative_to(root)) for target in staged)


def files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink()
                  and "__pycache__" not in path.parts)


def check_bounds(root: Path) -> None:
    listed = files(root)
    size = sum(path.stat().st_size for path in listed)
    if len(listed) > WORKSPACE_MAX_FILES or size > WORKSPACE_MAX_BYTES:
        raise ToolError(f"the workspace holds {len(listed)} files and {size} bytes, over its limits "
                        f"({WORKSPACE_MAX_FILES} files, {WORKSPACE_MAX_BYTES} bytes); remove files first",
                        code="workspace_full", retryable=True)


def snapshot(root: Path, objects: Path) -> list[dict[str, Any]]:
    """Each file's path, size and sha256; its bytes are kept once under ``objects``."""

    objects.mkdir(parents=True, exist_ok=True)
    listed = []
    for path in files(root):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        stored = objects / digest
        if not stored.exists():
            shutil.copyfile(path, stored)
        listed.append({"path": str(path.relative_to(root)), "bytes": len(data), "sha256": digest})
    return listed


def read_file(root: Path, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """``read``'s result: numbered lines of one file, or the file list for an empty path."""

    if not isinstance(arguments, Mapping) or set(arguments) - {"path", "offset", "limit"}:
        raise ToolError("read_file takes path, offset and limit", code="invalid_arguments", retryable=True)
    path = arguments.get("path")
    if path in ("", ".", "/work", None):
        return {"ok": True, "path": "", "files": [{"path": str(item.relative_to(root)), "bytes": item.stat().st_size}
                                                  for item in files(root)]}
    target = _target(root, path)
    if not target.is_file():
        raise ToolError(f"{path} does not exist in /work", code="not_found", retryable=True, field="path")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    offset, limit = int(arguments.get("offset") or 1), int(arguments.get("limit") or DEFAULT_READ_LINES)
    shown = lines[offset - 1: offset - 1 + limit]
    return {"ok": True, "path": str(target.relative_to(root)), "lines": len(lines),
            "content": "\n".join(f"{offset + index:5d}  {line}" for index, line in enumerate(shown))}


__all__ = ["apply_edits", "check_bounds", "files", "read_file", "snapshot", "workspace_root"]
