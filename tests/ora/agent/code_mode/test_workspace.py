"""The persistent code-mode workspace: edits apply all or none, paths stay in /work, files read back."""

from __future__ import annotations

import pytest

from ora.agent.code_mode import workspace
from ora.agent.error_classes import ToolError


def test_edits_apply_in_order_and_all_or_none(tmp_path) -> None:
    root = tmp_path
    assert workspace.apply_edits(root, [{"path": "policy.py", "content": "a = 1\nb = 2\n"},
                                        {"path": "policy.py", "old_string": "b = 2", "new_string": "b = 3"}]) \
        == ["policy.py"]
    assert (root / "policy.py").read_text() == "a = 1\nb = 3\n"
    with pytest.raises(ToolError, match="occurs 0 times"):
        workspace.apply_edits(root, [{"path": "other.py", "content": "x"},
                                     {"path": "policy.py", "old_string": "missing", "new_string": ""}])
    assert not (root / "other.py").exists(), "a failed edit leaves the workspace as it was"
    for bad in ("/etc/passwd", "../outside.py", "a/../../b"):
        with pytest.raises(ToolError, match="outside /work"):
            workspace.apply_edits(root, [{"path": bad, "content": "x"}])


def test_read_file_numbers_lines_and_lists_files(tmp_path) -> None:
    workspace.apply_edits(tmp_path, [{"path": "lib/geo.py", "content": "one\ntwo\nthree\n"}])
    read = workspace.read_file(tmp_path, {"path": "/work/lib/geo.py", "offset": 2, "limit": 1})
    assert read["lines"] == 3 and read["content"] == "    2  two"
    assert workspace.read_file(tmp_path, {"path": ""})["files"] == [{"path": "lib/geo.py", "bytes": 14}]
    snap = workspace.snapshot(tmp_path, tmp_path.parent / "objects")
    assert snap[0]["path"] == "lib/geo.py" and (tmp_path.parent / "objects" / snap[0]["sha256"]).is_file()
