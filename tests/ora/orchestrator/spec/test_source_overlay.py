"""Source overlays: tool and tool-surface files replaced on a base commit, checked byte for byte."""
import json
import subprocess

import pytest

from ora.orchestrator.spec import source_overlay as so

TOOL = "ora/agent/toolkit/tools/move_to/tool.py"


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "ora"
    (root / "ora/agent/toolkit/tools/move_to").mkdir(parents=True)
    (root / TOOL).write_text("def run(api, args):\n    return {}\n")
    (root / "ora/orchestrator").mkdir(parents=True)
    (root / "ora/orchestrator/launch.py").write_text("# kernel\n")
    (root / ".gitignore").write_text(".ora-source-overlay.json\n")
    git(tmp_path, "init", "-q", str(root))
    git(root, "add", "-A")
    git(root, "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "base")
    return root


def write_overlay(root, **provenance):
    overlay = so.build_overlay(root, base_commit=git(root, "rev-parse", "HEAD"), provenance=provenance)
    (root / so.OVERLAY_FILE).write_text(json.dumps(overlay))
    return overlay


def test_a_clean_checkout_has_no_overlay(checkout):
    state = so.source_state(checkout)
    assert state.commit == git(checkout, "rev-parse", "HEAD") and not state.dirty and state.overlay is None


def test_replaced_and_added_tool_files_run_as_the_base_commit_with_their_overlay(checkout):
    (checkout / TOOL).write_text("def run(api, args):\n    return {'v': 2}\n")
    (checkout / "ora/agent/toolkit/tools/new_tool").mkdir()
    (checkout / "ora/agent/toolkit/tools/new_tool/tool.py").write_text("def run(api, args):\n    return {}\n")
    assert so.source_state(checkout).dirty  # without the overlay file it is a dirty checkout
    overlay = write_overlay(checkout, agent_repo="/w", agent_commit="c" * 40)
    state = so.source_state(checkout)
    assert not state.dirty and state.error is None
    assert set(state.overlay["files"]) == {TOOL, "ora/agent/toolkit/tools/new_tool/tool.py"}
    assert state.overlay["provenance"] == {"agent_repo": "/w", "agent_commit": "c" * 40}
    plan = {"source": {"commit": overlay["base_commit"], "dirty": False}, "source_overlay": state.overlay}
    assert so.matches_plan(state, plan)
    assert not so.matches_plan(state, {"source": plan["source"]})  # a plan without the overlay is another source


def test_any_other_difference_or_changed_byte_is_refused(checkout):
    (checkout / TOOL).write_text("v2\n")
    write_overlay(checkout)
    (checkout / TOOL).write_text("v3\n")
    assert "does not match" in so.source_state(checkout).error
    (checkout / TOOL).write_text("v2\n")
    (checkout / "ora/orchestrator/launch.py").write_text("# edited kernel\n")
    state = so.source_state(checkout)
    assert state.dirty and "outside its source overlay" in state.error


def test_only_tools_surfaces_and_the_lock_may_be_replaced():
    base = {"schema": so.SCHEMA, "base_commit": "a" * 40}
    so.validate_overlay({**base, "files": {"ora/policy/tool_surfaces/x/surface.json": "b" * 64, "policy.lock": None}})
    for path in ("ora/orchestrator/launch.py", "pyproject.toml", "ora/agent/toolkit/lib/x.py",
                 "ora/agent/toolkit/tools/../lib/x.py"):
        with pytest.raises(so.SourceOverlayError, match="may not replace"):
            so.validate_overlay({**base, "files": {path: "b" * 64}})


def test_the_overlay_must_sit_on_head(checkout):
    (checkout / TOOL).write_text("v2\n")
    overlay = write_overlay(checkout)
    (checkout / so.OVERLAY_FILE).write_text(json.dumps({**overlay, "base_commit": "f" * 40}))
    assert "is not HEAD" in so.source_state(checkout).error


def test_task_files_may_be_overlaid():
    so.validate_overlay({"schema": so.SCHEMA, "base_commit": "a" * 40,
                         "files": {"ora/policy/task_files/align_blocks.md": "b" * 64}})
