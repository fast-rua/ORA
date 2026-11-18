"""The Pi core's Context Artifact catalog and the Python one agree on an artifact's identity.

Pi runs register into the TypeScript catalog (``ora-pi/src/context/memory/artifacts.ts``);
``ora/agent/memory/catalog.py`` is its Python counterpart. A rule changed on one side only
shows up here as a different outcome or index: the storage kind label was once ignored in
Python while Pi code-mode runs kept faulting on it as identity drift.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.memory.catalog import ContextArtifactCatalog

ROOT = Path(__file__).resolve().parents[2]

# (operation, bytes, storage kind, suffix, media type, artifact kind, round, action step)
OPERATIONS = [
    # A script's last frame as a tool image (core.registerToolImages), then the next
    # observation publishing the same frame (InlineMediaCache.admitImage).
    ["register", "frame-1", "native_tool_image", ".png", "image/png", "context_image", None, 0],
    ["put", "frame-1", "context_image", ".png", "image/png", "context_image", 1, 1],
    # The observation first, then the tool image.
    ["put", "frame-2", "context_image", ".png", "image/png", "context_image", 1, 2],
    ["register", "frame-2", "native_tool_image", ".png", "image/png", "context_image", None, 2],
    # Any other difference stays drift: the artifact kind, the media type.
    ["register", "frame-1", "native_tool_image", ".png", "image/png", "context_tool_result", 1, 3],
    ["put", "frame-1", "context_image", ".img", "image/jpeg", "context_image", 1, 3],
]


def _python(root: Path) -> dict[str, Any]:
    artifacts = ArtifactStore(root)
    catalog = ContextArtifactCatalog(root, "policy-session-1", artifacts)
    outcomes: list[Any] = []
    for operation, data, storage_kind, suffix, media_type, kind, round_index, action_step in OPERATIONS:
        try:
            if operation == "register":
                storage = artifacts.put_bytes(data.encode(), kind=storage_kind, suffix=suffix, media_type=media_type)
                outcomes.append(catalog.register_storage_ref(
                    storage, kind=kind, round_index=round_index, action_step=action_step))
            else:
                outcomes.append(catalog.put_bytes(data.encode(), kind=kind, suffix=suffix, media_type=media_type,
                                                  round_index=round_index, action_step=action_step))
        except ValueError as error:
            outcomes.append({"error": str(error)})
    index = json.loads((root / "native/context-artifacts/index.json").read_text(encoding="utf-8"))
    return {"outcomes": outcomes, "index": index}


@pytest.mark.usefixtures("node_runtime")
def test_the_typescript_catalog_registers_like_the_python_one(tmp_path: Path) -> None:
    python = _python(tmp_path / "python")
    assert [outcome.get("error") for outcome in python["outcomes"]] == [
        None, None, None, None, "logical artifact identity drift", "logical artifact identity drift"]
    operations = tmp_path / "operations.json"
    operations.write_text(json.dumps(OPERATIONS), encoding="utf-8")
    result = subprocess.run(
        [shutil.which("node"), str(ROOT / "ora-pi/test/context_artifact_parity.ts"), str(operations),
         str(tmp_path / "typescript")],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    typescript = json.loads(result.stdout)
    assert typescript["outcomes"] == python["outcomes"]
    assert typescript["index"] == python["index"]
