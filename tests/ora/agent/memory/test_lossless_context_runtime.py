"""Recoverable Context Artifacts of the lossless Context.

The projection cases of this module drove the Python Context pipeline, which ORA
removed with the Native core; they were recorded as Context fixtures
(``tests/data/context_differential/captured_runtime_*``) that the TypeScript
pipeline replays. The ActionStep inventory cases went with that pipeline's
remaining test-only Python code; the inventory is the TypeScript pipeline's.
"""

from __future__ import annotations

from ora.agent.error_classes import ToolError

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

pytestmark = pytest.mark.usefixtures("ready_model_services")

from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.memory.catalog import ContextArtifactCatalog
from ora.agent.memory.tool_runtime import RecoverableArtifactToolRuntime
from ora.agent.memory.markers import (
    decode_recoverable_artifact_marker,
    encode_recoverable_artifact_marker,
    recoverable_artifact_envelope,
    recoverable_artifact_pointer,
    validate_recoverable_artifact_pointer,
)


@dataclass
class _Surface:
    result_schemas: Mapping[str, Any] = field(default_factory=dict)
    mutation_tools: tuple[str, ...] = (
        "move_pose", "close_gripper", "open_gripper",
    )

    @staticmethod
    def supports(_capability: str) -> bool:
        return False


@dataclass
class _BaseRuntime:
    surface: _Surface = field(default_factory=_Surface)
    control_root: Path = Path("control")
    allowed_tools: tuple[str, ...] = ("observe",)
    reset_rounds: list[int] = field(default_factory=list)
    canonical_reader: bool = False

    def definitions(self) -> list[dict[str, Any]]:
        extra = [{"name": "vision_segment", "description": "fixture", "inputSchema": {"type": "object", "properties": {}}}] if self.canonical_reader else []
        return extra + [{
            "name": "observe", "description": "observe",
            "inputSchema": {
                "type": "object", "properties": {},
                "additionalProperties": False,
            },
        }]

    def classify_call_effect(self, name: str, arguments: Mapping[str, Any]) -> str:
        return "read_only"

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"status": "actual"}

    def reset_round(self, round_index: int) -> None:
        self.reset_rounds.append(round_index)


def test_a_tool_image_and_an_observation_frame_with_the_same_bytes_are_one_artifact(tmp_path: Path) -> None:
    """A script's observe returns the frame the next observation publishes: one content-addressed file."""
    root = tmp_path / "run"
    artifacts = ArtifactStore(root)
    catalog = ContextArtifactCatalog(root, "policy-session-1", artifacts)
    tool_image = artifacts.put_bytes(b"frame", kind="native_tool_image", suffix=".png", media_type="image/png")
    first = catalog.register_storage_ref(tool_image, kind="context_image", round_index=1, action_step=2)
    second = catalog.put_bytes(b"frame", kind="context_image", suffix=".png", media_type="image/png",
                               round_index=1, action_step=3)
    assert first["ref"] == second["ref"]
    with pytest.raises(ValueError, match="identity drift"):
        catalog.register_storage_ref(tool_image, kind="context_tool_result", round_index=1, action_step=4)


def test_artifacts_recover_exact_bytes_and_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "run"
    artifacts = ArtifactStore(root)
    catalog = ContextArtifactCatalog(root, "policy-session-1", artifacts)
    image = catalog.put_bytes(
        b"exact-image", kind="context_image", suffix=".png",
        media_type="image/png", round_index=1, action_step=2,
    )
    document = catalog.put_bytes(
        b'{"outer":{"value":"exact"}}', kind="context_tool_result",
        suffix=".json", media_type="application/json", round_index=1,
        action_step=2,
    )
    long_text = catalog.put_bytes(
        b"0123456789", kind="context_tool_result", suffix=".txt",
        media_type="text/plain", round_index=2, action_step=8,
    )
    legacy_envelope = recoverable_artifact_envelope(
        image, source={
            "origin": "tool_image", "message_role": "user",
            "tool": "vision.segment", "tool_call_id": "segment-0001",
            "image_index": 0,
        },
    )
    pointer = recoverable_artifact_pointer(image)
    marker = encode_recoverable_artifact_marker(pointer)
    legacy_marker = (
        "<recoverable_artifact>"
        + json.dumps(
            legacy_envelope, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        )
        + "</recoverable_artifact>"
    )
    assert decode_recoverable_artifact_marker(marker) == pointer
    assert decode_recoverable_artifact_marker(legacy_marker) == pointer
    assert len(marker.encode("utf-8")) < len(legacy_marker.encode("utf-8")) * 0.55
    with pytest.raises(ValueError, match="pointer fields"):
        validate_recoverable_artifact_pointer({**pointer, "path": "/host/private"})
    with pytest.raises(ValueError, match="pointer is malformed"):
        validate_recoverable_artifact_pointer({
            **pointer, "open": "artifact.read",
        })
    unicode_text = catalog.put_bytes(
        b"a" * 4095 + "中".encode("utf-8"), kind="context_tool_result",
        suffix=".txt", media_type="text/plain", round_index=2, action_step=9,
    )
    events: list[dict[str, Any]] = []
    runtime = RecoverableArtifactToolRuntime(
        base=_BaseRuntime(), run_root=root, event_path=root / "unused.jsonl",
    )
    runtime.bind_context_artifacts(
        catalog, event_sink=lambda kind, **fields: events.append({
            "type": kind, **fields,
        }),
    )
    observed = runtime.invoke_with_context(
        "observe", {"ref": image["ref"]}, call_context={},
    )
    assert base64.b64decode(observed["images_base64"][0]) == b"exact-image"
    selected = runtime.invoke_with_context(
        "artifact.read", {"ref": document["ref"], "json_pointer": "/outer/value"},
        call_context={},
    )
    assert selected["json_value"] == "exact"
    chunk = runtime.invoke_with_context(
        "artifact.read", {"ref": long_text["ref"], "offset": 2, "limit": 4},
        call_context={},
    )
    assert chunk["content"] == "2345" and chunk["next_offset"] == 6
    unicode_head = runtime.invoke_with_context(
        "artifact.read", {"ref": unicode_text["ref"]}, call_context={},
    )
    assert unicode_head["content"] == "a" * 4095
    unicode_tail = runtime.invoke_with_context(
        "artifact.read", {
            "ref": unicode_text["ref"], "offset": unicode_head["next_offset"],
        }, call_context={},
    )
    assert unicode_tail["content"] == "中" and unicode_tail["eof"] is True
    exact_eof = runtime.invoke_with_context(
        "artifact.read", {
            "ref": long_text["ref"], "offset": len(b"0123456789"), "limit": 4,
        }, call_context={},
    )
    assert exact_eof["content"] == ""
    assert exact_eof["next_offset"] == len(b"0123456789")
    assert exact_eof["eof"] is True
    # Argument errors are invalid_arguments naming the argument, store faults
    # keep infrastructure codes (refusal census 6, 8): never internal_tool_error.
    with pytest.raises(ToolError, match="past the artifact") as past:
        runtime.invoke_with_context(
            "artifact.read", {
                "ref": long_text["ref"], "offset": len(b"0123456789") + 1,
            }, call_context={},
        )
    assert (past.value.code, past.value.field) == ("invalid_arguments", "offset")
    runtime.reset_round(3)
    observed_after_reset = runtime.invoke_with_context(
        "observe", {"ref": image["ref"]}, call_context={},
    )
    assert base64.b64decode(
        observed_after_reset["images_base64"][0]
    ) == b"exact-image"
    assert runtime.base.reset_rounds == [3]
    with pytest.raises(ToolError, match="another PolicySession") as foreign:
        runtime.invoke_with_context(
            "artifact.read", {"ref": document["ref"].replace(
                "policy-session-1", "policy-session-2"
            )}, call_context={},
        )
    assert (foreign.value.code, foreign.value.field) == ("invalid_arguments", "ref")
    with pytest.raises(ToolError, match="image artifact") as image_read:
        runtime.invoke_with_context(
            "artifact.read", {"ref": image["ref"]}, call_context={},
        )
    assert (image_read.value.code, image_read.value.field) == ("invalid_arguments", "ref")
    with pytest.raises(ToolError, match="image artifact"):
        runtime.invoke_with_context(
            "observe", {"ref": long_text["ref"]}, call_context={},
        )
    with pytest.raises(ToolError, match="takes ref alone") as extra:
        runtime.invoke_with_context(
            "observe", {"ref": image["ref"], "views": ["agentview"]},
            call_context={},
        )
    assert (extra.value.code, extra.value.field) == ("invalid_arguments", "views")
    missing = "artifact://policy-session-1/" + "f" * 64
    with pytest.raises(ToolError, match="no artifact") as unknown:
        runtime.invoke_with_context(
            "artifact.read", {"ref": missing}, call_context={},
        )
    assert (unknown.value.code, unknown.value.field) == ("invalid_arguments", "ref")
    _entry, data = catalog.resolve(document["ref"])
    assert data == b'{"outer":{"value":"exact"}}'
    index = json.loads(catalog.index_path.read_text(encoding="utf-8"))
    stored = root / index["artifacts"][document["ref"]]["storage_ref"]["path"]
    stored.write_bytes(b"tampered")
    with pytest.raises(ToolError, match="hash") as drifted:
        runtime.invoke_with_context(
            "artifact.read", {"ref": document["ref"]}, call_context={},
        )
    assert drifted.value.code == "artifact_hash_mismatch"
    assert [event["payload"]["status"] for event in events] == [
        "ok", "ok", "ok", "ok", "ok", "ok", "rejected", "ok",
        "rejected", "rejected", "rejected", "rejected", "rejected", "rejected",
    ]
    assert all(
        isinstance(event["payload"]["duration_ms"], float)
        and event["payload"]["duration_ms"] >= 0
        for event in events
    )
