"""Test helpers for control plane, shared by several test modules.

They were defined in ``tests/ora/orchestrator/test_control_plane.py``; test modules
import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ora.orchestrator.preflight.gates import SERVICE_SET_BINDING_SCHEMA


ROOT = Path(__file__).resolve().parents[3]


def _graphics_stub(tmp_path: Path, body: str) -> Path:
    """Write an interpreter stand-in that emits one probe line."""

    stub = tmp_path / "python-stub"
    stub.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    stub.chmod(0o755)
    return stub


def _service_set_plan(*, drifted: str | None = None) -> dict:
    """A v2 set binding holding both resident services on one tool surface."""

    def child(schema: str, endpoint: str) -> dict:
        return {
            "schema": schema,
            "tool_surface": "unified_manip_v1",
            "endpoint": endpoint,
            "instance_id": "drifted" if schema == drifted else "bound",
        }

    return {
        "policy": {"tool_surface": {"profile": "unified_manip_v1"}},
        "prerequisite": {
            "schema": SERVICE_SET_BINDING_SCHEMA,
            "capability_scope": [
                "resident_sam3_text_grounding_v1", "resident_contact_graspnet_v1",
            ],
            "services": {
                "segmentation": child(
                    "ora.segmentation-service-binding.v1",
                    "http://127.0.0.1:8114",
                ),
                "grasp_proposals": child(
                    "ora.grasp-proposal-service-binding.v1",
                    "http://127.0.0.1:8115",
                ),
            },
        },
    }


def _install_fake_services(monkeypatch: pytest.MonkeyPatch, seen: list) -> None:
    from ora.agent.model_services import clients as service_clients

    def make(label: str):
        class Fake:
            def __init__(self, **kwargs):
                self.binding = kwargs["expected_binding"]
                seen.append(label)

            def require_ready(self):
                if self.binding.get("instance_id") == "drifted":
                    raise ValueError(f"{label} service differs from bound instance_id")
                return {
                    "instance_id": "bound", "pid": 4242,
                    "adapter_sha256": "a" * 64, "model": label,
                    "warmup_complete": True,
                }

            def close(self):
                pass

        return Fake

    monkeypatch.setattr(service_clients, "Sam3ApiSegmenter", make("SAM3"))
    monkeypatch.setattr(service_clients, "ContactGraspNetServiceClient", make("CGN"))
