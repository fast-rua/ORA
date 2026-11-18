"""Test helpers for robodojo operator, shared by several test modules.

They were defined in ``tests/ora/orchestrator/backends/robodojo/test_robodojo_operator.py``;
test modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ora.environment.benchmarks import robodojo


def _stub_probe_environment(
    module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
    device: str = "2",
) -> list[dict]:
    """Reach the probe body without an interpreter, a GPU or a worker."""

    from ora.orchestrator.backends.robodojo import embodiment as embodiment

    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path / "probes")
    monkeypatch.setattr(module, "_source_commit", lambda: "a" * 40)
    # The record is admitted by worker-kernel content; the fixture commit has
    # no git history behind it, so the digest is supplied here.
    monkeypatch.setattr(
        module, "worker_kernel_sha256", lambda commit, **_kwargs: "d" * 64,
    )
    monkeypatch.setattr(module, "python_stack", lambda: {"stub": True})
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    monkeypatch.setattr(
        module.subprocess, "check_output",
        lambda *_args, **_kwargs: f"{device}, 40000, 580.65.06\n",
    )
    monkeypatch.setattr(
        module, "claim_resource_lease", lambda *_args: {"claim_token": "token"},
    )
    monkeypatch.setattr(module, "assign_resource_lease", lambda *_args: None)
    monkeypatch.setattr(
        module, "release_resource_lease", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        embodiment, "create_activation", lambda root, episodes: {"stub": True},
    )
    seen: list[dict] = []

    def episode(runtime, root, *, scope="card"):
        seen.append(dict(runtime))
        names = (robodojo.CARD_QUALIFICATION_CHECKS if scope == "card"
                 else robodojo.TASK_QUALIFICATION_CHECKS)
        registration = {
            "schema": robodojo.SCENE_REGISTRATION_SCHEMA,
            "instances": {"mug_0": "rigid"},
            "instance_identities_sha256": robodojo.scene_identities_sha256(
                {"mug_0": "rigid"},
            ),
            "registered_instance_count": 1,
            "evaluator_registration": {
                "check_list": True, "final_check_list": False,
                "trigger_check_list": False,
            },
            "native_action_limit": int(runtime["native_action_limit"]),
        }
        return {
            "passed": True, "native_timeout_exercised": True, "scope": scope,
            "checks": {
                **{name: True for name in names},
                **{name: True for name in robodojo.QUALIFICATION_CHECKS},
            },
            "scene_registration": registration,
            "repeated_reset_scene_registration": registration,
            "reset": {
                "task_instruction": "Hang all the mugs on the mug rack.",
                "actual_source_fingerprint": {"loaded_files": {}},
                "source_binding": {
                    "source_revisions": robodojo.source_revisions(runtime),
                },
                "control_binding": {"control_hz": 25},
                "runtime_embodiment": {
                    "collector": (
                        "initialized_isaac_articulation_and_optical_camera_prim"
                    ),
                    "arms": {"left": {}, "right": {}},
                    "cameras": {
                        "head": {}, "left_wrist": {}, "right_wrist": {},
                    },
                },
                "device_binding": {"selected_gpu_uuid": "GPU-fixture"},
            },
        }

    monkeypatch.setattr(module, "_episode", episode)
    return seen
