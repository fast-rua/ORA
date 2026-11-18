"""LIBERO binds Mesa's software EGL device once per process and its resets record it, on stand-ins."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from ora.environment.backends.libero_pro import construction
from ora.environment.backends.libero_pro.session import (
    LiberoProSession,
    libero_environment_init_params,
    libero_public_environment_context,
)
from ora.environment.graphics_probe import SOFTWARE_EGL_DEVICE_SCHEMA
from tests.support.orchestrator.runtime_v1 import FakeLiberoEnvironment, _fake_libero_metadata

DEVICE = {"schema": SOFTWARE_EGL_DEVICE_SCHEMA, "index": 8, "vendor": "Mesa Project",
          "driver": "llvmpipe (LLVM 15.0.7, 256 bits)"}


def test_the_device_is_bound_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("robosuite", "robosuite.utils", "robosuite.utils.binding_utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    calls: list[None] = []

    def bind() -> dict[str, Any]:
        calls.append(None)
        return dict(DEVICE)

    monkeypatch.setattr(construction, "bind_software_egl_device", bind)
    monkeypatch.setattr(construction, "_EGL_DEVICE", None)

    assert construction.bind_egl_device() == DEVICE
    assert construction.bind_egl_device() == DEVICE
    assert len(calls) == 1  # robosuite's display belongs to the first context; it is never rebound


def test_each_reset_records_the_egl_device(tmp_path: Path) -> None:
    public_context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100),
    )
    session = LiberoProSession(
        libero_pro_root=tmp_path / "LIBERO-PRO", data_root=tmp_path / "data", output_root=tmp_path / "native",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate",
        required_libero_pro_revision="libero-test", required_data_revision="data-test",
        workspace_bounds_m={"x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3]}, max_sim_steps=100,
        environment_factory=lambda **kwargs: (
            FakeLiberoEnvironment(), {**_fake_libero_metadata(tmp_path, public_context), "egl_device": DEVICE}),
    )

    reset = session.start()

    assert reset["egl_device"] == DEVICE
    assert json.loads((tmp_path / "native" / "reset.json").read_text())["egl_device"] == DEVICE
    assert "egl_device" not in reset["public_environment_context"]
    assert session.restart_round(1)["egl_device"] == DEVICE
