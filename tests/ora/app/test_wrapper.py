"""scripts/ora runs the Harness in the locked project environment before any python3 on PATH."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _fake_python(path: Path, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho {label} \"$@\"\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _run(tmp_path: Path, **environment: str) -> subprocess.CompletedProcess[str]:
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts/ora", checkout / "scripts/ora")
    path = tmp_path / "path"
    _fake_python(path / "python3", "path-python3")
    values = {"PATH": f"{path}:/usr/bin:/bin", "HOME": str(tmp_path), **environment}
    return subprocess.run([str(checkout / "scripts/ora"), "status"], env=values, text=True, capture_output=True,
                          check=True)


def test_the_checkout_venv_then_python_then_uv_project_environment(tmp_path: Path) -> None:
    shared = _fake_python(tmp_path / "shared-venv/bin/python", "shared-venv")
    explicit = _fake_python(tmp_path / "explicit/python", "explicit")
    chosen = _run(tmp_path, UV_PROJECT_ENVIRONMENT=str(shared.parents[1]))
    assert chosen.stdout.split()[:3] == ["shared-venv", "-m", "ora.app"] and not chosen.stderr
    chosen = _run(tmp_path, UV_PROJECT_ENVIRONMENT=str(shared.parents[1]), PYTHON=str(explicit))
    assert chosen.stdout.split()[0] == "explicit"
    _fake_python(tmp_path / "checkout/.venv/bin/python", "checkout-venv")
    chosen = _run(tmp_path, UV_PROJECT_ENVIRONMENT=str(shared.parents[1]), PYTHON=str(explicit))
    assert chosen.stdout.split()[0] == "checkout-venv"


def test_python3_from_path_only_last_and_said_on_stderr(tmp_path: Path) -> None:
    chosen = _run(tmp_path, UV_PROJECT_ENVIRONMENT=str(tmp_path / "missing"))
    assert chosen.stdout.split()[0] == "path-python3"
    assert "no locked project environment" in chosen.stderr and os.fspath(tmp_path / "path/python3") in chosen.stderr
