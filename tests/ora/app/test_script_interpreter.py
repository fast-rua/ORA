"""scripts/lib/python.sh: the interpreter every shell wrapper but scripts/ora runs, and when it may run uv."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SYSTEM_PATH = "/usr/bin:/bin"
#: The wrappers that need only the standard library: python3 from PATH, never uv.
STDLIB_WRAPPERS = {"scripts/ora-remote", "scripts/setup-robodojo"}


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_python(path: Path, label: str) -> Path:
    return _executable(path, f'#!/bin/sh\necho {label} "$@"\n')


@pytest.fixture
def checkout(tmp_path: Path) -> dict[str, Path]:
    """A copy of scripts/ in an empty checkout, python3 on PATH, and a fake uv beside it."""

    root = tmp_path / "checkout"
    shutil.copytree(ROOT / "scripts", root / "scripts")
    bin_dir = tmp_path / "bin"
    _fake_python(bin_dir / "python3", "path-python3")
    uv_dir = tmp_path / "uv-bin"
    uv_log = tmp_path / "uv.log"
    uv_python = _fake_python(tmp_path / "uv-env/bin/python", "uv-python")
    # `uv run ... python -c 'print(sys.executable)'` prints the environment's interpreter.
    _executable(uv_dir / "uv", f'#!/bin/sh\necho "$@" >> {uv_log}\n'
                               f'[ -z "$FAKE_UV_FAIL" ] || {{ echo "uv: lock is stale" >&2; exit 2; }}\n'
                               f'echo {uv_python}\n')
    return {"root": root, "bin": bin_dir, "uv": uv_dir, "uv_log": uv_log, "uv_python": uv_python,
            "home": tmp_path}


def _environment(c: dict[str, Path], *, uv: bool = True, **extra: str) -> dict[str, str]:
    path = [str(c["bin"])] + ([str(c["uv"])] if uv else []) + [SYSTEM_PATH]
    return {"PATH": ":".join(path), "HOME": str(c["home"]), **extra}


def _select(c: dict[str, Path], *args: str, uv: bool = True, **extra: str) -> subprocess.CompletedProcess[str]:
    driver = '. "$ROOT/scripts/lib/python.sh"; ora_python "$@"; printf "%s\\n%s\\n" "$PYTHON" "$PYTHONPATH"'
    return subprocess.run(["sh", "-c", driver, "sh", *args], text=True, capture_output=True,
                          env={**_environment(c, uv=uv, **extra), "ROOT": str(c["root"])})


def _uv_calls(c: dict[str, Path]) -> list[str]:
    return c["uv_log"].read_text().splitlines() if c["uv_log"].exists() else []


def test_the_checkout_venv_then_python_then_the_shared_environment_and_none_runs_uv(checkout) -> None:
    c = checkout
    shared = _fake_python(c["home"] / "shared-venv/bin/python", "shared")
    chosen = _select(c, "probe", UV_PROJECT_ENVIRONMENT=str(shared.parents[1]))
    assert chosen.stdout.splitlines() == [str(shared), str(c["root"])] and not chosen.stderr
    chosen = _select(c, "probe", UV_PROJECT_ENVIRONMENT=str(shared.parents[1]), PYTHON="/opt/explicit/python")
    assert chosen.stdout.splitlines()[0] == "/opt/explicit/python"
    venv = _fake_python(c["root"] / ".venv/bin/python", "checkout-venv")
    chosen = _select(c, "probe", PYTHON="/opt/explicit/python")
    assert chosen.stdout.splitlines()[0] == str(venv)
    assert _uv_calls(c) == []


def test_without_an_environment_uv_run_locked_prepares_the_checkout(checkout) -> None:
    c = checkout
    chosen = _select(c, "probe")
    assert chosen.returncode == 0 and not chosen.stderr
    assert chosen.stdout.splitlines()[0] == str(c["uv_python"])
    assert _uv_calls(c) == [f"run --locked --project {c['root']} python -c import sys; print(sys.executable)"]

    failed = _select(c, "probe", FAKE_UV_FAIL="1")
    assert failed.returncode == 1 and failed.stdout == ""
    assert "uv: lock is stale" in failed.stderr and "probe: uv run --locked could not prepare" in failed.stderr


@pytest.mark.parametrize("setting, reason", [
    ({"UV_PROJECT_ENVIRONMENT": "/nowhere/shared-venv"},
     "$UV_PROJECT_ENVIRONMENT=/nowhere/shared-venv has no bin/python"),
    ({"UV_NO_SYNC": "1"}, "$UV_NO_SYNC keeps uv from creating one"),
])
def test_a_named_shared_environment_or_no_sync_never_leads_to_uv(checkout, setting, reason) -> None:
    c = checkout
    chosen = _select(c, "probe", **setting)
    assert chosen.stdout.splitlines()[0] == "python3"
    assert chosen.stderr.startswith("probe: no locked project environment") and reason in chosen.stderr
    assert _uv_calls(c) == []


def test_python3_from_path_comes_last_and_a_stdlib_entry_takes_it_silently(checkout) -> None:
    c = checkout
    if shutil.which("uv", path=SYSTEM_PATH):
        pytest.skip(f"uv is installed under {SYSTEM_PATH}")
    chosen = _select(c, "probe", uv=False)
    assert chosen.stdout.splitlines()[0] == "python3"
    assert "no locked project environment" in chosen.stderr and os.fspath(c["bin"] / "python3") in chosen.stderr
    chosen = _select(c, "--stdlib", "probe")
    assert chosen.stdout.splitlines()[0] == "python3" and not chosen.stderr
    assert _uv_calls(c) == []


def _wrappers() -> list[str]:
    return sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "scripts").rglob("*")
                  if path.is_file() and path.parts[-2] != "lib" and path.name not in {"check", "README.md"}
                  and "ora_python " in path.read_text(encoding="utf-8", errors="replace"))


def test_the_wrappers_that_use_the_library_are_the_standard_library_entries() -> None:
    # The other wrappers became ora subcommands and ora-viewer compile; scripts/dev/check selects its
    # interpreter with the library too, and scripts/ora keeps its own copy of the order.
    assert set(_wrappers()) == STDLIB_WRAPPERS


@pytest.mark.parametrize("wrapper", _wrappers())
def test_each_wrapper_runs_the_interpreter_the_library_selects(checkout, wrapper) -> None:
    c = checkout
    result = subprocess.run([str(c["root"] / wrapper)], text=True, capture_output=True, env=_environment(c),
                            cwd=c["home"])
    assert result.returncode == 0, result.stderr
    label, *arguments = result.stdout.split()
    if wrapper in STDLIB_WRAPPERS:
        assert (label, _uv_calls(c), result.stderr) == ("path-python3", [], "")
    else:
        assert label == "uv-python" and len(_uv_calls(c)) == 1 and not result.stderr
    assert arguments, wrapper
