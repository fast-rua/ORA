"""The sandbox's host prerequisites: where the interpreter is found, and what a refusal says to install or set.

``bwrap``, unprivileged user namespaces and the interpreter (``sandbox.py``). An explicit ``ORA_CODE_MODE_PYTHON``
wins; without one the interpreter ``ora setup --env robodojo`` built under its prefix is used. Preflight's
``code_mode_sandbox`` and ``ora doctor`` refuse with what to do. Fake ``bwrap`` scripts stand for the host; nothing
here needs a real sandbox.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ora.agent.code_mode import sandbox
from ora.agent.code_mode.sandbox import SandboxError


def _executable(path: Path, body: str = "exit 0") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A host with a setup prefix and nothing configured."""
    monkeypatch.setenv("ORA_GPU_PREFIX", str(tmp_path / "prefix"))
    monkeypatch.delenv("ORA_CODE_MODE_PYTHON", raising=False)
    monkeypatch.delenv("ORA_CODE_MODE_BWRAP", raising=False)
    return tmp_path


def test_without_the_variable_the_interpreter_setup_built_is_used(host):
    built = _executable(host / "prefix" / sandbox.DEFAULT_PYTHON)
    assert sandbox.default_python() == built
    assert sandbox.python_path() == str(built)


def test_an_explicit_variable_wins_over_the_one_setup_built(host, monkeypatch):
    _executable(host / "prefix" / sandbox.DEFAULT_PYTHON)
    explicit = _executable(host / "elsewhere/bin/python")
    monkeypatch.setenv("ORA_CODE_MODE_PYTHON", str(explicit))
    assert sandbox.python_path() == str(explicit)
    # A variable that names nothing runnable is refused, not passed over for the built one.
    monkeypatch.setenv("ORA_CODE_MODE_PYTHON", str(host / "missing/python"))
    with pytest.raises(SandboxError, match="ORA_CODE_MODE_PYTHON=.*missing/python is not an executable interpreter"):
        sandbox.python_path()


def test_without_either_the_refusal_names_setup_the_path_and_the_lock_build(host):
    with pytest.raises(SandboxError) as refused:
        sandbox.python_path()
    message = str(refused.value)
    assert f"ORA_CODE_MODE_PYTHON is unset and {host / 'prefix' / sandbox.DEFAULT_PYTHON} does not exist" in message
    assert "run ./scripts/ora setup --env robodojo" in message
    assert f"--require-hashes --no-deps --only-binary :all: --compile-bytecode -r {sandbox.LOCK}" in message


def test_the_default_prefix_is_setups_when_no_prefix_is_named(monkeypatch, tmp_path):
    monkeypatch.delenv("ORA_GPU_PREFIX", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert sandbox.default_python() == tmp_path / "data/ora/gpu" / sandbox.DEFAULT_PYTHON


def test_preflight_says_to_install_bubblewrap(host, monkeypatch):
    from ora.orchestrator.preflight.gates import _code_mode_sandbox

    monkeypatch.setenv("ORA_CODE_MODE_BWRAP", str(host / "no-bwrap"))
    check = _code_mode_sandbox({"policy": {"code_mode": {"contract": "ora.code-mode-script.v1"}}})
    assert check["check"] == "code_mode_sandbox" and check["passed"] is False
    assert "is not an executable bwrap; install bubblewrap" in check["detail"]["reason"]
    assert "apt-get install bubblewrap" in check["detail"]["reason"]


def test_preflight_says_to_enable_user_namespaces_when_bwrap_cannot_create_them(host, monkeypatch):
    from ora.orchestrator.preflight.gates import _code_mode_sandbox

    message = "bwrap: No permissions to creating new namespace, likely because the kernel does not allow it"
    bwrap = _executable(host / "bin/bwrap", f"echo '{message}' >&2; exit 1")
    monkeypatch.setenv("ORA_CODE_MODE_BWRAP", str(bwrap))
    reason = _code_mode_sandbox({"policy": {"code_mode": {}}})["detail"]["reason"]
    assert reason.startswith(f"bubblewrap cannot create the sandbox's namespaces on this host ({message})")
    assert "kernel.unprivileged_userns_clone=1" in reason and "user.max_user_namespaces" in reason
    assert "kernel.apparmor_restrict_unprivileged_userns=0" in reason


def test_preflight_says_where_the_interpreter_is_missing_once_the_sandbox_starts(host, monkeypatch):
    from ora.orchestrator.preflight.gates import _code_mode_sandbox

    monkeypatch.setenv("ORA_CODE_MODE_BWRAP", str(_executable(host / "bin/bwrap")))
    reason = _code_mode_sandbox({"policy": {"code_mode": {}}})["detail"]["reason"]
    assert reason.startswith("no code-mode interpreter: ORA_CODE_MODE_PYTHON is unset")
    assert "./scripts/ora setup --env robodojo" in reason


def test_doctor_checks_each_prerequisite_and_the_interpreter_against_its_lock(host, monkeypatch):
    import json

    monkeypatch.setenv("ORA_CODE_MODE_BWRAP", str(_executable(host / "bin/bwrap")))
    python, packages = sandbox.locked_packages(sandbox.LOCK)
    found = {"executable": "/fake/python", "python": f"{python}.9",
             "packages": {name: {"version": version, "error": None} for name, version in packages.items()}}
    # The interpreter answers the lock probe itself (outside the sandbox); the setup-built one is found.
    built = _executable(host / "prefix" / sandbox.DEFAULT_PYTHON, f"echo '{json.dumps(found)}'")
    checks = {item["check"]: item for item in sandbox.prerequisite_checks()}
    assert list(checks) == ["code_mode_bwrap", "code_mode_user_namespaces", "code_mode_interpreter"]
    assert checks["code_mode_bwrap"] == {"check": "code_mode_bwrap", "passed": True,
                                         "detail": str(host / "bin/bwrap")}
    assert checks["code_mode_user_namespaces"]["passed"] is True
    assert checks["code_mode_interpreter"] == {"check": "code_mode_interpreter", "passed": True,
                                               "detail": f"{built} matches {sandbox.LOCK.name}"}
    found["packages"]["numpy"]["version"] = "1.26.3"
    _executable(built, f"echo '{json.dumps(found)}'")
    interpreter = {item["check"]: item for item in sandbox.prerequisite_checks()}["code_mode_interpreter"]
    assert interpreter["passed"] is False
    assert f"does not match its lock {sandbox.LOCK}: numpy: the lock says 1.26.4" in interpreter["detail"]
    assert "Build one that does" in interpreter["detail"]


def test_an_interpreter_under_tmp_is_bound_over_the_fresh_tmp(monkeypatch, tmp_path):
    from ora.agent.code_mode import sandbox

    root = "/tmp/someone/.local/share/uv/python/cpython-3.11"
    monkeypatch.setattr(sandbox, "_interpreter_roots", lambda python: (root,))
    command = sandbox.bwrap_command("/tmp/someone/venv/bin/python", "bwrap", 3, tmp_path)
    assert command.index("--tmpfs") < command.index(root)


def test_a_managed_interpreter_reached_through_a_link_is_bound_under_both_names(tmp_path):
    import json, os, sys
    from ora.agent.code_mode import sandbox

    real = tmp_path / "cpython-3.11.18"
    (real / "bin").mkdir(parents=True)
    link = tmp_path / "cpython-3.11"
    link.symlink_to(real)
    fake = real / "bin" / "python3.11"  # answers the prefix probe the way a venv over the link does
    fake.write_text(f"#!{sys.executable}\nimport json; print(json.dumps([{str(tmp_path / 'venv')!r}, {str(link)!r}, "
                    f"{str(tmp_path / 'venv')!r}, {str(link)!r}]))\n")
    fake.chmod(0o755)
    sandbox._interpreter_roots.cache_clear()
    roots = sandbox._interpreter_roots(str(fake))
    assert str(link) in roots and str(real) in roots
    sandbox._interpreter_roots.cache_clear()
