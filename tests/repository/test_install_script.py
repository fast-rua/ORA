"""``scripts/install.sh``: the one-line user installer, run against a local repository with stub uv and npm."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "install.sh"


def _node_ok() -> bool:
    node = shutil.which("node")
    if not node:
        return False
    major, minor = (int(part) for part in subprocess.check_output([node, "-p", "process.versions.node"],
                                                                    text=True).split(".")[:2])
    return (major, minor) >= (22, 19)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"})


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "ora-pi").mkdir()
    ora = repo / "scripts" / "ora"
    ora.write_text('#!/bin/sh\necho "ora $*"\n')
    ora.chmod(0o755)
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name in ("uv", "npm"):  # the real ones download; these record that they ran
        stub = stubs / name
        stub.write_text(f'#!/bin/sh\necho "{name} $*" >> "{tmp_path}/calls"\n')
        stub.chmod(0o755)
    node = stubs / "node"  # the real Node, found beside the stub npm, as the installer looks for npm
    node.write_text(f'#!/bin/sh\nexec "{shutil.which("node")}" "$@"\n')
    node.chmod(0o755)
    return {"tmp": tmp_path, "repo": repo, "stubs": stubs}


def _install(world: dict[str, Path], *extra: str) -> subprocess.CompletedProcess[str]:
    tmp = world["tmp"]
    env = {"HOME": str(tmp / "home"), "PATH": f"{world['stubs']}{os.pathsep}{os.environ['PATH']}",
           "ORA_NODE": str(world["stubs"] / "node")}
    return subprocess.run(["sh", str(SCRIPT), "--repo", str(world["repo"]), "--ref", "main",
                           "--dir", str(tmp / "ora-home"), "--bin-dir", str(tmp / "bin"), "--no-doctor", *extra],
                          env=env, capture_output=True, text=True, timeout=120)


def test_install_script_parses() -> None:
    subprocess.run(["sh", "-n", str(SCRIPT)], check=True)
    result = subprocess.run(["sh", str(SCRIPT), "--bogus"], capture_output=True, text=True)
    assert result.returncode != 0 and "unknown option --bogus" in result.stderr


@pytest.mark.skipif(not _node_ok(), reason="needs Node.js 22.19 or newer on PATH")
def test_install_checks_out_syncs_and_writes_the_ora_command(world: dict[str, Path]) -> None:
    result = _install(world)
    assert result.returncode == 0, result.stderr
    checkout = world["tmp"] / "ora-home" / "ORA"
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    expected = subprocess.check_output(["git", "-C", str(world["repo"]), "rev-parse", "HEAD"], text=True).strip()
    assert head == expected
    calls = (world["tmp"] / "calls").read_text()
    assert "uv sync --all-extras --locked" in calls and "npm ci --prefix" in calls
    shim = world["tmp"] / "bin" / "ora"
    assert "Installed by ORA's scripts/install.sh" in shim.read_text()
    assert "export ORA_COMMAND=ora\n" in shim.read_text()         # ORA's hints then name `ora`, not ./scripts/ora
    out = subprocess.check_output([str(shim), "doctor"], text=True, env={**os.environ, "HOME": str(world["tmp"])})
    assert out.strip() == "ora doctor"

    # Rerunning updates the same checkout to the new commit.
    (world["repo"] / "NEWS").write_text("x\n")
    _git("add", "-A", cwd=world["repo"])
    _git("commit", "-qm", "next", cwd=world["repo"])
    assert _install(world).returncode == 0
    newer = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    assert newer != head


@pytest.mark.skipif(not _node_ok(), reason="needs Node.js 22.19 or newer on PATH")
def test_install_refuses_a_foreign_ora_command_and_a_dirty_checkout(world: dict[str, Path]) -> None:
    (world["tmp"] / "bin").mkdir()
    (world["tmp"] / "bin" / "ora").write_text("#!/bin/sh\necho mine\n")
    result = _install(world)
    assert result.returncode != 0 and "was not written by this installer" in result.stderr

    (world["tmp"] / "bin" / "ora").unlink()
    assert _install(world).returncode == 0
    (world["tmp"] / "ora-home" / "ORA" / "local-edit").write_text("x\n")
    result = _install(world)
    assert result.returncode != 0 and "has local changes" in result.stderr
