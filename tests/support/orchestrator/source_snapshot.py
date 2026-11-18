"""Show a CLI child the working tree as a clean commit, so preflight's ``source_identity`` check admits it.

Preflight admits only a clean checkout whose HEAD is the plan's commit, so a
CLI run from a contributor's checkout with local changes is refused before it
starts. ``commit_working_tree`` commits the working tree as it is now (tracked
and untracked files; ignored files excluded) into a scratch repository and
returns the Git environment (``GIT_DIR``, ``GIT_WORK_TREE``) under which every
``git`` the CLI runs sees that commit as HEAD and the tree as clean. The CLI
still runs the checkout's own files in place, with the contributor's changes;
the checkout's repository is only read. The scratch repository borrows the
checkout's objects as alternates, so only changed files are stored again.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
#: Git variables that would point a child's ``git`` somewhere else (a hook's index, another object store).
CLEARED_VARIABLES = ("GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
                     "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE")
_SCRATCH_CONFIG = ("-c", "user.name=ORA tests", "-c", "user.email=tests@localhost", "-c", "commit.gpgsign=false",
                   "-c", "core.hooksPath=/dev/null", "-c", "init.defaultBranch=main")


def _environment(**selected: str) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items()
                   if key not in {*CLEARED_VARIABLES, "GIT_DIR", "GIT_WORK_TREE"}}
    environment.update(selected)
    return environment


def _git(*arguments: str, environment: dict[str, str], cwd: Path) -> str:
    return subprocess.run(["git", *_SCRATCH_CONFIG, *arguments], cwd=cwd, env=environment, check=True,
                          capture_output=True, text=True).stdout.strip()


def commit_working_tree(scratch: Path, work_tree: Path = ROOT) -> dict[str, str]:
    """Commit ``work_tree`` as it is now into a repository under ``scratch``; return the variables that select it."""

    own = _environment()
    common = (work_tree / _git("rev-parse", "--git-common-dir", environment=own, cwd=work_tree)).resolve()
    _git("init", "-q", str(scratch), environment=own, cwd=work_tree)
    git_dir = (scratch / ".git").resolve()
    (git_dir / "objects/info/alternates").write_text(f"{common / 'objects'}\n", encoding="utf-8")
    selected = {"GIT_DIR": str(git_dir), "GIT_WORK_TREE": str(work_tree.resolve())}
    environment = _environment(**selected)
    _git("add", "-A", environment=environment, cwd=work_tree)
    _git("commit", "-q", "--no-verify", "--allow-empty", "-m", "working tree under test",
         environment=environment, cwd=work_tree)
    return selected
