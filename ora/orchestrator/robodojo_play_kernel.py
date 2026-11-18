"""The content identity of the RoboDojo free-play worker: the play kernel.

A free-play RoboDojo worker (backend ``robodojo_play``) runs the benchmark
worker kernel (``robodojo_kernel.WORKER_KERNEL_PATHS``) plus the play code:
the play modules, the scene registry's code, the layout generator and the
play factory. Its identity is the play kernel: those two manifests in order,
hashed by content at one commit exactly as ``worker_kernel_sha256`` hashes the
benchmark kernel (each path and its bytes, length-prefixed). The play kernel
is code only. A scene's data -- its index entry and its YAML -- is bound per
scene instead (``robodojo_play_scenes.source_binding``), so one scene's edit
does not touch another scene's identity.

A play-scene audit (``robodojo_play_audit``) records the play kernel and the
scene's source binding it measured; a host offers a scene, and launch
preflight admits a plan, only when both still hash to the audited values at
the plan's commit. So an edit to a play module, the registry's code, the
layout generator or any benchmark kernel file retires every scene's audit; an
edit to one scene's entry or YAML retires that scene's; any other change to
ORA leaves an audit valid.

``PLAY_KERNEL_PATHS`` lists what the play worker adds to the benchmark kernel:
``robodojo_kernel_review.play_kernel_gaps`` holds it to the play worker's
static import closure (``tests/ora/orchestrator/backends/robodojo/test_robodojo_kernel_paths.py``),
as ``kernel_manifest_gaps`` holds the benchmark manifest to the benchmark
worker's. No benchmark kernel member imports a play module: ``worker.py``
reaches ``play_scenes/session.py`` only on its ``robodojo_play`` branch, a
reviewed pruned dispatch edge.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path
from typing import Sequence

from ora.environment.benchmarks.robodojo import RoboDojoBenchmarkError

from .robodojo_kernel import WORKER_KERNEL_PATHS, worker_kernel_paths

PLAY_KERNEL_SCHEMA = "ora.robodojo-play-kernel.v2"
#: The play worker's own code: its host-side config composer, the registry's code, the layout
#: generator and the play modules (``audit.py`` judges what an audit admits). No scene data: a
#: scene's index entry and YAML are in its own source binding. Sorted, Python only.
PLAY_KERNEL_PATHS = (
    "ora/environment/backends/robodojo/play_scenes/__init__.py",
    "ora/environment/backends/robodojo/play_scenes/audit.py",
    "ora/environment/backends/robodojo/play_scenes/bootstrap.py",
    "ora/environment/backends/robodojo/play_scenes/lifecycle.py",
    "ora/environment/backends/robodojo/play_scenes/operator_view.py",
    "ora/environment/backends/robodojo/play_scenes/session.py",
    "ora/environment/benchmarks/robodojo_play_layouts.py",
    "ora/environment/benchmarks/robodojo_play_scenes.py",
    "ora/environment/robodojo_play_factory.py",
)
#: Where the play worker's executed ORA code starts (the play twin of
#: ``robodojo_kernel_review.ROBODOJO_WORKER_ENTRY_POINTS``).
PLAY_WORKER_ENTRY_POINTS = (
    "ora/environment/worker.py",
    "ora/environment/robodojo_play_factory.py",
    "ora/environment/backends/robodojo/play_scenes/bootstrap.py",
    "ora/environment/backends/robodojo/play_scenes/session.py",
    "ora/environment/backends/robodojo/play_scenes/audit.py",
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OWNER = "ora/orchestrator/robodojo_play_kernel.py"


def _require_commit(commit: str) -> None:
    import re

    if re.fullmatch(r"[0-9a-f]{7,40}", str(commit)) is None:
        raise RoboDojoBenchmarkError("a RoboDojo play kernel hash needs an exact commit id")


def play_kernel_paths(commit: str | None = None, *, project_root: Path | None = None) -> tuple[str, ...]:
    """The play manifest as it stood at ``commit`` (None: this tree), read without executing it."""
    root = _PROJECT_ROOT if project_root is None else project_root
    if commit is None:
        return PLAY_KERNEL_PATHS
    _require_commit(commit)
    result = subprocess.run(["git", "show", f"{commit}:{_OWNER}"], cwd=root, capture_output=True, check=False)
    if result.returncode:
        raise RoboDojoBenchmarkError(f"no RoboDojo play kernel manifest at {commit}")
    declarations = [node.value for node in ast.parse(result.stdout).body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "PLAY_KERNEL_PATHS"
                            for target in node.targets)]
    try:
        paths = ast.literal_eval(declarations[0]) if len(declarations) == 1 else None
    except (ValueError, TypeError):
        paths = None
    if (not isinstance(paths, tuple) or not paths or len(set(paths)) != len(paths)
            or any(not isinstance(path, str) or not path.startswith("ora/") or ".." in Path(path).parts
                   or not path.endswith(".py") for path in paths)):
        raise RoboDojoBenchmarkError("invalid source-bound RoboDojo play kernel manifest")
    return paths


def _blobs(commit: str | None, paths: Sequence[str], root: Path) -> list[bytes]:
    """Each path's bytes at ``commit`` (one ``git cat-file --batch``), or in this tree when None."""
    if commit is None:
        try:
            return [(root / path).read_bytes() for path in paths]
        except OSError as exc:
            raise RoboDojoBenchmarkError(f"cannot read a RoboDojo play kernel path: {exc}") from exc
    request = "".join(f"{commit}:{path}\n" for path in paths).encode()
    process = subprocess.run(["git", "cat-file", "--batch"], cwd=root, input=request, capture_output=True,
                             check=False)
    if process.returncode:
        raise RoboDojoBenchmarkError(f"cannot read the RoboDojo play kernel at {commit}")
    output, offset, blobs = process.stdout, 0, []
    for path in paths:
        end = output.index(b"\n", offset)
        header = output[offset:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise RoboDojoBenchmarkError(f"cannot resolve the RoboDojo play kernel path {path} at {commit}")
        size = int(header[2])
        blobs.append(output[end + 1:end + 1 + size])
        offset = end + 1 + size + 1
    return blobs


def kernel_digest(paths: Sequence[str], blobs: Sequence[bytes]) -> str:
    """``worker_kernel_sha256``'s scheme: each path and its bytes, length-prefixed, in order."""
    digest = hashlib.sha256()
    for relative, data in zip(paths, blobs, strict=True):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def play_kernel(commit: str | None = None, *, project_root: Path | None = None) -> dict[str, object]:
    """``{schema, commit, benchmark_paths, play_paths, sha256}`` of the play kernel at ``commit``.

    ``commit`` None hashes this working tree (what ``ora play`` lists scenes
    against); launch preflight hashes the plan's commit.
    """
    root = _PROJECT_ROOT if project_root is None else project_root
    benchmark = WORKER_KERNEL_PATHS if commit is None else worker_kernel_paths(commit, project_root=root)
    play = play_kernel_paths(commit, project_root=root)
    paths = (*benchmark, *play)
    return {"schema": PLAY_KERNEL_SCHEMA, "commit": commit, "benchmark_paths": len(benchmark),
            "play_paths": list(play), "sha256": kernel_digest(paths, _blobs(commit, paths, root))}


def play_kernel_sha256(commit: str | None = None, *, project_root: Path | None = None) -> str:
    return str(play_kernel(commit, project_root=project_root)["sha256"])


__all__ = ["PLAY_KERNEL_PATHS", "PLAY_KERNEL_SCHEMA", "PLAY_WORKER_ENTRY_POINTS", "kernel_digest", "play_kernel",
           "play_kernel_paths", "play_kernel_sha256"]
