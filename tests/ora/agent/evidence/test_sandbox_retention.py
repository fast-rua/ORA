"""save_sandbox: false keeps only the hashes of a sealed run's sandbox/ (agent/evidence/sandbox_retention.py)."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path

import pytest

from ora.agent.evidence import sandbox_retention
from ora.orchestrator.evaluation import evaluate as evaluation


def _sealed_run(tmp_path: Path, *, save_sandbox: bool | None) -> Path:
    run = tmp_path / "runs" / "run-a"
    files = {
        "sandbox/workspace/AGENTS.md": b"instructions",
        "sandbox/workspace/inputs/action-observations/a/head.png": b"\x89PNG" + b"1" * 100,
        "sandbox/workspace/inputs/current-visual/a/head.png": b"\x89PNG" + b"2" * 100,
        "policy-observations/step-0001/head.png": b"\x89PNG" + b"0" * 100,
        "result.json": b"{}",
    }
    for relative, data in files.items():
        (run / relative).parent.mkdir(parents=True, exist_ok=True)
        (run / relative).write_bytes(data)
    (run / "sandbox/workspace/scratch").mkdir()
    os.symlink("../inputs", run / "sandbox/workspace/scratch/inputs")
    plan = {} if save_sandbox is None else {"save_sandbox": save_sandbox}
    (run / "plan.json").write_text(json.dumps(plan))
    inventory = []
    for relative in sorted([*files, "plan.json"]):
        data = (run / relative).read_bytes()
        inventory.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        (run / relative).chmod(0o444)
    (run / "seal.json").write_text(json.dumps({"inventory": inventory, "inventory_sha256": "f" * 64}))
    return run


def test_off_replaces_the_sandbox_by_its_hashes(tmp_path: Path) -> None:
    run = _sealed_run(tmp_path, save_sandbox=False)
    seal = json.loads((run / "seal.json").read_text())
    result = sandbox_retention.omit(run)
    assert result["status"] == "omitted" and result["files"] == 3
    assert sorted(p.relative_to(run).as_posix() for p in (run / "sandbox").rglob("*")) == ["sandbox/hashes.json"]
    manifest = sandbox_retention.manifest(run)
    sealed = [r for r in seal["inventory"] if r["path"].startswith("sandbox/")]
    assert manifest["files"] == sealed and manifest["option"] == {"save_sandbox": False}
    assert all(sandbox_retention.omitted_record(run, record) for record in sealed)
    assert not sandbox_retention.omitted_record(run, next(r for r in seal["inventory"] if r["path"] == "result.json"))
    assert (run / "policy-observations/step-0001/head.png").is_file()  # only sandbox/ is covered
    assert sandbox_retention.omit(run)["status"] == "already_omitted"


@pytest.mark.parametrize("save_sandbox", [None, True])
def test_on_and_the_default_keep_every_file(tmp_path: Path, save_sandbox: bool | None) -> None:
    run = _sealed_run(tmp_path, save_sandbox=save_sandbox)
    assert sandbox_retention.omit(run) == {"status": "kept"}
    assert (run / "sandbox/workspace/inputs/current-visual/a/head.png").is_file()
    assert sandbox_retention.manifest(run) is None


def test_a_drifted_file_is_refused_and_the_files_stay(tmp_path: Path) -> None:
    run = _sealed_run(tmp_path, save_sandbox=False)
    path = run / "sandbox/workspace/AGENTS.md"
    path.chmod(0o644)
    path.write_bytes(b"changed")
    assert sandbox_retention.omit_after_seal(run)["status"] == "refused"
    assert path.is_file() and sandbox_retention.manifest(run) is None
    assert "drifted" in json.loads((run / sandbox_retention.REFUSAL).read_text())["reason"]


def test_export_carries_the_hashes_and_evaluation_says_the_files_were_not_saved(tmp_path: Path) -> None:
    run = _sealed_run(tmp_path, save_sandbox=False)
    sandbox_retention.omit(run)
    exported = evaluation.export(run, tmp_path / "export.tar.gz")
    with tarfile.open(exported) as archive:
        names = set(archive.getnames())
    assert "run-a/sandbox/hashes.json" in names and "run-a/result.json" in names
    assert not any(name.startswith("run-a/sandbox/workspace") for name in names)
    with pytest.raises(evaluation.EvaluationError, match="not saved .save_sandbox: false."):
        evaluation.evaluate(run, persist=False)
