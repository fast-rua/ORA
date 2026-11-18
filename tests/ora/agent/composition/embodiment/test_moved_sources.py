"""A recorded path answered elsewhere now: a moved file only with its bytes, a successor descriptor only with its geometry."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from ora.agent.composition.embodiment import EmbodimentDescriptorError, audit_embodiment_sources
from ora.agent.composition.embodiment.geometry import load_public_geometry_binding
from ora.agent.composition.embodiment.moved_sources import MOVED, resolve, succeeds

ROOT = Path(__file__).resolve().parents[5]
V5 = "ora/policy/embodiments/arx-x5/descriptor-public-geometry-v5.json"
V6 = "ora/policy/embodiments/arx-x5/descriptor-public-geometry-v6.json"
GRIPPER = "ora/agent/tools/grasp/arx_x5_gripper.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_entry_is_a_moved_file_or_a_successor_with_the_same_geometry() -> None:
    for (old, sha256), new in MOVED.items():
        if not (ROOT / old).exists():  # moved: the same bytes at the new place
            assert _sha(ROOT / new) == sha256, new
            continue
        assert _sha(ROOT / old) == sha256, old  # kept byte for byte for the runs sealed with it
        then = load_public_geometry_binding({"path": old, "sha256": sha256}, project_root=ROOT)
        now = load_public_geometry_binding({"path": new, "sha256": _sha(ROOT / new)}, project_root=ROOT)
        assert then["geometry"] == now["geometry"], new


def test_an_observation_verified_against_v5_holds_for_v6_and_nothing_else() -> None:
    v5, v6 = _sha(ROOT / V5), _sha(ROOT / V6)
    assert succeeds(ROOT, v5, {"path": V6, "sha256": v6})
    assert not succeeds(ROOT, "0" * 64, {"path": V6, "sha256": v6}), "another recorded descriptor"
    assert not succeeds(ROOT, v5, {"path": V6, "sha256": "0" * 64}), "a v6 binding to other bytes"
    assert not succeeds(ROOT, v6, {"path": V5, "sha256": v5}), "the table is one way"


def test_a_successor_with_other_geometry_is_refused(tmp_path: Path) -> None:
    for relative in (V5, V6):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, tmp_path / relative)
    descriptor = json.loads((tmp_path / V6).read_text())
    descriptor["geometry"]["grippers"][0]["contact_geometry"]["fingers"][0]["collision_bounds_m"]["minimum"][0] += 0.01
    (tmp_path / V6).write_text(json.dumps(descriptor))
    binding = {"path": V6, "sha256": _sha(tmp_path / V6)}
    assert load_public_geometry_binding(binding, project_root=tmp_path)["geometry"], "a valid descriptor"
    assert not succeeds(tmp_path, _sha(ROOT / V5), binding)


def test_the_v5_descriptor_still_audits_through_its_moved_gripper_source() -> None:
    descriptor = json.loads((ROOT / V5).read_text())
    assert GRIPPER in {source.get("local_path") for source in descriptor["provenance"]["sources"]}
    audit = audit_embodiment_sources(descriptor, project_root=ROOT)
    assert audit["passed"] is True
    assert {"source_id": "arx_x5_reviewed_nominal_gripper_parameters", "kind": "policy_runtime_adapter",
            "status": "repository_source", "local_path": GRIPPER} in audit["checks"]


def test_a_moved_file_whose_bytes_changed_is_refused_as_a_missing_one(tmp_path: Path) -> None:
    descriptor = json.loads((ROOT / V5).read_text())
    for source in descriptor["provenance"]["sources"]:
        if "local_path" in source:
            found = resolve(ROOT, source["local_path"]).relative_to(ROOT.resolve())
            (tmp_path / found).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / found, tmp_path / found)
    assert audit_embodiment_sources(descriptor, project_root=tmp_path)["passed"] is True
    (old, sha256), new = next(iter(MOVED.items()))
    assert resolve(tmp_path, old) == resolve(tmp_path, old, sha256) == (tmp_path / new).resolve()
    assert resolve(tmp_path, old, "0" * 64) is None, "a binding that recorded other bytes"
    (tmp_path / new).write_bytes((ROOT / new).read_bytes() + b"\n")
    assert resolve(tmp_path, old) is None
    with pytest.raises(EmbodimentDescriptorError, match=f"provenance source is unavailable: {GRIPPER}"):
        audit_embodiment_sources(descriptor, project_root=tmp_path)


def test_a_path_that_is_there_is_its_own_file_and_an_unknown_missing_path_is_none(tmp_path: Path) -> None:
    (tmp_path / "here.txt").write_text("x")
    assert resolve(tmp_path, "here.txt") == (tmp_path / "here.txt").resolve()
    assert resolve(tmp_path, "gone.txt") is None
    assert resolve(tmp_path, "../outside.txt") is None
