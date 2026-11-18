"""Real bootstrap source-manifest code, fake source trees, no Kit imports."""
import hashlib
from types import SimpleNamespace

import pytest

from ora.environment.backends.robodojo import bridge
from ora.environment.benchmarks import robodojo


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root, assets = tmp_path / "sdk", tmp_path / "assets"
    for base, names in ((assets, ("Robots/x5/robot_config.yml", "Robots/x5/curobo_tmp.yml", "Robots/x5/curobo.yml")),
                        (root, ("env_cfg/arx_x5.yml", "env_cfg/sim/sim_config.yml", "env_cfg/scene/default.yml",
                                "env_cfg/robot/dual_x5.yml", "env_cfg/camera/camera_config.yml",
                                "task/RoboDojo/config/align_blocks.yml", "task/RoboDojo/config/_task.yml",
                                "third_party/IsaacLab/apps/isaaclab.python.kit"))):
        for name in names:
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
    layout = assets / "layout.json"
    layout.write_text("{}")
    monkeypatch.setattr(robodojo, "selected_layout", lambda *a, **k: layout)
    monkeypatch.setattr(bridge, "revision", lambda path: "a" * 40)
    monkeypatch.setattr(bridge.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=b""))
    monkeypatch.setattr(bridge.importlib.metadata, "version", lambda name: "5.1.0")
    return dict(robodojo_root=str(root), asset_root=str(assets), task="align_blocks",
                action_codec="motion_action_v3", required_revision="a" * 40,
                required_isaaclab_revision="a" * 40, required_curobo_revision="a" * 40,
                required_xpolicylab_revision="a" * 40, required_asset_revision="a" * 40,
                layout_sha256=hashlib.sha256(layout.read_bytes()).hexdigest())


@pytest.mark.parametrize("drift", ["revision", "missing", "unversioned_checkout", "dirty", "layout", "package"])
def test_source_identity_only_warns_in_explicit_diagnostic_config(source_tree, monkeypatch, drift):
    config = source_tree
    if drift == "revision":
        config["required_revision"] = "b" * 40
    elif drift == "missing":
        config.pop("required_revision")
    elif drift == "unversioned_checkout":
        def absent(path):
            raise bridge.RobodojoError("source is not a Git checkout")
        monkeypatch.setattr(bridge, "revision", absent)
    elif drift == "dirty":
        monkeypatch.setattr(bridge.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=b"patch"))
    elif drift == "layout":
        config["layout_sha256"] = "c" * 64
    else:
        monkeypatch.setattr(bridge.importlib.metadata, "version", lambda name: "99.0")
    with pytest.raises(bridge.RobodojoError):
        bridge.source_manifest(config)
    manifest = bridge.source_manifest(dict(config, replay_identity_warnings=True))
    assert manifest["warnings"]
    assert all(w["code"] == "implementation_identity_mismatch" for w in manifest["warnings"])
    assert manifest["source_revisions"]["robodojo"] == (None if drift == "unversioned_checkout" else "a" * 40)
    assert all("recorded" in w and "current" in w for w in manifest["warnings"])
    # The explicit diagnostic flag doesn't leak into a subsequent production call.
    with pytest.raises(bridge.RobodojoError):
        bridge.source_manifest(config)


def test_matching_production_manifest_and_physics_binding_are_unchanged(source_tree):
    production = bridge.source_manifest(source_tree)
    diagnostic = bridge.source_manifest(dict(source_tree, replay_identity_warnings=True))
    assert diagnostic.pop("warnings") == []
    assert diagnostic == production
    with pytest.raises(bridge.RobodojoError, match="cadence"):
        bridge.source_manifest(dict(source_tree, replay_identity_warnings=True, physics_dt=0.01))
