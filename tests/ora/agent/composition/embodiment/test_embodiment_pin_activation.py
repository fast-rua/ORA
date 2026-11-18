"""Every descriptor this tree composes names its sources by path.

Since stage 2 unit L3 a descriptor names its repository sources by path only:
the RunPlan's source commit identifies their bytes, so an edit to a source no
longer invalidates the descriptors that cite it. Every descriptor in the tree is
composed (bound by a context profile, a ToolSurface or code) and audits
path-only, or is kept byte for byte for the runs sealed with it and answered by
a successor (``moved_sources.MOVED``) that does; nothing composes or audits
such a descriptor any more, and one may name a source that left the tree.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.composition.baseline import BASELINE_CATALOGS, load_native_baseline
from ora.agent.composition.embodiment import (
    EmbodimentDescriptorError,
    audit_embodiment_sources,
    load_embodiment_descriptor,
)
from ora.agent.composition.embodiment.geometry import PUBLIC_GEOMETRY_DESCRIPTOR_VERSIONS
from ora.agent.composition.embodiment.moved_sources import MOVED
from ora.agent.composition.embodiment.environment_context import resolve_environment_prompt_profile
from ora.agent.composition.prompt.bundle import load_prompt_bundle
from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG
from ora.orchestrator.spec.policy_pins import composable_pin_holders

ROOT = Path(__file__).resolve().parents[5]
ACTIVATION = json.loads((ROOT / "tests/data/default-expectations.json").read_text(encoding="utf-8"))
EMBODIMENTS = "ora/policy/embodiments/"

# The default baseline plus every maintained baseline selectable from this tree.
MAINTAINED_BASELINES = {ACTIVATION["default_baseline"], *ACTIVATION["maintained_baselines"]}

# Descriptors whose geometry a registered ToolSurface binds although no context
# profile compiles a card from them and no maintained bundle names them;
# None is left.
GEOMETRY_ONLY_DESCRIPTORS: dict[str, set[str]] = {}
# The descriptors the public-facts code resolves by controlled-frame convention.
CODE_REGISTERED_DESCRIPTORS = {
    EMBODIMENTS + item for paths in PUBLIC_GEOMETRY_DESCRIPTOR_VERSIONS.values() for item in paths
}
# Descriptors no composition binds any more, kept byte for byte for the runs sealed with them.
SEALED_ONLY_DESCRIPTORS = {old for (old, _), new in MOVED.items() if old.startswith(EMBODIMENTS)}

def _audits(path: str) -> bool:
    try:
        audit_embodiment_sources(load_embodiment_descriptor(ROOT / path), project_root=ROOT)
    except EmbodimentDescriptorError:
        return False
    return True


def _profiles() -> dict[str, tuple[str, dict]]:
    return {
        value["profile_id"]: (path.relative_to(ROOT).as_posix(), value)
        for path in sorted((ROOT / "ora/policy/environments").glob("*/context-profile-*.json"))
        for value in [json.loads(path.read_text(encoding="utf-8"))]
    }


def _registered_geometry() -> dict[str, set[str]]:
    bound: dict[str, set[str]] = {}
    for profile_id, path in PROFILE_CATALOG.items():
        geometry = (json.loads(path.read_text(encoding="utf-8")).get("view_config") or {}).get(
            "public_geometry_descriptor")
        if geometry:  # a registered manifest names its descriptor by path
            bound.setdefault(geometry, set()).add(profile_id)
    return bound


def test_every_baseline_is_the_default_or_maintained():
    ids = {json.loads(p.read_text(encoding="utf-8"))["id"]
           for p in (ROOT / "ora/policy/baselines").glob("*.json")}
    assert ids == MAINTAINED_BASELINES
    assert set(BASELINE_CATALOGS) == MAINTAINED_BASELINES


@pytest.mark.parametrize("baseline_id", sorted(MAINTAINED_BASELINES))
def test_maintained_baseline_pins_verify_on_every_instance(baseline_id):
    baseline = load_native_baseline(ROOT, baseline_id)
    for instance, config in baseline["instances"].items():
        bundle = load_prompt_bundle(ROOT / config["prompt_bundle"])
        reference = (bundle["layers"].get("embodiment") or {}).get("descriptor")
        assert reference is not None, (baseline_id, instance)
        assert reference not in GEOMETRY_ONLY_DESCRIPTORS, (baseline_id, instance)
        assert _audits(reference), (
            f"{baseline_id} {instance} still binds a superseded descriptor: {reference}. "
            "Rebind it to the current feedback-source successor instead of re-pinning the predecessor."
        )


def test_every_context_profile_resolves():
    for profile_id, (_, profile) in _profiles().items():
        metadata = {"schema": "ora.public-environment-context.v1", **profile["match"]}
        resolve_environment_prompt_profile(metadata, project_root=ROOT, profile_id=profile_id)


def test_every_descriptor_is_composed_and_path_only():
    """Every descriptor in the tree is composed and audits path-only."""
    holders = composable_pin_holders(ROOT)
    bound = _registered_geometry()
    compiled = {
        binding["descriptor"]
        for _, profile in _profiles().values()
        for binding in (profile.get("compiled_embodiment"),) if binding
    }
    assert all(_audits(new) for (old, _), new in MOVED.items() if old in SEALED_ONLY_DESCRIPTORS)
    for path in sorted((ROOT / EMBODIMENTS).glob("*/descriptor-*.json")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in SEALED_ONLY_DESCRIPTORS:
            assert relative not in holders and relative not in CODE_REGISTERED_DESCRIPTORS, relative
            continue
        assert relative in holders or relative in CODE_REGISTERED_DESCRIPTORS, f"descriptor not composed: {relative}"
        assert _audits(relative), relative
    geometry_only = {path for path, kind in holders.items() if kind == "geometry_descriptor"}
    assert geometry_only == set(GEOMETRY_ONLY_DESCRIPTORS)
    for relative, surfaces in GEOMETRY_ONLY_DESCRIPTORS.items():
        assert bound.get(relative) == surfaces, relative
        assert relative not in compiled


@pytest.mark.parametrize("backend", ["libero_pro", "robodojo"])
def test_the_default_surface_binds_the_descriptor_the_activation_contract_names(backend):
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface

    expected = ACTIVATION["instances"][backend]
    binding = resolve_tool_surface(expected["tool_surface"]).view_config["public_geometry_descriptor"]
    assert binding["path"] == expected["public_geometry_descriptor"]
    assert binding["path"] not in SEALED_ONLY_DESCRIPTORS
