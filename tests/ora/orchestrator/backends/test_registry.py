"""The orchestrator's backend registry (ora/orchestrator/backends/registry.py).

Every ``runtime.kind`` loads its backend with the two required hooks; every
optional hook is either ``None`` or callable with the arguments its comment in
``OrchestratorBackend`` says the core passes; and the capabilities, admitters,
play hooks and tools are what the core reads instead of naming kinds.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

from ora.orchestrator.backends import registry

ROOT = Path(__file__).resolve().parents[4]


def test_every_registered_backend_loads_with_its_required_hooks():
    for kind, reference in registry.BACKENDS.items():
        loaded = registry.backend(kind)
        assert loaded.kind == kind, reference
        assert callable(loaded.resolve_runtime) and callable(loaded.preflight_runtime)
        assert registry.backend(kind) is loaded  # imported once, then cached
    assert [item.kind for item in registry.backends()] == list(registry.BACKENDS)


def test_an_alias_names_its_kind_and_an_unknown_kind_has_no_backend():
    assert registry.backend("libero") is registry.backend("libero_pro")
    assert registry.canonical("libero") == "libero_pro"
    for kind in ("isaac", None, 3, ""):
        assert registry.find(kind) is None and registry.canonical(kind) is None
    with pytest.raises(registry.UnknownBackend, match="isaac"):
        registry.backend("isaac")


def test_the_registry_itself_imports_no_ora_module():
    tree = ast.parse((ROOT / "ora/orchestrator/backends/registry.py").read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not {name for name in imported if name and name.split(".")[0] == "ora"}


def _kinds(predicate) -> set[str]:
    return {item.kind for item in registry.backends() if predicate(item.capabilities)}


def test_the_capabilities_are_the_kind_lists_the_core_used_to_name():
    assert _kinds(lambda caps: caps.captures_execution) == set(registry.BACKENDS) - {"synthetic"}
    assert _kinds(lambda caps: caps.persistent_session) == {"synthetic", "libero_pro", "robotwin", "robodojo"}
    assert _kinds(lambda caps: caps.boolean_verdict_required) == {"robotwin", "robodojo", "robocasa", "robolab"}
    assert _kinds(lambda caps: caps.native_requires_images) == {"libero_pro", "robodojo", "robocasa", "robolab", "yam"}
    assert _kinds(lambda caps: caps.signal_policies != {"lifecycle_only"}) == {"robotwin"}
    # The YAM station is a real cell played freely: no benchmark instance, no task specs.
    assert _kinds(lambda caps: caps.baseline_instance) == set(registry.BACKENDS) - {"synthetic", "yam"}
    assert _kinds(lambda caps: caps.baseline_requires_seed_manifest) == {"robotwin"}
    assert _kinds(lambda caps: caps.physics_steps_per_native_action is not None) == {"robodojo"}
    assert _kinds(lambda caps: caps.kit_storage_reason is not None) == {"robodojo", "robolab"}
    descriptors = [path for item in registry.backends() for path in item.capabilities.pinned_descriptors]
    assert descriptors == ["ora/policy/embodiments/arx-x5/descriptor-v1.json"]
    assert all((ROOT / path).is_file() for path in descriptors)


def test_a_manifest_is_admitted_by_the_backend_that_names_its_schema():
    from ora.orchestrator.backends.robodojo import admission as robodojo
    from ora.orchestrator.backends.robotwin import admission as robotwin, play_audit, seed_manifest
    from ora.orchestrator.robodojo_play_audit import AUDIT_SCHEMA as ROBODOJO_AUDIT

    admitter = registry.manifest_admitter
    assert admitter(seed_manifest.SEED_MANIFEST_SCHEMA, kind="robotwin") is robotwin.admit_seed_manifest
    assert admitter(play_audit.AUDIT_SCHEMA, kind="robotwin") is robotwin.admit_play_audit
    assert admitter(ROBODOJO_AUDIT, kind="robodojo") is robodojo.admit_play_audit
    # Another backend's schema is still judged by that backend; a schema nobody names
    # (an older per-task RoboTwin manifest has none) by the default admitter.
    assert admitter(ROBODOJO_AUDIT, kind="robotwin") is robodojo.admit_play_audit
    assert admitter(None, kind="robodojo") is robotwin.admit_seed_manifest
    assert admitter("ora.unknown.v1", kind="isaac") is robotwin.admit_seed_manifest


def test_the_seed_manifest_rules_digest_as_the_evidence_reader_does():
    from ora.orchestrator.backends.robotwin import seed_manifest
    from ora.orchestrator.evaluation.evidence_io import digest

    value = {"b": [1, 2.5, "ü"], "a": {"z": None, "y": True}}
    assert seed_manifest.digest(value) == digest(value)


def test_each_play_environment_is_one_backends_play_hook():
    from ora.orchestrator import robodojo_play_audit
    from ora.orchestrator.play import scenes as play_scenes
    from ora.orchestrator.backends.robotwin import play_audit

    plays = {item.play.environment: item for item in registry.backends() if item.play is not None}
    assert set(plays) == set(play_scenes.ENVIRONMENTS)
    assert {name for name, item in plays.items() if item.play.scene_sets} == {"robotwin", "robodojo"}
    assert {name for name, item in plays.items() if item.play_audit is not None} == {"robotwin", "robodojo"}
    for name, item in plays.items():
        assert registry.backend(name) is item  # libero names libero_pro
    # Both scene-set audits write where play.scenes looks for them.
    assert play_audit.audit_root() == robodojo_play_audit.audit_root() == play_scenes.play_audit_root()


def test_doctor_and_setup_read_what_each_simulator_backend_declares():
    from ora.app import gpu_recipes

    probes = {item.kind: item.doctor for item in registry.backends() if item.doctor is not None}
    assert {kind: (probe.python_env, probe.module, probe.directory_envs) for kind, probe in probes.items()} == {
        "robotwin": ("ROBOTWIN_PYTHON", "sapien", ("ROBOTWIN_ROOT",)),
        "libero_pro": ("LIBERO_PRO_PYTHON", "libero", ("LIBERO_PRO_ROOT", "LIBERO_PRO_DATA_ROOT")),
        "robodojo": ("ROBODOJO_PYTHON", "isaacsim", ("ROBODOJO_ROOT", "ROBODOJO_ASSET_ROOT")),
    }
    for name in ("libero", "robotwin", "robodojo"):
        components = registry.backend(name).gpu_setup
        assert components and set(components) <= set(gpu_recipes.COMPONENTS)
    assert all(item.spec_tools for item in registry.backends() if item.kind not in {"synthetic", "yam"})
    tools = {item.kind: (sorted(item.spec_tools), sorted(item.qualify_tools)) for item in registry.backends()}
    assert tools == {
        "synthetic": ([], []),
        "robotwin": (["compose", "spec", "tasks"], ["configs", "merge", "prepare", "qualify", "seal"]),
        "libero_pro": (["compose", "spec", "tasks"], ["motion"]),
        "robodojo": (["compose", "spec", "tasks"], ["card", "motion", "motion-metrics", "motion-replay", "physics-probe",
                                                    "revalidate", "smoke", "task"]),
        "robocasa": (["compose", "language"], ["motion"]),
        "robolab": (["compose", "tasks"], ["contract-witness", "motion"]),
        "yam": ([], ["motion"]),
    }
    # Every tool names an importable function; the run cells start these two as subprocesses.
    for item in registry.backends():
        for tool in (*item.spec_tools.values(), *item.qualify_tools.values()):
            module, _, function = tool.target.partition(":")
            assert callable(getattr(importlib.import_module(module), function)), tool.target
    assert registry.backend("robotwin").qualify_tools["qualify"].command("py", "--plan", "p") == [
        "py", "-m", "ora.orchestrator.backends.robotwin.evaluation_seeds", "qualify", "--plan", "p"]
    assert registry.backend("robodojo").qualify_tools["card"].command("py", "stack_bowls")[2:] == [
        "ora.orchestrator.backends.robodojo.lane_qualification", "card", "stack_bowls"]


#: How the core calls each optional hook (the comment beside each field of
#: ``OrchestratorBackend``): positional and keyword arguments a hook must bind.
CALLED_AS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "resolve_runtime": (("runtime", "policy", "resolved_surface"), ("task", "seed", "task_instruction")),
    "preflight_runtime": (("plan", "context"), ()),
    "validate_session": (("runtime", "session"), ()),
    "expand_template": (("runtime", "template", "requested_baseline"), ()),
    "provider_defaults": (("policy", "runtime_spec"), ()),
    "revision_label": (("runtime",), ()),
    "supervisor_binding": (("run_root", "runtime"), ()),
    "official_complete": (("official", "round_results"), ("multi_round", "round_limit")),
    "evidence_errors": (("run_root", "plan", "result", "events"), ()),
    "round_record_errors": (("plan", "record", "index"), ()),
    "evaluation_extras": (("plan", "run_root"), ()),
    "seal_extras": (("evaluation",), ()),
}
#: Hooks that are a module or an object: attribute -> (positional, keyword) arguments it is called with.
MEMBERS: dict[str, dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "qualification_exception": {"resolve_exception": (("qualification", "bind"), ()),
                                "check_exception": (("plan", "workspace_root"), ())},
    "seed_manifest": {name: (("manifest",), ("task", "seed", "task_config", "revision"))
                      for name in ("seed_admission", "expert_unsolved")},
    "round_scene": {"require_round_scene": (("plan", "reset", "index"), ()),
                    "round_scene_field": (("plan", "reset"), ()),
                    "layout_change_sentence": (("plan", "index"), ())},
    "play": {"list_scenes": ((), ("task", "task_config", "scene_set", "root")),
             "draw": (("rng", "listing"), ()), "label": (("scene",), ()),
             "build_spec": (("spec", "chosen"), ("host", "resource_profile", "service_report", "root")),
             "refuse_benchmark": (("runtime",), ()), "bind_runtime": (("runtime", "scene", "scenes"), ()),
             "require_scenes": (("runtime", "scenes"), ())},
    "play_audit": {"audit_root": ((), ()), "main": (("argv",), ())},
    "run_cell": {"compose": (("args",), ("common", "host", "services", "confirm", "stream"))},
}


def _binds(function: Any, arguments: tuple[tuple[str, ...], tuple[str, ...]]) -> bool:
    positional, keywords = arguments
    try:
        inspect.signature(function).bind(*positional, **{name: name for name in keywords})
    except TypeError:
        return False
    return True


@pytest.mark.parametrize("kind", list(registry.BACKENDS))
def test_every_hook_is_absent_or_callable_as_the_core_calls_it(kind: str) -> None:
    item = registry.backend(kind)
    declared = {field.name for field in dataclasses.fields(item)}
    assert set(CALLED_AS) | set(MEMBERS) <= declared  # this table names real fields
    for name, arguments in CALLED_AS.items():
        hook = getattr(item, name)
        assert hook is None or (callable(hook) and _binds(hook, arguments)), (kind, name)
    for name, members in MEMBERS.items():
        hook = getattr(item, name)
        for member, arguments in ({} if hook is None else members).items():
            assert callable(getattr(hook, member, None)) and _binds(getattr(hook, member), arguments), (
                kind, name, member)
    for schema, admitter in item.manifest_admitters.items():
        assert _binds(admitter, (("manifest", "plan"), ("seed", "root", "memo"))), (kind, schema)
    if item.play is not None:
        assert isinstance(item.play.environment, str) and isinstance(item.play.display_name, str)
        assert isinstance(item.play.scene_sets, bool) and isinstance(item.play.bound_record, bool)
    if item.play_audit is not None:
        assert isinstance(item.play_audit.KIND, str) and isinstance(item.play_audit.AUDIT_SCHEMA, str)
        # Each backend's own play code calls its admission with the backend's evidence keywords besides these.
        parameters = inspect.signature(item.play_audit.play_scene_admission).parameters
        assert {"record", "scene_set", "scene", "seed"} <= set(parameters), kind
    if item.run_cell is not None:
        assert isinstance(item.run_cell.slot, str) and isinstance(item.run_cell.example, str)
        assert all(isinstance(name, str) for name in item.run_cell.required_env)
    for tool in (*item.spec_tools.values(), *item.qualify_tools.values()):
        assert isinstance(tool, registry.Tool) and tool.help.strip() and ":" in tool.target
    if item.doctor is not None:
        assert item.doctor.extra_checks is None or _binds(item.doctor.extra_checks, (("interpreter", "nvidia_smi"), ()))
    assert item.gpu_setup is None or all(isinstance(component, str) for component in item.gpu_setup)
