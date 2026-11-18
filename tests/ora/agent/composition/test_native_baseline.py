"""The Native robot default composition and its fail-closed live prerequisites.

The default is direct_v4 on LIBERO, RoboTwin and RoboDojo with its audited
current_v4 bundles; a tool order other than the canonical one is refused, and
explicit tool-surface and non-robot specs pass through default expansion
unchanged. The default tool count is per backend: RoboTwin carries
align_plan, LIBERO does not.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ora.agent.composition.surface import tool_names

from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from tests.support.orchestrator.run_specs import native_spec, robodojo_native_spec


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("environment,profile,tool_count", [
    # Twenty-one shared tools plus the Artifact reader on both backends, and
    # read-only align_plan on RoboTwin only: its eight registration layers end
    # at RoboTwinSession, and the approach poses the policy side synthesises
    # put the approach axis on local +X, which is ALOHA's convention and not
    # the Panda grip site's.  Advertising it on LIBERO would advertise a tool
    # that cannot answer.
    # The shared read-only geometry_query is answerable on both backends
    # because `_geometry` reads published RGB-D and calibration alone.
    # The count then falls by one: set_gripper replaces open_gripper and
    # close_gripper, two tools that pinned arm coordination to one arm per
    # call on the embodiment where coordination is what varies.
    # It rises by one again for compute, which is backend-neutral: the
    # evaluator reads no observation, so both surfaces take the same entry --
    # and falls by one for vision_offset, whose single subtraction compute
    # answers alongside everything else that step needs. The count is unchanged
    # by the pair; the tools are not.
    # direct_v4 (integration 0059 unit A0, user decision 2026-10-04): the v2
    # generation retires place_plan, place_verify, grasp_verify_retention and
    # servo on LIBERO and adds the shared wait_ticks (21 - 4 + 1). Integration
    # 0084 moves LIBERO to direct_v2's overhead successor, which adds no tool.
    ("libero_pro", "libero_placement_direct_overhead_v1", 18),
    # B2 closed skills: grasp_execute, place_execute and place_resume replace
    # servo and place_verify (+3 -2), then artifact_read; v2 retires
    # place_plan, place_execute, place_resume and grasp_verify_retention and
    # adds the shared wait_ticks (23 - 4 + 1).
    ("robotwin", "robotwin_toolchain_b2_closed_noservo_v2", 20),
])
def test_new_native_robot_spec_defaults_to_complete_baseline(environment, profile, tool_count):
    spec = native_spec(environment)
    original = copy.deepcopy(spec)
    plan = resolve_run_spec(spec, root=ROOT)
    assert spec == original
    policy = plan["policy"]
    assert policy["baseline"]["id"] == "native_full_explicit_motion_direct_v4"
    assert policy["provider_recovery"] == "first_token_deadline_v1"
    assert plan["runtime"]["public_geometry_descriptor"] == plan["policy_composition"]["tool_surface"]["view_config"]["public_geometry_descriptor"]
    assert policy["baseline"]["selection"] == "omitted_tool_surface"
    assert len(policy["baseline"]["sha256"]) == 64
    assert policy["tool_surface"]["profile"] == profile
    assert policy["action_codec"] == "motion_action_v3"
    # The baseline names its context half (context management v6) as a scheme; the plan records which.
    assert policy["context_scheme"] == {"id": "context_lossless_v6",
                                        "path": "ora/policy/context_schemes/lossless-v6.json",
                                        "sha256": policy["context_scheme"]["sha256"]}
    assert policy["context_profile"] == "native_causal_context_v1"
    assert policy["context_composition"] == "native_slots_v4"
    assert policy["context_retention"] == "action_step_lossless_v4"
    assert policy["context_management"]["schema"] == (
        "ora.native-context-management.v6"
    )
    assert policy["context_management"]["tail_aging_stride_action_steps"] == 4
    assert policy["context_management"]["trigger_inline_images"] == 24
    # v6 (user decision 2026-10-04): L5 is an absolute 150,000 tokens.
    assert policy["context_management"]["episode_checkpoint_trigger_tokens"] == 150000
    assert "episode_checkpoint_trigger_ratio" not in policy["context_management"]
    assert policy["memory_policy"] == "disabled"
    assert policy["provider"] == "openai_compatible"
    assert policy["model"] == "test-model"
    assert plan["budgets"] == spec["budgets"]
    assert "session" not in plan
    definitions = plan["policy_composition"]["tool_surface"]["definitions"]
    names = [item["name"] for item in definitions]
    assert len(names) == tool_count  # Direct-pose/Servo tools and Artifact reader.
    assert ("align_plan" in names) is (environment == "robotwin")
    assert "motion.probe" not in names
    # Named by the surface that carries it: canonical_tool_names_v1 now reaches
    # the runtime extension reader too, so a snake-case surface advertises no
    # dotted name at all.
    assert "artifact_read" in names
    # The live surfaces declare canonical_tool_names_v1; the assertion is about
    # the tool, not about which spelling it carries. Retired by undeclaration
    # on the v2 generation (P04 extended, user decision 2026-10-04).
    assert not tool_names.declares(names, "grasp_verify_retention")
    assert "servo" not in names and "place_plan" not in names
    assert not any(name.startswith("memory.") for name in names)
    for resources in plan["policy_context"].values():
        if isinstance(resources, list):
            assert all(not item["path"].startswith("experiments/") for item in resources)
    assert plan["prerequisite"]["status"] == "unbound_preflight_required"
    assert "required_host" not in plan["runtime"]
    assert "resource_binding" not in plan["runtime"]


@pytest.mark.parametrize("environment", ["libero_pro", "robotwin"])
def test_full_baseline_missing_services_fails_preflight_without_connecting(environment):
    from ora.orchestrator.preflight.gates import _resident_service_prerequisite

    plan = resolve_run_spec(native_spec(environment), root=ROOT)
    result = _resident_service_prerequisite(plan)
    assert result["passed"] is False
    assert "no services" in str(result["detail"])


def test_explicit_baseline_refuses_foreign_surface_and_resolver_owned_binding():
    spec = native_spec("libero_pro")
    spec["policy"]["baseline"] = "native_full_explicit_motion_direct_v4"
    spec["policy"]["tool_surface"] = {"profile": "robotwin_manipulation_v1"}
    with pytest.raises(RunSpecError, match="baseline.*tool_surface"):
        resolve_run_spec(spec, root=ROOT)
    spec = native_spec("libero_pro")
    spec["policy"]["baseline"] = {"id": "native_full_v1", "sha256": "0" * 64}
    with pytest.raises(RunSpecError, match="baseline.*id|baseline.*string"):
        resolve_run_spec(spec, root=ROOT)


@pytest.mark.parametrize("field,value", [
    ("memory_policy", "enabled"), ("context_retention", "rewrite_compact_v1"),
    ("require_live_marker", False), ("supports_images", False),
    ("tool_allowlist", ["observe"]),
])
def test_baseline_rejects_partial_or_incompatible_policy(field, value):
    spec = native_spec("libero_pro")
    spec["policy"][field] = value
    with pytest.raises(RunSpecError, match="baseline"):
        resolve_run_spec(spec, root=ROOT)


@pytest.mark.parametrize("value", [None, {}, {"schema": "unknown"}])
def test_baseline_prerequisite_cannot_be_disabled(value):
    spec = native_spec("libero_pro")
    spec["prerequisite"] = value
    with pytest.raises(RunSpecError, match="baseline.*prerequisite"):
        resolve_run_spec(spec, root=ROOT)


def test_baseline_prerequisites_require_both_services_and_robotwin_seed(monkeypatch):
    from ora.orchestrator.preflight import gates as preflight

    def no_network(*args, **kwargs):
        pytest.fail("an incomplete service set must fail before any network call")

    monkeypatch.setattr(preflight, "_verify_resident_service", no_network)
    plan = resolve_run_spec(native_spec("robotwin"), root=ROOT)
    plan["prerequisite"]["services"] = {"segmentation": {
        "schema": "ora.segmentation-service-binding.v1",
    }}
    assert preflight._resident_service_prerequisite(plan)["passed"] is False
    assert preflight._qualification(plan)["passed"] is False


@pytest.mark.parametrize("field,value", [
    ("require_gpu", False), ("required_revision", "foreign-source"),
    ("control_profile", {"path": "wrong", "sha256": "0" * 64}),
    ("public_environment_context", {}),
])
def test_baseline_rejects_controller_source_or_metadata_drift(field, value):
    spec = native_spec("robotwin")
    spec["runtime"][field] = value
    with pytest.raises(RunSpecError, match="baseline"):
        resolve_run_spec(spec, root=ROOT)


@pytest.mark.parametrize("kind,environment,surface", [
    ("native_api", "libero_pro", "libero_unified_manip_v1"),
    ("native_api", "robotwin", "robotwin_unified_manip_v1"),
    ("native_api", "synthetic", None),
])
def test_explicit_historical_and_nonrobot_defaults_are_unchanged(kind, environment, surface):
    from ora.orchestrator.spec.defaults import expand_native_defaults

    spec = native_spec(environment)
    spec["policy"]["kind"] = kind
    if surface:
        spec["policy"]["tool_surface"] = {"profile": surface}
    original = copy.deepcopy(spec)
    assert expand_native_defaults(spec, root=ROOT) == original
    assert spec == original


@pytest.mark.parametrize("backend,stem", [("libero_pro", "libero-panda"), ("robotwin", "robotwin-aloha")])
@pytest.mark.parametrize("kind", ["native_api"])
def test_current_robot_bundle_audits_and_matches_default_selection(backend, stem, kind):
    from ora.agent.composition.prompt.bundle import load_prompt_bundle, audit_prompt_bundle
    from ora.orchestrator.spec.defaults import expand_native_defaults

    spec = native_spec(backend)
    spec["policy"]["kind"] = kind
    original = copy.deepcopy(spec)
    expanded = expand_native_defaults(spec, root=ROOT)
    from ora.agent.composition.baseline import load_native_baseline
    selected = load_native_baseline(ROOT)["instances"][backend]
    bundle_path = selected["prompt_bundle"].replace("native_api", kind)
    bundle = load_prompt_bundle(ROOT / bundle_path)
    report = audit_prompt_bundle(bundle, project_root=ROOT)
    assert report["passed"], [check for check in report["checks"] if not check["passed"]]
    assert bundle["layers"]["tool_surface"]["profile"] == expanded["policy"]["tool_surface"]["profile"]
    assert bundle["runtime"]["policy_kind"] == kind
    # direct_v4 (integration 0059 unit A0) selects the current_v4 bundles; the
    # Native one carries core v8 (the round static-facts slot and the delivery
    # trim, revised in place over v7).
    assert bundle["bundle_id"] == f"direct_v4_{backend.removesuffix('_pro')}_{kind}_v1"
    core = bundle["layers"]["core"]["sources"]
    assert core == ["ora/policy/core/evidence-action-loop-v8.md"]
    assert spec == original


def test_catalog_inspection_and_explicit_selection(capsys):
    from ora.app.main import main

    assert main(["baselines", "--json"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["id"] == "native_full_explicit_motion_direct_v4"
    assert set(catalog["instances"]) == {"libero_pro", "robotwin", "robocasa", "robolab"}
    spec = native_spec("libero_pro")
    spec["policy"]["baseline"] = "native_full_explicit_motion_direct_v4"
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["policy"]["baseline"]["selection"] == "explicit"
    assert plan["policy"]["baseline"]["sha256"] == catalog["sha256"]


def test_a_fresh_robodojo_native_spec_inherits_the_robodojo_default_baseline():
    """RoboDojo's default is its own baseline (robodojo_toolkit_v7: toolkit v6, which has no official check), in code
    mode (its ``code_mode``: the extended library, prompt v3 and ORA's source, 2026-11-05)."""
    spec = robodojo_native_spec()
    original = copy.deepcopy(spec)
    plan = resolve_run_spec(spec, root=ROOT)
    assert spec == original
    policy = plan["policy"]
    assert policy["baseline"]["id"] == "robodojo_toolkit_v7" and policy["baseline"]["selection"] == "omitted_tool_surface"
    assert policy["baseline"]["instance"] == "robodojo_arx_x5"
    assert policy["tool_surface"]["profile"] == "robodojo_placement_toolkit_v6"
    assert plan["toolkit_grants"] == {"tools": [], "state_keys": []}
    assert policy["context_composition"] == "native_slots_v4"
    assert policy["context_scheme"]["id"] == "context_keep_written"
    assert policy["context_management"]["schema"] == "ora.native-context-management.v10"
    assert policy["context_management"]["compaction_unit"] == "exchange"
    assert policy["context_management"]["reasoning_replay"] == "all"
    assert policy["context_tools"] == ["artifact_read"]
    assert policy["baseline"]["prompt_bundle"]["path"].endswith("toolkit_v7/robodojo.yaml")
    # The baseline names no execution host; the RunSpec binds a registered one.
    assert "required_host" not in plan["runtime"]
    assert plan["runtime"]["workspace_profile"]["profile_id"] == "robodojo_arx_x5_tabletop_v1"
    assert policy["task_contract"]["delivered"] is True
    assert (policy["code_mode"]["library"], policy["code_mode"]["prompt"], policy["code_mode"]["source"]) == (
        "extended", "v3", True)
    assert "preview" not in policy["code_mode"]
    names = policy["tool_allowlist"]
    assert {"wait_ticks", "return_home"} <= set(names) and not {"check_task", "servo"} & set(names)
    # The model gets the code tools; the surface's other tools are functions of the robot library.
    advertised = {item["name"] for item in plan["policy_composition"]["tool_surface"]["definitions"]}
    assert {"read", "write", "edit", "bash", "finish_episode"} <= advertised and not {"wait_ticks", "move_pose"} & advertised


@pytest.mark.parametrize("task_config", ["demo_clean", "demo_randomized"])
def test_robotwin_baseline_accepts_either_official_scene_configuration(task_config):
    spec = native_spec("robotwin")
    spec["runtime"]["task_config"] = task_config
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["runtime"]["task_config"] == task_config


@pytest.mark.parametrize("task_config", ["demo_easy", ["demo_clean"]])
def test_robotwin_baseline_refuses_an_unofficial_scene_configuration(task_config):
    # A non-string value is refused through the same contract error, never a traceback.
    spec = native_spec("robotwin")
    spec["runtime"]["task_config"] = task_config
    with pytest.raises(RunSpecError, match="task_config"):
        resolve_run_spec(spec, root=ROOT)
