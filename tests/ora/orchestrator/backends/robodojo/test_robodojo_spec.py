"""The maintained RoboDojo builder: fresh RunSpecs on the default baseline.

New RoboDojo experiments compose here (ora/orchestrator/backends/robodojo/spec.py),
as RoboTwin and LIBERO-Pro experiments start from the default resolver. These
tests also carry the product rules that were reached only through the earlier
Dojo builder and the study launchers: the registry-v10 contract roster and delivery, each task's
reviewed runtime and physics device, the per-round layout schedule, the
RoboDojo recovery window, executor routes and scheduled-contract disclosure.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from ora.agent.composition.baseline import default_baseline_id, load_native_baseline
from ora.agent.composition.surface.toolkit_profiles import TOOLKIT_BASELINES
from ora.agent.composition.task_contract.robodojo import (
    REGISTRY_PATH, load_task_contract, resolve_reviewed_contract,
)
from ora.environment.benchmarks.robodojo import (
    NATIVE_ACTION_LIMITS, RoboDojoBenchmarkError, physics_step_ceiling,
)
from ora.environment.benchmarks.robodojo_layouts import REPEAT_ROUND0_LAYOUT
from ora.environment.benchmarks.robodojo_physics import physics_device_for
from ora.orchestrator.backends.robodojo import spec as robodojo_spec
from ora.orchestrator.evaluation.model_visible import model_visible_items
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from tests.support.robodojo import runtime as robodojo_runtime

ROOT = Path(__file__).resolve().parents[5]
ACTIVATION = json.loads((ROOT / "tests/data/default-expectations.json").read_text())
REGISTRY = json.loads((ROOT / REGISTRY_PATH).read_text())["tasks"]
ASSETS = Path(os.environ.get("ROBODOJO_ASSET_ROOT", str(robodojo_runtime.ASSET_ROOT)))
needs_assets = pytest.mark.skipif(not (ASSETS / "Eval_Layout/RoboDojo/arx_x5/0").is_dir(),
                                  reason="pinned RoboDojo Assets absent")
FLUID = ("pour_liquid_into_cup", "pour_by_language")
QUALIFICATION = {"path": "outputs/robodojo-spec-fixture/qualification/fixture.json", "sha256": "0" * 64}
EXPERIMENT = "robodojo-spec-fixture"
#: Policy fields a Provider profile owns; a fresh default spec and a builder spec differ only here.
PROVIDER_OWNED = {*robodojo_spec.PROVIDER_FIELDS, "provider", "provider_profile", "provider_session_affinity",
                  "provider_qualification_schema", "provider_qualification_version"}


@pytest.fixture(autouse=True)
def _registered_execution_host(monkeypatch):
    """RoboDojo runtimes bind the marker's registered host; tests run on the fixture host."""
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")


@pytest.fixture(autouse=True)
def _restore_service_endpoints():
    """compose and bind_service_environment write the tool-client endpoints into
    os.environ themselves; monkeypatch records nothing for an absent variable,
    so restore both here or the fixture endpoints reach later test modules."""
    saved = {variable: os.environ.pop(variable, None) for _, variable in robodojo_spec.SERVICE_VARIABLES}
    yield
    for variable, value in saved.items():
        if value is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = value



def _spec(task="stack_bowls", **kwargs):
    kwargs.setdefault("gpu", "0")
    return robodojo_spec.build_spec(task, experiment=EXPERIMENT, infrastructure_qualification=QUALIFICATION,
                                    **kwargs)


# --- the contract roster -------------------------------------------------------------------------

def test_the_roster_is_every_task_the_default_registry_contracts():
    roster = robodojo_spec.task_roster()
    assert roster == tuple(REGISTRY) and len(roster) == 54
    # registry-v10: every admitted task has a contract, the 42 official tasks and the twelve *_random
    # variants of the generalization tasks, which follow them.
    assert set(roster) == set(NATIVE_ACTION_LIMITS)
    assert all(not task.endswith("_random") for task in roster[:42])
    assert all(task.endswith("_random") for task in roster[42:])
    assert roster[40:42] == FLUID
    assert ACTIVATION["task_contracts"]["robodojo"]["registry"] == REGISTRY_PATH
    assert ACTIVATION["task_contracts"]["robodojo"]["task_count"] == 54
    with pytest.raises(ValueError, match="no reviewed contract"):
        robodojo_spec.task_contract("not_a_task")


@pytest.mark.parametrize("task", list(REGISTRY))
def test_each_task_contract_is_the_default_registry_record_and_layer(task):
    contract = robodojo_spec.task_contract(task)
    assert contract["layer"] == REGISTRY[task]["layer"]
    assert contract["record"] == json.loads((ROOT / REGISTRY[task]["record"]).read_text())


@pytest.mark.parametrize("task", ["stack_bowls", *FLUID])
def test_the_resolver_discloses_the_default_registry_and_refuses_another_instruction(task):
    record = robodojo_spec.task_contract(task)["record"]
    runtime = {"task_variant": task, "required_revision": record["source"]["robodojo_revision"],
               "required_asset_revision": record["layout"]["assets_revision"],
               "layout_file": record["layout"]["layout_file"], "layout_sha256": record["layout"]["layout_sha256"],
               "eval_seed": 0, "layout_id": 0, "expected_task_language": record["instruction"]["text"]}
    layer, disclosure = resolve_reviewed_contract(runtime, project_root=ROOT, disclosure={"backend": "robodojo"})
    assert disclosure["delivered"] is True and disclosure["registry"] == REGISTRY_PATH
    assert layer == REGISTRY[task]["layer"] and disclosure["record_sha256"] == hashlib.sha256(
        (ROOT / REGISTRY[task]["record"]).read_bytes()).hexdigest()
    runtime["expected_task_language"] = "A different task."
    with pytest.raises(RuntimeError, match="task instruction differs"):
        resolve_reviewed_contract(runtime, project_root=ROOT, disclosure={"backend": "robodojo"})


def test_a_layer_that_drifted_after_review_is_refused(tmp_path):
    entry = REGISTRY["stack_blocks"]
    for relative in (entry["record"], entry["layer"]):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    assert load_task_contract("stack_blocks", bindings={"stack_blocks": entry}, project_root=tmp_path)
    with (tmp_path / entry["layer"]).open("a") as stream:
        stream.write("- and one more thing.\n")
    with pytest.raises(RuntimeError, match="changed after review"):
        load_task_contract("stack_blocks", bindings={"stack_blocks": entry}, project_root=tmp_path)


# --- each task's reviewed runtime ------------------------------------------------------------------

@needs_assets
@pytest.mark.parametrize("task", list(REGISTRY))
def test_each_task_runs_on_its_reviewed_layout_limit_and_physics_device(task):
    record = robodojo_spec.task_contract(task)["record"]
    layout = ASSETS / record["layout"]["layout_file"]
    if layout.is_file() and robodojo_runtime.is_lfs_pointer(layout):
        robodojo_runtime.skip_lfs_pointers([layout], "every other task's layout was checked")
    runtime = robodojo_spec.runtime_for(task, gpu="4", resource_profile="robodojo-gpu4-d1")
    assert (runtime["task_variant"], runtime["layout_id"]) == (task, 0)
    assert (runtime["layout_file"], runtime["layout_sha256"]) == (
        record["layout"]["layout_file"], record["layout"]["layout_sha256"])
    assert runtime["native_action_limit"] == record["native_action_limit"] == NATIVE_ACTION_LIMITS[task]
    assert runtime["expected_task_language"] == record["instruction"]["text"]
    assert runtime["physics_device"] == physics_device_for(task)
    assert (runtime["cuda_visible_devices"], runtime["renderer_gpu"]) == ("4", 4)
    # The default instance owns the workspace profile and the resolver derives the public context.
    assert "workspace_bounds_m" not in runtime and "public_environment_context" not in runtime
    assert "layout_schedule" not in runtime


@needs_assets
@pytest.mark.parametrize("task", FLUID)
def test_the_fluid_tasks_run_one_round_on_cpu_physics_and_have_no_multi_round_pool(task):
    assert robodojo_spec.runtime_for(task, gpu="4", resource_profile="robodojo-gpu4-d1")["physics_device"] == "cpu"
    with pytest.raises(RoboDojoBenchmarkError, match="no reviewed layout pool"):
        robodojo_spec.runtime_for(task, gpu="4", resource_profile="robodojo-gpu4-d1", rounds=2)


@needs_assets
def test_a_persistent_session_starts_each_round_on_a_new_reviewed_layout():
    default = robodojo_spec.runtime_for("stack_bowls", gpu="0", rounds=3)
    rounds = default["layout_schedule"]["rounds"]
    assert [entry["layout_id"] for entry in rounds] == [23, 29, 51]
    assert {key: default[key] for key in ("layout_id", "layout_file", "layout_sha256")} == rounds[0]
    explicit = robodojo_spec.runtime_for("stack_bowls", gpu="0", rounds=2, layout_schedule=[29, 51])
    assert [entry["layout_id"] for entry in explicit["layout_schedule"]["rounds"]] == [29, 51]
    assert {key: explicit[key] for key in ("layout_id", "layout_file", "layout_sha256")} == (
        explicit["layout_schedule"]["rounds"][0])
    # The historical every-round-layout-0 treatment is an explicit selection, never a fallback.
    replay = robodojo_spec.runtime_for("stack_bowls", gpu="0", rounds=3, layout_schedule=REPEAT_ROUND0_LAYOUT)
    assert "layout_schedule" not in replay and replay["layout_id"] == 0
    with pytest.raises(RoboDojoBenchmarkError, match="no reviewed layout pool"):
        robodojo_spec.runtime_for("stack_blocks", gpu="0", rounds=2)
    with pytest.raises(ValueError, match="two or more rounds"):
        robodojo_spec.runtime_for("stack_bowls", gpu="0", layout_schedule=[29])


@needs_assets
def test_the_resource_profile_must_select_the_runtime_gpu():
    with pytest.raises(ValueError, match="resource profile and the runtime GPU differ"):
        robodojo_spec.runtime_for("stack_bowls", gpu="0", resource_profile="robodojo-gpu4-d1")
    with pytest.raises(ValueError, match="uncontracted"):
        robodojo_spec.runtime_for("not_a_task", gpu="0")


# --- the composed RunSpec ---------------------------------------------------------------------------

@needs_assets
@pytest.mark.parametrize("task", ["stack_bowls", "hang_mugs", "pack_objects_into_box"])
def test_a_new_robodojo_spec_resolves_the_default_instance_with_the_fresh_specs_model_visible_items(task):
    """What `ora baselines --explain --environment robodojo` resolves, plus the per-run inputs."""
    from tests.support.orchestrator.run_specs import robodojo_native_spec

    spec = _spec(task)
    for owned_by_the_baseline in ("tool_surface", "tool_allowlist", "semantic_manifest", "baseline",
                                  "context_profile", "context_composition", "action_codec"):
        assert owned_by_the_baseline not in spec["policy"]
    assert "policy_context" not in spec and "session" not in spec
    plan = resolve_run_spec(spec, root=ROOT)
    catalog = load_native_baseline(ROOT, default_baseline_id("robodojo"))
    instance = catalog["instances"]["robodojo"]
    baseline = plan["policy"]["baseline"]
    assert (baseline["id"], baseline["instance"], baseline["selection"]) == (
        catalog["id"], instance["id"], "omitted_tool_surface")
    assert plan["policy"]["tool_surface"]["profile"] == instance["tool_surface"]
    assert baseline["prompt_bundle"]["path"] == instance["prompt_bundle"]
    expected = ACTIVATION["instances"]["robodojo"]
    assert (baseline["id"], instance["tool_surface"], instance["prompt_bundle"]) == (
        expected["baseline"], expected["tool_surface"], expected["prompt_bundle"])
    assert plan["runtime"]["public_geometry_descriptor"]["path"] == expected["public_geometry_descriptor"]
    # The default runs code mode: the surface's tools are the allowlist, the robot library's functions.
    contract = plan["policy"]["code_mode"]
    assert {key: contract[key] for key in ACTIVATION["code_mode"]["robodojo"]} == ACTIVATION["code_mode"]["robodojo"]
    assert set(expected["required_tools"]) <= set(plan["policy"]["tool_allowlist"])
    assert plan["runtime"]["environment_prompt_profile"]["profile_id"] == instance["environment_prompt_profile"]
    fresh = resolve_run_spec(robodojo_native_spec(task), root=ROOT)
    ours, theirs = model_visible_items(plan, root=ROOT), model_visible_items(fresh, root=ROOT)
    for item in ("advertised", "prompt_layers", "embodiment", "observation_projection"):
        assert ours[item] == theirs[item], item
    assert ours["settings"]["runtime"] == theirs["settings"]["runtime"]
    policy, default = ours["settings"]["policy"], theirs["settings"]["policy"]
    assert {key for key in set(policy) | set(default) if policy.get(key) != default.get(key)} <= PROVIDER_OWNED
    assert plan["policy"]["task_contract"]["delivered"] is True
    assert plan["task_instruction"] == plan["runtime"]["expected_task_language"] == (
        robodojo_spec.task_contract(task)["record"]["instruction"]["text"])


@needs_assets
def test_the_builder_takes_the_shared_low_profile_budgets_and_admission_inputs():
    spec = _spec("stack_bowls")
    policy = spec["policy"]
    assert ACTIVATION["default_experiment_provider"]["profile_id"] == policy["provider_profile"] == (
        "deepseek_flash_openai_low_16k_v1")
    assert (policy["kind"], policy["reasoning_effort"], policy["max_output_tokens"]) == ("native_api", "low", 16384)
    assert spec["budgets"] == {"max_robot_decisions": 80, "max_model_calls": 200, "wall_time_s": 21600,
                               "max_sim_steps": physics_step_ceiling("stack_bowls", 1000)}
    assert spec["runtime"]["official_limit_multiple"] == ACTIVATION["robodojo_official_limit_multiple"] == 1000
    assert spec["runtime"]["native_action_limit"] == NATIVE_ACTION_LIMITS["stack_bowls"]
    official = _spec("stack_bowls", official_limit_multiple=1)
    assert official["runtime"]["official_limit_multiple"] == 1
    assert official["budgets"]["max_sim_steps"] == physics_step_ceiling("stack_bowls")
    assert (spec["config"], spec["seed"], spec["output_workspace"]) == (
        "stack_bowls-l0-s0", 0, f"outputs/{EXPERIMENT}")
    assert "launch_admission" not in spec and "prerequisite" not in spec
    chosen = _spec("stack_bowls", provider_profile="deepseek_flash_openai_high_16k_v1",
                   budgets={"max_robot_decisions": 40}, cohort_id="robodojo-spec-fixture-c1", rounds=1)
    assert (chosen["policy"]["provider_profile"], chosen["policy"]["reasoning_effort"]) == (
        "deepseek_flash_openai_high_16k_v1", "high")
    assert chosen["budgets"]["max_robot_decisions"] == 40
    assert chosen["session"] == {"cadence": "persistent_agent_k_rounds_v1", "rounds": 1}
    assert chosen["launch_admission"] == {"schema": "ora.run-launch-admission.v1",
                                          "mode": "single_attempt_cell", "cohort_id": "robodojo-spec-fixture-c1",
                                          "cell_id": "stack_bowls-l0-s0", "retry_allowed": False}
    plan = resolve_run_spec(chosen, root=ROOT)
    assert plan["launch_admission"]["cell_id"] == "stack_bowls-l0-s0" and plan["session"]["rounds"] == 1


@needs_assets
def test_the_default_robodojo_surface_derives_its_staged_approach_and_recovery_routes():
    plan = resolve_run_spec(_spec("stack_bowls"), root=ROOT)
    assert plan["runtime"]["contact_approach_route"] == "global_precontact_then_local"
    assert plan["runtime"]["recovery_route"] == "retract_then_unwind"
    assert plan["policy"]["provider_recovery"] == "first_token_deadline_v1"


@needs_assets
@pytest.mark.parametrize("window, recovery", [(10, None), (True, None), (600, "legacy_v1")])
def test_a_recovery_window_outside_its_range_or_policy_is_refused(window, recovery):
    accepted = _spec("stack_bowls", provider_recovery_window_s=600)
    assert resolve_run_spec(accepted, root=ROOT)["policy"]["provider_recovery_window_s"] == 600
    spec = _spec("stack_bowls", provider_recovery_window_s=window)
    if recovery is None:
        with pytest.raises(RunSpecError, match="provider_recovery_window_s must be 30..1800"):
            resolve_run_spec(spec, root=ROOT)
        return
    # Every RoboDojo surface brings its baseline (TOOLKIT_BASELINES), which owns provider_recovery, so a RunSpec
    # naming another recovery is refused there before the window rule; the rule still refuses the pair itself.
    from ora.orchestrator.spec.native_provider import resolve_native_provider_protocol

    spec["policy"]["provider_recovery"] = recovery
    with pytest.raises(RunSpecError, match="policy.baseline conflicts with policy.provider_recovery"):
        resolve_run_spec(spec, root=ROOT)
    policy = {**resolve_run_spec(accepted, root=ROOT)["policy"], "provider_recovery": recovery}
    with pytest.raises(RunSpecError, match="provider_recovery_window_s must be 30..1800"):
        resolve_native_provider_protocol(policy, accepted["runtime"])


@needs_assets
def test_a_spec_may_keep_the_earlier_deadline_recovery_over_the_baseline_but_not_legacy():
    """A RunSpec copied from a plan of the earlier default still resolves, on the recovery it names."""
    spec = _spec("stack_bowls", provider_recovery_window_s=900)
    spec["policy"]["provider_recovery"] = "same_credential_deadline_v1"
    policy = resolve_run_spec(spec, root=ROOT)["policy"]
    assert (policy["provider_recovery"], policy["provider_recovery_window_s"]) == ("same_credential_deadline_v1", 900)
    spec["policy"]["provider_recovery"] = "legacy_v1"
    spec["policy"].pop("provider_recovery_window_s")
    with pytest.raises(ValueError, match="policy.baseline conflicts with policy.provider_recovery"):
        resolve_run_spec(spec, root=ROOT)


@needs_assets
def test_robodojo_without_a_baseline_inherits_the_maintained_provider_recovery():
    from ora.orchestrator.backends.robodojo.runtime import provider_defaults

    # A RunSpec that names the default surface and no baseline gets the surface's baseline back, with its recovery.
    surface = ACTIVATION["instances"]["robodojo"]["tool_surface"]
    spec = robodojo_spec.select_policy(_spec("stack_bowls"), tool_surface=surface)
    assert "baseline" not in spec["policy"] and "provider_recovery" not in spec["policy"]
    policy = resolve_run_spec(spec, root=ROOT)["policy"]
    assert policy["baseline"]["id"] == TOOLKIT_BASELINES[surface] == default_baseline_id("robodojo")
    assert policy["provider_recovery"] == load_native_baseline(ROOT, TOOLKIT_BASELINES[surface])["policy"][
        "provider_recovery"] == "first_token_deadline_v1"
    # A policy that reaches the Provider phase without one still takes the maintained selector, not legacy retries.
    unnamed = {}
    provider_defaults(unnamed, spec["runtime"])
    assert unnamed == {"provider_recovery": "first_token_deadline_v1"}


@needs_assets
def test_a_scheduled_session_discloses_its_layouts_and_must_cover_every_round():
    spec = _spec("stack_bowls", rounds=3)
    assert spec["config"] == "stack_bowls-l23.29.51-s0"
    plan = resolve_run_spec(spec, root=ROOT)
    compatibility = plan["policy"]["task_contract"]["layout_compatibility"]
    assert [entry["layout_id"] for entry in compatibility["scheduled_layouts"]] == [23, 29, 51]
    assert plan["session"]["rounds"] == 3
    assert "layout_compatibility" not in resolve_run_spec(_spec("stack_bowls"), root=ROOT)["policy"]["task_contract"]
    spec["session"]["rounds"] = 2
    with pytest.raises(RunSpecError, match="per session round"):
        resolve_run_spec(spec, root=ROOT)


@needs_assets
def test_a_cpu_physics_cell_is_admitted_only_by_its_own_motion_record(monkeypatch):
    """CPU-physics lane admission: cpu joined cuda:0 with its own 24-case motion record (definitions.py)."""
    from ora.environment import definitions

    spec = _spec("press_by_number", gpu="4", resource_profile="robodojo-gpu4-d1")
    assert spec["runtime"]["physics_device"] == "cpu"
    with monkeypatch.context() as patch:  # without that record the table names cuda:0 only
        patch.setitem(definitions.SPARSE_MOTION_QUALIFIED_PHYSICS_DEVICES, ("motion_action_v3", "robodojo"),
                      ("cuda:0",))
        with pytest.raises(RunSpecError, match="rule=sparse_motion_physics_unqualified"):
            resolve_run_spec(spec, root=ROOT)
    assert resolve_run_spec(spec, root=ROOT)["runtime"]["physics_device"] == "cpu"
    with pytest.raises(RoboDojoBenchmarkError, match="no reviewed layout pool"):
        _spec("press_by_number", gpu="4", resource_profile="robodojo-gpu4-d1", rounds=2)


def test_qualification_names_keep_cuda_unchanged_and_mark_cpu_physics():
    from ora.orchestrator.backends.robodojo.runtime_config import qualification_device_label

    assert qualification_device_label("4", "cuda:0") == "cuda4"
    assert qualification_device_label("4", "cpu") == "cpu-gpu4"
    with pytest.raises(ValueError, match="physics device"):
        qualification_device_label("4", "cuda:1")


# --- qualification, services and the operator entry point -------------------------------------------

def test_qualification_bindings_come_from_workspace_records(tmp_path, monkeypatch):
    from ora.orchestrator import paths

    monkeypatch.setattr(robodojo_spec, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", tmp_path / "outputs")
    card = {"path": "outputs/robodojo-compat/qualification/card/card-qualification.json", "sha256": "1" * 64}
    task = tmp_path / "outputs/robodojo-compat/qualification/task/task-qualification.json"
    task.parent.mkdir(parents=True)
    task.write_text(json.dumps({"passed": True, "card_qualification": card}))
    layered = robodojo_spec.layered_qualification(task)
    assert layered["card"] == card
    assert layered["task"]["path"] == "outputs/robodojo-compat/qualification/task/task-qualification.json"
    assert len(layered["task"]["sha256"]) == 64
    task.write_text(json.dumps({"passed": False, "card_qualification": card}))
    with pytest.raises(ValueError, match="not a passing"):
        robodojo_spec.layered_qualification(task)
    outside = tmp_path.parent / f"{tmp_path.name}-elsewhere.json"
    outside.write_text(json.dumps({"passed": True, "card_qualification": card}))
    with pytest.raises(ValueError, match="shared workspace"):
        robodojo_spec.layered_qualification(outside)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(QUALIFICATION))
    assert robodojo_spec.qualification_reference(gate) == QUALIFICATION
    gate.write_text(json.dumps({"card": card, "task": card}))
    assert set(robodojo_spec.qualification_reference(gate)) == {"card", "task"}
    gate.write_text(json.dumps({"path": "x"}))
    with pytest.raises(ValueError, match="no RoboDojo qualification binding"):
        robodojo_spec.qualification_reference(gate)


def _readiness_report(tmp_path, monkeypatch):
    from ora.agent.model_services import operator as services

    monkeypatch.setenv("ORA_SAM_PORT", "18114")
    monkeypatch.setenv("ORA_GRASP_PORT", "18115")
    health = {**{key: "fixture" for key in services.FIELDS},
              "protocol_version": services.GROUNDING_GRASP_PROTOCOL_VERSION}
    report = {"schema": services.REPORT_SCHEMA, "created_unix_s": time.time(), "passed": True,
              "protocol_version": services.GROUNDING_GRASP_PROTOCOL_VERSION,
              "adapter_sha256": services.grounding_grasp_adapter_sha256(),
              "checks": {k: True for k in ("single_point_wire", "batch_point_wire", "text_wire", "grasp_warmup")},
              "services": {name: {"endpoint": endpoint, "health": health}
                           for name, endpoint in services.endpoints().items()}}
    path = tmp_path / (services.digest(report) + ".json")
    path.write_text(json.dumps(report))
    return path


@needs_assets
def test_a_readiness_report_binds_the_default_surface_and_its_endpoints(tmp_path, monkeypatch):
    report = _readiness_report(tmp_path, monkeypatch)
    for name, variable in robodojo_spec.SERVICE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    spec = robodojo_spec.bind_services(_spec("stack_bowls"), report)
    catalog = load_native_baseline(ROOT, default_baseline_id("robodojo"))
    surface = catalog["instances"]["robodojo"]["tool_surface"]
    services = spec["prerequisite"]["services"]
    assert {child["tool_surface"] for child in services.values()} == {surface}
    assert spec["prerequisite"]["capability_scope"] == catalog["required_service_capabilities"]
    assert resolve_run_spec(spec, root=ROOT)["prerequisite"] == spec["prerequisite"]
    robodojo_spec.bind_service_environment(spec)
    assert os.environ["ORA_SEGMENTATION_URL"] == "http://127.0.0.1:18114"
    assert os.environ["ORA_GRASP_URL"] == "http://127.0.0.1:18115"
    # Set directly: the autouse fixture restores it (monkeypatch would restore the bound value).
    os.environ["ORA_GRASP_URL"] = "http://127.0.0.1:9999"
    with pytest.raises(RuntimeError, match="conflicts with the bound resident service"):
        robodojo_spec.bind_service_environment(spec)
    robodojo_spec.bind_service_environment({"prerequisite": None})


@needs_assets
@pytest.mark.parametrize("composer", ["command_line", "task_file"])
def test_the_services_bind_to_the_selected_surface_and_preflight_accepts_it(tmp_path, monkeypatch, capsys, composer):
    """A non-default ``--tool-surface`` with ``--service-report``: every service binding names it."""
    from ora.agent.model_services import clients as service_clients
    from ora.orchestrator.preflight import gates

    report = _readiness_report(tmp_path, monkeypatch)
    surface = "robodojo_placement_toolkit_v5"
    if composer == "command_line":
        assert robodojo_spec.main(["spec", "stack_bowls", "--experiment", EXPERIMENT, "--gpu", "0",
                                   "--tool-surface", surface, "--service-report", str(report)]) == 0
        spec = json.loads(capsys.readouterr().out)
    else:
        task = {"schema": "ora.task.v1", "simulator": "robodojo", "experiment": EXPERIMENT, "task": "stack_bowls",
                "seed": 0, "layout": 0, "model": "deepseek-flash", "policy": {"tool_surface": surface}}
        spec = robodojo_spec.compose_task(task, {
            "gpu": "0", "provider_catalog": str(ROOT / "ora/agent/providers/builtin-profiles.yaml"),
            "provider_profile": "deepseek_flash_openai_low_16k_v1", "service_report": str(report)})
    default = load_native_baseline(ROOT, default_baseline_id("robodojo"))["instances"]["robodojo"]["tool_surface"]
    assert surface != default
    assert {child["tool_surface"] for child in spec["prerequisite"]["services"].values()} == {surface}
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["policy"]["tool_surface"]["profile"] == surface

    class Ready:  # the live identity the binding recorded; no network
        def __init__(self, *, endpoint, expected_binding, timeout_s):
            self.binding = expected_binding

        def require_ready(self):
            return dict(self.binding)

        def close(self):
            pass

    for name in ("Sam3ApiSegmenter", "ContactGraspNetServiceClient"):
        monkeypatch.setattr(service_clients, name, Ready)
    check = gates._resident_service_prerequisite(plan)
    assert check["passed"] is True, check["detail"]
    assert set(check["detail"]["services"]) == {"segmentation", "grasp_proposals"}
    # The binding the composer made before this fix, on the default surface, is refused.
    for child in plan["prerequisite"]["services"].values():
        child["tool_surface"] = default
    refused = gates._resident_service_prerequisite(plan)
    assert refused["passed"] is False and "different tool surface" in refused["detail"]
    # The policy is selected first; a bound spec cannot change its surface afterwards.
    with pytest.raises(ValueError, match="before binding the resident services"):
        robodojo_spec.select_policy(spec, tool_surface=default)


@needs_assets
def test_compose_writes_the_spec_under_its_experiment_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(robodojo_spec, "experiment_workspace", lambda name: tmp_path / name)
    spec = _spec("stack_bowls")
    path = robodojo_spec.compose(spec)
    assert path == tmp_path / EXPERIMENT / "specs/stack_bowls-l0-s0.json"
    assert json.loads(path.read_text()) == spec


@needs_assets
def test_the_entry_point_lists_tasks_and_takes_an_optional_qualification_source(tmp_path, capsys):
    assert robodojo_spec.main(["tasks"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert list(listed) == list(REGISTRY)
    assert listed["press_by_number"]["physics_device"] == "cpu"
    reference = tmp_path / "gate.json"
    reference.write_text(json.dumps(QUALIFICATION))
    assert robodojo_spec.main(["spec", "stack_bowls", "--experiment", EXPERIMENT, "--gpu", "0",
                               "--qualification-reference", str(reference)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == _spec("stack_bowls")
    # The qualification source is optional (M7.6): without it the spec carries no infrastructure binding.
    assert robodojo_spec.main(["spec", "stack_bowls", "--experiment", EXPERIMENT, "--gpu", "0"]) == 0
    assert "infrastructure_qualification" not in json.loads(capsys.readouterr().out)["runtime"]


@needs_assets
def test_the_default_surface_reaches_a_strict_false_gateway_with_nullable_optionals():
    """The wire chain an explicit_false/max_completion_tokens profile takes, on the default surface."""
    from ora.agent.composition import resolve_tool_surface
    from ora.agent.composition.surface.null_optional_arguments import deliver_tool_definitions
    from ora.agent.session_core.gateway import openai_tool_definitions
    from ora.agent.providers.factory import build_provider

    plan = resolve_run_spec(_spec("stack_bowls"), root=ROOT)
    surface = resolve_tool_surface(plan["policy"]["tool_surface"]["profile"], toolkit_grants=plan["toolkit_grants"])
    tools = openai_tool_definitions(plan["policy"]["tool_allowlist"],
                                    definitions=deliver_tool_definitions(surface.definitions, surface.capabilities))
    provider = build_provider("openai_compatible", base_url="http://127.0.0.1:1/v1", api_key="offline",
                                     output_token_parameter="max_completion_tokens", tool_strictness="explicit_false")
    try:
        body = provider.prepare_request({"model": "offline", "messages": [{"role": "user", "content": "x"}],
                                         "tools": tools, "max_tokens": 16384})
    finally:
        provider.close()
    assert body["tools"] and all(tool["function"]["strict"] is False for tool in body["tools"])
    assert body["max_completion_tokens"] == 16384 and "max_tokens" not in body
    query = next(tool for tool in body["tools"] if tool["function"]["name"] == "geometry_query")
    parameters = query["function"]["parameters"]
    optional = set(parameters["properties"]) - set(parameters.get("required", []))
    assert optional and all("null" in parameters["properties"][name]["type"] for name in optional)


def test_no_bound_tool_client_endpoint_outlives_this_module():
    """Runs last in the module: the fixture endpoints above must not leak."""
    for _, variable in robodojo_spec.SERVICE_VARIABLES:
        assert os.environ.get(variable) not in {
            "http://127.0.0.1:18114", "http://127.0.0.1:18115", "http://127.0.0.1:9999"}


# --- policy selection: tool surface, baseline and context scheme ---------------------------------

@needs_assets
def test_a_cell_selects_its_tool_surface_and_context_scheme():
    spec = robodojo_spec.select_policy(_spec("stack_bowls"), tool_surface="robodojo_placement_toolkit_v5",
                                       context_scheme="context_keep_written", record_toolkit_io=True,
                                       ground_truth_capture=True)
    # A file-backed surface's required grants come from its surface.json.
    assert spec["toolkit_grants"] == {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
    plan = resolve_run_spec(spec, root=ROOT)
    policy = plan["policy"]
    assert policy["baseline"]["id"] == "robodojo_toolkit_v6"
    assert policy["tool_surface"]["profile"] == "robodojo_placement_toolkit_v5"
    assert policy["context_scheme"]["id"] == "context_keep_written"
    assert (policy["context_composition"], policy["context_management"]["schema"]) == (
        "native_slots_v4", "ora.native-context-management.v10")
    assert plan["record_toolkit_io"] is True and plan["runtime"]["ground_truth_capture"] is True


@needs_assets
def test_naming_the_baselines_own_context_scheme_changes_nothing():
    """``robodojo_toolkit_v6`` names ``context_keep_written`` itself, so a RunSpec that names it too sees the same.

    The plan records the scheme; the model-visible items leave the record out (``PROVENANCE_KEYS``) and keep the
    fields it brought."""
    baseline = "robodojo_toolkit_v6"
    plain = robodojo_spec.select_policy(_spec("stack_bowls"), baseline=baseline)
    named = robodojo_spec.select_policy(_spec("stack_bowls"), baseline=baseline,
                                        context_scheme="context_keep_written")
    plans = [resolve_run_spec(spec, root=ROOT) for spec in (plain, named)]
    a, b = (model_visible_items(plan, root=ROOT) for plan in plans)
    assert all(plan["policy"]["context_scheme"]["id"] == "context_keep_written" for plan in plans)
    assert a == b and "context_scheme" not in a["settings"]["policy"]
    assert a["settings"]["policy"]["context_management"]["camera_images"] == "exchange_window_v1"


@needs_assets
def test_a_context_scheme_needs_a_baseline_and_a_registered_name():
    with pytest.raises((RunSpecError, ValueError), match="context_scheme must be one of"):
        resolve_run_spec(robodojo_spec.select_policy(_spec("stack_bowls"), baseline="robodojo_toolkit_v6",
                                                     context_scheme="context_unknown_v1"), root=ROOT)
    # Every RoboDojo surface brings its own baseline, so a scheme named beside the surface applies to that baseline;
    for surface, baseline in TOOLKIT_BASELINES.items():
        spec = robodojo_spec.select_policy(_spec("stack_bowls"), tool_surface=surface,
                                           context_scheme="context_keep_written")
        policy = resolve_run_spec(spec, root=ROOT)["policy"]
        assert (policy["baseline"]["id"], policy["context_scheme"]["id"]) == (baseline, "context_keep_written")
    # a surface no baseline claims (Synthetic's) has no context half for a scheme to replace.
    spec = robodojo_spec.select_policy(_spec("stack_bowls"), tool_surface="robot_runtime_v2",
                                       context_scheme="context_keep_written")
    with pytest.raises((RunSpecError, ValueError), match="select a baseline"):
        resolve_run_spec(spec, root=ROOT)


@needs_assets
def test_compose_takes_the_policy_selection_on_its_command_line(capsys):
    assert robodojo_spec.main(["spec", "stack_bowls", "--experiment", EXPERIMENT, "--gpu", "0",
                               "--tool-surface", "robodojo_placement_toolkit_v5",
                               "--context-scheme", "context_keep_written", "--record-toolkit-io"]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["policy"]["tool_surface"] == {"profile": "robodojo_placement_toolkit_v5"}
    assert spec["policy"]["context_scheme"] == "context_keep_written"
    assert spec["toolkit_grants"]["tools"] == ["check_task"] and spec["record_toolkit_io"] is True


@needs_assets
def test_a_task_file_run_composes_the_same_spec_as_the_command_line():
    """``ora task run``'s composer: the task's cell and policy, the binding's resources."""
    catalog = ROOT / "ora/agent/providers/builtin-profiles.yaml"
    task = {"schema": "ora.task.v1", "simulator": "robodojo", "experiment": EXPERIMENT, "task": "stack_bowls",
            "seed": 0, "layout": 0, "model": "deepseek-flash", "budgets": {"max_robot_decisions": 40},
            "policy": {"tool_surface": "robodojo_placement_toolkit_v5", "context_scheme": "context_keep_written"},
            "record_toolkit_io": True, "ground_truth_capture": True}
    binding = {"gpu": "0", "provider_catalog": str(catalog), "provider_profile": "deepseek_flash_openai_low_16k_v1"}
    spec = robodojo_spec.compose_task(task, binding)
    expected = robodojo_spec.select_policy(
        robodojo_spec.build_spec("stack_bowls", experiment=EXPERIMENT, gpu="0", provider_catalog=catalog,
                                 provider_profile="deepseek_flash_openai_low_16k_v1",
                                 budgets={"max_robot_decisions": 40}),
        tool_surface="robodojo_placement_toolkit_v5", context_scheme="context_keep_written",
        record_toolkit_io=True, ground_truth_capture=True)
    assert spec == expected and spec["budgets"]["max_robot_decisions"] == 40
    with pytest.raises(ValueError, match="seed 0"):
        robodojo_spec.compose_task({**task, "seed": 1}, binding)
