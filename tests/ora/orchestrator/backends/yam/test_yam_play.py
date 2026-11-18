"""``ora play --env yam``: the YAM station's cell as it stands, composed explicitly and bound to a rig."""

from __future__ import annotations

from pathlib import Path

import pytest

from ora.app import play as ora_play
from ora.app.main import build_parser
from ora.agent.code_mode.contract import script_reads
from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.orchestrator.backends.yam.play import (
    NO_PERCEPTION_BUNDLE, NO_PERCEPTION_SURFACE, PLAY, SURFACE, YAM_CODE_MODE, YAM_PLAY_REACH_QUERY,
    YAM_PLAY_SCRIPTED_MOVES,
    YAM_PROMPT_BUNDLE, with_rig,
)
from ora.orchestrator.play.scenes import PlaySceneError
from ora.orchestrator.spec.native_policy import _resolve_code_mode
from ora.orchestrator.spec.run_spec import resolve_run_spec
from ora.orchestrator.spec.values import RunSpecError
from tests.support.orchestrator.benchmark_specs import _readiness_report

ROOT = Path(__file__).resolve().parents[5]
PROVIDER = "deepseek_flash_openai_low_16k_v1"


def _parse(*arguments: str):
    return build_parser().parse_args(["play", "--env", "yam", *arguments])


def _compose(*arguments: str):
    """A session without the perception services, unless the arguments name a --service-report."""
    perception = ("--no-perception",) if "--service-report" not in arguments else ()
    return ora_play.compose_play(_parse("--provider", PROVIDER, "--draw-seed", "3", "--scenes", "2",
                                        *perception, *arguments))


def test_a_simulated_session_resolves_on_the_yam_surface_without_task_or_lifecycle(workspace):
    """--no-perception: yam_direct_v1, which hosts no services."""
    spec, path = _compose("--rig-mode", "sim")
    assert path.is_file() and spec["experiment"] == "ora-play" and spec["task"] == "operator_task"
    plan = resolve_run_spec(spec)

    runtime = plan["runtime"]
    assert (runtime["kind"], runtime["rig_mode"], runtime["python_env"]) == ("yam", "sim", "YAM_PYTHON")
    assert runtime["task_instruction"] == ora_play.PLAY_TASK == plan["task_instruction"]
    assert "operator_present" not in runtime and "required_host" not in runtime
    assert runtime["workspace_bounds_m"] == {"x": [0.0, 0.65], "y": [-0.65, 0.65], "z": [-0.1, 0.6]}
    # The station's surface, its lifecycle tool withheld as in every free-play session.
    surface = plan["policy_composition"]["tool_surface"]
    assert surface["profile_id"] == NO_PERCEPTION_SURFACE == "yam_direct_v1"
    assert "prerequisite" not in spec
    advertised = [item["name"] for item in surface["definitions"]]
    assert "finish_episode" not in advertised and "check_task" not in advertised
    assert {"observe", "move_pose", "set_gripper", "recover_pose", "wait_ticks"} <= set(advertised)
    assert plan["policy"]["task_contract"]["delivered"] is False
    layers = {channel: [item["path"] for item in value] for channel, value in plan["policy_context"].items()
              if isinstance(value, list)}
    assert layers["project_instructions"][-2:] == ["ora/policy/environments/yam/boundary-v1.md",
                                                  "ora/policy/embodiments/yam-station/manual-public-geometry-v1.md"]
    # The free-play task only: scripted moves and the reach query are off by default.
    assert layers["episode_task"] == ["ora/policy/prompt_bundles/templates/episode-task-play-v2.md"]
    assert "reach_query" not in runtime
    # /reset starts a new round of the same cell.
    assert [(scene["task"], scene["seed"]) for scene in plan["play"]["scenes"]] == [("operator_task", 0)] * 2
    assert plan["play"]["scene"]["qualification"]["kind"] == "yam_operator_cell"


def test_free_play_on_the_station_is_a_real_time_session_unless_asked_otherwise(workspace):
    """No execution capture and no preview pictures on the control path; --latency-profile overrides it."""
    plan = resolve_run_spec(_compose("--rig-mode", "sim")[0])
    assert plan["latency_profile"]["profile"] == "realtime"
    assert plan["latency_profile"]["switches"] == {"execution_capture": False, "reach_reference": False,
                                                   "preview_images": False}
    assert plan["runtime"]["execution_capture"]["enabled"] is False
    benchmark = resolve_run_spec(_compose("--rig-mode", "sim", "--latency-profile", "sim_benchmark")[0])
    assert benchmark["latency_profile"]["profile"] == "sim_benchmark"
    assert benchmark["runtime"]["execution_capture"]["enabled"] is True


def test_scripted_moves_and_the_reach_query_are_switches_each_adding_its_layer(workspace, monkeypatch):
    report = _readiness_report(workspace / "reports", monkeypatch)
    play_task = "ora/policy/prompt_bundles/templates/episode-task-play-v2.md"

    def layers_and_runtime(*switches: str):
        plan = resolve_run_spec(_compose("--rig-mode", "sim", "--service-report", str(report), *switches)[0])
        return [item["path"] for item in plan["policy_context"]["episode_task"]], plan["runtime"]

    layers, runtime = layers_and_runtime("--scripted-moves")
    assert layers == [play_task, YAM_PLAY_SCRIPTED_MOVES] and "reach_query" not in runtime
    layers, runtime = layers_and_runtime("--reach-query")
    assert layers == [play_task, YAM_PLAY_REACH_QUERY] and runtime["reach_query"] is True
    layers, runtime = layers_and_runtime("--scripted-moves", "--reach-query")
    assert layers == [play_task, YAM_PLAY_SCRIPTED_MOVES, YAM_PLAY_REACH_QUERY] and runtime["reach_query"] is True
    # The reach query needs no perception; scripted moves perceive with SAM3 and Contact-GraspNet.
    assert resolve_run_spec(_compose("--rig-mode", "sim", "--reach-query")[0])["runtime"]["reach_query"] is True
    with pytest.raises(ValueError, match="--scripted-moves perceives with the yam_perception_v1 tools"):
        _compose("--rig-mode", "sim", "--scripted-moves")
    with pytest.raises(ValueError, match="apply to --env yam only"):
        ora_play.compose_play(build_parser().parse_args(["play", "--env", "synthetic", "--reach-query"]))
    bad = _compose("--rig-mode", "sim", "--reach-query")[0]
    bad["runtime"]["reach_query"] = "yes"
    with pytest.raises(RunSpecError, match="reach_query is a boolean"):
        resolve_run_spec(bad)


def test_the_shared_motion_timing_is_a_switch_that_adds_no_layer_and_a_preflight_check(workspace, monkeypatch):
    """``--motion-timing`` (off by default): the timing worker times the arm paths; the model is told nothing."""
    from ora.environment.benchmarks.yam import MOTION_TIMING
    from ora.orchestrator.backends.yam import preflight

    default = resolve_run_spec(_compose("--rig-mode", "sim")[0])
    assert "motion_timing" not in default["runtime"]
    timed = resolve_run_spec(_compose("--rig-mode", "sim", "--motion-timing")[0])
    assert timed["runtime"]["motion_timing"] == MOTION_TIMING
    assert timed["policy_context"]["episode_task"] == default["policy_context"]["episode_task"]
    assert timed["policy_composition"] == default["policy_composition"]
    with pytest.raises(ValueError, match="apply to --env yam only"):
        ora_play.compose_play(build_parser().parse_args(["play", "--env", "synthetic", "--motion-timing"]))
    bad = _compose("--rig-mode", "sim", "--motion-timing")[0]
    bad["runtime"]["motion_timing"] = {"implementation": "toppra_ruckig", "python_env": "OTHER_PYTHON"}
    with pytest.raises(RunSpecError, match="motion_timing is absent or exactly"):
        resolve_run_spec(bad)
    # Preflight starts the worker the runtime names; the session binds what that check started.
    monkeypatch.delenv("YAM_PYTHON", raising=False)
    monkeypatch.delenv("ORA_TIMING_PYTHON", raising=False)
    names = lambda plan: [item["check"] for item in preflight.yam_runtime(plan)]
    assert names(default) == ["environment_python"]
    checks = preflight.yam_runtime(timed)
    assert [item["check"] for item in checks] == ["environment_python", "motion_timing_worker"]
    assert checks[-1] == {"check": "motion_timing_worker", "passed": False, "detail": "ORA_TIMING_PYTHON is missing"}


def test_code_mode_on_the_station_names_the_extended_contract_and_its_code_layers(workspace, monkeypatch):
    """ora play --env yam --code-mode: the RoboDojo toolkit baseline's contract (extended library, prompt v3); the
    model works in ORA's coding tools and calls the robot as a library; a dry run is its one read."""
    report = _readiness_report(workspace / "reports", monkeypatch)
    for spec in (_compose("--rig-mode", "sim", "--code-mode")[0],
                 _compose("--rig-mode", "sim", "--service-report", str(report), "--code-mode")[0]):
        assert spec["policy"]["code_mode"] == YAM_CODE_MODE and spec["play"]["code_mode"] is True
        plan = resolve_run_spec(spec)
        assert (plan["policy"]["code_mode"]["library"], plan["policy"]["code_mode"]["prompt"]) == ("extended", "v3")
        assert plan["policy"]["code_mode"]["script_reads"] == ["dry_run"]
        advertised = {item["name"] for item in plan["policy_composition"]["tool_surface"]["definitions"]}
        assert {"read", "write", "edit", "bash"} <= advertised and "move_pose" not in advertised
        harness = [item["path"].rsplit("/", 1)[-1] for item in plan["policy_context"]["harness_system"]]
        assert harness[0] == "code-mode-coding-v3.md" and "code-mode-play-v1.md" in harness
    # Without --code-mode the station is unchanged.
    assert "code_mode" not in resolve_run_spec(_compose("--rig-mode", "sim")[0])["policy"]
    # The pipelined next step predicts Pi's next tool call; ORA's code mode has none.
    with pytest.raises(RunSpecError, match="ORA's code mode has none to predict"):
        resolve_run_spec(_compose("--rig-mode", "sim", "--code-mode", "--pipeline-predict")[0])


def test_the_shared_planner_is_a_switch_that_adds_no_layer_and_a_preflight_check(workspace, monkeypatch):
    """``--motion-planner`` (off by default): the shared cuRobo V2 planner plans the arm paths; the model is told
    nothing, and preflight starts the worker the session then binds."""
    from ora.environment.benchmarks.yam import MOTION_PLANNER, MOTION_PLANNER_PREFLIGHT_CHECK as PREFLIGHT_CHECK
    from ora.orchestrator.backends.yam import preflight

    default = resolve_run_spec(_compose("--rig-mode", "sim")[0])
    assert "motion_planner" not in default["runtime"]
    planned = resolve_run_spec(_compose("--rig-mode", "sim", "--motion-planner")[0])
    assert planned["runtime"]["motion_planner"] == MOTION_PLANNER
    assert planned["policy_context"]["episode_task"] == default["policy_context"]["episode_task"]
    assert planned["policy_composition"] == default["policy_composition"]
    with pytest.raises(ValueError, match="apply to --env yam only"):
        ora_play.compose_play(build_parser().parse_args(["play", "--env", "synthetic", "--motion-planner"]))
    bad = _compose("--rig-mode", "sim", "--motion-planner")[0]
    bad["runtime"]["motion_planner"] = {**MOTION_PLANNER, "world_model": "none"}
    with pytest.raises(RunSpecError, match="motion_planner is absent or exactly"):
        resolve_run_spec(bad)
    monkeypatch.delenv("CUROBO_V2_PYTHON", raising=False)
    monkeypatch.delenv("CUROBO_V2_ROOT", raising=False)
    check = preflight.motion_planner_check(MOTION_PLANNER, "/i2rt/station.urdf")
    assert check == {"check": PREFLIGHT_CHECK, "passed": False,
                     "detail": {"missing": ["CUROBO_V2_PYTHON", "CUROBO_V2_ROOT"]}}


def test_the_station_resolves_on_a_toolkit_surface_whose_environment_plans_its_moves(workspace, monkeypatch):
    report = _readiness_report(workspace / "reports", monkeypatch)
    for spec in (_compose("--rig-mode", "sim")[0], _compose("--rig-mode", "sim", "--service-report", str(report))[0]):
        capabilities = resolve_run_spec(spec)["policy_composition"]["tool_surface"]["capabilities"]
        assert "toolkit_only_v1" in capabilities and "toolkit_synchronous_execution_v1" not in capabilities


def test_the_default_session_binds_sam3_and_contact_graspnet_from_the_readiness_report(workspace, monkeypatch):
    report = _readiness_report(workspace / "reports", monkeypatch)
    spec, _ = _compose("--rig-mode", "sim", "--service-report", str(report))
    plan = resolve_run_spec(spec)
    surface = plan["policy_composition"]["tool_surface"]
    assert surface["profile_id"] == SURFACE == "yam_perception_v1"
    advertised = [item["name"] for item in surface["definitions"]]
    assert {"vision_segment", "vision_ground", "grasp_plan"} <= set(advertised)
    assert "finish_episode" not in advertised and "grasp_execute" not in advertised
    assert set(spec["prerequisite"]["capability_scope"]) == {
        "resident_sam3_point_segmentation_v1", "resident_sam3_text_grounding_v1", "resident_contact_graspnet_v1"}
    # The surface delivers public robot facts, so the runtime binds the descriptor the worker verifies.
    assert plan["runtime"]["public_geometry_descriptor"]["path"] == (
        "ora/policy/embodiments/yam-station/descriptor-public-geometry-v1.json")


def test_a_session_names_its_services_or_declines_them(workspace, monkeypatch):
    with pytest.raises(ValueError, match="pass --service-report REPORT .*or --no-perception"):
        ora_play.compose_play(_parse("--provider", PROVIDER, "--rig-mode", "sim"))
    report = _readiness_report(workspace / "reports", monkeypatch)
    with pytest.raises(ValueError, match="--no-perception runs yam_direct_v1, which hosts no services"):
        ora_play.compose_play(_parse("--provider", PROVIDER, "--rig-mode", "sim", "--no-perception",
                                     "--service-report", str(report)))
    with pytest.raises(ValueError, match="apply to --env yam only"):
        ora_play.compose_play(build_parser().parse_args(["play", "--env", "synthetic", "--no-perception"]))


def test_code_mode_on_the_station_surface_tells_of_no_query():
    """Its worker's read-only bridge answers a dry run's plan and nothing else (``YamSession.align_plan``)."""
    policy = {"agent_core": "pi", "code_mode": {"library": "extended"}}
    _resolve_code_mode(policy, resolve_tool_surface(SURFACE), {"kind": "yam"})
    assert script_reads(policy["code_mode"]) == {"dry_run"}


def test_the_rig_choices_bind_the_runtime_and_refuse_the_unsafe_ones(workspace, tmp_path):
    spec, _ = _compose("--rig-mode", "sim")
    cell = tmp_path / "cell.json"
    cell.write_text("{}\n")

    read = with_rig(spec, rig_mode="read", operator_present=False, cell=cell, host="yam-cell")
    assert read["runtime"]["rig_mode"] == "read" and read["runtime"]["required_host"] == "yam-cell"
    assert read["runtime"]["cell_config"] == str(cell.resolve())
    live = with_rig(spec, rig_mode="on", operator_present=True, cell=None, host=None)
    assert live["runtime"]["operator_present"] is True
    assert live["runtime"]["required_host"] == "test-host"  # ORA_EXECUTION_HOST
    back = with_rig(live, rig_mode="sim", operator_present=False, cell=None, host=None)
    assert not {"operator_present", "required_host", "cell_config"} & set(back["runtime"])

    with pytest.raises(ValueError, match="--operator-present only with a person"):
        with_rig(spec, rig_mode="on", operator_present=False, cell=None, host=None)
    with pytest.raises(ValueError, match="applies to --rig-mode on only"):
        with_rig(spec, rig_mode="read", operator_present=True, cell=None, host=None)
    with pytest.raises(ValueError, match="names no file"):
        with_rig(spec, rig_mode="sim", operator_present=False, cell=tmp_path / "missing.json", host=None)


def test_auto_approve_binds_motion_approval_into_the_runtime_of_the_real_arms_only(workspace, tmp_path):
    spec, _ = _compose("--rig-mode", "sim")
    live = with_rig(spec, rig_mode="on", operator_present=True, cell=None, host=None, auto_approve=True)
    assert live["runtime"]["auto_approve_motions"] is True
    assert "auto_approve" not in live["runtime"], "RealRig's auto_approve would skip enabling and releasing too"
    plan = resolve_run_spec(live)
    assert plan["runtime"]["auto_approve_motions"] is True
    asked = with_rig(live, rig_mode="on", operator_present=True, cell=None, host=None)
    assert "auto_approve_motions" not in asked["runtime"]
    with pytest.raises(ValueError, match="--auto-approve applies to --rig-mode on only"):
        with_rig(spec, rig_mode="read", operator_present=False, cell=None, host=None, auto_approve=True)
    with pytest.raises(RunSpecError, match="auto_approve_motions is a boolean and applies to rig_mode 'on' only"):
        resolve_run_spec({**spec, "runtime": {**spec["runtime"], "auto_approve_motions": True}})

    # The command line: the flag reaches the runtime, and it belongs to a new YAM session.
    composed, _ = _compose("--rig-mode", "on", "--operator-present", "--auto-approve")
    assert composed["runtime"]["auto_approve_motions"] is True
    assert "auto_approve_motions" not in _compose("--rig-mode", "on", "--operator-present")[0]["runtime"]
    with pytest.raises(ValueError, match="--auto-approve applies to --rig-mode on only"):
        _compose("--rig-mode", "sim", "--auto-approve")
    with pytest.raises(ValueError, match="apply to --env yam only"):
        ora_play.compose_play(build_parser().parse_args(["play", "--env", "synthetic", "--auto-approve"]))
    with pytest.raises(ValueError, match="already binds the rig"):
        ora_play.compose_play(_parse("--spec", str(tmp_path / "play.json"), "--auto-approve"))
    help_text = " ".join(build_parser()._subparsers._group_actions[0].choices["play"].format_help().split())
    assert "motions run without asking; the operator still approves enabling the arms once" in help_text


def test_the_rig_options_belong_to_the_yam_station(workspace):
    with pytest.raises(ValueError, match="needs --rig-mode"):
        _compose()
    with pytest.raises(ValueError, match="apply to --env yam only"):
        ora_play.compose_play(build_parser().parse_args(["play", "--env", "synthetic", "--rig-mode", "sim"]))
    # Only the simulated cell launches without --confirm-launch.
    assert not ora_play._needs_launch_confirmation("yam", {"runtime": {"rig_mode": "sim"}})
    assert ora_play._needs_launch_confirmation("yam", {"runtime": {"rig_mode": "read"}})
    assert ora_play._needs_launch_confirmation("robotwin", {"runtime": {}})


def test_the_station_offers_one_cell_and_refuses_a_report_it_cannot_bind(tmp_path):
    scenes = PLAY.list_scenes(task=None, task_config=None, scene_set=None, root=ROOT)
    assert [(scene.task, scene.seed) for scene in scenes] == [("operator_task", 0)]
    assert PLAY.list_scenes(task="sim_pick_place", task_config=None, scene_set=None, root=ROOT) == []
    with pytest.raises(PlaySceneError, match="does not bind the YAM station's services"):
        PLAY.build_spec({}, scenes[0], host=None, resource_profile=None, service_report=tmp_path / "ready.json",
                        root=ROOT)


@pytest.mark.parametrize("path,surface", [(YAM_PROMPT_BUNDLE, SURFACE), (NO_PERCEPTION_BUNDLE, NO_PERCEPTION_SURFACE)])
def test_the_bundle_audits_and_names_the_layers_the_session_delivers(path, surface):
    import yaml

    from ora.agent.composition.prompt.bundle import audit_prompt_bundle
    from ora.environment.benchmarks.yam import public_context

    bundle = yaml.safe_load((ROOT / path).read_text())
    assert bundle["layers"]["tool_surface"]["profile"] == surface
    assert bundle["bindings"]["environment_prompt_profile"]["public_environment_context"] == public_context()
    report = audit_prompt_bundle(bundle, project_root=ROOT)
    assert report["passed"], [check for check in report["checks"] if not check["passed"]]


def test_setup_runs_the_cells_setup_first_on_the_real_arms_only(workspace):
    composed, _ = _compose("--rig-mode", "on", "--operator-present", "--setup")
    assert composed["play"]["cell_setup"] is True
    assert resolve_run_spec(composed)["play"]["cell_setup"] is True
    plain = resolve_run_spec(_compose("--rig-mode", "on", "--operator-present")[0])
    assert "cell_setup" not in plain["play"], "a plan without --setup is unchanged"
    with pytest.raises(ValueError, match="--setup moves the arms"):
        _compose("--rig-mode", "read", "--setup")
    with pytest.raises(ValueError, match="apply to --env yam only"):
        ora_play.compose_play(build_parser().parse_args(["play", "--env", "synthetic", "--setup"]))
    sim, _ = _compose("--rig-mode", "sim")
    with pytest.raises(RunSpecError, match="YAM on rig_mode on only"):
        resolve_run_spec({**sim, "play": {**sim["play"], "cell_setup": True}})
    with pytest.raises(RunSpecError, match="cell_setup is true or false"):
        resolve_run_spec({**composed, "play": {**composed["play"], "cell_setup": "yes"}})


def test_smooth_commands_is_a_runtime_switch_off_by_default(workspace):
    assert "smooth_arm_commands" not in resolve_run_spec(_compose("--rig-mode", "sim")[0])["runtime"]
    spec = _compose("--rig-mode", "sim", "--smooth-commands")[0]
    assert resolve_run_spec(spec)["runtime"]["smooth_arm_commands"] is True
    spec["runtime"]["smooth_arm_commands"] = "yes"
    with pytest.raises(RunSpecError, match="smooth_arm_commands is a boolean"):
        resolve_run_spec(spec)
    with pytest.raises(ValueError, match="apply to --env yam only"):
        ora_play.compose_play(build_parser().parse_args(["play", "--env", "synthetic", "--smooth-commands"]))
