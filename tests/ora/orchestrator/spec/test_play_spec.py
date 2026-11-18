"""A free-play RunSpec (``ora play``): what it binds, what it withholds, what it refuses.

The plan carries ``episode_mode: "play"``, the current scene, the scenes reset
may load and the per-instruction budgets; a benchmark instance is composed
without its task contract, with the play episode-task layer and without
check_task and the lifecycle tools; anything that would make it a scored,
multi-round or native-core run is refused. The evaluation never accepts a
free-play result that claims a verdict; every scene comes from
``play.scenes`` (a real environment's stubbed here, in its PlayScene shape),
stays on the scene's task and record, and binds the plan runtime to the scene's
play binding.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

from ora.orchestrator.spec import defaults
from ora.orchestrator.evaluation.play import verify_play_session
from ora.app.play import PLAY_TASK, draw_scenes, play_spec
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from tests.support.orchestrator.run_specs import native_spec
from tests.support.orchestrator.play_spec import POLICY

ROOT = Path(__file__).resolve().parents[4]


SYNTHETIC_SCENE = {"environment": "synthetic", "task": "synthetic-world", "seed": 0,
                   "scene_id": "synthetic:synthetic-world:s0", "label": "Synthetic world",
                   "qualification": {"kind": "synthetic_world"}}


def _synthetic(**changes) -> dict:
    """play.scenes' Synthetic scene, with a second scene bound for reset (another seed of its world)."""
    spec, _ = play_spec("synthetic", policy=POLICY, host=None, count=1, rng=random.Random(1),
                        draw={"draw_seed": 1})
    spec["play"]["scenes"].append({**spec["play"]["scene"], "seed": 7})
    spec.update(changes)
    return spec


def test_a_synthetic_play_plan_binds_its_scenes_and_both_budget_layers():
    plan = resolve_run_spec(_synthetic(), root=ROOT)
    assert plan["experiment"] == "ora-play" and plan["output_workspace"] == "outputs/ora-play"
    assert plan["episode_mode"] == "play" and plan["policy"]["agent_core"] == "pi"
    assert plan["task_instruction"] == PLAY_TASK and plan["seed"] == 0
    scene = SYNTHETIC_SCENE
    assert plan["play"] == {
        "schema": "ora.play-session.v1", "scene": scene, "scenes": [scene, {**scene, "seed": 7}],
        "instruction_budgets": {"max_robot_decisions": 24, "max_model_calls": 80, "wall_time_s": 1200},
        # The default: images leave in batches where an instruction begins with 20 messages of a kind still
        # carrying theirs, all but the latest three observations and tool results.
        "context_images": {"observations": 3, "tool_results": 3, "prune_at": 20},
        "draw": {"draw_seed": 1, "pool_size": 1},
    }
    assert plan["policy_context"]["episode_prefix"][-1]["path"] == defaults.PLAY_EPISODE_TASK
    assert plan["budgets"]["max_model_calls"] == 1600
    assert "session" not in plan and plan["operator_control"] == {"mode": "file_inbox_v1"}


@pytest.mark.parametrize(("change", "message"), [
    (lambda spec: spec.update(experiment="ora-interactive", output_workspace="outputs/ora-interactive"),
     "experiment ora-play"),
    (lambda spec: spec.pop("operator_control"), "needs operator_control"),
    (lambda spec: spec.update(session={"cadence": "persistent_agent_k_rounds_v1", "rounds": 2}),
     "takes no session block"),
    (lambda spec: spec["play"]["scenes"].reverse(), r"play.scenes\[0\] must be play.scene"),
    (lambda spec: spec["play"].update(scene={**spec["play"]["scene"], "seed": 9}), "play.scene.seed must be"),
    (lambda spec: spec.pop("episode_mode"), 'episode_mode "play" go together'),
    (lambda spec: spec["policy"].update(agent_core="native"), "runs on the Pi agent core"),
    (lambda spec: spec["play"]["instruction_budgets"].update(max_model_calls=5000), "exceeds the session"),
    (lambda spec: spec["play"].update(context_images={"observations": 0, "tool_results": 3}),
     "context_images takes observations and tool_results"),
    (lambda spec: spec["play"].update(context_images={"observations": 3}), "context_images takes"),
    (lambda spec: spec["play"].update(context_images={"observations": 3, "tool_results": 3, "prune": 20}),
     "context_images takes"),
    (lambda spec: spec["play"].update(context_images={"observations": 5, "tool_results": 3, "prune_at": 4}),
     "prune_at is at least"),
    (lambda spec: spec["play"].update(goal="stack the bowls"), "takes schema, scene, scenes"),
    (lambda spec: spec["play"].update(scenes=[spec["play"]["scene"]] * 33), "1..32 scenes"),
    (lambda spec: spec["play"].update(code_mode="yes"), "play.code_mode is true or false"),
    (lambda spec: spec["play"].update(code_mode=True), "play.code_mode needs policy.code_mode"),
    (lambda spec: spec["policy"].update(code_mode=True), r"policy.code_mode in free play needs play.code_mode"),
    (lambda spec: spec["play"]["scenes"][1].update(task="other"), r"scenes\[1\].task must be the plan's task"),
    (lambda spec: spec["play"]["scenes"][1].pop("qualification"), r"scenes\[1\] needs environment, task"),
])
def test_what_would_make_free_play_a_scored_or_other_run_is_refused(change, message):
    spec = _synthetic()
    change(spec)
    with pytest.raises(RunSpecError, match=message):
        resolve_run_spec(spec, root=ROOT)


@pytest.mark.parametrize("environment", ["robotwin", "libero_pro", "robodojo"])
def test_a_benchmark_instance_plays_without_its_task_contract_checker_or_finish(environment):
    spec = native_spec("robotwin" if environment == "robodojo" else environment)
    if environment == "robotwin":
        spec["runtime"]["task"] = "adjust_bottle"  # a task with a reviewed contract
    if environment == "robodojo":
        from tests.support.orchestrator.run_specs import robodojo_native_spec
        spec = robodojo_native_spec()
    benchmark = defaults.expand_native_defaults(spec, root=ROOT)
    play = defaults.expand_native_defaults({**copy.deepcopy(spec), "play": {}}, root=ROOT)
    assert play["policy_context"]["episode_task"] == [defaults.PLAY_EPISODE_TASK]
    assert play["policy"]["task_contract"] == {
        "schema": "ora.task-contract-disclosure.v1", "backend": environment, "delivered": False,
        "reason": "free_play_session_withholds_the_task",
    }
    assert "check_task" in benchmark["policy"].get("tool_allowlist", ["check_task"])
    assert not {"check_task", "finish_episode"} & set(play["policy"]["tool_allowlist"])
    # Everything else of the default bundle is reused as it is, but a context scheme's prompt: free play runs Pi's
    # own context (its compaction), never the scheme's stages the prompt describes.
    def without_scheme(paths: list) -> list:
        return [path for path in paths if not str(path).startswith("ora/policy/context_schemes/")]
    for layer in ("harness_system", "project_instructions", "tool_manual"):
        assert play["policy_context"][layer] == without_scheme(benchmark["policy_context"][layer])
    if environment == "robodojo":
        assert "ora/policy/context_schemes/keep-written.md" in benchmark["policy_context"]["harness_system"]
    if environment == "robotwin":
        assert benchmark["policy"]["task_contract"]["delivered"] is True


def test_the_play_episode_task_names_no_tool_task_or_benchmark():
    text = (ROOT / defaults.PLAY_EPISODE_TASK).read_text(encoding="utf-8")
    assert text.count("{{TASK}}") == 1
    for word in ("check_task", "finish_episode", "submit_action", "RoboTwin", "LIBERO", "RoboDojo", "success"):
        assert word not in text


def _play_result(**changes) -> tuple[dict, dict]:
    plan = {"experiment": "ora-play", "episode_mode": "play", "operator_control": {"mode": "file_inbox_v1"},
            "play": {"scenes": [{"seed": 3}, {"seed": 7}]}}
    instruction = {"schema": "ora.play-instruction.v1", "index": 0, "instruction_id": "i1", "delivered": True,
                   "ended": "reported", "decisions": 1, "refused_proposals": 0, "model_calls": 2}
    result = {"official_success": None, "verdict": None, "behavioral_sample_valid": False,
              "termination": "operator_quit", "decisions": 1, "refused_proposals": 0, "model_calls": 2,
              "play": {"schema": "ora.play-session.v1", "instruction_count": 1, "delivered_instruction_count": 1,
                       "instructions": [instruction], "scene_count": 1,
                       "scenes": [{"scene_index": 0, "seed": 3}]}}
    result.update(changes)
    return plan, result


def _events() -> list[dict]:
    record = {"schema": "ora.play-instruction.v1", "index": 0, "instruction_id": "i1", "delivered": True,
              "ended": "reported", "decisions": 1, "refused_proposals": 0, "model_calls": 2}
    return [{"type": "reset", "record": {"seed": 3}},
            {"type": "play_instruction_started", "record": {"instruction_id": "i1"}},
            {"type": "operator_message", "payload": {"id": "i1", "type": "instruction"}},
            {"type": "play_instruction_finished", "record": record}]


def test_a_consistent_free_play_result_passes_and_a_claimed_verdict_does_not(tmp_path):
    plan, result = _play_result()
    assert verify_play_session(tmp_path, plan, result, _events()) == []
    for change, reason in (({"official_success": True}, "play_official_verdict_claimed"),
                           ({"behavioral_sample_valid": True}, "play_behavioral_sample_claimed"),
                           ({"decisions": 4}, "play_decisions_mismatch")):
        assert reason in verify_play_session(tmp_path, plan, _play_result(**change)[1], _events())
    undelivered = [event for event in _events() if event["type"] != "operator_message"]
    assert "play_instruction_delivery_mismatch" in verify_play_session(tmp_path, plan, result, undelivered)
    (tmp_path / "native").mkdir()
    (tmp_path / "native/official-verdict.json").write_text("{}")
    assert "play_official_verdict_persisted" in verify_play_session(tmp_path, plan, result, _events())
    assert verify_play_session(tmp_path, {}, result, _events()) == ["play_result_without_plan"]


def test_a_message_that_joined_a_running_instruction_is_accounted_to_it(tmp_path):
    """A message sent while an instruction ran is delivered into it (joined_messages), not a new instruction."""

    plan, result = _play_result()
    result["play"]["instructions"][0]["joined_messages"] = ["m1"]
    events = _events()
    events[-1]["record"]["joined_messages"] = ["m1"]
    events.insert(3, {"type": "operator_message", "payload": {"id": "m1", "type": "instruction"}})
    assert verify_play_session(tmp_path, plan, result, events) == []
    # The same message, undeclared: an instruction-type delivery that opened no instruction.
    undeclared = [event for event in _events()]
    undeclared.insert(3, {"type": "operator_message", "payload": {"id": "m1", "type": "instruction"}})
    assert "play_instruction_delivery_mismatch" in verify_play_session(tmp_path, plan, _play_result()[1], undeclared)
    # Declared as joined but never delivered.
    undelivered = [event for event in events if event.get("payload", {}).get("id") != "m1"]
    assert "play_instruction_delivery_mismatch" in verify_play_session(tmp_path, plan, result, undelivered)


def test_scenes_are_drawn_from_the_pool_and_a_reset_changes_scene_where_it_can():
    scenes = draw_scenes([5, 6, 7], 8, random.Random(0))
    assert set(scenes) <= {5, 6, 7} and all(a != b for a, b in zip(scenes, scenes[1:]))
    assert draw_scenes([0], 3, random.Random(0)) == [0, 0, 0]
    with pytest.raises(RuntimeError, match="no qualified scene"):
        draw_scenes([], 1, random.Random(0))


def _scene(seed: int, task: str = "libero_goal-t03", sha256: str = "a"):
    """A ``play.scenes.PlayScene`` of a LIBERO-Goal task: its fields and its ``record()``."""
    import types

    record = {"environment": "libero", "task": task, "seed": seed, "scene_id": f"libero:{task}:s{seed}",
              "label": f"LIBERO kitchen table · initial state {seed}", "suite": "libero_goal", "task_id": 3,
              "qualification": {"kind": "libero_init_states", "sha256": sha256}}
    return types.SimpleNamespace(**{key: record[key] for key in ("environment", "task", "seed", "suite",
                                                                 "task_id", "qualification")},
                                 task_config=None, record=lambda: copy.deepcopy(record))


def test_ora_play_composes_a_synthetic_session_and_a_scene_from_play_scenes(monkeypatch, tmp_path, workspace):
    import sys
    import types

    import ora.orchestrator.play
    from ora.app import play as ora_play
    from ora.app.main import build_parser

    assert build_parser().parse_args(["play", "--env", "synthetic"]).command == "play"
    for key, value in {"ORA_PROVIDER": "openai_compatible", "ORA_BASE_URL": "http://127.0.0.1:9/v1",
                       "ORA_MODEL": "fake", "ORA_API_KEY": "fake-key"}.items():
        monkeypatch.setenv(key, value)
    def parse(arguments):
        return build_parser().parse_args(["play", *arguments])

    spec, path = ora_play.compose_play(parse(["--env", "synthetic", "--provider", "env", "--draw-seed", "4",
                                              "--scenes", "3"]))
    assert json.loads(path.read_text()) == spec and spec["experiment"] == "ora-play"
    # Synthetic offers its one world (play.scenes); reset reloads it.
    assert spec["play"]["scene"] == SYNTHETIC_SCENE and spec["play"]["scenes"] == [SYNTHETIC_SCENE] * 3
    assert spec["play"]["draw"]["draw_seed"] == 4 and spec["task_instruction"] == PLAY_TASK

    # A real environment's scene and its reset candidates come from play.scenes (ora-play-scenes).
    drawn = _scene(7)
    calls = []
    stub = types.ModuleType("ora.orchestrator.play.scenes")
    stub.draw_play_scene = lambda environment, rng: calls.append(("draw", environment)) or drawn
    stub.build_play_spec = lambda environment, *, host, rng, scene, service_report: calls.append(
        ("build", environment, host, scene.seed, service_report)) or {
        "schema": "ora.run-spec.v1", "task": scene.task, "seed": scene.seed, "episode_mode": "play",
        "runtime": {"kind": "libero_pro"}, "budgets": {"max_sim_steps": 9000, "max_model_calls": 5},
        "play": {"schema": "ora.play-session.v1", "scene": scene.record()}}
    # Another task, or a scene another record admits, is not a candidate for /reset.
    stub.list_play_scenes = lambda environment, *, task=None, task_config=None: calls.append(
        ("list", task, task_config)) or [_scene(seed) for seed in range(10)] + [_scene(9, task="libero_goal-t04"),
                                                                              _scene(42, sha256="b")]
    stub.write_play_spec = lambda spec, scene: calls.append(("write", scene.seed)) or tmp_path / "play.json"
    monkeypatch.setitem(sys.modules, "ora.orchestrator.play.scenes", stub)
    monkeypatch.setattr(ora.orchestrator.play, "scenes", stub, raising=False)
    with pytest.raises(ValueError, match="--provider env is for Synthetic"):
        ora_play.compose_play(parse(["--env", "libero", "--provider", "env"]))
    assert calls == []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    spec, path = ora_play.compose_play(parse(["--env", "libero", "--provider", "deepseek_flash_openai_high_v1",
                                              "--host", "h1", "--service-report", "ready.json", "--draw-seed", "2"]))
    assert calls == [("draw", "libero"), ("build", "libero", "h1", 7, Path("ready.json")),
                     ("list", "libero_goal-t03", None), ("write", 7)]
    assert path == tmp_path / "play.json"  # written for a launch too (a RoboTwin manifest is imported)
    assert spec["task_instruction"] == PLAY_TASK and spec["policy"]["agent_core"] == "pi"
    assert spec["play"]["scene"] == drawn.record() and spec["play"]["scenes"][0] == drawn.record()
    assert len(spec["play"]["scenes"]) == 8 and {item["seed"] for item in spec["play"]["scenes"]} <= set(range(10))
    assert spec["budgets"]["max_sim_steps"] == 9000 and spec["budgets"]["max_model_calls"] == 1600
    with pytest.raises(ValueError, match="1 and 32"):
        ora_play.compose_play(parse(["--env", "libero", "--scenes", "40"]))


def test_the_plan_runtime_is_the_scenes_play_binding(monkeypatch):
    import types

    from ora.orchestrator.spec import play as spec_play

    # A scene is play.scenes' record as it is: four keys required, the rest kept.
    spec = _synthetic()
    spec["play"]["scenes"][1].update(note={"lit": "dim"})
    assert resolve_run_spec(spec, root=ROOT)["play"]["scenes"][1]["note"] == {"lit": "dim"}
    spec["play"]["scenes"][1].update(task_id=-1)
    with pytest.raises(RunSpecError, match=r"scenes\[1\].task_id must be"):
        resolve_run_spec(spec, root=ROOT)
    play = {"scene": _scene(7).record()}
    benchmark = {"kind": "libero_pro", "environment_driver": "direct_libero_v1"}
    calls = []
    stub = types.ModuleType("ora.orchestrator.play.binding")
    stub.bind_play_runtime = lambda runtime, *, scene, scenes=None: calls.append(("bind", scene)) or {
        **runtime, **({"environment_driver": "direct_libero_play_v1"} if scene else {})}
    stub.verify_play_qualification = lambda play, qualification: calls.append(("verify", qualification))
    monkeypatch.setattr(spec_play, "binding", stub)
    assert spec_play.bind_play_runtime(benchmark, play)["environment_driver"] == "direct_libero_play_v1"
    assert spec_play.bind_play_runtime(benchmark, None) == benchmark  # the benchmark guard is play.binding's
    spec_play.verify_play_qualification(None, {"required": False})
    spec_play.verify_play_qualification(play, {"required": True})
    assert calls == [("bind", play["scene"]), ("bind", None), ("verify", {"required": True})]


def test_a_libero_robustness_case_keeps_its_own_initial_state_and_robodojo_its_layout():
    from ora.orchestrator.spec.play import _require_backend_scenes

    _require_backend_scenes({"kind": "libero_pro", "robustness_case": "case"}, [{"seed": 4}, {"seed": 4}])
    with pytest.raises(RunSpecError, match="robustness case"):
        _require_backend_scenes({"kind": "libero_pro", "robustness_case": "case"}, [{"seed": 4}, {"seed": 5}])
    with pytest.raises(RunSpecError, match="initial-state index below 50"):
        _require_backend_scenes({"kind": "libero_pro"}, [{"seed": 50}])
    # A RoboDojo session's scenes are seeds of one scene of ORA's own scene sets (relaunched, never restarted).
    with pytest.raises(RunSpecError, match="scene sets"):
        _require_backend_scenes({"kind": "robodojo", "eval_seed": 0}, [{"seed": 0}, {"seed": 1}])


def test_the_plans_image_window_reaches_the_pi_runtime(tmp_path, monkeypatch):
    from ora.agent.pi import factory

    monkeypatch.setattr(factory, "user_credential_store", lambda: tmp_path / "store" / "auth.json")
    plan = {"episode_mode": "play", "play": {"context_images": {"observations": 4, "tool_results": 2}}}
    assert factory._play_init(plan, tmp_path / "run")["context_images"] == {"observations": 4, "tool_results": 2}
    assert factory._play_init({"episode_mode": "benchmark"}, tmp_path / "run") is None


def test_a_spec_that_names_the_window_without_prune_at_keeps_the_sliding_window():
    """A RunSpec written before prune_at (and a sealed plan) resolves as it did; the default prunes in batches."""
    from ora.orchestrator.spec.play import CONTEXT_IMAGES

    assert CONTEXT_IMAGES == {"observations": 3, "tool_results": 3, "prune_at": 20}
    spec = _synthetic()
    spec["play"]["context_images"] = {"observations": 3, "tool_results": 3}
    assert resolve_run_spec(spec, root=ROOT)["play"]["context_images"] == {"observations": 3, "tool_results": 3}
    spec["play"]["context_images"] = {"observations": 3, "tool_results": 3, "prune_at": 50}
    assert resolve_run_spec(spec, root=ROOT)["play"]["context_images"]["prune_at"] == 50


def test_a_request_priority_is_optional_validated_and_reaches_the_pi_runtime(tmp_path, monkeypatch):
    """play.request_priority (ora play --request-priority): absent, requests carry no priority field."""
    from ora.agent.pi import factory
    from ora.app import play as ora_play

    spec = _synthetic()
    assert "request_priority" not in resolve_run_spec(spec, root=ROOT)["play"]
    spec["play"]["request_priority"] = {"correction": -1, "default": 0}
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["play"]["request_priority"] == {"correction": -1, "default": 0}
    monkeypatch.setattr(factory, "user_credential_store", lambda: tmp_path / "store" / "auth.json")
    assert factory._play_init(plan, tmp_path / "run")["request_priority"] == {"correction": -1, "default": 0}
    assert "request_priority" not in factory._play_init({"episode_mode": "play", "play": {}}, tmp_path / "run")
    for bad in ({"correction": -1}, {"correction": 1, "default": 0, "warmup": 5}, {"correction": 1.5, "default": 0},
                {"correction": True, "default": 0}, {"correction": 5000, "default": 0}, [1, 0]):
        spec["play"]["request_priority"] = bad
        with pytest.raises(RunSpecError, match="play.request_priority takes correction and default"):
            resolve_run_spec(spec, root=ROOT)
    assert ora_play._request_priority('{"correction": 10, "default": 0}') == {"correction": 10, "default": 0}
    for text, message in (("{", "not valid JSON"), ("null", "names correction and default"),
                          ('{"correction": 1}', "takes correction and default")):
        with pytest.raises(ValueError, match=message):
            ora_play._request_priority(text)


def test_a_prepared_play_spec_launches_with_its_profiles_base_url(tmp_path, monkeypatch):
    """`ora play --spec`: the profile's base-URL variable defaults to the spec's endpoint, as with --provider."""
    from ora.app import play as ora_play
    from ora.app.main import build_parser

    spec = {"schema": "ora.run-spec.v1", "episode_mode": "play", "task": "t", "seed": 0,
            "policy": {"kind": "native_api", "base_url_env": "ORA_TEST_PLAY_BASE_URL",
                       "endpoint": "https://api.example.invalid"}}
    path = tmp_path / "play.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr("ora.orchestrator.spec.run_spec.load_run_spec", lambda value: json.loads(Path(value).read_text()))
    monkeypatch.delenv("ORA_TEST_PLAY_BASE_URL", raising=False)
    loaded, written = ora_play.compose_play(build_parser().parse_args(
        ["play", "--env", "robodojo", "--spec", str(path)]))
    assert (loaded, written) == (spec, path)
    assert __import__("os").environ["ORA_TEST_PLAY_BASE_URL"] == "https://api.example.invalid"
    monkeypatch.setenv("ORA_TEST_PLAY_BASE_URL", "http://127.0.0.1:9/v1")  # an operator's own value stays
    ora_play.compose_play(build_parser().parse_args(["play", "--env", "robodojo", "--spec", str(path)]))
    assert __import__("os").environ["ORA_TEST_PLAY_BASE_URL"] == "http://127.0.0.1:9/v1"


def test_the_pipelined_next_step_is_a_switch_that_reaches_the_pi_runtime(tmp_path, monkeypatch):
    """play.pipeline_predict (ora play --pipeline-predict): off by default; on, Pi's driver predicts the next step
    while a motion runs (ora-pi/src/play/pipeline.ts)."""
    from ora.agent.pi import factory
    from ora.app import play as ora_play
    from ora.app.main import build_parser

    spec = _synthetic()
    assert "pipeline_predict" not in resolve_run_spec(spec, root=ROOT)["play"]
    spec["play"]["pipeline_predict"] = True
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["play"]["pipeline_predict"] is True
    monkeypatch.setattr(factory, "user_credential_store", lambda: tmp_path / "store" / "auth.json")
    assert factory._play_init(plan, tmp_path / "run")["pipeline_predict"] is True
    assert "pipeline_predict" not in factory._play_init({"episode_mode": "play", "play": {}}, tmp_path / "run")
    spec["play"]["pipeline_predict"] = "yes"
    with pytest.raises(RunSpecError, match="play.pipeline_predict is true or false"):
        resolve_run_spec(spec, root=ROOT)
    for name, value in (("ORA_API_KEY", "fixture"), ("ORA_MODEL", "fixture"),
                        ("ORA_BASE_URL", "https://example.invalid/v1")):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("ORA_PROVIDER", raising=False)
    composed, _ = ora_play.compose_play(build_parser().parse_args(
        ["play", "--env", "synthetic", "--provider", "env", "--pipeline-predict"]))
    assert composed["play"]["pipeline_predict"] is True


def _robodojo_play(**play) -> dict:
    from tests.support.orchestrator.run_specs import robodojo_native_spec

    return defaults.expand_native_defaults({**robodojo_native_spec(), "play": dict(play)}, root=ROOT)


def test_robodojo_free_play_takes_its_baselines_code_mode_only_through_play_code_mode():
    """``ora play --code-mode``: the baseline's code mode, the code planner and its free-play layer; Pi's own tools
    (no policy.code_mode) otherwise, and a policy.code_mode named without play.code_mode is refused."""
    from ora.agent.composition.baseline import default_baseline_id, load_native_baseline

    baseline = load_native_baseline(ROOT, default_baseline_id("robodojo"))
    assert baseline["code_mode"], "RoboDojo's baseline defaults to code mode"
    assert "code_mode" not in _robodojo_play()["policy"]
    coded = _robodojo_play(code_mode=True)
    assert coded["policy"]["code_mode"] == baseline["code_mode"]
    harness = [str(path) for path in coded["policy_context"]["harness_system"]]
    assert "ora/policy/planners/native/code-mode-play-v1.md" not in harness, "composed at resolution, not here"
    assert not [path for path in harness if path.startswith("ora/policy/context_schemes/")]
    # The baseline's own context scheme, not the code-mode one an episode chooses (free play's context is Pi's).
    assert coded["policy"]["context_scheme"]["id"] == baseline.get("context_scheme")
    from tests.support.orchestrator.run_specs import robodojo_native_spec

    named = {**robodojo_native_spec(), "play": {}}
    named["policy"] = {**named["policy"], "code_mode": True}
    with pytest.raises(ValueError, match=r"takes code mode only through play.code_mode"):
        defaults.expand_native_defaults(named, root=ROOT)


def _layer_paths(plan: dict) -> dict[str, list[str]]:
    """Each policy-context group's layer paths."""
    return {name: [item["path"] for item in value] for name, value in plan["policy_context"].items()
            if isinstance(value, list) and all(isinstance(item, dict) and "path" in item for item in value)}


def test_the_free_play_code_layer_follows_the_code_planner_in_play_only(tmp_path, monkeypatch):
    from ora.orchestrator.spec.composition import CODE_MODE_PLAY

    spec = _synthetic()
    spec["policy"] = {**spec["policy"], "code_mode": True}
    spec["play"]["code_mode"] = True
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["play"]["code_mode"] is True and plan["policy"]["code_mode"]["prompt"] in CODE_MODE_PLAY
    # The free-play layer closes the group the code planner leads (harness_system, or the quickstart's codex_project).
    (group,) = [paths for paths in _layer_paths(plan).values()
                if paths and paths[0].startswith("ora/policy/planners/native/code-mode-coding-")]
    assert group[-1] == CODE_MODE_PLAY[plan["policy"]["code_mode"]["prompt"]]
    text = (ROOT / CODE_MODE_PLAY["v3"]).read_text(encoding="utf-8")
    for word in ("check_task", "finish_episode", "RoboDojo", "success"):
        assert word not in text
    # An episode in code mode never reads it.
    from tests.support.agent_core.episodes import synthetic_plan

    episode = synthetic_plan(actions=2, policy_overrides={"code_mode": True})
    assert not [paths for paths in _layer_paths(episode).values() if CODE_MODE_PLAY["v3"] in paths]
    from ora.agent.pi import factory

    monkeypatch.setattr(factory, "user_credential_store", lambda: tmp_path / "store" / "auth.json")
    assert factory._play_init(plan, tmp_path / "run")["code_mode"] is True
    assert "code_mode" not in factory._play_init(resolve_run_spec(_synthetic(), root=ROOT), tmp_path / "run")


@pytest.mark.parametrize(("argv", "message"), [
    (["play", "--env", "robodojo", "--code-mode", "--spec", "play.json"], "its play.code_mode already decides"),
])
def test_ora_play_code_mode_refuses_a_prepared_spec(argv, message):
    from ora.app import play as ora_play
    from ora.app.main import build_parser

    with pytest.raises(ValueError, match=message):
        ora_play.compose_play(build_parser().parse_args(argv))
