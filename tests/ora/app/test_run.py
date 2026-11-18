"""``ora run``: environment binding, services, qualification reuse and spec composition.

No simulator, service, GPU or Provider is started; producers and builders are faked
at their module boundary.
"""

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import pytest

from ora.app import run as ora_run, run_services as services
from ora.app.main import build_parser
from ora.orchestrator import progress
from ora.orchestrator.backends.libero_pro.run_cell import libero_cell
from ora.orchestrator.backends.robodojo import run_cell as robodojo_cell
from ora.orchestrator.backends.robotwin import run_cell as robotwin_cell

PROVIDER = {"DEEPSEEK_LOW_API_KEY": "secret-never-printed"}
GPU_KEYS = {key: f"/srv/{key.lower()}" for environment in ora_run.ENVIRONMENTS
            for key in ora_run.required_keys(environment)}


def test_environment_binds_env_file_host_marker_and_provider_defaults(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_LOW_API_KEY=\n" + "".join(f"{k}={v}\n" for k, v in GPU_KEYS.items())
                        + "ROBOTWIN_ROOT=/from-file\n")
    environ = {**PROVIDER, "ROBOTWIN_ROOT": "/from-shell"}
    ora_run.prepare_environment("robotwin", env_file=env_file, environ=environ)
    assert environ["ORA_EXECUTION_HOST"] == "single-gpu"
    assert environ["DEEPSEEK_BASE_URL"] == "https://api.deepseek.com"
    assert environ["DEEPSEEK_LOW_API_KEY"] == "secret-never-printed"  # an empty .env line never blanks it
    assert environ["ROBOTWIN_ROOT"] == "/from-shell"
    assert environ["CUROBO_PYTHON"] == GPU_KEYS["CUROBO_PYTHON"]


def test_environment_accepts_the_service_keys_under_their_pre_1_0_names(tmp_path, monkeypatch):
    from ora.agent.model_services import service_env

    monkeypatch.setattr(service_env, "_deprecation_warned", False)
    settings = ora_run.service_settings()
    legacy = {(service_env.legacy_service_key(*settings[key]) if key in settings
               else key): value for key, value in GPU_KEYS.items()}
    assert not set(legacy) & set(settings)
    environ = {**PROVIDER, **legacy}
    with pytest.warns(FutureWarning, match="rename it to ORA_"):
        ora_run.prepare_environment("libero", env_file=tmp_path / ".env", environ=environ)
    del environ[service_env.legacy_service_key("GRASP", "CONFIG")]
    with pytest.raises(RuntimeError, match="ORA_GRASP_CONFIG unset"):
        ora_run.prepare_environment("libero", env_file=tmp_path / ".env", environ=environ)


def test_environment_follows_the_selected_provider_profile(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("".join(f"{k}={v}\n" for k, v in GPU_KEYS.items()))
    environ = {"OPENROUTER_API_KEY": "secret-never-printed", "DEEPSEEK_API_KEY": "other"}
    ora_run.prepare_environment("libero", env_file=env_file, environ=environ,
                                provider_profile="muse_spark_open_openrouter_high_16k_v1")
    assert environ["OPENROUTER_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert "DEEPSEEK_LOW_API_KEY" not in environ  # the DeepSeek alias belongs to the default profile only
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        ora_run.prepare_environment("libero", env_file=env_file, environ=dict(GPU_KEYS),
                                    provider_profile="nemotron_35_super_vl_openrouter_thinking_16k_v1")


def test_compose_passes_the_selected_provider_profile_to_the_builder(monkeypatch, tmp_path):
    from ora.orchestrator.spec import benchmark as benchmark_spec

    built = {}
    monkeypatch.setattr(benchmark_spec, "build_libero_spec", lambda suite, task_id, **kwargs: built.update(kwargs) or {})
    monkeypatch.setattr(benchmark_spec, "compose", lambda spec, **kwargs: tmp_path / "spec.json")
    args = _args(env="libero", task="libero_goal/0")
    args.provider_profile = "muse_spark_open_openrouter_high_16k_v1"
    ora_run.compose(args, services=lambda: tmp_path / "report.json", confirm=lambda message: True,
                    stream=io.StringIO())
    assert built["provider_profile"] == "muse_spark_open_openrouter_high_16k_v1"


def test_environment_puts_the_prefix_system_libraries_on_the_library_path(tmp_path):
    from ora.app.gpu_recipes import SYSTEM_LIBRARY_DIR

    (tmp_path / "gpu" / SYSTEM_LIBRARY_DIR).mkdir(parents=True)
    env_file = tmp_path / ".env"
    env_file.write_text("".join(f"{k}={v}\n" for k, v in GPU_KEYS.items()) + f"ORA_GPU_PREFIX={tmp_path / 'gpu'}\n")
    environ = {**PROVIDER, "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"}
    ora_run.prepare_environment("robodojo", env_file=env_file, environ=environ)
    assert environ["LD_LIBRARY_PATH"] == f"{tmp_path / 'gpu' / SYSTEM_LIBRARY_DIR}:/usr/local/cuda/lib64"


def test_environment_refuses_other_hosts_missing_provider_and_missing_install(tmp_path):
    env_file = tmp_path / ".env"
    with pytest.raises(RuntimeError, match="names another execution host"):
        ora_run.prepare_environment("libero", env_file=env_file,
                                    environ={**PROVIDER, **GPU_KEYS, "ORA_EXECUTION_HOST": "test-host"})
    with pytest.raises(RuntimeError, match="DEEPSEEK_BASE_URL and DEEPSEEK_LOW_API_KEY"):
        ora_run.prepare_environment("libero", env_file=env_file, environ=dict(GPU_KEYS))
    with pytest.raises(RuntimeError, match="base URL must be https://api.deepseek.com"):
        ora_run.prepare_environment("libero", env_file=env_file,
                                    environ={**PROVIDER, **GPU_KEYS, "DEEPSEEK_BASE_URL": "http://elsewhere"})
    with pytest.raises(RuntimeError, match=r"ROBODOJO_ROOT.*ora setup --env robodojo"):
        ora_run.prepare_environment("robodojo", env_file=env_file,
                                    environ={**PROVIDER, **{k: v for k, v in GPU_KEYS.items()
                                                            if k != "ROBODOJO_ROOT"}})


def test_libero_task_names_select_the_base_or_pro_builder():
    assert libero_cell("libero_goal/3") == ("libero", "libero_goal", 3)
    assert libero_cell("libero_10") == ("libero", "libero_10", 0)
    from ora.environment.benchmarks.libero_pro import CORE_SUITES
    assert libero_cell(f"{CORE_SUITES[0]}/1") == ("libero_pro", CORE_SUITES[0], 1)
    with pytest.raises(ValueError, match="unknown LIBERO suite"):
        libero_cell("libero_90/0")
    with pytest.raises(ValueError, match="libero_goal/0"):
        libero_cell("libero_goal/x")


def _args(**values):
    parsed = ora_run.gpu_task(build_parser().parse_args(["run", "--env", values.pop("env"), "--task", values.pop("task")]))
    for key, value in values.items():
        setattr(parsed, key, value)
    return parsed


def test_compose_libero_uses_the_shipped_host_slot_and_a_fresh_cohort(monkeypatch, tmp_path):
    from ora.orchestrator.spec import benchmark as benchmark_spec

    built = {}

    def build(suite, task_id, **kwargs):
        built.update(suite=suite, task_id=task_id, **kwargs)
        return {"config": "lb-cell"}

    monkeypatch.setattr(benchmark_spec, "build_libero_spec", build)
    monkeypatch.setattr(benchmark_spec, "compose", lambda spec, **kwargs: tmp_path / "spec.json")
    path = ora_run.compose(_args(env="libero", task="libero_goal/0"), services=lambda: tmp_path / "report.json",
                           confirm=lambda message: pytest.fail("LIBERO needs no qualification"), stream=io.StringIO())
    assert path == tmp_path / "spec.json"
    assert (built["suite"], built["task_id"], built["seed"]) == ("libero_goal", 0, 0)
    assert (built["host"], built["resource_profile"], built["experiment"]) == ("single-gpu", "libero-gpu0",
                                                                                "ora-interactive")
    assert built["service_report"] == tmp_path / "report.json"
    assert built["cohort_id"].startswith("ora-run-libero-")


def test_compose_robotwin_binds_the_qualified_seed_and_imports_its_manifest(monkeypatch, tmp_path):
    from ora.orchestrator.spec import benchmark as benchmark_spec

    manifest = tmp_path / "click_bell.json"
    monkeypatch.setattr(robotwin_cell, "robotwin_seed", lambda task, **kwargs: (manifest, 100000))
    built, written = {}, {}
    monkeypatch.setattr(benchmark_spec, "build_robotwin_spec", lambda task, **kwargs: built.update(task=task, **kwargs)
                        or {"config": "rt"})
    monkeypatch.setattr(benchmark_spec, "compose", lambda spec, **kwargs: written.update(kwargs) or tmp_path / "rt.json")
    ora_run.compose(_args(env="robotwin", task="click_bell"), services=lambda: tmp_path / "r.json", confirm=print,
                    stream=io.StringIO())
    assert (built["seed"], built["seed_manifest"], built["resource_profile"]) == (100000, manifest, "robotwin-gpu0")
    assert written == {"seed_manifest": manifest}


def test_compose_robodojo_binds_the_layered_qualification(monkeypatch, tmp_path):
    from ora.orchestrator.backends.robodojo import spec as robodojo_spec

    binding = {"card": {"path": "c", "sha256": "1"}, "task": {"path": "t", "sha256": "2"}}
    monkeypatch.setattr(robodojo_cell, "robodojo_qualification", lambda task, **kwargs: binding)
    built = {}
    monkeypatch.setattr(robodojo_spec, "build_spec", lambda task, **kwargs: built.update(task=task, **kwargs) or {})
    monkeypatch.setattr(robodojo_spec, "bind_services", lambda spec, report: built.update(report=report) or spec)
    monkeypatch.setattr(robodojo_spec, "compose", lambda spec: tmp_path / "rd.json")
    ora_run.compose(_args(env="robodojo", task="general_pickup"), services=lambda: tmp_path / "r.json", confirm=print,
                    stream=io.StringIO())
    assert built["infrastructure_qualification"] == binding and built["report"] == tmp_path / "r.json"
    assert (built["gpu"], built["resource_profile"]) == ("0", "robodojo-gpu0")
    with pytest.raises(ValueError, match="layout 0 of seed 0"):
        ora_run.compose(_args(env="robodojo", task="general_pickup", seed=3), services=lambda: tmp_path / "r.json",
                        confirm=print, stream=io.StringIO())


def test_seed_manifest_reuse_takes_the_newest_admitted_seed(monkeypatch, tmp_path):
    from ora.orchestrator.spec import benchmark as benchmark_spec

    old = tmp_path / "qualification/1/seed-manifests/click_bell.json"
    new = tmp_path / "qualification/2/seed-manifests/click_bell.json"
    for path, seeds in ((old, [100000]), (new, [100002, 100003])):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"accepted_seeds": seeds}))
    import os
    os.utime(old, (1, 1))

    def admitted(path, *, task, seed, task_config):
        # Launch preflight's rule needs the seed's own record in this manifest.
        if (seed not in json.loads(path.read_text())["accepted_seeds"] or seed == 100002
                or task_config != "demo_randomized"):
            raise ValueError("not admitted")
        return b""

    monkeypatch.setattr(benchmark_spec, "admitted_seed_manifest", admitted)
    assert robotwin_cell.find_seed_manifest("click_bell", task_config="demo_randomized", seed=None,
                                            workspace=tmp_path) == (new, 100003)
    assert robotwin_cell.find_seed_manifest("click_bell", task_config="demo_randomized", seed=100000,
                                            workspace=tmp_path) == (old, 100000)
    assert robotwin_cell.find_seed_manifest("click_bell", task_config="demo_clean", seed=None,
                                            workspace=tmp_path) is None


def test_robodojo_missing_task_does_not_start_qualification(monkeypatch, tmp_path):
    card, task_record = tmp_path / "card-qualification.json", tmp_path / "task-qualification.json"
    task_record.write_text("{}")
    monkeypatch.setattr(robodojo_cell, "_harness_commit", lambda: "a" * 40)
    monkeypatch.setattr(robodojo_cell, "_records", lambda kind: [card] if kind == "card" else [])
    monkeypatch.setattr(robodojo_cell, "live_identity", lambda gpu="0": {"gpu_uuid": "GPU-1", "environment": {}})
    monkeypatch.setattr(robodojo_cell, "card_record_admitted", lambda task, record, **kwargs: record == card)
    monkeypatch.setattr(robodojo_cell, "task_record_admitted",
                        lambda task, record, **kwargs: {"task": str(record)} if record == task_record else None)
    commands, prompts = [], []

    def produce(command, **kwargs):
        commands.append([str(item) for item in command])
        return f"noise\n{task_record}\n"

    monkeypatch.setattr(robodojo_cell, "run_with_progress", produce)
    binding = robodojo_cell.robodojo_qualification("general_pickup", requalify=False, confirm=prompts.append,
                                                   stream=io.StringIO())
    assert binding is None
    assert commands == [] and prompts == []  # this GPU's card record is admitted: nothing runs


def test_robodojo_first_start_on_a_gpu_runs_the_card_qualification_only(monkeypatch, tmp_path):
    monkeypatch.setattr(robodojo_cell, "_harness_commit", lambda: "a" * 40)
    monkeypatch.setattr(robodojo_cell, "_records", lambda kind: [])
    monkeypatch.setattr(robodojo_cell, "live_identity", lambda gpu="0": {"gpu_uuid": "GPU-1", "environment": {}})
    monkeypatch.setattr(robodojo_cell, "log_path", lambda name: tmp_path / f"{name}.log")
    commands, prompts = [], []
    monkeypatch.setattr(robodojo_cell, "run_with_progress",
                        lambda command, **kwargs: commands.append([str(item) for item in command]) or "")
    binding = robodojo_cell.robodojo_qualification("general_pickup", requalify=False, confirm=prompts.append,
                                                   stream=io.StringIO(), gpu="3")
    assert binding is None
    assert len(prompts) == 1 and "first start on GPU 3" in prompts[0]
    assert len(commands) == 1 and commands[0][-5:] == ["--gpu", "3", "--resource-profile", "robodojo-gpu3", "--yes"]


def test_robodojo_reuses_an_admitted_task_record_without_any_episode(monkeypatch, tmp_path):
    record = tmp_path / "task-qualification.json"
    monkeypatch.setattr(robodojo_cell, "_harness_commit", lambda: "a" * 40)
    monkeypatch.setattr(robodojo_cell, "_records", lambda kind: [record] if kind == "task" else [])
    monkeypatch.setattr(robodojo_cell, "live_identity", lambda gpu="0": {"gpu_uuid": "GPU-1", "environment": {}})
    monkeypatch.setattr(robodojo_cell, "task_record_admitted", lambda task, path, **kwargs: {"reused": True})
    monkeypatch.setattr(robodojo_cell, "run_with_progress", lambda *a, **k: pytest.fail("must reuse"))
    assert robodojo_cell.robodojo_qualification("general_pickup", requalify=False,
                                                confirm=lambda m: pytest.fail("nothing to confirm"),
                                                stream=io.StringIO()) == {"reused": True}


def test_confirmation_requires_yes_off_a_terminal(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(RuntimeError, match="--yes"):
        services.confirm("scan", assume_yes=False, stream=io.StringIO())
    services.confirm("scan", assume_yes=True, stream=io.StringIO())
    services.confirm("scan", assume_yes=False, stream=io.StringIO(), reader=lambda prompt: "y")
    with pytest.raises(RuntimeError, match="not confirmed"):
        services.confirm("scan", assume_yes=False, stream=io.StringIO(), reader=lambda prompt: "")


def test_progress_runner_returns_stdout_and_reports_failures_with_the_log(tmp_path):
    stream = io.StringIO()
    out = progress.run_with_progress([sys.executable, "-c", "import sys; print('/record.json'); "
                                      "print('noise', file=sys.stderr)"],
                                     label="probe", estimate="instant", log=tmp_path / "probe.log",
                                     stream=stream, poll_s=0.01)
    assert out.strip() == "/record.json" and "probe: done" in stream.getvalue()
    assert "noise" in (tmp_path / "probe.log").read_text()
    with pytest.raises(RuntimeError, match=r"probe failed .*exit 3.*\n.*boom"):
        progress.run_with_progress([sys.executable, "-c", "import sys; print('boom', file=sys.stderr); "
                                    "sys.exit(3)"], label="probe", estimate="instant",
                                   log=tmp_path / "fail.log", stream=io.StringIO(), poll_s=0.01)


def _fake_operator(monkeypatch, tmp_path, *, ready_after: int, reports=(), identity="matches"):
    from ora.agent.model_services import operator
    from ora.app import services as ora_services

    fields = {key: 1 for key in operator.FIELDS}
    calls = {"health": 0, "start": 0, "qualify": 0}

    def health():
        calls["health"] += 1
        ready = calls["health"] > ready_after
        return {name: {"ready": ready, "health": dict(fields)} for name in operator.NAMES}

    def lifecycle(command, **kwargs):
        calls[command] += 1
        return {name: {"status": "starting", "log": f"{name}.log"} for name in operator.NAMES}

    def qualify(directory):
        calls["qualify"] += 1
        return directory / "new.json"

    def binding(path, surface, capabilities):
        if path.name not in reports:
            raise ValueError("stale")
        return {"services": {name: {**fields, "qualification": {"created_unix_s": __import__("time").time()}}
                             for name in operator.NAMES}}

    monkeypatch.setattr(operator, "health", health)
    monkeypatch.setattr(operator, "lifecycle", lifecycle)
    monkeypatch.setattr(operator, "qualify", qualify)
    monkeypatch.setattr(operator, "binding", binding)
    monkeypatch.setattr(ora_services, "state_directory", lambda explicit: tmp_path)
    monkeypatch.setattr(services, "_identity_state", lambda record: identity)
    for name in operator.NAMES:
        (tmp_path / f"{name}.json").write_text(json.dumps({"pid": 1}))
    (tmp_path / "qualification").mkdir()
    for name in ("stale.json", *reports):
        (tmp_path / "qualification" / name).write_text("{}")
    return calls


def test_services_start_wait_and_qualify_when_nothing_answers(monkeypatch, tmp_path):
    calls = _fake_operator(monkeypatch, tmp_path, ready_after=2)
    monkeypatch.setattr(services.time, "sleep", lambda seconds: None)
    report = services.ensure_services(stream=io.StringIO(), poll_s=0)
    assert report == tmp_path / "qualification/new.json"
    assert (calls["start"], calls["qualify"]) == (1, 1)


def test_running_services_reuse_the_report_naming_their_instances(monkeypatch, tmp_path):
    calls = _fake_operator(monkeypatch, tmp_path, ready_after=0, reports=("current.json",))
    report = services.ensure_services(stream=io.StringIO())
    assert report == tmp_path / "qualification/current.json"
    assert (calls["start"], calls["qualify"]) == (0, 0)


def test_ready_services_this_checkout_did_not_start_are_refused(monkeypatch, tmp_path):
    calls = _fake_operator(monkeypatch, tmp_path, ready_after=0, reports=("current.json",), identity="drifted")
    with pytest.raises(RuntimeError, match="ORA_SAM_PORT"):
        services.ensure_services(stream=io.StringIO())
    assert (calls["start"], calls["qualify"]) == (0, 0)
    for name in ("segmentation", "grasp_proposals"):
        (tmp_path / f"{name}.json").unlink()
    with pytest.raises(RuntimeError, match="not started from this checkout"):
        services.ensure_services(stream=io.StringIO())


def test_an_occupied_port_says_how_to_choose_free_ones(monkeypatch, tmp_path):
    import errno as errno_module

    from ora.agent.model_services import operator

    _fake_operator(monkeypatch, tmp_path, ready_after=99)

    def occupied(command, **kwargs):
        raise OSError(errno_module.EADDRINUSE, "Address already in use")

    monkeypatch.setattr(operator, "lifecycle", occupied)
    with pytest.raises(RuntimeError, match="ORA_SAM_PORT / ORA_GRASP_PORT"):
        services.ensure_services(stream=io.StringIO())


def _run(argv, *, stream):
    """``ora run ARGV`` through the app's parser."""
    return ora_run.run(build_parser().parse_args(["run", *argv]), stream=stream)


def test_main_prepares_then_forwards_to_interactive_with_explicit_authorization(monkeypatch, tmp_path, capsys):
    from ora.app import main as app_main

    monkeypatch.setattr(ora_run, "prepare_environment", lambda environment, env_file, provider_profile=None, gpu=None: {})
    monkeypatch.setattr(ora_run, "require_clean_checkout", lambda: None)
    monkeypatch.setattr(services, "ensure_services", lambda stream: tmp_path / "report.json")
    monkeypatch.setattr(services, "check_service_ports", lambda: None)
    monkeypatch.setattr(ora_run, "compose", lambda args, services, **kwargs: services() and tmp_path / "spec.json")
    forwarded = []
    monkeypatch.setattr(app_main, "main", lambda argv: forwarded.append(argv) or 0)
    assert _run(["--env", "libero", "--task", "libero_goal/0", "--prepare-only"],
                        stream=io.StringIO()) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "spec.json") and not forwarded
    assert _run(["--env", "libero", "--task", "libero_goal/0", "--confirm-launch", "--emit", "jsonl"],
                        stream=io.StringIO()) == 0
    argv = forwarded[0]
    assert argv[:3] == ["interactive", "--spec", str(tmp_path / "spec.json")]
    assert "--confirm-launch" in argv and argv[-2:] == ["--emit", "jsonl"] and "--world-port" not in argv
    assert _run(["--env", "libero", "--task", "libero_goal/0"], stream=io.StringIO()) == 0
    assert "--confirm-launch" not in forwarded[1]


def test_main_reports_a_refusal_without_a_traceback(monkeypatch, capsys):
    def refuse(environment, env_file, provider_profile=None, gpu=None):
        raise RuntimeError("DEEPSEEK_BASE_URL and DEEPSEEK_LOW_API_KEY unset")

    monkeypatch.setattr(ora_run, "prepare_environment", refuse)
    assert _run(["--env", "robotwin", "--task", "click_bell"], stream=io.StringIO()) == 2
    assert "ora run: DEEPSEEK_BASE_URL" in capsys.readouterr().err


def test_ora_run_parses_the_gpu_options_and_refuses_them_on_synthetic(monkeypatch):
    from ora.app import main as app_main, quickstart

    seen = []
    monkeypatch.setattr(ora_run, "run", lambda args: seen.append((args.environment, args.task)) or 0)
    assert app_main.main(["run", "--env", "robodojo", "--task", "general_pickup"]) == 0
    assert seen == [("robodojo", "general_pickup")]
    monkeypatch.undo()
    monkeypatch.setattr(quickstart, "command_run", lambda args: pytest.fail("a GPU option on Synthetic must stop"))
    assert app_main.main(["run", "--task", "general_pickup"]) == 2
    with pytest.raises(ValueError, match="needs --task"):
        ora_run.gpu_task(build_parser().parse_args(["run", "--env", "robotwin"]))
    defaults = ora_run.gpu_task(build_parser().parse_args(["run", "--env", "robotwin", "--task", "click_bell"]))
    assert (defaults.task_config, defaults.candidates, defaults.poll) == ("demo_randomized", 4, 0.2)
    assert defaults.env_file == ora_run.PROJECT_ROOT / ".env"


def test_robodojo_records_must_name_this_gpu_and_interpreter(monkeypatch, tmp_path):
    from ora.orchestrator import robodojo_qualification
    from ora.orchestrator.backends.robodojo import spec as robodojo_spec

    record = {"device_binding": {"selected_gpu_uuid": "GPU-1"}, "environment": {"versions": {"torch": "2.7.0+cu128"}}}
    monkeypatch.setattr(robodojo_spec, "layered_qualification", lambda path: {"card": {}, "task": {}})
    monkeypatch.setattr(robodojo_cell, "_robodojo_runtime", lambda task, gpu="0": {})
    monkeypatch.setattr(robodojo_qualification, "load_qualification", lambda runtime, **kwargs: record)
    same = {"gpu_uuid": "GPU-1", "environment": {"versions": {"torch": "2.7.0+cu128"}}}
    path = tmp_path / "task-qualification.json"
    assert robodojo_cell.task_record_admitted("general_pickup", path, commit="c", identity=same) is not None
    assert robodojo_cell.task_record_admitted("general_pickup", path, commit="c",
                                              identity={**same, "gpu_uuid": "GPU-2"}) is None
    assert robodojo_cell.task_record_admitted("general_pickup", path, commit="c",
                                              identity={**same, "environment": {"versions": {}}}) is None


def test_robotwin_scan_without_an_accepted_seed_says_how_to_widen_it(monkeypatch, tmp_path):
    from ora.orchestrator.backends.robotwin import evaluation_seeds as robotwin_evaluation

    plan = {"qualification": {"estimated_wall_minutes": 37.5}, "plan_sha256": "abc"}
    monkeypatch.setattr(robotwin_evaluation, "WORKSPACE", tmp_path)
    monkeypatch.setattr(robotwin_evaluation, "prepare", lambda **kwargs: plan)
    monkeypatch.setattr(robotwin_evaluation, "save", lambda path, value: None)
    monkeypatch.setattr(robotwin_cell, "log_path", lambda name: tmp_path / f"{name}.log")
    accepted = []

    def scan(command, **kwargs):
        accepted.append(kwargs["accepted_codes"])
        record = tmp_path / "qualification/run-1"
        (record / "seed-manifests").mkdir(parents=True)
        (record / "prepared-plan.json").write_text(json.dumps({"plan_sha256": "abc"}))
        (record / "seed-manifests/click_bell.json").write_text(json.dumps({"accepted_seeds": []}))
        return ""

    monkeypatch.setattr(robotwin_cell, "run_with_progress", scan)
    with pytest.raises(RuntimeError, match="larger --candidates"):
        robotwin_cell.robotwin_seed("click_bell", task_config="demo_randomized", seed=None, candidates=4,
                                    requalify=True, confirm=lambda message: None,
                                    environ={"ROBOTWIN_ROOT": "/r", "ROBOTWIN_PYTHON": "/p"}, stream=io.StringIO())
    assert accepted == [(0, 2)]


def test_an_old_report_is_not_reused_for_a_new_session(monkeypatch, tmp_path):
    import time as clock
    from ora.agent.model_services import operator

    fields = {key: 1 for key in operator.FIELDS}
    created = {"value": clock.time() - 21 * 3600}
    monkeypatch.setattr(operator, "binding", lambda path, surface, capabilities: {"services": {
        name: {**fields, "qualification": {"created_unix_s": created["value"]}} for name in operator.NAMES}})
    (tmp_path / "report.json").write_text("{}")
    health = {name: {"ready": True, "health": dict(fields)} for name in operator.NAMES}
    assert services.reusable_report(tmp_path, health) is None
    created["value"] = clock.time() - 3600
    assert services.reusable_report(tmp_path, health) == tmp_path / "report.json"


def test_occupied_service_ports_are_refused_before_any_qualification(monkeypatch, tmp_path):
    # 2026-10-15, shared RTX PRO 6000 host: another user's services held 8114/8115;
    # ora run spent the 2 min RoboTwin seed scan before ensure_services refused.
    _fake_operator(monkeypatch, tmp_path, ready_after=99)
    monkeypatch.setattr(services, "_port_in_use", lambda port: True)
    monkeypatch.setattr(services, "_identity_state", lambda record: "drifted")
    with pytest.raises(RuntimeError, match="ORA_SAM_PORT / ORA_GRASP_PORT"):
        services.check_service_ports()
    # Our own services, still warming up, hold their ports legitimately.
    monkeypatch.setattr(services, "_identity_state", lambda record: "matches")
    services.check_service_ports()
    monkeypatch.setattr(services, "_port_in_use", lambda port: False)
    monkeypatch.setattr(services, "_identity_state", lambda record: "drifted")
    services.check_service_ports()


def test_own_services_from_another_namespace_are_not_called_another_users(monkeypatch, tmp_path):
    # 2026-10-15, RTX PRO 6000 acceptance: this checkout started the services from another
    # container; /proc/<pid>/exe was unreadable and the refusal blamed "another user's services?".
    _fake_operator(monkeypatch, tmp_path, ready_after=0, reports=("current.json",), identity="uninspectable")
    with pytest.raises(RuntimeError) as ready:
        services.ensure_services(stream=io.StringIO())
    (tmp_path / "warming").mkdir()
    _fake_operator(monkeypatch, tmp_path / "warming", ready_after=99, identity="uninspectable")
    monkeypatch.setattr(services, "_port_in_use", lambda port: True)
    with pytest.raises(RuntimeError) as warming:
        services.check_service_ports()
    for error in (str(ready.value), str(warming.value)):
        assert "segmentation at http://" in error and "grasp_proposals at http://" in error
        assert error.count("was started by this checkout's model-services, not by another user") == 2
        assert "another container" in error and "ora services stop" in error
        assert "another user's services?" not in error
        assert error.endswith("or set free ports with ORA_SAM_PORT / ORA_GRASP_PORT in .env")


def test_mixed_refusal_names_each_case_and_the_ports_remedy_once():
    message = services.refusal([
        ("uninspectable", "segmentation at http://127.0.0.1:8114 was started by this checkout's model-services"),
        ("foreign", "grasp_proposals at http://127.0.0.1:8115"),
    ])
    assert message.startswith("grasp_proposals at http://127.0.0.1:8115 " + services.FOREIGN_SERVICES_HINT)
    assert message.count("ORA_SAM_PORT") == 1 and "segmentation at" in message


def test_main_checks_ports_before_composing(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(ora_run, "prepare_environment", lambda environment, env_file, provider_profile=None, gpu=None: {})
    monkeypatch.setattr(ora_run, "require_clean_checkout", lambda: None)

    def occupied():
        order.append("ports")
        raise RuntimeError("segmentation at http://127.0.0.1:8114 " + services.FOREIGN_SERVICES_HINT)

    monkeypatch.setattr(services, "check_service_ports", occupied)
    monkeypatch.setattr(ora_run, "compose", lambda *args, **kwargs: order.append("compose"))
    assert _run(["--env", "robotwin", "--task", "click_bell", "--yes"], stream=io.StringIO()) == 2
    assert order == ["ports"]


def test_gpu_selects_that_gpu_s_slot_and_moves_the_perception_services_there(monkeypatch, tmp_path):
    from ora.agent.model_services.service_env import service_key
    from ora.orchestrator.preflight.resources import resolve_execution_resource
    from ora.orchestrator.spec import benchmark as benchmark_spec

    env_file = tmp_path / ".env"
    env_file.write_text("".join(f"{k}={v}\n" for k, v in GPU_KEYS.items()))
    environ = dict(PROVIDER)
    ora_run.prepare_environment("libero", env_file=env_file, environ=environ, gpu=3)
    assert environ[service_key("SAM", "GPU")] == environ[service_key("GRASP", "GPU")] == "3"
    untouched = dict(PROVIDER)
    ora_run.prepare_environment("libero", env_file=env_file, environ=untouched)
    assert untouched[service_key("SAM", "GPU")] == GPU_KEYS[service_key("SAM", "GPU")]

    built = {}
    monkeypatch.setattr(benchmark_spec, "build_libero_spec", lambda suite, task_id, **kwargs: built.update(kwargs)
                        or {"config": "lb"})
    monkeypatch.setattr(benchmark_spec, "compose", lambda spec, **kwargs: tmp_path / "spec.json")
    ora_run.compose(_args(env="libero", task="libero_goal/0", gpu=3), services=lambda: tmp_path / "r.json",
                    confirm=print, stream=io.StringIO())
    assert built["resource_profile"] == "libero-gpu3"
    for kind, family in (("libero_pro", "libero"), ("robotwin", "robotwin"), ("robodojo", "robodojo")):
        for index in range(8):
            binding = resolve_execution_resource(f"{family}-gpu{index}", runtime_kind=kind, required_host="single-gpu")
            assert binding["worker_environment"]["CUDA_VISIBLE_DEVICES"] == str(index)
            assert binding["slot_id"] == f"gpu{index}"
    assert robodojo_cell.slot(None) == robodojo_cell.SLOT == "robodojo-gpu0"
    assert robodojo_cell.slot(5) == "robodojo-gpu5" and robotwin_cell.slot(5) == "robotwin-gpu5"


def test_gpu_must_be_one_this_machine_lists(monkeypatch):
    monkeypatch.setattr(ora_run.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="0\n1\n"))
    ora_run.check_gpu(None)
    ora_run.check_gpu(1)
    with pytest.raises(ValueError, match="this machine's GPUs are 0, 1"):
        ora_run.check_gpu(3)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--env", "robodojo", "--task", "general_pickup", "--gpu", "9"])


def test_a_prepared_single_gpu_spec_gets_what_ora_run_would_have_given_it(monkeypatch, tmp_path):
    from ora.app import run_services

    calls = []
    monkeypatch.setattr(ora_run, "prepare_environment", lambda environment, **kwargs: calls.append((environment, kwargs)))
    monkeypatch.setattr(run_services, "ensure_services", lambda **kwargs: calls.append(("services",)))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"runtime": {"kind": "robodojo", "required_host": "single-gpu",
                                            "resource_profile": "robodojo-gpu6"},
                                "policy": {"provider_profile": "deepseek_flash_openai_low_16k_v1"}}))
    assert ora_run.bind_prepared_specs([spec], env_file=tmp_path / ".env") is True
    environment, kwargs = calls[0]
    assert environment == "robodojo" and kwargs["gpu"] == 6
    assert kwargs["provider_profile"] == "deepseek_flash_openai_low_16k_v1" and calls[1] == ("services",)

    calls.clear()
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"runtime": {"kind": "robodojo", "required_host": "lab-node",
                                             "resource_profile": "robodojo-019-g6s1"}, "policy": {}}))
    assert ora_run.bind_prepared_specs([other, tmp_path / "missing.json"]) is False and calls == []
