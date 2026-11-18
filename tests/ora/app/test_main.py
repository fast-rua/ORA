"""``ora``'s one parser and command table: registration, dispatch, the process entries and inventories.

The grouped help, ``ora help`` and the topic pages are ``test_help.py``; the generated command
reference is ``test_cli_reference.py``.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

from ora.app import main as app_main
from ora.app.main import COMMANDS, INTERNAL_COMMANDS, PUBLIC_COMMANDS, build_parser
from tests.support.orchestrator.control_plane import ROOT


def test_shared_provider_registry_is_inspectable_without_credentials(capsys) -> None:
    assert app_main.main(["providers"]) == 0
    output = capsys.readouterr().out
    assert "volcengine_agent_plan_glm_5_2_anthropic_v1" in output
    assert "admission=text_tool_qualified" in output
    assert "usage=agent_plan_api" in output
    # A Volcengine key starts with "ark-"; "muse-spark-open" is a model name.
    assert re.search(r"(?<![A-Za-z])ark-", output) is None


def test_one_parser_registers_every_command_of_the_table() -> None:
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == set(COMMANDS) == set(PUBLIC_COMMANDS + INTERNAL_COMMANDS)
    # Each command's module is the one the table names, with register and run.
    for name, target in COMMANDS.items():
        module, _, attribute = target.partition(":")
        assert attribute == "register" and module.startswith("ora.app."), name
        imported = __import__(module, fromlist=["run"])
        assert callable(imported.register) and callable(imported.run), module
    # No forwarded placeholders: every command parses its own options.
    assert not any(action.nargs == "..." for parser in choices.values() for action in parser._actions
                   if getattr(parser, "tool", None) is None)


def test_generic_cli_dispatches_every_public_command_without_live_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ora.agent import providers as provider_package
    from ora.agent.evidence import policy_state, sft_export, trace_dataset, trace_verify
    from ora.agent.model_services import operator
    from ora.app import doctor, evidence, interactive, local, plan as plan_command, play, quickstart, replay, runs
    from ora.app import cameras, setup, web
    from ora.orchestrator import launch, paths
    from ora.orchestrator.evaluation import context_audit, evaluate as evaluation, metrics, ownership_audit
    from ora.orchestrator.evaluation import revalidation
    from ora.orchestrator.spec import policy_lock, run_spec

    run = tmp_path / "run"
    run.mkdir()
    (run / "context-manifest.json").write_text(
        json.dumps({"schema": "context-test"}), encoding="utf-8"
    )
    plan = {"plan_id": "plan-test", "plan_sha256": "sha-test"}
    calls: list[str] = []

    def record(name, value=0):
        return lambda *_args, **_kwargs: calls.append(name) or value

    monkeypatch.setattr(run_spec, "load_run_spec", lambda _path: {"spec": True})
    monkeypatch.setattr(run_spec, "resolve_run_spec", lambda _spec: plan)
    monkeypatch.setattr(plan_command, "display_plan", record("show"))
    monkeypatch.setattr(plan_command, "command_context", record("context"))
    monkeypatch.setattr(launch, "_preflight", record("preflight", {"passed": True}))
    monkeypatch.setattr(launch, "request_stop", record("stop", None))
    for name in ("launch", "retry", "status", "watch"):
        monkeypatch.setattr(runs, f"command_{name}", record(name))
    monkeypatch.setattr(paths, "workspace_topology", record("workspace", {"workspace_root": str(tmp_path)}))
    monkeypatch.setattr(provider_package, "load_builtin_provider_profiles", record("providers", {
        "test": {"protocol": "anthropic_messages", "model": "test", "admission": "text_tool_qualified",
                 "usage_scope": "general_api", "supports_images": False},
    }))
    monkeypatch.setattr(paths, "resolve_run_root", lambda _run_id: run)
    monkeypatch.setattr(policy_state, "verify_policy_state", lambda _root: {"status": "valid", "errors": []})
    monkeypatch.setattr(policy_state, "load_policy_state", record("state", {"step": 0}))
    monkeypatch.setattr(context_audit, "write_context_audit", record("context-audit", {}))
    monkeypatch.setattr(metrics, "write_runtime_metrics", record("metrics", {"schema": "metrics-test"}))
    monkeypatch.setattr(ownership_audit, "write_ownership_audit", record("ownership", {"schema": "ownership-test"}))
    monkeypatch.setattr(evaluation, "evaluate", record("evaluate", {"status": "valid"}))
    monkeypatch.setattr(evaluation, "seal", record("seal", {"status": "sealed"}))
    monkeypatch.setattr(evaluation, "export", lambda _root, destination: calls.append("export") or destination)
    monkeypatch.setattr(trace_verify, "verify_agent_trace", record("trace-validate", {"status": "valid"}))
    monkeypatch.setattr(sft_export, "export_sft", lambda _root, destination: calls.append("export-sft") or destination)
    monkeypatch.setattr(trace_dataset, "read_entries", lambda _path: [{"run": str(run)}])
    monkeypatch.setattr(trace_dataset, "export_dataset", lambda _entries, _destination, **_options: calls.append(
        "export-traces") or {"refused": [], "failed": []})
    monkeypatch.setattr(revalidation, "revalidate_runs", record("revalidate", {"runs": []}))
    monkeypatch.setattr(policy_lock, "command_lock", record("lock"))
    monkeypatch.setattr(quickstart, "command_run", record("run"))
    monkeypatch.setattr(doctor, "command_doctor", record("doctor"))
    monkeypatch.setattr(local, "doctor", record("doctor-local", {"ready": True}))
    monkeypatch.setattr(operator, "lifecycle", record("services", {}))
    # The front ends' implementations start Node, a browser or a simulator: their run is the boundary.
    for module, name in ((setup, "setup"), (interactive, "interactive"), (web, "web"), (replay, "replay"),
                         (play, "play"), (cameras, "cameras")):
        monkeypatch.setattr(module, "run", lambda args, _name=name: calls.append(f"{_name}:{args.command}") or 0)
    tools: list[tuple[str, list[str]]] = []
    for target in ("ora.orchestrator.reports.run_report:main", "ora.orchestrator.spec.benchmark:robotwin_main",
                   "ora.orchestrator.backends.robotwin.evaluation_seeds:main", "ora.orchestrator.suite.queue:main",
                   "ora.agent.composition.prompt.bundle:main"):
        module_name, _, function = target.partition(":")
        module = __import__(module_name, fromlist=[function])
        monkeypatch.setattr(module, function, lambda argv, _name=module_name: tools.append((_name, argv)) or 0)

    spec = str(tmp_path / "spec.yaml")
    invocations = {
        "setup": ["setup", "--json"],
        "doctor": ["doctor", "--json"],
        "run": ["run", "--env", "synthetic"],
        "play": ["play", "--env", "synthetic"],
        "replay": ["replay", "run-test", "--export", str(tmp_path / "run.mp4")],
        "web": ["web", "run-test", "--no-browser"],
        "cameras": ["cameras", "run-test", "--once"],
        "interactive": ["interactive", "--spec", spec],
        "attach": ["attach", "run-test"],
        "sessions": ["sessions", "export", "run-test", "--output", str(tmp_path / "s.html")],
        "local": ["local", "--suite", "libero_goal"],
        "plan": ["plan", "show", spec],
        "launch": ["launch", spec, "--yes"],
        "retry": ["retry", "run-test", "--yes"],
        "status": ["status"],
        "stop": ["stop", "run-test"],
        "watch": ["watch", "run-test"],
        "workspace": ["workspace"],
        "task": ["task", "resolve", str(ROOT / "examples/tasks/robodojo-toolkit-v6.yaml")],
        "evaluate": ["evaluate", "run-test"],
        "seal": ["seal", "run-test"],
        "revalidate": ["revalidate", str(run), "--output-dir", str(tmp_path / "review")],
        "export": ["export", "run-test", "--destination", str(tmp_path / "run.tar.gz")],
        "export-sft": ["export-sft", "run-test", "--destination", str(tmp_path / "sft")],
        "export-traces": ["export-traces", "--runs", str(tmp_path / "runs.txt"), "--destination", str(tmp_path / "traces")],
        "report": ["report", "--namespace", "ns", "--cohort", "c1"],
        "metadata": ["metadata", "run-test"],
        "state": ["state", "run-test", "--step", "0"],
        "metrics": ["metrics", "run-test"],
        "ownership": ["ownership", "run-test"],
        "trace-validate": ["trace-validate", "run-test"],
        "spec": ["spec", "robotwin", "tasks"],
        "qualify": ["qualify", "robotwin", "configs"],
        "suite": ["suite", "status", "--experiment", "e", "--group", "g"],
        "providers": ["providers"],
        "services": ["services", "status"],
        "cell": ["cell", "init", "--example", str(tmp_path / "cell.yaml")],
        "baselines": ["baselines"],
        "lock": ["lock", "--check"],
        "toolkit": ["toolkit", "replay", str(run), "--list"],
        "dev": ["dev", "prompt-bundle", "audit", "--bundle", "x"],
        "help": ["help"],
    }
    assert tuple(invocations) == PUBLIC_COMMANDS
    for arguments in invocations.values():
        assert app_main.main(arguments) == 0, arguments
    for arguments in (["plan", "validate", spec], ["plan", "context", spec], ["plan", "preflight", spec],
                      ["doctor", "--env", "local", "--json"]):
        assert app_main.main(arguments) == 0, arguments

    assert {
        "show", "context", "preflight", "launch", "retry", "status", "workspace", "providers", "lock", "watch",
        "stop", "state", "context-audit", "metrics", "ownership", "evaluate", "seal", "export", "trace-validate",
        "export-sft", "export-traces", "revalidate", "run", "doctor", "doctor-local", "services",
        "setup:setup", "interactive:interactive", "interactive:attach", "interactive:sessions", "interactive:local",
        "web:web", "replay:replay", "play:play", "cameras:cameras",
    } <= set(calls)
    # The operator tools get the rest of the line, options and --help included, unparsed.
    assert tools == [
        ("ora.orchestrator.reports.run_report", ["--namespace", "ns", "--cohort", "c1"]),
        ("ora.orchestrator.spec.benchmark", ["tasks"]),
        ("ora.orchestrator.backends.robotwin.evaluation_seeds", ["configs"]),
        ("ora.orchestrator.suite.queue", ["status", "--experiment", "e", "--group", "g"]),
        ("ora.agent.composition.prompt.bundle", ["audit", "--bundle", "x"]),
    ]


def test_spec_qualify_and_play_audit_run_the_tools_each_backend_declares(monkeypatch, capsys) -> None:
    """``ora spec|qualify KIND TOOL`` and ``ora play audit --env KIND`` resolve through the backend registry."""
    from ora.orchestrator import robodojo_play_audit
    from ora.orchestrator.backends.robodojo import lane_qualification
    from ora.orchestrator.backends.robodojo.motion import metrics as motion_metrics
    from ora.orchestrator.backends.robotwin import play_audit
    from ora.orchestrator.spec import benchmark

    calls = []

    def recorder(name):
        return lambda argv: calls.append((name, list(argv), sys.argv[0])) or 0

    monkeypatch.setattr(benchmark, "libero_main", recorder("libero"))
    monkeypatch.setattr(lane_qualification, "main", recorder("lane"))
    monkeypatch.setattr(motion_metrics, "main", recorder("metrics"))
    monkeypatch.setattr(play_audit, "main", recorder("robotwin-audit"))
    monkeypatch.setattr(robodojo_play_audit, "main", recorder("robodojo-audit"))
    assert app_main.main(["spec", "libero", "compose", "libero_goal", "0", "--seed", "1"]) == 0
    assert app_main.main(["spec", "libero_pro", "tasks"]) == 0
    assert app_main.main(["qualify", "robodojo", "card", "stack_bowls", "--gpu", "0"]) == 0
    assert app_main.main(["qualify", "robodojo", "motion-metrics", "RUN"]) == 0
    assert app_main.main(["play", "audit", "--env", "robotwin", "--scene-set", "tabletop-v1"]) == 0
    assert app_main.main(["play", "--env", "robodojo", "audit", "--seeds", "2"]) == 0
    assert calls == [
        ("libero", ["compose", "libero_goal", "0", "--seed", "1"], "ora spec libero"),
        ("libero", ["tasks"], "ora spec libero_pro"),
        ("lane", ["card", "stack_bowls", "--gpu", "0"], "ora qualify robodojo"),
        ("metrics", ["RUN"], "ora qualify robodojo motion-metrics"),
        ("robotwin-audit", ["--scene-set", "tabletop-v1"], "ora play audit --env robotwin"),
        ("robodojo-audit", ["--seeds", "2"], "ora play audit --env robodojo"),
    ]
    capsys.readouterr()
    # Without a tool, each level lists what there is: on stdout with --help, else on stderr and exit 2.
    assert app_main.main(["spec", "--help"]) == 0
    listing = capsys.readouterr().out
    assert "usage: ora spec ENVIRONMENT TOOL" in listing and "libero | libero_pro" in listing
    assert "synthetic" not in listing
    assert app_main.main(["qualify", "robocasa"]) == 2
    assert "motion" in capsys.readouterr().err
    assert app_main.main(["qualify", "robotwin", "--help"]) == 0
    listing = capsys.readouterr().out
    assert all(f"  {name} " in listing for name in ("configs", "prepare", "qualify", "merge", "seal"))
    assert app_main.main(["spec", "synthetic", "tasks"]) == 2
    assert "synthetic has no spec tools" in capsys.readouterr().err
    assert app_main.main(["qualify", "nowhere"]) == 2
    assert "unknown environment 'nowhere'" in capsys.readouterr().err
    assert app_main.main(["qualify", "robodojo", "nothing"]) == 2
    assert "robodojo has no tool 'nothing'" in capsys.readouterr().err
    assert app_main.main(["play", "audit", "--help"]) == 0
    assert "environments: robotwin, robodojo" in capsys.readouterr().out
    assert app_main.main(["play", "audit", "--env", "libero"]) == 2
    assert "libero has no play-scene audit" in capsys.readouterr().err


def test_export_sft_refuses_redaction_options_where_they_do_not_apply(tmp_path: Path, monkeypatch, capsys) -> None:
    from ora.orchestrator import paths

    monkeypatch.setattr(paths, "resolve_run_root", lambda _run_id: tmp_path)
    export = ["export-sft", str(tmp_path), "--destination", str(tmp_path / "out")]
    for argv, message in (
        (["--view", "openai", "--redact-path", "/data/lab"], "--redact-path needs --redact-host-paths"),
        (["--view", "wire", "--check", "--redact-host-paths"], "it takes no --redact-host-paths"),
        (["--redact-host-paths"], "--redact-host-paths need --view wire or openai"),
    ):
        assert app_main.main([*export, *argv]) == 2
        assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_every_main_guard_is_a_listed_process_entry() -> None:
    """The operator tools are ora commands; what keeps ``python -m`` is the inventory in ora/app/README.md."""
    readme = (ROOT / "ora/app/README.md").read_text(encoding="utf-8")
    table = readme.split("| Process entry | Started by |", 1)[1].split("\n\n", 1)[0]
    listed = set()
    for row in table.splitlines()[2:]:
        entry = row.split("`")[1].split()[0]
        path = entry if entry.endswith(".py") else entry.replace(".", "/") + ".py"
        listed.add(path if (ROOT / path).is_file() else path.removesuffix(".py") + "/__main__.py")
    # Three pinned inputs keep theirs (their bytes are hashed: two qualification inputs and the perception
    # adapter identity); ora-remote is a console script.
    listed |= {"ora/orchestrator/robodojo_play_audit.py", "ora/environment/backends/robodojo/motion_qualification.py",
               "ora/agent/model_services/resident.py", "ora/app/remote.py", "ora/robot_client/cli.py"}
    guarded = {path.relative_to(ROOT).as_posix() for path in (ROOT / "ora").rglob("*.py")
               if path.name == "__main__.py" or 'if __name__ == "__main__":' in path.read_text(encoding="utf-8")}
    assert guarded == listed


def test_the_readme_lists_every_command_in_its_group_with_its_module() -> None:
    """ora/app/README.md's command table: the groups of ``ora --help``, each command with the module the table names."""
    from ora.app.main import GROUPS

    readme = (ROOT / "ora/app/README.md").read_text(encoding="utf-8")
    table = readme.split("| Group | Commands | Module |", 1)[1].split("\n\n", 1)[0]
    rows, group = [], None
    for row in table.splitlines()[2:]:
        title, commands, module = (cell.strip() for cell in row.strip("|").split("|"))
        group = title or group
        names = re.findall(r"`([^`]+)`", re.sub(r"\([^)]*\)", "", commands))
        rows += [(group, name, re.findall(r"`([^`]+)`", module)[0]) for name in names]
    assert [(group, name) for group, name, _ in rows] == [
        (titles[0], name) for titles, commands in GROUPS for name, *_ in commands]
    for _, name, module in rows:
        command = name.split()[0]
        expected = "main.py" if command == "help" else COMMANDS[command].split(":")[0].rsplit(".", 1)[1] + ".py"
        assert module == expected, name


def test_runtime_packages_are_in_installed_distribution() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    include = project["tool"]["setuptools"]["packages"]["find"]["include"]
    for package in (
        "ora.app",
        "ora.environment.benchmarks",
        "ora.environment.controllers",
    ):
        assert (ROOT / package.replace(".", "/") / "__init__.py").is_file(), package
        assert any(fnmatch.fnmatchcase(package, pattern) for pattern in include), package
    assert project["project"]["scripts"]["ora"] == "ora.app.main:main"
    assert project["project"]["scripts"]["ora-remote"] == "ora.app.remote:main"
    assert project["tool"]["setuptools"]["package-data"]["viewer"] == [
        "templates/*.html", "static/*.css", "provider-pricing.json", "dashboard/static/*"
    ]


def test_documented_script_and_test_module_inventory_is_current() -> None:
    scripts_doc = (ROOT / "scripts/README.md").read_text(encoding="utf-8")
    tests_doc = (ROOT / "tests/README.md").read_text(encoding="utf-8")
    for path in sorted((ROOT / "scripts").rglob("*")):
        if path.is_file() and path.name != "README.md" and "__pycache__" not in path.parts:
            relative = path.relative_to(ROOT).as_posix()
            assert f"`{relative}`" in scripts_doc, relative
    for relative in (
        "ora/app/main.py",
        "ora/environment/backends/robotwin/qualification.py",
        "viewer/app.py",
    ):
        assert f"`{relative}`" in scripts_doc, relative
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("test_*.py")
        if path.is_file()
    }
    # Every test module states the contract it protects in its docstring, sits
    # in an area directory the README describes, and shared helpers live only
    # in tests/support/.
    for relative in actual:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        assert ast.get_docstring(tree), f"{relative} has no module docstring stating its contract"
        directory = relative.rsplit("/", 1)[0]
        assert directory != "tests", f"{relative} is not in an area directory"
        assert f"`{directory}/`" in tests_doc, directory
    helpers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("*.py")
        if not path.name.startswith("test_") and path.name not in {"__init__.py", "conftest.py"}
    }
    assert {path for path in helpers if not path.startswith("tests/support/")} == set()


@pytest.mark.parametrize("runtime_driver", [{"environment_driver": "direct_libero_v1"}, {}])
def test_plan_display_names_the_libero_driver_it_resolved(runtime_driver) -> None:
    from ora.app.plan import _runtime_revision

    # A RunSpec that names no driver selects the direct driver.
    runtime = {
        "kind": "libero_pro", **runtime_driver,
        "required_libero_pro_revision": "2b910b5b5f53016bef9907632f6f840f1ce2229c",
        "required_data_revision": "c86fc3b8293b5f6f373677018ff3e37f8391602",
    }
    assert _runtime_revision(runtime) == "direct_libero_v1/2b910b5b5f53/c86fc3b8293b"
