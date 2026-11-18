"""``policy.agent_core``: the Pi core, the refusal of the retired Native core, the Pi runtime checks."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from ora.agent.contracts import PolicySessionError
from ora.app import doctor
from ora.app.main import build_parser, main
from ora.orchestrator.spec import quickstart
from ora.orchestrator.episode import agent_core
from ora.orchestrator.episode import policy_factory
from ora.orchestrator.preflight import gates as preflight
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec

MISSING_NODE = {"passed": False, "path": None, "version": None, "candidates": []}
#: What every refusal of the retired core says.
RETIRED = ("removed from ORA", "robot-harness repository", "Pi agent core only")


def _retired(message: str) -> bool:
    return all(part in message for part in RETIRED)


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setenv(agent_core.OVERRIDE_ENV, "")  # unset, restored after the test
    for name, value in (("ORA_API_KEY", "placeholder"), ("ORA_BASE_URL", "http://127.0.0.1:9/v1"),
                        ("ORA_MODEL", "fixture-model")):
        monkeypatch.setenv(name, value)
    return quickstart.synthetic_spec()


def _with(spec, **policy):
    return {**spec, "policy": {**spec["policy"], **policy}}


def test_a_fresh_plan_records_pi_and_a_runspec_naming_native_is_refused(synthetic, monkeypatch):
    default = resolve_run_spec(synthetic)
    named = resolve_run_spec(_with(synthetic, agent_core="pi"))
    assert default["policy"]["agent_core"] == named["policy"]["agent_core"] == "pi"
    assert default["plan_sha256"] == named["plan_sha256"]
    with pytest.raises(RunSpecError) as refused:
        resolve_run_spec(_with(synthetic, agent_core="native"))
    assert _retired(str(refused.value)) and "policy.agent_core: native" in str(refused.value)
    monkeypatch.setenv(agent_core.OVERRIDE_ENV, "native")
    with pytest.raises(RunSpecError) as refused:
        resolve_run_spec(synthetic)
    assert _retired(str(refused.value)) and "ORA_AGENT_CORE=native" in str(refused.value)
    monkeypatch.setenv(agent_core.OVERRIDE_ENV, "pi")  # the one value the override takes: no change
    assert resolve_run_spec(synthetic)["plan_sha256"] == default["plan_sha256"]
    monkeypatch.setenv(agent_core.OVERRIDE_ENV, "codex")
    with pytest.raises(RunSpecError, match="ORA_AGENT_CORE must be pi"):
        resolve_run_spec(synthetic)
    monkeypatch.setenv(agent_core.OVERRIDE_ENV, "")
    with pytest.raises(RunSpecError, match="policy.agent_core must be pi"):
        resolve_run_spec(_with(synthetic, agent_core="python"))


def test_the_pi_core_runs_every_context_combination(synthetic, monkeypatch, tmp_path):
    import ora.agent.pi.factory as pi_factory
    from tests.support.orchestrator.native_synthetic_spec import synthetic_spec

    spec = synthetic_spec(seed=0)  # native_slots_v3 with rewrite_compact_v1, the last Native-only pairing
    spec["policy"]["context_retention"] = "rewrite_compact_v1"
    plan = resolve_run_spec(spec)
    assert plan["policy"]["agent_core"] == "pi"
    monkeypatch.setattr(pi_factory, "build_pi_policy_session", lambda **kwargs: ("pi", kwargs["node"]))
    found = {"passed": True, "path": "/opt/node/bin/node", "version": "v22.19.0", "candidates": []}
    monkeypatch.setattr(agent_core, "find_node", lambda environ=None: found)
    monkeypatch.setattr(agent_core, "pi_dependencies",
                        lambda root=None: {"passed": True, "detail": "ready", "pi_version": "1.3.0"})
    assert _build(plan, tmp_path) == ("pi", "/opt/node/bin/node")
    assert preflight._agent_core(plan)["passed"] is True


@pytest.mark.parametrize("policy,named", [
    ({"kind": "native_api"}, "it predates policy.agent_core"),
    ({"kind": "native_api", "agent_core": "native"}, "policy.agent_core: native"),
], ids=["unrecorded", "native"])
def test_a_persisted_plan_resolved_for_native_does_not_start(policy, named, monkeypatch, tmp_path):
    import ora.agent.pi.factory as pi_factory

    monkeypatch.setenv(agent_core.OVERRIDE_ENV, "")
    monkeypatch.setattr(pi_factory, "build_pi_policy_session", lambda **kwargs: pytest.fail("a session started"))
    for environ in ({}, {agent_core.OVERRIDE_ENV: "pi"}):  # the override does not re-label a persisted plan
        with pytest.raises(agent_core.AgentCoreError) as refused:
            agent_core.selected(policy, environ)
        message = str(refused.value)
        assert named in message and "resolve its RunSpec again" in message
        assert "robot-harness repository" in message and "replays with ora watch and ora replay" in message
    with pytest.raises(PolicySessionError, match="resolved for the Python Native agent core"):
        _build({"policy": policy}, tmp_path)
    check = preflight._agent_core({"policy": policy})
    assert check["passed"] is False and named in json.dumps(check["detail"])
    assert agent_core.selected({"kind": "native_api", "agent_core": "pi"}) == "pi"


def test_the_override_takes_pi_only(monkeypatch):
    environ: dict[str, str] = {}
    agent_core.apply_cli_choice("pi", environ)
    assert environ == {agent_core.OVERRIDE_ENV: "pi"}
    with pytest.raises(agent_core.AgentCoreError) as refused:
        agent_core.apply_cli_choice("native", {})
    assert _retired(str(refused.value)) and "--agent-core native" in str(refused.value)
    with pytest.raises(agent_core.AgentCoreError, match="--agent-core must be pi"):
        agent_core.apply_cli_choice("codex", {})
    assert agent_core.AGENT_CORES == ("pi",) and agent_core.UNRECORDED_AGENT_CORE == "native"


def _build(plan, tmp_path):
    return policy_factory.build_policy_session(
        plan=plan, run_root=tmp_path, run_id="fixture", workspace=tmp_path, control_root=tmp_path,
        events_path=tmp_path / "events.jsonl", compiled_policy={}, tools=None, state_root=tmp_path,
        record_context_bundle=None,
    )


def test_the_pi_core_gets_the_checked_node_or_an_actionable_refusal(monkeypatch, tmp_path):
    import ora.agent.pi.factory as pi_factory

    monkeypatch.setenv(agent_core.OVERRIDE_ENV, "")
    monkeypatch.setattr(pi_factory, "build_pi_policy_session", lambda **kwargs: ("pi", kwargs["node"]))
    monkeypatch.setattr(agent_core, "pi_dependencies", lambda root=None: {"passed": True, "detail": "ready"})
    found = {"passed": True, "path": "/opt/node/bin/node", "version": "v22.19.0", "candidates": []}
    monkeypatch.setattr(agent_core, "find_node", lambda environ=None: found)
    plan = {"policy": {"kind": "native_api", "agent_core": "pi"}}
    assert _build(plan, tmp_path) == ("pi", "/opt/node/bin/node")
    monkeypatch.setattr(agent_core, "find_node", lambda environ=None: {
        **MISSING_NODE, "candidates": [{"path": "/usr/bin/node", "version": "v18.19.1"}]})
    with pytest.raises(PolicySessionError) as refused:
        _build(plan, tmp_path)
    message = str(refused.value)
    assert "Node.js >= 22.19 is required (found v18.19.1 at /usr/bin/node)" in message
    assert "curl -fsSL https://nodejs.org/dist/" in message and "native" not in message
    assert refused.value.__cause__ is None and refused.value.__suppress_context__


def test_preflight_and_doctor_report_the_pi_runtime_without_credentials(monkeypatch):
    monkeypatch.setenv(agent_core.OVERRIDE_ENV, "")
    monkeypatch.setenv("ORA_API_KEY", "private-value-must-never-appear")
    monkeypatch.setattr(agent_core, "find_node", lambda environ=None: MISSING_NODE)
    refused = preflight._agent_core({"policy": {"kind": "native_api", "agent_core": "pi"}})
    assert refused["passed"] is False and refused["detail"].startswith("the Pi agent core cannot start")
    assert "Node.js >= 22.19 is required" in refused["detail"] and "native" not in refused["detail"]
    assert preflight._agent_core({"policy": {"kind": "scripted"}})["passed"]
    report = doctor.report()
    check = {item["check"]: item for item in report["checks"]}["agent_core"]
    assert report["passed"] is False and check["detail"].startswith("the Pi agent core cannot start")
    assert "native" not in check["detail"]
    assert report["agent_core"]["selected"] == "pi" and report["agent_core"]["pi_ready"] is False
    assert "private-value-must-never-appear" not in json.dumps(report)
    monkeypatch.setenv(agent_core.OVERRIDE_ENV, "native")  # no longer a way around a missing Node
    report = doctor.report()
    check = {item["check"]: item for item in report["checks"]}["agent_core"]
    assert report["passed"] is False and _retired(check["detail"])


def test_dependencies_must_match_every_locked_runtime_package(tmp_path):
    lock = {"packages": {"": {"name": "ora-pi"}, "node_modules/a": {"version": "1.0.0"},
                         "node_modules/a/node_modules/b": {"version": "2.0.0"},
                         "node_modules/typescript": {"version": "5.9.3", "dev": True},
                         "node_modules/@earendil-works/pi-coding-agent": {"version": "1.3.0"}}}
    (tmp_path / "package-lock.json").write_text(json.dumps(lock))
    for name, version in (("a", "1.0.0"), ("a/node_modules/b", "1.9.0"), ("@earendil-works/pi-coding-agent", "1.3.0")):
        (tmp_path / "node_modules" / name).mkdir(parents=True)
        (tmp_path / "node_modules" / name / "package.json").write_text(json.dumps({"version": version}))
    status = agent_core.pi_dependencies(tmp_path)
    assert status["passed"] is False and status["pi_version"] == "1.3.0"
    assert status["detail"] == "1 of 3 locked packages differ: a/node_modules/b@2.0.0 (1.9.0)"
    (tmp_path / "node_modules/a/node_modules/b/package.json").write_text(json.dumps({"version": "2.0.0"}))
    assert agent_core.pi_dependencies(tmp_path) == {
        "passed": True, "pi_version": "1.3.0", "locked_packages": 3, "detail": "3 locked runtime packages installed"}


def _node(path: Path, version: str) -> Path:
    path.parent.mkdir(parents=True)
    path.write_text(f"#!/bin/sh\necho {version}\n")
    path.chmod(0o755)
    return path


def test_node_discovery_passes_over_an_old_node_to_ora_node_or_nvm(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    old = _node(tmp_path / "usr/bin/node", "v18.19.1")
    found = agent_core.find_node({"PATH": str(old.parent)})
    assert found["passed"] is False and found["candidates"] == [{"path": str(old), "version": "v18.19.1"}]
    nvm = _node(tmp_path / "home/.nvm/versions/node/v22.19.0/bin/node", "v22.19.0")
    _node(tmp_path / "home/.nvm/versions/node/v22.9.0/bin/node", "v22.9.0")
    assert agent_core.find_node({"PATH": str(old.parent)})["path"] == str(nvm)
    explicit = _node(tmp_path / "opt/node", "v24.1.0")
    assert agent_core.find_node({"PATH": str(old.parent), "ORA_NODE": str(explicit)})["version"] == "v24.1.0"


def test_the_cli_option_takes_pi_and_refuses_the_retired_core(synthetic, tmp_path, capsys, monkeypatch):
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump(synthetic))
    assert main(["plan", "validate", str(spec), "--agent-core", "pi"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["plan_sha256"] == resolve_run_spec(synthetic)["plan_sha256"]
    assert os.environ[agent_core.OVERRIDE_ENV] == "pi"
    monkeypatch.setenv(agent_core.OVERRIDE_ENV, "")
    with pytest.raises(SystemExit) as refused:
        main(["plan", "validate", str(spec), "--agent-core", "native"])
    assert refused.value.code == 2
    error = capsys.readouterr().err
    assert _retired(error) and "--agent-core native" in error
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--env", "libero", "--task", "libero_goal/0", "--agent-core", "native"])
    assert _retired(capsys.readouterr().err)
    assert os.environ[agent_core.OVERRIDE_ENV] == ""
    run_parser = build_parser()._subparsers._group_actions[0].choices["run"]
    assert "--agent-core" not in run_parser.format_help()  # hidden: pi is its only value
