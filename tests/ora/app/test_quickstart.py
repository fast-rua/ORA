"""Public installation checks and the documented native_api Synthetic quickstart on the Pi core.

The quickstart's CLI preflight admits only a clean checkout of the plan's
commit, so its Git sees the working tree committed as is
(``tests/support/orchestrator/source_snapshot.py``): a contributor's local changes are what
it exercises, not a reason it fails. Without Node.js >= 22.19 and the
``ora-pi`` packages, the cases that need the default Pi core skip with the
remedy ``ora doctor`` prints, and fail under ``ORA_TEST_REQUIRE_AGENT_CORE=1``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from ora.app import doctor
from ora.orchestrator.spec import quickstart
from ora.orchestrator.backends.robodojo import doctor as robodojo_doctor
from tests.support.orchestrator import source_snapshot
from tests.support.agent_core.preconditions import require_pi_runtime


def _agent_core_ready(monkeypatch):
    """Simulator doctor tests patch ``shutil.which`` and ``subprocess.run``; test_agent_core checks Node."""
    monkeypatch.setattr(doctor, "_agent_core_check",
                        lambda: ({"check": "agent_core", "passed": True, "detail": "fixture"}, {}))


def test_doctor_never_displays_credential_values(monkeypatch):
    require_pi_runtime()  # the doctor passes only with the default core ready
    monkeypatch.setenv("ORA_API_KEY", "private-value-must-never-appear")
    result = doctor.report()
    assert result["passed"]
    assert result["optional"]["credentials_present"]["ORA_API_KEY"] is True
    assert "private-value-must-never-appear" not in json.dumps(result)


def test_doctor_says_what_the_quickstart_needs_when_no_key_is_set(monkeypatch, capsys):
    """``ora doctor`` passes without a key, so it says which key (or profile) the quickstart needs."""
    import argparse

    monkeypatch.setattr(doctor, "_agent_core_check",
                        lambda: ({"check": "agent_core", "passed": True, "detail": "fixture"}, {}))
    for name in doctor.QUICKSTART_KEYS:
        monkeypatch.delenv(name, raising=False)
    assert doctor.command_doctor(argparse.Namespace(env="synthetic", json=False)) == 0
    assert doctor.QUICKSTART_HINT in capsys.readouterr().out.splitlines()
    monkeypatch.setenv("ORA_API_KEY", "placeholder")
    assert doctor.command_doctor(argparse.Namespace(env="synthetic", json=False)) == 0
    assert doctor.QUICKSTART_HINT not in capsys.readouterr().out


def test_quickstart_requires_a_key_and_an_explicit_anthropic_model(monkeypatch):
    for key in ("ORA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ORA_MODEL", "ORA_PROVIDER", "ORA_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="Set ORA_API_KEY"):
        quickstart.synthetic_spec()
    monkeypatch.setenv("ORA_PROVIDER", "anthropic_messages")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    with pytest.raises(ValueError, match="Set ORA_MODEL"):
        quickstart.synthetic_spec()


@pytest.fixture
def no_model_variables(monkeypatch):
    for key in ("ORA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ORA_MODEL", "ORA_PROVIDER", "ORA_BASE_URL",
                "DEEPSEEK_LOW_API_KEY", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"):
        monkeypatch.setenv(key, "")  # recorded, so a value the command sets is undone after the test
        monkeypatch.delenv(key)


def test_the_quickstart_runs_on_a_provider_profile(no_model_variables, monkeypatch, capsys):
    """``ora run --env synthetic --provider-profile ID``: the profile's policy, no ORA_* variable needed."""
    from ora.app import main as app_main
    from ora.orchestrator import launch

    seen = []
    monkeypatch.setattr(launch, "_preflight", lambda plan: seen.append(plan) or {"passed": False})
    monkeypatch.setenv("DEEPSEEK_LOW_API_KEY", "placeholder-never-sent")
    assert app_main.main(["run", "--env", "synthetic", "--provider-profile", "deepseek_flash_openai_low_16k_v1"]) == 2
    policy = seen[0]["policy"]
    assert (policy["provider_profile"], policy["endpoint"], policy["api_key_env"]) == (
        "deepseek_flash_openai_low_16k_v1", "https://api.deepseek.com", "DEEPSEEK_LOW_API_KEY")
    assert seen[0]["task"] == "move-once" and seen[0]["runtime"]["kind"] == "synthetic"
    assert os.environ["DEEPSEEK_BASE_URL"] == "https://api.deepseek.com"  # defaulted, as on a GPU task
    assert "placeholder-never-sent" not in json.dumps(seen[0])
    # Without the profile the quickstart still reads the ORA_* variables, and says so.
    assert app_main.main(["run"]) == 2
    assert "Set ORA_API_KEY or OPENAI_API_KEY" in capsys.readouterr().err


def test_a_provider_profile_without_its_key_or_on_another_endpoint_stops_before_preflight(
        no_model_variables, monkeypatch, capsys):
    from ora.app import main as app_main
    from ora.orchestrator import launch

    monkeypatch.setattr(launch, "_preflight", lambda plan: pytest.fail("refused before preflight"))
    profile = ["run", "--provider-profile", "deepseek_flash_openai_low_16k_v1"]
    assert app_main.main(profile) == 2
    assert ("deepseek_flash_openai_low_16k_v1 needs DEEPSEEK_LOW_API_KEY: export it, or store it once with "
            "./scripts/ora providers key deepseek_flash_openai_low_16k_v1") in capsys.readouterr().err
    monkeypatch.setenv("DEEPSEEK_LOW_API_KEY", "placeholder")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://elsewhere.example/v1")
    assert app_main.main(profile) == 2
    assert "DEEPSEEK_BASE_URL must be https://api.deepseek.com" in capsys.readouterr().err
    assert app_main.main(["run", "--provider-profile", "no_such_profile"]) == 2
    assert "unknown provider profile: no_such_profile" in capsys.readouterr().err


@pytest.fixture(scope="module")
def committed_source(tmp_path_factory):
    return source_snapshot.commit_working_tree(tmp_path_factory.mktemp("committed-source"))


def test_documented_quickstart_completes_and_seals_an_episode(committed_source, monkeypatch, tmp_path, capsys):
    """CI runs the smoke in a throwaway workspace; ``--keep DIR`` keeps the run and says how to replay it."""
    require_pi_runtime()
    from examples import quickstart_smoke
    for name in source_snapshot.CLEARED_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name, value in committed_source.items():
        monkeypatch.setenv(name, value)  # the CLI child inherits them
    # The CLI child (`python -m ora.app`) imports this checkout's ora, the one committed.
    monkeypatch.chdir(source_snapshot.ROOT)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(source_snapshot.ROOT),
                                                                    os.environ.get("PYTHONPATH")))))
    assert quickstart_smoke.main(["--keep", str(tmp_path / "kept")]) == 0
    lines = capsys.readouterr().out.splitlines()
    root = next(tmp_path.glob("kept/outputs/quickstart/runs/*"))
    assert (root / "seal.json").is_file()  # still there after the smoke returned
    assert lines[-3:] == [f"run ID: {root.name}", f"kept in: {root}", f"replay it: ./scripts/ora watch {root}"]


def test_every_synthetic_entry_point_names_its_surface(monkeypatch):
    """``robot_runtime_v2`` is chosen where Synthetic is composed, so none of them reaches the RunSpec fallback."""
    from ora.app.interactive import interactive_spec
    from ora.app.main import build_parser
    from ora.app.play import compose_play

    for name, value in (("ORA_API_KEY", "fixture"), ("ORA_MODEL", "fixture"),
                        ("ORA_BASE_URL", "https://example.invalid/v1")):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("ORA_PROVIDER", raising=False)
    specs = {"run": quickstart.synthetic_spec(), "interactive": interactive_spec(provider="env", task=None, actions=3),
             "play": compose_play(build_parser().parse_args(["play", "--env", "synthetic", "--provider", "env"]))[0]}
    for entry, spec in specs.items():
        assert spec["policy"]["tool_surface"] == {"profile": "robot_runtime_v2", "component_overrides": {}}, entry


def test_the_smoke_says_how_to_run_it_when_ora_is_not_importable():
    """A Python without the project environment (a conda base, say) stops at once with the documented form."""
    result = subprocess.run([sys.executable, "-S", "-I", "examples/quickstart_smoke.py"], cwd=source_snapshot.ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and not result.stdout
    assert result.stderr.count("\n") == 1, result.stderr
    assert "cannot import ORA" in result.stderr
    assert "run it from the checkout as `uv run --locked python -m examples.quickstart_smoke`" in result.stderr


@pytest.mark.parametrize("by_environment", [False, True], ids=["flag-native", "environment-native"])
def test_the_quickstart_refuses_the_retired_native_core(by_environment, tmp_path):
    """``--agent-core native`` and ``ORA_AGENT_CORE=native`` stop before anything runs, and say why."""
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("ORA_") and key not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}}
    environment.update(ORA_API_KEY="local-smoke-placeholder", ORA_MODEL="smoke-model",
                       ORA_BASE_URL="http://127.0.0.1:9/v1", ORA_WORKSPACE_ROOT=str(tmp_path),
                       PYTHONPATH=os.pathsep.join(filter(None, (str(source_snapshot.ROOT),
                                                                os.environ.get("PYTHONPATH")))))
    command = [sys.executable, "-m", "ora.app", "run", "--env", "synthetic"]
    if by_environment:
        environment["ORA_AGENT_CORE"] = "native"
    else:
        command += ["--agent-core", "native"]
    result = subprocess.run(command, env=environment, cwd=source_snapshot.ROOT, capture_output=True,
                            text=True, timeout=120)
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "removed from ORA" in output and "Pi agent core only" in output, output
    assert not list(tmp_path.glob("outputs/*/runs/*"))


def test_libero_doctor_uses_the_declared_source_without_starting_the_simulator(monkeypatch, tmp_path):
    from types import SimpleNamespace
    interpreter = tmp_path / "python"
    interpreter.touch()
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setenv("LIBERO_PRO_PYTHON", str(interpreter))
    monkeypatch.setenv("LIBERO_PRO_ROOT", str(source))
    monkeypatch.setenv("LIBERO_PRO_DATA_ROOT", str(source))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    calls = []
    def probe(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="0, test GPU, 8192\n")
    monkeypatch.setattr(doctor.subprocess, "run", probe)
    _agent_core_ready(monkeypatch)
    result = doctor.report("libero_pro")
    assert result["passed"] is True
    assert str(source / "libero") in calls[0][-1]
    assert "find_spec" in calls[0][-1] and "import libero" not in calls[0][-1]
    assert calls[1][1] == "--query-gpu=index,name,memory.total"


def _robodojo_doctor(monkeypatch, tmp_path, *, loader: bool, driver: str):
    """`ora doctor --env robodojo` with a fake interpreter, loader probe and driver."""
    from types import SimpleNamespace
    runtime = tmp_path / "robodojo-runtime"
    interpreter = runtime / "env/bin/python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.touch()
    for name in ("ROBODOJO_ROOT", "ROBODOJO_ASSET_ROOT"):
        (tmp_path / name).mkdir(exist_ok=True)
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.setenv("ROBODOJO_PYTHON", str(interpreter))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    calls = []
    real_run = doctor.subprocess.run

    def run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["-c", robodojo_doctor.RECEIPT_CHECK]:
            return real_run(command, **kwargs)  # the real receipt check, in a child process
        if command[-1] == "import ctypes; ctypes.CDLL('libvulkan.so.1')":
            return SimpleNamespace(returncode=0 if loader else 1, stdout="")
        if "--query-gpu=driver_version" in command:
            return SimpleNamespace(returncode=0, stdout=f"{driver}\n")
        return SimpleNamespace(returncode=0, stdout="0, test GPU, 8192\n")
    monkeypatch.setattr(doctor.subprocess, "run", run)
    _agent_core_ready(monkeypatch)
    # The code-mode sandbox's checks have their own tests below; these are the simulator's.
    monkeypatch.setattr(doctor, "_code_mode_checks", lambda environment: [
        {"check": "code_mode_sandbox_fixture", "passed": True, "detail": "fixture"}])
    result = doctor.report("robodojo")
    return {item["check"]: item for item in result["checks"]}, result, runtime, calls


def _write_compat_installation(runtime):
    import hashlib
    from pathlib import Path
    installation_dir = Path(doctor.__file__).resolve().parents[1] / "environment/backends/robodojo/installation"
    out = runtime / "graphics-compat"
    files = {"layer": out / "lib/libVkLayer_khronos_profiles.so",
             "profile": out / "profiles/VP_ROBODOJO_RTX_ALLOCATION_COMPAT.json",
             "manifest": out / "share/vulkan/implicit_layer.d/VkLayer_khronos_profiles.json",
             "patch": out / "vulkan-profiles-core-id.patch"}
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    files["layer"].write_bytes(b"layer")
    files["profile"].write_bytes((installation_dir / "VP_ROBODOJO_RTX_ALLOCATION_COMPAT.json").read_bytes())
    files["patch"].write_bytes((installation_dir / "vulkan-profiles-core-id.patch").read_bytes())
    files["manifest"].write_text(json.dumps({"layer": {
        "name": "VK_LAYER_KHRONOS_profiles", "type": "GLOBAL", "library_path": str(files["layer"]),
        "enable_environment": {"ROBODOJO_VULKAN_COMPAT_ENABLE": "1"},
        "disable_environment": {"ROBODOJO_VULKAN_COMPAT_DISABLE": ""}}}))
    refs = {role: {"path": path.relative_to(out).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in files.items()}
    (out / "installation.json").write_text(json.dumps({
        "schema": "ora.robodojo-graphics-compat-installation.v1",
        "source_url": "https://github.com/KhronosGroup/Vulkan-Profiles",
        "source_revision": "a400d0b10059b2cc3cd285475593eee79975a5c4", "patches": [refs["patch"]], **refs}))


def test_robodojo_doctor_reports_a_missing_vulkan_loader_and_compat_layer(monkeypatch, tmp_path):
    checks, result, _, calls = _robodojo_doctor(monkeypatch, tmp_path, loader=False, driver="595.91.07")
    assert result["passed"] is False
    assert checks["vulkan_loader"]["passed"] is False
    assert "LD_LIBRARY_PATH" in checks["vulkan_loader"]["detail"]
    assert "robodojo.md" in checks["vulkan_loader"]["detail"]
    assert checks["vulkan_compatibility_layer"]["passed"] is False
    assert "setup-robodojo graphics-compat" in checks["vulkan_compatibility_layer"]["detail"]
    assert any(command[-1].startswith("import ctypes") and command[0].endswith("env/bin/python") for command in calls)


def test_robodojo_doctor_passes_with_loader_and_verified_receipt(monkeypatch, tmp_path):
    runtime = tmp_path / "robodojo-runtime"
    _write_compat_installation(runtime)
    checks, result, _, _ = _robodojo_doctor(monkeypatch, tmp_path, loader=True, driver="595.91.07")
    assert checks["vulkan_loader"]["passed"] is True
    assert checks["vulkan_compatibility_layer"]["passed"] is True, checks["vulkan_compatibility_layer"]
    assert result["passed"] is True
    receipt = runtime / "graphics-compat/installation.json"
    value = json.loads(receipt.read_text())
    receipt.write_text(json.dumps({**value, "schema": "robot-harness.robodojo-graphics-compat-installation.v1"}))
    checks, result, _, _ = _robodojo_doctor(monkeypatch, tmp_path, loader=True, driver="595.91.07")
    assert checks["vulkan_compatibility_layer"]["passed"] is False
    assert "source drift" in checks["vulkan_compatibility_layer"]["detail"]


def test_robodojo_doctor_skips_the_compat_layer_on_other_drivers(monkeypatch, tmp_path):
    checks, result, _, _ = _robodojo_doctor(monkeypatch, tmp_path, loader=True, driver="570.86.15")
    assert checks["vulkan_compatibility_layer"]["passed"] is True
    assert "not required" in checks["vulkan_compatibility_layer"]["detail"]
    assert result["passed"] is True


def test_only_an_environment_whose_default_runs_code_mode_checks_the_sandbox(monkeypatch):
    """RoboDojo's default baseline runs code mode (``robodojo_toolkit_v7``'s ``code_mode``), so its doctor reports
    the sandbox's host prerequisites; the other environments' defaults use tool calls and do not."""
    assert doctor.default_code_mode("robodojo") is True
    assert not any(doctor.default_code_mode(name) for name in ("synthetic", "libero_pro", "robotwin", "local"))
    from ora.agent.code_mode import sandbox

    monkeypatch.setattr(sandbox, "prerequisite_checks", lambda: [{"check": "code_mode_bwrap", "passed": False,
                                                                   "detail": "fixture"}])
    assert doctor._code_mode_checks("robodojo") == [{"check": "code_mode_bwrap", "passed": False, "detail": "fixture"}]
    assert doctor._code_mode_checks("libero_pro") == []


def test_the_robodojo_doctor_fails_without_bwrap_and_says_what_to_install(monkeypatch, tmp_path):
    from ora.agent.code_mode import sandbox

    monkeypatch.setenv("ORA_CODE_MODE_BWRAP", str(tmp_path / "no-bwrap"))
    monkeypatch.setenv("ORA_GPU_PREFIX", str(tmp_path / "prefix"))
    monkeypatch.delenv("ORA_CODE_MODE_PYTHON", raising=False)
    checks = {item["check"]: item for item in doctor._code_mode_checks("robodojo")}
    assert list(checks) == ["code_mode_bwrap", "code_mode_user_namespaces", "code_mode_interpreter"]
    assert checks["code_mode_bwrap"]["passed"] is False
    assert "install bubblewrap" in checks["code_mode_bwrap"]["detail"]
    assert "ORA_CODE_MODE_BWRAP" in checks["code_mode_bwrap"]["detail"]
    assert checks["code_mode_user_namespaces"]["passed"] is False
    assert checks["code_mode_interpreter"]["passed"] is False
    detail = checks["code_mode_interpreter"]["detail"]
    assert str(sandbox.default_python()) in detail and "./scripts/ora setup --env robodojo" in detail
