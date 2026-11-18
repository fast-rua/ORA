"""``ora doctor``: read-only setup checks; never display credential values or start a simulator.

    ora doctor [--env synthetic|robotwin|libero_pro|robodojo] [--json]
    ora doctor --env local [--root DIR] [--json]     the local CPU setup (``local.doctor``)

A simulator backend says what to discover (``backends.registry.DoctorProbe``).
The ``agent_core`` check reports the Node.js and ora-pi package readiness of the
default Pi agent core (``episode/agent_core.py``); it runs ``node --version`` only.
Where the environment's default policy runs code mode (RoboDojo), the code-mode
sandbox's host prerequisites are checked too: ``bwrap``, unprivileged user
namespaces and the sandbox interpreter against its lock.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

from ora.orchestrator.episode import agent_core
from ora.orchestrator.backends.registry import find


def _agent_core_check() -> tuple[dict, dict]:
    """Node.js and the locked ora-pi packages for the default Pi core; versions and paths only."""
    try:
        forced = agent_core.override()
    except agent_core.AgentCoreError as exc:
        return {"check": "agent_core", "passed": False, "detail": str(exc)}, {"override": "invalid"}
    selected = forced or agent_core.DEFAULT_AGENT_CORE
    status = agent_core.pi_runtime()
    node, dependencies = status["node"], status["dependencies"]
    summary = {"default": agent_core.DEFAULT_AGENT_CORE, "selected": selected, "override": forced,
               "node": {"path": node["path"], "version": node["version"], "candidates": node["candidates"]},
               "pi_version": dependencies.get("pi_version"), "ora_pi_dependencies": dependencies["detail"],
               "pi_ready": status["passed"]}
    if status["passed"]:
        detail = (f"{selected}{' (ORA_AGENT_CORE)' if forced else ' (default)'}; Pi core ready: node "
                  f"{node['version']} at {node['path']}, Pi {dependencies.get('pi_version')}, {dependencies['detail']}")
    else:
        detail = status["remedy"]
    return {"check": "agent_core", "passed": status["passed"], "detail": detail}, summary


def default_code_mode(environment: str) -> bool:
    """Whether a RunSpec for ``environment`` that names no ``policy.code_mode`` runs code mode (its default baseline's
    ``code_mode``, ``spec/defaults.py``)."""
    from ora.agent.composition.baseline import default_baseline_id, load_native_baseline
    from ora.orchestrator.paths import PROJECT_ROOT

    selected = find(environment)
    if selected is None or not selected.capabilities.baseline_instance:
        return False
    catalog = load_native_baseline(PROJECT_ROOT, default_baseline_id(environment))
    return catalog.get("code_mode") is not None and environment in catalog["instances"]


def _code_mode_checks(environment: str) -> list[dict]:
    """The code-mode sandbox's host prerequisites, where the environment's default runs code mode."""
    if not default_code_mode(environment):
        return []
    from ora.agent.code_mode.sandbox import prerequisite_checks

    return prerequisite_checks()


def report(environment: str = "synthetic") -> dict:
    try:
        version = importlib.metadata.version("ora-agent")
    except importlib.metadata.PackageNotFoundError:
        version = "source checkout (not installed)"
    checks = [{"check": "python", "passed": sys.version_info >= (3, 11),
               "detail": platform.python_version()}]
    credentials = {name: bool(os.environ.get(name)) for name in (
        "ORA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NATIVE_MODEL_API_KEY",
        "DEEPSEEK_LOW_API_KEY", "DEEPSEEK_API_KEY",
    )}
    optional = {"credentials_present": credentials,
                "ffmpeg": bool(shutil.which("ffmpeg")),
                "egl_library": bool(ctypes.util.find_library("EGL")),
                "vulkan_library": bool(ctypes.util.find_library("vulkan"))}
    gpu = shutil.which("nvidia-smi")
    optional["nvidia_smi"] = bool(gpu)
    # A simulator backend says what to discover (registry.DoctorProbe); Synthetic needs nothing.
    selected = find(environment)
    probe_spec = selected.doctor if selected is not None else None
    if probe_spec is not None:
        var, module, asset_vars = probe_spec.python_env, probe_spec.module, probe_spec.directory_envs
        interpreter = os.environ.get(var, "")
        present = bool(interpreter and Path(interpreter).is_file())
        detail = f"Set {var} to the isolated simulator interpreter"
        if present:
            discovery = "import importlib.util; "
            if probe_spec.source_env is not None and os.environ.get(probe_spec.source_env[0]):
                source = str(Path(os.environ[probe_spec.source_env[0]]) / probe_spec.source_env[1])
                discovery += "import sys; sys.path.insert(0, " + repr(source) + "); "
            try:
                probe = subprocess.run([interpreter, "-c",
                    discovery + "raise SystemExit(0 if importlib.util.find_spec("
                    + repr(module) + ") else 1)"], capture_output=True, timeout=15,
                    env={k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}})
                present = probe.returncode == 0
                detail = f"{module}: {'discoverable' if present else 'missing'}; simulator not started"
            except (OSError, subprocess.TimeoutExpired):
                present = False
                detail = f"{var} probe failed or timed out"
        checks.append({"check": var, "passed": present, "detail": detail})
        for var in asset_vars:
            value = os.environ.get(var, "")
            checks.append({"check": var, "passed": bool(value and Path(value).is_dir()),
                           "detail": "directory exists" if value and Path(value).is_dir() else "unset or missing"})
        gpu_ready = False
        if gpu:
            try:
                # 30 s: on a busy multi-GPU machine nvidia-smi alone can take longer than 5 s.
                probe = subprocess.run([gpu, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=30)
                optional["gpus"] = probe.stdout.strip().splitlines() if probe.returncode == 0 else []
                gpu_ready = bool(optional["gpus"])
            except (OSError, subprocess.TimeoutExpired):
                optional["gpus"] = []
        checks.append({"check": "nvidia_driver", "passed": gpu_ready,
                       "detail": "visible GPUs discovered" if gpu_ready else "no GPU discovered by nvidia-smi"})
        if probe_spec.extra_checks is not None:
            checks.extend(probe_spec.extra_checks(interpreter if present else "", gpu))
    checks.extend(_code_mode_checks(environment))
    core_check, core = _agent_core_check()
    checks.append(core_check)
    return {"ora_version": version, "platform": platform.system(), "environment": environment,
            "passed": all(item["passed"] for item in checks), "checks": checks,
            "optional": optional, "agent_core": core,
            "note": "Discovery checks only. Launch preflight verifies exact sources, assets, services and qualification."}


ENVIRONMENTS = ("synthetic", "robotwin", "libero_pro", "robodojo", "local")
#: The keys ``ora run --env synthetic`` reads without ``--provider-profile`` (``spec.quickstart.environment_policy``).
QUICKSTART_KEYS = ("ORA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
QUICKSTART_HINT = ("The quickstart (ora run --env synthetic) needs ORA_API_KEY or OPENAI_API_KEY, or "
                   "--provider-profile ID with that profile's key (ora providers); without a key: "
                   "uv run --locked python -m examples.quickstart_smoke")


def register(sub: Any) -> None:
    doctor = sub.add_parser("doctor", help="check setup without displaying credentials or starting simulators",
                            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    doctor.add_argument("--env", choices=ENVIRONMENTS, default="synthetic")
    doctor.add_argument("--root", type=Path, help="with --env local: the local setup root")
    doctor.add_argument("--json", action="store_true", help="machine-readable readiness report")


def run(args: argparse.Namespace) -> int:
    if args.env != "local":
        return command_doctor(args)
    from .local import doctor, print_doctor

    result = doctor(args.root)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print_doctor(result)
    return 0 if result["ready"] else 2


def load_checkout_env(environment: str, environ=os.environ, env_path: Path | None = None) -> list[str]:
    """For a simulator environment, read the checkout's ``.env`` as ``ora run`` does (the shell's values win),
    with the system libraries ``ora setup`` unpacked on ``LD_LIBRARY_PATH``; the names it set."""
    if environment in ("synthetic", "local"):
        return []
    from . import env_file
    from .gpu_recipes import add_library_path

    loaded = env_file.load_into(environ, env_path or Path(__file__).resolve().parents[2] / ".env")
    if environ.get("ORA_GPU_PREFIX"):
        add_library_path(environ, Path(environ["ORA_GPU_PREFIX"]))
    return loaded


def command_doctor(args: argparse.Namespace) -> int:
    loaded = load_checkout_env(args.env)
    result = report(args.env)
    result["env_file_keys"] = sorted(loaded)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"ORA {result['ora_version']} · Python {platform.python_version()} · {result['platform']}")
        if loaded:
            print(f"Read {len(loaded)} setting(s) from the checkout's .env, as ora run does")
        for item in result["checks"]:
            print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['check']}: {item['detail']}")
        present = result["optional"]["credentials_present"]
        print("Credentials present: " + ", ".join(k for k, v in present.items() if v)
              if any(present.values()) else "Credentials present: none")
        if args.env == "synthetic" and not any(present[name] for name in QUICKSTART_KEYS):
            print(QUICKSTART_HINT)
        print(result["note"])
    return 0 if result["passed"] else 1
