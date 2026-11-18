"""``ora run``: one episode. Synthetic by default; ``--env libero|robotwin|robodojo`` runs one GPU task.

    ora run [--env synthetic] [--provider-profile ID]
                                       the bounded Synthetic quickstart episode (``quickstart.command_run``)
    ora run --env ENV --task TASK [--confirm-launch] [...]

On a GPU environment the command takes one task on the shipped ``single-gpu``
host from installed inputs to ORA. It does what ``docs/getting-started/single-gpu.md`` §9.6 does by
hand, in order, and stops at the first failure with the reason:

1. load the checkout's ``.env`` (environment values win; empty values are
   ignored, so a template line cannot blank a key), bind the ``single-gpu``
   host marker, default ``DEEPSEEK_BASE_URL`` to the profile endpoint and run
   preflight's offline ``provider_marker`` configuration check;
2. reuse or run the task's one-time qualification (the environment backend's
   run cell, ``backends/<backend>/run_cell.py``);
3. make the SAM3 / Contact-GraspNet services ready and bind a readiness
   report for the running instances (``run_services``), after step 2 so a
   long qualification cannot age the report out;
4. compose the RunSpec under ``outputs/ora-interactive/specs/`` with a fresh
   single-attempt cohort, then open ``ora interactive`` on it.

Launch authority is unchanged: ``ora interactive`` repeats the full preflight
and launches only with ``--confirm-launch`` and the operator's selection.
Credentials come from the environment (``.env`` included), else from the user's
ORA credential store (``ora providers``); ora run writes none.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

from ora.orchestrator.episode import agent_core  # standard library only
from ora.orchestrator.backends.registry import backend  # standard library only
from ora.version import ora_command

# Not ``paths.PROJECT_ROOT``: importing ``paths`` resolves the outputs root, and
# ``.env`` (which may set ORA_OUTPUTS_ROOT) is loaded first.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOST = "single-gpu"
HOST_MARKER_ENV = "ORA_EXECUTION_HOST"
EXPERIMENT = "ora-interactive"
#: The ``--env`` choices; each is a backend with a run cell (``registry.RunCell``).
ENVIRONMENTS = ("libero", "robotwin", "robodojo")
#: The perception services' settings, as (service, field); ``service_setting`` also accepts their pre-1.0 names.
SERVICE_FIELDS = (("SAM", "PYTHON"), ("SAM", "ROOT"), ("SAM", "CHECKPOINT"), ("SAM", "GPU"), ("GRASP", "PYTHON"),
                  ("GRASP", "ROOT"), ("GRASP", "CHECKPOINT"), ("GRASP", "CONFIG"), ("GRASP", "GPU"))


def service_settings() -> dict[str, tuple[str, str]]:
    """The services' settings by ``.env`` name; imported here, so building the parser loads no agent module."""

    from ora.agent.model_services.service_env import service_key

    return {service_key(service, field): (service, field) for service, field in SERVICE_FIELDS}


def required_keys(environment: str) -> tuple[str, ...]:
    """The ``.env`` keys ``ora run --env ENV`` needs: its run cell's, then the perception services'."""
    return (*backend(environment).run_cell.required_env, *service_settings())


#: Options of a GPU task only, as (dest, option); ``ora run`` on Synthetic refuses them.
GPU_ONLY = (("task", "--task"), ("seed", "--seed"), ("task_config", "--task-config"),
            ("candidates", "--candidates"), ("requalify", "--requalify"), ("yes", "--yes"),
            ("confirm_launch", "--confirm-launch"), ("prepare_only", "--prepare-only"),
            ("env_file", "--env-file"), ("emit", "--emit"), ("poll", "--poll"), ("gpu", "--gpu"))
#: The GPU options' defaults, applied when ``--env`` names a GPU environment.
GPU_DEFAULTS = {"task_config": "demo_randomized", "candidates": 4, "poll": 0.2}


def register(sub: Any) -> None:
    from .world import add_world_options

    parser = sub.add_parser(
        "run", help="run a bounded Synthetic episode, or one GPU task with "
                    "--env libero|robotwin|robodojo --task TASK (ora run --help)",
        description="Run the bounded Synthetic quickstart episode, or prepare one GPU task on the "
                    "single-gpu host and open ORA on it.",
        epilog="Examples: ora run; ora run --provider-profile deepseek_flash_openai_low_16k_v1; "
               "ora run --env libero --task libero_goal/0 --confirm-launch; "
               "ora run --env robotwin --task click_bell --confirm-launch; "
               "ora run --env robodojo --task general_pickup --confirm-launch")
    parser.add_argument("--env", choices=("synthetic", *ENVIRONMENTS), default="synthetic", dest="environment",
                        help="synthetic (the default) or a GPU environment")
    parser.add_argument("--provider-profile", metavar="ID",
                        help="the model's Provider profile (ora providers lists them; its key from the environment "
                             "or ora providers key). Synthetic: instead of ORA_API_KEY and the ORA_* variables. "
                             "GPU task: default deepseek_flash_openai_low_16k_v1, e.g. "
                             "muse_spark_open_openrouter_high_16k_v1")
    agent_core.add_cli_option(parser)
    gpu = parser.add_argument_group("a GPU task (--env libero|robotwin|robodojo)")
    gpu.add_argument("--task", help="LIBERO: <suite>/<task id> (base or LIBERO-Pro core suite); RoboTwin and "
                     "RoboDojo: the task name (required)")
    gpu.add_argument("--seed", type=int, help="LIBERO initial state 0-49 (default 0); RoboTwin: an admitted "
                     "seed (default: the first seed the qualification accepted)")
    gpu.add_argument("--task-config", choices=("demo_randomized", "demo_clean"),
                     help="RoboTwin scene configuration (default demo_randomized)")
    gpu.add_argument("--candidates", type=int, help="RoboTwin seeds a new scan may try (default 4)")
    gpu.add_argument("--gpu", type=int, choices=range(8), metavar="N",
                     help="this machine's CUDA device index to run on, 0-7 (default 0): the simulator's slot and "
                          "the perception services ora run starts")
    gpu.add_argument("--requalify", action="store_true", help="run the qualification even if a record is reusable")
    gpu.add_argument("--yes", action="store_true", help="confirm qualification simulator episodes without asking")
    gpu.add_argument("--confirm-launch", action="store_true",
                     help="authorize launching the selected task after its normal preflight passes")
    gpu.add_argument("--prepare-only", action="store_true",
                     help="stop after composing the RunSpec and print its path")
    gpu.add_argument("--env-file", type=Path, help="GPU task only: the .env with its simulator, service and key "
                     "settings (default: the checkout's .env); Synthetic reads only the environment")
    gpu.add_argument("--emit", choices=("jsonl",), help="JSON-lines interactive backend instead of the Pi UI")
    gpu.add_argument("--poll", type=float, help="event polling interval in seconds (default 0.2)")
    add_world_options(gpu)


def prepare_environment(environment: str, *, env_file: Path,
                        environ: MutableMapping[str, str] | None = None,
                        provider_profile: str | None = None, gpu: int | None = None) -> dict[str, Any]:
    """Bind ``.env``, the host marker and Provider defaults into ``environ``; refuse missing inputs.

    With ``gpu`` (``ora run --gpu``) the perception services ``ora run`` starts run on that GPU too,
    whatever ``.env`` says."""
    from . import env_file as dotenv

    values = os.environ if environ is None else environ
    loaded = dotenv.load_into(values, env_file)
    from .local import provider_check, provider_environment  # imports paths: after .env

    marker = values.get(HOST_MARKER_ENV)
    if marker and marker != HOST:
        raise RuntimeError(f"ora run uses the shipped {HOST} host; {HOST_MARKER_ENV}={marker} names another "
                           "execution host (unset it or use that host's own workflow)")
    values[HOST_MARKER_ENV] = HOST
    if values.get("ORA_GPU_PREFIX"):
        # libGLU / the Vulkan loader ``ora setup`` unpacked without sudo; workers inherit it.
        from .gpu_recipes import add_library_path
        add_library_path(values, Path(values["ORA_GPU_PREFIX"]))
    values.update(provider_environment(values, provider_profile))
    check = provider_check(values, provider_profile)
    if not check["passed"]:
        detail = check["detail"]
        raise RuntimeError("Provider configuration is incomplete: set "
                           + " and ".join(detail.get("required_environment", []))
                           + (f" (base URL must be {detail['expected_endpoint']})" if "expected_endpoint" in detail
                              else "") + f", or store the key once: {ora_command()} providers key DEEPSEEK_LOW_API_KEY")
    from ora.agent.model_services.service_env import service_key, service_setting

    if gpu is not None:
        for service in ("SAM", "GRASP"):
            values[service_key(service, "GPU")] = str(gpu)
    settings = service_settings()
    missing = [key for key in required_keys(environment)
               if not (service_setting(*settings[key], environ=values) if key in settings
                       else values.get(key))]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} unset; run {ora_command()} setup --env {environment} "
                           f"(it writes these keys to {env_file})")
    return {"env_file_keys": loaded}


#: A RunSpec's ``runtime.kind`` and the ``ora run --env`` name its run cell has.
KIND_ENVIRONMENT = {"libero_pro": "libero", "robotwin": "robotwin", "robodojo": "robodojo"}


def bind_prepared_specs(specs: list[Path], *, env_file: Path | None = None, stream=None) -> bool:
    """Give ``ora launch`` / ``ora interactive --spec`` of a spec ``ora run --prepare-only`` wrote what ``ora run``
    itself would have: the checkout's ``.env``, the host marker and library path, and ready perception services.

    Only for specs on the shipped single-gpu host; any other spec is left to its host's own workflow. The
    perception services run on the spec's GPU (its ``<family>-gpu<N>`` slot). False when no spec is one."""
    import json

    for path in specs:
        try:
            spec = json.loads(Path(path).expanduser().read_text())
        except (OSError, ValueError):
            continue  # the launch path reports an unreadable spec itself
        runtime, policy = spec.get("runtime") or {}, spec.get("policy") or {}
        if runtime.get("required_host") != HOST or runtime.get("kind") not in KIND_ENVIRONMENT:
            continue
        slot = re.fullmatch(r".+-gpu(\d+)", str(runtime.get("resource_profile") or ""))
        gpu = int(slot.group(1)) if slot else None
        prepare_environment(KIND_ENVIRONMENT[runtime["kind"]], env_file=env_file or PROJECT_ROOT / ".env",
                            provider_profile=policy.get("provider_profile"), gpu=gpu)
        from .run_services import ensure_services

        ensure_services(stream=stream or sys.stderr)
        return True
    return False


def require_clean_checkout(root: Path = PROJECT_ROOT) -> None:
    """Qualification and launch both refuse a dirty tree; say so before minutes of work."""
    try:
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
                                capture_output=True, text=True, timeout=180, check=False,
                                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot inspect the Harness checkout: {exc}") from exc
    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("the Harness checkout has local changes or untracked files; qualification and launch "
                           "require a clean committed tree (commit or move them, then rerun)")


def check_gpu(index: int | None) -> None:
    """Refuse a ``--gpu`` index this machine's driver does not list."""
    if index is None:
        return
    try:
        listed = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                                text=True, timeout=60, check=True).stdout.split()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"--gpu {index}: cannot list this machine's GPUs with nvidia-smi: {exc}") from exc
    if str(index) not in listed:
        raise ValueError(f"--gpu {index}: this machine's GPUs are {', '.join(listed) or 'none'}")


def cohort(environment: str) -> str:
    """A fresh single-attempt cohort per ``ora run``, so rerunning a cell is a new admission."""
    return f"ora-run-{environment}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def compose(args: argparse.Namespace, *, services: Callable[[], Path], confirm, stream) -> Path:
    """Run the needed qualification, then bind the service report and write the RunSpec.

    The environment backend's run cell does both. The report is chosen after
    the qualification, which can take most of an hour, so a reused report
    cannot age out between selection and composition.
    """
    environment = args.environment
    common = {"experiment": EXPERIMENT, "cohort_id": cohort(environment),
              "provider_profile": getattr(args, "provider_profile", None)}
    return backend(environment).run_cell.compose(args, common=common, host=HOST, services=services,
                                                 confirm=confirm, stream=stream)


def interactive_arguments(args: argparse.Namespace, spec: Path, *, world_explicit: bool) -> list[str]:
    forwarded = ["interactive", "--spec", str(spec), "--poll", str(args.poll),
                 "--world-view", args.world_view, "--world-browser", args.world_browser]
    if world_explicit:
        forwarded += ["--world-port", str(args.world_port)]
    if args.world_public_url:
        forwarded += ["--world-public-url", args.world_public_url]
    if args.confirm_launch:
        forwarded.append("--confirm-launch")
    if args.emit:
        forwarded += ["--emit", args.emit]
    return forwarded


def gpu_task(args: argparse.Namespace) -> argparse.Namespace:
    """``args`` of ``ora run --env ENV``, with the GPU options' defaults filled in; refuses a missing --task."""

    if not args.task:
        raise ValueError(f"ora run --env {args.environment} needs --task")
    for dest, default in GPU_DEFAULTS.items():
        if getattr(args, dest) is None:
            setattr(args, dest, default)
    if args.env_file is None:
        args.env_file = PROJECT_ROOT / ".env"
    return args


def run(args: argparse.Namespace, *, stream=None) -> int:
    if args.environment == "synthetic":
        if misplaced := [option for dest, option in GPU_ONLY if getattr(args, dest) not in (None, False)]:
            raise ValueError(f"{', '.join(misplaced)} apply to ora run --env libero|robotwin|robodojo")
        from .quickstart import command_run

        return command_run(args)
    gpu_task(args)
    output = stream or sys.stderr
    try:
        prepare_environment(args.environment, env_file=args.env_file, provider_profile=args.provider_profile,
                            gpu=args.gpu)
        check_gpu(args.gpu)
        from .run_services import check_service_ports, confirm as ask, ensure_services

        require_clean_checkout()
        check_service_ports()  # before a qualification that can take most of an hour
        print(f"ora run: {args.environment} {args.task} on the {HOST} host, GPU {args.gpu or 0}", file=output,
              flush=True)
        spec = compose(args, services=lambda: ensure_services(stream=output),
                       confirm=lambda message: ask(message, assume_yes=args.yes, stream=output), stream=output)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"ora run: {exc}", file=sys.stderr)
        return 2
    print(f"RunSpec {spec}", file=output, flush=True)
    if args.prepare_only:
        print(spec)
        return 0
    from .main import main
    from .world import world_explicit

    return main(interactive_arguments(args, spec, world_explicit=world_explicit(args)))


__all__ = ["ENVIRONMENTS", "GPU_ONLY", "HOST", "bind_prepared_specs", "check_gpu", "compose", "gpu_task", "prepare_environment", "register",
           "require_clean_checkout", "required_keys", "run"]
