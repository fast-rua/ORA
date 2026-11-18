"""``ora setup``: install the local CPU simulator, or with ``--env`` the single-GPU stacks.

    ora setup [--root DIR] [--python PY] [--graphics MODE] [--json]
        the locked Harness, the agent core's Pi packages and terminal UI, pinned CPU
        LIBERO sources and assets and a separate simulator runtime (``local_setup``)
    ora setup --env libero|robotwin|robodojo|perception [--env ...] [--prefix DIR] [--cuda-home DIR]
              [--cuda-arch X.Y] [--env-file .env] [--dry-run] [--json]
        the pinned single-GPU simulator, CuRobo and perception stacks and their .env keys
        (``gpu_setup`` with ``gpu_recipes``)

Both are idempotent and start no simulator, service or Provider call.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

#: Options of one installation only, as (dest, option).
LOCAL_ONLY = (("root", "--root"), ("python", "--python"), ("graphics", "--graphics"))
GPU_ONLY = (("prefix", "--prefix"), ("cuda_home", "--cuda-home"), ("cuda_arch", "--cuda-arch"),
            ("env_file", "--env-file"), ("dry_run", "--dry-run"))


def register(sub: Any) -> None:
    from .gpu_setup import ENVIRONMENTS  # standard library only

    setup = sub.add_parser(
        "setup", help="install the pinned local CPU simulator and terminal UI; with "
                      "--env libero|robotwin|robodojo|perception, the single-GPU stacks",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    setup.add_argument("--json", action="store_true", help="print the report as JSON")
    local = setup.add_argument_group("the local CPU installation (without --env)")
    local.add_argument("--root", type=Path, help="local setup root (default: ORA_LOCAL_ROOT or the user data directory)")
    local.add_argument("--python", type=Path, help="existing Python 3.11 interpreter")
    local.add_argument("--graphics", choices=("auto", "cgl", "glfw", "egl", "osmesa"),
                       help="auto (the default): CGL on macOS; on Linux EGL when an NVIDIA EGL vendor file "
                            "exists, else OSMesa")
    gpu = setup.add_argument_group("the single-GPU stacks (--env)")
    gpu.add_argument("--env", action="append", choices=ENVIRONMENTS, dest="environments",
                     help="libero, robotwin and robodojo include the perception services (repeatable)")
    gpu.add_argument("--prefix", type=Path, help="persistent install directory (default: ORA_GPU_PREFIX "
                     "from the environment or .env, else ~/.local/share/ora/gpu)")
    gpu.add_argument("--cuda-home", type=Path, help="CUDA 12.8 toolkit for the CuRobo build "
                     "(default: CUDA_HOME, /usr/local/cuda-12.8, /usr/local/cuda)")
    gpu.add_argument("--cuda-arch", help="TORCH_CUDA_ARCH_LIST base for CuRobo, e.g. 12.0 "
                     "(default: nvidia-smi --query-gpu=compute_cap)")
    gpu.add_argument("--env-file", type=Path, help="the .env to update (default: the checkout's .env)")
    gpu.add_argument("--dry-run", action="store_true", help="list each step as done or pending; change nothing")


def _given(args: argparse.Namespace, options: tuple[tuple[str, str], ...]) -> list[str]:
    return [option for dest, option in options if getattr(args, dest) not in (None, False)]


def run(args: argparse.Namespace) -> int:
    if args.environments:
        if misplaced := _given(args, LOCAL_ONLY):
            raise ValueError(f"{', '.join(misplaced)} apply to the local CPU setup, not to ora setup --env")
        from .gpu_setup import command_setup

        return command_setup(args)
    if misplaced := _given(args, GPU_ONLY):
        raise ValueError(f"{', '.join(misplaced)} apply to ora setup --env only")
    from .local_setup import command_setup

    return command_setup(args)


__all__ = ["GPU_ONLY", "LOCAL_ONLY", "register", "run"]
