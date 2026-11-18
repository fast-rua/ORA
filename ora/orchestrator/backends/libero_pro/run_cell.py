"""``ora run --env libero``: a LIBERO or LIBERO-Pro cell; the pinned checkout is its qualification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping

SLOT = "libero-gpu0"


def slot(gpu: int | None) -> str:
    """The single-gpu host's LIBERO slot on GPU ``gpu`` (``ora run --gpu``; None is GPU 0)."""
    return f"libero-gpu{gpu or 0}"
#: The ``.env`` keys the cell runs on, beside the perception services'.
REQUIRED_ENV = ("LIBERO_PRO_PYTHON", "LIBERO_PRO_ROOT", "LIBERO_PRO_DATA_ROOT", "CUROBO_V2_ROOT", "CUROBO_V2_PYTHON",
                "ORA_TIMING_PYTHON")
EXAMPLE = "libero_goal/0"


def libero_cell(task: str) -> tuple[str, str, int]:
    """``(builder, suite, task_id)`` for ``<suite>/<id>`` or a bare suite (task 0)."""
    from ora.environment.benchmarks.libero_pro import BASE_SUITES, CORE_SUITES

    suite, _, number = task.partition("/")
    try:
        task_id = int(number) if number else 0
    except ValueError as exc:
        raise ValueError(f"LIBERO task must look like libero_goal/0, not {task!r}") from exc
    if suite in BASE_SUITES:
        return "libero", suite, task_id
    if suite in CORE_SUITES:
        return "libero_pro", suite, task_id
    raise ValueError(f"unknown LIBERO suite {suite!r}; base suites {list(BASE_SUITES)}, "
                     f"LIBERO-Pro core suites {list(CORE_SUITES)}")


class LiberoRunCell:
    """``registry.RunCell`` of LIBERO-Pro (``ora run --env libero``)."""

    slot = SLOT
    required_env = REQUIRED_ENV
    example = EXAMPLE

    def compose(self, args: argparse.Namespace, *, common: Mapping[str, Any], host: str,
                services: Callable[[], Path], confirm: Callable[[str], None], stream: Any) -> Path:
        from ora.orchestrator.spec.benchmark import build_libero_pro_spec, build_libero_spec, compose as write

        builder, suite, task_id = libero_cell(args.task)
        build = build_libero_spec if builder == "libero" else build_libero_pro_spec
        spec = build(suite, task_id, seed=0 if args.seed is None else args.seed,
                     resource_profile=slot(getattr(args, "gpu", None)), host=host, service_report=services(),
                     **common)
        return write(spec)


RUN_CELL = LiberoRunCell()

__all__ = ["EXAMPLE", "REQUIRED_ENV", "RUN_CELL", "SLOT", "LiberoRunCell", "libero_cell"]
