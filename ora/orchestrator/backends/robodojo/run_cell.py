"""``ora run --env robodojo``: the card and task qualification records a one-round cell needs, reused when valid.

Reuse needs a matching task/card pair for the device and interpreter. The first
start on a GPU (no card record admitted for this GPU, driver and installation)
runs the card qualification once, after a confirmation: two full-length episodes
that check the card and warm the renderer's caches, without which a fresh
installation's first episode can time out compiling materials. A missing task
record does not trigger a task qualification: run preflight records a warning
and continues. Explicit ``--requalify`` retains the operator-requested card/task
qualification workflow. Live source, GPU and graphics checks still apply at launch.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.orchestrator.paths import PROJECT_ROOT, WORKSPACE_ROOT
from ora.orchestrator.progress import log_path, newest, run_with_progress

SLOT = "robodojo-gpu0"
GPU = "0"


def slot(gpu: int | None) -> str:
    """The single-gpu host's RoboDojo slot on GPU ``gpu`` (``ora run --gpu``; None is GPU 0)."""
    return f"robodojo-gpu{gpu or 0}"
#: The ``.env`` keys the cell runs on, beside the perception services'. Unset ROBODOJO_* silently
#: fall back to ~/robodojo-runtime in the runtime configuration. The cell runs the default policy, code mode, so it
#: needs the sandbox interpreter ``ora setup --env robodojo`` builds (ORA_CODE_MODE_PYTHON).
REQUIRED_ENV = ("ROBODOJO_ROOT", "ROBODOJO_ASSET_ROOT", "ROBODOJO_PYTHON", "CUROBO_V2_PYTHON", "CUROBO_V2_ROOT",
                "ORA_TIMING_PYTHON", "ORA_CODE_MODE_PYTHON")
EXAMPLE = "general_pickup"
#: Measured on the 2026-10-14 verification machine (RTX PRO 6000 Blackwell).
CARD_ESTIMATE = "about 9 min on an RTX PRO 6000; 35-56 min on slower cards"
TASK_ESTIMATE = "about 4 min"

Confirm = Callable[[str], None]


def _harness_commit() -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()


def _robodojo_runtime(task: str, gpu: str = GPU) -> dict[str, Any]:
    """The one-round runtime a composed RunSpec resolves to, as preflight compares it."""
    from ora.orchestrator.backends.robodojo.spec import qualification_runtime

    return qualification_runtime(task, gpu=gpu, resource_profile=slot(int(gpu)), rounds=1)


def live_identity(gpu: str = GPU) -> dict[str, Any]:
    """What launch preflight compares a RoboDojo record with: GPU ``gpu``'s UUID and the interpreter readback."""
    from ora.orchestrator.backends.robodojo.card_qualification import python_stack

    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                   text=True, timeout=60).splitlines()
    uuids = [row.split(", ")[1].strip() for row in rows if row.split(", ")[0].strip() == gpu]
    return {"gpu_uuid": uuids[0] if len(uuids) == 1 else None, "environment": python_stack()}


def _matches_live(record: Mapping[str, Any], identity: Mapping[str, Any] | None) -> bool:
    if identity is None:
        return True
    device = record.get("device_binding") or {}
    return (device.get("selected_gpu_uuid") == identity["gpu_uuid"]
            and record.get("environment") == identity["environment"])


def task_record_admitted(task: str, record: Path, *, commit: str,
                         identity: Mapping[str, Any] | None = None, gpu: str = GPU) -> Mapping[str, Any] | None:
    """The ``{card, task}`` binding of ``record`` when this checkout admits it for ``task``.

    Besides the offline admission, the record must name this GPU and this
    interpreter readback, which launch preflight compares live; a new card or a
    reinstalled RoboDojo environment therefore requalifies here, not at launch.
    """
    from ora.orchestrator.backends.robodojo.spec import layered_qualification
    from ora.orchestrator.robodojo_qualification import load_qualification

    try:
        binding = layered_qualification(record)
        runtime = {**_robodojo_runtime(task, gpu), "infrastructure_qualification": binding}
        loaded = load_qualification(runtime, workspace_root=WORKSPACE_ROOT, harness_commit=commit)
    except Exception:  # every refusal means "not reusable"; the producer decides next
        return None
    return binding if _matches_live(loaded, identity) else None


def card_record_admitted(task: str, record: Path, *, commit: str,
                         identity: Mapping[str, Any] | None = None, gpu: str = GPU) -> bool:
    from ora.orchestrator.backends.robodojo.lane_qualification import card_reference
    from ora.orchestrator.robodojo_qualification import load_card_qualification

    try:
        loaded = load_card_qualification(_robodojo_runtime(task, gpu), workspace_root=WORKSPACE_ROOT,
                                         harness_commit=commit, reference=card_reference(record))
    except Exception:
        return False
    return _matches_live(loaded, identity)


def _records(kind: str) -> list[Path]:
    from ora.orchestrator.backends.robodojo.runtime_config import OUTPUT_ROOT

    return newest((OUTPUT_ROOT / "qualification").glob(f"robodojo-lane-{kind}*/{kind}-qualification.json"))


def _produced(stdout: str, label: str) -> Path:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    path = Path(lines[-1]) if lines else None
    if path is None or not path.is_file():
        raise RuntimeError(f"{label} printed no record path")
    return path


def robodojo_qualification(task: str, *, requalify: bool, confirm: Confirm, stream=None,
                           gpu: str = GPU) -> Mapping[str, Any] | None:
    """The card/task qualification binding for a one-round ``task`` cell on GPU ``gpu``."""
    output = stream or sys.stderr
    commit = _harness_commit()
    identity = None if requalify else live_identity(gpu)
    if not requalify:
        for record in _records("task"):
            binding = task_record_admitted(task, record, commit=commit, identity=identity, gpu=gpu)
            if binding is not None:
                print(f"RoboDojo {task}: reusing task qualification {record}", file=output, flush=True)
                return binding
    card = None if requalify else next(
        (record for record in _records("card")
         if card_record_admitted(task, record, commit=commit, identity=identity, gpu=gpu)), None)
    from ora.orchestrator.backends.registry import backend

    tools = backend("robodojo").qualify_tools
    lane = [task, "--gpu", gpu]
    if not requalify:
        if card is None:
            confirm(f"RoboDojo: first start on GPU {gpu}: one-time card qualification for this GPU, driver and "
                    f"installation (two full-length {task} episodes; {CARD_ESTIMATE}). It also warms the "
                    "renderer's caches, which a fresh installation's first episode needs.")
            run_with_progress(tools["card"].command(sys.executable, *lane, "--resource-profile", slot(int(gpu)),
                                                    "--yes"), label="RoboDojo card qualification",
                              estimate=CARD_ESTIMATE, log=log_path("robodojo-card"), stream=output)
        else:
            print(f"RoboDojo: reusing card qualification {card}", file=output, flush=True)
        print(f"RoboDojo {task}: warning: no task qualification; continuing without one (--requalify runs it).",
              file=output, flush=True)
        return None
    if card is None:
        confirm(f"RoboDojo: one-time card qualification for this GPU, driver and installation (two full-length "
                f"{task} episodes; {CARD_ESTIMATE}), then the {task} task qualification "
                f"({TASK_ESTIMATE}).")
        stdout = run_with_progress(tools["card"].command(sys.executable, *lane, "--resource-profile",
                                                         slot(int(gpu)), "--yes"),
                                   label="RoboDojo card qualification", estimate=CARD_ESTIMATE,
                                   log=log_path("robodojo-card"), stream=output)
        card = _produced(stdout, "card qualification")
    else:
        print(f"RoboDojo: reusing card qualification {card}", file=output, flush=True)
        confirm(f"RoboDojo {task}: one-time task qualification on that card ({TASK_ESTIMATE}).")
    stdout = run_with_progress(tools["task"].command(sys.executable, *lane, "--resource-profile", slot(int(gpu)), "--card",
                                                     card, "--yes"), label=f"RoboDojo {task} task qualification",
                               estimate=TASK_ESTIMATE, log=log_path(f"robodojo-{task}-task"), stream=output)
    record = _produced(stdout, "task qualification")
    binding = task_record_admitted(task, record, commit=commit, gpu=gpu)
    if binding is None:
        raise RuntimeError(f"the new task qualification {record} is not admitted for {task}; "
                           "run ./scripts/ora plan preflight on a composed spec for the reason")
    return binding


class RoboDojoRunCell:
    """``registry.RunCell`` of RoboDojo."""

    slot = SLOT
    required_env = REQUIRED_ENV
    example = EXAMPLE

    def compose(self, args: argparse.Namespace, *, common: Mapping[str, Any], host: str,
                services: Callable[[], Path], confirm: Confirm, stream: Any) -> Path:
        from ora.orchestrator.backends.robodojo.spec import bind_services, build_spec, compose as write

        if args.seed not in (None, 0):
            raise ValueError("a RoboDojo cell runs official layout 0 of seed 0; --seed does not apply")
        gpu = str(getattr(args, "gpu", None) or 0)
        binding = robodojo_qualification(args.task, requalify=args.requalify, confirm=confirm, stream=stream, gpu=gpu)
        spec = build_spec(args.task, gpu=gpu, resource_profile=slot(int(gpu)),
                          infrastructure_qualification=binding, **common)
        return write(bind_services(spec, services()))


RUN_CELL = RoboDojoRunCell()

__all__ = ["EXAMPLE", "GPU", "REQUIRED_ENV", "RUN_CELL", "RoboDojoRunCell", "SLOT", "card_record_admitted",
           "live_identity", "robodojo_qualification", "task_record_admitted"]
