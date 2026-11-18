"""Host-safe validation of selected LIBERO driver configuration."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping
from ora.environment.benchmarks.libero_pro import PLAY_DRIVER, PRACTICE_DRIVER, PRACTICE_SEED_FLOOR
from ora.orchestrator.spec.values import RunSpecError, _positive_int

if TYPE_CHECKING:
    from ora.agent.composition import ResolvedToolSurface


def resolve_runtime(
    runtime: dict[str, Any], policy: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface, *, task: str, seed: int,
    task_instruction: str,
) -> None:
    """The LIBERO-Pro runtime of a RunSpec: its driver configuration (spec.runtime)."""
    validate_libero_configuration(runtime, seed=seed)


def revision_label(runtime: Mapping[str, Any]) -> str:
    """The plan display's runtime revision: the driver and the LIBERO-Pro code and data pins."""
    return "/".join((
        runtime.get("environment_driver", "direct_libero_v1"),
        runtime["required_libero_pro_revision"][:12],
        runtime["required_data_revision"][:12],
    ))


def validate_libero_configuration(runtime: dict[str, Any], *, seed: int | None = None) -> None:
    if runtime["transport"] != "local_rpc":
        raise RunSpecError("libero_pro runtime.transport must be local_rpc")
    # The direct driver is the only LIBERO driver; a RunSpec that names none selects it.
    # Its play variant (a play episode's scene, play.binding.bind_play_runtime) withholds the predicate;
    # its practice variant samples the initial state from the seed instead of loading an official one.
    driver = runtime.get("environment_driver", "direct_libero_v1")
    if driver not in {"direct_libero_v1", PLAY_DRIVER, PRACTICE_DRIVER}:
        raise RunSpecError("runtime.environment_driver is unsupported")
    if driver == PRACTICE_DRIVER:
        if runtime.get("robustness_case") is not None:
            raise RunSpecError("practice initial states are not qualified for robustness cases")
        if seed is not None and seed < PRACTICE_SEED_FLOOR:
            raise RunSpecError(f"a LIBERO practice episode needs a seed of at least {PRACTICE_SEED_FLOOR}")
    if any(key in runtime for key in ("rlinf_root_env", "required_rlinf_revision")):
        raise RunSpecError("the direct LIBERO runtime binds no external driver checkout")
    for field in (
        "python_env", "libero_pro_root_env", "data_root_env",
        "required_libero_pro_revision", "required_data_revision", "suite",
        "expected_task_language",
    ):
        if not isinstance(runtime.get(field), str) or not runtime[field]:
            raise RunSpecError(f"runtime.{field} must be non-empty")
    task_id = runtime.get("task_id")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
        raise RunSpecError("runtime.task_id must be a non-negative integer")
    runtime.setdefault("main_camera", "agentview")
    if runtime["main_camera"] != "agentview":
        raise RunSpecError("libero_pro runtime.main_camera must be agentview")
    runtime.setdefault("startup_timeout_s", 600)
    runtime.setdefault("request_timeout_s", 1800)
    runtime.setdefault("servo_max_steps", 80)
    for field in ("startup_timeout_s", "request_timeout_s", "servo_max_steps"):
        runtime[field] = _positive_int(runtime[field], f"runtime.{field}")
