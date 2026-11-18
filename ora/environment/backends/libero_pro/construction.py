"""Construct the source-bound direct LIBERO driver inside the isolated worker."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from ora.environment.controllers.libero_osc import libero_osc_action_contract
from ora.environment.graphics_probe import bind_software_egl_device, configure_libero_graphics

#: The EGL device this process renders on, bound once by ``bind_egl_device``.
_EGL_DEVICE: dict[str, Any] | None = None


def bind_egl_device() -> dict[str, Any] | None:
    """Point a Mesa lane at Mesa's software EGL device, once per process, before LIBERO creates a context.

    Mesa lists every DRM render node before its software device, so device 0 is
    the software device only on a host without ``/dev/dri``. robosuite asserts
    at import that ``MUJOCO_EGL_DEVICE_ID`` is one of ``CUDA_VISIBLE_DEVICES``
    (a Mesa lane meets it with ``"K,0"``) and reads the variable again when it
    creates its first context, so robosuite is imported first and the device is
    bound in between. Returns the record (``graphics_probe.bind_software_egl_device``),
    or ``None`` when software rendering is not asked for.
    """

    global _EGL_DEVICE
    if _EGL_DEVICE is None:
        import robosuite.utils.binding_utils  # noqa: F401  runs robosuite's import-time device check first

        _EGL_DEVICE = bind_software_egl_device()
    return _EGL_DEVICE


def build_environment(
    *, libero_pro_root: Path, data_root: Path,
    runtime_config_root: Path, suite: str, task_id: int, seed: int,
    max_sim_steps: int, robustness_case: str | None = None,
    environment_driver: str = "direct_libero_v1",
) -> tuple[Any, dict[str, Any]]:
    """Construct the source-bound direct LIBERO driver."""
    from .session import (
        LiberoProError, _prepare_pinned_libero_imports, _git,
        libero_environment_init_params, _camera_metadata_env_factory,
        libero_public_environment_context,
    )

    configure_libero_graphics()
    runtime_bindings = _prepare_pinned_libero_imports(
        libero_pro_root=libero_pro_root, data_root=data_root,
        runtime_config_root=runtime_config_root, suite=suite,
    )
    from robosuite.controllers import load_controller_config
    import robosuite
    import robosuite.models.grippers.panda_gripper as panda_gripper_module
    try:
        egl_device = bind_egl_device()
    except (OSError, RuntimeError) as exc:
        raise LiberoProError(f"LIBERO software EGL device: {exc}") from exc
    from .direct import DRIVERS, PLAY_DRIVER, PRACTICE_DRIVER, DirectLiberoEnvironment, withhold_official_predicate
    from ora.environment.benchmarks.libero_pro import PRACTICE_SEED_FLOOR
    if environment_driver not in DRIVERS:
        raise LiberoProError("unsupported LIBERO environment driver")
    if environment_driver == PLAY_DRIVER and robustness_case is not None:
        raise LiberoProError("a LIBERO play scene loads no robustness case")
    practice = environment_driver == PRACTICE_DRIVER
    if practice and robustness_case is not None:
        raise LiberoProError("practice initial states are not qualified for robustness cases")
    if practice and seed < PRACTICE_SEED_FLOOR:
        raise LiberoProError(f"a practice episode needs a practice seed of at least {PRACTICE_SEED_FLOOR}")
    from libero.libero import benchmark
    runtime_bindings["environment_driver"] = environment_driver
    from .registration import ensure_perturbation_suites

    from ora.environment.benchmarks.libero_pro import CORE_SUITES

    if suite in CORE_SUITES:
        ensure_perturbation_suites(benchmark)

    benchmark_suite = benchmark.get_benchmark(suite)()
    if not 0 <= task_id < benchmark_suite.get_num_tasks():
        raise LiberoProError(f"task_id {task_id} is outside suite {suite}")
    trials = len(benchmark_suite.get_task_init_states(task_id))
    if trials <= 0:
        raise LiberoProError(f"suite {suite} task {task_id} has no init states")
    first_id = sum(
        len(benchmark_suite.get_task_init_states(index)) for index in range(task_id)
    )
    reset_id = first_id + (seed % trials)
    task = benchmark_suite.get_task(task_id)
    robustness_bddl: Path | None = None
    robustness_config: dict[str, Any] | None = None
    if robustness_case is not None:
        from .robustness import (
            ROBUSTNESS_CODE_REVISION,
            parse_and_validate_robustness_bddl,
            robustness_bddl_path,
        )

        actual_revision = _git(libero_pro_root, "rev-parse", "HEAD")
        if actual_revision != ROBUSTNESS_CODE_REVISION:
            raise LiberoProError(
                "BDDL robustness requires the paired LIBERO-Pro runtime: "
                f"expected {ROBUSTNESS_CODE_REVISION}, got {actual_revision}"
            )
        robustness_bddl = robustness_bddl_path(
            data_root, case=robustness_case, suite=suite,
            task_name=str(task.name),
        )
        robustness_config = parse_and_validate_robustness_bddl(
            robustness_bddl, case=robustness_case, suite=suite,
            task_name=str(task.name),
        )
    init_params = libero_environment_init_params(max_sim_steps=max_sim_steps)
    controller_action_contract = libero_osc_action_contract(
        load_controller_config(default_controller="OSC_POSE")
    )
    robosuite_root = Path(str(robosuite.__file__)).resolve().parent
    controller_config_file = robosuite_root / "controllers/config/osc_pose.json"
    panda_gripper_file = robosuite_root / "models/assets/grippers/panda_gripper.xml"
    for dependency in (controller_config_file, panda_gripper_file):
        if not dependency.is_file():
            raise LiberoProError(f"RoboSuite control dependency is unavailable: {dependency}")
    from libero.libero.envs import OffScreenRenderEnv
    active_bddl = robustness_bddl or Path(benchmark_suite.get_task_bddl_file_path(task_id))
    factory = lambda: OffScreenRenderEnv(
        bddl_file_name=str(active_bddl), **{**init_params, "ignore_done": True})
    if environment_driver == PLAY_DRIVER:
        scene_factory = factory
        factory = lambda: withhold_official_predicate(scene_factory())
    if robustness_bddl is not None:
        from .robustness import robustness_env_factory
        factory = robustness_env_factory(factory, case=str(robustness_case), suite=suite,
            task_name=str(task.name), bddl_path=robustness_bddl, seed=seed)
    environment = DirectLiberoEnvironment(
        factory=_camera_metadata_env_factory(factory),
        init_states=benchmark_suite.get_task_init_states(task_id),
        first_reset_id=first_id, reset_id=reset_id, seed=seed,
        task_language=str(task.language), max_sim_steps=max_sim_steps, practice=practice)
    public_environment_context = libero_public_environment_context(
        task, init_params=init_params,
    )
    init_states_path = (
        Path(benchmark.get_libero_path("init_states"))
        / suite / str(task.init_states_file)
    )
    active_bddl = (
        robustness_bddl
        if robustness_bddl is not None
        else Path(benchmark_suite.get_task_bddl_file_path(task_id))
    )
    robustness_runtime_module: Path | None = None
    robustness_parser_module: Path | None = None
    if robustness_case is not None:
        robustness_runtime_module = (
            libero_pro_root / "libero/libero/envs/robustness_perturbations.py"
        ).resolve()
        robustness_parser_module = (
            libero_pro_root / "libero/libero/envs/perturbation_config.py"
        ).resolve()
    return environment, {
        "suite": suite,
        "task_id": task_id,
        "task_language": str(task.language).strip(),
        "task_name": str(task.name),
        "bddl_file": str(active_bddl),
        "init_states_file": str(init_states_path),
        # A practice episode loads no official state, so it has no reset id.
        "reset_id": None if practice else reset_id,
        "trial_count": trials,
        "environment_driver": environment_driver,
        "egl_device": egl_device,
        "libero_pro_root": str(libero_pro_root),
        "data_root": str(data_root),
        "runtime_bindings": {
            **runtime_bindings,
            "direct_driver_module": str(Path(__file__).with_name("direct.py").resolve()),
            "robosuite_module": str(Path(robosuite.__file__).resolve()),
            "controller_config_file": str(controller_config_file.resolve()),
            "panda_gripper_module": str(
                Path(panda_gripper_module.__file__).resolve()
            ),
            "panda_gripper_file": str(panda_gripper_file.resolve()),
            **(
                {
                    "robustness_runtime_module": str(robustness_runtime_module),
                    "robustness_parser_module": str(robustness_parser_module),
                }
                if robustness_runtime_module is not None
                and robustness_parser_module is not None else {}
            ),
        },
        "controller_action_contract": controller_action_contract,
        "public_environment_context": public_environment_context,
        "robustness": (
            None if robustness_case is None else {
                "case": robustness_case,
                "config_sha256": hashlib.sha256(json.dumps(
                    robustness_config, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest(),
                "official_predicate_changed": False,
            }
        ),
    }
