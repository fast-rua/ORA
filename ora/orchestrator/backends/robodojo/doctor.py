"""What ``ora doctor --env robodojo`` checks beyond the interpreter: the Vulkan loader and its compatibility layer.

Read-only: the loader is probed in the simulator interpreter, and the
compatibility layer's receipt is checked by running the worker's own check by
path in a child process; nothing is started and no backend module is imported.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

from ora.orchestrator.backends.registry import DoctorProbe

ROBODOJO_GRAPHICS_DOCS = "docs/environments/robodojo.md#vulkan-loader-and-compatibility-layer"
#: Drivers whose Vulkan allocation limit overflows Isaac Sim 5.1 (robodojo_qualification.validate_runtime_driver).
COMPAT_LAYER_DRIVERS = ("595.",)
GRAPHICS_COMPAT_SCRIPT = Path(__file__).resolve().parents[3] / "environment/backends/robodojo/graphics_compat.py"
RECEIPT_CHECK = ("import importlib.util, sys; from pathlib import Path; "
                 "spec = importlib.util.spec_from_file_location('graphics_compat', sys.argv[1]); "
                 "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
                 "module.installation(Path(sys.argv[2]))")


def graphics_checks(interpreter: str, nvidia_smi: str | None) -> list[dict]:
    """Vulkan loader and, where the driver needs it, the compatibility-layer receipt.

    The loader is probed in the simulator interpreter (falling back to this one), which
    inherits ``LD_LIBRARY_PATH`` exactly as the isolated worker does. Nothing is started.
    """
    probe = "import ctypes; ctypes.CDLL('libvulkan.so.1')"
    try:
        if interpreter:
            loaded = subprocess.run([interpreter, "-c", probe], capture_output=True, timeout=15, env={
                k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}).returncode == 0
        else:
            ctypes.CDLL("libvulkan.so.1")
            loaded = True
    except (OSError, subprocess.TimeoutExpired):
        loaded = False
    checks = [{"check": "vulkan_loader", "passed": loaded, "detail": "libvulkan.so.1 loadable" if loaded else
               "libvulkan.so.1 is not loadable: install the Vulkan loader (e.g. libvulkan1) or add its "
               f"directory to LD_LIBRARY_PATH; see {ROBODOJO_GRAPHICS_DOCS}"}]
    driver = ""
    if nvidia_smi:
        try:
            result = subprocess.run([nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=5)
            driver = (result.stdout.strip().splitlines() or [""])[0].strip() if result.returncode == 0 else ""
        except (OSError, subprocess.TimeoutExpired):
            driver = ""
    if not driver.startswith(COMPAT_LAYER_DRIVERS):
        checks.append({"check": "vulkan_compatibility_layer", "passed": True,
                       "detail": f"not required by driver {driver or 'unknown'}"})
        return checks
    runtime = Path(interpreter or os.environ.get("ROBODOJO_PYTHON", "")).parent.parent.parent
    # The worker's own receipt check, run by path in a child: host layers never import backends.
    try:
        result = subprocess.run([sys.executable, "-c", RECEIPT_CHECK, str(GRAPHICS_COMPAT_SCRIPT), str(runtime)],
                                capture_output=True, text=True, timeout=30)
        ok = result.returncode == 0
        error = (result.stderr.strip().splitlines() or ["receipt check failed"])[-1]
    except (OSError, subprocess.TimeoutExpired) as exc:
        ok, error = False, f"{type(exc).__name__}: {exc}"
    detail = (f"receipt verified under {runtime / 'graphics-compat'} (driver {driver})" if ok else
              f"driver {driver} needs the Vulkan compatibility layer, but "
              f"{runtime / 'graphics-compat/installation.json'} is missing or invalid ({error}); "
              f"build it with `scripts/setup-robodojo graphics-compat`, see {ROBODOJO_GRAPHICS_DOCS}")
    checks.append({"check": "vulkan_compatibility_layer", "passed": ok, "detail": detail})
    return checks


DOCTOR = DoctorProbe(python_env="ROBODOJO_PYTHON", module="isaacsim",
                     directory_envs=("ROBODOJO_ROOT", "ROBODOJO_ASSET_ROOT"), extra_checks=graphics_checks)

__all__ = ["COMPAT_LAYER_DRIVERS", "DOCTOR", "GRAPHICS_COMPAT_SCRIPT", "RECEIPT_CHECK", "ROBODOJO_GRAPHICS_DOCS",
           "graphics_checks"]
