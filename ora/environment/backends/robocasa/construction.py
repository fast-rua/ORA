"""Build one RoboCasa365 environment inside the isolated worker interpreter."""

from __future__ import annotations

import importlib.metadata
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ora.environment.benchmarks.robocasa import (
    IMAGE_SIZE,
    PACKAGE_VERSIONS,
    PUBLIC_CAMERAS,
    REVISIONS,
    ROBOT,
    SPLIT,
    TASKS,
)
from ora.environment.controllers import robocasa_omron as C
from ora.environment.graphics_probe import bind_software_egl_device

from .common import RoboCasaError

#: The EGL device this process renders on, bound by the first ``build_environment`` (``egl_device``).
_EGL_DEVICE: dict[str, Any] | None = None


def egl_device() -> dict[str, Any] | None:
    """The software EGL device record (``graphics_probe.bind_software_egl_device``), once an environment is built."""

    return _EGL_DEVICE


def _git_head(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = root / ".git" / value[5:]
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
        packed = root / ".git" / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + value[5:]):
                    return line.split()[0]
        return None
    return value


def source_manifest() -> dict[str, Any]:
    """Verify the loaded robosuite/robocasa against the pinned revisions."""

    import robocasa
    import robosuite

    manifest: dict[str, Any] = {"packages": {}, "revisions": {}}
    for package in ("robocasa", "robosuite", "mujoco"):
        manifest["packages"][package] = importlib.metadata.version(package)
    if manifest["packages"] != PACKAGE_VERSIONS:
        raise RoboCasaError(
            f"RoboCasa package versions differ from the pins: {manifest['packages']}"
        )
    for name, module in (("robocasa", robocasa), ("robosuite", robosuite)):
        root = Path(module.__file__).resolve().parents[1]
        head = _git_head(root)
        manifest["revisions"][name] = {"root": str(root), "head": head}
        if head != REVISIONS[name]:
            raise RoboCasaError(
                f"{name} checkout {root} is at {head}, expected {REVISIONS[name]}"
            )
    return manifest


def _region_key(region: Mapping[str, Any]) -> str:
    return json.dumps({field: [float(value) for value in region[field]] for field in ("offset", "size")}, sort_keys=True)


@contextmanager
def region_order(recorded: Sequence[Sequence[str]] | None = None) -> Iterator[list[list[str]]]:
    """Keep the order counters list their regions in here, or hand back the order a run kept.

    The episode's seed does not fix a kitchen. ``Counter.get_reset_regions`` dedupes its top geoms with
    ``list(set(valid_geoms))``, and an XML element hashes by its address, so the regions come out in an order that
    changes between processes. The seeded generator then picks by index, and the same draw lands on another region: a
    stand mixer stands at either end of one island, whatever the hash seed. A run only reads the method's answer; a
    replay puts the same regions in the run's order, and refuses a counter that offers others. Each geom's regions are
    computed apart from the others', so the order of the answer is the whole of what ``set`` decided.
    """

    try:
        from robocasa.models.fixtures.counter import Counter
    except ImportError:  # a stand-in kitchen has no counters
        yield []
        return
    original = Counter.get_reset_regions
    supply = None if recorded is None else iter(recorded)
    kept: list[list[str]] = []

    def get_reset_regions(fixture: Any, *args: Any, **kwargs: Any) -> Any:
        regions = original(fixture, *args, **kwargs)
        keys = [_region_key(region) for region in regions.values()]
        if supply is not None:
            wanted = next(supply, None)
            if wanted is None or sorted(wanted) != sorted(keys):
                raise RoboCasaError("the replay's counter offers other regions than the run's")
            by_key = dict(zip(keys, regions.values()))
            regions, keys = dict(zip(regions, (by_key[key] for key in wanted))), list(wanted)
        kept.append(keys)
        return regions

    Counter.get_reset_regions = get_reset_regions
    try:
        yield kept
    finally:
        Counter.get_reset_regions = original
    if recorded is not None and len(kept) != len(recorded):
        raise RoboCasaError(f"the replay's counters listed regions {len(kept)} times, the run's {len(recorded)}")


def build_environment(*, task: str, seed: int) -> Any:
    """Create the target-split RoboCasa environment for one episode.

    The keyword set mirrors ``robocasa.utils.env_utils.create_env`` for
    ``split="target"`` (target object instances, layout/style pairs 1..10)
    except that camera observations are off (the session renders the public
    cameras only when it publishes an observation, so control steps cost
    physics only) and that the arm part of the composite controller is
    ``robocasa_omron.ARM_CONTROLLER`` (absolute joint targets) instead of the
    default OSC_POSE delta; base, torso and gripper parts stay as shipped.
    """

    if os.environ.get("MUJOCO_GL") not in {"egl", "osmesa"}:
        raise RoboCasaError("RoboCasa worker requires headless MUJOCO_GL (egl or osmesa)")
    import robocasa  # noqa: F401  registers the kitchen environments
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    # A Mesa lane renders on Mesa's software EGL device, which follows any DRM render node in Mesa's list. robosuite
    # asserts at import that MUJOCO_EGL_DEVICE_ID is one of CUDA_VISIBLE_DEVICES (a GPU rule a lane meets with "K,0")
    # and reads it again when it creates its first context, so the device is bound here, between the two.
    global _EGL_DEVICE
    if _EGL_DEVICE is None:
        try:
            _EGL_DEVICE = bind_software_egl_device()
        except (OSError, RuntimeError) as exc:
            raise RoboCasaError(f"RoboCasa software EGL device: {exc}") from exc

    if SPLIT != "target":
        raise RoboCasaError("only the target split is admitted")
    controller = load_composite_controller_config(controller=None, robot=ROBOT)
    shipped = controller["body_parts"]["right"]
    if shipped.get("type") != "OSC_POSE" or controller.get("type") != "HYBRID_MOBILE_BASE":
        raise RoboCasaError("PandaOmron default controller differs from the pinned robosuite")
    controller["body_parts"]["right"] = {**C.ARM_CONTROLLER, "gripper": shipped["gripper"]}
    return robosuite.make(
        env_name=task,
        robots=ROBOT,
        controller_configs=controller,
        camera_names=list(PUBLIC_CAMERAS.values()),
        camera_widths=IMAGE_SIZE,
        camera_heights=IMAGE_SIZE,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        horizon=TASKS[task]["horizon_control_steps"],
        use_object_obs=True,
        use_camera_obs=False,
        camera_depths=False,
        seed=int(seed),
        obj_instance_split="target",
        generative_textures=None,
        randomize_cameras=False,
        layout_and_style_ids=list(zip(range(1, 11), range(1, 11))),
        layout_ids=None,
        style_ids=None,
        translucent_robot=False,
    )
