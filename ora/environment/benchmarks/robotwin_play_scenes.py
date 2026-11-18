"""Host-safe registry of ORA's own RoboTwin free-play scenes: tables with many objects, no task.

A scene set is a named group of scenes ORA owns. Each scene is a RoboTwin task
class (a ``Base_Task`` subclass) that places objects from RoboTwin's asset
library, and RoboTwin's own box helper where a scene needs plain blocks, on
the standard table, with the standard ALOHA embodiment and cameras. It has no
predicate (``check_success`` is always false), no expert and no instruction:
it exists only to be played in. The isolated RoboTwin worker loads it in place
of ``envs.{task}`` when the plan's runtime names the set
(``runtime.play_scene_set``, bound only by ``play.binding.bind_play_runtime`` in
``ora/orchestrator``); the RoboTwin checkout and its pinned revision are not
changed.

This module is host-safe: it names the scenes, their assets and the ORA files
that define them, and never imports the simulator. The scene classes live in
the worker-only package ``ora/environment/backends/robotwin/play_scenes/``
(``base.py``, ``tabletop_v1.py``), whose modules import RoboTwin's ``envs``
package; only ``scene_instance``, called in the worker, imports them. A host offers a scene only when a local play-scene audit
(``robotwin_play_scene_audit``, ``ora/orchestrator/backends/robotwin/play_audit.py``)
admits its seed for the current scene sources and the pinned revision.

``tabletop-v1`` rebuilds three of the six lost ``robotwin-long-horizon-v1``
prototypes from their asset lists and camera images: the layouts are like the
prototypes', not identical to them, and each seed jitters them.

``household-v1`` sets a breakfast table and a kitchen counter from the asset
library, each with containers to put things in and nine or eight objects at
household masses (``play_scenes/household_v1.py``); a set of its own, so
tabletop-v1's source binding and audits are untouched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The ORA repository root (this module is ``ora/environment/benchmarks/robotwin_play_scenes.py``).
ROOT = Path(__file__).resolve().parents[3]
SOURCE_SCHEMA = "ora.robotwin-play-scene-source.v1"
#: The worker module every scene set's classes derive from.
BASE_MODULE = "ora.environment.backends.robotwin.play_scenes.base"
#: Height of RoboTwin's table top in the world frame (``create_table_and_wall``).
TABLE_TOP_Z_M = 0.74
#: Where an audited object must rest: the envelope of the positions the 50
#: official tasks sample their objects from (``rand_pose`` limits in
#: ``envs/*.py`` at the pinned revision), which both ALOHA arms reach.
PLAY_WORKSPACE_XY_M = {"x": (-0.40, 0.40), "y": (-0.30, 0.25)}
#: A scene-set play plan's ``runtime.execution_capture``: the capture's highest rate, so the operator's page
#: and a video of the run show the motion (an object knocked over, falling, rolling) rather than a few jumps.
#: The capture samples simulated time on RoboTwin's 4 ms physics grid, so 30 fps is a frame every ninth
#: physics step (27.8 per simulated second). It records the operator camera beside the head and front
#: cameras (``play_scenes/base.py``). Benchmark plans, and play on an official task, keep the default 10 fps.
PLAY_EXECUTION_CAPTURE = {"enabled": True, "fps": 30}
#: The post-action settle of a scene-set play scene (``backends/robotwin/play_settle.py``): after each action
#: the scene steps on, every drive target held, ``check_interval_physics_steps`` at a time, until over one
#: such window no dynamic object moved more than ``max_displacement_m`` or turned more than
#: ``max_rotation_deg``, at most ``max_physics_steps`` (2 s of simulated time at 4 ms). An object more than
#: ``off_table_drop_m`` below the table top (within 0.3 m of the floor) has fallen off the table: its fall is
#: in the settle, and it no longer holds one open while it rolls on the floor.
PLAY_POST_ACTION_SETTLE = {"check_interval_physics_steps": 10, "max_physics_steps": 500,
                           "max_displacement_m": 0.001, "max_rotation_deg": 1.0, "off_table_drop_m": 0.44}


class PlaySceneSetError(ValueError):
    """An unknown scene set or scene."""


@dataclass(frozen=True)
class SceneDefinition:
    """One free-play scene: what loads, for the operator and the audit; never policy input."""

    scene_set: str
    name: str
    label: str
    module: str
    task_config: str
    assets: tuple[str, ...]
    prototype: str
    description: str

    @property
    def class_name(self) -> str:
        """The scene's task class, named like a RoboTwin task module's class."""
        return self.name


_TABLETOP = "ora.environment.backends.robotwin.play_scenes.tabletop_v1"
_HOUSEHOLD = "ora.environment.backends.robotwin.play_scenes.household_v1"
_PROTOTYPES = "agentic-robot-benchmark archive/prototypes/robotwin-long-horizon-v1/scenes"
SCENE_SETS: dict[str, tuple[SceneDefinition, ...]] = {
    "tabletop-v1": (
        SceneDefinition(
            "tabletop-v1", "table_setting_sequence", "table setting", _TABLETOP, "demo_clean",
            ("003_plate/base0", "002_bowl/base3", "021_cup/base0", "033_fork/base0", "034_knife/base0",
             "019_coaster/base0"),
            f"{_PROTOTYPES}/table_setting_sequence",
            "A plate, a bowl and a cup in a row near the robot; a fork, a knife and a coaster beyond them."),
        SceneDefinition(
            "tabletop-v1", "shell_game_memory_3", "three cups and a ball", _TABLETOP, "demo_clean",
            ("021_cup/base0 x3", "027_table-tennis/base1"),
            f"{_PROTOTYPES}/shell_game_memory_3",
            "Three upright cups in a row and an orange ball in front of one of them."),
        SceneDefinition(
            "tabletop-v1", "obstacle_rearrangement_6", "blocks, walls and pads", _TABLETOP, "demo_clean",
            ("RoboTwin create_box: 6 blocks, 6 pads, 3 walls",),
            f"{_PROTOTYPES}/obstacle_rearrangement_6",
            "Six coloured blocks near the robot, two low walls with a gap and a divider, "
            "and six matching flat pads beyond the walls."),
    ),
    "household-v1": (
        SceneDefinition(
            "household-v1", "breakfast", "breakfast table", _HOUSEHOLD, "demo_clean",
            ("003_plate/base0", "076_breadbasket/base0", "002_bowl/base7", "031_jam-jar/base1", "035_apple/base0",
             "075_bread/base4", "039_mug/base3", "068_boxdrink/base3", "113_coffee-box/base1"),
            "none: ORA's own layout",
            "A plate, a bread basket and a bowl at the back; a loaf of bread, a mug, a bottle of juice and a box of "
            "coffee in front; a jar of jam on the robot's left and an apple on its right."),
        SceneDefinition(
            "household-v1", "kitchen", "kitchen counter", _HOUSEHOLD, "demo_clean",
            ("104_board/base0", "067_steamer/base1", "091_kettle/base3", "105_sauce-can/base0", "069_vagetable/base3",
             "065_soy-sauce/base4", "069_vagetable/base1", "062_plasticbox/base3"),
            "none: ORA's own layout",
            "A cutting board, a bamboo steamer and a red kettle at the back; a tomato, a bottle of soy sauce, an "
            "onion and a blue tub in front; a jar of sauce on the robot's left."),
    ),
}


def scene_sets() -> tuple[str, ...]:
    return tuple(sorted(SCENE_SETS))


def scene_definitions(scene_set: str) -> tuple[SceneDefinition, ...]:
    try:
        return SCENE_SETS[scene_set]
    except KeyError:
        raise PlaySceneSetError(f"unknown play scene set {scene_set!r}; one of {list(scene_sets())}") from None


def scene_definition(scene_set: str, name: str) -> SceneDefinition:
    for definition in scene_definitions(scene_set):
        if definition.name == name:
            return definition
    raise PlaySceneSetError(f"play scene set {scene_set} has no scene {name!r}")


def find_scene(name: str) -> SceneDefinition | None:
    """The scene of that name in any set (names are unique across sets), or None."""
    for definitions in SCENE_SETS.values():
        for definition in definitions:
            if definition.name == name:
                return definition
    return None


def source_files(scene_set: str) -> tuple[Path, ...]:
    """The ORA files whose bytes define the set's scenes: the shared base and the set's module."""
    modules = {BASE_MODULE, *(definition.module for definition in scene_definitions(scene_set))}
    return tuple(sorted(ROOT / (module.replace(".", "/") + ".py") for module in modules))


def source_binding(scene_set: str, *, root: Path = ROOT) -> dict[str, Any]:
    """``{schema, scene_set, files: [{path, sha256}], sha256}`` of the set's scene sources.

    An audit records it and admission compares it with the files as they are
    now, so a changed scene module needs a new audit; any other change to ORA
    leaves an audit valid.
    """
    files = []
    for path in source_files(scene_set):
        relative = path.relative_to(ROOT)
        files.append({"path": relative.as_posix(),
                      "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest()})
    body = {"schema": SOURCE_SCHEMA, "scene_set": scene_set, "files": files}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**body, "sha256": digest}


def scene_instance(scene_set: str, name: str) -> Any:
    """A fresh scene task object (worker only: imports RoboTwin's ``envs``)."""
    import importlib

    definition = scene_definition(scene_set, name)
    return getattr(importlib.import_module(definition.module), definition.class_name)()


__all__ = [
    "PLAY_EXECUTION_CAPTURE", "PLAY_POST_ACTION_SETTLE", "PLAY_WORKSPACE_XY_M", "PlaySceneSetError", "SCENE_SETS",
    "SceneDefinition", "TABLE_TOP_Z_M", "find_scene", "scene_definition", "scene_definitions", "scene_instance",
    "scene_sets", "source_binding", "source_files",
]
