"""A RoboDojo scene's trace (``ora/environment/scene_trace.py``) over a control replay (``control_replay.py``).

The Isaac part is ``isaac_scene.py``; this says where RoboDojo keeps its things. The bodies are the rigid prims under
the env root, the two arms' links (``robot``) and the layout's instances without a rigid body (``static``). A row is
one native action, 10 physics steps: the replay reads the scene between ticks only. ``physics_step`` counts PhysX steps
since the round reset, the execution capture's ``sim_step``, so a row at ``n`` physics steps is native action ``n / 10``
for the dashboard's clock. Not a kernel file: the worker never imports it, only ``control_replay --scene-trace`` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ora.environment.benchmarks.robodojo import PHYSICS_DT_S
from ora.environment.scene_trace import SceneTraceWriter

from .isaac_scene import RigidPoses, scene_bodies

CLOCK_UNIT = "physx_step_since_round_reset"
#: One row per native action: 10 PhysX steps at 250 Hz, 25 Hz.
STRIDE = 10
#: The layout's instance kinds with no rigid body; every other prim that has one is found on the stage.
STATIC_KINDS = frozenset({"geometry", "table", "room", "ground"})


def static_prims(stage: Any, root: str, instances: Mapping[str, str]) -> list[str]:
    """The prims of the layout's static instances, found by name: only these have no rigid body to find them by."""
    from pxr import Usd

    names = {name for name, kind in instances.items() if kind in STATIC_KINDS}
    return [str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath(root)) if prim.GetName() in names]


def bodies_of(session: Any) -> tuple[list[Any], RigidPoses]:
    """The round's bodies and their poses, from the session's live stage."""
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager

    stage = omni.usd.get_context().get_stage()
    scene = session.environment.scene_manager
    root = scene.env_roots[0]
    robots = [str(arm.root_physx_view.prim_paths[0]) for arm in session.environment.robot_manager.robot_key]
    return scene_bodies(stage, root, SimulationManager.get_physics_sim_view(), robots=robots,
                        statics=static_prims(stage, root, scene.layout_manager.instance_type_by_env[0]),
                        origin=scene.env_origins[0])


class RoboDojoSceneTrace:
    """What ``control_replay`` calls around a replay; it writes the trace, and reading the scene changes no state."""

    def __init__(self, directory: Path, *, run_id: str) -> None:
        self.writer = SceneTraceWriter(directory, run_id=run_id, backend="robodojo", clock_unit=CLOCK_UNIT,
                                       stride=STRIDE)
        self.poses: RigidPoses | None = None

    def begin_round(self, session: Any, round_index: int) -> None:
        bodies, self.poses = bodies_of(session)
        self.writer.begin_round(round_index, bodies, timestep_s=PHYSICS_DT_S)

    def begin_action(self, action: Mapping[str, Any]) -> None:
        self.writer.begin_action({"action_sequence": int(action["sequence"]), "action_id": action["action_id"]})

    def after_tick(self, session: Any) -> None:
        """The tick's steps on the capture's clock; the read happens at the row, after its last step."""
        self.poses.refresh()
        while self.writer.physics_step < session._clock() - session.reset_physics_steps:
            self.writer.after_step()

    def end_action(self) -> None:
        self.writer.end_action()

    def finish(self) -> dict[str, Any]:
        return self.writer.finish()


__all__ = ["CLOCK_UNIT", "STATIC_KINDS", "STRIDE", "RoboDojoSceneTrace", "bodies_of", "static_prims"]
