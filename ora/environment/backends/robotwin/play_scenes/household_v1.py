"""``household-v1``: a breakfast table and a kitchen counter set from RoboTwin's asset library.

Worker only (imports RoboTwin's ``envs``). Frame: RoboTwin's world, the table
centred at the origin, ``x`` to the robot's right, ``y`` away from the robot,
the table top at z 0.74 m; the arms' bases stand at y -0.65. Every seed
jitters the layout; nothing here is a task.

Each scene has containers (a plate, a basket and a bowl; a cutting board, a
steamer and a tub) and graspable things, the big ones in a back row at y about
-0.04 and the small ones in a front row at y about -0.24 and at the sides, 15
cm or more apart, so an arm carrying one has a lane between the others. Under
the grippers' home poses (x ±0.30, y -0.19, z 0.94) stand only objects lower
than 12 cm, so a resting arm touches nothing. Every object rests on the
table, its centre where both arms (in the middle) or one arm (at a side)
reach it from above: the read-only top-down reach map of the ALOHA arms at
the pinned revision covers, at grasp height, y up to about 0 at the centre and
0.10 at x ±0.3, and y down to the robot's edge (measured 2026-10-21).
The small objects are at most 8 cm across where a gripper closes on them;
nothing is thin or flat.

Physics. RoboTwin builds a library object at 10 g (``envs.utils.Actor``) and
keeps the rotational inertia PhysX computed for its shape at 1000 kg/m^3, so
its inertia is tens of times what its mass implies: pushed, it slides and
spins slowly rather than tipping, and a jaw or arm that pinches it can throw
it across the room. A household object here is built from the same mesh and
collision files with a household mass (``mass_kg``) and the inertia of that
mass at the shape's uniform density (PhysX's inertia, scaled by the mass
ratio), and with a static/dynamic friction of 0.6/0.5 (RoboTwin's shapes
default to 0.3/0.3). Nothing global changes: no depenetration or velocity
limit, the robot and the table keep RoboTwin's values. In scripted sweeps on
both scenes (2026-10-21) nothing left the table; a milk carton (since
replaced) the arm hit from above slid 3.5 cm at no more than 0.76 m/s and the
arm stopped on the contact, where at RoboTwin's 10 g it was pushed 9 cm at
1.29 m/s. Jaws that close on a thin part and open again can still throw an
object: no mass prevents a deep penetration from resolving fast. The scenes
therefore hold nothing with a thin fin or blade to pinch.

An asset is built at its model data's calibrated scale, else, as RoboTwin's
``create_actor`` does, in its mesh's own units; the assets whose meshes are
about 2 units across with no calibrated scale (the jam jar, the steamer, the
tomato, the onion and the cutting board) are placed at an explicit scale
(``EXPLICIT_SCALES``) chosen for their real size.

The operator camera frames each scene's whole table and both arms
(``operator_view``, 1280 x 720, as ``base.OPERATOR_VIEW`` for tabletop-v1).
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import sapien

from envs.utils import Actor  # type: ignore[import-not-found]

from ora.environment.benchmarks.robotwin_play_scenes import TABLE_TOP_Z_M
from .base import (
    DROP_CLEARANCE_M, OPERATOR_CAMERA, OPERATOR_VIEW, PlayObject, PlaySceneTask, Y_UP, _model_files, jitter,
    look_at_pose, quaternion_multiply, rigid_component, yaw_quaternion,
)

#: Scale of library assets whose model data carries none (their meshes are about 2 units across): the
#: object's size at that scale, measured on the built collision body, is in the comment.
EXPLICIT_SCALES: Mapping[tuple[str, int], float] = {
    ("031_jam-jar", 1): 0.0442,     # strawberry jam jar, 5.8 cm across, 8.5 cm tall
    ("067_steamer", 1): 0.0892,     # bamboo steamer, 17 cm across, 7.5 cm tall
    ("069_vagetable", 1): 0.0392,   # onion, 6.6 cm across
    ("069_vagetable", 3): 0.0362,   # tomato, 6.4 cm across
    ("104_board", 0): 0.1450,       # cutting board, 28 x 18.5 cm, 2.7 cm thick
}
#: Static and dynamic friction of a household object's collision shapes.
OBJECT_FRICTION = (0.6, 0.5)
#: The density PhysX computes a built body's mass and inertia at (SAPIEN's builder default).
BUILD_DENSITY_KG_M3 = 1000.0


class HouseholdScene(PlaySceneTask):
    """A play scene of library objects at household masses, with its own operator-camera framing."""

    operator_view: Mapping[str, Any] = OPERATOR_VIEW

    def _add_operator_camera(self) -> Any:
        view = {**OPERATOR_VIEW, **self.operator_view}
        camera = self.scene.add_camera(name=OPERATOR_CAMERA, width=int(view["width"]), height=int(view["height"]),
                                       fovy=math.radians(float(view["fovy_deg"])), near=float(view["near_m"]),
                                       far=float(view["far_m"]))
        camera.set_property("spp", int(view["samples_per_pixel"]))
        camera.entity.set_pose(look_at_pose(view["eye"], view["target"], view["up"]))
        return camera

    def place(self, name: str, model: str, model_id: int, xy: Sequence[float], *, mass_kg: float,
              yaw: float = 0.0, upright: Sequence[float] = Y_UP) -> PlayObject:
        """A library model at ``mass_kg`` resting on the table at ``xy``, turned by ``yaw`` about the vertical."""
        collision, visual, data = _model_files(model, model_id)
        model_data = json.loads(data.read_text(encoding="utf-8")) if data.exists() else {}
        explicit = EXPLICIT_SCALES.get((model, model_id))
        # As create_actor: the model data's calibrated scale, else the mesh's own units.
        scales = ([float(explicit)] * 3 if explicit is not None
                  else [float(value) for value in model_data.get("scale", (1.0, 1.0, 1.0))])
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        material = self.scene.create_physical_material(OBJECT_FRICTION[0], OBJECT_FRICTION[1], 0.0)
        builder.add_multiple_convex_collisions_from_file(filename=str(collision), scale=scales, material=material,
                                                        density=BUILD_DENSITY_KG_M3)
        builder.add_visual_from_file(filename=str(visual), scale=scales)
        entity = builder.build(name=model)
        orientation = quaternion_multiply(yaw_quaternion(yaw), upright)
        entity.set_pose(sapien.Pose([float(xy[0]), float(xy[1]), TABLE_TOP_Z_M + 0.3], list(orientation)))
        body = rigid_component(entity)
        built_mass, built_inertia = float(body.mass), np.asarray(body.inertia, dtype=float)
        actor = Actor(entity, {**model_data, "scale": scales}, mass=float(mass_kg))
        # The household mass at the shape's uniform density: PhysX's inertia scaled by the same ratio.
        body.set_inertia(built_inertia * (float(mass_kg) / built_mass))
        actor.set_name(name)
        self._rest_on_table(entity, xy, clearance=DROP_CLEARANCE_M)
        files = tuple(sorted({path.as_posix() for path in (collision, visual, data) if path.exists()}))
        return self._register(PlayObject(name, actor, f"{model}/base{model_id}", True, files))


def _xy(x: float, y: float, spread: float = 0.012) -> tuple[float, float]:
    return (x + jitter(spread), y + jitter(spread))


class breakfast(HouseholdScene):
    """A plate, a bread basket and a bowl at the back; bread, jam, a mug, juice, coffee and an apple in front."""

    operator_view = {"eye": (0.60, 0.50, 1.40), "target": (0.0, -0.20, 0.84), "fovy_deg": 46.0}

    def build_scene(self) -> None:
        self.place("plate", "003_plate", 0, _xy(-0.22, -0.05), mass_kg=0.35)
        self.place("bread_basket", "076_breadbasket", 0, _xy(0.0, -0.03), mass_kg=0.25, yaw=jitter(0.15))
        self.place("bowl", "002_bowl", 7, _xy(0.22, -0.05), mass_kg=0.25)
        self.place("jam_jar", "031_jam-jar", 1, _xy(-0.37, -0.15), mass_kg=0.30, yaw=jitter(np.pi))
        self.place("apple", "035_apple", 0, _xy(0.37, -0.15), mass_kg=0.18, yaw=jitter(np.pi))
        self.place("bread", "075_bread", 4, _xy(-0.24, -0.24), mass_kg=0.15, yaw=jitter(0.3))
        self.place("mug", "039_mug", 3, _xy(-0.08, -0.23), mass_kg=0.30, yaw=jitter(np.pi))
        # A slim juice bottle, not RoboTwin's milk carton: in the first breakfast session the jaws closed on the
        # carton's sealed top fin and, opening, threw the carton off the table (ora-play-play-0a94e1a6c96f).
        self.place("juice_bottle", "068_boxdrink", 3, _xy(0.08, -0.24), mass_kg=0.30, yaw=jitter(0.4))
        self.place("coffee_box", "113_coffee-box", 1, _xy(0.25, -0.23), mass_kg=0.20, yaw=jitter(0.4))


class kitchen(HouseholdScene):
    """A cutting board, a steamer and a kettle at the back; a tomato, soy sauce, an onion and a tub in front; a jar."""

    operator_view = {"eye": (0.60, 0.50, 1.40), "target": (0.0, -0.20, 0.84), "fovy_deg": 46.0}

    def build_scene(self) -> None:
        self.place("cutting_board", "104_board", 0, _xy(-0.22, -0.05, 0.008), mass_kg=0.60)
        self.place("steamer", "067_steamer", 1, _xy(0.03, -0.03, 0.008), mass_kg=0.30, yaw=jitter(np.pi))
        self.place("kettle", "091_kettle", 3, _xy(0.27, -0.05, 0.008), mass_kg=0.60, yaw=jitter(0.3))
        self.place("sauce_jar", "105_sauce-can", 0, _xy(-0.38, -0.18), mass_kg=0.30, yaw=jitter(np.pi))
        self.place("tomato", "069_vagetable", 3, _xy(-0.26, -0.24), mass_kg=0.12, yaw=jitter(np.pi))
        self.place("soy_sauce", "065_soy-sauce", 4, _xy(-0.10, -0.23), mass_kg=0.25, yaw=jitter(np.pi))
        self.place("onion", "069_vagetable", 1, _xy(0.06, -0.24), mass_kg=0.15, yaw=jitter(np.pi))
        self.place("tub", "062_plasticbox", 3, _xy(0.27, -0.24, 0.008), mass_kg=0.20, yaw=jitter(0.15))


__all__ = ["BUILD_DENSITY_KG_M3", "EXPLICIT_SCALES", "HouseholdScene", "OBJECT_FRICTION", "breakfast", "kitchen"]
