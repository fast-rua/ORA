"""A RoboTwin scene's bodies for the scene trace: stand-ins shaped like SAPIEN's objects (the reader goes by
component class name), so no simulator. The vertex rules were measured on SAPIEN 3.0.0b1 (each shape's world AABB)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.environment.backends.robotwin.scene_trace import RoboTwinSceneTrace, body_mesh, scene_bodies
from ora.environment.scene_trace import SceneTraceError

IDENTITY = SimpleNamespace(p=[0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])
QUARTER_ABOUT_Z = SimpleNamespace(p=[0.0, 0.0, 1.0], q=[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
UNIT_CUBE = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 1.0]])


class Part:
    def __init__(self, vertices=UNIT_CUBE, *, color=(1.0, 0.0, 0.0, 1.0), texture=None, uv=None) -> None:
        self.vertices = np.asarray(vertices, np.float32)
        self.triangles = np.array([[0, 1, 2]], np.uint32)
        self.material = SimpleNamespace(base_color=list(color), base_color_texture=texture)
        self._uv = uv

    def get_vertex_uv(self):
        return self._uv


class Shape:
    local_pose = IDENTITY

    def __init__(self, *parts, local_pose=None, **fields) -> None:
        self.parts = list(parts) or [Part()]
        if local_pose is not None:
            self.local_pose = local_pose
        self.__dict__.update(fields)


class RenderShapeBox(Shape):
    pass


class RenderShapeSphere(Shape):
    pass


class RenderShapeCylinder(Shape):
    pass


class RenderShapeCapsule(Shape):
    pass


class RenderShapeTriangleMesh(Shape):
    pass


class RenderBodyComponent:
    def __init__(self, *shapes) -> None:
        self.render_shapes = list(shapes)


class PhysxRigidStaticComponent:
    pass


class PhysxRigidDynamicComponent:
    pass


class PhysxArticulationLinkComponent:
    def __init__(self, articulation, entity) -> None:
        self.articulation, self.entity = articulation, entity


class RenderPointLightComponent:
    pass


class Entity:
    count = 0

    def __init__(self, name: str, *components, p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0)) -> None:
        Entity.count += 1
        self.name, self.per_scene_id, self.components = name, Entity.count, list(components)
        self.pose = SimpleNamespace(p=list(p), q=list(q))

    def get_components(self):
        return self.components


class Articulation:
    def __init__(self, name: str) -> None:
        self.name, self.links = name, []

    def add(self, entity_name: str, *components) -> Entity:
        entity = Entity(entity_name, *components)
        link = PhysxArticulationLinkComponent(self, entity)
        entity.components.append(link)
        self.links.append(link)
        return entity

    def get_links(self):
        return self.links


def _world(robot: Articulation, *entities) -> SimpleNamespace:
    return SimpleNamespace(robot=SimpleNamespace(left_entity=robot, right_entity=robot),
                           scene=SimpleNamespace(get_entities=lambda: list(entities), get_timestep=lambda: 0.004))


@pytest.mark.parametrize(("shape", "extent"), [
    (RenderShapeBox(half_size=[0.1, 0.2, 0.3]), [0.1, 0.2, 0.3]),
    (RenderShapeSphere(radius=0.25), [0.25, 0.25, 0.25]),
    (RenderShapeCylinder(radius=0.1, half_length=0.3), [0.3, 0.1, 0.1]),
    (RenderShapeCapsule(radius=0.1, half_length=0.3), [1.0, 1.0, 1.0]),     # already in metres
    (RenderShapeTriangleMesh(scale=[0.04, 0.04, 0.04]), [0.04, 0.04, 0.04]),
])
def test_each_shape_s_vertices_are_scaled_as_sapien_scales_them(shape, extent) -> None:
    mesh = body_mesh(RenderBodyComponent(shape))

    assert mesh.positions.max(axis=0) == pytest.approx(extent)


def test_a_shape_s_local_pose_places_it_in_the_body_s_frame() -> None:
    mesh = body_mesh(RenderBodyComponent(RenderShapeBox(Part([[1.0, 0.0, 0.0]] * 3), half_size=[2.0, 1.0, 1.0],
                                                        local_pose=QUARTER_ABOUT_Z)))

    assert mesh.positions[0] == pytest.approx([0.0, 2.0, 1.0])  # scaled first, then turned, then moved


def test_a_body_s_parts_join_into_one_mesh_with_their_own_colours() -> None:
    mesh = body_mesh(RenderBodyComponent(RenderShapeTriangleMesh(Part(color=(1, 0, 0, 1)), Part(color=(0, 0.5, 0, 1)),
                                                                 scale=[1.0, 1.0, 1.0])))

    assert mesh.faces.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert mesh.colors[0].tolist() == [255, 0, 0, 255] and mesh.colors[3].tolist() == [0, 128, 0, 255]
    assert body_mesh(RenderBodyComponent()) is None


def _texture(pixels: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(download=lambda: pixels)


def test_a_texture_is_sampled_at_the_vertices_and_turned_linear() -> None:
    pixels = np.zeros((2, 2, 4), np.uint8)
    pixels[0, 1, :3] = 255            # row 0, column 1
    pixels[1, 0, :3] = 188            # sRGB 188 is about 0.5 linear
    n = 256
    uv = np.tile([[0.75, 0.25]], (n, 1))
    uv[1] = [0.25, 0.75]
    vertices = np.zeros((n, 3))
    mesh = body_mesh(RenderBodyComponent(RenderShapeTriangleMesh(Part(vertices, texture=_texture(pixels), uv=uv),
                                                                 scale=[1.0, 1.0, 1.0])))

    assert mesh.colors[0].tolist() == [255, 255, 255, 255]
    assert mesh.colors[1, 0] == pytest.approx(128, abs=2)


def test_a_coarse_textured_part_takes_the_texture_s_mean() -> None:
    pixels = np.full((4, 4, 4), 255, np.uint8)
    pixels[:2, :, :3] = 0             # half black, half white: mean sRGB 0.5
    mesh = body_mesh(RenderBodyComponent(RenderShapeBox(Part(texture=_texture(pixels), uv=np.zeros((3, 2))),
                                                        half_size=[1.0, 1.0, 1.0])))

    assert mesh.colors[:, 0].tolist() == [mesh.colors[0, 0]] * 3
    assert mesh.colors[0, 0] == pytest.approx(55, abs=2)      # the mean of the picture, linear


def test_actors_links_and_the_robot_are_bodies_and_lights_and_cameras_are_not() -> None:
    box = RenderBodyComponent(RenderShapeBox(half_size=[1.0, 1.0, 1.0]))
    robot, switch = Articulation(""), Articulation("056_switch")
    bodies = {body.key: body for body in scene_bodies(_world(
        robot,
        Entity("table", PhysxRigidStaticComponent(), box),
        Entity("cup", PhysxRigidDynamicComponent(), box, p=(1.0, 2.0, 3.0), q=(0.5, 0.1, 0.2, 0.3)),
        Entity("cup", PhysxRigidDynamicComponent(), box),
        Entity("", RenderPointLightComponent()),
        Entity("head_camera", SimpleNamespace()),
        robot.add("fl_link1", box), robot.add("footprint"), switch.add("link_1", box),
    ))}

    assert set(bodies) == {"table", "cup", "cup#2", "robot/fl_link1", "robot/footprint", "056_switch/link_1"}
    assert {key: body.kind for key, body in bodies.items()} == {
        "table": "static", "cup": "object", "cup#2": "object", "robot/fl_link1": "robot", "robot/footprint": "robot",
        "056_switch/link_1": "object"}
    assert bodies["robot/footprint"].mesh is None and bodies["056_switch/link_1"].mesh is not None
    assert bodies["cup"].read_pose() == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.5]     # SAPIEN's wxyz as xyzw


class Scene:
    """A native scene's stand-in: ``PhysicsStepTap`` wraps its class's ``step``, as it does SAPIEN's."""

    def __init__(self, entities) -> None:
        self.entities = entities

    def get_entities(self):
        return self.entities

    def get_timestep(self):
        return 0.004

    def step(self) -> None:
        pass


def test_the_trace_counts_every_scene_step_from_each_round_s_reset(tmp_path: Path) -> None:
    robot = Articulation("")

    def world() -> SimpleNamespace:
        scene = Scene([Entity("cup", PhysxRigidDynamicComponent(), RenderBodyComponent(RenderShapeBox(half_size=[1.0] * 3)))])
        return SimpleNamespace(robot=SimpleNamespace(left_entity=robot, right_entity=robot), scene=scene)

    first, second = world(), world()
    trace = RoboTwinSceneTrace(tmp_path / "t", run_id="r")
    trace.begin_round(first, 0)
    trace.begin_action({"action_sequence": 0, "action_id": "a"})
    for _ in range(6):
        first.scene.step()
    trace.end_action()
    trace.begin_round(second, 1)
    first.scene.step()                          # the old scene is no longer watched
    trace.begin_action({"action_sequence": 1, "action_id": "b"})
    for _ in range(2):
        second.scene.step()
    trace.end_action()
    manifest = trace.finish()

    assert [(s["round"], s["first_step"], s["last_step"]) for s in manifest["spans"]] == [(0, 0, 6), (1, 0, 2)]
    assert manifest["clock"]["unit"] == "sapien_step_since_round_reset" and manifest["clock"]["hz"] == 250.0
    assert Scene.step.__qualname__ == "Scene.step"      # the tap is off the class again


def test_a_pose_the_tap_could_not_read_fails_the_trace(tmp_path: Path) -> None:
    robot = Articulation("")
    entity = Entity("cup", PhysxRigidDynamicComponent(), RenderBodyComponent(RenderShapeBox(half_size=[1.0] * 3)))
    world = SimpleNamespace(robot=SimpleNamespace(left_entity=robot, right_entity=robot), scene=Scene([entity]))
    trace = RoboTwinSceneTrace(tmp_path / "t", run_id="r")
    trace.begin_round(world, 0)
    del entity.pose
    for _ in range(5):
        world.scene.step()

    with pytest.raises(SceneTraceError, match="after a physics step"):
        trace.finish()
