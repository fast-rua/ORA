"""A RoboTwin scene's bodies for the scene trace (``ora/environment/scene_trace.py``), read from SAPIEN.

Runs in the replay worker around an episode's re-execution. A body is a scene entity with a rigid body or
an articulation link component: the task's actors and clutter (RoboTwin's ``create_*`` loads them from
``assets/objects``), the table, wall and ground, each articulated object's links and the aloha-agilex's
links. Lights and cameras are not bodies. A body's visual geometry is its ``RenderBodyComponent``'s shapes:
each shape's parts hold unit or unscaled vertices, placed by the shape's scale and ``local_pose`` into the
entity's frame. A part's colour is its base-colour texture sampled at the vertices, or the texture's mean on
a part too coarse for that (a box has only its corners), else the material's base colour.

Nothing here imports SAPIEN: it reads the objects the running scene hands it, by component class name.
Reading changes no simulator state; ``--check``'s comparison runs on the same re-execution and says so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ora.environment.scene_trace import (
    MIN_SAMPLED_VERTICES, Body, Mesh, SceneTraceWriter, StepTrace, linear, merge, rotation,
)

CLOCK_UNIT = "sapien_step_since_round_reset"


def _shape_scale(shape: Any) -> list[float]:
    """What SAPIEN scales a shape's vertices by: boxes, spheres and cylinders are unit shapes, a capsule is built
    in metres, a triangle mesh or plane has its own ``scale``."""
    kind = type(shape).__name__
    if kind == "RenderShapeBox":
        return [float(value) for value in shape.half_size]
    if kind == "RenderShapeSphere":
        return [float(shape.radius)] * 3
    if kind == "RenderShapeCylinder":
        return [float(shape.half_length), float(shape.radius), float(shape.radius)]
    if kind == "RenderShapeCapsule":
        return [1.0, 1.0, 1.0]
    return [float(value) for value in shape.scale]


def _part_colors(part: Any, count: int) -> Any:
    import numpy as np

    material = part.material
    texture = material.base_color_texture
    if texture is None:
        rgb = np.tile(np.asarray(material.base_color, dtype=np.float64)[:3], (count, 1))
    else:
        pixels = np.asarray(texture.download())[..., :3].astype(np.float64) / 255.0
        uv = part.get_vertex_uv()
        if uv is None or len(uv) != count or count < MIN_SAMPLED_VERTICES:
            rgb = np.tile(linear(pixels.reshape(-1, 3).mean(axis=0)), (count, 1))
        else:
            height, width = pixels.shape[:2]
            fraction = np.mod(np.asarray(uv, dtype=np.float64), 1.0)
            columns = np.minimum((fraction[:, 0] * width).astype(np.int64), width - 1)
            rows = np.minimum((fraction[:, 1] * height).astype(np.int64), height - 1)
            rgb = linear(pixels[rows, columns])
    alpha = np.ones((count, 1))
    return np.clip(np.rint(np.concatenate([rgb, alpha], axis=1) * 255.0), 0, 255).astype(np.uint8)


def body_mesh(render: Any) -> Mesh | None:
    """A render body's shapes as one mesh in the entity's frame."""
    import numpy as np

    parts = []
    for shape in render.render_shapes:
        scale = np.asarray(_shape_scale(shape), dtype=np.float64)
        turn, origin = rotation(shape.local_pose.q), np.asarray(shape.local_pose.p, dtype=np.float64)
        for part in shape.parts:
            vertices = np.asarray(part.vertices, dtype=np.float64).reshape(-1, 3)
            parts.append(Mesh((vertices * scale) @ turn.T + origin, _part_colors(part, len(vertices)),
                              np.asarray(part.triangles, dtype=np.int64).reshape(-1, 3)))
    return merge(parts)


def _pose_reader(entity: Any) -> Any:
    def read() -> list[float]:
        pose = entity.pose
        p, q = pose.p, pose.q
        return [p[0], p[1], p[2], q[1], q[2], q[3], q[0]]
    return read


def _robot_entities(environment: Any) -> set[int]:
    robot = environment.robot
    return {link.entity.per_scene_id for articulation in (robot.left_entity, robot.right_entity)
            for link in articulation.get_links()}


def scene_bodies(environment: Any) -> list[Body]:
    """Every body of the round's scene (see the module docstring)."""
    robot = _robot_entities(environment)
    bodies: list[Body] = []
    seen: dict[str, int] = {}
    for entity in environment.scene.get_entities():
        components = {type(component).__name__: component for component in entity.get_components()}
        link = components.get("PhysxArticulationLinkComponent")
        static = "PhysxRigidStaticComponent" in components
        if link is None and not static and "PhysxRigidDynamicComponent" not in components:
            continue
        if link is not None:
            name = f"{link.articulation.name or 'robot'}/{entity.name}"
            kind = "robot" if entity.per_scene_id in robot else "object"
        else:
            name, kind = entity.name or "body", "static" if static else "object"
        seen[name] = seen.get(name, 0) + 1
        key = name if seen[name] == 1 else f"{name}#{seen[name]}"
        render = components.get("RenderBodyComponent")
        bodies.append(Body(key=key, name=name, kind=kind, read_pose=_pose_reader(entity),
                           mesh=body_mesh(render) if render is not None else None))
    return bodies


class RoboTwinSceneTrace(StepTrace):
    """A RoboTwin replay's trace; ``StepTrace`` holds the step tap."""

    def __init__(self, directory: Path, *, run_id: str) -> None:
        super().__init__(SceneTraceWriter(directory, run_id=run_id, backend="robotwin", clock_unit=CLOCK_UNIT))

    def begin_round(self, environment: Any, round_index: int) -> None:
        self.close()
        self.writer.begin_round(round_index, scene_bodies(environment), timestep_s=environment.scene.get_timestep())
        self.watch(environment.scene)


__all__ = ["CLOCK_UNIT", "RoboTwinSceneTrace", "body_mesh", "scene_bodies"]
