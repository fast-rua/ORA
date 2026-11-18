"""The RoboLab grasp_plan result schema publishes the compiler its runtime selects.

``robolab_direct_v1`` declares ``contact_graspnet_robotiq_2f85_tcp_compiler_v1``,
which makes the runtime publish the Robotiq TCP compiler identity; its result
schema, inherited from the LIBERO family, pinned the Panda controlled point, so
every RoboLab grasp_plan call failed validation. A binder rebound that one
constant on this surface only; its grasp_plan file now holds what it wrote.
LIBERO and RoboCasa keep the Panda constant, and a whole grasp_plan result
produced by the runtime validates on RoboLab.
"""
from __future__ import annotations

import copy

import pytest

from ora.agent.composition.surface.tool_surface import (
    ToolSurfaceError, resolve_tool_surface, validate_schema_instance,
)
from ora.agent.toolkit.host.grasp.compiler import (
    PANDA_GRIP_SITE_COMPILER_IDENTITY, ROBOTIQ_TCP_COMPILER_IDENTITY,
)
from tests.support.agent.grounding_grasp_runtime import _runtime


ROBOLAB = "robolab_direct_v1"
CAPABILITY = "contact_graspnet_robotiq_2f85_tcp_compiler_v1"
PANDA_SURFACES = ("libero_placement_direct_overhead_v1", "robocasa_mobile_direct_v1")


def _compiler_schema(profile: str) -> dict:
    defs = resolve_tool_surface(profile).result_schemas["grasp_plan"]["$defs"]
    return {"$ref": "#/$defs/compiler", "$defs": defs}


def test_robolab_accepts_the_robotiq_identity_and_refuses_the_panda_one():
    surface = resolve_tool_surface(ROBOLAB)
    assert surface.supports(CAPABILITY)
    assert surface.view_config["controlled_frame_convention"] == ROBOTIQ_TCP_COMPILER_IDENTITY["controlled_point"]
    schema = _compiler_schema(ROBOLAB)
    validate_schema_instance(copy.deepcopy(ROBOTIQ_TCP_COMPILER_IDENTITY), schema)
    with pytest.raises(ToolSurfaceError, match="controlled_point differs from result const"):
        validate_schema_instance(copy.deepcopy(PANDA_GRIP_SITE_COMPILER_IDENTITY), schema)


@pytest.mark.parametrize("profile", PANDA_SURFACES)
def test_libero_and_robocasa_keep_the_panda_constant(profile):
    assert not resolve_tool_surface(profile).supports(CAPABILITY)
    schema = _compiler_schema(profile)
    validate_schema_instance(copy.deepcopy(PANDA_GRIP_SITE_COMPILER_IDENTITY), schema)
    with pytest.raises(ToolSurfaceError, match="controlled_point differs from result const"):
        validate_schema_instance(copy.deepcopy(ROBOTIQ_TCP_COMPILER_IDENTITY), schema)


def test_a_runtime_grasp_plan_result_validates_on_robolab(tmp_path):
    from tests.support.agent.toolkit_run import served

    runtime, observation_id = _runtime(tmp_path, profile=ROBOLAB, toolkit=True)
    served(runtime)
    grounded = runtime.invoke("vision_ground", {"camera": "agentview", "text": "red object", "max_candidates": 2})
    planned = runtime.invoke("grasp_plan", {"target_ref": grounded["candidates"][0]["target_ref"], "max_candidates": 3})
    assert planned["observation_id"] == observation_id
    assert planned["status"] == "candidates" and planned["candidates"]
    assert planned["compiler"] == ROBOTIQ_TCP_COMPILER_IDENTITY
    visible = {key: value for key, value in planned.items() if key not in {"image_base64", "images_base64", "_runtime"}}
    validate_schema_instance(visible, runtime.surface.result_schemas["grasp_plan"])
