"""The Runtime's own five-tool catalog, the base of a surface that names none.

``tool_surface.resolve_tool_surface`` starts from these definitions when a
manifest declares no ``base_definitions`` component (``robot_runtime_v2``);
``tools/runtime/base.py`` serves them. They are composition
data, so the resolver reads them without loading the tool runtime.
"""

from __future__ import annotations


TOOL_DEFINITIONS = (
    {
        "name": "observation.dump_state",
        "description": (
            "Read the latest or one saved policy-visible state step. Returns public state, "
            "camera/modality inventory and the selected step's main-camera image."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"step": {"type": ["integer", "null"], "minimum": 0}},
            "additionalProperties": False,
        },
    },
    {
        "name": "vision.inspect",
        "description": "Inspect a current or saved policy-visible camera image or crop.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "camera": {"type": "string"},
                "modality": {"type": "string", "enum": ["rgb", "depth"]},
                "crop": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                "resolution": {"type": "string", "enum": ["thumbnail", "full"]},
                "step": {"type": ["integer", "null"], "minimum": 0},
            },
            "required": ["camera"],
            "additionalProperties": False,
        },
    },
    {
        "name": "geometry.query",
        "description": "Back-project pixels, project world points, or summarize region depth.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "camera": {"type": "string"},
                "points": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, "minItems": 1, "maxItems": 16},
                "region": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                "world_points_m": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "minItems": 1, "maxItems": 16},
                "window_px": {"type": "integer", "minimum": 0, "maximum": 15},
                "step": {"type": ["integer", "null"], "minimum": 0},
            },
            "required": ["camera"],
            "oneOf": [
                {"required": ["points"]},
                {"required": ["region"]},
                {"required": ["world_points_m"]},
            ],
            "additionalProperties": False,
        },
    },
    {
        "name": "robot.read_manual",
        "description": "Read one named section of the robot manual.",
        "inputSchema": {
            "type": "object",
            "properties": {"section": {"type": "string"}},
            "required": ["section"],
            "additionalProperties": False,
        },
    },
    {
        "name": "robot.submit_action",
        "description": "Submit one pose-only motion or one close-in-place gripper command.",
        "inputSchema": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "const": "move_pose"},
                        "arm": {"type": "string", "enum": ["left", "right"]},
                        "position_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                        "orientation_xyzw": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    },
                    "required": ["action", "arm", "position_m", "orientation_xyzw"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "const": "set_gripper"},
                        "arm": {"type": "string", "enum": ["left", "right"]},
                        "gripper": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["action", "arm", "gripper"],
                    "additionalProperties": False,
                },
            ],
        },
    },
)


__all__ = ["TOOL_DEFINITIONS"]
