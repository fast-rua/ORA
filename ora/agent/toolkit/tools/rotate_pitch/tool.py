"""A rotation about a declared local robot axis, with position held."""
import math
from ..move_pose.tool import compile_and_submit
from ..move_pose._refusals import PrimitiveAngleError


def run(api, arguments):
    """Replay: recorded current state and IK; changed queries fail unrecorded.

    World position remains fixed; public angles are degrees. Live calls use the
    same trusted executor and native budget as move_pose, never legacy submit.
    """
    config = api.artifacts.tool_context()
    capabilities = config["motion"]["capabilities"]
    axis = "y"

    def request(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("each rotate_pitch target must be a finite angle in degrees")
        if abs(float(value)) > 180:
            raise PrimitiveAngleError("each rotate_pitch target must be within [-180, 180] degrees")
        return {"position": {"mode": "hold_current"},
                "orientation": {"mode": "axis_angle_delta", "axis": axis,
                                "angle_deg": float(value), "frame": "end_effector"}}

    return compile_and_submit(api, arguments, config=config, primitive="rotate_pitch", request_for_target=request,
                              scalar_key="angle_deg")
