"""Render one candidate grip-site pose on the current agentview, without acting."""

from ...lib import observation
from ..errors import ToolRefusal
from ..move_pose import _atomic
from ..observe import _annotated

MARK_COLOUR = [255, 0, 255]


def run(api, arguments):
    """Replay: recorded observation and tool.previews; pure compilation and rendering, nothing moves.

    Pose metres and xyzw in the world frame; the returned ``preview_id`` holds for this observation only.
    """
    config = api.artifacts.tool_context()
    atomic = config.get("atomic")
    if atomic is None:
        raise ToolRefusal("preview_move runs only on an atomic surface", code="tool_unavailable", retryable=False)
    obs = observation.observe(api)
    manifest = obs.manifest.json()
    current, _ = _atomic.grip_state(manifest)
    target, compiled = _atomic.compile_target(arguments, current=current, grip_arm=_atomic.arm(manifest, config),
                                              config=config)
    observed = obs.identity.json()["observation_id"]
    preview_id = _atomic.preview_id(observed, target)
    previews = api.state.read("tool.previews")
    previews[preview_id] = {"observation_id": observed, "action": "move_pose", "arguments": target,
                            **({"compiled_pose": compiled} if compiled is not None else {})}
    api.state.write("tool.previews", previews)
    rendered = _annotated.camera_view(api, obs, atomic, "agentview", [["target", list(target["position_m"]), MARK_COLOUR]])
    image = _annotated.published(api, rendered, "agentview", manifest, "move_preview")
    result = {"status": "candidate", "preview_id": preview_id, "views": ["candidate_agentview"], "images_base64": [image],
              "_runtime": {"directive": "preview_only"}}
    if compiled is not None:
        result["compiled_pose"] = compiled
    if atomic["view_dimensions"]:
        result["view_dimensions"] = {"candidate_agentview": {"width": rendered.width, "height": rendered.height}}
    return result
