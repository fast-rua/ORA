"""A translation-only target preserves the measured action-start orientation."""

from ..move_pose.tool import compile_and_submit


def run(api, arguments):
    """Replay: recorded observation/IK or 未记录; compare one action, never execute online."""
    return compile_and_submit(
        api, arguments, config=api.artifacts.tool_context(), primitive="move_to"
    )
