"""Official judgment, exclusively through the capability-gated interface."""

from ...interface.api import DataInterface
from ..errors import ToolRefusal
from ...lib.values import PublicResult


def run(api: DataInterface, args: dict) -> dict:
    """Replay: recorded same-boundary verdict only; missing data fails.

    No fallback to final results or tool configuration is permitted.
    """
    if args:
        raise ToolRefusal("check_task accepts no arguments")
    api.state.observation()  # Preserve the live tool's open-observation precondition.
    data = PublicResult.of(api.state.check_task()).data.json()
    return {key: value for key, value in data.items() if key != "verdict"}
