"""return_home: each selected arm straight back to the episode reset joints, whatever path it came by.

One smooth trajectory planned in lib (``lib.motion_plan.recovery``): link6
lifts up to 0.10 m straight up, orientation held, then the joints unwind to
``exec.recovery.<arm>.home`` in the first screened order (the shared code is
recover_pose's ``_trajectory.run_home``).
"""

from ...interface import dry_run
from .. import _dry_run
from ..errors import ToolRefusal
from ..recover_pose import _trajectory


def run(api, arguments):
    """Replay: recorded observation, home anchors, kinematics answers and the one recorded trajectory primitive.

    Missing or non-finite reset joints and blocked paths refuse before anything moves (zero spend); the host
    admits only the current home anchors."""
    config = api.artifacts.tool_context()
    refusal = config["motion"]["mutation_refusals"].get("return_home")
    if refusal is not None:
        raise ToolRefusal.from_public(refusal)
    arms = _trajectory.arms_argument(arguments, "return_home")
    if dry_run.requested():  # a plan the real call would refuse is the dry run's answer
        return _dry_run.answered("return_home", lambda: _trajectory.run_home(api, config, arms))
    return _trajectory.run_home(api, config, arms)
