"""Assemble immutable categorized capabilities, never a mutable world snapshot.

Robot assembly deliberately has no privileged category. A transitional task
check requires both the public tool and its one state-key grant. Same-process
capabilities are an API boundary, not a Python sandbox.
"""
from dataclasses import dataclass
from typing import Any

from .perception.live import Perception
from .state_access.live import State
from .sim_query.live import Planner
from .sim_exec.live import Actions, SynchronousActions
from .lifecycle.live import Episode
from .artifacts.live import Artifacts
from .compute.live import Compute
from .context.live import Context
from .privileged.live import Privileged
from .types import AccessDenied


@dataclass(frozen=True, slots=True)
class RobotAPI:
    perception: Perception
    state: State
    planner: Planner
    act: Actions | SynchronousActions
    episode: Episode
    artifacts: Artifacts
    compute: Compute
    context: Context


@dataclass(frozen=True, slots=True)
class RsiAPI(RobotAPI):
    privileged: Privileged


def assemble_robot(*, backend: Any, state: Any, context: Any = None,
                   privileged_tools: tuple[str, ...] = (), privileged_keys: tuple[str, ...] = (),
                   synchronous_execution: bool = False) -> RobotAPI:
    """Read state on every call. Only tool.* writes are granted by the owner.

    The caller supplies a robot StateAccess, never a supervisor accessor.
    check_task is the only transitional privilege and requires both grants.
    """
    if state is None or state._reader != 'robot' or state._writer not in {None, 'tool'}:
        raise AccessDenied('robot assembly requires a robot state accessor')
    check = ('check_task' in privileged_tools and 'privileged.task_check' in privileged_keys
             and 'privileged.task_check' in state._privileged_keys)
    return RobotAPI(Perception(backend), State(backend, state, check), Planner(backend),
                    (SynchronousActions(backend) if synchronous_execution else Actions(backend)), Episode(backend), Artifacts(backend), Compute(backend), Context(context))


def assemble_rsi(*, backend: Any, state: Any, context: Any = None) -> RsiAPI:
    """RSI assembly alone includes the privileged namespace; it cannot write it."""
    if state._reader != 'rsi' or state._writer not in {None, 'tool'}:
        raise AccessDenied('RSI assembly requires an RSI state accessor')
    return RsiAPI(Perception(backend), State(backend, state, True), Planner(backend),
                  Actions(backend), Episode(backend), Artifacts(backend), Compute(backend),
                  Context(context), Privileged(state))


def assemble_snapshot(*, backend: Any, state: Any, context: Any = None,
                      control_recorder: Any = None) -> RsiAPI:
    """Compose RSI snapshot reads/services with no simulator or online actions.

    Round control goes to the supplied replay owner, which records requests
    only in the output. Phase four binds this composition to the replay CLI.
    """
    from dataclasses import replace
    from .lifecycle.replay import Episode as SnapshotEpisode
    from .state_access.replay import State as SnapshotState
    from .sim_query.replay import Planner as SnapshotPlanner
    from .sim_exec.replay import Actions as SnapshotActions
    from .privileged.replay import Privileged as SnapshotPrivileged
    api = assemble_rsi(backend=backend, state=state, context=context)
    return replace(api, state=SnapshotState(backend, state, True),
                   planner=SnapshotPlanner(), act=SnapshotActions(),
                   episode=SnapshotEpisode(control_recorder), privileged=SnapshotPrivileged(state))
