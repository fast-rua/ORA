"""The CuRobo IK seed generators as episode state (``exec.ik_sampler_state``).

Every SDK IK solve draws its random seeds from a Halton sample buffer whose
index generator advances on every draw, so the same request answers
differently depending on the solves before it. With CuRobo's default LM seed
solver (the SDK keeps ``use_lm_seed=True``) that is the ``SeedIKSolver``'s
``act_sample_gen`` (``sampler_seed`` 451); without it the main solver's seed
manager buffer (seed 123). That generator state is therefore part of the episode state: the
worker reports it after every request (``ExecutionRequest``), so every
publication snapshot holds it, and a request whose host values carry it sets
the generators back to it first. A restore or a fork continues from the state
its boundary recorded and later solves advance it naturally; each new
snapshot records its own. A run recorded before this key existed has no
state to restore (``unrecorded``): its recorded IK answers stand as they were.

One CuroboPlanner serves both X5 arms (``RobotManager`` keys planners by
robot name); the generator that advances is the LM seed solver's when it is
enabled, else the IK solver's (``ik.seed_generator``). A solver exposing
neither records no state; a kinematics query on it is refused
(``seeding_unavailable``), never answered with unreproducible seeds.
"""

from __future__ import annotations

from typing import Any, Mapping

from .bridge import RobodojoError
from .ik import generator_state, restore_generator_state, seed_generator

SCHEMA = "ora.robodojo-ik-sampler-state.v1"
UNRECORDED = "unrecorded"
RESTORED = "restored"
UNRECORDED_WARNING = ("this run predates IK generator-state recording; IK seeds restart from the initial state, "
                      "so solutions may differ from what the original run would have computed")


def _planners(session: Any) -> dict[str, Any]:
    manager = session.environment.robot_manager
    planners = getattr(manager, "ik_solver", None) or {}
    return {str(robot.robot_name): planners[robot.robot_name]
            for robot in getattr(session, "robots", {}).values() if robot.robot_name in planners}


def sampler_states(session: Any) -> dict[str, Any] | None:
    """Each planner's seed generator state, or ``None`` when no planner exposes one."""
    planners = {}
    for name, planner in _planners(session).items():
        buffer, generator, owner = seed_generator(planner)
        state = generator_state(generator)
        if state is None:
            continue
        sequencer = getattr(buffer, "sequencer", None)
        planners[name] = {"generator": owner, "state_hex": state,
                          "random_seed": getattr(sequencer, "seed", None),
                          "sample_buffer": "curobo_halton_store_buffer"}
    return {"schema": SCHEMA, "planners": planners} if planners else None


def restore_sampler_states(session: Any, value: Mapping[str, Any]) -> list[str]:
    """Set each recorded planner's generator to its recorded state; returns the planners restored."""
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise RobodojoError("IK sampler state is not ora.robodojo-ik-sampler-state.v1")
    planners = _planners(session)
    restored = []
    for name, record in dict(value.get("planners") or {}).items():
        if name not in planners:
            raise RobodojoError(f"recorded IK sampler state names an unknown planner: {name}")
        _, generator, owner = seed_generator(planners[name])
        if owner != record.get("generator"):
            raise RobodojoError("recorded IK sampler state belongs to another seed generator")
        if generator_state(generator) != record["state_hex"]:
            restore_generator_state(planners[name], record["state_hex"])
        restored.append(name)
    return restored


def resume(session: Any, value: Mapping[str, Any] | None) -> str:
    """Continue a restored or forked run from its boundary's generator state.

    Without one (a run recorded before the key existed) every planner's
    sample buffer restarts from its initial state; the session remembers it,
    so every later kinematics answer carries the warning.
    """
    if value is not None:
        restore_sampler_states(session, value)
        session.ik_sampler_provenance = RESTORED
        return RESTORED
    for planner in _planners(session).values():
        buffer, _, _ = seed_generator(planner)
        if callable(getattr(buffer, "reset", None)):
            buffer.reset()
    session.ik_sampler_provenance = UNRECORDED
    return UNRECORDED


def provenance(session: Any) -> dict[str, Any]:
    """The warning fields of an answer computed on a restored run without recorded generator state."""
    if getattr(session, "ik_sampler_provenance", None) == UNRECORDED:
        return {"ik_sampler_state": UNRECORDED, "warning": UNRECORDED_WARNING}
    return {}


__all__ = ["RESTORED", "SCHEMA", "UNRECORDED", "UNRECORDED_WARNING", "provenance", "restore_sampler_states",
           "resume", "sampler_states"]
