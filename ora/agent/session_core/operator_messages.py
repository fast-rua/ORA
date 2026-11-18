"""Free-play operator commands that end the policy's current turn.

The orchestrator owns command admission and the inbox (``episode_play``); the
Pi agent core delivers operator messages itself (``ora-pi/src/core/session.ts``).
"""

from __future__ import annotations


#: Free-play operator commands the Supervisor turns into the end of the current turn.
STOP_TURN_REASONS = frozenset({
    "operator_instruction_stop", "operator_scene_reset", "operator_quit", "operator_instruction_superseded",
    "instruction_model_call_budget", "instruction_wall_time_budget",
})
