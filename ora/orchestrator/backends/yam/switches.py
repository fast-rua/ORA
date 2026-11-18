"""The YAM station's free-play switches: every ``ora play --env yam`` option that turns one station behaviour on.

One row per switch (``SWITCHES``): its flag and help, the ``runtime`` key it sets, and the episode-task layer it adds
for the model. The value a runtime key may hold is ``ora.environment.benchmarks.yam.RUNTIME_SWITCHES``, which the
runtime validation reads; ``tests/ora/orchestrator/backends/yam/test_switches.py`` holds the two to each other.
``ora play`` builds its options, the check that they apply to ``--env yam`` only and its message from this table
(``ora/app/play.py``); ``play.with_station_options`` applies the rows chosen.

A new station switch is one row here, and its value in ``RUNTIME_SWITCHES`` when it sets a runtime key. Every switch
is off by default; one qualified on the real cell becomes the default and its old path is removed, with its row.

``ora`` builds its parser from this module on every start, so it imports nothing heavy.
"""

from __future__ import annotations

from dataclasses import dataclass

PERCEPTION_SURFACE = "yam_perception_v1"


@dataclass(frozen=True)
class StationSwitch:
    name: str                          # the option's argparse dest
    flag: str
    help: str
    runtime_key: str | None = None     # the runtime key it sets (its value: RUNTIME_SWITCHES)
    layer: str | None = None           # the episode-task layer it adds, which tells the model
    surface: str | None = None         # the tool surface it needs


SCRIPTED_MOVES = StationSwitch(
    "scripted_moves", "--scripted-moves",
    "YAM: the model runs a pick and place it can already plan as one checked codemode script instead of a turn "
    "per move (off by default)",
    layer="ora/policy/environments/yam/play-scripted-moves-v1.md", surface=PERCEPTION_SURFACE)
REACH_QUERY = StationSwitch(
    "reach_query", "--reach-query",
    "YAM: an unreachable target's refusal names the nearest position that solves at its height, and the model is "
    "told to read it rather than search a grid (off by default)",
    runtime_key="reach_query", layer="ora/policy/environments/yam/play-reach-query-v1.md")
MOTION_TIMING = StationSwitch(
    "motion_timing", "--motion-timing",
    "YAM: ORA's shared timing worker (TOPP-RA + Ruckig, ORA_TIMING_PYTHON) times the arm paths under the station's "
    "speed caps, with acceleration limits and a start and stop at rest, instead of the station's fixed-speed timing "
    "(off by default)",
    runtime_key="motion_timing")
SMOOTH_COMMANDS = StationSwitch(
    "smooth_commands", "--smooth-commands",
    "YAM --rig-mode on/read: stream each control tick's arm target along a line at 250 Hz with its velocity "
    "feed-forward instead of one position step per tick (off by default)",
    runtime_key="smooth_arm_commands")
MOTION_PLANNER = StationSwitch(
    "motion_planner", "--motion-planner",
    "YAM: ORA's shared cuRobo V2 planner (CUROBO_V2_PYTHON, CUROBO_V2_ROOT) plans the arm paths in the public-depth "
    "world, a free-space move collision-free, instead of the station's own IK lines (off by default)",
    runtime_key="motion_planner")

SWITCHES: tuple[StationSwitch, ...] = (SCRIPTED_MOVES, REACH_QUERY, MOTION_TIMING, SMOOTH_COMMANDS, MOTION_PLANNER)


def flags_sentence(leading: tuple[str, ...] = ()) -> str:
    """``leading`` and every switch's flag as one list: "a, b and c"."""

    names = [*leading, *(switch.flag for switch in SWITCHES)]
    return ", ".join(names[:-1]) + " and " + names[-1]


__all__ = ["MOTION_PLANNER", "MOTION_TIMING", "PERCEPTION_SURFACE", "REACH_QUERY", "SCRIPTED_MOVES", "SMOOTH_COMMANDS",
           "SWITCHES", "StationSwitch", "flags_sentence"]
