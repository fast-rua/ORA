"""State keys, writers and readers. Poses use episode-local metres / xyzw.

Joint positions are radians; gripper directives use normalized opening.
Camera arrays retain their published dtype, shape and order in PNG/NPY files.
Tools can define JSON values under tool.*; other keys are a closed vocabulary.
"""

from dataclasses import dataclass
import re


class StateError(ValueError):
    """Invalid state, absent evidence or an incomplete snapshot boundary."""


class StateAccessDenied(StateError):
    """The accessor cannot read or write this namespace."""


@dataclass(frozen=True)
class Key:
    pattern: str
    writer: str
    units: str


KEYS = (
    Key(r"env.observation", "environment", "observation identity and publication step"),
    Key(r"env.camera.(head|left_wrist|right_wrist)", "environment", "PNG RGB; float32 NPY metres; calibration JSON"),
    Key(r"env.proprioception.(left|right)", "environment", "episode-local metres / xyzw; radians; normalized opening"),
    Key(r"env.terminal", "environment", "terminal Boolean and reason"),
    Key(r"exec.command.(left|right)", "execution", "episode-local metres / xyzw; normalized opening directive; radians"),
    Key(r"exec.recovery.(left|right).(previous|home|undo)", "execution", "radians; episode-local metres / xyzw"),
    Key(r"exec.session_home.(left|right)", "execution", "first-reset radians; episode-local metres / xyzw"),
    Key(r"exec.last_receipt", "execution", "public execution receipt"),
    Key(r"exec.world_mutation_sequence", "execution", "count"),
    Key(r"exec.native_actions", "execution", "native action count: limit and used"),
    Key(r"exec.tool_calls_with_actions", "execution", "count"),
    Key(r"exec.round", "execution", "round index and official instruction"),
    Key(r"exec.ik_sampler_state", "execution", "CuRobo IK seed generator state per planner (hex bytes)"),
    Key(r"privileged.task_check", "environment", "official predicate"),
    Key(r"privileged.ground_truth", "environment", "capture-defined native frames and units"),
    Key(r"tool.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", "tool", "tool-defined finite JSON"),
)


def key_definition(name: str) -> Key:
    if not isinstance(name, str):
        raise StateError("state key must be a string")
    for key in KEYS:
        # Dots in the fixed path are literals; only tool's explicit suffix is regex.
        pattern = key.pattern.replace(".", r"\.") if not key.pattern.startswith("tool.") else r"tool\." + key.pattern[5:]
        if re.fullmatch(pattern, name):
            return key
    raise StateError(f"unknown state key: {name}")


def require_read(name: str, *, reader: str, privileged_keys: frozenset[str]) -> None:
    key_definition(name)
    if reader not in {"robot", "rsi", "supervisor"}:
        raise StateAccessDenied(f"unknown state reader: {reader}")
    if name.startswith("privileged.") and reader == "robot" and name not in privileged_keys:
        raise StateAccessDenied(f"privileged state is unavailable: {name}")
