"""``latency_profile``: which costs a run keeps on its control path.

A run is either a simulation benchmark, where every cost that serves people,
evaluation or evidence is paid on the control path because the world waits for
it, or a real-time cell, where the world does not wait. The RunSpec names one of
two profiles and the plan records the explicit switches it resolved to
(``ora.latency-profile.v1``), so a run's evidence says what it ran with and a
later reader never infers it from a profile name.

- ``sim_benchmark`` (the default, and every plan resolved before the switches
  existed): today's behaviour. Nothing in the runtime, the tool context or what
  the model reads changes.
- ``realtime``: execution capture is off in the loop, ``move_pose``'s reach
  reference is not computed and dry runs and refused real calls carry no
  rendered preview picture.

Each switch is one Boolean; ``SWITCHES`` says what it does, ``MODEL_FACING``
which of them change what the model reads when they are off. A RunSpec may
name a profile (``latency_profile: realtime``) or a profile with switches set
one by one (``{profile: realtime, preview_images: true}``); a switch the RunSpec
sets is listed in ``set_by_spec``. ``runtime.execution_capture.enabled``, where
the RunSpec names it, sets the ``execution_capture`` switch; naming both with
different values is refused. The other costs a real-time cell drops (the
per-step journal, the official reward and terminal checks, the planner's scene
rebuild) live in the RoboDojo worker kernel and are not switches yet
(docs/reference/latency-profile.md).
"""

from __future__ import annotations

from typing import Any, Mapping

from .values import RunSpecError

SCHEMA = "ora.latency-profile.v1"
DEFAULT_PROFILE = "sim_benchmark"
#: Each switch, on is the benchmark behaviour.
SWITCHES = {
    # Control-only pixels of the original execution (runtime.execution_capture): people, never the policy or the
    # verdict, read them. Off, the worker renders nothing for them during a motion.
    "execution_capture": "execution capture records camera frames during motions (runtime.execution_capture)",
    # move_pose's, move_to's, rotate_wrist's and rotate_pitch's compiled_pose.targets.<arm>.reach_along_request:
    # a read-only IK question asked before the move is submitted.
    "reach_reference": "pose tools ask the planner's IK for compiled_pose.targets.<arm>.reach_along_request",
    # The picture of a dry run's plan and of a real call refused by its plan's verdict.
    "preview_images": "dry runs and refused real calls carry a rendered preview picture",
}
PROFILES = {
    "sim_benchmark": {"execution_capture": True, "reach_reference": True, "preview_images": True},
    "realtime": {"execution_capture": False, "reach_reference": False, "preview_images": False},
}
#: The switches whose off value changes what the model reads (``model_visible.model_visible_items`` adds them).
MODEL_FACING = ("reach_reference", "preview_images")


def select(value: Any) -> dict[str, Any]:
    """The profile and switch values a RunSpec's ``latency_profile`` names, before the runtime resolves."""

    if value is None:
        value = DEFAULT_PROFILE
    if isinstance(value, str):
        value = {"profile": value}
    if not isinstance(value, Mapping):
        raise RunSpecError("latency_profile must be a profile name or an object with profile and switches")
    unknown = sorted(set(value) - {"profile", *SWITCHES})
    if unknown:
        raise RunSpecError(f"latency_profile has unknown fields: {', '.join(unknown)}")
    profile = value.get("profile", DEFAULT_PROFILE)
    if profile not in PROFILES:
        raise RunSpecError(f"latency_profile.profile must be one of {sorted(PROFILES)}")
    switches = dict(PROFILES[profile])
    named = []
    for name in SWITCHES:
        if name in value:
            if type(value[name]) is not bool:
                raise RunSpecError(f"latency_profile.{name} must be true or false")
            switches[name] = value[name]
            named.append(name)
    return {"profile": profile, "switches": switches, "set_by_spec": named}


def with_capture(spec: Mapping[str, Any], selection: Mapping[str, Any], *, captures: bool) -> Mapping[str, Any]:
    """``spec`` with the profile's ``execution_capture`` as ``runtime.execution_capture.enabled``.

    ``captures`` is whether the runtime's backend records execution without being asked. A RunSpec that names
    ``runtime.execution_capture.enabled`` keeps it, and its value is the switch's; a plan whose capture the profile
    leaves as it was (on, or a backend that records nothing unasked) keeps its runtime unchanged.
    """

    runtime = spec.get("runtime") if isinstance(spec.get("runtime"), Mapping) else {}
    capture = runtime.get("execution_capture")
    if isinstance(capture, Mapping) and "enabled" in capture:
        if "execution_capture" in selection["set_by_spec"] and capture["enabled"] != selection["switches"][
                "execution_capture"]:
            raise RunSpecError("latency_profile.execution_capture and runtime.execution_capture.enabled differ")
        return spec
    if selection["switches"]["execution_capture"] or not (captures or isinstance(capture, Mapping)):
        return spec
    runtime = dict(runtime, execution_capture={**dict(capture or {}), "enabled": False})
    return {**spec, "runtime": runtime}


def record(selection: Mapping[str, Any], spec: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    """The plan's ``latency_profile``: the profile and the switches the run executes with.

    ``execution_capture`` is what the resolved runtime records; a RunSpec that named it
    (``runtime.execution_capture.enabled``) set it.
    """

    switches = dict(selection["switches"])
    capture = runtime.get("execution_capture")
    switches["execution_capture"] = bool(isinstance(capture, Mapping) and capture.get("enabled"))
    named = list(selection["set_by_spec"])
    given = (spec.get("runtime") or {}).get("execution_capture") if isinstance(spec.get("runtime"), Mapping) else None
    if isinstance(given, Mapping) and "enabled" in given and "execution_capture" not in named:
        named.append("execution_capture")
    return {"schema": SCHEMA, "profile": selection["profile"], "switches": switches,
            "set_by_spec": [name for name in SWITCHES if name in named],
            "model_facing_off": [name for name in MODEL_FACING if not switches[name]]}


def tool_switches(plan: Mapping[str, Any]) -> dict[str, bool] | None:
    """The model-facing switches a plan turns off, for the tool context; ``None`` when all are on.

    Only such a plan's tool calls carry ``latency_switches`` in their recorded configuration, so a benchmark run's
    tool inputs are byte for byte what they were before the switches (and replay reads what was recorded).
    """

    value = plan.get("latency_profile")
    switches = value.get("switches") if isinstance(value, Mapping) else None
    if not isinstance(switches, Mapping):
        return None
    return {name: False for name in MODEL_FACING if switches.get(name) is False} or None


__all__ = ["DEFAULT_PROFILE", "MODEL_FACING", "PROFILES", "SCHEMA", "SWITCHES", "record", "select",
           "tool_switches", "with_capture"]
