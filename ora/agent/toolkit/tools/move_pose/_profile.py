"""The robot motion profile a call's context binds: what ``fast`` is and how the jaws move.

A RoboDojo plan names its profile by path (``runtime.motion_profile``,
``ora/policy/embodiments/arx-x5/motion-profile-v<N>.json``); the host verifies the
file and hands the tool context its ``fast`` pace and settle and its jaw forms
under ``motion.profile``. A context without one -- every run sealed before
profile v2, every frozen plan that binds none -- runs ``LEGACY``, the values the
toolkit always had (``motion-profile-v1.json`` states the same values), so its
calls build the requests they recorded and replay unchanged.

Only ``fast`` and the jaws are the profile's: ``normal``, ``careful``,
``speed_m_s`` (``_smooth.custom_pace``) and the recovery retrace keep their own
caps whatever the profile.
"""

from collections.abc import Mapping

from ...lib.motion_plan import timing

#: The schema of an ARX X5 motion profile file, and of the part a tool context carries.
SCHEMA = "ora.arx-x5-motion-profile.v1"
CONTEXT_SCHEMA = "ora.motion-profile-context.v1"
#: The jaw forms a profile names: a close, an open that releases a held close, any other open.
GRIPPER_FORMS = ("close", "release", "open")
_PACE_FIELDS = (*timing.CAPS, "joint_jerk_rad_s3")
_SETTLE_FIELDS = ("native_actions", "joint_tolerance_rad", "joint_speed_rad_s")
_FORM_FIELDS = {"rate_per_s", "max_lead_m", "arm_settle"}
_FORM_OPTIONAL = {"hold_lead_m", "end"}

#: The toolkit's values before profiles (``motion-profile-v1.json``): fast is 80% of the X5's 5 rad/s joint
#: limit at 40 rad/s^2, certified at the robot's jerk; a close moves at 2 openings/s leading the finger by at
#: most 5 mm after one settle look at the arm; an open releasing a held close at 2 openings/s, any other at full rate.
LEGACY = {
    "profile_id": None,
    "fast": {"pace": timing.Pace(linear_m_s=None, angular_deg_s=None, linear_accel_m_s2=None,
                                 angular_accel_deg_s2=None, joint_speed_rad_s=4.0, joint_accel_rad_s2=40.0),
             "settle": {"native_actions": 5, "joint_tolerance_rad": 0.003, "joint_speed_rad_s": 0.05}},
    "gripper": {"close": {"rate_per_s": 2.0, "max_lead_m": 0.005,
                          "arm_settle": {"joint_speed_rad_s": 0.05, "max_native_actions": 5}},
                "release": {"rate_per_s": 2.0, "max_lead_m": None, "arm_settle": None},
                "open": {"rate_per_s": None, "max_lead_m": None, "arm_settle": None}},
}


class ProfileError(ValueError):
    """A motion profile file or context entry that is not one this toolkit knows."""


def _pace(value):
    if not isinstance(value, Mapping) or set(value) != set(_PACE_FIELDS):
        raise ProfileError(f"a profile pace names exactly {sorted(_PACE_FIELDS)}")
    try:
        return timing.Pace(**dict(value))
    except (TypeError, ValueError) as error:
        raise ProfileError(f"a profile pace is invalid: {error}") from error


def _settle(value):
    if not isinstance(value, Mapping) or set(value) != set(_SETTLE_FIELDS):
        raise ProfileError(f"a profile settle names exactly {sorted(_SETTLE_FIELDS)}")
    return dict(value)


def _form(name, value):
    if (not isinstance(value, Mapping) or not _FORM_FIELDS <= set(value)
            or set(value) - _FORM_FIELDS - _FORM_OPTIONAL):
        raise ProfileError(f"the profile's {name} jaw form names {sorted(_FORM_FIELDS)} and optionally "
                           f"{sorted(_FORM_OPTIONAL)}")
    return dict(value)


def context_entry(profile, *, path, sha256):
    """Pure computation: the ``motion.profile`` a tool context carries for a verified profile file's JSON.

    Raises ``ProfileError``. Example: context_entry(json.loads(text), path=p, sha256=h)['profile_id']."""
    if not isinstance(profile, Mapping) or profile.get("schema") != SCHEMA:
        raise ProfileError(f"a motion profile is an {SCHEMA} file")
    fast, gripper = profile.get("fast"), profile.get("gripper")
    if not isinstance(fast, Mapping) or set(fast) != {"pace", "settle"}:
        raise ProfileError("a profile's fast names pace and settle")
    if not isinstance(gripper, Mapping) or set(gripper) != set(GRIPPER_FORMS):
        raise ProfileError(f"a profile's gripper names {list(GRIPPER_FORMS)}")
    _pace(fast["pace"])
    return {"schema": CONTEXT_SCHEMA, "profile_id": str(profile["profile_id"]), "path": str(path),
            "sha256": str(sha256), "fast": {"pace": dict(fast["pace"]), "settle": _settle(fast["settle"])},
            "gripper": {name: _form(name, gripper[name]) for name in GRIPPER_FORMS}}


def bound(config):
    """Pure computation: the call's profile as ``{"profile_id", "fast": {"pace": Pace, "settle"}, "gripper"}``.

    ``LEGACY`` where the context binds none. Example: bound(config)['fast']['pace'].joint_speed_rad_s."""
    entry = (config.get("motion") or {}).get("profile") if isinstance(config, Mapping) else None
    if entry is None:
        return LEGACY
    if not isinstance(entry, Mapping) or entry.get("schema") != CONTEXT_SCHEMA:
        raise ProfileError(f"motion.profile is an {CONTEXT_SCHEMA} entry")
    return {"profile_id": entry["profile_id"],
            "fast": {"pace": _pace(entry["fast"]["pace"]), "settle": _settle(entry["fast"]["settle"])},
            "gripper": {name: _form(name, entry["gripper"][name]) for name in GRIPPER_FORMS}}


def fast(config):
    """Pure computation: ``(pace, settle)`` of the call's ``fast``. Example: fast({})[0].joint_speed_rad_s == 4.0."""
    value = bound(config)["fast"]
    return value["pace"], dict(value["settle"])


def gripper_form(config, form):
    """Pure computation: one jaw form of the call's profile (``GRIPPER_FORMS``), a fresh dict."""
    value = bound(config)["gripper"][form]
    return {key: (dict(item) if isinstance(item, Mapping) else item) for key, item in value.items()}


__all__ = ["CONTEXT_SCHEMA", "GRIPPER_FORMS", "LEGACY", "ProfileError", "SCHEMA", "bound", "context_entry", "fast",
           "gripper_form"]
