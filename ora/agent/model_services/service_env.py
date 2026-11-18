"""The resident perception services' settings, read from the environment.

``ORA_SAM_*`` configures the SAM3 segmentation service and ``ORA_GRASP_*`` the
Contact-GraspNet grasp-proposal service: ``PYTHON``, ``ROOT``, ``CHECKPOINT``,
``GPU`` and ``PORT`` for both, and ``CONFIG`` for the grasp service.

Releases before 1.0 used another prefix for the same settings. A setting that
is unset (or empty) under its ``ORA_`` name is still read under the earlier
name, and the first such read in a process warns once (``FutureWarning``).
When both names are set to different values, the ``ORA_`` value is used and
the first such read of each setting warns once (``UserWarning``) that the
earlier name is ignored. ``ora setup --env`` moves the earlier names in
``.env`` to the new ones (``legacy_renames``). That spelling lives only in
``legacy_service_key``; everything else names a setting through
``service_key``.
"""

from __future__ import annotations

import os
import warnings
from typing import Mapping

#: Every setting of each service, by service.
SERVICE_FIELDS: dict[str, tuple[str, ...]] = {
    "SAM": ("PYTHON", "ROOT", "CHECKPOINT", "GPU", "PORT"),
    "GRASP": ("PYTHON", "ROOT", "CHECKPOINT", "CONFIG", "GPU", "PORT"),
}

_deprecation_warned = False
_conflicts_warned: set[str] = set()


def service_key(service: str, field: str) -> str:
    """The name of one service setting, such as ``ORA_SAM_PORT``."""
    return f"ORA_{service}_{field}"


def legacy_service_key(service: str, field: str) -> str:
    """The deprecated pre-1.0 name of the same setting; read only by ``service_setting``."""
    return f"B2_{service}_{field}"


def legacy_renames() -> dict[str, str]:
    """Each setting's deprecated name mapped to its ``ORA_`` name, for every field of every service."""
    return {legacy_service_key(service, field): service_key(service, field)
            for service, fields in SERVICE_FIELDS.items() for field in fields}


def service_setting(service: str, field: str, environ: Mapping[str, str] | None = None) -> str:
    """The value of ``service_key(service, field)``, else of its deprecated name, else ``""``."""
    values = os.environ if environ is None else environ
    name, legacy = service_key(service, field), legacy_service_key(service, field)
    value, legacy_value = values.get(name), values.get(legacy)
    if value:
        if legacy_value and legacy_value != value and legacy not in _conflicts_warned:
            _conflicts_warned.add(legacy)
            warnings.warn(f"{name} and {legacy} are both set and differ ({value!r} and {legacy_value!r}); using "
                          f"{name}={value!r} and ignoring {legacy}: remove {legacy} (in .env too)",
                          UserWarning, stacklevel=2)
        return value
    if not legacy_value:
        return ""
    global _deprecation_warned
    if not _deprecation_warned:
        _deprecation_warned = True
        warnings.warn(f"{legacy} is deprecated: rename it to {name}, and the other "
                      f"perception-service settings to {service_key('SAM', '*')} and {service_key('GRASP', '*')} "
                      "likewise (in .env too)", FutureWarning, stacklevel=2)
    return legacy_value


__all__ = ["SERVICE_FIELDS", "legacy_renames", "legacy_service_key", "service_key", "service_setting"]
