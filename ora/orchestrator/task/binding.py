"""``ora.run-binding.v1``: where one run of a task executes.

A binding holds what a task file may not: the GPU, the RoboDojo resource
profile, the Provider catalog and profile of the model replica, the
perception readiness report, an optional admission cohort, the environment
the run needs (service ports, execution host, workspace) and the model names
the replica serves. ``ora task run`` reads it from ``--binding FILE``, or
from the allocation file a scheduler names in ``ORA_ROUTER_ALLOCATION``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

SCHEMA = "ora.run-binding.v1"
ALLOCATION_VARIABLE = "ORA_ROUTER_ALLOCATION"
KEYS = {"schema", "gpu", "resource_profile", "provider_catalog", "provider_profile", "service_report",
        "cohort", "config", "env", "models"}
REQUIRED = ("gpu", "provider_catalog", "provider_profile")


class BindingError(ValueError):
    """No usable run binding."""


def validate_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise BindingError(f"a run binding has schema {SCHEMA}")
    binding = {"env": {}, "models": [], **dict(value)}
    unknown = sorted(set(binding) - KEYS)
    missing = [key for key in REQUIRED if binding.get(key) in (None, "")]
    if unknown or missing:
        raise BindingError(f"run binding: unknown keys {unknown}, missing {missing}")
    if not isinstance(binding["env"], Mapping) or not isinstance(binding["models"], list):
        raise BindingError("run binding env is a mapping and models a list")
    binding["gpu"] = str(binding["gpu"])
    binding["env"] = {str(key): str(value) for key, value in binding["env"].items()}
    return binding


def binding_from_allocation(value: Mapping[str, Any]) -> dict[str, Any]:
    """A scheduler allocation (its ``allocation`` object, or the object itself) as a run binding."""

    allocation = value.get("allocation") or value
    build = allocation.get("build_spec") or {}
    lane = allocation.get("lane") or {}
    llm = allocation.get("llm") or {}
    if lane.get("card") is None or not build.get("provider_profile"):
        raise BindingError("the allocation has no lane card or no model replica")
    return validate_binding({
        "schema": SCHEMA, "gpu": lane["card"],
        "resource_profile": build.get("resource_profile") or lane.get("resource_profile"),
        "provider_catalog": build["provider_catalog"], "provider_profile": build["provider_profile"],
        "service_report": build.get("service_report"),
        "env": {**(lane.get("env") or {}), **(allocation.get("env") or {})},
        "models": [name for name in (llm.get("model"), llm.get("served_model")) if name],
    })


def load_binding(path: Path | None = None, environ: Mapping[str, str] = os.environ) -> dict[str, Any]:
    """``path`` (a run binding or an allocation file), else the allocation ``ORA_ROUTER_ALLOCATION`` names."""

    if path is None:
        named = environ.get(ALLOCATION_VARIABLE)
        if not named:
            raise BindingError(f"no --binding and no {ALLOCATION_VARIABLE}")
        path = Path(named)
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, Mapping) and value.get("schema") == SCHEMA:
        return validate_binding(value)
    return binding_from_allocation(value or {})


def provider_profile(binding: Mapping[str, Any]) -> dict[str, Any]:
    """The binding's Provider profile from its catalog."""

    catalog = yaml.safe_load(Path(binding["provider_catalog"]).read_text(encoding="utf-8")) or {}
    profile = (catalog.get("profiles") or {}).get(binding["provider_profile"])
    if not isinstance(profile, Mapping):
        raise BindingError(f"{binding['provider_catalog']} has no profile {binding['provider_profile']}")
    return dict(profile)


def check_model(task: Mapping[str, Any], binding: Mapping[str, Any], profile: Mapping[str, Any]) -> None:
    """The replica the run is bound to must serve the model the task names."""

    served = {*binding.get("models", ()), profile.get("model")} - {None}
    if task["model"] not in served:
        raise BindingError(f"task model {task['model']!r} is not what the bound replica serves: {sorted(served)}")


def run_environment(binding: Mapping[str, Any], profile: Mapping[str, Any],
                    environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` with the binding's variables and the profile's endpoint and model variables set."""

    result = {**environ, **binding.get("env", {})}
    for variable, value in ((profile.get("base_url_env"), profile.get("endpoint")),
                            (profile.get("model_env"), profile.get("model"))):
        if variable and value and not result.get(variable):
            result[variable] = str(value)
    return result
