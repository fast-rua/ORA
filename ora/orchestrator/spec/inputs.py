"""RunSpec inputs the maintained builders share: Provider, resident services, admission.

The maintained builders (``backends.robodojo.spec`` for RoboDojo, ``benchmark`` for
RoboTwin and LIBERO-Pro) name no ToolSurface, prompt bundle or Context, so the
default resolver composes the default catalog's instance for the backend. What
they do supply that does not depend on the benchmark lives here: the policy
block of one Provider profile, the resident-service binding of a readiness
report to the surface the RunSpec runs, and the single-attempt launch
admission of one cell.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

from ora.agent.composition.baseline import default_baseline_id, load_native_baseline
from ora.orchestrator.paths import PROJECT_ROOT

SPEC_SCHEMA = "ora.run-spec.v1"
#: The provider recovery a profile's ``recovery_window_s`` applies under (``providers.recovery``):
#: there the window is each attempt's first-token limit and the call's no-content window.
RECOVERY_POLICY = "first_token_deadline_v1"
ADMISSION_SCHEMA = "ora.run-launch-admission.v1"
#: Provider-profile fields a native_api policy takes when the profile has them.
PROVIDER_FIELDS = (
    "model", "endpoint", "endpoint_transport", "gateway_client_binding", "model_env", "base_url_env", "api_key_env",
    "proxy_env", "auth_scheme", "reasoning_effort", "context_window_tokens", "max_output_tokens",
    "stream", "supports_images", "output_token_parameter", "tool_strictness", "reasoning_field",
    "request_options",
)
#: Resident services and the tool-client variables that must name the bound endpoint.
SERVICE_VARIABLES = (("segmentation", "ORA_SEGMENTATION_URL"),
                     ("grasp_proposals", "ORA_GRASP_URL"))
#: How the backends are named in prose and refusals.
ENVIRONMENT_NAMES = {"robodojo": "RoboDojo", "robotwin": "RoboTwin", "libero_pro": "LIBERO-Pro"}


def default_instance(environment: str, root: Path = PROJECT_ROOT,
                     baseline: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The environment's default catalog (``default_baseline_id``, or the registered ``baseline``) and its instance for
    ``environment``; refused when it has none."""
    catalog = load_native_baseline(root, default_baseline_id(environment) if baseline is None else baseline)
    instance = catalog["instances"].get(environment)
    if instance is None:
        name = ENVIRONMENT_NAMES.get(environment, environment)
        raise ValueError(f"the {'default ' if baseline is None else ''}baseline {catalog['id']} has no {name} instance")
    return catalog, instance


def provider_policy(profile_id: str | None = None, *, catalog: Path | None = None) -> dict[str, Any]:
    """A native_api policy bound to one Provider profile; the shared default when none is named.

    Without ``catalog``, ``profile_id`` may also name one of the user's own profiles
    (``ora providers add``).
    """
    from ora.agent.providers.profiles import select_provider_profile
    from ora.agent.providers.user_store import select_profile

    selected, profile = (select_provider_profile(catalog, profile_id=profile_id) if catalog is not None
                         else select_profile(profile_id))
    policy = {"kind": "native_api", "provider": profile["protocol"], "provider_profile": selected,
              **{key: copy.deepcopy(profile[key]) for key in PROVIDER_FIELDS if key in profile},
              "provider_qualification_schema": profile["qualification_schema"],
              "provider_qualification_version": profile["qualification_version"]}
    if profile.get("session_affinity") is not None:
        policy["provider_session_affinity"] = profile["session_affinity"]
    if profile.get("recovery_window_s") is not None:
        # A slow self-hosted model can take past the 180 s default to its first
        # token. The window belongs to the first-token recovery, which the default
        # baseline already names; a spec outside it (the Synthetic one of `ora
        # interactive`) would otherwise stay on legacy retries and refuse the window.
        policy["provider_recovery"] = RECOVERY_POLICY
        policy["provider_recovery_window_s"] = profile["recovery_window_s"]
    return policy


def service_prerequisite(report: Path, environment: str, *, root: Path = PROJECT_ROOT,
                         baseline: str | None = None, tool_surface: str | None = None) -> dict[str, Any]:
    """Bind a readiness report (``ora services qualify``) to the ``environment`` surface.

    The surface is the ``tool_surface`` the RunSpec selects (``policy.tool_surface``),
    else that of the registered ``baseline`` it selects (``policy.baseline``), else
    the default baseline's. A selected toolkit surface without a baseline takes the
    baseline it brings (``TOOLKIT_BASELINES``), as the resolver does, for the
    capability scope. Launch preflight requires the binding to name the surface
    the plan resolves, so bind after the policy is selected.

    ``operator.binding`` refuses a report that is older than 24 hours, failed,
    of another service protocol version (unless ``ORA_SERVICE_PROTOCOL_CHECK=off``)
    or for other endpoints than the configured ones
    (``ORA_SAM_PORT``/``ORA_GRASP_PORT``); its error is raised unchanged.
    """
    from ora.agent.composition import resolve_tool_surface
    from ora.agent.composition.surface.toolkit_surface import declared_required_grants
    from ora.agent.composition.surface.toolkit_profiles import TOOLKIT_BASELINES
    from ora.agent.model_services import operator as services

    if baseline is None and tool_surface is not None:
        baseline = TOOLKIT_BASELINES.get(tool_surface)
    catalog, instance = default_instance(environment, root, baseline)
    profile = tool_surface or instance["tool_surface"]
    binding = services.binding(Path(report), profile, catalog["required_service_capabilities"])
    surface = resolve_tool_surface(profile, toolkit_grants=declared_required_grants(profile))
    missing = set(binding["capability_scope"]) - set(surface.capabilities)
    if missing:
        name = ENVIRONMENT_NAMES.get(environment, environment)
        raise ValueError(f"the selected {name} surface lacks qualified service capabilities: {sorted(missing)}")
    return binding


def single_attempt_admission(cohort_id: str, cell: str) -> dict[str, Any]:
    """One attempt for ``cell`` in ``cohort_id``; a retry needs a new cohort or cell."""
    return {"schema": ADMISSION_SCHEMA, "mode": "single_attempt_cell", "cohort_id": cohort_id,
            "cell_id": cell, "retry_allowed": False}


def bind_service_environment(spec: Mapping[str, Any]) -> None:
    """Point the tool clients at the endpoints the bound readiness report names.

    A conflicting value is an error rather than something to overwrite: an
    operator who set it meant something by it. A spec without a bound service
    set needs nothing here; preflight binds and checks it.
    """
    services = (spec.get("prerequisite") or {}).get("services")
    if not isinstance(services, Mapping):
        return
    for name, variable in SERVICE_VARIABLES:
        endpoint = services[name]["endpoint"]
        if os.environ.get(variable, endpoint) != endpoint:
            raise RuntimeError(f"{variable} conflicts with the bound resident service "
                               f"({os.environ[variable]} against {endpoint})")
        os.environ[variable] = endpoint


__all__ = [
    "ADMISSION_SCHEMA", "ENVIRONMENT_NAMES", "PROVIDER_FIELDS", "SERVICE_VARIABLES", "SPEC_SCHEMA",
    "bind_service_environment", "default_instance", "provider_policy", "service_prerequisite",
    "single_attempt_admission",
]
