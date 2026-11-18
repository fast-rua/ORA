"""``policy.context_compactor``: the rollout compactor on its own Provider profile.

The lossless Context retention's compactor (the Pi core's ``RolloutCheckpointCompactor``)
writes the episode (L5) and emergency (L6) checkpoints, one tool-free call each.
By default it calls the policy's own provider and model. A RunSpec may bind it
to another Provider profile instead, for example a small local policy model with
its checkpoints written by an API model::

    policy:
      context_compactor:
        provider_profile: deepseek_flash_openai_low_16k_v1
        provider_catalog: path/to/catalog.yaml   # optional

The profile resolves as ``provider_policy`` resolves a policy's: built-in
profiles first, then the user's ``providers.yaml``, or only the named catalog.
The plan records the expanded binding under the same key (schema
``ora.context-compactor-binding.v1``): wire fields and credential variable
names, never a key. Without the field nothing is added to the plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ora.agent.memory.artifacts import LOSSLESS_CONTEXT_RETENTION

from .native_provider import resolve_native_provider_endpoint, resolve_native_provider_protocol
from .inputs import provider_policy
from .play import episode_mode
from .values import RunSpecError, file_sha256

SCHEMA = "ora.context-compactor-binding.v1"
#: What a RunSpec names; the resolver writes the binding in their place.
INPUT_FIELDS = frozenset({"provider_profile", "provider_catalog"})
#: The output ceiling of every compactor call (``maxOutputTokens`` in ``ora-pi/src/context/rollout_compactor.ts``).
COMPACTOR_OUTPUT_TOKENS = 8192
#: Wire fields every binding records, then those it records when the profile has them.
REQUIRED_FIELDS = (
    "provider", "model", "endpoint", "endpoint_transport", "base_url_env", "api_key_env",
    "proxy_env", "reasoning_effort", "stream", "supports_images", "context_window_tokens",
    "max_output_tokens", "provider_recovery",
)
OPTIONAL_FIELDS = (
    "auth_scheme", "output_token_parameter", "reasoning_field", "request_options",
    "provider_recovery_window_s", "provider_session_affinity",
)


def resolve_context_compactor(
    policy: dict[str, Any], spec: Mapping[str, Any], *, root: Path,
) -> None:
    """Replace ``policy["context_compactor"]`` by its resolved binding, or refuse it."""

    raw = policy["context_compactor"]
    if not isinstance(raw, Mapping) or set(raw) - INPUT_FIELDS or "provider_profile" not in raw:
        raise RunSpecError(
            "policy.context_compactor takes provider_profile and an optional provider_catalog"
        )
    profile_id = raw["provider_profile"]
    if not isinstance(profile_id, str) or not profile_id:
        raise RunSpecError("policy.context_compactor.provider_profile must be a non-empty string")
    if policy.get("kind") != "native_api":
        raise RunSpecError("policy.context_compactor requires a native_api policy")
    if (
        policy.get("context_retention") != LOSSLESS_CONTEXT_RETENTION
        or policy.get("summary_policy") != "rollout_checkpoint_v1"
    ):
        raise RunSpecError(
            f"policy.context_compactor requires context_retention={LOSSLESS_CONTEXT_RETENTION} "
            "and summary_policy=rollout_checkpoint_v1 (the only retention with a compactor)"
        )
    if episode_mode(spec) == "play":
        raise RunSpecError(
            "policy.context_compactor does not apply to a free-play session (Pi compacts its own context)"
        )
    catalog = _catalog(raw.get("provider_catalog"), root)
    binding = _binding(profile_id, catalog, policy)
    if binding["supports_images"] is not True:
        raise RunSpecError(
            "policy.context_compactor needs an image-capable profile: compactor packets are multimodal"
        )
    if binding["max_output_tokens"] < COMPACTOR_OUTPUT_TOKENS:
        raise RunSpecError(
            f"policy.context_compactor needs max_output_tokens >= {COMPACTOR_OUTPUT_TOKENS}, "
            "the compactor's fixed output ceiling"
        )
    if binding["context_window_tokens"] < policy["context_window_tokens"]:
        raise RunSpecError(
            "policy.context_compactor needs a context window at least the policy's "
            f"({binding['context_window_tokens']} < {policy['context_window_tokens']})"
        )
    policy["context_compactor"] = binding


def _catalog(value: Any, root: Path) -> tuple[Path, dict[str, Any]] | None:
    """The named catalog file and its ``{path, sha256}`` record, or None for the default lookup."""

    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RunSpecError("policy.context_compactor.provider_catalog must be a path")
    given = Path(value)
    expanded = given.expanduser()
    if not expanded.is_absolute() and ".." in given.parts:
        raise RunSpecError(
            "policy.context_compactor.provider_catalog is an absolute path or one inside the project"
        )
    path = expanded if expanded.is_absolute() else root / given
    if not path.is_file():
        raise RunSpecError(f"policy.context_compactor.provider_catalog is unavailable: {value}")
    return path, {"path": given.as_posix(), "sha256": file_sha256(path)}


def _binding(
    profile_id: str, catalog: tuple[Path, dict[str, Any]] | None, policy: Mapping[str, Any],
) -> dict[str, Any]:
    """The selected profile's wire fields, checked as a native_api policy's are."""

    from ora.agent.providers.profiles import load_builtin_provider_profiles

    try:
        selected = provider_policy(profile_id, catalog=None if catalog is None else catalog[0])
    except (RuntimeError, ValueError, OSError) as exc:
        raise RunSpecError(f"policy.context_compactor: {exc}") from exc
    if selected.get("gateway_client_binding") is True:
        raise RunSpecError("policy.context_compactor does not take a gateway-bound profile")
    # A profile's recovery window selects the deadline recovery (provider_policy);
    # otherwise the compactor recovers as the policy does.
    selected.setdefault("provider_recovery", policy.get("provider_recovery", "legacy_v1"))
    try:  # the policy's own checks: a binding is admitted exactly as such a policy would be
        resolve_native_provider_protocol(selected, None)
        resolve_native_provider_endpoint(selected)
    except RunSpecError as exc:
        raise RunSpecError(f"policy.context_compactor: {exc}") from exc
    if catalog is not None:
        source = "catalog"
    elif profile_id in load_builtin_provider_profiles():
        source = "builtin"
    else:
        source = "user"
    binding: dict[str, Any] = {
        "schema": SCHEMA, "provider_profile": profile_id, "profile_source": source,
        **({"provider_catalog": catalog[1]} if catalog is not None else {}),
    }
    binding.update({field: selected[field] for field in REQUIRED_FIELDS})
    binding.update({field: selected[field] for field in OPTIONAL_FIELDS if selected.get(field) is not None})
    return binding


__all__ = ["COMPACTOR_OUTPUT_TOKENS", "SCHEMA", "resolve_context_compactor"]
