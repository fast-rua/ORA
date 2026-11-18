"""Closed configuration contracts for Native Context management v4-v10.

v4-v7 are recoverable-first: older ActionSteps age into recoverable artifact
Pointers (L1-L4) before anything is summarized. v8 is append-only: the
projected history only grows until a rollout compaction (L5/L6), and fixed
camera images are delivered by a camera-image mode instead of aging. v9 is v8
plus exchange aging: at a lossless Provider-input watermark, one rewrite
externalizes the large parts of every exchange (an assistant message and what
follows it) older than an exact exchange tail into recoverable-artifact
Pointers, and a hard backstop keeps every request under the hard limit; its
camera-image mode ``exchange_window_v1`` ages camera images with their
exchange too. v10 is v9 whose arguments may never age, whose aged results may
keep their structured fields, whose L5 may be by ratio and whose L6 and exact
tail may count exchanges. A v6 object may also carry two optional fields
(2026-11-14): ``backstop_exact_tail_exchanges`` (the hard backstop as its guard)
and ``truncated_reply_reasoning`` (a truncated reply sent without its
reasoning); without them it is the v6 every earlier plan carries.

Each version's fields and their meanings are the ledger in
``management_fields.py``; this module validates against it and re-exports it.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping

from .management_fields import (
    APPEND_ONLY_SCHEMAS, CAMERA_EXCHANGE_WINDOW, CAMERA_IMAGE_MODES, CAMERA_INLINE_UNTIL_COMPACTION,
    CAMERA_ROLLING_TAIL, COMPACTION_UNITS, CONTEXT_MANAGEMENT_V10_FIELDS, CONTEXT_MANAGEMENT_V10_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V10_SCHEMA, CONTEXT_MANAGEMENT_V4_FIELDS, CONTEXT_MANAGEMENT_V4_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V4_SCHEMA, CONTEXT_MANAGEMENT_V5_FIELDS, CONTEXT_MANAGEMENT_V5_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V5_SCHEMA, CONTEXT_MANAGEMENT_V6_FIELDS, CONTEXT_MANAGEMENT_V6_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V6_OPTIONAL_FIELD_SPECS, CONTEXT_MANAGEMENT_V6_SCHEMA, CONTEXT_MANAGEMENT_V7_FIELDS, CONTEXT_MANAGEMENT_V7_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V7_SCHEMA, CONTEXT_MANAGEMENT_V8_FIELDS, CONTEXT_MANAGEMENT_V8_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V8_SCHEMA, CONTEXT_MANAGEMENT_V9_FIELDS, CONTEXT_MANAGEMENT_V9_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V9_SCHEMA, DEFAULT_IMAGE_ESTIMATE_TOKENS, EPISODE_TRIGGER_RATIO_FIELD, EXCHANGE_AGING_SCHEMAS,
    IMAGE_ESTIMATE_TOKENS_FIELD, LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS, MIN_CAMERA_TAIL_OBSERVATIONS,
    POINTER_FREE_SCHEMAS, REASONING_BINDINGS, REASONING_REPLAY_ALL, REASONING_REPLAY_MODES, REASONING_REPLAY_RECENT,
    REASONING_REPLAY_UNAGED, RESULT_PREVIEWS, TAIL_AGING_SCHEMAS, TAIL_AGING_STRIDE_FIELD,
    TRUNCATED_REPLY_REASONING_WITHHELD, WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS,
)


class ContextManagementConfigError(ValueError):
    """The closed v4-v9 Context configuration is incomplete or inconsistent."""


def validate_context_management_v4(value: Any) -> dict[str, Any]:
    """Validate and normalize the one closed Context Management v4 object."""

    return _validate_closed(
        value, version="v4", schema=CONTEXT_MANAGEMENT_V4_SCHEMA,
        specs=CONTEXT_MANAGEMENT_V4_FIELD_SPECS,
    )


def validate_context_management_v5(value: Any) -> dict[str, Any]:
    """Validate and normalize the one closed Context Management v5 object."""

    return _validate_closed(
        value, version="v5", schema=CONTEXT_MANAGEMENT_V5_SCHEMA,
        specs=CONTEXT_MANAGEMENT_V5_FIELD_SPECS,
    )


def validate_context_management_v6(value: Any) -> dict[str, Any]:
    """Validate and normalize the one closed Context Management v6 object."""

    return _validate_closed(
        value, version="v6", schema=CONTEXT_MANAGEMENT_V6_SCHEMA,
        specs=CONTEXT_MANAGEMENT_V6_FIELD_SPECS, optional=CONTEXT_MANAGEMENT_V6_OPTIONAL_FIELD_SPECS,
    )


def validate_context_management_v7(value: Any) -> dict[str, Any]:
    """Validate and normalize the one closed Context Management v7 object."""

    return _validate_closed(
        value, version="v7", schema=CONTEXT_MANAGEMENT_V7_SCHEMA,
        specs=CONTEXT_MANAGEMENT_V7_FIELD_SPECS,
    )


def validate_context_management_v8(value: Any) -> dict[str, Any]:
    """Validate and normalize the one closed Context Management v8 object."""

    return _validate_closed(
        value, version="v8", schema=CONTEXT_MANAGEMENT_V8_SCHEMA,
        specs=CONTEXT_MANAGEMENT_V8_FIELD_SPECS,
    )


def validate_context_management_v9(value: Any) -> dict[str, Any]:
    """Validate and normalize the one closed Context Management v9 object."""

    return _validate_closed(
        value, version="v9", schema=CONTEXT_MANAGEMENT_V9_SCHEMA,
        specs=CONTEXT_MANAGEMENT_V9_FIELD_SPECS,
    )


def validate_context_management_v10(value: Any) -> dict[str, Any]:
    """Validate and normalize the one closed Context Management v10 object."""

    return _validate_closed(
        value, version="v10", schema=CONTEXT_MANAGEMENT_V10_SCHEMA, specs=CONTEXT_MANAGEMENT_V10_FIELD_SPECS,
    )


_VALIDATORS: dict[str, Callable[[Any], dict[str, Any]]] = {
    CONTEXT_MANAGEMENT_V4_SCHEMA: validate_context_management_v4,
    CONTEXT_MANAGEMENT_V5_SCHEMA: validate_context_management_v5,
    CONTEXT_MANAGEMENT_V6_SCHEMA: validate_context_management_v6,
    CONTEXT_MANAGEMENT_V7_SCHEMA: validate_context_management_v7,
    CONTEXT_MANAGEMENT_V8_SCHEMA: validate_context_management_v8,
    CONTEXT_MANAGEMENT_V9_SCHEMA: validate_context_management_v9,
    CONTEXT_MANAGEMENT_V10_SCHEMA: validate_context_management_v10,
}


def validate_context_management(value: Any) -> dict[str, Any]:
    """Validate the lossless-retention object against the schema it names."""

    if not isinstance(value, Mapping):
        raise ContextManagementConfigError("context_management must be an object")
    schema = value.get("schema")
    validator = _VALIDATORS.get(schema) if isinstance(schema, str) else None
    if validator is None:
        raise ContextManagementConfigError("unsupported context_management schema")
    return validator(value)


def _validate_closed(
    value: Any, *, version: str, schema: str,
    specs: Mapping[str, Mapping[str, str]],
    optional: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """``specs`` are required; an ``optional`` field may be absent and is checked like the others when present."""

    if not isinstance(value, Mapping):
        raise ContextManagementConfigError(
            f"context_management {version} must be an object"
        )
    optional = optional or {}
    fields = frozenset(specs) | (set(optional) & set(value))
    if set(value) != fields:
        missing = sorted(fields - set(value))
        extra = sorted(set(value) - fields)
        raise ContextManagementConfigError(
            f"context_management {version} fields drifted; "
            f"missing={missing}, extra={extra}"
        )
    specs = {**specs, **{field: spec for field, spec in optional.items() if field in value}}
    if value.get("schema") != schema:
        raise ContextManagementConfigError(
            f"unsupported context_management {version} schema"
        )

    normalized = copy.deepcopy(dict(value))
    for field, spec in specs.items():
        kind = spec["kind"]
        item = normalized[field]
        if kind == "schema":
            continue
        if kind in {"positive_int", "positive_int_or_null"}:
            nullable = kind == "positive_int_or_null"
            if (item is not None or not nullable) and (isinstance(item, bool) or not isinstance(item, int) or item <= 0):
                raise ContextManagementConfigError(
                    f"context_management.{field} must be positive{' or null' if nullable else ''}"
                )
        elif kind == "nonnegative_int":
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ContextManagementConfigError(
                    f"context_management.{field} must be non-negative"
                )
        elif kind == "required_true":
            if item is not True:
                raise ContextManagementConfigError(
                    f"context_management.{field} must be true"
                )
        elif kind == "ratio":
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not 0 < float(item) < 1
            ):
                raise ContextManagementConfigError(
                    f"context_management.{field} must be in (0,1)"
                )
            normalized[field] = float(item)
        elif kind == "camera_images":
            if item not in CAMERA_IMAGE_MODES:
                raise ContextManagementConfigError(
                    f"context_management.{field} must be one of {sorted(CAMERA_IMAGE_MODES)}"
                )
        elif kind in {"camera_window", "reasoning_window"}:
            if item is not None and (
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
            ):
                raise ContextManagementConfigError(
                    f"context_management.{field} must be a positive integer or null"
                )
        elif kind == "reasoning_replay":
            if item not in REASONING_REPLAY_MODES:
                raise ContextManagementConfigError(
                    f"context_management.{field} must be one of {sorted(REASONING_REPLAY_MODES)}"
                )
        elif kind == "ratio_or_null":
            if item is not None and (isinstance(item, bool) or not isinstance(item, (int, float))
                                     or not 0 < float(item) < 1):
                raise ContextManagementConfigError(f"context_management.{field} must be in (0,1)")
            normalized[field] = None if item is None else float(item)
        elif kind == "truncated_reply_reasoning":
            if item != TRUNCATED_REPLY_REASONING_WITHHELD:
                raise ContextManagementConfigError(
                    f"context_management.{field} must be {TRUNCATED_REPLY_REASONING_WITHHELD!r} when present"
                )
        elif kind in {"result_preview", "compaction_unit", "reasoning_binding"}:
            allowed = {"result_preview": RESULT_PREVIEWS, "compaction_unit": COMPACTION_UNITS,
                       "reasoning_binding": REASONING_BINDINGS}[kind]
            if item not in allowed:
                raise ContextManagementConfigError(f"context_management.{field} must be one of {sorted(allowed)}")
        else:  # pragma: no cover - the static ledger is reviewed with tests.
            raise ContextManagementConfigError(
                f"unsupported context_management field kind: {kind}"
            )

    if normalized["causal_exact_tail_action_steps"] != 3:
        raise ContextManagementConfigError(
            "context_management.causal_exact_tail_action_steps must equal 3"
        )
    if (
        normalized["checkpoint_target_estimated_tokens"]
        > normalized["checkpoint_max_estimated_tokens"]
    ):
        raise ContextManagementConfigError(
            "checkpoint target must not exceed its configured soft maximum"
        )
    if schema in APPEND_ONLY_SCHEMAS:
        _check_append_only(normalized, version=version, schema=schema)
        if schema in EXCHANGE_AGING_SCHEMAS:
            _check_exchange_aging(normalized, version=version)
        return normalized
    if normalized["eager_preview_bytes"] >= normalized["eager_large_result_bytes"]:
        raise ContextManagementConfigError(
            "eager preview must be smaller than the large-result threshold"
        )
    _check_layer_order(normalized, version=version, schema=schema)
    return normalized


def _check_append_only(normalized: Mapping[str, Any], *, version: str, schema: str) -> None:
    """The camera window matches its mode, which the schema supports; L6 targets below its fallback."""

    window = normalized["camera_tail_observations"]
    if normalized["camera_images"] == CAMERA_ROLLING_TAIL:
        if window is None or window < MIN_CAMERA_TAIL_OBSERVATIONS:
            raise ContextManagementConfigError(
                f"context_management.camera_tail_observations must be at least "
                f"{MIN_CAMERA_TAIL_OBSERVATIONS} for {CAMERA_ROLLING_TAIL}"
            )
    elif window is not None:
        raise ContextManagementConfigError(
            f"context_management.camera_tail_observations must be null for "
            f"{normalized['camera_images']}"
        )
    if normalized["camera_images"] == CAMERA_EXCHANGE_WINDOW and schema not in EXCHANGE_AGING_SCHEMAS:
        raise ContextManagementConfigError(
            f"context_management.camera_images {CAMERA_EXCHANGE_WINDOW} needs exchange aging "
            f"(Context {version} has none)"
        )
    if not normalized["emergency_target_ratio"] < normalized["emergency_fallback_ratio"]:
        raise ContextManagementConfigError(
            f"Context {version} requires emergency target < emergency fallback"
        )


def _check_exchange_aging(normalized: Mapping[str, Any], *, version: str) -> None:
    """Aging precedes L6 on every window (ratios always, absolute watermarks when L6 has one); tails agree."""

    absolute = normalized["compaction_trigger_tokens"]
    if not (normalized["exchange_aging_trigger_ratio"] < normalized["emergency_fallback_ratio"]
            and (absolute is None or normalized["exchange_aging_trigger_tokens"] < absolute)):
        raise ContextManagementConfigError(
            f"Context {version} requires the exchange aging watermark below the L6 watermark "
            "(exchange_aging_trigger_ratio < emergency_fallback_ratio, and exchange_aging_trigger_tokens < "
            "compaction_trigger_tokens when L6 has an absolute watermark)"
        )
    if not normalized["backstop_exact_tail_exchanges"] <= normalized["exact_tail_exchanges"]:
        raise ContextManagementConfigError(
            "context_management.backstop_exact_tail_exchanges must not exceed exact_tail_exchanges"
        )
    preview, arguments = normalized["aged_preview_bytes"], normalized["aged_argument_bytes"]
    if not ((arguments is None or preview < arguments) and preview < normalized["aged_result_bytes"]):
        raise ContextManagementConfigError(
            "context_management.aged_preview_bytes must be smaller than the aged argument and result thresholds"
        )
    if "compaction_unit" in normalized:
        _check_v10(normalized)
    window = normalized["reasoning_tail_exchanges"]
    if normalized["reasoning_replay"] != REASONING_REPLAY_RECENT and window is not None:
        raise ContextManagementConfigError(
            f"context_management.reasoning_tail_exchanges must be null for {normalized['reasoning_replay']}"
        )
    if normalized["reasoning_replay"] == REASONING_REPLAY_RECENT and window is None:
        raise ContextManagementConfigError(
            f"context_management.reasoning_tail_exchanges must be a positive integer for {REASONING_REPLAY_RECENT}"
        )


def _check_v10(normalized: Mapping[str, Any]) -> None:
    """L5 by tokens or by ratio, the ratio below L6's."""

    ratio = normalized[EPISODE_TRIGGER_RATIO_FIELD]
    if (normalized["episode_checkpoint_trigger_tokens"] is None) == (ratio is None):
        raise ContextManagementConfigError(
            "context_management sets exactly one of episode_checkpoint_trigger_tokens and "
            f"{EPISODE_TRIGGER_RATIO_FIELD}"
        )
    if ratio is not None and not ratio < normalized["emergency_fallback_ratio"]:
        raise ContextManagementConfigError(
            f"Context v10 requires {EPISODE_TRIGGER_RATIO_FIELD} < emergency_fallback_ratio"
        )


def _check_layer_order(
    normalized: Mapping[str, Any], *, version: str, schema: str,
) -> None:
    """Window-independent order of the L4, L5 and L6 watermarks.

    v4/v5 order the L5 ratio below the L6 target. v6 has no L5 ratio; its
    absolute L5 must lie above the absolute L4 estimate floor, which is the
    order at the 200K window both figures were chosen for. Above that window
    the order depends on the bound model (see ``thresholds``).
    """

    if schema in WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS:
        if not (
            normalized[EPISODE_TRIGGER_RATIO_FIELD]
            < normalized["emergency_target_ratio"]
            <= normalized["trigger_provider_input_ratio"]
            < normalized["emergency_fallback_ratio"]
        ):
            raise ContextManagementConfigError(
                f"Context {version} ratios require episode trigger < emergency target "
                "<= pressure trigger < emergency fallback"
            )
        return
    if not (
        normalized["emergency_target_ratio"]
        <= normalized["trigger_provider_input_ratio"]
        < normalized["emergency_fallback_ratio"]
    ):
        raise ContextManagementConfigError(
            f"Context {version} ratios require emergency target "
            "<= pressure trigger < emergency fallback"
        )
    if not (
        normalized["trigger_estimated_tokens"]
        < normalized["episode_checkpoint_trigger_tokens"]
    ):
        raise ContextManagementConfigError(
            f"Context {version} requires the L4 estimate floor "
            "(trigger_estimated_tokens) below the absolute L5 watermark "
            "(episode_checkpoint_trigger_tokens)"
        )


__all__ = [
    "APPEND_ONLY_SCHEMAS",
    "CAMERA_EXCHANGE_WINDOW",
    "CAMERA_IMAGE_MODES",
    "CAMERA_INLINE_UNTIL_COMPACTION",
    "CAMERA_ROLLING_TAIL",
    "COMPACTION_UNITS",
    "CONTEXT_MANAGEMENT_V10_FIELDS",
    "CONTEXT_MANAGEMENT_V10_FIELD_SPECS",
    "CONTEXT_MANAGEMENT_V10_SCHEMA",
    "CONTEXT_MANAGEMENT_V4_FIELDS",
    "CONTEXT_MANAGEMENT_V4_FIELD_SPECS",
    "CONTEXT_MANAGEMENT_V4_SCHEMA",
    "CONTEXT_MANAGEMENT_V5_FIELDS",
    "CONTEXT_MANAGEMENT_V5_FIELD_SPECS",
    "CONTEXT_MANAGEMENT_V5_SCHEMA",
    "CONTEXT_MANAGEMENT_V6_FIELDS",
    "CONTEXT_MANAGEMENT_V6_FIELD_SPECS",
    "CONTEXT_MANAGEMENT_V6_OPTIONAL_FIELD_SPECS",
    "CONTEXT_MANAGEMENT_V6_SCHEMA",
    "CONTEXT_MANAGEMENT_V7_FIELDS",
    "CONTEXT_MANAGEMENT_V7_FIELD_SPECS",
    "CONTEXT_MANAGEMENT_V7_SCHEMA",
    "CONTEXT_MANAGEMENT_V8_FIELDS",
    "CONTEXT_MANAGEMENT_V8_FIELD_SPECS",
    "CONTEXT_MANAGEMENT_V8_SCHEMA",
    "CONTEXT_MANAGEMENT_V9_FIELDS",
    "CONTEXT_MANAGEMENT_V9_FIELD_SPECS",
    "CONTEXT_MANAGEMENT_V9_SCHEMA",
    "ContextManagementConfigError",
    "DEFAULT_IMAGE_ESTIMATE_TOKENS",
    "EPISODE_TRIGGER_RATIO_FIELD",
    "EXCHANGE_AGING_SCHEMAS",
    "IMAGE_ESTIMATE_TOKENS_FIELD",
    "LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS",
    "MIN_CAMERA_TAIL_OBSERVATIONS",
    "POINTER_FREE_SCHEMAS",
    "REASONING_BINDINGS",
    "REASONING_REPLAY_ALL",
    "REASONING_REPLAY_MODES",
    "REASONING_REPLAY_RECENT",
    "REASONING_REPLAY_UNAGED",
    "RESULT_PREVIEWS",
    "TAIL_AGING_SCHEMAS",
    "TAIL_AGING_STRIDE_FIELD",
    "TRUNCATED_REPLY_REASONING_WITHHELD",
    "WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS",
    "validate_context_management",
    "validate_context_management_v10",
    "validate_context_management_v4",
    "validate_context_management_v5",
    "validate_context_management_v6",
    "validate_context_management_v7",
    "validate_context_management_v8",
    "validate_context_management_v9",
]
