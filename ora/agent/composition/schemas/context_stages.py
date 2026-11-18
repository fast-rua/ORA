"""Context stage lists: the parameter schemas and the resolution of a scheme's stages.

An ``ora.context-scheme.v2`` scheme (``ora.agent.composition.context_scheme``)
lists the stages the agent core runs (``ora-pi/src/context/stages/``), each with
its parameters, and the scheme-level settings of the shared services
(``segments``, ``limits``, ``estimator``, ``references``). Each stage's
parameters have one closed schema, ``ora/policy/context_stages/<stage>.schema.json``,
and the settings ``scheme.schema.json``. Python validates schemes with them here
(filling the defaults they declare) and sends them to the core at session start
(``ora.agent.pi.domain.context_stage_schemas``); the core validates every stage
it builds with the same files and keeps no rules of its own.

A resolved list (``ora.context-stages.v1``) is the plan's ``policy.context_stages``.
The projection manifest, the evaluation and the plan's Context checks still read
a management object, so a stage list resolves only when it is the stage list of
a management version, v6, v8 or v9: ``management_equivalent`` derives that object
(``ora.agent.context.lossless.management`` validates it at plan resolution),
the plan carries both, and the agent core refuses a list that differs from its
own translation of the object (``ora-pi/src/context/stages/scheme.ts``). A stage
list without a management equivalent waits for per-stage evidence checks.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..locked_assets import locked_asset
from ..surface.schema_validation import (
    SchemaViolation, ToolSurfaceError, lint_input_schema, validate_schema_instance,
)

#: The directory of the parameter schemas, relative to the source tree.
STAGES_DIR = "ora/policy/context_stages"
#: This code's source tree, whose schemas a plan's stage list is checked with.
SOURCE_ROOT = Path(__file__).resolve().parents[4]
#: The schema of the scheme-level settings (not a stage).
SCHEME_SETTINGS = "scheme"
#: The plan's ``policy.context_stages``.
STAGE_LIST_SCHEMA = "ora.context-stages.v1"
#: The scheme-level settings, in the order a stage list carries them.
SETTINGS = ("segments", "limits", "estimator", "references")


def stage_schemas(root: Path) -> dict[str, dict[str, Any]]:
    """``{id: schema}`` of every ``<id>.schema.json`` in ``root``'s ``STAGES_DIR`` (the settings as ``scheme``)."""

    schemas: dict[str, dict[str, Any]] = {}
    for file in sorted((Path(root) / STAGES_DIR).glob("*.schema.json")):
        relative = f"{STAGES_DIR}/{file.name}"
        asset = locked_asset(root, relative, label=f"context stage schema {file.name}")
        schema = json.loads(asset.data.decode("utf-8"))
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError(f"{relative} is not an object schema")
        lint_input_schema(schema, source=relative)
        schemas[file.name[: -len(".schema.json")]] = schema
    if SCHEME_SETTINGS not in schemas:
        raise ValueError(f"{STAGES_DIR} has no {SCHEME_SETTINGS}.schema.json")
    return schemas


def _with_defaults(value: Any, schema: Mapping[str, Any]) -> Any:
    """``value`` with every absent property that declares a default filled in (objects, recursively)."""

    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    filled = {key: copy.deepcopy(item) for key, item in value.items()}
    for name, prop in (schema.get("properties") or {}).items():
        if name not in filled and isinstance(prop, Mapping) and "default" in prop:
            filled[name] = copy.deepcopy(prop["default"])
        elif name in filled and isinstance(prop, Mapping) and prop.get("type") == "object":
            filled[name] = _with_defaults(filled[name], prop)
    return filled


def _validated(value: Any, schema: Mapping[str, Any], label: str) -> Any:
    try:
        validate_schema_instance(value, schema, root_path="$input")
    except (SchemaViolation, ToolSurfaceError) as error:
        raise ValueError(f"{label}: {error}") from None
    return value


def resolve_stage_list(root: Path, scheme: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """A scheme's settings and stages as the ``ora.context-stages.v1`` list the core runs."""

    schemas = stage_schemas(root)
    settings = {key: scheme.get(key) for key in SETTINGS}
    settings = _validated(_with_defaults(settings, schemas[SCHEME_SETTINGS]), schemas[SCHEME_SETTINGS],
                          f"{label} settings")
    raw = scheme.get("stages")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label}: stages must be a non-empty list")
    stages: list[dict[str, Any]] = []
    for index, entry in enumerate(raw):
        if (not isinstance(entry, Mapping) or set(entry) - {"stage", "params"}
                or not isinstance(entry.get("stage"), str)):
            raise ValueError(f"{label}: stages[{index}] must be {{stage, params}}")
        stage = entry["stage"]
        if stage == SCHEME_SETTINGS or stage not in schemas:
            raise ValueError(f"{label}: stages[{index}] names an unknown stage {stage!r}")
        if any(item["stage"] == stage for item in stages):
            raise ValueError(f"{label}: stage {stage} is listed twice")
        params = _with_defaults(entry.get("params", {}), schemas[stage])
        stages.append({"stage": stage, "params": _validated(params, schemas[stage], f"{label} {stage}")})
    return {"schema": STAGE_LIST_SCHEMA, **settings, "stages": stages}


# -- the bridge to the management object the evidence reads ----------------------

#: The management versions a stage list may be the stage list of (``context/lossless/management.py``).
CONTEXT_MANAGEMENT_V6_SCHEMA = "ora.native-context-management.v6"
CONTEXT_MANAGEMENT_V8_SCHEMA = "ora.native-context-management.v8"
CONTEXT_MANAGEMENT_V9_SCHEMA = "ora.native-context-management.v9"
CONTEXT_MANAGEMENT_V10_SCHEMA = "ora.native-context-management.v10"
#: The ActionStep exact tail every management version fixes; with an exact tail in exchanges (v10) the segments
#: count exchanges instead and it has no effect.
ACTION_STEP_EXACT_TAIL = 3
#: The L4 estimate share the v6 translation uses (ora-pi pressure_externalize/trigger.ts).
ESTIMATE_PRESSURE_WINDOW_SHARE = 48_000 / (200_000 - 2_048)
#: The reserve and per-image estimate v6 and v8 imply.
RESERVE_TOKENS = 2048
V6_IMAGE_TOKENS = 4096
#: v6's and v8's stage ids, in order.
V6_STAGES = (
    "materialize", "stride_aging", "eager_externalize", "media_window", "pressure_externalize",
    "episode_capsule", "emergency_checkpoint", "over_limit_guard",
)
#: v6 with the hard backstop as its guard (``backstop_exact_tail_exchanges``, since 2026-11-14).
V6_BACKSTOP_STAGES = (*V6_STAGES[:-1], "hard_backstop")
V8_STAGES = ("materialize", "camera_delivery", "episode_capsule", "emergency_checkpoint", "over_limit_guard")
V9_STAGES = (
    "materialize", "exchange_aging", "reasoning_replay", "camera_delivery", "episode_capsule",
    "emergency_checkpoint", "hard_backstop",
)


def _params(stage_list: Mapping[str, Any], expected: Sequence[str], label: str) -> dict[str, dict[str, Any]]:
    ids = [item["stage"] for item in stage_list["stages"]]
    if tuple(ids) != tuple(expected):
        raise ValueError(f"{label}: stages {ids} are not the stages of a management version")
    return {item["stage"]: item["params"] for item in stage_list["stages"]}


def _require(condition: bool, label: str, what: str) -> None:
    if not condition:
        raise ValueError(f"{label}: {what} has no management equivalent")


def _v10_fields(
    stage_list: Mapping[str, Any], params: Mapping[str, Mapping[str, Any]], label: str,
) -> dict[str, Any]:
    """v10's fields beyond v9's, from v9's stages that state an aged result's preview (v10 always does)."""

    segments, aging, reasoning = stage_list["segments"], params["exchange_aging"], params["reasoning_replay"]
    episode, emergency = params["episode_capsule"], params["emergency_checkpoint"]
    unit = segments.get("exact_tail_unit", "action_step")
    _require(emergency.get("unit", "action_step") == unit, label, "an emergency checkpoint and an exact tail in "
             "different units")
    _require(aging.get("keep_anchor", False) == (unit == "exchange"), label,
             "an aging that keeps the anchor exchange other than exactly with an exact tail in exchanges")
    _require(unit != "exchange" or segments["exact_tail"] == aging["exact_tail_exchanges"], label,
             "an exact tail in exchanges other than the aging's")
    _require((episode["trigger_tokens"] is None) != (episode["trigger_ratio"] is None), label,
             "an L5 watermark by both tokens and ratio")
    _require("binding" in reasoning, label, "v10 stages that do not state the reasoning binding")
    return {
        **({"causal_exact_tail_action_steps": ACTION_STEP_EXACT_TAIL} if unit == "exchange" else {}),
        "episode_checkpoint_trigger_ratio": episode["trigger_ratio"],
        "aged_result_preview": aging["result_preview"],
        "compaction_unit": unit,
        "reasoning_binding": reasoning["binding"],
    }


def _v9_management(
    stage_list: Mapping[str, Any], params: Mapping[str, Mapping[str, Any]], label: str,
) -> dict[str, Any]:
    """v9: v8's fields plus the exchange stages' (``stage_list`` has v9's stages); v10's when it states
    an aged result's preview."""

    references, aging = stage_list["references"], params["exchange_aging"]
    reasoning, preview = params["reasoning_replay"], aging["preview"]
    v10 = "result_preview" in aging
    _require(v10 or (aging["argument_bytes"] is not None and "keep_anchor" not in aging
                     and "binding" not in reasoning and params["episode_capsule"]["trigger_tokens"] is not None),
             label, "v9 stages with arguments that never age, a kept anchor, a reasoning binding or an L5 by ratio "
             "(v10 states result_preview)")
    _require(stage_list["segments"]["unit"] == "exchange", label, "v9 stages counted in another unit than exchange")
    _require(references == {"pointers": True, "advertise_over_bytes": None, "preview": {"head_bytes": 0}}, label,
             "v9 references other than Pointers shown, none advertised, no head preview")
    budget = preview["head_bytes"] + preview["tail_bytes"]
    _require(preview["tail_bytes"] == budget // 2, label, "an aged preview split other than half and half")
    return {
        "exchange_aging_trigger_tokens": aging["trigger_tokens"],
        "exchange_aging_trigger_ratio": aging["trigger_ratio"],
        "exact_tail_exchanges": aging["exact_tail_exchanges"],
        "aged_argument_bytes": aging["argument_bytes"],
        "aged_result_bytes": aging["result_bytes"],
        "aged_preview_bytes": budget,
        "backstop_exact_tail_exchanges": params["hard_backstop"]["exact_tail_exchanges"],
        "reasoning_replay": reasoning["mode"],
        "reasoning_tail_exchanges": reasoning["tail_exchanges"],
        **(_v10_fields(stage_list, params, label) if v10 else {}),
    }


def management_equivalent(stage_list: Mapping[str, Any], *, label: str = "context stages") -> dict[str, Any]:
    """The v6, v8, v9 or v10 management object whose translation is ``stage_list`` (the evidence still reads it)."""

    if stage_list.get("schema") != STAGE_LIST_SCHEMA:
        raise ValueError(f"{label}: not an {STAGE_LIST_SCHEMA} stage list")
    segments, limits = stage_list["segments"], stage_list["limits"]
    estimator, references = stage_list["estimator"], stage_list["references"]
    _require(limits["reserve_tokens"] == RESERVE_TOKENS, label, "a reserve other than 2048")
    ids = [item["stage"] for item in stage_list["stages"]]
    episode = emergency = None
    v9 = "exchange_aging" in ids
    v10 = v9 and "result_preview" in next(item["params"] for item in stage_list["stages"]
                                         if item["stage"] == "exchange_aging")
    _require(v10 or segments.get("exact_tail_unit", "action_step") == "action_step"
             and all(item["params"].get("unit", "action_step") == "action_step"
                     for item in stage_list["stages"] if item["stage"] == "emergency_checkpoint"),
             label, "an exact tail or an emergency checkpoint in exchanges before v10")
    _require(v9 or segments["unit"] == "action_step", label, "a unit other than action_step")
    if "camera_delivery" in ids:
        params = _params(stage_list, V9_STAGES if v9 else V8_STAGES, label)
        _require("truncated_reply_reasoning" not in params["materialize"], label,
                 "a truncated reply's reasoning withheld outside v6")
        _require(not v9 or not {"aging", "trigger_ratio"} & set(params["hard_backstop"]), label,
                 "a hard backstop that ages by itself beside the exchange stages")
        camera = params["camera_delivery"]
        episode, emergency = params["episode_capsule"], params["emergency_checkpoint"]
        if not v9:
            _require(emergency["trigger_tokens"] is not None, label, "a v8 L6 without an absolute watermark")
            _require(not references["pointers"], label, "a v8 scheme with references shown")
            _require(params["over_limit_guard"]["over_limit"] == "send_and_record", label, "a v8 guard that refuses")
        management = {
            "schema": CONTEXT_MANAGEMENT_V10_SCHEMA if v10 else
            CONTEXT_MANAGEMENT_V9_SCHEMA if v9 else CONTEXT_MANAGEMENT_V8_SCHEMA,
            "causal_exact_tail_action_steps": segments["exact_tail"],
            "camera_images": camera["mode"],
            "camera_tail_observations": camera["tail_observations"],
            "episode_checkpoint_trigger_tokens": episode["trigger_tokens"],
            "compaction_trigger_tokens": emergency["trigger_tokens"],
            "emergency_fallback_ratio": emergency["trigger_ratio"],
            "emergency_target_ratio": emergency["target_ratio"],
            "checkpoint_target_estimated_tokens": episode["checkpoint_target_tokens"],
            "checkpoint_max_estimated_tokens": episode["checkpoint_max_tokens"],
            "image_estimate_tokens": estimator["image_tokens"],
            **(_v9_management(stage_list, params, label) if v9 else {}),
        }
    else:
        backstop = "hard_backstop" in ids
        params = _params(stage_list, V6_BACKSTOP_STAGES if backstop else V6_STAGES, label)
        stride, eager = params["stride_aging"], params["eager_externalize"]
        pressure = params["pressure_externalize"]
        episode, emergency = params["episode_capsule"], params["emergency_checkpoint"]
        _require(references["pointers"] and references["advertise_over_bytes"] == eager["large_result_bytes"],
                 label, "v6 references that do not advertise over the L1 size")
        _require(estimator["image_tokens"] == V6_IMAGE_TOKENS, label, "a v6 per-image estimate other than 4096")
        _require(stride["strided"], label, "unstrided aging")
        _require(pressure["window_share"] == ESTIMATE_PRESSURE_WINDOW_SHARE, label, "another L4 window share")
        _require(emergency["trigger_tokens"] is None, label, "a v6 L6 with an absolute watermark")
        if backstop:
            _require(params["hard_backstop"].get("aging") == "own", label,
                     "a v6 hard backstop that does not age by itself (v6 has no exchange stages)")
            _require(params["hard_backstop"].get("trigger_ratio") == emergency["trigger_ratio"], label,
                     "a v6 hard backstop whose watermark is not L6's")
        else:
            _require(params["over_limit_guard"]["over_limit"] == "refuse", label, "a v6 guard that sends")
        truncated = params["materialize"].get("truncated_reply_reasoning")
        management = {
            "schema": CONTEXT_MANAGEMENT_V6_SCHEMA,
            "causal_exact_tail_action_steps": segments["exact_tail"],
            "inline_media_tail_action_steps": params["media_window"]["tail_action_steps"],
            "eager_large_result_bytes": eager["large_result_bytes"],
            "eager_preview_bytes": references["preview"]["head_bytes"],
            "eager_exact_deduplication": eager["deduplicate"],
            "trigger_inline_images": pressure["inline_images"],
            "trigger_estimated_tokens": pressure["estimated_tokens"],
            "trigger_provider_input_ratio": pressure["provider_input_ratio"],
            "episode_checkpoint_trigger_tokens": episode["trigger_tokens"],
            "emergency_fallback_ratio": emergency["trigger_ratio"],
            "emergency_target_ratio": emergency["target_ratio"],
            "checkpoint_target_estimated_tokens": episode["checkpoint_target_tokens"],
            "checkpoint_max_estimated_tokens": episode["checkpoint_max_tokens"],
            "tail_aging_stride_action_steps": stride["stride"],
            # The optional v6 fields (management_fields.py CONTEXT_MANAGEMENT_V6_OPTIONAL_FIELD_SPECS).
            **({"backstop_exact_tail_exchanges": params["hard_backstop"]["exact_tail_exchanges"]} if backstop else {}),
            **({"truncated_reply_reasoning": truncated} if truncated is not None else {}),
        }
        _require(pressure["enabled"], label, "L4 turned off in the scheme (a context pack turns it off)")
    _require(episode["trigger_ratio"] is None or v10 and episode["trigger_tokens"] is None, label,
             "a window-scaled L5 watermark")
    _require((episode["retry_attempts"], episode["retry_backoff_action_steps"]) == (4, 4), label,
             "another L5 retry schedule")
    _require(episode["enabled"] and emergency["enabled"], label,
             "rollout compaction turned off in the scheme (a context pack turns it off)")
    _require((episode["checkpoint_target_tokens"], episode["checkpoint_max_tokens"])
             == (emergency["checkpoint_target_tokens"], emergency["checkpoint_max_tokens"]),
             label, "checkpoint lengths that differ between L5 and L6")
    return management


def check_stage_list(
    stage_list: Any, management: Mapping[str, Any], *, root: Path = SOURCE_ROOT,
) -> dict[str, Any]:
    """A plan's ``policy.context_stages``, validated, when it is the stage list of its ``context_management``."""

    if not isinstance(stage_list, Mapping) or stage_list.get("schema") != STAGE_LIST_SCHEMA:
        raise ValueError(f"policy.context_stages must be an {STAGE_LIST_SCHEMA} stage list")
    resolved = resolve_stage_list(root, stage_list, label="policy.context_stages")
    if management_equivalent(resolved, label="policy.context_stages") != dict(management):
        raise ValueError("policy.context_stages is not the stage list of policy.context_management")
    return resolved
