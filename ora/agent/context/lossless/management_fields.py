"""The field ledger of Native Context management: each version's schema literal, its modes and its field specs.

``management.py`` validates a ``context_management`` object against these specs and re-exports every name here.
"""

from __future__ import annotations


CONTEXT_MANAGEMENT_V4_SCHEMA = "ora.native-context-management.v4"
CONTEXT_MANAGEMENT_V5_SCHEMA = "ora.native-context-management.v5"
CONTEXT_MANAGEMENT_V6_SCHEMA = "ora.native-context-management.v6"
CONTEXT_MANAGEMENT_V7_SCHEMA = "ora.native-context-management.v7"
CONTEXT_MANAGEMENT_V8_SCHEMA = "ora.native-context-management.v8"
CONTEXT_MANAGEMENT_V9_SCHEMA = "ora.native-context-management.v9"
CONTEXT_MANAGEMENT_V10_SCHEMA = "ora.native-context-management.v10"
LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS = frozenset({
    CONTEXT_MANAGEMENT_V4_SCHEMA, CONTEXT_MANAGEMENT_V5_SCHEMA,
    CONTEXT_MANAGEMENT_V6_SCHEMA, CONTEXT_MANAGEMENT_V7_SCHEMA,
    CONTEXT_MANAGEMENT_V8_SCHEMA, CONTEXT_MANAGEMENT_V9_SCHEMA,
    CONTEXT_MANAGEMENT_V10_SCHEMA,
})
#: Schemas whose history is append-only between the rewrites they admit: no
#: L1-L4 externalization, fixed camera images by ``camera_images``.
APPEND_ONLY_SCHEMAS = frozenset({
    CONTEXT_MANAGEMENT_V8_SCHEMA, CONTEXT_MANAGEMENT_V9_SCHEMA, CONTEXT_MANAGEMENT_V10_SCHEMA,
})
#: Append-only schemas that show no recoverable-artifact Pointer and provide no reader.
POINTER_FREE_SCHEMAS = frozenset({CONTEXT_MANAGEMENT_V8_SCHEMA})
#: Schemas that age exchanges into recoverable-artifact Pointers at a watermark.
EXCHANGE_AGING_SCHEMAS = frozenset({CONTEXT_MANAGEMENT_V9_SCHEMA, CONTEXT_MANAGEMENT_V10_SCHEMA})
#: v10 ``aged_result_preview``: the head and tail of the text, or the result's structured fields.
RESULT_PREVIEWS = frozenset({"head_and_tail", "structured"})
#: v10 ``compaction_unit``: what L6 absorbs and the exact tail counts in.
COMPACTION_UNITS = frozenset({"action_step", "exchange"})
#: v10 ``reasoning_binding``: an assistant message an aging rewrote is sent without its reasoning (strict), or
#: keeps it when the Provider's reasoning is unsigned and so can be bound to the rewrite (rebind_unsigned).
REASONING_BINDINGS = frozenset({"strict", "rebind_unsigned"})
#: v9 ``reasoning_replay`` modes: every assistant message an aging leaves
#: unchanged keeps its provider continuation state, only the exchanges that have
#: not aged keep it (an aging withholds the aged ones' in the same rewrite), or
#: only the latest ``reasoning_tail_exchanges`` do.
REASONING_REPLAY_ALL = "all"
REASONING_REPLAY_UNAGED = "unaged_exchanges"
REASONING_REPLAY_RECENT = "recent_exchanges"
REASONING_REPLAY_MODES = frozenset({REASONING_REPLAY_ALL, REASONING_REPLAY_UNAGED, REASONING_REPLAY_RECENT})
#: ``camera_images`` modes. ``exchange_window_v1`` needs exchange aging (v9):
#: camera images stay in their observation message until its exchange ages.
CAMERA_ROLLING_TAIL = "rolling_tail_v1"
CAMERA_INLINE_UNTIL_COMPACTION = "inline_until_compaction_v1"
CAMERA_EXCHANGE_WINDOW = "exchange_window_v1"
CAMERA_IMAGE_MODES = frozenset({CAMERA_ROLLING_TAIL, CAMERA_INLINE_UNTIL_COMPACTION, CAMERA_EXCHANGE_WINDOW})
#: The fewest observations a rolling camera tail carries: the current one and
#: the one before it, so an action's input and result are both visible.
MIN_CAMERA_TAIL_OBSERVATIONS = 2
#: Schemas whose v4 tails are minimums that age in strides (v5 onward).
TAIL_AGING_SCHEMAS = frozenset({
    CONTEXT_MANAGEMENT_V5_SCHEMA, CONTEXT_MANAGEMENT_V6_SCHEMA,
    CONTEXT_MANAGEMENT_V7_SCHEMA,
})
#: Schemas whose L5 watermark is max(tokens, ratio x hard window). Maintained
#: configurations no longer select them; HEAD keeps the formula so recorded
#: v4/v5 runs, frozen archives and explicit historical treatments keep the
#: exact threshold they ran with.
WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS = frozenset({
    CONTEXT_MANAGEMENT_V4_SCHEMA, CONTEXT_MANAGEMENT_V5_SCHEMA,
})
TAIL_AGING_STRIDE_FIELD = "tail_aging_stride_action_steps"
EPISODE_TRIGGER_RATIO_FIELD = "episode_checkpoint_trigger_ratio"
IMAGE_ESTIMATE_TOKENS_FIELD = "image_estimate_tokens"
#: The tokens per inline image the v4-v6 pressure estimate charges (``IMAGE_PRESSURE_ESTIMATE_TOKENS``).
DEFAULT_IMAGE_ESTIMATE_TOKENS = 4096

# This ledger is the machine-readable owner of the v4 field set and field-level
# semantics. Callers own explicit treatment values; the lifecycle reference
# renders the reviewed field meanings and one historical value set.
CONTEXT_MANAGEMENT_V4_FIELD_SPECS: dict[str, dict[str, str]] = {
    "schema": {
        "layer": "Contract",
        "phase": "RunSpec and pipeline construction",
        "authority": "exact literal",
        "kind": "schema",
        "meaning": "Select the closed v4 configuration and reject mixed versions.",
    },
    "causal_exact_tail_action_steps": {
        "layer": "Guard",
        "phase": "every projection",
        "authority": "Harness ActionStep inventory",
        "kind": "positive_int",
        "meaning": "Completed ActionSteps whose causal and read-only bodies stay exact through L1-L6; media has its own current-Episode tail.",
    },
    "inline_media_tail_action_steps": {
        "layer": "L3",
        "phase": "every media lifecycle projection",
        "authority": "Harness ActionStep and Session-round inventory",
        "kind": "nonnegative_int",
        "meaning": "Completed steps in the current Episode whose media remains inline; earlier and prior-Episode media becomes recoverable Pointer-only.",
    },
    "eager_large_result_bytes": {
        "layer": "L1",
        "phase": "after a result leaves the protected near window",
        "authority": "exact UTF-8 byte length",
        "kind": "positive_int",
        "meaning": "Strict byte threshold for eager result externalization.",
    },
    "eager_preview_bytes": {
        "layer": "L1/L4",
        "phase": "recoverable result replacement",
        "authority": "exact UTF-8 byte slice",
        "kind": "positive_int",
        "meaning": "Maximum inline preview bytes retained beside a Pointer.",
    },
    "eager_exact_deduplication": {
        "layer": "L2",
        "phase": "after repeated results leave the protected near window",
        "authority": "content SHA-256 and exact bytes",
        "kind": "required_true",
        "meaning": "Require beneficial byte-exact duplicate replacement.",
    },
    "trigger_inline_images": {
        "layer": "L4",
        "phase": "post-L3 pressure admission",
        "authority": "Harness inline image count",
        "kind": "positive_int",
        "meaning": "Run L4 when images remaining after the fixed current-Episode media tail are strictly above this soft threshold.",
    },
    "trigger_estimated_tokens": {
        "layer": "L4",
        "phase": "post-L3 pressure admission",
        "authority": "model-dependent Harness estimate",
        "kind": "positive_int",
        "meaning": "Run L4 when estimated request tokens are strictly above this value.",
    },
    "trigger_provider_input_ratio": {
        "layer": "L4",
        "phase": "post-L3 pressure admission",
        "authority": "same-epoch Provider input tokens",
        "kind": "ratio",
        "meaning": "Run L4 when exact Provider input reaches this hard-window ratio.",
    },
    "episode_checkpoint_trigger_tokens": {
        "layer": "L5",
        "phase": "closed-Episode admission",
        "authority": "same-epoch Provider input tokens",
        "kind": "positive_int",
        "meaning": "Absolute lower bound in the L5 max(tokens, ratio) watermark.",
    },
    "episode_checkpoint_trigger_ratio": {
        "layer": "L5",
        "phase": "closed-Episode admission",
        "authority": "same-epoch Provider input tokens",
        "kind": "ratio",
        "meaning": "Hard-window lower bound in the L5 max(tokens, ratio) watermark.",
    },
    "emergency_fallback_ratio": {
        "layer": "L6",
        "phase": "post-L5 emergency admission",
        "authority": "same-epoch Provider input tokens",
        "kind": "ratio",
        "meaning": "Admit L6 only at or above this hard-window ratio.",
    },
    "emergency_target_ratio": {
        "layer": "L6",
        "phase": "next Provider report after L6",
        "authority": "same-epoch Provider input tokens",
        "kind": "ratio",
        "meaning": "Nonterminal post-L6 request-size target.",
    },
    "checkpoint_target_estimated_tokens": {
        "layer": "L5/L6",
        "phase": "compactor generation",
        "authority": "compactor output target",
        "kind": "positive_int",
        "meaning": "Requested semantic checkpoint output size.",
    },
    "checkpoint_max_estimated_tokens": {
        "layer": "L5/L6",
        "phase": "checkpoint telemetry",
        "authority": "Provider output usage, else Harness estimate",
        "kind": "positive_int",
        "meaning": "Soft checkpoint-size maximum; exceeding it is nonterminal telemetry.",
    },
}

CONTEXT_MANAGEMENT_V4_FIELDS = frozenset(
    CONTEXT_MANAGEMENT_V4_FIELD_SPECS
)

# v5 is v4 plus one field: the v4 tails become minimums and older ActionSteps
# age in batches, so a Provider prefix cache sees one history rewrite per
# stride instead of one per ActionStep. The entries whose meaning changes are
# restated; every other v4 field keeps its exact v4 semantics.
CONTEXT_MANAGEMENT_V5_FIELD_SPECS: dict[str, dict[str, str]] = {
    **CONTEXT_MANAGEMENT_V4_FIELD_SPECS,
    "schema": {
        **CONTEXT_MANAGEMENT_V4_FIELD_SPECS["schema"],
        "meaning": "Select the closed v5 configuration and reject mixed versions.",
    },
    "causal_exact_tail_action_steps": {
        **CONTEXT_MANAGEMENT_V4_FIELD_SPECS["causal_exact_tail_action_steps"],
        "meaning": "Minimum completed ActionSteps whose causal and read-only bodies stay exact through L1-L6; older bodies age only at a stride event.",
    },
    "inline_media_tail_action_steps": {
        **CONTEXT_MANAGEMENT_V4_FIELD_SPECS["inline_media_tail_action_steps"],
        "meaning": "Minimum completed current-Episode steps whose media stays inline; older media ages only at a stride event, and a reset releases prior-Episode media at once.",
    },
    "trigger_inline_images": {
        **CONTEXT_MANAGEMENT_V4_FIELD_SPECS["trigger_inline_images"],
        "meaning": "Run L4 when images remaining after the minimum current-Episode media tail are strictly above this soft threshold; not evaluated while stride-deferred aging is pending.",
    },
    "trigger_estimated_tokens": {
        **CONTEXT_MANAGEMENT_V4_FIELD_SPECS["trigger_estimated_tokens"],
        "meaning": "Run L4 when estimated request tokens are strictly above this value; not evaluated while stride-deferred aging is pending.",
    },
    "trigger_provider_input_ratio": {
        **CONTEXT_MANAGEMENT_V4_FIELD_SPECS["trigger_provider_input_ratio"],
        "meaning": "Run L4 when exact Provider input reaches this hard-window ratio; not evaluated while stride-deferred aging is pending.",
    },
    TAIL_AGING_STRIDE_FIELD: {
        "layer": "L1-L4",
        "phase": "every lifecycle projection",
        "authority": "Harness ActionStep and Session-round inventory",
        "kind": "positive_int",
        "meaning": "Age older ActionSteps only once this many are pending beyond a minimum tail, then all at once to exactly the minimum tails; 1 reproduces v4 byte for byte.",
    },
}

CONTEXT_MANAGEMENT_V5_FIELDS = frozenset(
    CONTEXT_MANAGEMENT_V5_FIELD_SPECS
)

# v6 is v5 with an absolute L5 watermark: the ratio
# field is gone, so L5 no longer scales with the bound window. Under v5 a 1M
# window put L5 at 0.60 x 1,030,144 = 618,086 tokens, where it effectively
# never fired. Every other v5 field keeps its exact v5 semantics.
CONTEXT_MANAGEMENT_V6_FIELD_SPECS: dict[str, dict[str, str]] = {
    **{
        field: spec for field, spec in CONTEXT_MANAGEMENT_V5_FIELD_SPECS.items()
        if field != EPISODE_TRIGGER_RATIO_FIELD
    },
    "schema": {
        **CONTEXT_MANAGEMENT_V5_FIELD_SPECS["schema"],
        "meaning": "Select the closed v6 configuration and reject mixed versions.",
    },
    "episode_checkpoint_trigger_tokens": {
        **CONTEXT_MANAGEMENT_V5_FIELD_SPECS["episode_checkpoint_trigger_tokens"],
        "meaning": "Absolute L5 watermark: admit one closed Episode Capsule once same-epoch Provider input reaches this many tokens, whatever the bound window.",
    },
}

CONTEXT_MANAGEMENT_V6_FIELDS = frozenset(
    CONTEXT_MANAGEMENT_V6_FIELD_SPECS
)

#: v6 ``truncated_reply_reasoning``: a reply the output limit cut off before any tool call loses its reasoning.
TRUNCATED_REPLY_REASONING_WITHHELD = "withheld"

# Fields a v6 object may leave out. v6 objects resolved before 2026-11-14 have
# neither and keep their behaviour: L6 and L4 react to the Provider's report of
# the request before, and a request the Provider reported over the hard limit
# ends the turn afterwards (seven RoboLab tool-call runs sent requests their own
# estimate put 9,000-31,000 tokens over the limit and died on the report). With
# the backstop field every request is estimated before it is sent, and one at or
# above L6's watermark has all but the latest exchanges aged, the open
# ActionStep's and the exact ActionStep tail's included, which no v6 layer ages.
CONTEXT_MANAGEMENT_V6_OPTIONAL_FIELD_SPECS: dict[str, dict[str, str]] = {
    "backstop_exact_tail_exchanges": {
        "layer": "Hard backstop",
        "phase": "before every request",
        "authority": "provider-anchored request estimate",
        "kind": "positive_int",
        "meaning": (
            "When a request's own estimate after every layer reaches L6's watermark (emergency_fallback_ratio "
            "of the hard limit), the reasoning, read-only tool results and tool-returned images of every "
            "exchange but this many latest ones are aged, the open ActionStep's included (observations stay "
            "whole); a request still over the hard limit ends the turn with context_exhausted before it is "
            "sent. Absent: a request the Provider reported over the hard limit ends the turn afterwards."
        ),
    },
    "truncated_reply_reasoning": {
        "layer": "Delivery",
        "phase": "every projected request",
        "authority": "Harness history (the truncation continuation that follows the reply)",
        "kind": "truncated_reply_reasoning",
        "meaning": (
            "withheld: a reply the Provider's output limit cut off before any tool call is sent without its "
            "reasoning from the first request that carries it; the continuation after it says the reply was "
            "cut, and the trace keeps the reply whole. Absent: replayed like any other reply."
        ),
    },
}

# v7 is v6 with the pressure estimate's per-image charge as a field instead of
# the fixed 4096. Against DeepSeek-V4.1-Flash-1017 usage on 4,186 RoboDojo
# requests (measured 2026-10-24) one inline camera image cost about 530 input
# tokens, so a 4096 charge overstated a 23-image request several-fold. Every
# other v6 field keeps its exact v6 semantics.
CONTEXT_MANAGEMENT_V7_FIELD_SPECS: dict[str, dict[str, str]] = {
    **CONTEXT_MANAGEMENT_V6_FIELD_SPECS,
    "schema": {
        **CONTEXT_MANAGEMENT_V6_FIELD_SPECS["schema"],
        "meaning": "Select the closed v7 configuration and reject mixed versions.",
    },
    IMAGE_ESTIMATE_TOKENS_FIELD: {
        "layer": "L4-L6",
        "phase": "every lifecycle projection",
        "authority": "Harness pressure estimate, calibrated against the bound model's reported usage",
        "kind": "positive_int",
        "meaning": "Tokens the request-pressure estimate charges per inline image; text keeps its byte estimate.",
    },
}

CONTEXT_MANAGEMENT_V7_FIELDS = frozenset(
    CONTEXT_MANAGEMENT_V7_FIELD_SPECS
)

# v8 is a different contract, not v7 plus a field: the projected history only
# grows between rollout compactions, so a Provider prefix cache is rewritten
# only when a summary is admitted. The L1-L4 fields (eager externalization,
# media tails, pressure triggers, aging strides) are gone: nothing ages into a
# Pointer, and no recoverable-artifact Pointer is shown to the model. Fixed
# camera images follow ``camera_images``; tool-returned images stay inline
# until a compaction absorbs their ActionStep.
CONTEXT_MANAGEMENT_V8_FIELD_SPECS: dict[str, dict[str, str]] = {
    "schema": {
        **CONTEXT_MANAGEMENT_V7_FIELD_SPECS["schema"],
        "meaning": "Select the closed v8 configuration and reject mixed versions.",
    },
    "causal_exact_tail_action_steps": {
        **CONTEXT_MANAGEMENT_V7_FIELD_SPECS["causal_exact_tail_action_steps"],
        "layer": "L5/L6",
        "meaning": "Completed ActionSteps a rollout compaction always keeps exact, with their images.",
    },
    "camera_images": {
        "layer": "Delivery",
        "phase": "every projected request",
        "authority": "Harness observation inventory",
        "kind": "camera_images",
        "meaning": (
            f"{CAMERA_ROLLING_TAIL}: each observation's fixed camera images are shown once, at the end "
            "of the request, while the observation is among the latest camera_tail_observations; the "
            f"history keeps a per-observation reference. {CAMERA_INLINE_UNTIL_COMPACTION}: they stay "
            "inline in the history until a compaction absorbs their ActionStep."
        ),
    },
    "camera_tail_observations": {
        "layer": "Delivery",
        "phase": "every projected request",
        "authority": "Harness observation inventory",
        "kind": "camera_window",
        "meaning": (
            f"Observations whose camera images the rolling tail carries (at least "
            f"{MIN_CAMERA_TAIL_OBSERVATIONS}); null for {CAMERA_INLINE_UNTIL_COMPACTION}."
        ),
    },
    "episode_checkpoint_trigger_tokens": CONTEXT_MANAGEMENT_V7_FIELD_SPECS["episode_checkpoint_trigger_tokens"],
    "compaction_trigger_tokens": {
        "layer": "L6",
        "phase": "rollout compaction admission",
        "authority": "same-epoch Provider input tokens",
        "kind": "positive_int",
        "meaning": (
            "Absolute L6 watermark: summarize every ActionStep before the exact tail once "
            "same-epoch Provider input reaches min(this, emergency_fallback_ratio x hard window)."
        ),
    },
    "emergency_fallback_ratio": {
        **CONTEXT_MANAGEMENT_V7_FIELD_SPECS["emergency_fallback_ratio"],
        "meaning": "Hard-window cap of the L6 watermark (see compaction_trigger_tokens).",
    },
    "emergency_target_ratio": CONTEXT_MANAGEMENT_V7_FIELD_SPECS["emergency_target_ratio"],
    "checkpoint_target_estimated_tokens": CONTEXT_MANAGEMENT_V7_FIELD_SPECS["checkpoint_target_estimated_tokens"],
    "checkpoint_max_estimated_tokens": CONTEXT_MANAGEMENT_V7_FIELD_SPECS["checkpoint_max_estimated_tokens"],
    IMAGE_ESTIMATE_TOKENS_FIELD: {
        **CONTEXT_MANAGEMENT_V7_FIELD_SPECS[IMAGE_ESTIMATE_TOKENS_FIELD],
        "layer": "L5/L6",
    },
}

CONTEXT_MANAGEMENT_V8_FIELDS = frozenset(
    CONTEXT_MANAGEMENT_V8_FIELD_SPECS
)

# v9 is v8 plus exchange aging. v8's L6 absorbs only completed ActionSteps, so
# a code-mode episode whose model calls complete none (perception, failed
# scripts) never shrinks and was sent over the window; v9's aging unit is the
# exchange, and its rewrite is lossless. Every v8 field keeps its v8 meaning.
CONTEXT_MANAGEMENT_V9_FIELD_SPECS: dict[str, dict[str, str]] = {
    **CONTEXT_MANAGEMENT_V8_FIELD_SPECS,
    "schema": {
        **CONTEXT_MANAGEMENT_V8_FIELD_SPECS["schema"],
        "meaning": "Select the closed v9 configuration and reject mixed versions.",
    },
    "camera_images": {
        **CONTEXT_MANAGEMENT_V8_FIELD_SPECS["camera_images"],
        "meaning": (
            f"{CONTEXT_MANAGEMENT_V8_FIELD_SPECS['camera_images']['meaning']} {CAMERA_EXCHANGE_WINDOW} (v9 "
            "only): they stay inline in their observation message until the exchange the observation "
            "arrived in ages (an aging, the hard backstop or a compaction of its span), and then leave "
            "the request as the recoverable reference of each image."
        ),
    },
    "camera_tail_observations": {
        **CONTEXT_MANAGEMENT_V8_FIELD_SPECS["camera_tail_observations"],
        "meaning": (
            f"Observations whose camera images the rolling tail carries (at least "
            f"{MIN_CAMERA_TAIL_OBSERVATIONS}); null for {CAMERA_INLINE_UNTIL_COMPACTION} and "
            f"{CAMERA_EXCHANGE_WINDOW}."
        ),
    },
    "compaction_trigger_tokens": {
        **CONTEXT_MANAGEMENT_V8_FIELD_SPECS["compaction_trigger_tokens"],
        "kind": "positive_int_or_null",
        "meaning": ("Absolute L6 watermark: summarize every ActionStep before the exact tail once same-epoch Provider "
                    "input reaches min(this, emergency_fallback_ratio x hard window); null: the ratio alone."),
    },
    "exchange_aging_trigger_tokens": {
        "layer": "Exchange aging",
        "phase": "exchange aging admission",
        "authority": "same-epoch Provider input tokens",
        "kind": "positive_int",
        "meaning": (
            "Absolute aging watermark: while same-epoch Provider input is at or above "
            "min(this, exchange_aging_trigger_ratio x hard window), every exchange older than the exact "
            "exchange tail is aged in one rewrite once exact_tail_exchanges exchanges have left that tail "
            "since the last aging. Below compaction_trigger_tokens."
        ),
    },
    "exchange_aging_trigger_ratio": {
        "layer": "Exchange aging",
        "phase": "exchange aging admission",
        "authority": "same-epoch Provider input tokens",
        "kind": "ratio",
        "meaning": "Hard-window cap of the aging watermark; below emergency_fallback_ratio.",
    },
    "exact_tail_exchanges": {
        "layer": "Exchange aging",
        "phase": "exchange aging admission",
        "authority": "Harness exchange inventory",
        "kind": "positive_int",
        "meaning": (
            "Latest exchanges (assistant message plus what follows it) an aging keeps exact, and the batch: "
            "an aging waits until this many exchanges have left the exact tail since the last one."
        ),
    },
    "aged_argument_bytes": {
        "layer": "Exchange aging",
        "phase": "aged exchanges",
        "authority": "exact UTF-8 byte length",
        "kind": "positive_int",
        "meaning": "An aged tool call's arguments longer than this become a recoverable-artifact reference.",
    },
    "aged_result_bytes": {
        "layer": "Exchange aging",
        "phase": "aged exchanges",
        "authority": "exact UTF-8 byte length",
        "kind": "positive_int",
        "meaning": "An aged tool result longer than this becomes a preview and a recoverable-artifact reference.",
    },
    "aged_preview_bytes": {
        "layer": "Exchange aging",
        "phase": "aged exchanges",
        "authority": "exact UTF-8 byte slices",
        "kind": "positive_int",
        "meaning": (
            "The preview budget of an aged result or argument string: its first and last half are kept beside its "
            "reference, never splitting a character, around a marker of the bytes cut between them."
        ),
    },
    "backstop_exact_tail_exchanges": {
        "layer": "Hard backstop",
        "phase": "before every request",
        "authority": "provider-anchored request estimate",
        "kind": "positive_int",
        "meaning": (
            "When a request would exceed the hard limit after every layer, all but this many latest "
            "exchanges are aged (reasoning included); a request that still exceeds it ends the turn "
            "with context_exhausted."
        ),
    },
    "reasoning_replay": {
        "layer": "Delivery",
        "phase": "every projected request",
        "authority": "Harness exchange inventory",
        "kind": "reasoning_replay",
        "meaning": (
            f"{REASONING_REPLAY_ALL}: every assistant message an aging leaves unchanged keeps its provider "
            f"continuation state (reasoning). {REASONING_REPLAY_UNAGED}: only exchanges that have not aged "
            "keep it; an aging withholds it from every exchange it ages, in the same rewrite, and nothing "
            f"is withheld below the aging watermark. {REASONING_REPLAY_RECENT}: only the latest "
            "reasoning_tail_exchanges..2 x reasoning_tail_exchanges - 1 exchanges keep it; older ones are "
            "sent without it, in strides of reasoning_tail_exchanges. A withheld state is stored as an "
            "artifact."
        ),
    },
    "reasoning_tail_exchanges": {
        "layer": "Delivery",
        "phase": "every projected request",
        "authority": "Harness exchange inventory",
        "kind": "reasoning_window",
        "meaning": (
            f"Exchanges whose reasoning {REASONING_REPLAY_RECENT} keeps; null for {REASONING_REPLAY_ALL} "
            f"and {REASONING_REPLAY_UNAGED}."
        ),
    },
}

CONTEXT_MANAGEMENT_V9_FIELDS = frozenset(
    CONTEXT_MANAGEMENT_V9_FIELD_SPECS
)


def _v10_field_specs() -> dict[str, dict[str, str]]:
    """v10 is v9 with arguments that may never age and an L5 by ratio, plus three more fields.

    ``aged_result_preview`` says what an aged result keeps, ``compaction_unit``
    what L6 absorbs and the exact tail counts in, ``reasoning_binding`` whether
    a rewritten assistant message keeps its reasoning. Every other v9 field
    keeps its v9 semantics.
    """

    restated = {
        "causal_exact_tail_action_steps": ("positive_int", "With compaction_unit exchange the exact tail counts "
                                           "exact_tail_exchanges exchanges instead, and this field has no effect."),
        "episode_checkpoint_trigger_tokens": ("positive_int_or_null", "null: L5 at episode_checkpoint_trigger_ratio "
                                              "of the hard input limit instead."),
        "aged_argument_bytes": ("positive_int_or_null", "null: arguments never age, so no assistant message is "
                                "rewritten and its reasoning stays bound."),
    }
    specs: dict[str, dict[str, str]] = {}
    for field, spec in CONTEXT_MANAGEMENT_V9_FIELD_SPECS.items():
        spec = dict(spec)
        if field in restated:
            spec["kind"], note = restated[field]
            spec["meaning"] = f"{spec['meaning']} {note}"
        specs[field] = spec
        if field == "episode_checkpoint_trigger_tokens":
            specs[EPISODE_TRIGGER_RATIO_FIELD] = {
                "layer": "L5", "phase": "episode capsule admission", "authority": "same-epoch Provider input tokens",
                "kind": "ratio_or_null",
                "meaning": "L5 by ratio: int(hard input limit x this), below emergency_fallback_ratio; exactly one of "
                           "it and episode_checkpoint_trigger_tokens is set."}
        if field == "aged_preview_bytes":
            specs["aged_result_preview"] = {
                "layer": "Exchange aging", "phase": "aged exchanges", "authority": "the result's own fields",
                "kind": "result_preview",
                "meaning": "head_and_tail: aged_preview_bytes of the text, head and tail. structured: the result's "
                           "status fields (ok, status, exit code, actions executed), a failed one's error type and "
                           "message and its size, no output text."}
    specs["schema"] = {**specs["schema"], "meaning": "Select the closed v10 configuration and reject mixed versions."}
    specs["compaction_unit"] = {
        "layer": "L6", "phase": "rollout compaction admission", "authority": "Harness exchange inventory",
        "kind": "compaction_unit",
        "meaning": "action_step: as v9. exchange: L6 summarizes whole exchanges, and the exact tail of L6 and of the "
                   "aging is the latest exact_tail_exchanges exchanges plus the latest robot action's exchange, "
                   "however old."}
    specs["reasoning_binding"] = {
        "layer": "Exchange aging", "phase": "aged exchanges", "authority": "the Provider continuation's native kind",
        "kind": "reasoning_binding",
        "meaning": "strict: an assistant message whose arguments an aging replaced is sent without its reasoning. "
                   "rebind_unsigned: it keeps its reasoning, bound to the rewritten message, when the Provider's "
                   "reasoning carries no signature (OpenAI-compatible reasoning_content); signed reasoning "
                   "(Anthropic) is still withheld."}
    return specs


# The keep-written schemes and the script-aging comparison select v10 (ora/policy/context_schemes).
CONTEXT_MANAGEMENT_V10_FIELD_SPECS = _v10_field_specs()

CONTEXT_MANAGEMENT_V10_FIELDS = frozenset(
    CONTEXT_MANAGEMENT_V10_FIELD_SPECS
)


__all__ = [
    "APPEND_ONLY_SCHEMAS",
    "CAMERA_EXCHANGE_WINDOW",
    "CAMERA_IMAGE_MODES",
    "CAMERA_INLINE_UNTIL_COMPACTION",
    "CAMERA_ROLLING_TAIL",
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
    "CONTEXT_MANAGEMENT_V10_FIELDS",
    "CONTEXT_MANAGEMENT_V10_FIELD_SPECS",
    "CONTEXT_MANAGEMENT_V10_SCHEMA",
    "COMPACTION_UNITS",
    "REASONING_BINDINGS",
    "RESULT_PREVIEWS",
    "DEFAULT_IMAGE_ESTIMATE_TOKENS",
    "EPISODE_TRIGGER_RATIO_FIELD",
    "EXCHANGE_AGING_SCHEMAS",
    "IMAGE_ESTIMATE_TOKENS_FIELD",
    "LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS",
    "MIN_CAMERA_TAIL_OBSERVATIONS",
    "POINTER_FREE_SCHEMAS",
    "REASONING_REPLAY_ALL",
    "REASONING_REPLAY_MODES",
    "REASONING_REPLAY_RECENT",
    "REASONING_REPLAY_UNAGED",
    "TAIL_AGING_SCHEMAS",
    "TAIL_AGING_STRIDE_FIELD",
    "TRUNCATED_REPLY_REASONING_WITHHELD",
    "WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS",
]
