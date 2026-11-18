"""The robot-specific rules the TypeScript context policy applies, sent from Python.

The TypeScript port keeps these values as defaults
(``ora-pi/src/context/domain_defaults.ts``) only so its differential tests can
run alone. In a run they come from here, read from the Python modules that own
them, so Python stays the single owner of robot vocabulary and privacy rules.
"""

from __future__ import annotations

from typing import Any, Sequence

from ora.agent.context import causal_value
from ora.agent.context.lossless import compactor_privacy
from ora.agent.context.lossless.compactor import COMPACTOR_SYSTEM_PROMPT
from ora.agent.composition.surface.tool_names import CANONICAL_TOOL_NAMES
from ora.agent.evidence.trace_privacy import SCRIPT_AUTHORED_FIELDS, SCRIPT_HANDED_BACK_FIELDS
from ora.agent.composition.schemas.context_stages import SOURCE_ROOT, stage_schemas


def context_domain(*, mutation_tools: Sequence[str], script_tools: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "mutationTools": list(mutation_tools),
        "scriptTools": sorted(script_tools),
        "scriptAuthoredFields": sorted(SCRIPT_AUTHORED_FIELDS),
        "scriptHandedBackFields": sorted(SCRIPT_HANDED_BACK_FIELDS),
        "causalForbiddenKeys": sorted(causal_value._CAUSAL_FORBIDDEN_KEYS),
        "compactorPrivateKeys": sorted(causal_value._COMPACTOR_PRIVATE_KEYS),
        "finishVerdictRefusalCodes": sorted(compactor_privacy.FINISH_VERDICT_REFUSAL_CODES),
        "checkpointForbiddenPattern": causal_value._CHECKPOINT_FORBIDDEN.pattern,
        "checkpointOutcomeAssertionPattern": causal_value._CHECKPOINT_OUTCOME_ASSERTION.pattern,
        "compactorSystemPrompt": COMPACTOR_SYSTEM_PROMPT,
        "canonicalToolNames": sorted(CANONICAL_TOOL_NAMES),
    }


def context_stage_schemas() -> dict[str, Any]:
    """Every context stage's parameter schema and the scheme settings' (``ora/policy/context_stages``).

    The core validates each stage it builds with these; it keeps no copy of the rules.
    """

    return stage_schemas(SOURCE_ROOT)
