"""Finish argument policy; no access to official judgment."""

from __future__ import annotations
import re
from typing import Any, Mapping
from ..errors import ToolRefusal as ToolError

CATEGORIES = {
    "unreachable": "a pose the task needs cannot be reached by the arm",
    "object_lost": "a task object left the workspace or is no longer visible in any camera",
    "scene_damaged": "the scene changed so that the task can no longer be completed",
    "repeated_failure": "the same step failed repeatedly from the current state",
    "other": "none of the above; the reason names the obstacle",
}
#: Categories only the file-backed surfaces declare (2026-10-28). Which ones a call accepts is the surface's
#: own declared enum, handed over in the lifecycle configuration (``categories``); a configuration recorded
#: before that key existed accepts CATEGORIES, as it did then.
TOOLKIT_CATEGORIES = {
    "insufficient_budget": "an allowance left in remaining_budget (native actions or robot decisions) is too small to finish the remaining steps; the reason says what is done, what remains and what it would cost",
}
MIN_REASON_CHARACTERS = 24
MIN_REASON_WORDS = 4
EVIDENCE_TERMS = (
    "check_task",
    "observation",
    "camera",
    "image",
    "depth",
    "measured",
    "visible",
    "verified",
    "receipt",
    "sensor",
)
EVIDENCE_STEMS = (
    "check task",
    "task check",
    "checker",
    "observ",
    "camera",
    "image",
    "depth",
    "measur",
    "visib",
    "seen",
    "shows",
    "verif",
    "receipt",
    "sensor",
    "检查",
    "相机",
    "摄像",
    "图像",
    "画面",
    "观测",
    "观察",
    "深度",
    "测量",
    "可见",
    "看到",
    "显示",
)
REASON_REFUSAL_CODE = "finish_reason_insufficient"
CATEGORY_REFUSAL_CODE = "finish_category_required"
ARGUMENTS = ("outcome", "reason", "category")
#: Words a reason may not consist of alone: placeholders, the outcome and
#: category names, and filler.
TOKEN_WORDS = frozenset(
    {
        "n",
        "a",
        "an",
        "na",
        "none",
        "null",
        "nil",
        "nothing",
        "todo",
        "tbd",
        "reason",
        "reasons",
        "placeholder",
        "test",
        "testing",
        "ok",
        "okay",
        "yes",
        "done",
        "finish",
        "finished",
        "complete",
        "completed",
        "success",
        "successful",
        "succeeded",
        "failure",
        "failed",
        "fail",
        "give",
        "up",
        "gave",
        "giving",
        "abort",
        "aborted",
        "aborting",
        "stop",
        "stopped",
        "unknown",
        "no",
        "not",
        "applicable",
        "see",
        "above",
        "same",
        "as",
        "the",
        "task",
        "is",
        "was",
        "episode",
        "round",
        "outcome",
        "x",
        "xx",
        "xxx",
        "category",
        "end",
        "ended",
        "here",
        "this",
        "it",
        "its",
        "give_up",
        *CATEGORIES,
        *TOOLKIT_CATEGORIES,
    }
)
_WORD = re.compile(r"\w+")
#: CJK ideographs, kana and Hangul: one character carries about a syllable or a
#: morpheme, so it counts as two characters and half a word.
_CJK = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]"
)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def reason_problem(outcome: str, reason: Any) -> str | None:
    """Why a reason is not admissible, or ``None``."""

    if not isinstance(reason, str):
        return "missing" if reason is None else "not_text"
    text = reason.strip()
    if not text:
        return "missing"
    cjk = _CJK.findall(text)
    words = _WORD.findall(_CJK.sub(" ", text.lower()))
    if len(text) + len(cjk) < MIN_REASON_CHARACTERS:
        return "too_short"
    if (
        len(words) + (len(cjk) + 1) // 2 < MIN_REASON_WORDS
        or len(set(words) | set(cjk)) < 3
    ):
        return "too_few_words"
    if not cjk and all(word in TOKEN_WORDS for word in words):
        return "placeholder"
    if outcome == "success" and not names_evidence(text):
        return "no_evidence_source"
    return None


def names_evidence(text: str) -> bool:
    """Whether a reason names an observable evidence source (loose stem match)."""

    spaced = re.sub(r"[\s_\-]+", " ", _CAMEL.sub(" ", text).lower())
    return any(stem in spaced for stem in EVIDENCE_STEMS)


def _refuse_reason(
    outcome: str, reason: Any, problem: str, *, without_check_task: bool = False
) -> ToolError:
    constraints: dict[str, Any] = {
        "problem": problem,
        "outcome": outcome,
        "minimum_characters": MIN_REASON_CHARACTERS,
        "minimum_words": MIN_REASON_WORDS,
    }
    rule = (
        f"every finish_episode outcome needs a reason of at least {MIN_REASON_CHARACTERS} characters "
        f"and {MIN_REASON_WORDS} words that is not a placeholder"
    )
    if outcome == "success":
        # A surface whose success is the official boundary's alone has no check_task to name.
        constraints["evidence_terms"] = [
            term
            for term in EVIDENCE_TERMS
            if not (without_check_task and term == "check_task")
        ]
        rule += "; a success reason names its observable evidence source"
    received = reason[:200] if isinstance(reason, str) else reason
    return ToolError(
        f"finish_episode reason refused ({problem}): {rule}",
        code=REASON_REFUSAL_CODE,
        retryable=True,
        field="reason",
        received=received,
        constraints=constraints,
    )


def _check_arguments(arguments: Any, allowed: tuple[str, ...]) -> str:
    """Every malformed argument is the policy's refusal, never an internal error.

    An exception here would be logged as ``internal_tool_error``, which the
    abort gate counts as infrastructure evidence: the policy could then
    manufacture its own admissible abort. Every check is type-safe.
    """
    if not isinstance(arguments, Mapping):
        raise ToolError(
            "finish_episode arguments must be an object",
            code="invalid_arguments",
            retryable=True,
            constraints={"accepted": list(ARGUMENTS)},
        )
    unknown = sorted(str(name) for name in set(arguments) - set(ARGUMENTS))
    if unknown:
        raise ToolError(
            f"finish_episode does not accept {', '.join(unknown)}",
            code="invalid_arguments",
            retryable=True,
            field=unknown[0],
            constraints={"accepted": list(ARGUMENTS)},
        )
    outcome = arguments.get("outcome")
    if not isinstance(outcome, str) or outcome not in allowed:
        raise ToolError(
            f"finish_episode accepts outcome {', '.join(allowed)}",
            code="invalid_arguments",
            retryable=True,
            field="outcome",
            received=outcome,
            constraints={"allowed": list(allowed)},
        )
    return str(outcome)


def _check_category(outcome: str, category: Any, declared: Any = None) -> None:
    allowed = list(declared) if declared else list(CATEGORIES)
    if outcome == "give_up" and not (
        isinstance(category, str) and category in allowed
    ):
        raise ToolError(
            "give_up requires a category",
            code=CATEGORY_REFUSAL_CODE,
            retryable=True,
            field="category",
            received=category,
            constraints={"allowed": allowed},
        )
    if outcome != "give_up" and category is not None:
        raise ToolError(
            "category belongs to give_up only",
            code="invalid_arguments",
            retryable=True,
            field="category",
            received=category,
            constraints={"outcome": outcome},
        )
