"""Thin access to public receipts/control; sequencing and task preferences belong to tools."""
from ..interface.api import DataInterface
from .values import JsonValue, PublicResult


def receipt(api: DataInterface, *, action_id: str) -> PublicResult:
    """Replay: from record, missing fails. Example: receipt(api, action_id='action-1').

    All exported per-arm/phase details survive; metres/radians/seconds stay named in
    the envelope. No final-state backfill and no promise of unexported controller internals.
    """
    return PublicResult.of(api.state.receipt(action_id=action_id))


def check_task(api: DataInterface) -> PublicResult:
    """Replay: from record at this boundary, missing fails; capability disabled raises AccessDenied.

    Example: check_task(api).data.json(). Official result only, no truth/predicate API.
    """
    return PublicResult.of(api.state.check_task())


def submit_action(api: DataInterface, action: JsonValue) -> PublicResult:
    """Replay: exact recorded action only; changed action raises Divergence, missing cannot execute.

    Example: submit_action(api, JsonValue.of(validated_proposal)). Keep original schema,
    frame/unit bindings. Returns submission acknowledgement,
    never future execution receipt. Runtime uses the trusted existing admission path.
    """
    return PublicResult.of(api.act.submit_action(action=action.json()))


def report_issue(api: DataInterface, issue: JsonValue) -> PublicResult:
    """Replay: exact recorded control request only; mismatch fails, never writes source run.

    Example: report_issue(api, JsonValue.of({'message':'blocked'})); original schema applies.
    """
    return PublicResult.of(api.episode.report_issue(issue=issue.json()))


def finish_episode(api: DataInterface, request: JsonValue) -> PublicResult:
    """Replay: exact recorded finish only; mismatch fails, no simulator continuation.

    Example: finish_episode(api, JsonValue.of(valid_finish_request)); no automatic stage decisions.
    """
    return PublicResult.of(api.episode.finish_episode(request=request.json()))


def execute_action(api: DataInterface, action: JsonValue, budget: int = 20) -> PublicResult:
    """Execute one primitive through the trusted owner; return receipt and parent balance.

    Live execution publishes a new state before returning. No observation is
    included: read state separately. Replay: exact recorded receipt only; no online
    physics fallback. Snapshot-only assembly refuses execution.
    Example: execute_action(api, JsonValue.of(bound_primitive), budget=20).
    """
    return PublicResult.of(api.act.execute_action(action.json(), budget))
