"""Trusted lifecycle evidence projection and fixed-path finish publication."""

import json
import time
from ora.agent.composition.schemas.abort_evidence import ABORT_REQUIRES, counted_classes
from ora.agent.toolkit.host.finish import abort_evidence, candidates, outcomes
from ora.agent.toolkit.host.shared.artifact_io import _atomic_json
from ora.agent.toolkit.interface.types import AccessDenied, PublicData
from ora.agent.toolkit.interface.blobs import BlobStore

from .publication import published_manifest

#: What a reported issue is (default ``defect``): an interface, tool or evidence defect, or the policy's own confusion
#: about the task description, a tool or the instructions, so the maintainers and RSI agents see where it was unsure.
REPORT_ISSUE_KINDS = ("defect", "confusion")

#: The rules the handler has always enforced -- one ``message`` string of at
#: least one character, and an optional ``kind`` -- written as the schema most surfaces
#: declare. Checked on every surface, so a refusal
#: names its field and rule whether or not the surface validated its declared
#: schema before the handler.
REPORT_ISSUE_INPUT = {
    "type": "object", "additionalProperties": False, "required": ["message"],
    "properties": {"message": {"type": "string", "minLength": 1},
                   "kind": {"type": "string", "enum": list(REPORT_ISSUE_KINDS)}},
}


def declared_categories(surface):
    """The give_up categories the surface's own finish_episode declares (the file-backed surfaces add
    insufficient_budget, 2026-10-28); the released list when it declares none."""
    for definition in surface.definitions:
        if definition.get("name") == "finish_episode":
            field = ((definition.get("inputSchema") or {}).get("properties") or {}).get("category") or {}
            declared = [item for item in field.get("enum") or [] if item is not None]
            if declared:
                return declared
    return list(outcomes.CATEGORIES)


def released_arguments(surface):
    """What ``finish_episode`` accepts on a surface with no outcome contract: its three, and whatever else it advertises."""
    definition = next((item for item in surface.definitions if item.get("name") == "finish_episode"), {})
    return sorted({"outcome", "reason", "attempts", *(definition.get("inputSchema", {}).get("properties") or {})})


def configuration(host):
    """Public evidence only: no official task check is stored in tool inputs. ``released_arguments`` is there only on a
    surface with no outcome contract, so a context without it is a v2 one."""
    accepted = outcomes.accepted_outcomes(host.surface)
    return dict(
        faithful_check_task=host.surface.supports("faithful_check_task"),
        official_verdict_finish=host.surface.supports(
            "finish_success_official_verdict_v1"
        ),
        outcomes=list(outcomes.RELEASED_OUTCOMES if accepted is None else accepted),
        **({"released_arguments": released_arguments(host.surface)} if accepted is None else {}),
        categories=declared_categories(host.surface),
        action_pending=(host.control_root / "submitted-action.json").is_file(),
        remaining_budget=host._runtime_remaining_budget(),
        infrastructure_signals=abort_evidence.recorded_signals(host.event_path),
        abort_requires=ABORT_REQUIRES,
        counted_classes=counted_classes(),
        candidates=candidates.for_host(host, ""),
    )


def check_adapter(host):
    def read(parameters, inputs, store):
        from .state import task_check
        value = task_check(host)
        if value is None:
            return PublicData.from_json(
                dict(
                    success=False,
                    reason="task_checker_unavailable",
                    retryable=True,
                    verdict={},
                )
            )
        verdict = {
            key: value[key] for key in ("task_success", "terminal") if key in value
        }
        return PublicData.from_json(
            dict(
                success=True,
                task_success=bool(value.get("task_success", False)),
                verdict=verdict,
            )
        )

    return read


def finish_adapter(host):
    def submit(parameters, inputs, store):
        record = parameters["request"]
        required = {
            "schema",
            "outcome",
            "runtime_remaining_budget",
            "early_exit",
            "early_exit_argument_required",
        }
        if outcomes.accepted_outcomes(host.surface) is None:
            optional = set(released_arguments(host.surface))
            accepted = outcomes.RELEASED_OUTCOMES
        else:
            required |= {"reason", "finish_contract", "category"}
            optional = {
                "abort_admission",
                "abort_infrastructure_evidence",
                "untried_candidates",
            }
            accepted = ("success", "give_up", "abort")
        if (
            not required <= record.keys()
            or record.keys() - required - optional
            or record["schema"] != "ora.openeta-policy-finish.v1"
            or record["outcome"] not in accepted
        ):
            raise AccessDenied("invalid portable finish record")
        _atomic_json(
            host.control_root / "policy-finish.json", dict(record, time=time.time())
        )
        return PublicData.from_json(
            dict(
                success=True,
                outcome=record["outcome"],
                _runtime=dict(directive="episode_finished"),
            )
        )

    return submit


def context_publication(host):
    """Identify even a closed publication; this grants no observation/query capability."""
    path = host.control_root / "current-observation.json"
    if not path.is_file():
        return {}, {}
    control = json.loads(path.read_bytes())
    _, manifest, _ = published_manifest(BlobStore(host.control_root.parent), control)
    return control, manifest


def report_adapter(host):
    """Validate an episode issue and append only to its declared control artifact."""
    from ora.agent.composition.surface.schema_validation import ToolSurfaceError, validate_schema_instance
    from ora.agent.error_classes import schema_refusal, blank_text_refusal
    from ora.agent.toolkit.host.shared.artifact_io import _append

    def report(parameters, inputs, store):
        arguments = parameters["issue"]
        try:
            validate_schema_instance(arguments, REPORT_ISSUE_INPUT, root_path="$input")
        except ToolSurfaceError as error:
            raise schema_refusal(error) from error
        message = arguments["message"]
        if not message.strip():
            raise blank_text_refusal("message", message,
                message="report_issue requires a non-empty message: the message contains only whitespace")
        _append(host.control_root / "openeta-reported-issues.jsonl",
                {"message": message.strip(), "kind": arguments.get("kind", "defect"), "time": time.time()})
        return PublicData.from_json({"success": True})

    return report
