"""Deliver the reviewed public completion contract as an episode-task layer.

A benchmark's short instruction and its official predicate are not the same
statement. Where they differ, the reviewed contract carries the human-readable
requirements the instruction leaves unstated. It is semantic content only: no
coordinates, thresholds, actor handles or evaluator code, and it never displaces
the environment's own Boolean verdict.

The failure this module exists to prevent is silence. A contract that exists,
is not delivered, and leaves no record reads exactly like a task that has no
contract. Every resolution therefore returns a disclosure the RunPlan records,
including the reason a contract was withheld.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ora.agent.composition.task_contract.robodojo import resolve_reviewed_contract

REGISTRY_SCHEMA = "ora.robotwin-task-contract-registry.v1"
DISCLOSURE_SCHEMA = "ora.task-contract-disclosure.v1"
#: The reason a contract listed under a registry's ``pending_review`` is withheld.
PENDING_REASON = "contract_review_pending"
#: Disclosure reasons that withhold a contract on purpose (not a missing or unreviewed one).
DELIBERATE_WITHHOLDING = frozenset({"replaced_by_task_file", "free_play_session_withholds_the_task"})
REGISTRIES = {
    "robotwin": "ora/policy/environments/robotwin/task-contracts/registry-v4.json",
    "robocasa": "ora/policy/environments/robocasa/task-contracts/registry-v1.json",
    "robolab": "ora/policy/environments/robolab/task-contracts/registry-v1.json",
}
# The pinned source revision a contract is verified against. RoboCasa pins two
# repositories; its task predicates live in robocasa, so that pin scopes them.
REVISION_FIELDS = {"robocasa": ("required_revisions", "robocasa")}


def _runtime_revision(kind: str, runtime: Mapping[str, Any]) -> Any:
    field = REVISION_FIELDS.get(kind)
    if field is None:
        return runtime.get("required_revision")
    pins = runtime.get(field[0])
    return pins.get(field[1]) if isinstance(pins, Mapping) else None


class TaskContractError(RuntimeError):
    """A registered contract could not be resolved to an exact layer."""


def _registry(kind: str, root: Path) -> tuple[dict[str, Any], str] | None:
    relative = REGISTRIES.get(kind)
    if relative is None:
        return None
    path = (root / relative).resolve()
    if not path.is_file():
        raise TaskContractError(f"task contract registry is missing: {relative}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != REGISTRY_SCHEMA:
        raise TaskContractError(f"task contract registry schema differs: {relative}")
    return value, relative


#: Where a RunSpec's own task file may live (policy.task_file); a source overlay may replace it.
TASK_FILE_ROOT = "ora/policy/task_files/"


def task_file_layer(
    policy: Mapping[str, Any], runtime: Mapping[str, Any], *, root: Path,
) -> tuple[str, dict[str, Any]] | None:
    """The RunSpec's own task file in place of the reviewed contract layer, or None.

    A task file (``policy.task_file``) is task knowledge its author maintains, such as an
    RSI agent's completion conditions and experience. It takes the contract layer's place
    in ``episode_task``; the reviewed contract is then not delivered and no review vouches
    for the text, so it is not bound to the layouts a contract record was reviewed on.
    """
    from ora.agent.composition.locked_assets import locked_asset

    path = policy.get("task_file")
    if path is None:
        return None
    if (not isinstance(path, str) or not path.startswith(TASK_FILE_ROOT) or not path.endswith(".md")
            or ".." in Path(path).parts):
        raise TaskContractError(f"policy.task_file must be a Markdown file under {TASK_FILE_ROOT}")
    asset = locked_asset(root, path, label="task file", error=TaskContractError)
    task = runtime.get("task_variant") or runtime.get("task")
    return path, {
        "schema": DISCLOSURE_SCHEMA, "backend": runtime.get("kind"), "delivered": False,
        "reason": "replaced_by_task_file", "task": task,
        "task_file": {"path": path, "sha256": asset.sha256},
    }


def resolve_task_contract_layer(
    runtime: Mapping[str, Any], *, root: Path,
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    """Return the episode-task layer for this task and why it was or was not sent.

    The layer is `None` whenever the contract cannot be delivered against the
    exact source the runtime pins. The disclosure is never `None`: an absent
    contract, an unregistered task and a revision the contract was not verified
    against are three different recorded states, not one silence.
    """
    kind = runtime.get("kind")
    disclosure: dict[str, Any] = {
        "schema": DISCLOSURE_SCHEMA, "backend": kind, "delivered": False,
    }
    if kind == "robodojo":
        try:
            return resolve_reviewed_contract(runtime, project_root=root,
                                             disclosure=disclosure)
        except (OSError, ValueError, RuntimeError) as exc:
            raise TaskContractError(str(exc)) from exc
    relative = REGISTRIES.get(str(kind))
    if relative is None:
        disclosure["reason"] = "backend_has_no_reviewed_task_contract_registry"
        return None, disclosure
    disclosure["registry"] = relative

    task = runtime.get("task")
    if not isinstance(task, str) or not task:
        # A spec may resolve defaults before it binds a task. That is a recorded
        # state, not a crash: the launch-time guard is the delivery assertion
        # over real run specs, not an exception on a generic composition path.
        # The registry is read after this check rather than before it, so a
        # root with no task to look up is not also asserted to hold the file.
        disclosure["reason"] = "runtime_task_identity_unavailable"
        return None, disclosure
    disclosure["task"] = task

    registry, _relative = _registry(str(kind), root)

    required = _runtime_revision(str(kind), runtime)
    disclosure["runtime_revision"] = required

    entry = (registry.get("tasks") or {}).get(task)
    pending = (registry.get("pending_review") or {}).get(task)
    if entry is None and pending is not None:
        # A generated contract whose review still waits for its witness (RoboLab's
        # ``pending_review``). Its text exists and is hash-bound, but nothing has
        # vouched for it yet: withhold it and name the review that would.
        disclosure.update(reason=PENDING_REASON, semantic_review=pending.get("semantic_review"),
                          semantic_review_status="needs_review")
        return None, disclosure
    if entry is None:
        # Either the task has no reviewed contract, or its predicate changed
        # after the contract was written and it was deliberately left out of
        # the verified set. Both are recorded, neither is delivered.
        disclosure["reason"] = "task_absent_from_verified_contract_registry"
        return None, disclosure

    verified = list(entry.get("verified_revisions") or ())
    disclosure["verified_revisions"] = verified

    if required not in verified:
        # Delivering a contract verified against a different predicate would
        # state requirements the running evaluator may not hold. Withhold and
        # say so; re-verify and regenerate to deliver it.
        disclosure["reason"] = "contract_verified_against_a_different_source_revision"
        return None, disclosure

    path = (root / entry["path"]).resolve()
    if not path.is_file() or not path.is_relative_to(root.resolve()):
        raise TaskContractError(f"task contract layer is missing: {entry['path']}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != entry["sha256"]:
        raise TaskContractError(f"task contract layer changed after review: {entry['path']}")

    # A matching source/text hash is not semantic review. Retained legacy
    # entries stay explicit; newly reviewed entries bind the private record
    # and its exact allowlisted public projection, never its private contents.
    review_ref = entry.get("semantic_review")
    disclosure["semantic_review_status"] = "legacy_unreviewed"
    if review_ref is not None:
        from .review import load_review_record, render_public_contract, review_errors
        try:
            review, review_sha256 = load_review_record(root, review_ref)
            errors = review_errors(review)
            if review.get("task") != task or review.get("backend") != kind:
                errors.append("review_identity_mismatch")
            if review.get("verified_revisions") != verified:
                errors.append("review_revision_scope_mismatch")
            if render_public_contract(review) != path.read_text(encoding="utf-8"):
                errors.append("public_projection_differs_from_review")
            if errors:
                raise ValueError(", ".join(errors))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise TaskContractError(f"task contract semantic review invalid for {task}: {exc}") from exc
        disclosure["semantic_review_status"] = "review_record_valid"
        disclosure["semantic_review_sha256"] = review_sha256

    disclosure.update(
        delivered=True, reason="reviewed_contract_matches_the_pinned_source",
        # The disclosure is what a reader partitions results on, so it names the
        # exact text that was sent rather than pointing at the composed layer.
        contract_sha256=entry["sha256"],
        check_success_sha256=entry["check_success_sha256"],
        instruction_gap_verdict=entry.get("verdict"),
        clause_count=entry.get("clause_count"),
    )
    # Every policy_context layer is a project-relative path string; the RunSpec
    # resolver hashes it itself. The reviewed hash is already verified above and
    # is carried in the disclosure, so returning the path loses no integrity.
    return entry["path"], disclosure



def _task_channel(context: Mapping[str, Any]) -> str:
    return ("episode_task" if context.get("schema") == "ora.policy-context.v3"
            else "episode_prefix")


def with_task_contract(spec: dict[str, Any], *, root: Path) -> dict[str, Any]:
    """Complete a builder-owned spec with the reviewed task layer and disclosure.

    Builders call this after selecting their runtime and episode-task template.
    The shared RunSpec guard independently verifies the resulting delivery.
    """
    layer, disclosure = resolve_task_contract_layer(spec["runtime"], root=root)
    spec["policy"]["task_contract"] = disclosure
    layers = spec["policy_context"][_task_channel(spec["policy_context"])]
    if layer is not None and layer not in layers:
        layers.append(layer)
    return spec


def free_play_disclosure(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """The disclosure of a free-play session: the scene's task, and so its contract, is withheld."""
    return {
        "schema": DISCLOSURE_SCHEMA, "backend": runtime.get("kind"), "delivered": False,
        "reason": "free_play_session_withholds_the_task",
    }


def bind_task_contract(
    spec: Mapping[str, Any], policy: dict[str, Any], runtime: Mapping[str, Any],
    *, root: Path,
) -> None:
    """Attach the disclosure and assert a deliverable contract was delivered.

    The baseline resolver composes the layer; a package that assembles its own
    `policy_context` composes nothing and the omission leaves no trace, which
    is the silence this module exists to prevent. Asserting it here rather than
    in any one package puts the check on the single path every RunSpec takes.

    The caller translates TaskContractError to its own public error type.
    """

    if "play" in spec:
        layer, disclosure = None, free_play_disclosure(runtime)
    else:
        layer, disclosure = (task_file_layer(policy, runtime, root=root)
                             or resolve_task_contract_layer(runtime, root=root))
    declared = policy.get("task_contract")
    if declared is None:
        policy["task_contract"] = disclosure
    elif declared != disclosure:
        raise TaskContractError(
            "policy.task_contract disagrees with the contract this runtime "
            "resolves; it is resolver-owned and must not be restated"
        )
    if layer is None:
        return
    context = spec.get("policy_context") or {}
    # Typed context materializes episode_task for every runtime; the legacy
    # managed context materializes episode_prefix. Check the actual schema's
    # delivered channel, never accept an unused alias to bypass delivery.
    channel = _task_channel(context)
    episode_task = context.get(channel)
    if not isinstance(episode_task, list) or layer not in episode_task:
        raise TaskContractError(
            f"a reviewed task contract is deliverable for {runtime.get('task')!r} "
            f"but policy_context.{channel} does not carry {layer}"
        )


__all__ = [
    "DELIBERATE_WITHHOLDING",
    "DISCLOSURE_SCHEMA",
    "PENDING_REASON",
    "bind_task_contract",
    "free_play_disclosure",
    "with_task_contract",
    "REGISTRY_SCHEMA",
    "TaskContractError",
    "resolve_task_contract_layer",
    "task_file_layer",
]
