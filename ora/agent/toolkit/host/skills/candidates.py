"""Resolve observation-bound grasp candidates for committed skills."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ora.agent.composition.surface import tool_names
from ora.agent.error_classes import ToolError
from ora.agent.toolkit.tools.grasp_execute._candidates import _WITHHELD, _withheld


def _advertised(host: Any, tool: str) -> str:
    """The spelling this host's surface shows the policy for one tool.

    These error messages direct a policy to another tool by name, which is the
    shape that misleads when the name is not the one the policy holds.
    `canonicalise_prose` rewrites help text and schemas at composition; a
    ToolError built at call time is not composed, so the spelling is resolved
    here instead. A surface that does not declare the tool gets the literal
    back, naming what the message meant.
    """

    surface = getattr(host, "surface", None)
    definitions = getattr(surface, "definitions", ()) if surface is not None else ()
    return tool_names.as_advertised(
        [str(item.get("name")) for item in definitions], tool,
    )


def _grasp_candidate(host: Any, candidate_id: str, observation_id: str) -> dict[str, Any]:
    """Find one compiled grasp candidate produced from the current observation.

    Candidate identities are content hashes of the proposal, so the lookup is
    unambiguous; what the scan enforces is the binding. A candidate compiled
    from an earlier observation describes geometry the robot has since moved
    through, and executing against it is the exact mistake a committed sequence
    must not be able to make quietly.

    An id ``grasp_plan`` published only as rejected, or kept in its pool but
    not displayed, is refused as such, with the ids that can execute now
    "no candidate carries that id" was false for those.
    """

    root = Path(host.artifact_root) / "grasp-queries"
    stale = False
    current: list[str] = []
    disposition: dict[str, Any] | None = None
    for manifest_path in sorted(root.glob("*/manifest.json")) if root.is_dir() else ():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bound = str(manifest.get("identity", {}).get("observation_id")) == observation_id
        if bound:
            current.extend(str(item.get("candidate_id")) for item in manifest.get("compiled_candidates", ()))
            disposition = disposition or _withheld(manifest, candidate_id)
        for candidate in manifest.get("compiled_candidates", ()):
            if str(candidate.get("candidate_id")) != candidate_id:
                continue
            if not bound:
                stale = True
                continue
            compiler = manifest.get("identity", {}).get("compiler")
            compiler = compiler if isinstance(compiler, Mapping) else {}
            return {
                "candidate": dict(candidate),
                "target_ref": str(manifest["identity"]["target_ref"]),
                "query_ref": str(manifest["query_ref"]),
                # The arm a candidate was compiled for is a property of the
                # plan, not of the moment it executes in. A dual-arm backend
                # names it here because it has a choice to record; a
                # single-arm one leaves it out and the observation resolves it.
                "arm": str(compiler["arm"]) if compiler.get("arm") else None,
            }
    valid = {"current_observation_id": observation_id, "valid_candidate_ids": sorted(set(current))}
    if disposition is not None:
        raise ToolError(
            f"candidate_id {candidate_id} was {_WITHHELD[disposition['disposition']]} by "
            f"{_advertised(host, 'grasp_plan')} on the current observation and is not executable",
            code="candidate_unavailable", field="candidate_id", received=candidate_id,
            retryable=True, retry_requires_changed_arguments=True,
            constraints={**disposition, **valid},
        )
    raise ToolError(
        "grasp candidate was compiled from an earlier observation" if stale else
        "no grasp candidate carries that candidate_id",
        code="stale_candidate_observation" if stale else "candidate_unavailable",
        field="candidate_id", received=candidate_id, retryable=True,
        retry_requires_changed_arguments=True, constraints=valid,
    )
