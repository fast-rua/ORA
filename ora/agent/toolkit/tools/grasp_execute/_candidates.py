"""Resolve candidates only among this boundary's granted query manifests."""
import json
from typing import Any, Mapping
from ..errors import ToolRefusal as ToolError


def resolve(api, config, candidate_id, observation_id):
    stale=False;current=[];disposition=None
    for key in sorted(config['grasp_query_refs']):
        record=api.artifacts.tool_artifact(kind='grasp_query',key=key)
        if record is None:
            from ...interface.types import NotRecorded
            raise NotRecorded('未记录: granted grasp query unavailable')
        manifest=json.loads(api.artifacts.read_blob(record['files']['manifest.json']))
        bound=str(manifest.get('identity',{}).get('observation_id'))==observation_id
        if bound:
            current.extend(str(row.get('candidate_id')) for row in manifest.get('compiled_candidates',()))
            disposition=disposition or _withheld(manifest,candidate_id)
        for row in manifest.get('compiled_candidates',()):
            if str(row.get('candidate_id'))!=candidate_id:continue
            if not bound:stale=True;continue
            return dict(candidate=dict(row),target_ref=str(manifest['identity']['target_ref']),
                query_ref=key,arm=manifest['identity']['compiler'].get('arm'))
    valid=dict(current_observation_id=observation_id,valid_candidate_ids=sorted(set(current)))
    if disposition is not None:
        raise ToolError(f"candidate_id {candidate_id} was {_WITHHELD[disposition['disposition']]} by grasp_plan on the current observation and is not executable",
            code='candidate_unavailable',field='candidate_id',received=candidate_id,retry_requires_changed_arguments=True,
            constraints={**disposition,**valid})
    raise ToolError('grasp candidate was compiled from an earlier observation' if stale else 'no grasp candidate carries that candidate_id',
        code='stale_candidate_observation' if stale else 'candidate_unavailable',field='candidate_id',received=candidate_id,
        retry_requires_changed_arguments=True,constraints=valid)


_WITHHELD = {
    "rejected": "rejected", "rejected_by_compiler": "rejected at compilation",
    "not_displayed": "kept in the pool but not displayed",
}


def _withheld(manifest: Mapping[str, Any], candidate_id: str) -> dict[str, Any] | None:
    """Why one current ``grasp_plan`` call did not publish ``candidate_id``, if it did not."""

    diagnostics = manifest.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    for key, disposition in (("rejected", "rejected"), ("compiler_rejected", "rejected_by_compiler")):
        for entry in diagnostics.get(key) or ():
            if isinstance(entry, Mapping) and str(entry.get("candidate_id")) == candidate_id:
                reasons = entry.get("reasons", entry.get("reason"))
                return {"disposition": disposition,
                        "reasons": list(reasons) if isinstance(reasons, list) else [reasons]}
    selection = manifest.get("candidate_selection")
    if isinstance(selection, Mapping) and any(
        isinstance(item, Mapping) and str(item.get("candidate_id")) == candidate_id
        for item in selection.get("pool") or ()
    ):
        return {"disposition": "not_displayed",
                "displayed_candidate_ids": list(selection.get("displayed_candidate_ids") or ())}
    return None
