"""Tool-owned holding intentions. No attachment claim and no hidden cache."""

from ..lib import clearance
from ..lib.values import JsonValue, PublicResult
from ..lib.blobs import read_json
from ..interface.types import ActionNotStarted


def intent(source, target_ref, pose):
    return {"source": source, "target_ref": target_ref,
            "attach_pose_xyz_xyzw": list(pose), "role": "dispatch_intent_not_grasp_evidence"}


def stage(api, updates):
    """Write before dispatch so the action's completed snapshot includes intent."""
    previous = {arm: api.state.read(f"tool.held.{arm}") for arm in updates}
    for arm, value in updates.items():
        api.state.write(f"tool.held.{arm}", value)
    return previous


def restore_if_unspent(api, previous, receipt):
    # Unknown physical spend must never be treated as an unapplied command.
    if receipt.get("native_actions") == 0:
        stage(api, previous)


def execute(api, action, previous, budget=20):
    """Roll back only a zero receipt or the owner's explicit pre-intent failure."""
    try:
        response = api.act.execute_action(action, budget)
    except ActionNotStarted:
        stage(api, previous)
        raise
    envelope = response.json()
    failure = (envelope.get("error") or {}).get("execution", {})
    if (envelope.get("status") != "ok" and failure.get("status") == "not_started"
            and failure.get("native_actions") == 0 and failure.get("stage") in {"before_intent", "begin"}):
        stage(api, previous)
    value = PublicResult.of(response).data.json()
    restore_if_unspent(api, previous, value["receipt"])
    return value


def extents(api, jaws):
    """Recompute from public target bytes; replay requires those same inputs."""
    result = {}
    for arm, jaw in (jaws or {}).items():
        held = api.state.read(f"tool.held.{arm}")
        if held is None:
            continue
        try:
            artifact = api.artifacts.tool_artifact(kind="target", key=held["target_ref"])
            manifest = (read_json(api, JsonValue.of(artifact["files"]["manifest.json"])).json()
                        if artifact is not None else {})
            result[arm] = clearance.held_extent(
                localization=JsonValue.of(manifest.get("localization") or {}), intent=JsonValue.of(held),
                contact_center_local=tuple(jaw["contact_center_local"])).json()
        except Exception:
            # Like the released clearance reference, optional payload geometry
            # must never block a move. Free text/candidate IDs are valid intent
            # annotations, and missing/corrupt target artifacts mean unknown size.
            result[arm] = {"target_ref": held["target_ref"], "local_corners": None, "recentered": False}
    return result
