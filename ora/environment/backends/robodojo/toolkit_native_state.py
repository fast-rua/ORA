"""Diagnostic native endpoint bookkeeping; no robot/object setters or actions."""
from copy import deepcopy
from ora.contracts.execution_state import unpack


def bind(session, boundary):
    """Carry pre-execution owner state and the current tick's explicit directives.

    Completed post-action target buffers and receipts are not applied to an
    intermediate tick. The actual command buffer is already restored by control
    replay. The host separately supplies the current pre-action recovery anchor.
    """
    fields = unpack(boundary["exec_values"])
    directives = deepcopy(session.gripper_directives)
    for name, value in fields.items():
        setattr(session, name, deepcopy(value))
    if boundary["expected"]["native_actions"]:
        session.gripper_directives = directives
    expected = boundary["expected"]
    session.observation_sequence = expected["observation_sequence"]
    session.action_sequence = expected["native_receipts"]
    session.last_receipt = deepcopy(expected["previous_execution_receipt"])
    previous = boundary["exec_values"].get("exec.world_mutation_sequence", 0)
    session.world_mutation_sequence = previous
    session.world_mutation_actions = []


def source_warnings(session, reset, native):
    """Version/fingerprint differences are evidence; physical checks still gate."""
    import json
    recorded = json.loads((native / "source-manifest.json").read_bytes())
    original_reset = json.loads((native / "reset.json").read_bytes())
    current = getattr(session, "source", {}) or {}
    fields = {
        "source_revisions": (recorded.get("source_revisions"), current.get("source_revisions")),
        "layout_file.sha256": (recorded.get("layout_file", {}).get("sha256"),
                               current.get("layout_file", {}).get("sha256")),
        "actual_source_fingerprint": (original_reset.get("actual_source_fingerprint"),
                                      reset.get("actual_source_fingerprint")),
    }
    from ora.contracts.toolkit_protocol import identity_warning
    return [*current.get("warnings", []), *[
        {**identity_warning(name, left, right), "kind": "source_identity",
         "physical_verification_required": True}
        for name, (left, right) in fields.items() if left is None or left != right]]
