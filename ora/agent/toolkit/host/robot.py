"""The host side of ``robot_runtime_v2``'s five tools on a ``RobotToolRuntime``.

What each tool reads of the run is frozen into its call's context here, a refusal among it, and the tool raises that at the
point of its own argument checks. The sections are per call: nothing here runs on a RoboDojo or environment-plans context.
"""

from ora.agent.evidence.policy_state import PolicyStateError, load_policy_state
from ora.agent.toolkit.interface.types import AccessDenied, PublicData
from ora.contracts.action import validate_runtime_action

from ora.agent.error_classes import ToolError
from .shared.artifact_io import _hash
from .failures import control_adapter
from .geometry import selection
from .lifecycle import context_publication
from .observe import adapter, selected_adapter


def _main_image_refusal(host, state):
    """Why the main camera of a snapshot has no readable image, or None; the snapshot names the run's files."""
    main_camera = state.get("main_camera", "head_camera")
    camera = state.get("cameras", {}).get(main_camera)
    if not isinstance(camera, dict):
        return ToolError(f"policy-state main camera is unavailable: {main_camera}")
    reference = camera.get("files", {}).get("rgb_lossless_png")
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        return ToolError("policy-state main image reference is invalid")
    run_root = host.state_root.parent.resolve()
    image_path = (run_root / reference["path"]).resolve()
    if not image_path.is_file() or not image_path.is_relative_to(run_root):
        return ToolError("policy-state main image escapes the run")
    if _hash(image_path) != reference.get("sha256"):
        return ToolError("policy-state main image hash drift")
    return None


def _dump_state(host, arguments):
    """The snapshot of ``step`` and its publication; a call the tool itself refuses reads nothing."""
    step = arguments.get("step")
    if set(arguments) - {"step"} or (step is not None and (type(step) is not int or step < 0)):
        return {}
    try:
        if host.state_root is None:
            raise ToolError("policy-state evidence is unavailable")
        try:
            state = load_policy_state(host.state_root.parent, step)
        except PolicyStateError as error:
            raise ToolError(str(error)) from error
        refusal = _main_image_refusal(host, state)
        if refusal is not None:
            raise refusal
    except ToolError as error:
        return {"error": error.as_dict()}
    return {"state": state, "main_camera": state.get("main_camera", "head_camera"),
            "identity": {"observation_id": state["observation"]["id"],
                         "observation_sha256": state["observation"]["manifest"]["sha256"]}}


def _read_manual(host):
    return {"text": host.manual_path.read_text(encoding="utf-8"), "sections": list(host._manual_sections),
            "sha256": _hash(host.manual_path)}


def _submit_action(host):
    control, _ = context_publication(host)
    return {"action_pending": (host.control_root / "submitted-action.json").exists(),
            "generation": control.get("generation"), "workspace_bounds_m": dict(host.workspace_bounds_m)}


def configure(host, name, arguments, api, store):
    """``name``'s section of the call's context; the observation adapter the section selects goes on ``api``."""
    if name == "robot.read_manual":
        return _read_manual(host)
    if name == "robot.submit_action":
        api.adapters["observation"] = control_adapter(adapter(host, store, {}))
        return _submit_action(host)
    section = _dump_state(host, arguments) if name == "observation.dump_state" else selection(host, arguments)
    if "identity" in section:
        api.adapters["observation"] = control_adapter(selected_adapter(host, store, section["identity"]))
    return section


def submit_adapter(host):
    """Admit this call's target state and persist it; the Supervisor executes it, never this."""

    def submit(parameters, inputs, store):
        control = host._current()[0]
        try:
            action = validate_runtime_action(parameters["action"])
        except ValueError as error:
            raise ToolError(str(error), field="arguments") from error
        if (inputs or action["action"] != "target_state" or action["observation_id"] != control["observation_id"]
                or action["bound_observation_sha256"] != control["observation_sha256"]
                or action["action_id"] != f"action-{control['generation']}"):
            raise AccessDenied("target-state action boundary mismatch")
        return PublicData.from_json(host._persist_action(host.control_root / "submitted-action.json", dict(control), action))

    return submit
