"""``observation.dump_state``: one current or historical policy-state snapshot and its main image."""

import hashlib

from ...lib import images, observation
from ...lib.values import JsonValue
from ..errors import ToolRefusal
from ..observe import _annotated

THUMBNAIL = (512, 512)


def run(api, arguments):
    """Replay: the recorded snapshot and its publication's image; a missing one fails. No simulator query.

    ``step`` is a saved policy-state step, null the latest. An older snapshot says nothing of the current scene.
    """
    if set(arguments) - {"step"}:
        raise ToolRefusal("observation.dump_state accepts only optional step")
    step = arguments.get("step")
    if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
        raise ToolRefusal("step must be a non-negative integer or null")
    selected = api.artifacts.tool_context()["observation_dump_state"]
    if "error" in selected:
        raise ToolRefusal.from_public(selected["error"])
    state, camera = selected["state"], selected["main_camera"]
    obs = observation.observe(api, identity=JsonValue.of(selected["identity"]))
    source = obs.camera(camera).rgb.json()
    encoded = images.inspected(api.artifacts.read_blob(source), fit=THUMBNAIL).png
    return {
        "state": state,
        "camera_inventory": {
            name: {"mount": record.get("mount"), "calibration_dynamic": record.get("calibration_dynamic", False),
                   "modalities": record.get("modalities", []), "size": [record.get("width"), record.get("height")]}
            for name, record in sorted(state.get("cameras", {}).items()) if isinstance(record, dict)
        },
        "image_camera": camera,
        "image_source_sha256": source["sha256"],
        "image_sha256": hashlib.sha256(encoded).hexdigest(),
        "image_base64": _annotated.encoded(encoded),
    }
