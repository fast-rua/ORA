"""A dry run where the environment plans the move: its tool stops at ``dry_run.Proposed``, and this answers instead.

The tool has compiled and validated the action as for a real call. A jaw command needs no planner; anything that moves
the arm or the base is asked of the environment on the read-only bridge (``contracts/action_plan.py``), which never steps
the world and keeps its plan for the real call. A compound action is its stages in order: the first path is asked the
same way, what leads to contact is its commanded line, and what follows contact is marked as not predicted. A backend
that cannot answer without stepping the world says so, and the call is refused ``dry_run_unavailable``: never a fake.
"""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ora.agent.toolkit.interface import dry_run
from ora.agent.toolkit.lib import action_preview
from ora.agent.toolkit.tools import _dry_run, _latency
from ora.agent.error_classes import ToolError
from ora.contracts.action import MOTION_ACTION_V3_SCHEMA
from ora.contracts.action_plan import is_request, request, segment

from .motion import preview_geometry

#: The stages of a compound action that reach the contact, and how each is commanded.
APPROACH = "the approach is commanded along this line, up to the first contact"
AFTER_CONTACT = "after contact: not predicted"


def _reader(root: Path, manifest: Mapping[str, Any]) -> Any:
    """Decoded camera files of this observation, hash-checked as every camera read is."""
    resolved = Path(root).resolve()

    def read(name: str, role: str) -> Any:
        reference = ((manifest["cameras"][name].get("files") or {}).get(role)) or {}
        path = (resolved / str(reference.get("path", ""))).resolve()
        if (not path.is_file() or not path.is_relative_to(resolved)
                or hashlib.sha256(path.read_bytes()).hexdigest() != reference.get("sha256")):
            return None
        data = path.read_bytes()
        if role == "calibration":
            return json.loads(data)
        if role == "depth_numpy_m_f32":
            return np.load(BytesIO(data), allow_pickle=False)
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    return read


def view(host: Any) -> _dry_run.View:
    """The preview's inputs on this host's published observation."""
    _control, root, manifest = host._current()
    hands, jaws, held = preview_geometry(host)
    read = _reader(root, manifest)
    return _dry_run.View(manifest, lambda: action_preview.cameras_from(manifest, read), hands,
                         json.loads(json.dumps(jaws)) if jaws else None, held,
                         render=_latency.host_on(host, "preview_images"))


def _asked(host: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    """The environment's read-only answer for ``action``; an environment that cannot plan refuses the dry run."""
    if host.align_plan_callback is None:
        raise ToolError("the environment serves no read-only plan", code=dry_run.UNAVAILABLE, retryable=False,
                        constraints={"reason": "this environment has no read-only bridge"})
    payload = request(action)
    assert is_request(payload)
    answer = dict(host.align_plan_callback(payload))
    if answer["status"] == "unavailable":
        raise ToolError(f"a dry run of this call is unavailable: {answer['reason']}", code=dry_run.UNAVAILABLE,
                        retryable=False, constraints={"reason": answer["reason"]})
    return answer


def _goals(action: Mapping[str, Any]) -> dict[str, list[float]]:
    targets = (action.get("arguments") or {}).get("targets")
    if action.get("action") != "move_pose" or not isinstance(targets, Mapping):
        return {}
    return {arm: [*target["position_m"], *target["orientation_xyzw"]] for arm, target in targets.items()}


def _pose(value: Mapping[str, Any]) -> list[float]:
    return [*value["position_m"], *value["orientation_xyzw"]]


def _compound(host: Any, name: str, action: Mapping[str, Any], commitment: Mapping[str, Any],
              seen: _dry_run.View) -> dict[str, Any]:
    """The stages of a committed sequence: the first path asked, then lines; nothing after the contact is predicted."""
    arm, poses, grippers = commitment["arm"], commitment["poses"], commitment.get("grippers") or {}
    effector = seen.manifest["proprioception"]["end_effectors"][arm]
    last = [float(value) for value in effector["pose_xyz_xyzw"]]
    segments: list[dict[str, Any]] = []
    contact, asked = False, False
    for stage in action["arguments"]["sequence"]:
        target = poses.get(stage)
        if target is None:
            if stage in grippers:
                segments.extend(_dry_run.jaw_segments({arm: grippers[stage]}, {arm: {**effector, "pose_xyz_xyzw": last}},
                                                      seen.hands))
                contact = contact or stage == "close"
            continue
        goal = _pose(target)
        if contact:
            segments.append(segment("arm", arm, AFTER_CONTACT, planned=False, poses=[last, goal], not_predicted=True))
        elif asked:
            segments.append(segment("arm", arm, APPROACH, planned=False, poses=[last, goal]))
        else:
            asked = True
            plan = _asked(host, {**action, "schema": MOTION_ACTION_V3_SCHEMA, "action": "move_pose",
                                 "arguments": {"targets": {arm: dict(target)}, "interactions": {
                                     arm: "contact_approach" if stage == "approach" else "free_space"}}})
            if plan["status"] == "refused":
                return _dry_run.from_plan(name, plan, seen, targets={arm: goal})
            segments.extend(plan["segments"])
        last = goal
    return _dry_run.composed(name, segments, seen, route="grasp_sequence",
                             arms={arm: {"target_pose": _pose(poses["approach"])}}, stages=list(action["arguments"]["sequence"]))


def answer(host: Any, name: str, proposed: dry_run.Proposed) -> dict[str, Any]:
    """The dry-run record of the action a tool proposed."""
    action, seen = proposed.action, view(host)
    if proposed.commitment is not None:
        return _compound(host, name, action, proposed.commitment, seen)
    if action.get("action") == "set_gripper":
        arguments = action["arguments"]
        targets = arguments.get("targets") or {arguments["arm"]: arguments["gripper"]}
        return _dry_run.jaws(name, targets, seen)
    return _dry_run.from_plan(name, _asked(host, action), seen, targets=_goals(action))


__all__ = ["answer", "view"]
