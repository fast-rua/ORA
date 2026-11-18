"""What a dry run of a moving action returns (``interface/dry_run.py``): the plan the real call would run, not its effect.

One result shape, whoever made the plan. The motion tools call these where they would execute a planned trajectory:
``planned`` for a move's ``move.Planned``, ``recovered`` for a recovery's ``recovery.Planned``, ``jaws`` for a jaw
command and ``composed`` for a compound action. Where the environment plans, ``from_plan`` reads its answer
(``ora.action-plan.v1``). A refusal the real call would give for a request that cannot be planned (an unreachable pose,
a blocked path, a target outside the workspace, a budget too small) is the dry run's answer, not an error: ``answered``
turns it into an infeasible record (``refused``) with the real call's code. A refusal of the request itself (a malformed
argument) and a planner that could not be asked stay refusals, as they are for the real call. An infeasible record says
why in one line (``summary``, also the picture's legend): its code and the refusal's own words, which for an
environment's answer without a message are the planner's first sentence and the refusal detail's
(``_refusal_words.py``), as the real call's receipt says them. Poses are world metres
and xyzw, angles degrees; the numbers are the planner's own, never a guess. Every record that has a path carries the
preview of ``lib.action_preview``. A run whose latency profile turns ``preview_images`` off (``View.render`` False;
``_latency.py``) keeps a dry run's numbers without the picture, says so in ``image``, and draws no picture for a real
call's refusal.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..interface import dry_run
from ..interface.dry_run import body_points, segment
from ..lib import action_preview
from ..lib._clearance import PATH_CLAIM
from ..lib.blobs import read_array, read_json
from ..lib.motion_plan import kinematics
from ..lib.motion_plan.checks import PLANNER_REFUSALS
from . import _latency
from ._refusal_words import refusal_words
from .errors import ToolRefusal

#: What a dry run does not predict, said once in every answer.
CLAIM = ("The robot's own motion on the measured start state: inverse kinematics, forward kinematics and collision "
         "checks (the arm itself, the table top, the other arm), and on a planned route what the current depth images "
         "show; no payload or contact. It does not predict the world's response or that the arm arrives. "
         + action_preview.CLAIM + "Clearance: " + PATH_CLAIM)
#: Refusal codes that are the plan's verdict (the real call refuses with the same code before anything moves). Any
#: other refusal of a dry run is one of the real call too: the request is malformed, or no plan could be made.
VERDICT_CODES = frozenset({
    "paced_path_unreachable", "paced_path_joint_jump", "paced_path_wrist_unwind", "paced_motion_budget",
    "mixed_interactions", "route_contact_approach_exceeds_local_bound", "self_collision", "table_collision",
    "inter_arm_collision", "path_blocked", "contact_path_blocked", "workspace_bounds", "plan_rejected",
    *PLANNER_REFUSALS})
#: Where in the call each verdict code arises (``failure_stage``).
STAGES = {
    "workspace_bounds": "request", "mixed_interactions": "request", "route_contact_approach_exceeds_local_bound": "route",
    "paced_path_unreachable": "ik", "paced_path_joint_jump": "ik", "paced_path_wrist_unwind": "ik",
    "self_collision": "screening", "table_collision": "screening", "inter_arm_collision": "screening",
    "path_blocked": "screening", "contact_path_blocked": "screening", "recovery_self_collision_path": "screening",
    "recovery_inter_arm_collision_path": "screening", "recovery_table_collision_path": "screening",
    "paced_motion_budget": "budget", "plan_rejected": "plan"}
#: What a screen checks.
SCREENED_AGAINST = "the arm itself, the table top and the other arm (SDK collision spheres; no objects or payload)"


#: What a jaw command's segment says of where the jaws end.
JAWS_CLOSE = "the jaws close until they meet something; where they stop is not predicted"
JAWS_OPEN = "the jaws open"


@dataclass(frozen=True)
class View:
    """What a preview reads: the current manifest, its cameras and each hand's public geometry."""
    manifest: Mapping[str, Any]
    cameras: Callable[[], list]
    hands: Mapping[str, Mapping[str, Any]]
    jaws: Any
    held: Any
    #: Each arm's chain of joints at the measured state; an arm not given is not left out of what is measured against.
    body: Callable[[str], Sequence] = lambda arm: ()
    #: Whether the preview draws its picture (the run's ``preview_images`` switch, ``_latency.py``).
    render: bool = True


def view(api, obs, config: Mapping[str, Any]) -> View:
    """The preview's inputs on a toolkit tool's data interface."""
    from . import _held

    refs = {camera.name: camera for camera in getattr(obs, "cameras", ())}

    def read(name: str, role: str) -> Any:
        camera = refs[name]
        if role == "rgb_lossless_png":
            return read_array(api, camera.rgb).numpy()
        if role == "depth_numpy_m_f32":
            return None if camera.depth_m is None else read_array(api, camera.depth_m).numpy()
        return read_json(api, camera.calibration).json()

    def body(arm: str) -> list:
        joints = manifest["proprioception"]["end_effectors"][arm]["arm_joint_qpos"]
        return body_points(lambda rows: kinematics.fk(api, arm=arm, joints=rows), joints)

    motion = config.get("motion") or {}
    jaws = motion.get("jaws")
    manifest = obs.manifest.json()
    return View(manifest, lambda: action_preview.cameras_from(manifest, read), motion.get("preview_hands") or {}, jaws,
                _held.extents(api, jaws) if jaws and config.get("synchronous_execution") else motion.get("held"), body,
                render=_latency.on(config, "preview_images"))


def previewed(record: dict[str, Any], segments: Sequence[Mapping[str, Any]], seen: View | None) -> dict[str, Any]:
    """``record`` with its preview. A preview that cannot be made says so; the plan's answer stands without it.

    Of a refused action only the picture: what was asked has no plan to cost."""
    if seen is None:
        return record
    feasible = record["feasible"]
    if not feasible and not seen.render:  # a refused request's preview is only its picture
        return {**record, "image": dict(_latency.PICTURE_OFF)}
    try:
        arms = {item["arm"] for item in segments if item["kind"] == "arm" and item["arm"] is not None}
        added = action_preview.preview(segments, manifest=seen.manifest, cameras=seen.cameras(), hands=seen.hands,
                                       jaws=seen.jaws, held=seen.held,
                                       body=({arm: seen.body(arm) for arm in arms}
                                             if feasible and seen.jaws else None),
                                       headline=None if feasible else _headline(record),
                                       refused=None if feasible else str(record["failure_code"]),
                                       picture=seen.render)
    except Exception as error:  # noqa: BLE001 - advice never takes the answer down
        return {**record, "image": {"status": "unavailable", "reason": type(error).__name__}}
    png = added.pop("png", None)
    if png is not None:
        added["images_base64"] = [base64.b64encode(png).decode("ascii")]
    arms = dict(record.get("arms") or {})
    for arm, numbers in added.pop("arms").items():
        arms[arm] = {**dict(arms.get(arm) or {}), **numbers}
        arms[arm].get("clearance", {}).pop("claim_boundary", None)
    if not feasible:
        return {**record, **{key: added[key] for key in ("image", "images_base64") if key in added}}
    return {**record, **({"arms": arms} if arms else {}), **added}


#: The most of a refusal's own words the picture's legend repeats.
HEADLINE_REASON_CHARS = 320


def _headline(record: Mapping[str, Any], limit: int | None = HEADLINE_REASON_CHARS) -> str:
    """The legend of a refused request: its code and, where it has one, the refusal's own reason, at most ``limit``
    characters of it."""
    reason = str(((record.get("refusal") or {}).get("message")) or "").strip()
    if limit is not None and len(reason) > limit:
        reason = reason[:limit - 3].rstrip() + "..."
    return f"refused: {record['failure_code']}; nothing moves" + (f". {reason}" if reason else "")


def path_segments(expected: Mapping[str, Sequence], knots: Mapping[str, Any], effectors: Mapping[str, Any]) -> list:
    """A planned trajectory as segments: each arm's measured start and then the pose every knot expects."""
    return [segment("arm", arm, "planned path", planned=True,
                    poses=[[float(value) for value in effectors[arm]["pose_xyz_xyzw"]], *poses],
                    native_actions=knots["native_actions"], duration_s=knots["duration_s"])
            for arm, poses in expected.items()]


def _effectors(seen: View | None) -> Mapping[str, Any]:
    return (((seen.manifest.get("proprioception") or {}).get("end_effectors") or {}) if seen is not None else {})


def jaw_segments(targets: Mapping[str, Any], effectors: Mapping[str, Any], hands: Mapping[str, Any]) -> list:
    """Each arm's jaw command as a segment: from the measured opening to the commanded one, at the measured pose."""
    result = []
    for arm, target in targets.items():
        effector = effectors.get(arm) or {}
        pose = [float(value) for value in effector.get("pose_xyz_xyzw") or ()]
        now = effector.get("gripper")
        goal = {"open": 1.0, "close": 0.0}.get(target, target)
        aperture = (hands.get(arm) or {}).get("maximum_aperture_m")
        width = (lambda opening: None if aperture is None or type(opening) not in (int, float)
                 else round(float(opening) * float(aperture), 5))
        result.append(segment("gripper", arm, JAWS_CLOSE if float(goal) < float(now if now is not None else 1.0)
                              else JAWS_OPEN, poses=[pose, pose], planned=True, opening_from=now, opening_to=goal,
                              jaw_width_from_m=width(now), jaw_width_to_m=width(goal)))
    return result


def jaws(name: str, targets: Mapping[str, Any], seen: View | None) -> dict[str, Any]:
    """The record of a jaw command: nothing is solved, and where a closing jaw stops is the world's."""
    segments = jaw_segments(targets, _effectors(seen), seen.hands if seen is not None else {})
    arms = {item["arm"]: {"opening_from": item["opening_from"], "opening_to": item["opening_to"],
                          "jaw_width_from_m": item["jaw_width_from_m"], "jaw_width_to_m": item["jaw_width_to_m"]}
            for item in segments}
    return previewed(_record(name, arms=arms, route="jaw_command", failure_code=None, failure_stage=None), segments, seen)


def composed(name: str, segments: Sequence[Mapping[str, Any]], seen: View | None, *, route: str,
             **fields: Any) -> dict[str, Any]:
    """The record of a compound action: every segment up to the first contact, and what follows it as not predicted."""
    return previewed(_record(name, route=route, failure_code=None, failure_stage=None, **fields), segments, seen)


def from_plan(name: str, plan: Mapping[str, Any], seen: View | None, *, targets: Mapping[str, Sequence] = ()) -> dict:
    """The record of an environment's answer (``ora.action-plan.v1``); ``unavailable`` never reaches here.

    ``targets`` are the action's goals per arm: a refused action draws its request (``with_request``). The environment's
    answer names the arm's chain of joints, where it could."""
    if seen is not None and plan.get("body"):
        seen = View(seen.manifest, seen.cameras, seen.hands, seen.jaws, seen.held, lambda arm: plan["body"].get(arm, ()),
                    seen.render)
    if plan["status"] == "refused":
        code, stage = str(plan["failure_code"]), str(plan["failure_stage"])
        constraints = plan.get("constraints") or {}
        message = plan.get("message") or refusal_words(constraints, constraints.get("planning_diagnostics")) or None
        refusal = {key: value for key, value in {
            "code": code, "message": message, "constraints": plan.get("constraints")}.items()
            if value not in (None, {})}
        record = _record(name, feasible=False, failure_code=code, failure_stage=stage, refusal=refusal)
        effectors = _effectors(seen)
        return previewed(record, [segment("arm", arm, f"refused: {code}", planned=False,
                                          poses=[effectors[arm]["pose_xyz_xyzw"], goal])
                                  for arm, goal in dict(targets).items() if arm in effectors], seen)
    segments = plan["segments"]
    arms: dict[str, Any] = {}
    for item in segments:
        if item["arm"] is not None and item["kind"] == "arm" and not item["not_predicted"]:
            arms.setdefault(item["arm"], {})["predicted_pose"] = _round(item["poses"][-1])
            if "jaw_flip_retry" in item:  # the path runs the target's jaw twin, as the real call would
                arms[item["arm"]]["jaw_flip_retry"] = dict(item["jaw_flip_retry"])
    for arm, goal in dict(targets).items():
        arms.setdefault(arm, {})["target_pose"] = _round(goal)
    return previewed(_record(name, arms=arms, route=plan["route"], failure_code=None, failure_stage=None), segments, seen)


def with_request(error: ToolRefusal, starts: Mapping[str, Sequence], goals: Mapping[str, Sequence]) -> ToolRefusal:
    """``error`` with the request it refused, which is all a refused plan has to draw."""
    if dry_run.requested():
        error.requested = {arm: [starts[arm], goals[arm]] for arm in goals}
    return error


def pictured(name: str, error: ToolRefusal, starts: Mapping[str, Sequence], goals: Mapping[str, Sequence],
             view: Callable[[], View]) -> ToolRefusal:
    """``error`` with its request (``with_request``) and, on a real call refused by the plan's verdict, the picture a
    dry run would have answered with: the gateway attaches it to the failed result, so the model sees where the refused
    request was and why without asking again. A picture that cannot be drawn leaves the refusal as it was."""
    error = with_request(error, starts, goals)
    if dry_run.requested() or not verdict(error):
        return error
    seen = _seen(view)
    if seen is not None and seen.render:
        error.requested = {arm: [starts[arm], goals[arm]] for arm in goals}
        try:
            drawn = refused(name, error, seen).get("images_base64")
        except Exception:  # noqa: BLE001 - a refusal's picture is a courtesy, its answer stands
            drawn = None
        if drawn:
            error.images_base64 = list(drawn)
    return error


def unavailable(reason: str, **constraints: Any) -> ToolRefusal:
    """The refusal of a dry run no policy-side plan can answer; ``constraints.reason`` says why."""

    return ToolRefusal(f"a dry run of this call is unavailable: {reason}", code=dry_run.UNAVAILABLE, retryable=False,
                       constraints={"reason": reason, **constraints})


def _deg(values) -> list[float]:
    return [round(math.degrees(float(value)), 4) for value in values]


def _round(values, places: int = 5) -> list[float]:
    return [round(float(value), places) for value in values]


def _record(name: str, **fields: Any) -> dict[str, Any]:
    feasible = fields.pop("feasible", True)
    record = {"dry_run": True, "action": name, "status": "planned" if feasible else "infeasible", "feasible": feasible,
              **fields, "claim_boundary": CLAIM}
    if not feasible:  # a refusal has no plan to cost: its line says why, unshortened
        record["summary"] = _headline(record, None)
    return record


def screening(screen: Mapping[str, Any]) -> dict[str, Any]:
    """The screen's answer (``kinematics.screen_path``) as a record: its status, how many segments it checked and the
    least inter-arm clearance where two arms were screened."""

    clearance = screen.get("min_inter_arm_clearance_m")
    return {"status": screen.get("status"), "segments_screened": screen.get("segments_screened"),
            "min_inter_arm_clearance_m": None if clearance is None else round(float(clearance), 5),
            "against": SCREENED_AGAINST}


def planned(name: str, plan, compiled: Mapping[str, Any] | None, *, speed: Mapping[str, Any] | None = None,
            expect_contact: bool = False, view: View | None = None) -> dict:
    """The record of a move (``lib.motion_plan.move.Planned``) the call would run, with its compilation as the real
    call returns it. ``motion.expect_contact`` is the call's flag, as the real call reports it."""

    from ..lib.motion_plan import checks, move

    arms = {}
    for arm, rows in plan.knots["joints"].items():
        arms[arm] = {"target_pose": _round(plan.goals[arm]), "predicted_pose": _round(plan.expected[arm][-1]),
                     "end_joints_deg": _deg(rows[-1]),
                     "joint6": checks.joint6_extremes(plan.path.grid.joints[arm][0], rows)}
    motion = {"native_actions": plan.knots["native_actions"], "whole_native_actions": plan.whole_native_actions,
              "settle_native_actions": plan.settle["native_actions"], "duration_s": plan.knots["duration_s"],
              "fraction_of_move": round(plan.fraction, 3), "native_action_budget": plan.native_action_budget,
              "expect_contact": bool(expect_contact)}
    if plan.path.bridged:
        motion["joint_line_bridges"] = list(plan.path.bridged)
    if plan.path.warnings:
        motion["ik_warning"] = move.IK_UNRECORDED_WARNING
    if plan.fraction < 1.0:
        motion["remaining_m"] = {arm: round(math.dist(plan.expected[arm][-1][:3], goal[:3]), 4)
                                 for arm, goal in plan.goals.items()}
    record = _record(name, arms=arms, route=plan.path.route, screening=screening(plan.screen), motion=motion,
                     **({"speed": dict(speed)} if speed else {}), **({"compiled_pose": compiled} if compiled else {}),
                     failure_code=None, failure_stage=None)
    return previewed(record, path_segments(plan.expected, plan.knots, _effectors(view)), view)


def recovered(name: str, plan, *, route: str, undo: Mapping[str, Any] | None = None, view: View | None = None) -> dict:
    """The record of a recovery (``lib.motion_plan.recovery.Planned``) the call would run: the candidate the first
    clear screen kept, per arm its method and where it ends. ``undo`` (recover_pose) says per arm which call it would
    undo."""

    from ..lib.motion_plan import checks

    arms = {}
    for arm, option in plan.options.items():
        arms[arm] = {"target_pose": _round(plan.goals[arm]), "predicted_pose": _round(plan.expected[arm][-1]),
                     "end_joints_deg": _deg(plan.knots["joints"][arm][-1]), "method": option.method,
                     "joint6": checks.joint6_extremes(plan.grid.joints[arm][0], plan.knots["joints"][arm]),
                     **({"unwind_order": option.detail["unwind_order"]} if "unwind_order" in option.detail else {}),
                     **({"lift": option.detail["retract"]["status"]} if "retract" in option.detail else {}),
                     **({"undo": undo[arm]} if undo and arm in undo else {})}
    motion = {"native_actions": plan.knots["native_actions"], "whole_native_actions": plan.whole_native_actions,
              "settle_native_actions": plan.settle["native_actions"], "duration_s": plan.knots["duration_s"],
              "fraction_of_move": round(plan.fraction, 3), "native_action_budget": plan.native_action_budget}
    kept = plan.screens[-1] if plan.screens else {}
    record = _record(name, arms=arms, route=route, motion=motion, failure_code=None, failure_stage=None,
                     screening={"status": kept.get("status"), "candidates_screened": len(plan.screens),
                                "against": SCREENED_AGAINST})
    return previewed(record, path_segments(plan.expected, plan.knots, _effectors(view)), view)


def refused(name: str, error: ToolRefusal, view: View | None = None) -> dict:
    """The record of a call the real call would refuse: its code, stage and the refusal itself, and, where the refusal
    names its request (``with_request``), a picture of what was asked."""

    constraints = dict(error.constraints)
    code = str(constraints.get("failure_code") or error.code) if error.code == "plan_rejected" else error.code
    refusal = {key: value for key, value in {"code": error.code, "message": str(error), "field": error.fields.get("field"),
                                             "constraints": constraints}.items() if value not in (None, {})}
    record = _record(name, feasible=False, failure_code=code, failure_stage=STAGES.get(code, "plan"), refusal=refusal)
    asked = getattr(error, "requested", None)
    if not asked:
        return record
    return previewed(record, [segment("arm", arm, f"refused: {code}", planned=False, poses=poses)
                              for arm, poses in asked.items()], view)


def verdict(error: ToolRefusal) -> bool:
    """Whether a refusal is the plan's verdict (``VERDICT_CODES``) rather than an error of the request."""

    return error.code in VERDICT_CODES


def answered(name: str, call: Callable[[], dict], view: Callable[[], View] | None = None) -> dict:
    """``call()`` (a tool's dry run) with a refusal that is the plan's verdict turned into its infeasible record."""

    try:
        return call()
    except ToolRefusal as error:
        if verdict(error):
            return refused(name, error, None if view is None or not getattr(error, "requested", None) else _seen(view))
        raise


def _seen(view: Callable[[], View]) -> View | None:
    try:
        return view()
    except Exception:  # noqa: BLE001 - a refusal's picture is a courtesy, its answer stands
        return None


__all__ = ["CLAIM", "JAWS_CLOSE", "JAWS_OPEN", "SCREENED_AGAINST", "STAGES", "VERDICT_CODES", "View", "answered", "pictured",
           "composed", "from_plan", "jaw_segments", "jaws", "path_segments", "planned", "previewed", "recovered",
           "refused", "screening", "unavailable", "verdict", "view", "with_request"]
