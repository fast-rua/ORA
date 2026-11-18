"""One call on the toolkit: the result, refusal and action the toolkit answered while the handlers were its oracle.

A row is one call on one surface, on a stand-in host on the toolkit. It checks what the model gets back and the
``submitted-action.json`` the call leaves against the digest in ``tests/data/toolkit-goldens``
(``tests/support/agent/toolkit_goldens.py``). Each digest was recorded beside the tool handlers (``ora/agent/tools``),
which answered every row alike, with one known difference: RoboDojo's ``return_home`` refusal names its own reason on
each stack, and its digest is the toolkit's.

**A stand-in host sees no simulator behaviour.** The host is the real ``ManipulationToolRuntime`` on the real surface, but
its observation is one fabricated image per camera, its perception services are fakes and its environment bridge is a toy
that says yes (``tests/support/agent/stack_parity.py``). No worker answers it and nothing moves. These rows show what the
toolkit says and proposes for one call on one fresh observation. They cannot show what a worker does with the action, a
contact or physics outcome, a receipt, the next turn, or what a sequence of calls builds up (a held target, a retrace),
so a green row is no substitute for a smoke episode on the backend.

The RoboDojo rows run its toolkit surface without synchronous execution, so its calls propose their actions: a path no
run takes. ``robot`` is ``robot_runtime_v2``, the surface of ``ora run --env synthetic``, on ``RobotToolRuntime``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pytest

from tests.support.agent.stack_parity import (
    LAST_PREVIEW, MANUAL_SECTION, SURFACES, Call, answered, on_toolkit, outcome, resolved, stand_in,
)
from tests.support.agent.toolkit_goldens import Goldens
from tests.support.agent.toolkit_recording import proposal

GOLDEN = Goldens(__file__)

BLOCK_UV = [64, 64]
POSITION = [0.02, 0.0, 1.05]
OUTSIDE = [3.0, 3.0, 3.0]
REASON = "the stand-in world offers nothing to pick up here"
EDGE = "no_such_camera"
#: ``rotation`` is a quarter turn about the world's +Z.
TURN_XYZW = [0.0, 0.0, 0.70710678, 0.70710678]
SOLVED = {"terminal": False, "task_success": True}
UNSOLVED = {"terminal": False, "task_success": False}
ENDED = {"terminal": True, "task_success": False}


def atomic_rows(key: str, camera: str) -> dict[str, Call]:
    """What only the atomic surfaces have: their one pixel per ``mark_point`` call and their poses. A pose is spelled as
    the surface publishes it, and every sequence runs on each stack alone."""

    frame_contract = SURFACES[key].atomic_position == "target_xyz_world_m"
    position, delta, frame, approach, jaw = (
        ("target_xyz_world_m", "translation_delta_mm", "translation_delta_frame", "grip_site_z_axis_world",
         "grip_site_x_axis_world") if frame_contract
        else ("xyz_m", "delta_mm", "delta_frame", "approach_world", "jaw_world"))
    aimed = {position: POSITION}
    mark = {"point_id": "block", "view": camera, "x": BLOCK_UV[0], "y": BLOCK_UV[1]}
    top = {"point_id": "cloud", "view": "pointcloud_top", "x": 20, "y": 30}
    front = {"point_id": "cloud", "view": "pointcloud_front", "x": 20, "y": 30}
    marked = (Call("mark_point", mark),)
    clicked = (Call("mark_point", top),)
    move = Call("move_pose", aimed)
    previewed = (Call("preview_move", aimed),)
    rows = {
        "observe/pointcloud": Call("observe", {"views": ["pointcloud_top", "agentview"]}),
        "observe/history": Call("observe", {"history_point_ids": ["block"]}, first=marked),
        "observe/history_unsolved": Call("observe", {"history_point_ids": ["ghost"]}),
        "observe/unknown_field": Call("observe", {"step": 0}),
        "mark_point/candidate": Call("mark_point", {**mark, "view": "candidate_agentview"}),
        "mark_point/outside": Call("mark_point", {**mark, "x": 99999}),
        "mark_point/batched": Call("mark_point", {"view": camera, "point_ids": ["block"], "points_uv": [BLOCK_UV]}),
        "mark_point/missing_field": Call("mark_point", {"view": camera}),
        "mark_point/not_integer": Call("mark_point", {**mark, "x": 1.5}),
        "mark_point/empty_name": Call("mark_point", {**mark, "point_id": ""}),
        "mark_point/repeat": Call("mark_point", mark, first=marked),
        "mark_point/verified": Call("mark_point", {**mark, "x": BLOCK_UV[0] + 1}, first=marked),
        "mark_point/other_view": Call("mark_point", {**mark, "view": "wrist"}, first=marked),
        "mark_point/disagrees": Call("mark_point", {**mark, "x": 10, "y": 10}, first=marked),
        "mark_point/cloud_pending": Call("mark_point", top),
        "mark_point/cloud_solved": Call("mark_point", front, first=clicked),
        "mark_point/cloud_repeat": Call("mark_point", top, first=clicked),
        "mark_point/cloud_disagrees": Call("mark_point", {**top, "view": "pointcloud_side", "x": 200, "y": 30}, first=clicked),
        "mark_point/cloud_outside": Call("mark_point", {**top, "x": 99999}),
        "mark_point/cloud_verified": Call("mark_point", {**mark, "view": "pointcloud_top", "x": 20, "y": 30}, first=marked),
        "mark_point/camera_after_click": Call("mark_point", {**top, "view": camera, **dict(zip("xy", BLOCK_UV))}, first=clicked),
        "preview_move/delta": Call("preview_move", {delta: [10.0, 0.0, -5.0]}),
        "preview_move/delta_local": Call("preview_move", {delta: [10.0, 0.0, -5.0], frame: "grip_site"}),
        "preview_move/axes": Call("preview_move", {approach: [0.0, 0.0, -1.0], jaw: [1.0, 0.0, 0.0]}),
        "preview_move/parallel_axes": Call("preview_move", {approach: [0.0, 0.0, -1.0], jaw: [0.0, 0.0, -1.0]}),
        "preview_move/both": Call("preview_move", {position: POSITION, delta: [1.0, 0.0, 0.0]}),
        "preview_move/bad_frame": Call("preview_move", {delta: [1.0, 0.0, 0.0], frame: "camera"}),
        "preview_move/empty": Call("preview_move", {}),
        "preview_move/outside": Call("preview_move", {position: OUTSIDE}),
        "preview_move/short": Call("preview_move", {position: [0.1]}),
        "preview_move/unknown_field": Call("preview_move", {"nowhere": 1}),
        "move_pose/previewed": Call("move_pose", {"preview_id": LAST_PREVIEW}, first=previewed),
        "move_pose/delta": Call("move_pose", {delta: [10.0, 0.0, -5.0]}),
        "move_pose/axes": Call("move_pose", {approach: [0.0, 0.0, -1.0]}),
        "move_pose/second": Call("move_pose", aimed, first=(move,)),
        "move_pose/terminal": Call("move_pose", aimed, task_check=ENDED),
        "close_gripper/second": Call("close_gripper", {}, first=(move,)),
        "close_gripper/terminal": Call("close_gripper", {}, task_check=ENDED),
        "close_gripper/arguments": Call("close_gripper", {"force": 1}),
        "open_gripper/terminal": Call("open_gripper", {}, task_check=ENDED),
        "check_task/confirmed": Call("check_task", {}, task_check=SOLVED),
        "finish_episode/success": Call("finish_episode", {"outcome": "success", "reason": REASON}, task_check=SOLVED),
        "finish_episode/unconfirmed": Call("finish_episode", {"outcome": "success"}, task_check=UNSOLVED),
        "finish_episode/ended": Call("finish_episode", {"outcome": "success"}, task_check=ENDED),
        "finish_episode/latched": Call("finish_episode", {"outcome": "abort", "reason": REASON}, task_check=SOLVED),
        "finish_episode/no_reason": Call("finish_episode", {"outcome": "failure", "attempts": ["one"]}),
        "finish_episode/no_attempts": Call("finish_episode", {"outcome": "failure", "reason": REASON}),
        "finish_episode/blank_attempt": Call("finish_episode", {"outcome": "failure", "reason": REASON, "attempts": [" "]}),
        "finish_episode/unknown": Call("finish_episode", {"outcome": "give_up", "reason": REASON}),
        "finish_episode/pending": Call("finish_episode", {"outcome": "abort", "reason": REASON}, first=(move,)),
    }
    if frame_contract:
        rows.update({
            "preview_move/quaternion": Call("preview_move", {"target_orientation_world_xyzw": TURN_XYZW}),
            "preview_move/quaternion_and_axis": Call("preview_move", {"target_orientation_world_xyzw": TURN_XYZW, approach: [0.0, 0.0, 1.0]}),
            "move_pose/quaternion": Call("move_pose", {position: POSITION, "target_orientation_world_xyzw": TURN_XYZW}),
        })
    return rows


def robot_rows() -> dict[str, Call]:
    """``robot_runtime_v2``'s five tools, what ``ora run --env synthetic`` and ``ora interactive`` advertise."""

    pose = {"position_m": POSITION, "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    submit = Call("robot.submit_action", {"arm": "left", "pose": pose})
    submitted = (submit,)
    inspect = {"camera": "agentview"}
    query = {"camera": "agentview", "points": [BLOCK_UV]}
    return {
        "dump_state": Call("observation.dump_state", {}),
        "dump_state/step": Call("observation.dump_state", {"step": 0}),
        "dump_state/past_the_end": Call("observation.dump_state", {"step": 9}),
        "dump_state/negative": Call("observation.dump_state", {"step": -1}),
        "dump_state/flag": Call("observation.dump_state", {"step": True}),
        "dump_state/unknown_field": Call("observation.dump_state", {"views": []}),
        "dump_state/pending": Call("observation.dump_state", {}, first=submitted),
        "inspect": Call("vision.inspect", inspect),
        "inspect/full": Call("vision.inspect", {**inspect, "resolution": "full"}),
        "inspect/depth": Call("vision.inspect", {**inspect, "modality": "depth"}),
        "inspect/crop": Call("vision.inspect", {**inspect, "crop": [40, 40, 90, 90], "resolution": "full"}),
        "inspect/depth_crop": Call("vision.inspect", {**inspect, "modality": "depth", "crop": [0, 0, 60, 60]}),
        "inspect/step": Call("vision.inspect", {**inspect, "step": 0}),
        "inspect/past_the_end": Call("vision.inspect", {**inspect, "step": 9}),
        "inspect/unknown_camera": Call("vision.inspect", {"camera": EDGE}),
        "inspect/unknown_field": Call("vision.inspect", {**inspect, "scale": 2}),
        "inspect/no_camera": Call("vision.inspect", {}),
        "inspect/bad_modality": Call("vision.inspect", {**inspect, "modality": "normals"}),
        "inspect/bad_crop": Call("vision.inspect", {**inspect, "crop": [1, 2]}),
        "inspect/crop_outside": Call("vision.inspect", {**inspect, "crop": [0, 0, 9999, 10]}),
        "inspect/bad_resolution": Call("vision.inspect", {**inspect, "resolution": "tiny"}),
        "inspect/pending": Call("vision.inspect", inspect, first=submitted),
        "inspect/pending_step": Call("vision.inspect", {**inspect, "step": 0}, first=submitted),
        "query": Call("geometry.query", query),
        "query/region": Call("geometry.query", {"camera": "agentview", "region": [40, 40, 90, 90]}),
        "query/world": Call("geometry.query", {"camera": "agentview", "world_points_m": [POSITION]}),
        "query/step": Call("geometry.query", {**query, "step": 0}),
        "query/past_the_end": Call("geometry.query", {**query, "step": 9}),
        "query/unknown_camera": Call("geometry.query", {**query, "camera": EDGE}),
        "query/neither": Call("geometry.query", {"camera": "agentview"}),
        "query/both": Call("geometry.query", {**query, "region": [0, 0, 1, 1]}),
        "query/unknown_field": Call("geometry.query", {**query, "scale": 2}),
        "query/bad_window": Call("geometry.query", {**query, "window_px": 99}),
        "query/pending": Call("geometry.query", query, first=submitted),
        "query/pending_step": Call("geometry.query", {**query, "step": 0}, first=submitted),
        "manual": Call("robot.read_manual", {"section": MANUAL_SECTION}),
        "manual/unknown_section": Call("robot.read_manual", {"section": "Nowhere"}),
        "manual/no_section": Call("robot.read_manual", {}),
        "manual/not_text": Call("robot.read_manual", {"section": 3}),
        "manual/pending": Call("robot.read_manual", {"section": MANUAL_SECTION}, first=submitted),
        "submit/pose": submit,
        "submit/gripper": Call("robot.submit_action", {"arm": "right", "gripper": 0.5}),
        "submit/both": Call("robot.submit_action", {"arm": "right", "pose": pose, "gripper": 1}),
        "submit/arm_alone": Call("robot.submit_action", {"arm": "right"}),
        "submit/null_gripper": Call("robot.submit_action", {"arm": "right", "pose": pose, "gripper": None}),
        "submit/second": Call("robot.submit_action", {"arm": "right", "gripper": 0.5}, first=submitted),
        "submit/second_bad": Call("robot.submit_action", {"arm": "nowhere"}, first=submitted),
        "submit/unknown_field": Call("robot.submit_action", {"arm": "right", "action": "move_pose"}),
        "submit/no_arm": Call("robot.submit_action", {"gripper": 0.5}),
        "submit/bad_arm": Call("robot.submit_action", {"arm": "middle"}),
        "submit/bad_pose": Call("robot.submit_action", {"arm": "left", "pose": {"position_m": POSITION}}),
        "submit/short_position": Call("robot.submit_action", {"arm": "left", "pose": {**pose, "position_m": [1.0]}}),
        "submit/outside": Call("robot.submit_action", {"arm": "left", "pose": {**pose, "position_m": OUTSIDE}}),
        "submit/not_unit": Call("robot.submit_action", {"arm": "left", "pose": {**pose, "orientation_xyzw": [0.0, 0.0, 0.0, 2.0]}}),
        "submit/bad_gripper": Call("robot.submit_action", {"arm": "left", "gripper": 2.0}),
        "submit/flag_gripper": Call("robot.submit_action", {"arm": "left", "gripper": True}),
    }


def calls(key: str) -> dict[str, Call]:
    """Every row's call on one surface, spelled as its moving tools take a target."""

    surface = SURFACES[key]
    if surface.robot:
        return robot_rows()
    camera = surface.cameras[0]

    def targets(flat: Mapping, nested: object = None) -> dict:
        return dict(flat) if surface.flat else {"targets": {"right": dict(flat) if nested is None else nested}}

    pose = {"position_mode": "absolute", "position_m": POSITION, "orientation_mode": "hold_current"}
    outside = {**pose, "position_m": OUTSIDE}
    atomic = surface.atomic_position
    if atomic:
        pose, outside = {atomic: POSITION}, {atomic: OUTSIDE}
    marks = ({"point_id": "block", "x": BLOCK_UV[0], "y": BLOCK_UV[1]} if atomic
             else {"point_ids": ["block"], "points_uv": [BLOCK_UV]})
    finished = ({"outcome": "failure", "attempts": ["one attempt"]} if key in {"yam", "synthetic", "local"}
                else {"outcome": "give_up", "category": "other"})
    rows = {
        "observe": Call("observe", {}),
        "observe/unknown_view": Call("observe", {"views": [EDGE]}),
        "mark_point": Call("mark_point", {"view": camera, **marks}),
        "mark_point/unknown_view": Call("mark_point", {"view": EDGE, **marks}),
        "geometry_query": Call("geometry_query", {"camera": camera, "points": [BLOCK_UV]}),
        "geometry_query/unknown_camera": Call("geometry_query", {"camera": EDGE, "points": [BLOCK_UV]}),
        "compute": Call("compute", {"expressions": "x = 1 + 2", "outputs": ["x"]}),
        "vision_segment": Call("vision_segment", {"camera": camera, "points_uv": [BLOCK_UV]}),
        "vision_ground": Call("vision_ground", {"camera": camera, "text": "block"}),
        "grasp_plan": Call("grasp_plan", {} if surface.flat else {"arm": "right"}, needs="target"),
        "align_plan": Call("align_plan", {"approach_side": "top", "arm": "right"}, needs="target"),
        "grasp_execute": Call("grasp_execute", {"rationale": "the one candidate on the block"}, needs="candidate"),
        "set_gripper/close": Call("set_gripper", targets({"action": "close"}, "close")),
        "set_gripper/open": Call("set_gripper", targets({"action": "open"}, "open")),
        "set_gripper/close_target": Call("set_gripper", {**targets({"action": "close"}, "close"), "target_ref": "block"}),
        "move_to": Call("move_to", targets({"position_m": POSITION})),
        "move_to/outside": Call("move_to", targets({"position_m": OUTSIDE})),
        "move_to/speed": Call("move_to", {**targets({"position_m": POSITION}), "speed": "careful"}),
        "move_pose": Call("move_pose", targets(pose)),
        "move_pose/outside": Call("move_pose", targets(outside)),
        "move_pose/speed": Call("move_pose", {**targets(pose), "speed": "fast"}),
        "move_pose/dry_run": Call("move_pose", {**targets(pose), "dry_run": True}),
        "rotate_wrist": Call("rotate_wrist", targets({"angle_deg": 10.0}, 10.0)),
        "rotate_pitch": Call("rotate_pitch", targets({"angle_deg": 10.0}, 10.0)),
        "recover_pose": Call("recover_pose", {"arms": ["right"]} if surface.handlers_only
                             else targets({"strategy": "previous"}, "previous")),
        "return_home": Call("return_home", {"arms": ["right"]}),
        "wait_ticks": Call("wait_ticks", {"ticks": 2}),
        "wait_ticks/too_many": Call("wait_ticks", {"ticks": 21}),
        "move_base": Call("move_base", {"forward_m": 0.1}),
        "move_base/absolute": Call("move_base", {"x_m": 0.5, "y_m": -0.25, "yaw_deg": 200.0}),
        "move_base/relative": Call("move_base", {"forward_m": 0.3, "left_m": -0.2, "turn_deg": 45.0, "x_m": None}),
        "move_base/mixed": Call("move_base", {"x_m": 0.5, "y_m": 0.0, "yaw_deg": 0.0, "forward_m": 0.1}),
        "move_base/too_far": Call("move_base", {"forward_m": 6.5}),
        "move_base/dry_run": Call("move_base", {"turn_deg": -30.0, "dry_run": True}),
        "report_issue": Call("report_issue", {"message": "stand-in"}),
        "report_issue/blank": Call("report_issue", {"message": "   "}),
        "check_task": Call("check_task", {}),
        "finish_episode": Call("finish_episode", {**finished, "reason": REASON}),
        "finish_episode/abort": Call("finish_episode", {"outcome": "abort", "reason": REASON}),
    }
    if atomic:
        rows.update({"move_pose/preview": Call("move_pose", {"preview_id": "preview-unknown"}),
                     "preview_move": Call("preview_move", pose), "open_gripper": Call("open_gripper", {}),
                     "close_gripper": Call("close_gripper", {}), **atomic_rows(key, camera)})
    return rows


def rows() -> list[tuple[str, str]]:
    return [(key, name) for key in SURFACES for name, call in calls(key).items() if call.tool in _tools(key)]


def _tools(key: str) -> set[str]:
    return {item["name"] for item in resolved(SURFACES[key].profile).definitions}


@pytest.mark.parametrize("key,name", rows(), ids=lambda value: value)
def test_the_toolkit_answers_a_call_as_it_did_beside_the_handlers(tmp_path, monkeypatch, key, name):
    GOLDEN.check(f"{key}/{name}", answered(tmp_path / "run", key, calls(key)[name], monkeypatch))


@pytest.mark.parametrize("key", SURFACES)
def test_every_tool_of_a_surface_has_a_row(key):
    assert _tools(key) <= {call.tool for call in calls(key).values()}


@pytest.mark.parametrize("key", SURFACES)
def test_a_toolkit_host_runs_the_toolkit(tmp_path, monkeypatch, key):
    """A call that reached no toolkit tool would leave no record of its inputs; this one does."""
    host = on_toolkit(stand_in(tmp_path, key, monkeypatch))
    assert "result" in outcome(host, calls(key)["manual" if SURFACES[key].robot else "report_issue"])
    assert (host.control_root.parent / "native/toolkit-io/tool-inputs.jsonl").is_file()


def _committed_and_driven(host):
    """Ground, plan and commit a grasp on ``host``, then drive its stages as the Supervisor does."""
    def call(index, tool, arguments):
        context = proposal(host, tool, arguments, index)
        invoke = getattr(host, "invoke_with_context", None)
        return invoke(tool, arguments, call_context=context) if invoke else host.invoke(tool, arguments)

    grounded = call(0, "vision_ground", {"camera": SURFACES["robotwin"].cameras[0], "text": "block"})
    planned = call(1, "grasp_plan", {"target_ref": grounded["candidates"][0]["target_ref"], "arm": "right"})
    committed = call(2, "grasp_execute", {"candidate_id": planned["candidates"][0]["candidate_id"], "rationale": REASON})
    action = json.loads((host.control_root / "submitted-action.json").read_bytes())
    root = Path(json.loads((host.control_root / "current-observation.json").read_bytes())["source_root"])
    observation = json.loads((root / "observation.json").read_bytes())
    stages = []
    for index, stage in enumerate(action["arguments"]["sequence"]):
        driven = host.drive_skill_stage(action, stage=stage, stage_index=index, observation=observation,
                                        observation_root=root, completed=list(stages),
                                        capture_observation=lambda: (observation, root))
        stages.append({"stage": stage, "status": driven.get("status", "executed"), "reason": driven.get("reason", ""),
                       "action_id": (driven.get("action") or {}).get("action_id"), "driven": driven})
    receipt = host.build_skill_receipt(action=action, status="completed", stages=stages, observation=observation,
                                       stop_reason="sequence_completed")
    seen = {"committed": committed, "action": action, "stages": stages, "receipt": receipt}
    root = str(host.control_root.parent)
    # The delivered body names the run's root once, and the receipt states the body's size.
    seen["receipt"]["delivery"]["body_bytes"] -= len(root)
    return json.loads(json.dumps(seen).replace(root, "<root>"))


def test_the_stage_loop_drives_what_the_toolkit_committed(tmp_path, monkeypatch):
    """RoboTwin's ``grasp_execute`` hands over a committed sequence, and the Supervisor drives its stages through the
    host's ``drive_skill_stage`` and ``build_skill_receipt``, which stay as they are (decision D7): the stage loop
    reads what the toolkit published."""
    toolkit = _committed_and_driven(on_toolkit(stand_in(tmp_path / "new", "robotwin", monkeypatch)))
    GOLDEN.check("stage_loop", toolkit)
    assert [item["driven"]["kind"] for item in toolkit["stages"]] == ["mutation"] * 5 + ["measurement"]
    assert toolkit["receipt"]["ledger_released"] is True


def test_the_runtime_factory_serves_these_surfaces_from_the_toolkit():
    """What a run builds: the toolkit serves every surface with a row, LIBERO's and RoboTwin's last."""
    from ora.agent.toolkit.host.runtime.dispatch import served

    assert {key for key, surface in SURFACES.items() if not served(resolved(surface.profile))} == set()

