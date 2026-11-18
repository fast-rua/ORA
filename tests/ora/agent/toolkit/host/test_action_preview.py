"""A dry run where the environment plans (``runtime/action_preview.py``): the tool stops where it would hand the action
over, and the read-only bridge answers. Every moving action gets the one record, a compound action its stages, and a
backend that cannot plan without stepping the world refuses instead of faking a preview.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ora.agent.toolkit.interface import dry_run
from ora.agent.toolkit.tools import _dry_run
from ora.agent.toolkit.host import action_preview
from ora.agent.error_classes import ToolError
from ora.contracts import action_plan

START = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
GOAL = [0.1, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
MANIFEST = {"proprioception": {"end_effectors": {"right": {"pose_xyz_xyzw": START, "gripper": 1.0}}}}
HANDS = {"right": {"contact_center_local": [0, 0, 0], "opening_axis_local": [0, 1, 0], "maximum_aperture_m": 0.08}}


def _view(monkeypatch, *_):
    seen = _dry_run.View(MANIFEST, lambda: [], HANDS, None, None)
    monkeypatch.setattr(action_preview, "view", lambda host: seen)


class Host:
    def __init__(self, answer=None):
        self.asked, self.answer = [], answer

    def align_plan_callback(self, request):
        self.asked.append(request)
        return self.answer


def _move():
    return {"action": "move_pose", "arguments": {"targets": {"right": {"position_m": GOAL[:3], "orientation_xyzw": GOAL[3:]}},
                                                 "interactions": {"right": "free_space"}}}


def test_a_move_is_asked_of_the_environment_and_answered_in_the_one_shape(monkeypatch):
    _view(monkeypatch)
    host = Host(action_plan.planned("global", [
        action_plan.segment("arm", "right", "planned path", planned=True, poses=[START, GOAL], native_actions=20),
        action_plan.segment("arm", "right", "the servo settles", planned=False, poses=[GOAL, GOAL])]))
    record = action_preview.answer(host, "move_pose", dry_run.Proposed(_move()))
    assert action_plan.is_request(host.asked[0]) and host.asked[0]["action"]["action"] == "move_pose"
    assert record["dry_run"] is True and record["feasible"] is True and record["route"] == "global"
    arm = record["arms"]["right"]
    assert arm["predicted_pose"] == GOAL and arm["target_pose"] == GOAL and arm["native_actions"] == 20
    assert arm["servo_not_counted"] is True and "share_of_native_actions_left" not in arm
    assert [item["planned"] for item in record["segments"]] == [True, False]
    assert "claim_boundary" in record and "world's response" in record["claim_boundary"]


def test_a_refused_action_is_an_answer_with_the_real_call_s_code(monkeypatch):
    _view(monkeypatch)
    record = action_preview.answer(Host(action_plan.refused("goal_unreachable", "planner", "no solution")), "move_pose",
                                   dry_run.Proposed(_move()))
    assert record["feasible"] is False and record["status"] == "infeasible"
    assert (record["failure_code"], record["failure_stage"]) == ("goal_unreachable", "planner")
    assert record["refusal"]["message"] == "no solution"


def test_a_backend_that_cannot_plan_without_moving_refuses_rather_than_fakes_a_preview(monkeypatch):
    _view(monkeypatch)
    with pytest.raises(ToolError) as refused:
        action_preview.answer(Host(action_plan.unavailable("it retreats first")), "recover_pose", dry_run.Proposed(_move()))
    assert refused.value.code == dry_run.UNAVAILABLE and refused.value.constraints["reason"] == "it retreats first"


def test_the_jaws_need_no_planner_and_a_closing_jaw_s_stop_is_not_predicted(monkeypatch):
    _view(monkeypatch)
    host = Host()
    action = {"action": "set_gripper", "arguments": {"targets": {"right": "close"}}}
    record = action_preview.answer(host, "set_gripper", dry_run.Proposed(action))
    assert host.asked == [] and record["feasible"] is True and record["route"] == "jaw_command"
    arm = record["arms"]["right"]
    assert (arm["opening_from"], arm["opening_to"]) == (1.0, 0.0)
    assert (arm["jaw_width_from_m"], arm["jaw_width_to_m"]) == (0.08, 0.0)
    assert record["segments"][0]["label"] == _dry_run.JAWS_CLOSE and "not predicted" in _dry_run.JAWS_CLOSE


def _pose(pose):
    return {"position_m": pose[:3], "orientation_xyzw": pose[3:]}


def test_a_compound_action_shows_each_segment_up_to_contact_and_what_follows_as_not_predicted(monkeypatch):
    _view(monkeypatch)
    pregrasp, contact, lift = GOAL, [0.1, 0.0, 0.4, 0, 0, 0, 1], [0.1, 0.0, 0.5, 0, 0, 0, 1]
    host = Host(action_plan.planned("global", [action_plan.segment("arm", "right", "planned path", planned=True,
                                                                   poses=[START, pregrasp], native_actions=12)]))
    commitment = {"arm": "right", "poses": {"pregrasp": _pose(pregrasp), "approach": _pose(contact), "lift": _pose(lift)},
                  "grippers": {"open": 1.0, "close": 0.0}}
    action = {"action": "execute_skill_sequence", "observation_id": "o", "bound_observation_sha256": "s",
              "action_id": "action-3",
              "arguments": {"sequence": ["open", "pregrasp", "approach", "close", "lift", "verify"]}}
    record = action_preview.answer(host, "grasp_execute", dry_run.Proposed(action, commitment))
    kinds = [(item["kind"], item["planned"], item["not_predicted"]) for item in record["segments"]]
    assert kinds == [("gripper", True, False), ("arm", True, False), ("arm", False, False), ("gripper", True, False),
                     ("arm", False, True)]
    assert "not predicted" in record["segments"][-1]["label"] and "not predicted" in record["segments"][3]["label"]
    assert len(host.asked) == 1, "only the first path can be asked: the world is measured here"
    assert host.asked[0]["action"]["action"] == "move_pose"
    assert record["arms"]["right"]["native_actions"] == 12, "the lift is not a cost a plan can state"


def test_a_compound_action_whose_first_path_is_refused_refuses(monkeypatch):
    _view(monkeypatch)
    commitment = {"arm": "right", "poses": {"pregrasp": _pose(GOAL), "approach": _pose(GOAL)}, "grippers": {}}
    action = {"action": "execute_skill_sequence", "arguments": {"sequence": ["pregrasp", "approach"]}}
    record = action_preview.answer(Host(action_plan.refused("path_blocked", "planner")), "grasp_execute",
                                   dry_run.Proposed(action, commitment))
    assert record["feasible"] is False and record["failure_code"] == "path_blocked"


def test_the_preview_reads_only_what_the_manifest_lists_and_its_hashes_hold(tmp_path):
    from PIL import Image

    (tmp_path / "eye").mkdir()
    rgb, depth, calibration = tmp_path / "eye/rgb.png", tmp_path / "eye/depth-m.npy", tmp_path / "eye/calibration.json"
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(rgb)
    np.save(depth, np.ones((4, 6), dtype=np.float32), allow_pickle=False)
    calibration.write_text(json.dumps({"intrinsic_cv": np.eye(3).tolist(), "extrinsic_cv": np.eye(4).tolist()}))

    def ref(path):
        import hashlib
        return {"path": path.relative_to(tmp_path).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    manifest = {"cameras": {"eye": {"files": {"rgb_lossless_png": ref(rgb), "depth_numpy_m_f32": ref(depth),
                                              "calibration": ref(calibration)}}}}
    read = action_preview._reader(tmp_path, manifest)
    assert read("eye", "rgb_lossless_png").shape == (4, 6, 3) and read("eye", "depth_numpy_m_f32").shape == (4, 6)
    manifest["cameras"]["eye"]["files"]["rgb_lossless_png"]["sha256"] = "0" * 64
    assert read("eye", "rgb_lossless_png") is None, "a file that is not the published one is not read"


def test_a_dry_run_is_a_read_for_the_supervisor_whatever_the_tool_does_otherwise():
    from types import SimpleNamespace

    from ora.agent.session_core.gateway import NativeToolGateway

    class Runtime:
        surface = SimpleNamespace(capabilities=(), mutation_tools=("move_pose", "wait_ticks"), lifecycle_tools=())
        allowed_tools = ("move_pose", "wait_ticks")

        def classify_call_effect(self, name, arguments):
            return "mutation"

    gateway = NativeToolGateway(runtime=Runtime())
    assert gateway.classify_effect("move_pose", {"targets": {}, "dry_run": True}) == "read_only"
    assert gateway.classify_effect("move_pose", {"targets": {}, "dry_run": False}) == "mutation"
    assert gateway.classify_effect("wait_ticks", {"ticks": 3, "dry_run": True}) == "mutation", "it moves nothing to plan"


@pytest.mark.parametrize("profile", ["robotwin_toolchain_b2_closed_noservo_v2", "libero_placement_direct_overhead_v1",
                                     "robocasa_mobile_direct_v1", "robolab_direct_v1", "robodojo_placement_toolkit_v5"])
def test_every_default_surface_can_read_its_hands_where_the_tool_context_could_not(profile):
    """A live cell found this: the toolkit's tool context needs a reviewed contact centre only the X5 has, so a preview
    built on it failed on every other backend."""
    from types import SimpleNamespace

    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from ora.agent.toolkit.host.motion import preview_geometry
    from tests.support.agent.surface_catalog import catalog_grants

    surface = resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile))
    hands, jaws, held = preview_geometry(SimpleNamespace(surface=surface))
    assert hands and all(len(hand["opening_axis_local"]) == 3 for hand in hands.values())
    assert set(hands) <= set(jaws) and held == {}
    json.dumps(jaws)


def test_a_surface_that_binds_no_jaw_geometry_still_previews_its_path():
    """The YAM surface binds no public geometry descriptor: its path and numbers stand, its clearance is unknown."""
    from types import SimpleNamespace

    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from ora.agent.toolkit.host.motion import preview_geometry

    assert preview_geometry(SimpleNamespace(surface=resolve_tool_surface("yam_direct_v1"))) == ({}, None, {})
