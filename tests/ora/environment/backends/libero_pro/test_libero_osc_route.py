"""v4's free-space route choice on LIBERO, the straight OSC line's commands and the step budget's count.

A planned ``free_space`` move whose plan is clear and nearly straight runs as one straight OSC line;
every other planned move keeps the planner's joint path, and the move's sparse receipt says which
route ran and why (``arms.right.route_choice``). Every LIBERO receipt counts its control ticks as
``native_actions``, the units of the episode's step limit.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ora.environment.backends.libero_pro.session import (
    LiberoProSession, libero_environment_init_params, libero_public_environment_context,
)
from ora.environment.controllers import osc_transit
from ora.orchestrator.episode.native_allowance import NativeActionAllowance, effective_native_action_limit
from tests.support.libero.correctness_repair import ROOT, TARGET, _session
from tests.support.orchestrator.runtime_v1 import FakeLiberoEnvironment, _fake_libero_metadata

PROFILES = ROOT / "ora/policy/embodiments/franka-panda"


def _action() -> dict[str, Any]:
    return {
        "schema": "ora-motion-action.v1", "action_id": "action-0", "observation_id": "observation-0",
        "bound_observation_sha256": "0" * 64, "action": "move_free_space",
        "arguments": {"arm": "right", "target_pose": {"position_m": TARGET[:3], "orientation_xyzw": TARGET[3:]}},
    }


def _planned(tmp_path, monkeypatch, profile_name: str, interaction: str | None):
    session, plant = _session(tmp_path, monkeypatch, profile_name=profile_name)
    session._active_motion_interaction = interaction
    session._active_sparse_plan = SimpleNamespace(routes=["libero_curobo_joint_position"], route_choice=None)
    ran: dict[str, Any] = {}

    def osc_line(action, target_pose, planner_evidence, choice):
        ran.update(target=list(target_pose), choice=choice, planner=planner_evidence)
        return {"osc": True}

    session._run_osc_transit = osc_line  # type: ignore[method-assign]
    return session, plant, ran


def test_a_clear_straight_free_space_plan_runs_as_the_osc_line(tmp_path, monkeypatch) -> None:
    session, plant, ran = _planned(tmp_path, monkeypatch, "motion-profile-v4.json", "free_space")
    assert session._execute_planned_motion(_action()) == {"osc": True}
    assert plant.commands == []  # nothing ran on the joint route
    choice = ran["choice"]
    assert (choice["route"], choice["reason"]) == (osc_transit.ROUTE_OSC, "planner_path_clear_and_straight")
    assert choice["collision_checked"] == "planner_gripper_path_only"
    assert choice["path_deviation_m"] == pytest.approx(0.0, abs=1e-9)
    assert ran["target"] == TARGET and ran["planner"]["canonical_artifact"]
    assert session._active_sparse_plan.route_choice is choice


@pytest.mark.parametrize("profile_name,interaction,reason", [
    ("motion-profile-v4.json", "contact_approach", "not_free_space"),
    ("motion-profile-v4.json", None, "not_free_space"),
    ("motion-profile-v3.json", "free_space", None),
])
def test_every_other_planned_move_keeps_the_joint_path(tmp_path, monkeypatch, profile_name, interaction, reason):
    session, plant, ran = _planned(tmp_path, monkeypatch, profile_name, interaction)
    receipt = session._execute_planned_motion(_action())
    assert ran == {} and plant.commands and receipt["motion_status"] == "reached"
    choice = session._active_sparse_plan.route_choice
    assert (choice or {}).get("reason") == reason
    assert choice is None or choice["route"] == osc_transit.ROUTE_JOINT


def _local_session(tmp_path: Path) -> tuple[LiberoProSession, FakeLiberoEnvironment]:
    environment = FakeLiberoEnvironment()
    context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100))
    session = LiberoProSession(
        libero_pro_root=tmp_path / "LIBERO-PRO", data_root=tmp_path / "data", output_root=tmp_path / "run",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate", required_libero_pro_revision="libero-test",
        required_data_revision="data-test", workspace_bounds_m={"x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3]},
        max_sim_steps=100, environment_factory=lambda **_kwargs: (environment, _fake_libero_metadata(tmp_path, context)),
    )
    session.start()
    session.motion_profile = json.loads((PROFILES / "motion-profile-v4.json").read_text(encoding="utf-8"))
    return session, environment


def test_the_osc_line_aims_each_tick_at_its_reference_within_the_controller_reach(tmp_path) -> None:
    session, environment = _local_session(tmp_path)
    commands: list[list[float]] = []
    step = environment.step

    def record(action):
        commands.append(np.asarray(action)[0].tolist())
        return step(action)

    environment.step = record  # type: ignore[method-assign]
    start = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
    goal = [0.0, 0.0, 0.53, 0.0, 0.0, 0.0, 1.0]
    line = osc_transit.straight_line_reference(
        start, goal, linear_m_s=0.25, angular_rad_s=1.5, linear_accel_m_s2=2.0, angular_accel_rad_s2=12.0,
        tick_s=0.05, lead_s=0.163)
    # One aim beyond the reach of a tick, so the clip to the controller's 0.05 m shows.
    line["aims"][0] = [0.0, 0.0, 0.62, 0.0, 0.0, 0.0, 1.0]
    session._active_osc_transit = {**line, "settle_ticks": 3}
    try:
        receipt = session.execute({
            "schema": "ora-action.v3", "observation_id": "observation-0", "action_id": "line",
            "action": "move_pose", "bound_observation_sha256": "bound",
            "arguments": {"arm": "right", "position_m": goal[:3], "orientation_xyzw": goal[3:]},
        })
    finally:
        session.close()
    # The fake moves the hand by the commanded offset: tick 1 is clipped to 0.05 m (normalized 1).
    assert commands[0][2] == pytest.approx(1.0)
    assert receipt["steps_used"] <= line["ticks"] + 3
    assert receipt["native_actions"] == receipt["steps_used"]


def test_a_libero_plan_reports_its_step_limit_as_the_native_action_allowance() -> None:
    plan = {"runtime": {"kind": "libero_pro"}, "budgets": {"max_sim_steps": 220}}
    assert effective_native_action_limit(plan) == 220
    allowance = NativeActionAllowance(plan)
    allowance.record({"receipt": {"native_actions": 107}})
    allowance.record({"receipt": {"native_actions": 17}})
    assert allowance.fields() == {"native_actions_limit": 220, "native_actions_remaining": 96}
    allowance.record({"receipt": {"steps_used": 3}})  # an uncounted receipt: the remainder is unknown
    assert allowance.fields()["native_actions_remaining"] is None
    # RoboTwin declares no such limit and still reports nothing.
    assert effective_native_action_limit({"runtime": {"kind": "robotwin"}, "budgets": {"max_sim_steps": 400}}) is None


def test_every_libero_receipt_counts_its_control_ticks(tmp_path, monkeypatch) -> None:
    session, plant = _session(tmp_path, monkeypatch, profile_name="motion-profile-v4.json")
    receipt = session._execute_planned_motion(_action())
    assert receipt["native_actions"] == receipt["steps_used"] == len(plant.commands) > 0
