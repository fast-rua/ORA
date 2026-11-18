"""A LIBERO contact move stops when it presses without advancing or leaves its target."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ora.agent.session_core.contact_detail import contact_detail
from ora.agent.session_core.receipt_projection import public_motion_failure_code
from ora.environment.backends.libero_pro.session import (
    LiberoProSession,
    libero_environment_init_params,
    libero_public_environment_context,
)
from ora.environment.controllers.force_feedback import (
    contact_divergence_stop_reason,
    contact_force_stop_reason,
    sensor_reaction_feedback,
)
from tests.support.libero.correctness_repair import JOINT_NAMES, ROOT
from tests.support.orchestrator.runtime_v1 import FakeLiberoEnvironment, _fake_libero_metadata

# Recorded ticks of sealed runs: force in N and distance to the target in mm.
L4_ERRORS_MM = [40.7, 37.0, 30.5, 25.1, 24.1, 26.2, 28.6, 31.5, 35.0, 39.3, 44.4]
KNOB_TURNED = ([33.3, 4.7, 133.2, 86.2, 69.6, 67.5, 4.8, 4.9, 5.0, 44.1, 5.0, 5.1],
               [24.0, 21.9, 19.0, 16.6, 14.4, 12.5, 10.8, 9.2, 7.7, 6.4, 5.1, 3.9])
HAND_STALLED = ([5.1, 4.9, 206.5, 107.9, 105.0, 104.4, 103.8, 103.4],
                [36.6, 33.2, 28.8, 29.0, 28.9, 28.9, 28.9, 28.9])
HELD_WEIGHT_NO_CONTACT = ([45.0] * 8, [30.0, 28.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0])


def first_force_stop(forces: list[float], errors_mm: list[float], interaction: str = "contact_approach",
                     enabled: bool = True) -> int | None:
    for tick in range(1, len(forces) + 1):
        if contact_force_stop_reason(
            interaction=interaction, forces_n=forces[:tick],
            errors_m=[error / 1000 for error in errors_mm[:tick]], threshold_n=40.0, enabled=enabled,
        ):
            return tick - 1
    return None


def first_divergence_stop(errors_mm: list[float], interaction: str = "contact_approach") -> int | None:
    for tick in range(1, len(errors_mm) + 1):
        if contact_divergence_stop_reason(
            interaction=interaction, errors_m=[error / 1000 for error in errors_mm[:tick]], enabled=True,
        ):
            return tick - 1
    return None


def test_a_push_that_is_still_advancing_keeps_going_however_hard() -> None:
    assert first_force_stop(*KNOB_TURNED) is None


def test_a_hand_pressing_without_advancing_stops() -> None:
    assert first_force_stop(*HAND_STALLED) == 4


def test_force_is_read_above_the_moves_lowest_reading() -> None:
    assert first_force_stop(*HELD_WEIGHT_NO_CONTACT) is None


@pytest.mark.parametrize("interaction,enabled", [("precision", True), ("free_space", True),
                                                 ("contact_approach", False)])
def test_the_force_stop_is_for_a_contact_move_of_the_current_profile(interaction: str, enabled: bool) -> None:
    assert first_force_stop(*HAND_STALLED, interaction=interaction, enabled=enabled) is None


def test_a_hand_leaving_its_target_stops_before_it_presses() -> None:
    assert first_divergence_stop(L4_ERRORS_MM) == 7


def test_an_error_that_rises_a_little_or_stops_rising_does_not_stop() -> None:
    assert first_divergence_stop([20.0, 16.0, 17.0, 18.0, 19.0, 19.0, 19.0]) is None
    assert first_divergence_stop([20.0, 10.0, 20.0, 12.0, 21.0, 11.0]) is None


def test_the_divergence_stop_is_for_a_contact_move() -> None:
    assert first_divergence_stop(L4_ERRORS_MM, interaction="precision") is None


def test_the_policy_sees_the_named_code() -> None:
    assert public_motion_failure_code("execution", "tracking_diverged") == "tracking_diverged"


class ScriptedPlant(FakeLiberoEnvironment):
    """The grip site follows a height per tick, and the episode never ends."""

    def __init__(self, heights: list[float]) -> None:
        super().__init__()
        self.heights = heights

    def step(self, action: Any) -> tuple[Any, ...]:
        observation, reward, _terminated, truncated, info = super().step(action)
        selected = dict(self.current_raw_obs[0])
        height = self.heights[min(self.step_count, len(self.heights) - 1)]
        selected["robot0_eef_pos"] = np.asarray([0.0, 0.0, height], dtype=np.float32)
        self.current_raw_obs = np.asarray([selected], dtype=object)
        return observation, reward, np.asarray([False]), truncated, info


def contact_move(tmp_path: Path, heights: list[float], forces: list[float]) -> dict[str, Any]:
    environment = ScriptedPlant(heights)
    public_context = libero_public_environment_context(
        {"problem": "Libero", "problem_folder": "libero_goal_task"},
        init_params=libero_environment_init_params(max_sim_steps=100),
    )
    session = LiberoProSession(
        libero_pro_root=tmp_path / "LIBERO-PRO", data_root=tmp_path / "data", output_root=tmp_path / "native",
        task="libero_goal_task-t00", seed=0, suite="libero_goal_task", task_id=0,
        expected_task_language="put the bowl on the plate",
        required_libero_pro_revision="libero-test", required_data_revision="data-test",
        workspace_bounds_m={"x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3]}, max_sim_steps=100,
        environment_factory=lambda **kwargs: (environment, _fake_libero_metadata(tmp_path, public_context)),
    )
    session.start()
    session.action_codec = "motion_action_v3"
    session.motion_profile = json.loads(
        (ROOT / "ora/policy/embodiments/franka-panda/motion-profile-v2.json").read_text()
    )

    def measured_force() -> dict[str, Any]:
        force = forces[min(environment.step_count, len(forces) - 1)]
        return {"joint_names": JOINT_NAMES, "joint_positions": [0.0] * 7, "joint_velocities": [0.0] * 7,
                "external_force_magnitude_n": force, "external_wrench_clipped": [0, 0, min(force, 40)],
                **sensor_reaction_feedback([0, 0, force], [[1, 0, 0], [0, -1, 0], [0, 0, -1]])}

    session._motion_state = measured_force  # type: ignore[method-assign]
    receipt = session.execute({
        "schema": "ora-motion-action.v3", "observation_id": "observation-0", "action_id": "contact-move",
        "bound_observation_sha256": "bound", "action": "move_pose",
        "arguments": {"targets": {"right": {"position_m": [0.0, 0.0, 0.46],
                                            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}},
                      "interactions": {"right": "contact_approach"}},
    })
    receipt["steps_counted"] = environment.step_count
    session.close()
    return receipt


def test_a_stalled_contact_move_ends_as_a_precision_stop_does(tmp_path: Path) -> None:
    receipt = contact_move(tmp_path, [0.5, 0.49, 0.48, 0.48], [5.0, 5.0, 5.0, 90.0])
    segment = receipt["segment"]
    assert receipt["failure_stage"] == "execution" and receipt["failure_code"] == "contact_stop"
    assert receipt["execution_status"] == "not_reached" and segment["motion_status"] == "stopped"
    assert receipt["steps_counted"] == 5
    assert segment["control_health"]["status"] == "degraded"
    assert segment["control_health"]["force_telemetry"]["stop_authority"] is True
    assert segment["control_health"]["force_telemetry"]["stop_reason"] == "contact_stop"
    assert contact_detail(receipt)["force_stop_threshold_n"] == 40.0
    assert contact_detail(receipt)["external_force_magnitude_n"] == 90.0


def test_a_contact_move_leaving_its_target_stops_and_says_how_far(tmp_path: Path) -> None:
    receipt = contact_move(tmp_path, [0.5, 0.48, 0.47, 0.47, 0.474, 0.478, 0.482, 0.486], [5.0])
    guard = receipt["segment"]["control_health"]["tracking_guard"]
    assert receipt["failure_stage"] == "execution" and receipt["failure_code"] == "tracking_diverged"
    assert receipt["segment"]["motion_status"] == "stopped"
    assert guard["outcome"] == "contact_divergence_stop" and guard["reason"] == "tracking_diverged"
    assert guard["stopped_tick"] == 5
    detail = contact_detail(receipt)
    assert detail["position_error_growth_m"] == pytest.approx(0.008, abs=1e-5)
    assert detail["position_error_growth_limit_m"] == 0.005


def test_a_contact_move_that_arrives_under_load_is_reached(tmp_path: Path) -> None:
    receipt = contact_move(tmp_path, [0.5, 0.49, 0.48, 0.47, 0.46], [5.0, 5.0, 90.0, 90.0, 90.0])
    assert receipt["execution_status"] == "reached" and receipt["failure_code"] is None
