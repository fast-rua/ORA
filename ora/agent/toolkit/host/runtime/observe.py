"""Observation state every tool reads.

The current grip-site state, a saved step of this round, the cameras a
surface admits, the public robot facts ``observe`` attaches, the marked
points and the read-only feedback-batch pair. The toolkit serves
``observe`` and ``mark_point`` (``dispatch.install``).

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.composition.surface.view_config import (
    controlled_point_projection_delivery, gripper_command_delivery,
)
from ora.agent.error_classes import PublicFactsError
from ora.agent.toolkit.host.backend_selection import ROBODOJO_RGBD_CAPABILITY, YAM_RGBD_CAPABILITY

from ora.agent.toolkit.host.shared import saved_steps
from ora.agent.toolkit.host.runtime.common import ROBODOJO_POLICY_CAMERAS, YAM_POLICY_CAMERAS
from ora.agent.error_classes import ToolError


class ObserveRuntimeMixin:
    """Observation state every tool reads."""

    def _derived_resolution(self) -> tuple[int, int]:
        configured = self.view_config["derived_resolution"]
        return int(configured[0]), int(configured[1])

    def _current_grip_state(self) -> tuple[dict[str, Any], Path, dict[str, Any], list[float], float]:
        control, root, observation = self._current()
        end_effectors = observation.get("proprioception", {}).get("end_effectors", {})
        state = end_effectors.get("right") or end_effectors.get("left")
        if not isinstance(state, Mapping):
            raise ToolError("current grip-site state is unavailable", code="pose_state_unavailable")
        pose = state.get("pose_xyz_xyzw")
        if not isinstance(pose, list) or len(pose) != 7:
            raise ToolError("current grip-site pose is unavailable", code="pose_state_unavailable")
        return control, root, observation, [float(item) for item in pose], float(state.get("gripper", 0))

    def begin_read_only_feedback_batch(
        self, name: str, calls: Sequence[Mapping[str, Any]],
    ) -> bool:
        """No batch: only a surface that aggregates parallel marks (``batched_mark_overlay_v1``) holds their
        images back, and there the toolkit binds its own pair (``dispatch.install``)."""
        del name, calls
        return False

    def finish_read_only_feedback_batch(
        self, name: str, results: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Nothing to compose: no batch was begun (``begin_read_only_feedback_batch``)."""
        del results
        if name != "mark_point":
            raise ToolError("active read-only feedback batch changed tool identity")
        return {}

    def _attach_observe_public_facts(self, result: dict[str, Any], root: Path,
                                    observation: Mapping[str, Any]) -> dict[str, Any]:
        if not self.surface.supports("current_public_robot_facts_v1"):
            return result
        # The context layer's projection, handed in by whoever builds the
        # runtime (build_tool_runtime's public_facts_projector): the tool layer
        # sits below context and does not import it.
        project = self.public_facts_projector
        if project is None:
            raise ToolError(
                "the surface declares current_public_robot_facts_v1 but the runtime has no "
                "public-facts projector", code="tool_surface_capability_mismatch", retryable=False,
            )
        # Only physical camera panels have pinhole calibration. Point-cloud
        # projections remain explicitly named derived views, not virtual sensors.
        views = [view for view in result["views"] if view in observation["cameras"]]
        try:
            facts = project(
                observation=observation, observation_root=root, view_config=self.view_config,
                selected_views=views, snapshot_role=("historical" if result.get("status")
                    == "actual_historical_observation" else "current"),
                gripper_command_delivered=gripper_command_delivery(self.view_config),
                controlled_point_projection=controlled_point_projection_delivery(self.view_config),
            )
        except PublicFactsError as exc:
            # The Harness's own published robot facts failed to project: an
            # infrastructure code from the table, never retryable by the policy.
            raise ToolError(str(exc), code="embodiment_binding", retryable=False) from exc
        # Measured joint evidence replaces the old hard-coded 80 mm proxy and
        # command-only summary. Neither establishes attachment or grasp success.
        for key in ("grip_xyz_m", "aperture_mm", "aperture_limits_mm", "end_effectors"):
            result.pop(key, None)
        result["public_robot_facts"] = facts
        return result

    def _retention_observation(self, step: Any):
        """Select saved public evidence in the current round; never reset the world."""
        control, _root, _observation, _pose, _gripper = self._current_grip_state()
        states, round_start = saved_steps.round_timeline(self, control)
        refusal = saved_steps.round_step_refusal(step, states, round_start)
        if refusal is not None:
            raise refusal
        _step, state, root, observation = self._select_observation(step)
        effector = observation["proprioception"]["end_effectors"]
        effector = effector.get("right") or effector.get("left")
        selected = {**control, "observation_id": state["observation"]["id"],
                    "observation_sha256": state["observation"]["manifest"]["sha256"]}
        return selected, root, observation, effector["pose_xyz_xyzw"], float(effector.get("gripper", 0))

    def _physical_policy_cameras(self) -> set[str]:
        """The cameras this surface's observe schema admits, for every backend."""
        if self.surface.supports(YAM_RGBD_CAPABILITY):
            return set(YAM_POLICY_CAMERAS)
        return (self._robotwin_policy_cameras() if self.surface.supports("robotwin_rgbd_v1") else set(
            ROBODOJO_POLICY_CAMERAS) if self.surface.supports(ROBODOJO_RGBD_CAPABILITY) else {"agentview", "wrist"})

    def _robotwin_policy_cameras(self) -> set[str]:
        return {"head_camera", "front_camera", "left_camera", "right_camera"}

    def _load_points(self) -> dict[str, Any]:
        if not self.point_path.is_file():
            return {}
        value = json.loads(self.point_path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
