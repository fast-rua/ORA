"""One method surface for live adapters and evidence-only simulator replay."""
from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

from .types import Response
from .tool_data import ToolDataMethods


if TYPE_CHECKING:
    from .perception.live import Perception
    from .state_access.live import State
    from .sim_query.live import Planner
    from .sim_exec.live import Actions
    from .lifecycle.live import Episode
    from .artifacts.live import Artifacts
    from .compute.live import Compute
    from .context.live import Context

EXTERNAL = frozenset({"segment", "segment_points", "ground", "grasp_candidates"})
EFFECTS = frozenset({"execute_action", "submit_action", "report_issue", "finish_episode"})
SIMULATOR = frozenset({"observation", "proprioception", "receipt", "probe_motion",
                       "align_plan", "reach_check", "kinematics", "check_task"})
METHODS = EXTERNAL | EFFECTS | SIMULATOR


class DataInterface(Protocol):
    """The classified tool-facing API; robot instances have no privileged category."""
    perception: Perception
    state: State
    planner: Planner
    act: Actions
    episode: Episode
    artifacts: Artifacts
    compute: Compute
    context: Context


class Methods(ToolDataMethods):
    """Explicit signatures expand defaults before the shared request encoder."""

    def _invoke(self, method: str, parameters: dict, inputs: dict | None = None) -> Response:
        raise NotImplementedError

    def read_blob(self, ref: dict) -> bytes:
        """Read already granted public bytes; no discovery, registration or cursor step."""
        raise NotImplementedError

    def save_array(self, value: object) -> dict:
        """Persist caller-derived numeric input in the output CAS; no simulator query."""
        raise NotImplementedError

    def observation(self, *, observation: dict | None = None) -> Response:
        return self._invoke("observation", {"observation": observation})

    def proprioception(self, *, observation: dict | None = None) -> Response:
        return self._invoke("proprioception", {"observation": observation})

    def receipt(self, *, action_id: str) -> Response:
        return self._invoke("receipt", {"action_id": action_id})

    def probe_motion(self, *, request: dict) -> Response:
        return self._invoke("probe_motion", {"request": request})

    def align_plan(self, *, request: dict) -> Response:
        return self._invoke("align_plan", {"request": request})

    def reach_check(self, *, request: dict) -> Response:
        return self._invoke("reach_check", {"request": request})

    def kinematics(self, *, request: dict) -> Response:
        """Read-only ``ora.kinematics-request.v1``: IK chains, FK and joint-path screens of the backend's model."""
        return self._invoke("kinematics", {"request": request})

    def check_task(self) -> Response:
        return self._invoke("check_task", {})

    def segment(self, *, rgb: dict, depth: dict, point_uv: tuple[int, int]) -> Response:
        return self._invoke("segment", {"point_uv": list(point_uv)}, {"rgb": rgb, "depth": depth})

    def segment_points(self, *, rgb: dict, depth: dict, points_uv: tuple[tuple[int, int], ...]) -> Response:
        return self._invoke("segment_points", {"points_uv": [list(point) for point in points_uv]},
                            {"rgb": rgb, "depth": depth})

    def ground(self, *, rgb: dict, text: str, max_candidates: int = 3) -> Response:
        return self._invoke("ground", dict(text=text, max_candidates=max_candidates), {"rgb": rgb})

    def grasp_candidates(self, *, scene: dict, target: dict, camera_to_world: list,
                         workspace_bounds_m: dict, max_candidates: int = 32,
                         forward_passes: int = 1) -> Response:
        return self._invoke("grasp_candidates", dict(camera_to_world=camera_to_world,
                            workspace_bounds_m=workspace_bounds_m, max_candidates=max_candidates,
                            forward_passes=forward_passes), {"scene": scene, "target": target})

    def execute_action(self, *, action: dict, budget: int) -> Response:
        return self._invoke("execute_action", {"action": action, "budget": budget})

    def submit_action(self, *, action: dict, preparation: dict | None = None) -> Response:
        return self._invoke("submit_action", {"action": action}, preparation)

    def report_issue(self, *, issue: dict) -> Response:
        return self._invoke("report_issue", {"issue": issue})

    def finish_episode(self, *, request: dict) -> Response:
        return self._invoke("finish_episode", {"request": request})

    @property
    def perception(self):
        from .perception.live import Perception
        return Perception(self)

    @property
    def planner(self):
        from .sim_query.live import Planner
        return Planner(self)

    @property
    def act(self):
        from .sim_exec.live import Actions
        return Actions(self)

    @property
    def episode(self):
        from .lifecycle.live import Episode
        return Episode(self)

    @property
    def artifacts(self):
        from .artifacts.live import Artifacts
        return Artifacts(self)

    @property
    def compute(self):
        from .compute.live import Compute
        return Compute(self)

    @property
    def state(self):
        from .state_access.live import State
        return State(self, getattr(self, "state_access", None), self.context.json()["check_task"] or getattr(self, "allow_check_task", False))
