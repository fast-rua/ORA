"""RoboDojo's read-only motion bridge behind ``session.align_plan``.

``plan_alignment`` routes three requests through it: an ``align_plan`` batch
(``solve_batch_reachability``) and a reach check (``reach_check_batch``), both
answered by the shared planner's IK (``reachability.solve_batch``,
``reach_check``), and the kinematics query (``kinematics``,
``kinematics_query``). Nothing here writes a controller or advances physics:
every move runs as ``follow_joint_trajectory`` (``trajectory``).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ora.environment.controllers.motion_transaction import MotionTransactionError


class RobodojoMotionBackendAdapter:
    """The read-only answers ``plan_alignment`` asks an ARX X5 session for."""

    arms = ("left", "right")

    def __init__(
        self, *, reachability: Callable[..., Any], reach_check: Callable[..., Any],
        kinematics: Callable[[Mapping[str, Any]], Any],
    ) -> None:
        self._reachability = reachability
        self._reach_check = reach_check
        # The read-only kinematics query (``kinematics_query``); routed by ``plan_alignment``.
        self.kinematics = kinematics

    @classmethod
    def from_session(cls, session: Any) -> "RobodojoMotionBackendAdapter":
        from .kinematics_query import answer as kinematics_answer
        from .reach_check import reach_check_batch
        from .reachability import solve_batch

        return cls(
            reachability=lambda arm, poses, **kwargs: solve_batch(session, arm, poses, **kwargs),
            reach_check=lambda arm, queries, **kwargs: reach_check_batch(session, arm, queries, **kwargs),
            kinematics=lambda request: kinematics_answer(session, request),
        )

    def _require_arm(self, arm: str) -> None:
        if arm not in self.arms:
            raise MotionTransactionError(f"RoboDojo motion adapter has no arm {arm!r}")

    def solve_batch_reachability(self, arm: str, target_poses_xyzw: Sequence[Sequence[float]], **kwargs: Any):
        self._require_arm(arm)
        return self._reachability(arm, target_poses_xyzw, **kwargs)

    def reach_check_batch(self, arm: str, queries: Sequence[Mapping[str, Any]], **kwargs: Any):
        self._require_arm(arm)
        return self._reach_check(arm, queries, **kwargs)


__all__ = ["RobodojoMotionBackendAdapter"]
