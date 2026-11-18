"""A fake data interface for the motion tools' dry runs: toy kinematics and the actions it was asked to run.

Joints 0-2 of the toy arm are the link6 position, so an IK answer is the target position and an FK answer the joints'
first three numbers; a screen is clear unless the caller queues a verdict. The interface records every action it is
asked to execute, which a dry run must leave empty.
"""
import json
from types import SimpleNamespace

from ora.agent.toolkit.interface.types import Response
from ora.agent.toolkit.tools.move_pose import _smooth
from tests.support.agent.paced_kinematics import paced

START = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
RIGHT = [0.3, -0.4, 0.2, 0.0, 0.0, 0.0, 1.0]
GOAL = [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
CONFIG = {"synchronous_execution": True, "motion": {"joint_trajectory": True, "capabilities": [], "examples": []}}


def response(data, method="kinematics"):
    return Response(json.dumps({"status": "ok", "method": method, "data": data}).encode())


class Fixture:
    """``api`` is the data interface a tool gets; ``executed`` the actions it executed, ``queries`` the kinematics ops."""

    def __init__(self, config=None, screens=(), planned=None):
        self.executed, self.screens, self.queries = [], list(screens), []
        self.config = CONFIG if config is None else config

        def kinematics(*, request):
            self.queries.append(request["operation"])
            if request["operation"] == "screen_path":
                count = len(next(iter(request["knots_rad"].values())))
                return response(self.screens.pop(0) if self.screens else {
                    "status": "clear", "arm": None, "first_blocked_segment": None, "segments_screened": count - 1,
                    "min_inter_arm_clearance_m": 0.125})
            if request["operation"] == "fk":
                return response({"rows": [{"index": i, "pose_xyz_xyzw": [*q[:3], 0, 0, 0, 1]}
                                          for i, q in enumerate(request["joints_rad"])]})
            if request["operation"] == "pace_path":
                return response(paced(request))
            if request["operation"] == "plan_path":
                return response(planned() if callable(planned) else planned)
            return response({"rows": [{"index": i, "status": "ok", "joints_rad": [*pose[:3], 0, 0, 0]}
                                      for i, pose in enumerate(request["poses_xyz_xyzw"])]})

        def execute_action(action, budget):
            self.executed.append((action, budget))
            knots = next(iter(action["arguments"]["arms"].values()))["knots_rad"]
            status = "not_reached" if "part_of_move" in action["arguments"] else "reached"
            trajectory = {"status": "completed", "knots_planned": len(knots), "knots_executed": len(knots), "stop": None}
            return response({"remaining_budget": 50 - budget, "receipt": {
                "status": status, "native_actions": len(knots), "stage_evidence": {"trajectory": trajectory}}},
                "execute_action")

        self.state = {}

        def read(key):
            if key not in self.state:
                raise KeyError(key)
            return json.loads(self.state[key])

        def write(key, value):
            self.state[key] = json.dumps(value, allow_nan=False)

        self.api = SimpleNamespace(artifacts=SimpleNamespace(tool_context=lambda: self.config),
                                   planner=SimpleNamespace(kinematics=kinematics),
                                   act=SimpleNamespace(execute_action=execute_action),
                                   state=SimpleNamespace(read=read, write=write))


def compiled(goal, arms=("left",)):
    """The (action, compilation) ``compile_action`` returns for a free-space move of each arm to ``goal``."""
    targets = {arm: {"position_m": goal[:3], "orientation_xyzw": goal[3:], "controlled_point": "tcp"} for arm in arms}
    return ({"action_id": "action-7", "arguments": {"targets": targets, "interactions": {arm: "free_space" for arm in arms}}},
            {"targets": {arm: {"controlled_point": "end_effector"} for arm in arms}})


def measured(monkeypatch, goal=GOAL):
    """The measured start state of both arms, and ``compile_action`` returning the move to ``goal`` (move_pose's tool
    module); returns the observation."""
    obs = SimpleNamespace(
        manifest=SimpleNamespace(json=lambda: {"proprioception": {"end_effectors": {
            arm: {"pose_xyz_xyzw": pose, "arm_joint_qpos": [*pose[:3], 0, 0, 0]}
            for arm, pose in (("left", START), ("right", RIGHT))}}}),
        identity=SimpleNamespace(json=lambda: {"observation_id": "o", "observation_sha256": "s"}))
    monkeypatch.setattr(_smooth.observation, "observe", lambda api: obs)
    monkeypatch.setattr("ora.agent.toolkit.tools.move_pose.tool.compile_action",
                        lambda api, arguments, **kwargs: compiled(goal, tuple(arguments.get("arms") or ("left",))))
    return obs
