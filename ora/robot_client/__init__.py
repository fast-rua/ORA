"""The robot of an ORA free-play run, for the agent's own scripts (standard library only).

Inside a free-play run (``ora play``) the agent's shell carries
``ORA_ROBOT_SOCKET``, ``ora-robot`` on ``PATH`` and this package on
``PYTHONPATH``::

    from ora.robot_client import Robot

    robot = Robot()
    robot.tools()                                  # the run's ToolSurface: names and schemas
    robot.call("observe", views=["agentview"])     # a read-only tool, answered at once
    robot.call("close_gripper")                    # one world-changing action, through the Supervisor
    robot.observe(save="view.png")                 # the latest published observation

Every ``call`` takes the path the agent's own tool calls take: the run's
Supervisor admits it, executes a world-changing one alone and returns its
receipt with the new observation, or returns why it refused. Each call is
recorded in the run's agent trace, marked as coming from a script. The
package imports nothing outside the standard library, so it runs under any
``python3`` without ORA installed. See ``docs/reference/public-interfaces.md``.
"""

from .client import PROTOCOL, SOCKET_ENV, Robot, RobotError

__all__ = ["PROTOCOL", "SOCKET_ENV", "Robot", "RobotError"]
