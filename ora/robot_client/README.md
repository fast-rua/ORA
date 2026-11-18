# ora/robot_client

## Responsibility

The robot of a free-play run (`ora play`), for the agent's own scripts and
shell. Inside such a run the agent's environment carries `ORA_ROBOT_SOCKET`,
an `ora-robot` command on `PATH` and this package on `PYTHONPATH`; each call
goes over the run's robot socket (`ora.robot-socket.v1`) to the Supervisor,
which admits it, executes a world-changing call alone and returns its receipt
with the new observation, or says why it refused. Every call is recorded in
the run's agent trace, marked as coming from a script.

| Module | Holds |
|---|---|
| `client.py` | `Robot`: one connection to the socket (`tools()`, `call(name, **arguments)`, `observe()`), `RobotError` |
| `cli.py`, `__main__.py` | `ora-robot tools \| call \| observe \| status` from a shell |

## What belongs here, and what does not

Here: the client side of the robot socket, in the standard library only, so
it runs under any `python3` without ORA installed. Not here: the socket server
and the script calls' delivery (`ora/agent/pi/robot_socket.py`), and their
admission and execution (`ora/orchestrator/episode/script_actions.py`).
Nothing here can bypass the Supervisor or reach the simulator.

## Dependency direction

Imports the standard library only, never another `ora` package. No Python
module imports it to run it: `ora/agent/pi/robot_socket.py` copies the package
by path into the run's `native/robot-client/` directory with an `ora-robot`
shim that runs `python3 -S -m ora.robot_client`.

## Entry points

- `from ora.robot_client import Robot` in a free-play script.
- `ora-robot` (`ora.robot_client.cli:main`; the console script and the copied
  shim) and `python -m ora.robot_client`.

## Documentation

- [Scripting the robot in free play](../../docs/usage/play.md#scripting-the-robot)
- [Public interfaces](../../docs/reference/public-interfaces.md)
- [Run ORA safely](../../docs/usage/safety.md)
