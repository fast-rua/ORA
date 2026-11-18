"""``code_mode.shell.run_command`` without bubblewrap, for the agent-core tests of code mode.

The same ``robot`` library and protocol, run by the host's Python in the command's workspace
(``tests/ora/agent/code_mode/test_sandbox.py`` covers the sandbox itself).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def local_run_command(command: str, *, handle_call: Any, wall_s: float, cpu_s: int, memory_mib: int,
                      workspace: Path, uncounted_s: Any = lambda: 0.0, extended: bool = False,
                      handle_op: Any = None, observation: Any = None, source: bool = False,
                      preview: Any = (), should_stop: Any = None) -> Any:
    """``shell.run_command`` without bubblewrap: the same robot library and protocol, the host's Python."""

    import os
    import socket
    import subprocess
    import sys
    import tempfile

    from ora.agent.code_mode import shell

    library = Path(tempfile.mkdtemp())
    (library / "robot.py").write_text(shell.ROBOT_LIBRARY.read_text())
    host, child = socket.socketpair()
    env = {**os.environ, "ORA_CODE_MODE_FD": str(child.fileno()), "PYTHONPATH": str(library),
           "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
           **({shell.LIBRARY_ENV: "extended"} if extended or handle_op is not None else {})}
    process = subprocess.Popen(["/bin/bash", "-c", command], cwd=workspace, env=env, pass_fds=(child.fileno(),),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    child.close()
    output = bytearray()
    try:
        outcome = shell._serve(host, process, handle_call, output, deadline=__import__("time").monotonic() + wall_s,
                               uncounted_s=uncounted_s, handle_op=handle_op, should_stop=should_stop)
    finally:
        host.close()
        shell._stop(process)
        process.stdout.close()
    outcome.output = output.decode()
    if outcome.exit_code is None and outcome.status in {"completed", "failed"}:
        outcome.exit_code = process.returncode
    return outcome
