"""Independent toolkit live diagnostic entry; never the qualified policy worker.

Uses the existing RPC wire and session dispatcher. Restoration and explicit
host authorization precede every tool-facing query; execution also requires
its own launch flag. This module has a separate source hash in launch evidence.
"""

import json
import os
from pathlib import Path
import socket
import traceback

from ora.environment.rpc import (
    MAX_MESSAGE_BYTES,
    RPC_SCHEMA,
    PID_COLLISION_FILE,
    PID_SEGMENT_COLLISION_EXIT,
)
from ora.environment.worker import (
    WorkerProtocolError,
    _dispatch,
    _error_envelope,
    _load_config,
    build_parser,
    kit_pid_collision,
)
from .control_replay import _bounded_close
from .session import RobodojoSession
from .toolkit_restore import inspect, restore


class Diagnostic:
    def __init__(self, session, config):
        self.session, self.config = session, config
        self.started = self.restored = self.verified = self.prepared = False
        self.restore_attempted = False
        self.reset = None

    def dispatch(self, method, params):
        if method in {"ping", "close"}:
            if method == "close":
                return None, True  # bounded close belongs to the transport finally.
            return {"ready": True}, False
        if method == "start" and not self.started:
            self.started = True
            self.reset = self.session.start()
            return self.reset, False
        if method == "toolkit_restore" and self.started and not self.restore_attempted:
            self.restore_attempted = True
            report = restore(self.session, self.config, self.reset)
            self.restored = True
            return report, False
        if method == "toolkit_prepare" and self.restored and not self.prepared:
            # This capture is verifier-only; no ordinary observe is granted yet.
            # A failed render remains diagnostic evidence. The host must still
            # verify physical state before authorize can grant ordinary queries.
            self.prepared = True
            return _dispatch(self.session, self.config, "observe", params)
        if method == "toolkit_verify_state" and self.prepared:
            return inspect(self.session), False
        if method == "toolkit_authorize" and self.prepared and not self.verified:
            self.verified = True
            return {"verified": True}, False
        if not self.verified:
            raise WorkerProtocolError(
                "toolkit boundary has not been verified by the host"
            )
        if method == "toolkit_state":
            return inspect(self.session, objects=params.get("objects") is True), False
        allowed = {"observe", "align_plan", "probe_motion", "status", "is_terminal"}
        if self.config.get("toolkit_execute_action") is True:
            allowed |= {"begin_action_sequence", "end_action_sequence", "execute"}
        if method not in allowed:
            raise WorkerProtocolError("method is not a granted toolkit live capability")
        return _dispatch(self.session, self.config, method, params)


def serve(socket_path, config_path):
    config = _load_config(config_path)
    if config.get("backend") != "robodojo" or not config.get("session_id"):
        raise WorkerProtocolError(
            "toolkit live requires an identified RoboDojo session"
        )
    collision = kit_pid_collision(config, pid=os.getpid(), uid=os.getuid())
    if collision:
        (socket_path.parent / PID_COLLISION_FILE).write_text(
            json.dumps(dict(worker_pid=os.getpid(), segment=collision))
        )
        return PID_SEGMENT_COLLISION_EXIT
    output, shared = (
        Path(config["output_root"]).resolve(),
        Path(config["shared_root"]).resolve(),
    )
    if not output.is_relative_to(shared):
        raise WorkerProtocolError("diagnostic output escapes shared root")
    session = RobodojoSession(
        config=config, output_root=output, exit_process_on_close=False
    )
    diagnostic = Diagnostic(session, config)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    mask = os.umask(0o077)
    try:
        server.bind(str(socket_path))
    finally:
        os.umask(mask)
    server.listen(1)
    try:
        connection, _ = server.accept()
        with (
            connection,
            connection.makefile("r", encoding="utf-8") as reader,
            connection.makefile("w", encoding="utf-8") as writer,
        ):
            while True:
                line = reader.readline(MAX_MESSAGE_BYTES + 1)
                if not line:
                    break
                request_id, closing = None, False
                execution = None
                try:
                    if len(line.encode()) > MAX_MESSAGE_BYTES:
                        raise WorkerProtocolError("request exceeds size limit")
                    request = json.loads(line)
                    request_id = request.get("request_id")
                    if (
                        request.get("schema") != RPC_SCHEMA
                        or request.get("session_id") != config["session_id"]
                    ):
                        raise WorkerProtocolError("diagnostic RPC identity mismatch")
                    if not isinstance(request.get("method"), str) or not isinstance(
                        request.get("params"), dict
                    ):
                        raise WorkerProtocolError("invalid RPC request shape")
                    execution = session.execution_request(request.get("execution_state"))
                    with execution:
                        value, closing = diagnostic.dispatch(request["method"], request["params"])
                    response = dict(ok=True, result=value)
                except Exception as error:
                    traceback.print_exc()
                    response = dict(ok=False, error=_error_envelope(error))
                if execution is not None and execution.updates is not None:
                    response["execution_state"] = execution.updates
                writer.write(
                    json.dumps(
                        dict(
                            response,
                            schema=RPC_SCHEMA,
                            session_id=config["session_id"],
                            request_id=request_id,
                        )
                    )
                    + "\n"
                )
                writer.flush()
                if closing:
                    break
    finally:
        _bounded_close(session)
        server.close()
        socket_path.unlink(missing_ok=True)
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    return serve(args.socket.resolve(), args.config.resolve())


if __name__ == "__main__":
    code = main()
    # The Kit process may retain native threads after bounded session close.
    os._exit(code)
