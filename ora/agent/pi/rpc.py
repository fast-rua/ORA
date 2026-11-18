"""JSON-lines RPC with the TypeScript agent core (``ora-pi/src/rpc.ts``).

The Supervisor thread drives the peer synchronously: :meth:`RpcPeer.call`
writes one request and then serves every request the TypeScript side sends
(tool calls, observation builds) until its own response arrives. Other threads
may only :meth:`notify` (the SIGTERM handler uses it to interrupt a turn).
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Any, Callable, Mapping


class RpcError(RuntimeError):
    def __init__(self, message: str, code: str = "rpc_error", data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


Handler = Callable[[Mapping[str, Any]], Any]


class RpcPeer:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.reader = sock.makefile("rb")
        self.write_lock = threading.Lock()
        self.handlers: dict[str, Handler] = {}
        self.next_id = 1
        self.closed = False

    def on(self, method: str, handler: Handler) -> None:
        self.handlers[method] = handler

    def _write(self, message: Mapping[str, Any]) -> None:
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        with self.write_lock:
            if self.closed:
                raise RpcError("RPC peer is closed", "closed")
            self.sock.sendall(data)

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        self._write({"method": method, "params": dict(params or {})})

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        request_id = f"p{self.next_id}"
        self.next_id += 1
        self._write({"id": request_id, "method": method, "params": dict(params or {})})
        while True:
            message = self._read()
            if isinstance(message.get("method"), str):
                self._serve(message)
                continue
            if message.get("id") != request_id:
                raise RpcError(f"unexpected RPC response {message.get('id')!r}", "protocol")
            error = message.get("error")
            if error is not None:
                raise RpcError(str(error.get("message")), str(error.get("code")), error.get("data"))
            return message.get("result")

    def _read(self) -> dict[str, Any]:
        line = self.reader.readline()
        if not line:
            self.closed = True
            raise RpcError("TypeScript agent core closed the RPC channel", "closed")
        return json.loads(line.decode("utf-8"))

    def _serve(self, message: Mapping[str, Any]) -> None:
        request_id = message.get("id")
        handler = self.handlers.get(str(message["method"]))
        try:
            if handler is None:
                raise RpcError(f"unknown method {message['method']}", "method_not_found")
            result = handler(message.get("params") or {})
        except Exception as exc:  # the remote side decides what a failure means
            if request_id is None:
                raise
            self._write({"id": request_id, "error": {
                "message": str(exc), "code": getattr(exc, "code", type(exc).__name__),
                "data": getattr(exc, "data", None),
            }})
            return
        if request_id is not None:
            self._write({"id": request_id, "result": result})

    def close(self) -> None:
        with self.write_lock:
            self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.reader.close()
        self.sock.close()
