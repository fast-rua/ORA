"""The Pi agent core's process and RPC, seen from Python (a ``PiPlanner`` mixin).

Starts the TypeScript core (or accepts the Pi TUI that attaches to a run),
binds the RPC handlers, and restarts and resumes a core that ended: a call
made from the top restarts it once; a call made while serving one of the
core's own requests does not (see ``_call``). Tool executions of one act are
cached by model call and tool call for legacy sessions. Synchronous toolkit
parents instead terminate on loss before delivery and never recover cached results.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from ora.agent.contracts import PolicySessionError

from .context_pack_rpc import context_pack_handlers
from .robot_socket import ScriptAwareRpcPeer
from .rpc import RpcError, RpcPeer
from ora.agent.providers.user_store import credentials_path

ROOT = Path(__file__).resolve().parents[3]
PI_CORE_ENTRY = ROOT / "ora-pi/src/main.ts"
#: Credential-shaped environment variable names; a free-play agent's shell never sees them.
CREDENTIAL_VARIABLE = re.compile(r"(API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY|ACCESS_KEY|AUTH)",
                                 re.IGNORECASE)


def user_credential_store() -> Path:
    """Where /login keeps provider credentials: user-level ORA config, never inside a run.

    ``ora providers`` and the provider selector keep ORA's keys in the same store
    (``ora.agent.providers.user_store``).
    """

    return credentials_path()


def play_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The user's own environment for a free-play agent process, without credentials."""

    return {key: value for key, value in environ.items() if not CREDENTIAL_VARIABLE.search(key)}
# V8 grows the core's heap by up to 4x its live size between full collections;
# 1.5x keeps a long episode's Node process near its live size at no measured
# cost in time (481 Synthetic calls: peak RSS 1,101 -> 530 MB). 0 = V8's own.
HEAP_GROWING_PERCENT = int(os.environ.get("ORA_PI_HEAP_GROWING_PERCENT", "50"))
#: The keys Python hands the core; a parent's own value of these names never passes through.
CORE_KEY_VARIABLES = frozenset({"ORA_PI_API_KEY", "ORA_PI_STANDBY_API_KEYS", "ORA_PI_COMPACTOR_API_KEY"})


class PiCoreProcess:
    """Process, RPC, restart and the per-act execution cache of ``PiPlanner``."""

    def _spawn(self, provider: Mapping[str, Any], *, node: str | None, interactive_socket: str | None,
               extra_env: Mapping[str, str] | None, compactor_provider: Mapping[str, Any] | None = None) -> None:
        self.node = node or shutil.which("node")
        if self.node is None:
            raise PolicySessionError("the Pi agent core needs node >= 22.19 on PATH")
        native = self.run_root / "native"
        native.mkdir(parents=True, exist_ok=True)
        events = self.run_root / "agent-events.jsonl"
        ora_pi = ROOT / "ora-pi"
        self.credential_store = user_credential_store()
        self.permission = [
            "--permission",
            f"--allow-fs-read={ora_pi}",
            f"--allow-fs-read={native}",
            f"--allow-fs-read={events}",
            f"--allow-fs-write={native}",
            f"--allow-fs-write={events}",
            # /login and /logout: the user-level credential store, outside the run.
            f"--allow-fs-read={self.credential_store.parent}",
            f"--allow-fs-write={self.credential_store.parent}",
        ]
        if os.environ.get("ORA_PI_NO_PERMISSION") == "1":
            self.permission = []
        if self.play is not None:
            # Free play runs Pi as the user runs it (their Pi config, the agent's shell with
            # their privileges), from the run's workspace, with every credential variable removed.
            self.permission = []
            env = play_environment(os.environ)
        else:
            env = {key: value for key, value in os.environ.items()
                   if key in {"PATH", "LANG", "LC_ALL", "TERM", "COLORTERM", "TZ", "NODE_OPTIONS",
                              "NODE_EXTRA_CA_CERTS"}
                   or (key.startswith("ORA_PI_") and key not in CORE_KEY_VARIABLES)}
        if "NODE_EXTRA_CA_CERTS" not in env and os.environ.get("SSL_CERT_FILE"):
            # An internal-CA endpoint Python trusts through SSL_CERT_FILE; node reads its CA at startup.
            env["NODE_EXTRA_CA_CERTS"] = os.environ["SSL_CERT_FILE"]
        if self.play is None:
            env.update({"PI_OFFLINE": "1", "HOME": str(native / "pi-agent")})
        api_key = provider.get("api_key")
        if api_key:
            env["ORA_PI_API_KEY"] = str(api_key)
        standby = provider.get("standby_api_keys") or []
        if standby:
            # Credentials never enter the provider spec the core logs; only the environment.
            env["ORA_PI_STANDBY_API_KEYS"] = ",".join(str(key) for key in standby)
        if compactor_provider is not None:  # the bound compactor's own key (policy.context_compactor)
            env["ORA_PI_COMPACTOR_API_KEY"] = str(compactor_provider["api_key"])
        if getattr(self, "code_mode", None):
            # Code-mode scripts (code_mode.py), free play's (``play.code_mode``) included: no client, no shell.
            env.update(self._open_robot_socket("", shell=False))
        elif self.episode_mode == "play":  # the robot socket of a free-play run (robot_socket.py)
            env.update(self._open_robot_socket(env.get("PATH", "")))
        env.update(dict(extra_env or {}))
        self.child_env = env
        self.provider_spec = {key: value for key, value in provider.items()
                              if key not in {"api_key", "standby_api_keys"}}
        self.compactor_provider_spec = None if compactor_provider is None else {
            key: value for key, value in compactor_provider.items() if key != "api_key"}
        # The protocol name the Python provider object would report.
        self.provider_protocol = {
            "fake": "fake", "anthropic_messages": "anthropic_messages",
        }.get(str(provider.get("kind")), "openai_chat_completions")
        self.process = None
        self.server = None
        if interactive_socket is not None:  # a Pi TUI attaches (ora-pi/src/main.ts --attach)
            self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server.bind(interactive_socket)
            self.server.listen(1)
        self.restarts = 0
        self._connect()

    def _connect(self) -> None:
        """Start the TypeScript core (or accept the attaching TUI) and bind the RPC."""

        native = self.run_root / "native"
        if self.server is None:
            parent, child = socket.socketpair()
            with (native / "pi-core.log").open("ab") as log:
                self.process = subprocess.Popen(
                    [self.node, *self.permission, f"--heap-growing-percent={HEAP_GROWING_PERCENT}",
                     str(PI_CORE_ENTRY), "--rpc-fd", str(child.fileno())],
                    pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    env=self.child_env, cwd=str(self.play["workspace"] if self.play is not None else native),
                    start_new_session=True,
                )
            child.close()
            self.rpc = self._rpc_peer(parent)
        else:
            self.server.settimeout(float(os.environ.get("ORA_PI_ATTACH_TIMEOUT_S", "900")))
            try:
                connection, _ = self.server.accept()
            except socket.timeout as exc:
                raise PolicySessionError("no Pi TUI attached to this run in time") from exc
            connection.settimeout(None)
            self.rpc = self._rpc_peer(connection)
        for method, handler in {
            "tools.classify": self._rpc_classify,
            "tools.invoke": self._rpc_invoke,
            "tools.settled": self._rpc_tools_settled,
            "tools.project": self._rpc_project,
            "tools.read_only_batch": self._rpc_read_only_batch,
            "tools.read_only_batch_run": self._rpc_read_only_batch_run,
            "operator.submit": self._rpc_operator_submit,
            "operator.poll": self._rpc_operator_poll,
            "operator.stop": self._rpc_operator_stop,
            **context_pack_handlers(self),  # a plan's policy.context_pack only
        }.items():
            self.rpc.on(method, self._serving_handler(handler))

    def _rpc_peer(self, sock: socket.socket) -> RpcPeer:
        """The RPC peer; a free-play run's also serves its robot socket while ``act`` waits."""

        return RpcPeer(sock) if self.robot is None else ScriptAwareRpcPeer(sock, self.robot)

    def _serving_handler(self, handler: Any) -> Any:
        def serve(params: Mapping[str, Any]) -> Any:
            self._serving += 1
            try:
                return handler(params)
            finally:
                self._serving -= 1
        return serve

    def _call(self, method: str, params: Mapping[str, Any]) -> Any:
        """One RPC; a lost agent process is restarted (or re-attached) and resumed once.

        Only a call made from the top (the Supervisor's own thread) restarts.
        A call made while serving a request of the core (an operator message
        delivered from ``operator.poll``) loses the request it serves with
        the core: the request's outer call then fails with a clear fault
        instead of resuming an act whose nested record may or may not have
        landed. Tools never call back into the core (``_remote_event``).
        """

        try:
            result = self.rpc.call(method, params)
            if method == "act":
                getattr(self, "_pending_tool_deliveries", {}).clear()
            return result
        except (RpcError, OSError) as exc:
            if self._interrupt_pending_tool():
                raise PolicySessionError("agent core interrupted during current tool delivery") from exc
            if isinstance(exc, RpcError) and exc.code != "closed":
                raise
            if self._serving:
                self._lost_while_serving = self._lost_while_serving or method
                raise PolicySessionError(f"the agent core ended during the nested {method} call") from exc
            lost, self._lost_while_serving = self._lost_while_serving, None
            if lost is not None:
                raise PolicySessionError(
                    f"the agent core ended while Python was serving it, inside a nested {lost} call; "
                    f"whether that call was recorded is unknown, so the {method} is not resumed"
                ) from exc
            # A free-play agent process is not resumed: Pi's own conversation lives in that process.
            limit = 0 if self.play is not None else int(os.environ.get("ORA_PI_MAX_RESTARTS", "3"))
            if self.closed or self.restarts >= limit:
                raise
        self.restarts += 1
        self._resume_record({"restart": self.restarts, "method": method, "unix_s": time.time()})
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
        self._connect()
        self.rpc.call("init", {**self.init_params, "resume": True})
        resumed = dict(params)
        if method == "act":
            resumed["resume"] = True
        if getattr(self, "tool_action_executor", None) is not None:
            # A second loss during a resumed synchronous parent is fatal too.
            return self._call(method, resumed)
        return self.rpc.call(method, resumed)

    def _resume_record(self, record: Mapping[str, Any]) -> None:
        """Restarts and cache answers, beside the trace (native/pi-restarts.jsonl)."""

        with (self.run_root / "native" / "pi-restarts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(record), sort_keys=True) + "\n")

    def _once(self, key: tuple[str, ...], run: Any) -> Any:
        """Run a tool execution once per act; a replayed request gets the first result."""

        if key in self._executed:
            if key[0] == "tools.invoke" and getattr(self, "tool_action_executor", None) is not None:
                self.tool_action_executor.interrupt({"model_call_id": key[1], "tool_call_id": key[2]})
                raise PolicySessionError("interrupted tool result recovery is disabled")
            self._resume_record({"cached": list(key), "unix_s": time.time()})
            return copy.deepcopy(self._executed[key])
        value = run()
        self._executed[key] = copy.deepcopy(value)
        return value
