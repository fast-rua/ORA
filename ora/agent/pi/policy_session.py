"""The Python half of the Pi agent core: the policy session the Supervisor drives.

``PiPlanner`` is the policy delegate of a
``DelegatingPolicySession``. It keeps everything that knows robots: the
observation envelope (built from the policy-state snapshot and the last
receipt), the static prompt composition, tool execution through the existing
ToolRuntime, the result projection and delivery rules (sent as generic hints),
and the authority over submitted actions. The TypeScript process
(``ora-pi/src/main.ts``) owns the conversation, the provider loop, context
management and the agent-side evidence; it never reads Supervisor or worker
files.

One RPC exchange per observation: ``act`` carries the observation draft and
returns how the turn ended. While ``act`` is pending, the TypeScript side calls
back into ``tools.*`` on this thread.

The process and RPC (``core_process``), the observation drafts
(``observation_draft``) and the tool RPC (``tool_rpc``) are mixins; this module
keeps the session's lifecycle, turns, runtime events and operator control.
"""

from __future__ import annotations

import copy
import os
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.evidence.image_binding import PostActionObservationError, resolve_post_action_images
from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.memory.markers import LOSSLESS_CONTEXT_RETENTION
from ora.agent.session_core.actual_snapshot import NativeActualSnapshotMixin
from ora.agent.session_core.constants import COMPOSITION_CONTEXT_CAPABILITIES, NATIVE_CAUSAL_SLOTS
from ora.agent.session_core.gateway import NativeToolGateway
from ora.agent.session_core.domain import PolicyDomainMixin
from ora.agent.error_classes import ARGUMENT_REFUSAL_CODES
from ora.agent.error_classes import INFRASTRUCTURE_CODES
from ora.agent.evidence.trace_privacy import SCRIPT_TOOLS
from ora.agent.composition.surface.tool_names import DOTTED_TOOL_NAMES

from .core_process import PiCoreProcess
from .domain import context_domain, context_stage_schemas
from .observation_draft import PiObservationDrafts
from .code_mode import PiCodeMode
from .robot_socket import PiRobotSocket
from .rpc import RpcError
from .tool_rpc import PiToolRpc
from .catalog_view import ReadOnlyContextCatalog

EVENT_SCHEMA = os.environ.get("ORA_PI_EVENT_SCHEMA", "ora.agent-event.v1")


class PiPlanner(PiCoreProcess, PiObservationDrafts, PiToolRpc, PiRobotSocket, PiCodeMode, NativeActualSnapshotMixin):
    """Observation-scoped policy delegate whose agent core runs in TypeScript."""

    # Domain helpers of every native_api session (ora.agent.session_core.domain).
    _observation_step = staticmethod(PolicyDomainMixin._observation_step)
    _post_action_view_feedback = staticmethod(PolicyDomainMixin._post_action_view_feedback)
    _receipt_record = staticmethod(PolicyDomainMixin._receipt_record)
    _pose_residual = staticmethod(PolicyDomainMixin._pose_residual)
    _minimal_action_effect = PolicyDomainMixin._minimal_action_effect
    _skill_observation_blocks = PolicyDomainMixin._skill_observation_blocks
    _delivery_trim_rules = PolicyDomainMixin._delivery_trim_rules
    _delivery_trim = PolicyDomainMixin._delivery_trim
    _causal_tool_result = PolicyDomainMixin._causal_tool_result
    _current_observation_id = PolicyDomainMixin._current_observation_id
    _turn_instruction = PolicyDomainMixin._turn_instruction
    _legacy_observation_message = PolicyDomainMixin._observation_message
    _rewrite_causal_observation_message = PolicyDomainMixin._rewrite_causal_observation_message

    def __init__(
        self, *, run_root: Path, run_id: str, tools: Any, policy: Mapping[str, Any],
        provider: Mapping[str, Any], project_instructions_path: Path,
        episode_prefix_path: Path, harness_system_path: Path | None, task_id: str,
        node: str | None = None, interactive_socket: str | None = None,
        extra_env: Mapping[str, str] | None = None, episode_mode: str | None = None,
        play: Mapping[str, Any] | None = None, compactor_provider: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_root = Path(run_root).resolve()
        self.episode_mode = episode_mode
        self.run_id = run_id
        self.tools = tools
        self.policy = dict(policy)
        self.task_id = task_id
        self.context_composition = policy.get("context_composition")
        self.context_retention = policy.get("context_retention", "rewrite_compact_v1")
        self.supports_images = bool(policy["supports_images"])
        self.initial_observation = policy.get("initial_observation", "automatic")
        self.policy_session_id = f"native-policy-{uuid.uuid4().hex}"
        self.project_template = Path(project_instructions_path).read_text(encoding="utf-8").strip()
        self.episode_prefix_template = Path(episode_prefix_path).read_text(encoding="utf-8").strip()
        self.harness_system_template = (
            Path(harness_system_path).read_text(encoding="utf-8").strip()
            if harness_system_path is not None else None
        )
        self.artifacts = ArtifactStore(self.run_root)
        self.gateway = NativeToolGateway(tools)
        # Engine state the observation envelope reads (Python-owned domain state).
        self.pending_receipt: dict[str, Any] | None = None
        self.pending_receipt_body: dict[str, Any] | None = None
        self.last_actual_snapshot: dict[str, Any] | None = None
        self.actual_sequence = -1  # the first observation is sequence 0
        self.last_action_event_id: str | None = None
        self.last_action_step: int | None = None
        self.last_action_id: str | None = None
        self.last_action_public_name: str | None = None
        self.last_fresh_observation_event_id: str | None = None
        self.bound_task_instruction: str | None = None
        self.submitted_action: dict[str, Any] | None = None
        # Tool executions of the current act by (method, model call, tool calls): a
        # restarted core replays its last reply, and a call that already ran here
        # is answered from this cache instead of running twice.
        self._executed: dict[tuple[str, ...], Any] = {}
        # Events a tool records while tools.invoke runs it, returned with its result.
        self._tool_events: list[dict[str, Any]] | None = None
        # Requests of the core being served here, and a nested call that lost the core.
        self._serving = 0
        self._lost_while_serving: str | None = None
        self.closed = False
        self.interrupted = False
        self._operator_decision_callback = None
        self._operator_submitter = None
        # A free-play run (plan episode_mode "play"): Pi's own agent in the run's workspace.
        self.play = dict(play) if play is not None else None
        # Code mode (plan policy.code_mode, code_mode.py): scripts reach the robot through the socket's path.
        self.code_mode = dict(policy["code_mode"]) if policy.get("code_mode") else None
        self._spawn(provider, node=node, interactive_socket=interactive_socket, extra_env=extra_env,
                    compactor_provider=compactor_provider)
        bind_context_artifacts = getattr(self.tools, "bind_context_artifacts", None)
        if self.context_retention == LOSSLESS_CONTEXT_RETENTION and callable(bind_context_artifacts):
            bind_context_artifacts(
                ReadOnlyContextCatalog(self.run_root, self.policy_session_id, self.artifacts),
                event_sink=self._remote_event,
            )
        self.started = False
        self.start()

    def start(self) -> None:
        if self.started:
            return
        surface = getattr(self.tools, "surface", None)
        definitions = self.gateway.definitions()
        delivered = self.gateway.delivered() if callable(getattr(self.tools, "definitions", None)) else []
        payload = {
            "provider_protocol": self.provider_protocol,
            "model": self.policy["model"],
            "reasoning_effort": self.policy["reasoning_effort"],
            "context_profile": self.policy["context_profile"],
            "context_retention": self.context_retention,
            "context_management": self.policy.get("context_management"),
            "max_inline_images_per_request": self.policy.get("max_inline_images_per_request"),
            "context_window_tokens": self.policy["context_window_tokens"],
            "max_output_tokens": self.policy["max_output_tokens"],
            "stream": self.policy["stream"],
            "supports_images": self.supports_images,
            "initial_observation": self.initial_observation,
            "automatic_observation_image": self.supports_images and self.initial_observation == "automatic",
            "summary_policy": self.policy["summary_policy"],
            "context_composition": self.context_composition,
        }
        if self.context_composition in COMPOSITION_CONTEXT_CAPABILITIES:
            payload["context_capabilities"] = list(COMPOSITION_CONTEXT_CAPABILITIES[self.context_composition])
        self.init_params = {
            **({"play": self.play} if self.play is not None else {}),
            # /login keeps provider credentials here, outside the run (any mode).
            "credentials": {"auth_path": str(self.credential_store)},
            "run_root": str(self.run_root),
            "run_id": self.run_id,
            "policy_session_id": self.policy_session_id,
            "event_schema": EVENT_SCHEMA,
            "session_started_payload": payload,
            "policy": self.policy,
            "provider": self.provider_spec,
            # A compactor bound to its own profile (policy.context_compactor); absent otherwise.
            **({"compactor_provider": self.compactor_provider_spec}
               if self.compactor_provider_spec is not None else {}),
            "tools": {
                "definitions": list(definitions),
                "delivered": list(delivered),
                "mutation_tools": list(getattr(surface, "mutation_tools", ("robot.submit_action",))),
                "lifecycle_tools": list(getattr(surface, "lifecycle_tools", ())),
                "capabilities": list(getattr(surface, "capabilities", ()) or ()),
                "typed_effects": self.context_composition in NATIVE_CAUSAL_SLOTS,
                "null_as_omitted": "optional_argument_null_as_omitted_v1" in tuple(
                    getattr(surface, "capabilities", ()) or ()),
            },
            "domain": {
                "argument_refusal_codes": sorted(ARGUMENT_REFUSAL_CODES),
                "infrastructure_codes": sorted(INFRASTRUCTURE_CODES),
                "turn_instruction": self._code_mode_turn_instruction() or self._turn_instruction(),
                "terminal_refusals": self._code_mode_terminal_refusals(),
                "composition_context_capabilities": list(
                    COMPOSITION_CONTEXT_CAPABILITIES.get(self.context_composition, ())),
                "dotted_tool_names": dict(DOTTED_TOOL_NAMES),
                "context": context_domain(
                    mutation_tools=list(getattr(surface, "mutation_tools", ("robot.submit_action",))),
                    script_tools=SCRIPT_TOOLS if self.code_mode else (),
                ),
                "artifact_reader_name": getattr(self.tools, "artifact_read_tool_name", "artifact_read"),
                # Each context stage's parameter schema: the core validates the stages it builds with them.
                "context_stage_schemas": context_stage_schemas(),
            },
        }
        self.rpc.call("init", self.init_params)
        self.started = True

    def _remote_event(self, event_type: str, **fields: Any) -> dict[str, Any] | None:
        """Tools (the artifact reader) append agent events through the TS recorder.

        While tools.invoke runs a tool, its events travel back with the result
        and the core appends them before the tool_result (where the Native
        session appends them), so a core that dies under a running tool never
        leaves a half-recorded event: the restarted core reproduces them from
        the cached execution. Such an event's id is not known yet (None).
        """

        if self._tool_events is not None:
            self._tool_events.append({"type": event_type, "fields": fields})
            return None
        return self._call("record", {"type": event_type, "fields": fields})

    # -- Turns ---------------------------------------------------------------------

    def _turn_inputs(self, *, task_instruction: str, context: Mapping[str, Any], image_path: Path,
                     terminal_lifecycle: bool, image_paths: Sequence[Path],
                     turn_budget: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.closed or self.interrupted:
            raise PolicySessionError("native policy session is not active")
        if image_paths:
            try:
                verified = resolve_post_action_images(context, root=self.run_root, path_key="run_path")
            except PostActionObservationError as exc:
                raise PolicySessionError(str(exc)) from exc
            if tuple(item["path"].resolve() for item in verified) != tuple(Path(p).resolve() for p in image_paths):
                raise PolicySessionError("policy image vector differs from action-bound fresh views")
        static = self._static_context(task_instruction)
        step = self._observation_step(context)
        draft = self._observation_draft(
            task_instruction=task_instruction, context=context, image_path=image_path,
            terminal_lifecycle=terminal_lifecycle,
        )
        inputs = {
            "static": static, "step": step, "draft": draft,
            "receipt_event_id": (self.pending_receipt or {}).get("event_id"),
            "last_fresh_observation_event_id": self.last_fresh_observation_event_id,
            "remaining_budget": self._enforced_budget(context, turn_budget),
            "terminal_lifecycle": terminal_lifecycle,
        }
        self.pending_receipt = None
        self.pending_receipt_body = None
        return inputs

    @staticmethod
    def _enforced_budget(context: Mapping[str, Any], turn_budget: Mapping[str, Any] | None) -> dict[str, Any]:
        """The budget the core enforces in this turn (``remaining_budget`` of ``act``).

        The observation's budget was published with it; on a further turn on the
        same observation the turn's own ``model_calls`` replaces it, so the core
        refuses every model call past the round's budget. The observation the
        model sees keeps the published budget.
        """

        remaining = dict(context.get("remaining_budget", {}))
        calls = (turn_budget or {}).get("model_calls")
        if isinstance(calls, int) and not isinstance(calls, bool):
            remaining["model_calls"] = calls
        return remaining

    def run_observation(self, *, task_instruction: str, context: Mapping[str, Any], image_path: Path,
                        allow_reminder: bool = True, terminal_lifecycle: bool = False,
                        image_paths: tuple[Path, ...] = (),
                        turn_budget: Mapping[str, Any] | None = None) -> dict[str, Any]:
        inputs = self._turn_inputs(
            task_instruction=task_instruction, context=context, image_path=image_path,
            terminal_lifecycle=terminal_lifecycle, image_paths=image_paths, turn_budget=turn_budget,
        )
        self.submitted_action = None
        self._executed.clear()
        self._code_mode_new_act(inputs.get("remaining_budget") or {})
        self.inflight_model_calls = 0
        try:
            outcome = self._call("act", {
                **inputs, "allow_reminder": allow_reminder,
                "operator_poll": self._operator_decision_callback is not None,
            })
        finally:
            # The Supervisor settles outcome.attempts after act returns. Keeping
            # this count beyond that boundary would charge those calls twice.
            self.inflight_model_calls = 0
        if outcome.get("actions_executed"):
            execution = getattr(getattr(self, "tool_action_executor", None), "completed", None)
            if execution is None or execution != outcome.get("execution"):
                raise PolicySessionError("agent core execution differs from the Supervisor ledger")
            return {"actions_executed": True, "execution": execution, "attempts": outcome["attempts"]}
        if outcome.get("submitted"):
            if self.submitted_action is None:
                raise PolicySessionError("the agent core reported a submission no tool persisted")
            action = self.submitted_action
            self.last_action_event_id = outcome["action_event_id"]
            self.last_action_step = inputs["step"]
            self.last_action_id = str(action.get("action_id"))
            self.last_action_public_name = outcome.get("tool_name")
            return {"submitted": True, "action": action, "attempts": outcome["attempts"]}
        if outcome.get("termination") == "operator_interrupt":
            self.interrupted = True
        result = {"submitted": False, "termination": outcome["termination"], "attempts": outcome["attempts"]}
        if outcome.get("answers"):
            # Free play: the joined messages this text-only reply answers (ora-pi/src/play/driver.ts).
            result["answers"] = [str(item) for item in outcome["answers"]]
        return result

    act = run_observation

    def preserve_observation_without_policy_turn(
        self, *, task_instruction: str, context: Mapping[str, Any], image_path: Path,
        terminal_lifecycle: bool = True, allow_reminder: bool = False, image_paths: tuple[Path, ...] = (),
        turn_budget: Mapping[str, Any] | None = None,
    ) -> None:
        del allow_reminder, image_paths, turn_budget
        inputs = self._turn_inputs(
            task_instruction=task_instruction, context=context, image_path=image_path,
            terminal_lifecycle=terminal_lifecycle, image_paths=(),
        )
        self._executed.clear()
        self._call("preserve_observation", inputs)

    # -- Runtime events ------------------------------------------------------------

    def record_environment_receipt(self, receipt: Mapping[str, Any]) -> None:
        ref = self.artifacts.put_json(receipt, kind="environment_receipt")
        nested = receipt.get("receipt")
        nested_action = nested.get("action") if isinstance(nested, Mapping) else None
        action_ids = {
            value for value in (
                receipt.get("action_id"),
                nested.get("action_id") if isinstance(nested, Mapping) else None,
                nested_action.get("action_id") if isinstance(nested_action, Mapping) else None,
            ) if isinstance(value, str) and value
        }
        if len(action_ids) > 1:
            raise PolicySessionError("environment receipt contains conflicting action identities")
        action_id = next(iter(action_ids), None)
        if isinstance(self.last_action_id, str) and self.last_action_id and action_id != self.last_action_id:
            raise PolicySessionError("environment receipt does not match the submitted action")
        event = self._call("record", {"type": "environment_receipt", "fields": {
            "causal_parent_id": self.last_action_event_id, "visibility": "exact",
            "observation_step": self.last_action_step,
            "payload": {
                "receipt_id": receipt.get("receipt_id") or receipt.get("id")
                or f"receipt-sha256-{ref['sha256'][:16]}",
                "action_id": action_id,
            },
            "artifact_ref": ref,
        }})
        self.pending_receipt = {"receipt_ref": ref, "event_id": event["event_id"], "action_id": action_id}
        self.pending_receipt_body = copy.deepcopy(dict(receipt))

    def record_action_discarded(self, record: Mapping[str, Any]) -> None:
        """The Supervisor ended the round on a budget before executing the submitted action."""

        action_id = record.get("action_id")
        if not isinstance(self.last_action_event_id, str) or action_id != self.last_action_id:
            raise PolicySessionError("the discarded action is not the pending submitted action")
        self._call("record", {"type": "action_discarded", "fields": {
            "causal_parent_id": self.last_action_event_id, "visibility": "derived",
            "observation_step": self.last_action_step,
            "payload": {"action_id": action_id, "reason": record.get("reason")},
        }})
        self.last_action_event_id = None
        self.last_action_step = None
        self.last_action_id = None

    def record_fresh_observation(self, context: Mapping[str, Any]) -> None:
        state = context.get("policy_state")
        step = state.get("step") if isinstance(state, Mapping) else None
        event = self._call("record", {"type": "fresh_observation_published", "fields": {
            "observation_step": step if isinstance(step, int) else None,
            "payload": {"observation_id": self._current_observation_id(), "state_ref": state,
                        "after_receipt": self.pending_receipt},
        }})
        self.last_fresh_observation_event_id = str(event["event_id"])
        self._robot_observed(context)

    def record_official_verdict(self, verdict: Mapping[str, Any] | None) -> None:
        ref = None if verdict is None else self.artifacts.put_json(verdict, kind="official_verdict")
        self._call("official_verdict", {
            "artifact_ref": ref,
            "official_success": None if verdict is None else verdict.get("official_success"),
            "has_verdict": verdict is not None,
        })

    def record_run_termination(self, result: Mapping[str, Any]) -> None:
        self._call("record", {"type": "run_terminated", "fields": {
            "visibility": "derived",
            "payload": {key: result.get(key) for key in ("state", "termination", "decisions", "model_calls")},
        }})

    def bind_operator_decision_callback(self, callback: Any) -> None:
        if not callable(callback) or self._operator_decision_callback is not None:
            raise PolicySessionError("Native operator callback must be bound once")
        self._operator_decision_callback = callback

    def record_operator_message(self, payload: Mapping[str, Any]) -> None:
        self._call("operator_message", {"payload": dict(payload)})

    def bind_operator_submitter(self, submit: Any) -> None:
        """The Supervisor's operator inbox, for steers typed into the Pi TUI (bound once)."""

        if not callable(submit) or self._operator_submitter is not None:
            raise PolicySessionError("the operator submitter must be bound once")
        self._operator_submitter = submit

    def _rpc_operator_submit(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Operator input typed into the Pi TUI enters the run's own plan-bound operator inbox.

        A steer by default; a free-play session's terminal also sends instructions
        and its controls (stop the instruction, reset the scene, quit). The inbox
        admits only the types the plan binds.
        """

        if self._operator_decision_callback is None or self._operator_submitter is None:
            raise RpcError("this run accepts no operator messages", "operator_closed")
        return self._operator_submitter({
            "id": f"pi-{uuid.uuid4().hex[:16]}", "type": str(params.get("type") or "steer"),
            "message": str(params["message"]),
        })

    def _rpc_operator_poll(self, params: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """The Supervisor delivers admitted operator messages before a model call.

        A free-play Supervisor may answer ``{"stop_turn": reason}``: the core then
        ends the turn before that call.
        """

        if self._operator_decision_callback is None:
            return None
        return self._operator_decision_callback(
            observation_step=int(params["observation_step"]),
            before_model_call_id=str(params["before_model_call_id"]),
            # A free-play run (Pi's own agent) says how many model calls its turn made so far.
            **({"calls_in_act": int(params["calls_in_act"])} if "calls_in_act" in params else {}),
        )

    def serve_operator_input(self) -> None:
        """Between turns, answer what the operator's terminal sent the core (one round trip).

        The core's requests reach Python only while it calls the core; a
        free-play Supervisor waiting for an instruction calls this so an
        instruction typed into the TUI is admitted without waiting for a turn.
        """

        self._call("ping", {})

    def announce_play_state(self, record: Mapping[str, Any]) -> None:
        """A free-play Supervisor tells the operator's terminal where the session is (display only)."""

        try:
            self.rpc.notify("play.state", dict(record))
        except RpcError:
            pass

    def _rpc_operator_stop(self, params: Mapping[str, Any]) -> None:
        """A Pi abort is `ora stop`: SIGTERM to this Supervisor, the operator's authority path."""

        os.kill(os.getpid(), signal.SIGTERM)

    def interrupt(self, reason: str = "operator_interrupt") -> None:
        self.interrupted = True
        try:
            self.rpc.notify("interrupt", {"reason": reason})
        except RpcError:
            pass

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._call("close", {})
        finally:
            self.closed = True
            self._close_robot_socket()
            self.rpc.close()
            if self.server is not None:
                self.server.close()
            if self.process is not None:
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=10)


__all__ = ["PiPlanner"]
