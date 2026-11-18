"""Synchronous toolkit execution on the Supervisor thread.

The parent invocation owns its budget and durable ledger. The data interface
only forwards one action at a time; this owner admits, executes and publishes.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import time
from typing import Any, Mapping

from ora.contracts import validate_runtime_action
from ora.contracts.joint_trajectory import native_action_cap
from .actions import dispatch_action
from .native_allowance import receipt_native_actions
from .observation import publish_observation
from .records import _atomic_json
from .script_actions import ScriptActionExecutor
from .tool_budget import ToolBudget
from .tool_execution_log import ToolExecutionInterrupted, ToolExecutionLog, WorldStateUnknown
from .termination import infrastructure_boundary


@dataclass
class Invocation:
    name: str
    context: dict
    budget: ToolBudget
    log: ToolExecutionLog
    recovered: bool = False
    publications: dict = field(default_factory=dict)
    attempted: bool = False
    reentrant: bool = False
    #: Inside a reentrant parent: the tool call running in it (``tool_call_id``, ``tool``), which its actions name.
    nested: dict | None = None


class ToolActionExecutor(ScriptActionExecutor):
    """Benchmark counterpart of the existing free-play inline executor."""

    def __init__(self, loop):
        super().__init__(loop)
        self.active: Invocation | None = None
        self.completed: dict | None = None
        self.blocked = False
        self.interruption: dict | None = None

    def observation(self) -> dict:
        current = json.loads((self.loop.control / "current-observation.json").read_text())
        return {"observation_id": current["observation_id"],
                "step": self.loop.interaction.observation_step}

    def native_actions_per_action(self) -> int:
        """The budget one primitive requests by default (a toolkit tool's ``native_action_budget``)."""
        return 20

    def _action_cap(self, action: Mapping[str, Any], remaining: int) -> int:
        """The most native actions one action may use (``joint_trajectory.native_action_cap``: 20 for a legacy
        primitive, more for a joint trajectory or a rate-limited gripper); the worker enforces it
        (``limited_action``)."""
        return native_action_cap(action)

    def _native_remaining(self) -> int | None:
        return self.loop.native_allowance.remaining()

    def nested_allowance(self) -> int | None:
        """What the reentrant parent (a code-mode script) has left for the tool running inside it, which its
        admission may lend no more of; None for a tool that is its own parent (tool-call mode), whose own budget
        is that limit."""
        call = self.active
        return call.budget.remaining if call is not None and call.reentrant else None

    def _native_used(self, raw: Any) -> int | None:
        return receipt_native_actions(raw)

    def _state_owner(self) -> Any:
        return self.loop.execution_state_owner

    def _accepts(self, action: Mapping[str, Any]) -> bool:
        """Whether one execution takes this action: a primitive (a composite needs the coordinator's stage loop)."""
        return action["schema"] in {"ora-motion-action.v3", "ora-action.v3"}

    def _interruption_cause(self, error: BaseException, stage: str) -> dict:
        """The exception behind an interruption; stage: "action" (one the ledger holds as begun, not completed),
        "tool" (the script or a tool, no action pending), "ledger_completion" or "ledger_recovery"."""
        return {"type": type(error).__name__, "message": str(error)[:2000], "stage": stage}

    @contextmanager
    def invocation(self, name: str, context: dict, requested: int):
        if not self.blocked and self.active is not None and getattr(self.active, "reentrant", False):
            # A tool called from inside a reentrant parent (a code-mode script) spends that parent's budget; its
            # actions name the tool call itself (the script's ``<call>#NNN``), as a tool call's name it.
            parent = self.active
            outer = parent.nested
            parent.nested = ({"tool_call_id": str(context["tool_call_id"]), "tool": name}
                             if context.get("tool_call_id") else None)
            try:
                yield parent
            finally:
                parent.nested = outer
            return
        if self.blocked or self.active is not None:
            raise WorldStateUnknown("tool execution is blocked or already active")
        remaining = self._native_remaining()
        if type(remaining) is not int:
            raise RuntimeError("tool execution requires a known native-action allowance")
        call_id = context["tool_call_id"]
        budget = ToolBudget(call_id, min(requested, remaining))
        log = ToolExecutionLog(self.loop.run_root, call_id, context["model_call_id"])
        try:
            recovered = log.recover()
        except WorldStateUnknown as error:
            self.interrupt(context, unknown=True, cause=self._interruption_cause(error, "ledger_recovery"))
            raise
        if recovered is not None:
            self.completed = recovered
            self.interrupt(context)
            raise ToolExecutionInterrupted("interrupted parent results cannot be resumed")
        log.create(requested=requested, admitted=budget.limit, observation=self.observation())
        self.completed = None
        self.active = call = Invocation(name, dict(context), budget, log)
        try:
            yield call
        except BaseException as error:
            from ora.agent.error_classes import ToolError
            from ora.agent.toolkit.interface.types import ActionNotStarted
            expected = isinstance(error, ActionNotStarted) or (
                isinstance(error, ToolError) and error.code != "internal_tool_error")
            if (call.attempted and not expected) or not isinstance(error, Exception):
                pending = isinstance(log.value, dict) and log.value.get("pending") is not None
                self.interrupt(context, cause=self._interruption_cause(error, "action" if pending else "tool"))
            if not isinstance(error, Exception):
                raise ToolExecutionInterrupted("tool process interrupted") from error
            raise
        finally:
            try:
                self.completed = log.finish()
            except Exception as error:
                evidence = self.interrupt(context, cause=self._interruption_cause(error, "ledger_completion"))
                failure = (WorldStateUnknown if evidence["reason"] == "world_state_unknown"
                           else ToolExecutionInterrupted)
                raise failure("execution ledger completion is not durable") from error
            finally:
                self.active = None

    def interrupt(self, context: dict, *, unknown: bool = False, cause: dict | None = None) -> dict:
        """Stop this run; a durable closed ledger establishes state, not permission to resume.

        The first interruption is kept: a later failure (the ledger's completion after an action failed)
        does not replace its ``cause``."""
        if self.interruption is not None:
            return self.interruption
        log = ToolExecutionLog(self.loop.run_root, context["tool_call_id"], context["model_call_id"])
        try:
            recovered = log.recover()
            if recovered is None and self.active is not None:
                current = self.active.log.value or {}
                unknown = unknown or bool(current.get("pending") or current.get("actions"))
        except WorldStateUnknown:
            unknown = True
        value = log.value if isinstance(log.value, dict) else {}
        actions = value.get("actions")
        actions = actions if isinstance(actions, list) else []
        last = value.get("pending") or (actions[-1] if actions else {})
        last = last if isinstance(last, dict) else {}
        reason = "world_state_unknown" if unknown else "tool_execution_interrupted"
        self.interruption = {
            "schema": "ora.tool-execution-interruption.v1", "reason": reason,
            "tool_call_id": context["tool_call_id"], "model_call_id": context["model_call_id"],
            "last_action_id": last.get("action_id"),
            "world_state_certainty": "unknown" if unknown else "known",
            **({"cause": cause} if cause is not None else {}),
        }
        self.blocked = True
        decision = infrastructure_boundary(reason, scope="session")
        self.loop.termination = decision.reason
        self.loop.termination_authority = decision.authority
        self.loop.stop_session = True
        try:
            _atomic_json(self.loop.run_root / "native/tool-execution-interruption.json", self.interruption)
            self.loop.record_termination_decision(decision, source="tool_execution_ledger")
        finally:
            self.loop.runtime.abort(decision.reason)
        return self.interruption

    def finish_skill_feedback(self, result: dict, feedback: dict) -> None:
        """Persist a derived observation projection without inventing an action."""
        call, loop = self.active, self.loop
        if (call is None or call.name != "grasp_execute" or call.recovered
                or not loop.plan.get("toolkit_observation_receipts", False)
                or call.budget.used == 0):
            return
        from ora.agent.toolkit.host.skill_feedback import assemble
        body = assemble(result, feedback, call.log.summary(), call.publications,
                        terminal=loop.environment_status.get("terminal") is True)
        # Separate provenance artifact: the last real receipt binding and the
        # trusted _execution chain remain untouched (including refused actions).
        ref = loop.planner.artifacts.put_json(body, kind="tool_skill_observation_feedback")
        _atomic_json(loop.run_root / "native/tool-observation-feedback" / call.log.path.name, {
            "schema": "ora.tool-skill-observation-feedback.v1",
            "model_call_id": call.context["model_call_id"],
            "tool_call_id": call.context["tool_call_id"],
            "source": "tool_interpretation_of_execution_receipts",
            "feedback_ref": ref,
            "actions": [{key: item[key] for key in (
                "action_id", "receipt_event_id", "receipt_ref", "after_observation")}
                for item in call.log.value["actions"]],
        })
        if loop.plan.get("toolkit_receipt_delivery") is not None:
            loop.planner.set_tool_observation_feedback(body)
        else:
            loop.planner.pending_receipt_body = body

    def _admit(self) -> str | None:
        # The last admitted parent may spend its own remaining budget even when
        # its first action raised the effective-tool count to the run limit.
        refused = self.admit()
        if refused == "robot_decision_budget" and self.active.budget.used > 0:
            if self.loop.round_model_calls >= self.loop.plan["budgets"]["max_model_calls"]:
                return "model_call_budget"
            return None
        return refused

    def execute_action(self, action: dict, budget: int) -> dict:
        call, loop = self.active, self.loop
        if self.blocked or call is None or call.recovered:
            raise WorldStateUnknown("no fresh active tool invocation may execute")
        call.attempted = True
        try:
            action = validate_runtime_action(action)
            if not self._accepts(action):
                raise ValueError("toolkit execution accepts one primitive action, not a composite")
            current = json.loads((loop.control / "current-observation.json").read_text())
            if (action["observation_id"] != current["observation_id"]
                    or action["bound_observation_sha256"] != current["observation_sha256"]):
                raise ValueError("tool action is not bound to the current publication")
            remaining = self._native_remaining()
            admitted = call.budget.admit(budget, remaining, self._action_cap(action, remaining))
            refusal = (self._admit()
                       or self._state_owner().recovery_refusal(action)
                       or ("native_action_budget_exhausted" if admitted == 0 else None))
            before = self.observation()
        except Exception as error:
            from ora.agent.toolkit.interface.types import ActionNotStarted
            # Only this pre-intent region can assert zero physical spend.
            # Failures after begin remain uncertain and must never roll back intent.
            raise ActionNotStarted(str(error)) from error
        try:
            call.log.begin(action=action, admitted=admitted, observation=before)
        except Exception as error:
            from ora.agent.toolkit.interface.types import ActionNotStarted
            # No worker dispatch can occur before begin returns. A partial
            # journal write may still block the session at invocation finish.
            raise ActionNotStarted(str(error), stage="begin") from error
        loop.planner.begin_tool_action(action["action_id"])
        loop.planner.record_tool_action(action, call.context, before, call.name)
        if refusal is not None:
            raw = {"schema": "ora.tool-action-refusal.v1", "action_id": action["action_id"],
                   "status": "refused", "reason": refusal, "native_actions": 0}
            loop.planner.record_environment_receipt(raw)
            public = dict(raw)
            used = 0
        else:
            _atomic_json(loop.control / "submitted-action.json", action)
            accepted, pointer = loop._accept_submitted_action(0)
            started = time.monotonic()
            # The tool call the action belongs to (the worker keeps one undo entry per arm-moving call) and
            # whether its parent's robot decision is already counted (an earlier action of it spent).
            identity = call.nested or {"tool_call_id": call.context["tool_call_id"], "tool": call.name}
            with self._state_owner().limited_action(admitted, {**identity, "decision_counted": call.budget.used > 0}):
                raw, captured = dispatch_action(loop, accepted, pointer, should_stop=loop._should_stop)
            used = self._native_used(raw)
            if used is None:
                raise WorldStateUnknown("worker receipt omitted native-action consumption")
            first_spend = call.budget.charge(used, admitted)
            loop._record_receipt(action, raw, time.monotonic() - started,
                                 inline_consumed=first_spend, inline_used=used)
            loop.pointer, loop.environment_status = publish_observation(
                loop, "after_action", captured_observation=captured)
            loop.interaction.begin_policy()
            public = dict(loop.planner.project_tool_receipt(raw))
            public["native_actions"] = used
            public["action_id"] = action["action_id"]
        receipt_binding = loop.planner.tool_receipt_binding(action["action_id"])
        if not isinstance(receipt_binding, Mapping) or receipt_binding.get("action_id") != action["action_id"]:
            raise WorldStateUnknown("receipt binding belongs to another action")
        call.log.complete(used=used, observation=self.observation(), receipt=public,
                          receipt_event_id=receipt_binding["event_id"],
                          receipt_ref=receipt_binding["receipt_ref"], refusal=refusal)
        if call.name == "grasp_execute" and loop.plan.get("toolkit_observation_receipts", False):
            pointer = json.loads((loop.control / "current-observation.json").read_text())
            if loop.plan.get("toolkit_receipt_delivery") is not None:
                from ora.agent.toolkit.host.publication import published_manifest
                from ora.agent.toolkit.interface.blobs import BlobStore
                root, _, _ = published_manifest(BlobStore(loop.run_root), pointer)
                pointer["source_root"] = str(root)
            call.publications[action["action_id"]] = pointer
        return {"receipt": public, "remaining_budget": call.budget.remaining}
