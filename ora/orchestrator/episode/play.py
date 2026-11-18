"""Free play: one live scene, the operator's instructions in turn (``ora play``).

A free-play RunPlan (``plan.play``, experiment ``ora-play``) binds no benchmark
task: the policy sees a scene and the operator's instructions, never a task
text, contract or success predicate. The Supervisor keeps the world open and
runs one *instruction* at a time::

    scene observation -> awaiting_instruction
      -> operator instruction -> a fresh observation (``instruction_start``)
      -> the instruction, delivered before the next model call
      -> policy turns and executions, as in an episode
      -> the policy's text report, Esc (stop_instruction) or a budget
      -> awaiting_instruction -> ...

A new instruction (not a message joining one) starts on a fresh, read-only
observation (``_observe_for_instruction``); if it fails, on the last one with an
``observation_note``.
``reset`` loads the next scene the plan bound (``plan.play.scenes``, through
``runtime.restart_round`` with that scene's seed); ``quit`` ends the session.
A message the operator sends while an instruction runs joins that instruction:
it is delivered before the next model call and the agent decides whether it
answers a question and keeps working or takes it as the new objective
(``episode-task-play-v2.md``); only Esc (``stop_instruction``) stops a running
instruction. A text-only reply that is the model's first reply after it read
a joined message is that message's *answer*, not the instruction's report: it
is recorded as a ``play_answer`` event and the instruction goes on with the
next act on the same observation. The agent core knows which model call first
read each joined message (Pi steers a message in after the current tool batch,
which may be the next act's first call) and names the messages a text-only
last reply answers (the act outcome's ``answers``). Each joined message earns
at most one such continuation; a reply with a tool call answers it while
working, and a later text-only reply is the report. ``plan.play.instruction_budgets`` bound each instruction (checked
before every model call) and ``plan.budgets`` the whole session. A free-play
scene latches no success: should the environment still end a scene (its step
limit), the instruction ends and only ``reset`` or ``quit`` follow; no verdict
is persisted. Each instruction is recorded as
``play_instruction_started``/``play_instruction_finished`` events; operator
controls as ``play_control`` and scene changes as ``play_scene_reset``. After
each published observation the environment's optional operator view (a
picture for the human, never the policy's) is stored under
``native/operator-view/`` with an ``operator_view_frame`` event. A turn
may be stopped only at a boundary: a model call in flight is recorded and an
action it proposed executes, then no further call is made. On the Pi core a
free-play turn is Pi's own agent run (ora-pi/src/play/driver.ts).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from ora.agent import ActionProposal, PolicyTurnInput
from ora.agent.session_core.operator_messages import STOP_TURN_REASONS

from .actions import dispatch_action, post_action_boundary
from ..faults import SupervisorError
from .loop import EpisodeLoop
from .observation import _session_round_budget_fields, publish_observation
from .records import _append, _status
from .result import _close_rollout_evidence, _publish_session_result, _session_result
from .operator_stop import act_unless_stopped
from .play_execution import script_stop_request, settle_inline_execution
from .play_operator_view import store_operator_view
from ..evaluation.run_artifacts import _reconciled_provider_model_calls
from .stagnation import StagnationGuard
from .termination import TerminationAuthority, TerminationDecision, operator_boundary

PLAY_SCHEMA = "ora.play-session.v1"
INSTRUCTION_SCHEMA = "ora.play-instruction.v1"
ANSWER_SCHEMA = "ora.play-answer.v1"
#: Operator controls and the turn-ending reason each becomes (operator_messages.STOP_TURN_REASONS).
CONTROL_REASONS = {
    "stop_instruction": "operator_instruction_stop",
    "reset": "operator_scene_reset",
    "quit": "operator_quit",
}
_PRIORITY = {"operator_instruction_stop": 1, "operator_scene_reset": 2, "operator_quit": 3}
#: How an instruction ends when the operator's control ends it.
_ENDED_BY = {
    "operator_instruction_stop": "operator_stopped",
    "operator_scene_reset": "operator_reset",
    "operator_quit": "operator_quit",
}
#: Turn endings the Supervisor's own poll answer causes (operator_messages.STOP_TURN_REASONS).
#: An instruction no longer ends as ``superseded``: a newer message joins it (``joined_messages``).
_INSTRUCTION_STOPS = {
    "instruction_model_call_budget": "instruction_model_call_budget",
    "instruction_wall_time_budget": "instruction_wall_time_budget",
}
_POLL_INTERVAL_S = 0.2
#: Consecutive operator-view failures after which a session stops asking (each one is recorded).


class PlayLoop(EpisodeLoop):
    """`EpisodeLoop` with an awaiting-instruction phase and two budget layers."""

    def __init__(self, *, run_root: Path, plan: Mapping[str, Any], run_id: str, live_port: Any) -> None:
        super().__init__(run_root=run_root, plan=plan, run_id=run_id, live_port=live_port)
        play = plan["play"]
        self.play_scenes = [dict(scene) for scene in play["scenes"]]
        self.instruction_budgets = dict(play["instruction_budgets"])
        self.segment: dict[str, Any] | None = None
        self.segments: list[dict[str, Any]] = []
        self.scene_records: list[dict[str, Any]] = []
        #: Operator messages admitted and not yet delivered (instructions and steers, in arrival order).
        self.play_pending: list[dict[str, Any]] = []
        self.play_queued: list[dict[str, Any]] = []      # queued_instruction: held until the running one ends
        self.play_seen: set[str] = set()
        self.play_stop: str | None = None
        self.play_controls = 0
        # The environment ended the current scene; only reset or quit follow.
        self.scene_over = False
        #: The environment's operator view (native/operator-view/), never the policy's.
        self.operator_view: dict[str, Any] | None = None
        self.operator_view_failures = 0
        #: The action executing now, for the operator's "moving" status (display only).
        self._moving: dict[str, Any] | None = None
        #: When the latest observation was published (the age a failed instruction-start look reports).
        self.observation_unix_s: float | None = None

    # -- the session ----------------------------------------------------------------

    def _start_operator_control(self) -> None:
        if "operator_control" not in self.plan:
            raise SupervisorError("a free-play session needs plan-bound operator control")
        super()._start_operator_control()

    def _start_environment(self) -> None:
        super()._start_environment()
        self._require_scene(self.reset_record, 0)

    def _require_scene(self, reset: Mapping[str, Any], index: int) -> None:
        expected = self.play_scenes[index]["seed"]
        if reset.get("seed") != expected:
            raise SupervisorError(f"free-play scene {index} expected seed {expected}, got {reset.get('seed')!r}")

    def _run_rounds(self) -> None:
        try:
            self._play()
        except Exception:
            if self.segment is not None:
                self._finish_segment("trusted_fault")
            raise

    def _play(self) -> None:
        self.pointer, self.environment_status = publish_observation(self, "reset")
        self._scene_started(cause="session_start")
        while self.termination is None:
            if self.segment is None:
                self._await_instruction()
                continue
            self._take_commands()
            if self.play_stop is not None:
                self._apply_stop()
                continue
            if self.interrupted:
                self._end_on_operator_signal()
                continue
            if self.environment_status["terminal"]:
                self._scene_ended()
                continue
            if self._check_session_budget() or self._check_instruction_budget():
                continue
            outcome, call_increment = self._policy_turn()
            if settle_inline_execution(self, outcome, call_increment):  # play_execution.py
                continue
            if isinstance(outcome, ActionProposal):
                self._execute_proposal(call_increment)
            else:
                self._settle_play_outcome(outcome, call_increment)

    def _finish(self) -> dict[str, Any]:
        return finish_play_session(self)

    # -- waiting for the operator ----------------------------------------------------

    def _await_instruction(self) -> None:
        """Wait for the next instruction, a scene reset, quit, a stop signal or the session's end."""

        if self.interaction.phase != "awaiting_instruction":
            self.interaction.await_instruction(reason="instruction_ended" if self.segments else "scene_ready")
            self._announce("awaiting_instruction")
        while self.termination is None:
            if self.interrupted:
                self._end_on_operator_signal()
                return
            if self._check_session_budget():
                return
            serve = getattr(self.planner, "serve_operator_input", None)
            if callable(serve):
                serve()  # an instruction typed into the Pi TUI reaches the inbox now
            self._take_commands()
            if self.play_stop is not None:
                self._apply_stop()
                return
            # Between instructions any message (an instruction, or a steer that arrived as one ended) opens the next.
            instruction = self.play_pending[0] if self.play_pending else None
            if instruction is not None and self.scene_over:
                self.play_pending.remove(instruction)
                self._control(instruction, "scene_ended_reset_required")
                continue
            if instruction is not None:
                self.play_pending.remove(instruction)
                look = self._observe_for_instruction()
                self._begin_segment(instruction, look)
                return
            time.sleep(_POLL_INTERVAL_S)

    def _observe_for_instruction(self) -> dict[str, Any]:
        """Publish a fresh observation for a new instruction (the opening scene's path; it moves nothing).

        A look that fails before publishing is recorded and the instruction runs
        on the last observation; one that fails after publishing is a fault.
        """

        previous = self.interaction.observation_step
        try:
            self.pointer, self.environment_status = publish_observation(self, "instruction_start")
        except Exception as exc:
            if self.interaction.phase != "awaiting_instruction" or self.interaction.observation_step != previous:
                raise
            self.record_auxiliary_failure("play:instruction_observation", exc)
            self.interaction.begin_instruction()
            return {"fresh": False, "observation_step": previous, "error": f"{type(exc).__name__}: {exc}"}
        return {"fresh": True, "observation_step": self.interaction.observation_step}

    def _observation_note(self, look: Mapping[str, Any] | None) -> str | None:
        """What the model reads with an instruction whose fresh look failed (None when the look is fresh)."""
        if look is None or look.get("fresh", True):
            return None
        minutes, seconds = divmod(max(0, int(round(time.time() - (self.observation_unix_s or time.time())))), 60)
        age = f", captured {minutes} min {seconds} s ago" if minutes else f", captured {seconds} s ago"
        error = str(look.get("error") or "unknown error")[:200]
        return (f"ORA could not take a fresh observation at the start of this instruction ({error}). "
                f"The observation with it is the latest one, step {look.get('observation_step')}{age}; "
                "the scene may have changed since.")

    def _take_commands(self) -> None:
        """Admit new inbox commands: instructions and steers wait for delivery, controls apply."""

        for command in self.operator_inbox.poll():
            if command["id"] in self.play_seen:
                continue
            self.play_seen.add(command["id"])
            kind = command["type"]
            if kind == "queued_instruction" and self.segment is not None:
                # Held until the running instruction ends: then it is the next one.
                self.play_queued.append(command)
                self._announce("message_queued_next", message=self._message_record(command))
                continue
            if kind in {"instruction", "steer", "queued_instruction"}:
                self.play_pending.append(command)
                if self.segment is not None:
                    # It joins the running instruction, delivered before the next model call.
                    self._announce("message_queued", message=self._message_record(command))
                continue
            reason = CONTROL_REASONS.get(kind)
            refusal = None
            if reason is None:
                refusal = "not_a_free_play_command"
            elif kind == "stop_instruction" and self.segment is None:
                refusal = "no_instruction_running"
            elif kind == "reset" and self.scene_index + 1 >= len(self.play_scenes):
                refusal = "no_scene_left"
            self._control(command, refusal)
            if refusal is None and _PRIORITY[reason] > _PRIORITY.get(self.play_stop, 0):
                self.play_stop = reason

    def _control(self, command: Mapping[str, Any], refusal: str | None) -> None:
        status = "applied" if refusal is None else "rejected"
        self.operator_inbox.acknowledge(command, status, **({} if refusal is None else {"reason": refusal}))
        self.play_controls += 1
        record = {
            "id": command["id"], "command": command["type"], "status": status,
            "scene_index": self.scene_index,
            "instruction_index": None if self.segment is None else self.segment["index"],
            **({} if refusal is None else {"reason": refusal}),
        }
        _append(self.events, {"type": "play_control", "record": record, "time": time.time()})
        self._announce("control", control=record)

    def _apply_stop(self) -> None:
        reason, self.play_stop = self.play_stop, None
        if self.segment is not None:
            self._finish_segment(_ENDED_BY[reason])
        if reason == "operator_scene_reset":
            self._reset_scene(cause="operator")
        elif reason == "operator_quit":
            self._end_session("operator_quit", TerminationAuthority.OPERATOR)

    # -- delivery before each model call ---------------------------------------------

    def _deliver_operator_messages(self, *, observation_step: int, before_model_call_id: str,
                                   calls_in_act: int = 0) -> Mapping[str, Any] | None:
        """Before a model call: deliver this instruction and the messages that joined it, or end the turn.

        A pending control or this instruction's spent budget ends the turn
        here. Every other message the operator sent since the last call joins
        the running instruction, in arrival order: the agent reads it before
        this call and decides what it changes. ``calls_in_act`` counts the
        model calls the running turn already made.
        """

        self._take_commands()
        if self.play_stop is not None:
            return {"stop_turn": self.play_stop}
        if self.segment is None:
            return None
        stop = self._instruction_spent(calls_in_act)
        if stop is not None:
            return {"stop_turn": stop}
        delivery = {"observation_step": observation_step, "before_model_call_id": before_model_call_id,
                    "instruction_id": self.segment["instruction_id"]}
        if not self.segment["delivered"]:
            note = self._observation_note(self.segment["observation"])
            self._deliver_operator_message(self.segment["command"], **delivery, joined=False,
                                           **({} if note is None else {"observation_note": note}))
            self.segment["delivered"] = True
            self._announce_delivery(self.segment["command"], observation_step, joined=False)
        while self.play_pending:
            command = self.play_pending.pop(0)
            self._deliver_operator_message(command, **delivery, joined=True)
            self.segment["joined_messages"].append(command["id"])
            self._announce_delivery(command, observation_step, joined=True)
        return None

    def _message_record(self, command: Mapping[str, Any]) -> dict[str, Any]:
        """An operator message as the operator's terminal and page show it (display only)."""

        return {"id": command["id"], "text": str(command.get("message") or ""),
                "instruction": None if self.segment is None else self.segment["index"] + 1}

    def _announce_delivery(self, command: Mapping[str, Any], observation_step: int, *, joined: bool) -> None:
        self._announce("message_delivered", message={
            **self._message_record(command), "joined": joined, "observation_step": observation_step,
        })

    def _instruction_spent(self, calls_in_act: int) -> str | None:
        """The instruction budget a model call would exceed, checked before every model call."""

        budgets, segment = self.instruction_budgets, self.segment
        if time.time() - segment["started_unix_s"] >= budgets["wall_time_s"]:
            return "instruction_wall_time_budget"
        if self.model_calls - segment["model_calls_at_start"] + calls_in_act >= budgets["max_model_calls"]:
            return "instruction_model_call_budget"
        return None

    def script_action_refusal(self) -> str | None:
        """Why a script's robot action may not run now (an Esc sent meanwhile included); None when it may."""

        self._take_commands()
        if self.play_stop is not None:
            return "operator_stop_pending"
        if self.segment is None:
            return "no_instruction_running"
        if self.scene_over:
            return "scene_ended"
        if self.decisions - self.segment["decisions_at_start"] >= self.instruction_budgets["max_robot_decisions"]:
            return "instruction_robot_decision_budget"
        return None

    script_stop_request = script_stop_request  # a running code-mode script's stop (play_execution.py)

    # -- one instruction ----------------------------------------------------------------

    def _begin_segment(self, command: Mapping[str, Any], look: Mapping[str, Any] | None = None) -> None:
        look = None if look is None else dict(look)  # _observe_for_instruction's record
        self.segment = {
            "index": len(self.segments), "instruction_id": command["id"], "command": dict(command),
            "delivered": False, "joined_messages": [], "answered_messages": [],
            "scene_index": self.scene_index,
            "started_unix_s": time.time(), "started_monotonic_ns": time.monotonic_ns(),
            "decisions_at_start": self.decisions, "refused_at_start": self.refused_proposals,
            "model_calls_at_start": self.model_calls,
            "observation_step_at_start": self.interaction.observation_step,
            "observation": look,
        }
        _append(self.events, {"type": "play_instruction_started", "record": {
            "schema": INSTRUCTION_SCHEMA, "index": self.segment["index"], "instruction_id": command["id"],
            "scene_index": self.scene_index, "observation_step": self.interaction.observation_step,
            "observation": look,
        }, "time": self.segment["started_unix_s"]})
        self._announce("instruction", instruction_id=command["id"], message=self._message_record(command))

    def _finish_segment(self, ended: str) -> None:
        segment, self.segment = self.segment, None
        if segment is None:
            return
        if not segment["delivered"]:
            self.operator_inbox.acknowledge(segment["command"], "rejected", reason="instruction_ended_before_delivery")
        record = {
            "schema": INSTRUCTION_SCHEMA, "index": segment["index"], "instruction_id": segment["instruction_id"],
            "scene_index": segment["scene_index"], "delivered": segment["delivered"],
            "joined_messages": list(segment["joined_messages"]),
            "answered_messages": list(segment["answered_messages"]), "ended": ended,
            "decisions": self.decisions - segment["decisions_at_start"],
            "refused_proposals": self.refused_proposals - segment["refused_at_start"],
            "model_calls": self.model_calls - segment["model_calls_at_start"],
            "observation_steps": {"start": segment["observation_step_at_start"],
                                  "end": self.interaction.observation_step},
            "started_unix_s": segment["started_unix_s"], "finished_unix_s": time.time(),
            "wall_duration_s": round(max(0, time.monotonic_ns() - segment["started_monotonic_ns"]) / 1e9, 6),
            "wall_duration_clock": "monotonic",
        }
        self.segments.append(record)
        self.play_pending.extend(self.play_queued)          # the operator's queued instructions, in order
        self.play_queued.clear()
        _append(self.events, {"type": "play_instruction_finished", "record": record, "time": record["finished_unix_s"]})
        self._announce("instruction_finished", instruction=record)

    def _policy_turn(self) -> tuple[Any, int]:
        """One policy turn for the current instruction; a text-only reply is its report."""

        turn = PolicyTurnInput(
            task_id=str(self.plan["task"]),
            task_instruction=self.plan["task_instruction"],
            context=self.pointer,
            image_path=self.workspace / self.pointer["main_image"],
            observation_id=self.interaction.observation_id,
            observation_step=self.interaction.observation_step,
            budget={key: value for key, value in self.play_remaining_budget().items()
                    if not key.startswith("session_round")},
            interaction_phase=self.interaction.phase,
            fresh_observation_required=self.interaction.fresh_observation_required,
            allow_reminder=False,
        )
        self.interaction.begin_policy()
        outcome = act_unless_stopped(self.planner, turn, self._should_stop, self.events)
        call_increment = outcome.model_calls if self.planner.capabilities.runtime_kind != "scripted" else 0
        self.model_calls += call_increment
        self.round_model_calls += call_increment
        return outcome, call_increment

    def _settle_play_outcome(self, outcome: Any, call_increment: int) -> None:
        reason = str(getattr(outcome, "reason", "policy_no_action"))
        if outcome.kind == "lifecycle":
            raise SupervisorError("a free-play session admits no lifecycle request")
        if reason in {"operator_interrupt", "context_exhausted"}:
            # The run's operator stop and an exhausted session keep their episode boundaries.
            if self.segment is not None:
                self._finish_segment(reason)
            self._settle_policy_outcome(outcome, call_increment)
            return
        self.interaction.record_policy_outcome(
            kind="no_action", model_calls=call_increment, reason=reason, terminal=False,
        )
        if reason in _INSTRUCTION_STOPS:
            self._finish_segment(_INSTRUCTION_STOPS[reason])
        elif reason in STOP_TURN_REASONS:
            self.play_stop = self.play_stop or reason
            self._apply_stop()
        elif reason == "policy_no_action":
            if not self._answered_joined_messages(outcome):
                self._finish_segment("reported")
        elif reason == "provider_error":
            self._finish_segment("provider_error")  # e.g. a model the operator chose answers with an error
        elif reason == "output_truncated":
            # The provider's output limit cut the last reply off before any tool call, and the act's budget
            # left no call to continue it (ora-pi/src/play/driver.ts `afterTurn`): no report.
            self._finish_segment("output_truncated")
        elif call_increment == 0:
            self._finish_segment("no_progress")

    def _answered_joined_messages(self, outcome: Any) -> bool:
        """A text-only reply answering joined messages: record each answer and keep the instruction running.

        The agent core names the joined messages this reply was the first to
        read (``outcome.details["answers"]``); each one that joined this
        instruction and has no answer yet gets one ``play_answer`` event, and
        the instruction goes on with the next act on the same observation.
        False when the reply answers none (it is the instruction's report).
        """
        segment = self.segment
        if segment is None:
            return False
        named = (getattr(outcome, "details", None) or {}).get("answers") or ()
        answered = [item for item in dict.fromkeys(str(value) for value in named)
                    if item in segment["joined_messages"] and item not in segment["answered_messages"]]
        if not answered:
            return False
        now = time.time()
        for message_id in answered:
            segment["answered_messages"].append(message_id)
            record = {"schema": ANSWER_SCHEMA, "instruction_id": segment["instruction_id"],
                      "instruction_index": segment["index"], "message_id": message_id,
                      "observation_step": self.interaction.observation_step}
            _append(self.events, {"type": "play_answer", "record": record, "time": now})
            # Display only: the terminal and the page show the reply as the answer, the instruction running.
            self._announce("answer", answer=record)
        return True

    def _execute_proposal(self, call_increment: int) -> None:
        """Execute the proposal and observe again; an environment terminal ends the scene at the loop top."""

        action, current = self._accept_submitted_action(call_increment)
        started = time.monotonic_ns()
        receipt, captured = dispatch_action(self, action, current, should_stop=self._should_stop)
        self._record_receipt(action, receipt, round(max(0, time.monotonic_ns() - started) / 1e9, 6))
        self.pointer, self.environment_status = publish_observation(
            self, "after_action", captured_observation=captured,
        )
        boundary = post_action_boundary(receipt, self.environment_status)
        if boundary is not None and boundary[0].authority is TerminationAuthority.INFRASTRUCTURE:
            # A control fault: the world can no longer be trusted, so the session ends.
            decision, environment_reason = boundary
            self.record_termination_decision(decision, source="environment_status")
            self.interaction.mark_terminal(environment_reason)
            if self.segment is not None:
                self._finish_segment("control_fault")
            self.termination = environment_reason
            self.termination_authority = decision.authority

    # -- the operator view -------------------------------------------------------------

    def observation_published(self, step: int) -> None:
        """Store the environment's operator view of observation ``step`` (play_operator_view.py)."""

        self.observation_unix_s = time.time()
        store_operator_view(self, step)

    # -- the robot moving (the operator's liveness) --------------------------------------

    def _accept_submitted_action(self, call_increment: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """The policy's or a script's action is about to execute: the operator sees the robot moving."""

        action, current = super()._accept_submitted_action(call_increment)
        self._moving = {"action_id": action.get("action_id"), "observation_step": self.interaction.observation_step,
                        "started_monotonic_ns": time.monotonic_ns(), "started_unix_s": time.time()}
        self._announce("action_started", action={key: self._moving[key]
                                                  for key in ("action_id", "observation_step", "started_unix_s")})
        return action, current

    def _record_receipt(self, action: dict, receipt: dict, execute_duration_s: float, **accounting: Any) -> None:
        super()._record_receipt(action, receipt, execute_duration_s, **accounting)  # an inline spend's accounting
        moving, self._moving = self._moving, None
        if moving is not None:
            self._announce("action_finished", action={
                "action_id": moving["action_id"], "observation_step": moving["observation_step"],
                "duration_s": execute_duration_s, "started_unix_s": moving["started_unix_s"],
                "finished_unix_s": time.time(), "approval_wait_s": _approval_wait_s(receipt),
            })

    # -- budgets ----------------------------------------------------------------------

    def play_remaining_budget(self) -> dict[str, Any]:
        """What the policy sees: the session's remaining budget (each instruction is capped too)."""

        budgets = self.plan["budgets"]
        remaining = {
            "robot_decisions": budgets["max_robot_decisions"] - self.decisions,
            "model_calls": budgets["max_model_calls"] - self.model_calls,
            "wall_time_s": max(0, round(budgets["wall_time_s"] - (time.time() - self.started), 3)),
            **self.native_allowance.fields(),
        }
        remaining.update(_session_round_budget_fields(
            self.policy, round_index=self.round_index, round_limit=len(self.play_scenes), multi_round=False,
        ))
        return remaining

    def _session_exhausted(self) -> str | None:
        budgets = self.plan["budgets"]
        if time.time() - self.started >= budgets["wall_time_s"]:
            return "wall_time_budget"
        if self.decisions >= budgets["max_robot_decisions"]:
            return "robot_decision_budget"
        if self.model_calls >= budgets["max_model_calls"]:
            return "model_call_budget"
        return None

    def _check_session_budget(self) -> bool:
        reason = self._session_exhausted()
        if reason is None:
            return False
        if self.segment is not None:
            self._finish_segment(f"session_{reason}")
        self._end_session(reason, TerminationAuthority.EXPERIMENT_BUDGET)
        return True

    def _check_instruction_budget(self) -> bool:
        budgets, segment = self.instruction_budgets, self.segment
        reason = None
        if time.time() - segment["started_unix_s"] >= budgets["wall_time_s"]:
            reason = "instruction_wall_time_budget"
        elif self.decisions - segment["decisions_at_start"] >= budgets["max_robot_decisions"]:
            reason = "instruction_robot_decision_budget"
        elif self.model_calls - segment["model_calls_at_start"] >= budgets["max_model_calls"]:
            reason = "instruction_model_call_budget"
        if reason is None:
            return False
        self._finish_segment(reason)
        return True

    # -- scenes and the end --------------------------------------------------------------

    def _scene_started(self, *, cause: str) -> None:
        scene = self.play_scenes[self.scene_index]
        record = {"scene_index": self.scene_index, "seed": scene["seed"], "cause": cause,
                  "observation_step": self.interaction.observation_step, "started_unix_s": time.time()}
        self.scene_records.append(record)
        if self.scene_index:
            _append(self.events, {"type": "play_scene_reset", "record": record, "time": record["started_unix_s"]})
        view = self.operator_view if (self.operator_view or {}).get("observation_step") == record["observation_step"] else None
        self._announce("scene", cause=cause, image=self._scene_image(),
                       **({} if view is None else {"operator_view": view["path"]}))

    def _scene_image(self) -> str | None:
        """A display copy of the scene's main camera under native/ (the agent process may read it).

        The operator sees the scene before the first instruction; the model
        receives it only with that instruction's observation.
        """

        source = self.last_observation_image
        if source is None or not Path(source).is_file():
            return None
        target = self.native / "play-scenes" / f"scene-{self.scene_index:04d}{Path(source).suffix}"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        except OSError as exc:
            self.record_auxiliary_failure("play:scene_image", exc)
            return None
        return target.relative_to(self.run_root).as_posix()

    def _scene_ended(self) -> None:
        """The environment ended this scene by itself (its step limit): only reset or quit follow."""

        reason = str(self.environment_status.get("reason") or "environment_terminal")
        if self.segment is not None:
            self._finish_segment("scene_ended")
        self.scene_over = True
        self.scene_records[-1]["environment_ended"] = reason
        _append(self.events, {"type": "play_scene_ended", "scene_index": self.scene_index,
                              "reason": reason, "time": time.time()})
        self._announce("scene_ended", reason=reason)

    @property
    def scene_index(self) -> int:
        return self.round_index

    def _reset_scene(self, *, cause: str) -> None:
        """Load the next pre-drawn scene in the same worker, as a multi-round session resets a round."""

        if self.interaction.phase != "awaiting_instruction":
            self.interaction.await_instruction(reason=f"scene_reset:{cause}")
        try:
            self.projector.close_observation()
        except Exception as exc:
            self.record_auxiliary_failure("play:projector_close", exc)
        index = self.round_index + 1
        reset_tools = getattr(self.tools, "reset_round", None)
        if not callable(reset_tools):
            raise SupervisorError("ToolRuntime does not support a scene reset")
        reset_tools(index)
        self.round_index = index
        self.round_decisions = self.round_refused_proposals = self.round_model_calls = 0
        self.native_allowance.reset()
        self.round_started = time.time()
        self.round_started_monotonic_ns = time.monotonic_ns()
        self.round_start_step = len(self.state_store.steps)
        self.previous_public_action = None
        self.pending_execution = None
        self.last_observation_image = None
        self.stagnation = StagnationGuard(self.control / "rounds" / f"round-{index:04d}" / "stagnation.json")
        self.interaction.restart_scene()
        self.scene_over = False
        self.reset_record = self.runtime.restart_round(index, seed=self.play_scenes[index]["seed"])
        self._require_scene(self.reset_record, index)
        self.audit_round_isolation(self.reset_record, index)
        self.require_reset_descriptor(
            self.reset_record, invalid="environment scene reset descriptor is invalid",
            drift="environment runtime descriptor drift after scene reset",
        )
        _append(self.events, {"type": "reset", "record": self.reset_record, "round_index": index, "time": time.time()})
        self.round_reset_notice = self.scene_notice(index)
        _status(self.status_path, run_id=self.run_id, state="running", decisions=self.decisions,
                refused_proposals=self.refused_proposals, model_calls=self.model_calls, official_success=None)
        self.pointer, self.environment_status = publish_observation(self, "reset")
        self._scene_started(cause=cause)

    def scene_notice(self, index: int) -> dict[str, Any]:
        """The scene change as the policy reads it (rendered like a session round notice)."""

        return {
            "schema": "ora.session-round-notice.v1",
            "kind": "session_round_start",
            "round": index + 1,
            "total_rounds": len(self.play_scenes),
            "budget_scope": "per_instruction",
            "instruction": (
                "The operator's scene was replaced by a new one in the same AgentSession. "
                "Prior scene state and coordinates are stale. Re-ground from the current "
                "observation and wait for the operator's next instruction."
            ),
        }

    def _end_on_operator_signal(self) -> None:
        """`ora stop` (SIGTERM) ends the whole session, as for any run."""

        if self.segment is not None:
            self._finish_segment("operator_interrupt")
        decision = operator_boundary()
        self.record_termination_decision(decision, source="operator_signal")
        self.interaction.abort("operator_interrupt")
        self.termination = decision.reason
        self.termination_authority = decision.authority

    def _end_session(self, reason: str, authority: TerminationAuthority) -> None:
        """/quit, the session budget or the last scene's end: the session completes, unscored."""

        if self.segment is not None:
            self._finish_segment(reason)
        self.record_termination_decision(
            TerminationDecision(reason=reason, authority=authority, scope="session"), source="play_session",
        )
        self.termination = reason
        self.termination_authority = authority
        self._announce("ended", termination=reason)

    # -- the operator's terminal ---------------------------------------------------------

    def _announce(self, state: str, **fields: Any) -> None:
        """Tell the live port and the agent core's terminal (display only) where the session is."""

        record = {
            "schema": "ora.play-state.v1", "state": state, "unix_s": time.time(),
            "scene": self.round_index + 1, "scenes": len(self.play_scenes),
            "instructions": len(self.segments) + (self.segment is not None),
            "running": None if self.segment is None else self.segment["instruction_id"],
            **fields,
        }
        self.publish_live("play_state", record)
        notify = getattr(self.planner, "announce_play_state", None)
        if callable(notify):
            try:
                notify(record)
            except Exception as exc:
                self.record_auxiliary_failure("play_state", exc)

    def play_summary(self) -> dict[str, Any]:
        return {
            "schema": PLAY_SCHEMA,
            "instruction_count": len(self.segments),
            "delivered_instruction_count": sum(item["delivered"] for item in self.segments),
            "instructions": list(self.segments),
            "scene_count": len(self.scene_records),
            "scenes": list(self.scene_records),
            "control_count": self.play_controls,
        }


def _approval_wait_s(receipt: Any) -> float | None:
    """How long a motion waited for the operator, from its receipt's ``timestamps.approval``
    (a real cell's worker records it; nested receipts and phase receipts are searched)."""

    stack = [receipt]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            approval = (value.get("timestamps") or {}).get("approval") if isinstance(value.get("timestamps"), Mapping) \
                else None
            if isinstance(approval, Mapping) and isinstance(approval.get("created_unix_s"), (int, float)) \
                    and isinstance(approval.get("answered_unix_s"), (int, float)):
                return round(max(0.0, approval["answered_unix_s"] - approval["created_unix_s"]), 3)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return None


def finish_play_session(loop: PlayLoop) -> dict[str, Any]:
    """A free-play result: no official verdict, never a behavioral sample.

    `/quit`, the session budget and running out of scenes complete the
    session; `ora stop` (operator_interrupt) and infrastructure leave it
    incomplete, as for any run.
    """

    authority = loop.termination_authority
    termination = loop.termination
    state = ("incomplete" if authority is TerminationAuthority.INFRASTRUCTURE
             or termination == "operator_interrupt" else "completed")
    video = _close_rollout_evidence(loop, state, termination)
    loop.model_calls = _reconciled_provider_model_calls(loop.run_root, loop.plan, loop.model_calls)
    result = _session_result(
        loop, state=state, official=None, video=video,
        session_termination=termination, session_authority=authority,
    )
    result["behavioral_sample_valid"] = False
    result["play"] = loop.play_summary()
    _publish_session_result(
        loop, result, state=state, official=None,
        session_termination=termination, session_authority=authority,
    )
    return result


__all__ = ["PlayLoop", "finish_play_session"]
