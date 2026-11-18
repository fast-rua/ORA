"""Tool calls the Pi agent core sends to Python (a ``PiPlanner`` mixin).

Tools run through the existing ToolRuntime; each result is projected the way
a native_api session delivers it (delivery, units, causal shape), with the trims
returned as retention hints and images stored as content-addressed references.
A mutation that succeeded hands its persisted action proposal back to the core.
"""

from __future__ import annotations

import base64
import binascii
import copy
import json
import os
from typing import Any, Mapping

from ora.agent.context.delivery.trim import CAPABILITY as TRIM, trim_tool_result
from ora.agent.context.delivery.dedup import CAPABILITY as DEDUP
from ora.agent.context.delivery.dedup import hoist_fact_conventions
from ora.agent.context.delivery.tool_results import _robot_tool
from ora.agent.composition.surface.units import CAPABILITY as DEGREES, policy_degrees
from ora.agent.contracts import PolicySessionError
from ora.agent.session_core.constants import COMPOSITION_CONTEXT_CAPABILITIES, NATIVE_CAUSAL_SLOTS, TOOL_IMAGE_CALL_LABELS
from ora.agent.session_core.tool_arguments import _tool_error
from ora.agent.toolkit.host.observe_labels import observe_image_labels

from . import hints as retention_hints


class PiToolRpc:
    """``tools.*`` RPC handlers of ``PiPlanner``."""

    def bind_tool_action_executor(self, executor, *, observation_receipts: bool) -> None:
        self.tool_action_executor = executor
        self.toolkit_observation_receipts = observation_receipts
        self._pending_tool_deliveries = {}

    def set_tool_observation_feedback(self, body: Mapping[str, Any]) -> None:
        """Deliver derived feedback through the session facade, not an attribute shadow.

        Receipt references and execution authority remain bound to real actions.
        This body is only the subsequent observation's public interpretation.
        """
        self.pending_receipt_body = copy.deepcopy(dict(body))

    def _rpc_tools_settled(self, params: Mapping[str, Any]) -> None:
        """Pi persisted these read-only/zero-spend parents before its next request."""
        pending = getattr(self, "_pending_tool_deliveries", {})
        for identity in params["calls"]:
            pending.pop((str(identity["model_call_id"]), str(identity["tool_call_id"])), None)

    def _interrupt_pending_tool(self) -> bool:
        executor = getattr(self, "tool_action_executor", None)
        if executor is None:
            return False
        if executor.blocked:
            return True
        pending = getattr(self, "_pending_tool_deliveries", {})
        context = (executor.active.context if executor.active is not None
                   else next(reversed(pending.values()), None))
        if context is None:
            # A core may die after proposing a read-only tool but before its RPC
            # reaches Python. Inspect only the current provider exchange.
            events = self.run_root / "agent-events.jsonl"
            settled = set()
            for line in reversed(events.read_text().splitlines() if events.exists() else []):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # A crashed writer may leave an incomplete final line.
                if event.get("type") == "provider_request":
                    break
                identity = (event.get("model_call_id"), event.get("tool_call_id"))
                if event.get("type") == "tool_result":
                    settled.add(identity)
                if event.get("type") == "tool_call_proposed" and identity not in settled:
                    context = {"model_call_id": identity[0], "tool_call_id": identity[1]}
                    break
        if context is None:
            return False
        executor.interrupt(context)
        return True

    def record_tool_action(self, action: dict, context: dict, before: dict, name: str) -> None:
        event = self._call("record", {"type": "action_proposed", "fields": {
            "observation_step": before["step"],
            "model_call_id": context.get("model_call_id"),
            "tool_call_id": context["tool_call_id"],
            "payload": {"action_id": action["action_id"],
                        "observation_id": before["observation_id"]},
            "artifact_ref": self.artifacts.put_json(action, kind="native_action_proposal"),
        }})
        self.last_action_event_id = event["event_id"]
        self.last_action_step = before["step"]
        self.last_action_id = action["action_id"]
        self.last_action_public_name = name

    def project_tool_receipt(self, receipt: dict) -> dict:
        result = self._minimal_action_effect(receipt)
        record = self._receipt_record(receipt)
        # Explicit public controller evidence needed by tool-local stage policy.
        # Never expose the worker object or arbitrary receipt keys.
        if record.get("schema") == "ora.robodojo-sparse-motion-receipt.v1":
            result["stage_evidence"] = {
                "execution_status": record.get("execution_status"),
                "failure_code": record.get("failure_code"),
                "consumed": record.get("consumed") is True,
                "command_invariant_verified": record.get("command_invariant", {}).get("verified") is True,
                "gripper_directives": {arm: value for arm, value in record.get("gripper_directives", {}).items()
                                       if arm in {"left", "right"} and type(value) in (int, float)},
            }
            # Per-physics-step executors: the outcome a lib planner reports
            # back (raw residuals; the joint arrays stay Control-only).
            trajectory = record.get("trajectory")
            if isinstance(trajectory, Mapping):
                result["stage_evidence"]["trajectory"] = {
                    **{key: copy.deepcopy(trajectory.get(key)) for key in (
                        "status", "knots_planned", "knots_executed", "native_actions", "stop", "settle")},
                    **({"speed_segments": copy.deepcopy(trajectory["speed_segments"])}
                       if isinstance(trajectory.get("speed_segments"), Mapping) else {}),
                    "arms": {arm: {key: copy.deepcopy(value.get(key)) for key in (
                        "final_joint_error_rad", "final_joint_error_per_joint_rad", "max_tracking_error_rad",
                        "final_link6_error")}
                        for arm, value in (trajectory.get("arms") or {}).items() if arm in {"left", "right"}}}
            ramp = record.get("gripper_ramp")
            if isinstance(ramp, Mapping):
                result["stage_evidence"]["gripper_ramp"] = {
                    key: copy.deepcopy(ramp.get(key)) for key in ("native_actions", "ramp_native_actions", "arm_settle")}
        return result

    def begin_tool_action(self, action_id: str) -> None:
        """Invalidate the previous receipt before issuing a new trusted action."""
        self.pending_receipt = self.pending_receipt_body = None
        self._pending_tool_action_id = action_id

    def tool_receipt_binding(self, action_id: str) -> dict:
        # Event delivery is best effort elsewhere; execution authority must fail closed.
        binding = self.pending_receipt
        if (getattr(self, "_pending_tool_action_id", None) != action_id
                or not isinstance(binding, Mapping) or binding.get("action_id") != action_id):
            raise PolicySessionError("current tool action lacks its own delivered receipt")
        self._call("execution_evidence_barrier", {})
        return copy.deepcopy(binding)

    def _rpc_classify(self, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        effects = []
        for call in params["calls"]:
            try:
                effect = self.gateway.classify_effect(call["name"], call["arguments"])
                effects.append({"effect": effect})
            except Exception as exc:  # the call becomes a refused parse error
                effects.append({"error": _tool_error(exc)})
        return effects

    def _project(self, name: str, result: Mapping[str, Any], *, call_id: str,
                 step: int) -> dict[str, Any]:
        """Delivery and causal projection, trims returned as hints."""

        surface = getattr(self.tools, "surface", None)
        declared = frozenset(getattr(surface, "capabilities", ()) or ()) & {DEDUP, DEGREES, TRIM}
        untrimmed = trimmed = result
        if declared and isinstance(result, Mapping) and _robot_tool(surface, name):
            # delivery.tool_results.deliver_tool_result, with its last step (the trim) kept apart.
            untrimmed = dict(result)
            if DEDUP in declared and isinstance(untrimmed.get("public_robot_facts"), Mapping):
                untrimmed["public_robot_facts"] = hoist_fact_conventions(untrimmed["public_robot_facts"])
            untrimmed = policy_degrees(untrimmed) if DEGREES in declared else untrimmed
            stated = getattr(getattr(self, "gateway", None), "stated_prose", None)
            trimmed = (trim_tool_result(untrimmed, stated() if callable(stated) else None)
                       if TRIM in declared else untrimmed)
        canonical_untrimmed = copy.deepcopy(dict(untrimmed))
        canonical_trimmed = copy.deepcopy(dict(trimmed))
        images_b64 = []
        for record in (canonical_untrimmed, canonical_trimmed):
            single = record.pop("image_base64", None)
            multiple = record.pop("images_base64", None)
            if record is canonical_trimmed:
                if single is not None and multiple is not None:
                    raise PolicySessionError("tool result cannot mix image_base64 and images_base64")
                images_b64 = [single] if single is not None else multiple or []
        if not isinstance(images_b64, list) or not all(isinstance(item, str) for item in images_b64):
            raise PolicySessionError("tool returned invalid image payloads")
        if self.context_composition in NATIVE_CAUSAL_SLOTS:
            visible_untrimmed = self._causal_tool_result(name, canonical_untrimmed)
            visible_trimmed = self._causal_tool_result(name, canonical_trimmed)
        else:
            visible_untrimmed = copy.deepcopy(canonical_untrimmed)
            visible_untrimmed.pop("_runtime", None)
            visible_trimmed = copy.deepcopy(canonical_trimmed)
            visible_trimmed.pop("_runtime", None)
        rule = "observation_delivery_trim_v2"
        stored = canonical_trimmed if self.context_composition in NATIVE_CAUSAL_SLOTS else visible_trimmed
        stored_untrimmed = canonical_untrimmed if self.context_composition in NATIVE_CAUSAL_SLOTS else visible_untrimmed
        images = []
        for index, encoded in enumerate(images_b64):
            try:
                data = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise PolicySessionError("tool returned invalid image_base64") from exc
            historical = canonical_trimmed.get("media_type")
            media_type = (
                str(historical) if canonical_trimmed.get("status") == "actual_historical_artifact"
                and isinstance(historical, str) and historical.startswith("image/") else "image/png"
            )
            images.append(self.artifacts.put_bytes(
                data, kind="native_tool_image", suffix=".png" if media_type == "image/png" else ".img",
                media_type=media_type,
            ))
        image_text = None
        if images and self.supports_images:
            image_label = (
                f"HISTORICAL ACTUAL observation step {result['observation_step']}"
                if name == "observe" and result.get("status") == "actual_historical_observation"
                else "REFUSED request, drawn where it was asked"
                if result.get("ok") is False
                else "CANDIDATE preview"
                if self.context_composition in NATIVE_CAUSAL_SLOTS and name == "preview_move"
                else "CURRENT ACTUAL tool observation"
                if self.context_composition in NATIVE_CAUSAL_SLOTS and name == "observe"
                else "READ-ONLY tool feedback"
            )
            returned_by = f"returned by {name} ({call_id})"
            image_text = (
                f"{image_label}: {len(images)} image(s) {returned_by}."
                if TOOL_IMAGE_CALL_LABELS in COMPOSITION_CONTEXT_CAPABILITIES.get(self.context_composition, ())
                else f"{image_label}: {len(images)} image(s)."
                if self.context_composition in NATIVE_CAUSAL_SLOTS else
                f"{len(images)} image(s) {returned_by}."
            )
        projected = {
            # Free play puts each label before its image (ora-pi/src/play/driver.ts toolImageContent).
            "image_labels": observe_image_labels(surface, name, result, len(images)) if images else None,
            "canonical": stored_untrimmed,
            "canonical_hints": retention_hints.omit_hints(stored_untrimmed, stored, rule=rule),
            "visible": visible_untrimmed,
            "visible_hints": retention_hints.omit_hints(visible_untrimmed, visible_trimmed, rule=rule),
            "images": images,
            "image_text": image_text,
        }
        self._dump_hints("tool_result", {
            "name": name, "visible": visible_untrimmed, "hints": projected["visible_hints"],
            "python_delivered_text": json.dumps(visible_trimmed, ensure_ascii=False, sort_keys=True),
        })
        return projected

    def _dump_hints(self, kind: str, record: Mapping[str, Any]) -> None:
        """Retention-hint check (ORA_PI_HINT_DUMP): what Python decided and delivers itself."""

        path = os.environ.get("ORA_PI_HINT_DUMP")
        if path:
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"kind": kind, **record}, ensure_ascii=False) + "\n")

    def _rpc_invoke(self, params: Mapping[str, Any]) -> dict[str, Any]:
        # Trusted core metadata, outside model-supplied tool arguments. This is
        # the current act's cumulative count, not a delta per tool/primitive.
        inflight = params.get("model_calls_in_turn", 0)
        if type(inflight) is not int or inflight < 0:
            raise PolicySessionError("invalid core model-call count")
        self.inflight_model_calls = inflight
        name = str(params["name"])
        call_context = dict(params["call_context"])
        call_context.update({
            "preceding_action_step": self.last_action_step,
            "preceding_action_event_id": self.last_action_event_id,
        })
        if params.get("effect_is_mutation"):
            refused = self._script_moved_refusal(name, call_context)
            if refused is not None:
                return refused

        def run() -> dict[str, Any]:
            self._tool_events = []
            try:
                try:
                    invoke = getattr(self.tools, "invoke_with_context", None)
                    code_mode = getattr(self, "_code_mode_tool", None)
                    if callable(code_mode) and code_mode(name):  # a code-mode script or workspace file (code_mode.py)
                        result = dict(self._code_mode_execute(name, params["arguments"], call_context))
                    elif callable(invoke):
                        result = dict(invoke(name, params["arguments"], call_context=call_context))
                    else:
                        result = dict(self.tools.invoke(name, params["arguments"]))
                except Exception as exc:
                    result = _tool_error(exc)
                execution = getattr(getattr(self, "tool_action_executor", None), "completed", None)
                trusted = (execution if isinstance(execution, dict)
                           and execution.get("tool_call_id") == call_context["tool_call_id"]
                           and execution.get("model_call_id") == call_context["model_call_id"] else None)
                # A terminal read (e.g. check_task) must still allow finish_episode.
                # Only this parent's real execution attempt can require a boundary.
                return {"result": result, "events": self._tool_events,
                        **({"execution": copy.deepcopy(trusted)} if trusted else {}),
                        "execution_interruption": copy.deepcopy(getattr(
                            getattr(self, "tool_action_executor", None), "interruption", None)),
                        "execution_boundary": bool(trusted and trusted.get("actions")
                            and getattr(self, "tool_action_executor", None)
                            and getattr(self.tool_action_executor, "loop", None)
                            and self.tool_action_executor.loop.environment_status.get("terminal"))}
            finally:
                self._tool_events = None

        executor = getattr(self, "tool_action_executor", None)
        if executor is not None:
            self._pending_tool_deliveries[(str(call_context["model_call_id"]),
                                           str(call_context["tool_call_id"]))] = call_context
        executed = self._once(("tools.invoke", str(call_context["model_call_id"]),
                               str(call_context["tool_call_id"])), run)
        result = executed["result"]
        runtime_metadata = result.get("_runtime", {})
        directive = runtime_metadata.get("directive") if isinstance(runtime_metadata, Mapping) else None
        if directive is None:
            directive = result.get("directive")
        if directive in {"actions_executed", "world_state_unknown", "tool_execution_interrupted", "execution_boundary"}:
            directive = None  # Only the Supervisor ledger may issue these outcomes.
        if directive == "action_submitted" and getattr(self, "tool_action_executor", None) is not None:
            result = _tool_error(PolicySessionError("legacy action submission is disabled on the synchronous surface"))
            directive = None
        projected = self._project(name, result, call_id=str(call_context["tool_call_id"]),
                                  step=int(call_context["observation_step"]))
        projected["directive"] = directive
        projected["raw_ok"] = result.get("ok", True)
        projected["runtime_events"] = executed["events"]
        if "execution" in executed:
            projected["execution"] = executed["execution"]
            if executed["execution"]["native_actions_used"] > 0:
                projected["directive"] = "actions_executed"
                directive = "actions_executed"
        if executed.get("execution_boundary") and directive != "actions_executed":
            projected["directive"] = "execution_boundary"
            directive = "execution_boundary"
        if executed.get("execution_interruption"):
            projected["directive"] = executed["execution_interruption"]["reason"]
            directive = projected["directive"]
        if executor is not None:
            projected["execution_delivery_guard"] = True
        if params.get("effect_is_mutation") and projected["raw_ok"] and directive == "action_submitted":
            action_path = self.tools.control_root / "submitted-action.json"
            if not action_path.is_file():
                raise PolicySessionError("successful mutation did not persist an action proposal")
            action = json.loads(action_path.read_text(encoding="utf-8"))
            self.submitted_action = action
            projected["action"] = {
                "artifact_ref": self.artifacts.put_json(action, kind="native_action_proposal"),
                "action_id": action.get("action_id"),
                "observation_id": self._current_observation_id(),
            }
        return projected

    def _rpc_project(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Project a result the TypeScript dispatcher produced itself (a refusal)."""

        return self._project(str(params["name"]), params["result"], call_id=str(params["tool_call_id"]),
                             step=int(params["observation_step"]))

    def _rpc_read_only_batch(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """``begin_read_only_feedback_batch``: whether the runtime aggregates this batch."""

        key = ("tools.read_only_batch", str(params["model_call_id"]), *map(str, params["tool_call_ids"]))
        begun = self._once(key, lambda: bool(self.gateway.begin_read_only_feedback_batch(params["name"], params["calls"])))
        return {"aggregated": begun}

    def _rpc_read_only_batch_run(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Execute an aggregated same-tool read-only batch, as ``_execute_aggregated_read_only_batch``.

        Every call runs without a call context (the Native path uses
        ``gateway.execute`` without one); the runtime's one composite feedback
        image goes to the last successful result; each result is then
        projected exactly like a single call's. TypeScript has already recorded
        every proposal and records the results in call order.
        """

        name = str(params["name"])
        step = int(params["observation_step"])
        calls = list(params["calls"])

        def run() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            for call in calls:
                try:
                    results.append(dict(self.gateway.execute(name, call["arguments"] or {})))
                except Exception as exc:
                    results.append(_tool_error(exc))
            feedback = dict(self.gateway.finish_read_only_feedback_batch(name, results))
            if set(feedback) - {"image_base64", "images_base64"}:
                raise PolicySessionError("read-only batch feedback contains visible fields")
            if feedback:
                target = next((result for result in reversed(results) if result.get("success") is True), None)
                if target is None:
                    raise PolicySessionError("read-only batch produced feedback without a successful result")
                target.update(feedback)
            return results

        key = ("tools.read_only_batch_run", str(params["model_call_id"]), *(str(call["tool_call_id"]) for call in calls))
        results = self._once(key, run)
        projected = []
        for call, result in zip(calls, results):
            runtime_metadata = result.get("_runtime", {})
            directive = runtime_metadata.get("directive") if isinstance(runtime_metadata, Mapping) else None
            item = self._project(name, result, call_id=str(call["tool_call_id"]), step=step)
            item["directive"] = directive if directive is not None else result.get("directive")
            item["raw_ok"] = result.get("ok", True)
            projected.append(item)
        return {"results": projected}
