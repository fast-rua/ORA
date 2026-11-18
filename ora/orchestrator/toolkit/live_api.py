"""Toolkit live assembly after restoration: fresh queries, separate action proposals."""

from ora.agent.toolkit.interface.api import EXTERNAL, Methods
from ora.agent.toolkit.interface.experiment import RecordedExperiment
from ora.agent.toolkit.interface.runtime import Runtime, request_wire
from ora.agent.toolkit.interface.projections import (
    alignment,
    control_result,
    kinematics,
    motion_probe,
    reach,
    receipt,
)
from ora.agent.toolkit.interface.types import (
    AccessDenied,
    ActionNotExecuted,
    CallContext,
    NotRecorded,
    PublicData,
    canonical,
    digest,
    decode,
)
from ora.contracts import validate_runtime_action

from .boundary import Publications


class Live(Methods):
    """One validated proposal may run after tool code. A synchronous tool's action runs in it.

    A proposal waits for the tool to return. A synchronous tool (toolkit v4 and later) needs
    its receipt to go on, so ``execute_action`` runs at once, with ``execute_requested``.
    Tools retain the original published sensor bytes; fresh renders are diagnostic;
    new probes and official status come exclusively from the restored worker.
    Perception reuses the external request/comparison path, never its old output.
    """

    def __init__(
        self, original, session, source, output, selected, events, surface, host,
        *, execute_requested=False,
    ):
        self.allow_check_task = True
        self.execute_requested = execute_requested
        self.executed = []
        self.objects_before = None
        self.context, self.store = original.context, original.evaluation.store
        self.evaluation = original.evaluation
        self.tool_data = original.tool_data
        self.external = RecordedExperiment(original, alternative=True)
        self.proposed_actions, self.proposed_controls = [], []
        self.session, self.host = session, host
        self.publications = Publications(original, source, selected, events, surface)
        self.bindings = original.bindings
        context = CallContext.create(
            **dict(
                self.context.json(),
                run_id="live:" + self.context.json()["run_id"],
                check_task=True,
            )
        )

        def publication(method):
            def read(parameters, inputs, store):
                request = request_wire(
                    method,
                    parameters,
                    inputs,
                    context=self.context,
                    bindings=self.bindings,
                )
                response = self.publications.read(method, request).json()
                return PublicData.from_json(
                    response["data"], sources=response["provenance"]["sources"]
                )

            return read

        adapters = dict(
            observation=publication("observation"),
            proprioception=publication("proprioception"),
            reach_check=lambda p, i, s: reach(session.align_plan(p["request"])),
            align_plan=lambda p, i, s: alignment(session.align_plan(p["request"])),
            kinematics=lambda p, i, s: kinematics(session.align_plan(p["request"])),
            probe_motion=lambda p, i, s: motion_probe(
                session.probe_motion(p["request"])
            ),
            check_task=lambda p, i, s: PublicData.from_json(self.check()),
            receipt=self._receipt_adapter,
            submit_action=self.propose,
            execute_action=self.execute_now,
            report_issue=lambda p, i, s: self.control("report_issue", p["issue"]),
            finish_episode=lambda p, i, s: self.control("finish_episode", p["request"]),
        )
        self.runtime = Runtime(
            context=context,
            store=self.store,
            journal=original.evaluation.journal,
            adapters=adapters,
            bindings=self.bindings,
        )

    def _invoke(self, method, parameters, inputs=None):
        if method in EXTERNAL:
            return self.external._invoke(method, parameters, inputs)
        return self.runtime._invoke(method, parameters, inputs)

    def tool_artifact(self, *, kind, key, cache=False):
        if cache and kind in {"grounding_query", "grasp_query", "segmentation_query"}:
            return None
        return super().tool_artifact(kind=kind, key=key, cache=cache)

    def read_blob(self, ref):
        return self.store.read(ref)

    def save_array(self, value):
        return self.store.put_array(value)

    def check(self):
        status = self.session.status()
        success = bool(
            status["terminal"] and status.get("reason") == "official_success"
        )
        return dict(
            success=True,
            task_success=success,
            verdict=dict(task_success=success, terminal=bool(status["terminal"])),
        )

    def _receipt_adapter(self, parameters, inputs, store):
        previous = self.host._current()[2].get("previous_execution_receipt")
        if (
            previous is None
            or previous["receipt"]["action_id"] != parameters["action_id"]
        ):
            raise NotRecorded("receipt is not the completed boundary receipt")
        return receipt(previous["receipt"])

    def propose(self, parameters, inputs, store):
        action = validate_runtime_action(parameters["action"])
        if action["action"] == "execute_skill_sequence":
            from ora.agent.toolkit.host.execution import submit_adapter

            answer = submit_adapter(self.host)(parameters, inputs, store)
        elif action["action"] == "move_pose":
            from ora.agent.toolkit.host.motion import submit_adapter

            answer = submit_adapter(self.host, "move_pose")(parameters, inputs, store)
        else:
            raise AccessDenied(
                "live executor accepts migrated motion and grasp skill actions only"
            )
        request = request_wire(
            "submit_action",
            parameters,
            inputs,
            context=self.runtime.context,
            bindings=self.bindings,
        )
        self.proposed_actions.append(
            dict(
                action=decode(canonical(action)),
                request=request,
                semantic_request_hash=digest(canonical(request["identity"])),
                executed=False,
                compared=False,
            )
        )
        return answer

    def execute_now(self, parameters, inputs, store):
        """A synchronous tool's action, under its budget, with the envelope a live episode returns."""
        from ora.orchestrator.episode.action_executor import execute_runtime_action
        from ora.orchestrator.episode.native_allowance import receipt_native_actions

        action, budget = validate_runtime_action(parameters["action"]), parameters["budget"]
        if not self.execute_requested:
            raise ActionNotExecuted("execute_action", decode(canonical(action)), budget)
        if not self.executed:
            self.objects_before = self.session.diagnostic("state", objects=True).get("objects")
        call = self.context.json()
        with self.session.execution_state_owner.limited_action(
            budget, {"tool_call_id": call["tool_call_id"], "tool": "live_diagnostic"}
        ):
            raw = execute_runtime_action(self.session, action)
        self.executed.append(raw)
        used = receipt_native_actions(raw) or 0
        public = decode(receipt(raw.get("receipt", raw)).data)
        return PublicData.from_json(dict(
            receipt=dict(public, native_actions=used, action_id=action["action_id"]),
            remaining_budget=budget - used,
        ))

    def control(self, method, value):
        self.proposed_controls.append(
            dict(method=method, request=value, executed=False)
        )
        if method == "finish_episode":
            return control_result(
                dict(
                    success=True,
                    outcome=value["outcome"],
                    _runtime=dict(directive="episode_finished"),
                )
            )
        return control_result(dict(accepted=True))

    def execute(self, output):
        """Trusted host-only continuation after tool return and explicit execute flag."""
        if self.executed:
            return self._measured(output, dict(attempted=True, executed=True, receipts=self.executed))
        if not self.proposed_actions:
            return dict(executed=False, reason="no_action_proposed")
        from ora.orchestrator.episode.action_executor import execute_runtime_action
        from ora.orchestrator.episode.skill_sequence_job import run_skill_sequence

        action = self.proposed_actions[0]["action"]
        current = decode(
            (self.host.control_root / "current-observation.json").read_bytes()
        )
        report = self.execution_report = dict(attempted=True, executed=None)
        error = None
        try:
            if action["action"] == "execute_skill_sequence":
                raw, _ = run_skill_sequence(
                    self.session,
                    action,
                    tools=self.host,
                    current=current,
                    observation_root=self.session.output_root.parent
                    / "post-observations",
                    main_camera="head",
                    record=lambda event: self._event(output, event),
                )
            else:
                raw = execute_runtime_action(self.session, action)
            self.proposed_actions[0]["executed"] = True
            report.update(executed=True, receipt=raw)
        except (Exception, KeyboardInterrupt, SystemExit) as caught:
            error = caught
            self.proposed_actions[0]["executed"] = None
            report["error"] = dict(type=type(caught).__name__, message=str(caught))
        self._measured(output, report)
        if error is not None:
            raise error
        return report

    def _measured(self, output, report):
        """A refused or partly executed action still deserves measured evidence."""
        for name, read in (
            ("state", lambda: self.session.diagnostic("state", objects=self.objects_before is not None)),
            ("check_task", self.check),
            (
                "observation",
                lambda: self.session.observe(
                    self.session.output_root.parent / "final-observations"
                )[0],
            ),
        ):
            try:
                report[name] = read()
            except Exception as caught:
                report[name] = dict(
                    status="unavailable",
                    error=dict(type=type(caught).__name__, message=str(caught)),
                )
        report["objects_before"] = self.objects_before
        return report

    @staticmethod
    def _event(output, event):
        with (output / "execution-events.jsonl").open("ab") as stream:
            stream.write(canonical(dict(event)) + b"\n")
