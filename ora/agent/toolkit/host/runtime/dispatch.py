"""Opt-in coordinator adaptation; portable tools never receive this host object."""

from collections.abc import Mapping
from contextlib import nullcontext
from contextvars import ContextVar
import dataclasses
import json
import uuid
from ora.agent.composition.surface import tool_names
from ora.agent.composition.surface.schema_validation import ToolSurfaceError, validate_schema_instance
from ora.agent.error_classes import schema_refusal
from ora.agent.composition.surface.toolkit_profiles import FIRST_WAVE_TOOLS
from ora.agent.toolkit.registry import load_tools
from .base import ToolError
from ora.agent.toolkit.interface import dry_run
from ora.agent.toolkit.interface.types import CallContext, canonical
from ora.agent.toolkit.interface.tool_data import ToolData
from ora.agent.toolkit.interface.journal import Journal
from ora.agent.toolkit.interface.gaps import write_gap
from ora.agent.toolkit.host.numeric import evaluate
from ora.agent.toolkit.host.lifecycle import (
    configuration as lifecycle_configuration,
    check_adapter,
    finish_adapter,
)
from ora.agent.toolkit.host.adapter import bind_runtime
from ora.agent.toolkit.host.event_index import EventIndex
from .tool_data import log_query, publish
from .inputs import freeze, index_record
from ora.agent.toolkit.host.execution import submit_adapter, reconcile_before_grasp
from ora.agent.toolkit.host.failures import invoke, service_adapter, control_adapter
from ora.agent.toolkit.host.live import LiveStore, LiveRuntime
from ora.agent.toolkit.host.recording import journal_for
from ora.agent.toolkit.host.state import ATOMIC_CAPABILITY

TOOLS = load_tools()
#: Parallel single-pixel marks of one turn share one overlay (``mark_point/_feedback.py``).
MARK_AGGREGATION_CAPABILITY = "batched_mark_overlay_v1"
#: ``robot_runtime_v2`` advertises five dotted names, what its runs were shown; a tool directory has none.
ROBOT_TOOLS = {"observation.dump_state": "observation_dump_state", "vision.inspect": "vision_inspect",
               "geometry.query": "geometry_query", "robot.read_manual": "robot_read_manual",
               "robot.submit_action": "robot_submit_action"}
#: These run on no open observation, each refusing for itself where it needs one.
UNBOUND_TOOLS = frozenset({"compute", "finish_episode", "report_issue", *ROBOT_TOOLS})
#: The attribute of ``RobotToolRuntime`` each of them replaces.
ROBOT_HANDLERS = {"observation.dump_state": "_dump_state", "vision.inspect": "_inspect", "geometry.query": "_geometry",
                  "robot.read_manual": "_manual", "robot.submit_action": "_submit"}


def object_arguments(args):
    """A call's arguments are one JSON object. Anything else is the policy's malformed call: it is refused as every
    tool's input schema refuses it (``$input has the wrong input type; expected object``), before any part of the
    runtime reads it, and never reaches a tool as a fault of its own."""
    if isinstance(args, Mapping):
        return args
    try:
        validate_schema_instance(args, {"type": "object"}, root_path="$input")
    except ToolSurfaceError as error:
        raise schema_refusal(error) from error
    raise AssertionError("a non-object passed the object schema")


class Dispatch:
    def __init__(self, host):
        self.host = host
        self.supplied = ContextVar("toolkit_parent", default={})
        self.index = EventIndex(host.control_root.parent / "agent-events.jsonl")
        self.completed = set()

    def context(self, name, *, own_call=True):
        """The parent of one tool run. ``own_call`` false is work that no proposal stands for, so there is none to find."""
        host = self.host
        supplied = self.supplied.get()
        if name in UNBOUND_TOOLS:
            # These tools remain valid without an open observation. A failed
            # recording context must not change their ordinary result.
            from ora.agent.toolkit.host.lifecycle import context_publication

            try:
                control, manifest = context_publication(host)
            except Exception as error:
                control, manifest = {}, {}
                write_gap(
                    host.control_root.parent,
                    tool_call_id=supplied.get("tool_call_id"),
                    method=name,
                    stage="context",
                    error=error,
                )
        else:
            control, _, manifest = host._current()
        try:
            if not own_call:
                parent = "unindexed-" + uuid.uuid4().hex
                return CallContext.create(
                    run_id=host.control_root.parent.name, round_index=int(manifest.get("round_index", 0)),
                    tool_call_id=parent, boundary=dict(kind="before_call", call=0, tool_call_id=parent),
                    state=dict(observation=dict(observation_id=control.get("observation_id"),
                                                observation_sha256=control.get("observation_sha256"))),
                    check_task=False,
                )
            row = self.index.select(name, supplied, self.completed)
            parent = row["tool_call_id"]
            boundary = dict(
                kind="before_call",
                call=row["call"],
                seq=row["seq"],
                tool_call_id=parent,
                model_call_id=row["model_call_id"],
            )
            run_id = row["run_id"]
        except Exception as error:
            known_parent = supplied.get("tool_call_id")
            parent = (
                known_parent
                if known_parent and known_parent not in self.completed
                else "unindexed-" + uuid.uuid4().hex
            )
            run_id = host.control_root.parent.name
            boundary = dict(kind="before_call", call=0, tool_call_id=parent)
            write_gap(
                host.control_root.parent,
                tool_call_id=known_parent or parent,
                method=name,
                stage="context",
                error=error,
            )
        return CallContext.create(
            run_id=run_id,
            round_index=int(manifest.get("round_index", 0)),
            tool_call_id=parent,
            boundary=boundary,
            state=dict(
                observation=dict(
                    observation_id=control.get("observation_id"),
                    observation_sha256=control.get("observation_sha256"),
                )
            ),
            check_task=("check_task" in host.allowed_tools
                        and host.surface.supports("faithful_check_task")),
        )

    def parent_boundary(self, context, api):
        """A proposing tool's boundary in the tool-state journal, named by its recorded proposal.

        The synchronous path refuses a call without a trusted identity; this one has run such calls all along. With no
        identity there is no boundary, so a write freezes the journal with a gap and the call goes on (``deltas.py``)."""
        value, supplied = context.json(), self.supplied.get()
        boundary = value["boundary"]
        identity = {key: supplied.get(key) or boundary.get(key) for key in ("model_call_id", "tool_call_id")}
        if not all(identity.values()):
            return nullcontext()
        return self.host.toolkit_state_store.tool_journal.boundary(
            dict(boundary, kind="parent", run_id=value["run_id"], round_index=value["round_index"], **identity),
            lambda: api.recording.ordinal)

    def run(self, name, args):
        object_arguments(args)
        if not self.host.surface.supports("toolkit_synchronous_execution_v1"):
            return self._run(name, args)
        args = dict(args)
        # The surface's mutation tools are its execution tools: the built-in set on every shipped
        # surface, plus an action tool a toolset adds and declares there (I-0005). The executor's
        # per-invocation budget stays the only way any tool reaches the robot.
        execution_tool = name in self.host.surface.mutation_tools
        legacy_visible = self.host.surface.supports("toolkit_legacy_projection_v1")
        # Each tool's budget is offline configuration (the toolkit v2 surface, which took one from the call,
        # retired on 2026-11-01).
        from ora.agent.toolkit.tools.budgets import native_action_budget
        if "native_action_budget" in args:
            raise ToolError("native_action_budget is offline configuration, not a tool argument",
                            code="invalid_arguments", field="native_action_budget", retryable=False)
        budget = native_action_budget(name)
        if type(budget) is not int or budget < 0 or (not execution_tool and budget != 0):
            raise ToolError("invalid native_action_budget", code="invalid_arguments",
                            field="native_action_budget", retryable=False)
        executor = getattr(self.host, "toolkit_executor", None)
        if executor is None:
            raise ToolError("synchronous tool execution is unavailable", code="execution_unavailable")
        context = dict(self.supplied.get())
        if not context.get("tool_call_id") or not context.get("model_call_id"):
            raise ToolError("trusted parent invocation identity is required", code="execution_unavailable")
        with executor.invocation(name, context, budget) as invocation:
            result = self._run(name, args)
            metadata = {}
            if name == "grasp_execute" and isinstance(result, dict):
                # Tool-computed semantic feedback is not execution authority.
                # The owner binds it to its receipts/publications before use.
                metadata = result.pop("_runtime", {})
                feedback = metadata.get("skill_feedback")
                if feedback is not None:
                    executor.finish_skill_feedback(result, feedback)
            if legacy_visible and execution_tool and executor.loop.plan.get("toolkit_receipt_delivery", "observation") == "observation":
                from ora.agent.toolkit.host.legacy_results import submitted
                result = submitted(name, result, metadata, execution=invocation.log.summary())
            return result

    def run_dry(self, name, args):
        """A motion tool's dry run (``interface/dry_run.py``): its plan and nothing else. It opens no execution
        invocation, and the tool's data interface has no action capability (``_run``)."""
        object_arguments(args)
        if name not in dry_run.TOOLS:
            raise ToolError(f"{name} has no dry run", code=dry_run.UNAVAILABLE, retryable=False,
                            constraints={"reason": "only the arm-moving tools plan a call before it executes",
                                         "allowed": list(dry_run.TOOLS)})
        try:
            return self._run(name, dict(args))
        except dry_run.Proposed as proposed:  # the environment plans this move: it answers for it
            from ora.agent.toolkit.host.action_preview import answer

            return answer(self.host, name, proposed)

    def mark_feedback(self, calls, results):
        """The one image of a parallel ``mark_point`` batch (``mark_point/_feedback.py``)."""
        from ora.agent.toolkit.tools.mark_point import _feedback

        return self._run("mark_point", dict(calls=list(calls), results=[
            {key: result.get(key) for key in ("success", "status")} for result in results]), entry=_feedback.run)

    def _run(self, name, args, *, entry=None):
        host = self.host
        context = self.context(name, own_call=entry is None)
        parent = context.json()["tool_call_id"]
        store = LiveStore(host.control_root.parent, parent)
        try:
            journal = journal_for(host)
        except Exception as error:
            write_gap(
                store.root, tool_call_id=parent, method=name, stage="begin", error=error
            )
            # capture sees the closed journal and runs the original adapter once.
            journal = object.__new__(Journal)
            journal.events = []
            journal._closed = True
        if name in UNBOUND_TOOLS or name == "observe":
            api = LiveRuntime(
                context=context, store=store, journal=journal, adapters={}, bindings={}
            )
            config, inputs = {}, {}
            if name == "finish_episode":
                config["lifecycle"] = lifecycle_configuration(host)
            if name in ROBOT_TOOLS:
                from ora.agent.toolkit.host.robot import configure as configure_robot

                config[ROBOT_TOOLS[name]] = configure_robot(host, name, args, api, store)
        else:
            learned = getattr(host.grasp_backend, "contact_backend", host.grasp_backend)
            api = bind_runtime(
                runtime=host,
                context=context,
                store=store,
                journal=journal,
                bindings={},
                sam3=host.segmentation_backend if name == "vision_segment" else host.grounding_backend,
                grasp=learned,
                runtime_type=LiveRuntime,
            )
            if name == "grasp_execute" and not host.surface.supports("toolkit_synchronous_execution_v1"):
                reconcile_before_grasp(host, args)
            config, inputs = freeze(host, store)
        if name == "observe":
            from ora.agent.toolkit.host.observe import (
                configuration as observation_configuration,
                adapter,
            )

            config["observe"], publications = observation_configuration(host)
            api.adapters["observation"] = control_adapter(
                adapter(host, store, publications)
            )
        if name == "vision_segment":
            from ora.agent.toolkit.host.segment import configure
            config["vision_segment"] = configure(host, args, api, store)
        if name == "mark_point":
            from ora.agent.toolkit.host.observe import adapter
            api.adapters["observation"] = control_adapter(adapter(host, store, {}))
        if name == "geometry_query":
            from ora.agent.toolkit.host.geometry import configure
            config["geometry_query"] = configure(host, args, api, store)
        synchronous = host.surface.supports("toolkit_synchronous_execution_v1")
        config["synchronous_execution"] = synchronous
        switches = getattr(host, "latency_switches", None)
        if switches:  # only a plan whose latency profile turns a model-facing switch off carries the key
            config["latency_switches"] = dict(switches)
        if host.surface.supports("toolkit_legacy_projection_v1"):
            config["legacy_visible"] = True
        if synchronous:
            # Public allowance (the observation's remaining_budget): a motion tool fits its plan to
            # min(its own budget, this), so the episode's last native actions never cut it mid-cruise.
            allowance = getattr(getattr(getattr(host, "toolkit_executor", None), "loop", None), "native_allowance", None)
            remaining = allowance.fields().get("native_actions_remaining") if allowance is not None else None
            config["episode_native_actions_remaining"] = remaining if type(remaining) is int else None
            # Inside a code-mode script the call spends the script's parent allowance: a joint-trajectory move
            # (toolkit v4/v5) fits to what it has left too. Only there: a tool-call context and the released
            # primitives' (v1-v3), whose tools never read it, carry no script_native_actions_remaining.
            from ora.agent.toolkit.host.motion import joint_trajectory_surface

            if joint_trajectory_surface(host.surface):
                nested = getattr(getattr(host, "toolkit_executor", None), "nested_allowance", None)
                nested = nested() if callable(nested) else None
                if type(nested) is int:
                    config["script_native_actions_remaining"] = nested
            from ora.agent.toolkit.interface.types import PublicData
            api.adapters["execute_action"] = lambda parameters, inputs, store: PublicData.from_json(
                host.toolkit_executor.execute_action(parameters["action"], parameters["budget"]))
        api.numeric_evaluator = evaluate
        api.adapters["check_task"] = control_adapter(check_adapter(host))
        api.adapters["finish_episode"] = control_adapter(finish_adapter(host))
        from ora.agent.toolkit.host.lifecycle import report_adapter
        from ora.agent.toolkit.host.motion import alignment_adapter, kinematics_adapter
        api.adapters["report_issue"] = control_adapter(report_adapter(host))
        api.adapters["align_plan"] = control_adapter(alignment_adapter(host))
        api.adapters["kinematics"] = control_adapter(kinematics_adapter(host))
        api.tool_data = ToolData(
            store=store,
            configuration=config,
            inputs=inputs,
            publisher=lambda *values: publish(host, *values),
        )
        api.adapters["submit_action"] = submit_adapter(host)
        from ora.agent.toolkit.host.motion import PROPOSED_ACTIONS, environment_plans, submit_adapter as submit_motion, reach_adapter

        if name == "robot.submit_action":
            from ora.agent.toolkit.host.robot import submit_adapter as submit_target

            api.adapters["submit_action"] = control_adapter(submit_target(host))
        if name in PROPOSED_ACTIONS:
            api.adapters["submit_action"] = control_adapter(submit_motion(host, name, target_ref=args.get("target_ref")))
        if name in {"move_pose", "move_to", "rotate_wrist", "rotate_pitch"}:
            api.adapters["reach_check"] = reach_adapter(host)
        # The validating probe wrapper preserves plan-handle registration and old
        # public refusal fields, while the interface records the worker projection.
        from ora.agent.toolkit.interface.projections import motion_probe

        api.adapters["probe_motion"] = service_adapter(
            lambda p, i, s: motion_probe(
                host._motion_probe(
                    {k: p["request"][k] for k in ("queries", "variant_policy")}
                )
            ),
            "grasp_proposal_failed",
        )
        for method, code in [
            ("ground", "grounding_failed"),
            ("segment_points", "segmentation_failed"),
            ("grasp_candidates", "grasp_proposal_failed"),
        ]:
            if method in api.adapters:
                api.adapters[method] = service_adapter(
                    api.adapters[method], code, target_ref=args.get("target_ref")
                )
        try:
            path = store.root / "native/toolkit-io/tool-inputs.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as stream:
                stream.write(canonical(index_record(context, config, inputs)) + b"\n")
        except Exception as error:
            write_gap(
                store.root, tool_call_id=parent, method=name, stage="store", error=error
            )
        try:
            from ora.agent.toolkit.interface.assemble import assemble_robot
            access = getattr(host, "toolkit_state_access", None)
            granted = context.json()["check_task"]
            if synchronous:
                api.adapters.pop("submit_action", None)
            public_api = assemble_robot(
                backend=api, state=access, synchronous_execution=synchronous,
                privileged_tools=("check_task",) if "check_task" in host.allowed_tools else (),
                privileged_keys=("privileged.task_check",) if granted else (),
            )
            if dry_run.requested():  # whatever the tool does, it cannot act
                from ora.agent.toolkit.host.motion import admitted

                public_api = dataclasses.replace(public_api, act=dry_run.NoActions(
                    admit=admitted if environment_plans(host.surface) else None))
            if synchronous:
                journal = host.toolkit_state_store.tool_journal
                parent_context = dict(
                    context.json()["boundary"], kind="parent",
                    run_id=context.json()["run_id"],
                    round_index=context.json()["round_index"],
                    model_call_id=self.supplied.get()["model_call_id"],
                    tool_call_id=self.supplied.get()["tool_call_id"],
                )
                with journal.boundary(parent_context, lambda: api.recording.ordinal):
                    return invoke(TOOLS[name], public_api, args)
            tool = entry or TOOLS[ROBOT_TOOLS.get(name, name)]
            if not environment_plans(host.surface):
                return invoke(tool, public_api, args)
            with self.parent_boundary(context, api):
                result = invoke(tool, public_api, args)
            log_query(host, name, result)
            return result
        finally:
            self.completed.add(parent)


def served(surface):
    """Whether the toolkit serves this surface's tools: its manifest declares ``toolkit_only_v1``, as every registered
    surface does."""
    return surface.supports("toolkit_only_v1")


def install(host):
    """Install the portable handlers of a surface the toolkit serves (``served``)."""
    from ora.agent.toolkit.host.motion import environment_plans

    if not served(host.surface):
        raise ValueError("portable tools require a toolkit-only surface")
    advertised = {d["name"] for d in host.surface.definitions}
    known = TOOLS.keys() | ROBOT_TOOLS.keys()
    if not advertised <= known or not set(host.allowed_tools) <= known:
        raise ValueError("toolkit-only surface names a tool this toolset does not have")
    from ora.agent.toolkit.host.state import bind
    from ora.agent.toolkit.host.shared.artifact_io import _atomic_json
    grants = bind(host)
    _atomic_json(host.control_root.parent / "native/toolkit-state-grants.json", grants)
    dispatch = Dispatch(host)
    host._toolkit_tools_enabled = True
    # Every tool the surface advertises runs here, and each records its own interface.
    host._toolkit_installed_tools = frozenset({*FIRST_WAVE_TOOLS, *advertised})
    host._observe = lambda args: dispatch.run("observe", args)
    host._restricted_compute = lambda args: dispatch.run("compute", args)
    host._check_task = lambda args: dispatch.run("check_task", args)
    host._finish = lambda args: dispatch.run("finish_episode", args)
    host._atomic_move_pose = lambda args: dispatch.run("move_pose", args)
    host._atomic_preview_move = lambda args: dispatch.run("preview_move", args)
    host._atomic_close_gripper = lambda args: dispatch.run("close_gripper", args)
    host._atomic_open_gripper = lambda args: dispatch.run("open_gripper", args)
    host._move_to = lambda args: dispatch.run("move_to", args)
    host._vision_ground = lambda args: dispatch.run("vision_ground", args)
    host._grasp_plan = lambda args: dispatch.run("grasp_plan", args)
    host._align_plan = lambda args: dispatch.run("align_plan", args)
    host._motion_rotate_wrist_v2 = lambda args: dispatch.run("rotate_wrist", args)
    host._motion_rotate_pitch_v2 = lambda args: dispatch.run("rotate_pitch", args)
    host._wait_ticks = lambda args: dispatch.run("wait_ticks", args)
    host._move_base = lambda args: dispatch.run("move_base", args)
    host._report_issue = lambda args: dispatch.run("report_issue", args)
    host._motion_set_gripper = lambda args: dispatch.run("set_gripper", args)
    host._motion_recover_pose_v2 = lambda args: dispatch.run("recover_pose", args)
    host._motion_return_home = lambda args: dispatch.run("return_home", args)
    host._vision_segment = lambda args: dispatch.run("vision_segment", args)
    host._mark_point = lambda args: dispatch.run("mark_point", args)
    host._geometry = lambda args: dispatch.run("geometry_query", args)
    for name in advertised & ROBOT_HANDLERS.keys():
        setattr(host, ROBOT_HANDLERS[name], lambda args, name=name: dispatch.run(name, args))
    # A tool the toolset adds under a new name has no runtime handler of its own; it runs
    # through the same dispatch (core.invoke falls back to these).
    host._toolkit_handlers = {
        name: (lambda args, name=name: dispatch.run(name, args)) for name in advertised
    }
    reset_round = host.reset_round

    def reset_tool_state(next_round_index):
        reset_round(next_round_index)
        journal = host.toolkit_state_store.tool_journal
        with journal.boundary({"kind": "owner", "operation": "reset", "round_index": next_round_index}):
            host.toolkit_state_access.write("tool.marks", {})
            for arm in ("left", "right"):
                host.toolkit_state_access.write(f"tool.held.{arm}", None)
            if host.surface.supports(ATOMIC_CAPABILITY):
                host.toolkit_state_access.write("tool.previews", {})

    host.reset_round = reset_tool_state

    def closed_skill(skill, args):
        if skill == "grasp_execute":
            return dispatch.run(skill, args)
        return refuse_unknown(skill)

    host._closed_skill = closed_skill
    original = host.invoke
    contextual = getattr(host, "invoke_with_context", None)

    def refuse_unknown(name):
        raise ToolError(
            f"unknown tool: {name}", code="unknown_tool", retryable=False,
            field="tool", received=name,
        )

    def invoke_parent(name, args, *, call_context=None):
        # Where the environment plans, the runtime's own invoke refuses a tool the surface withholds, in its own words.
        if not environment_plans(host.surface) and (name not in host.allowed_tools or name not in TOOLS):
            return refuse_unknown(name)
        object_arguments(args)  # before the runtime's own invoke reads them
        args, previewed = dry_run.split(name, args)
        if previewed:
            with dry_run.dry():
                return invoke_parent(name, args, call_context=call_context)
        token = dispatch.supplied.set(call_context or {})
        try:
            if dry_run.requested() and not environment_plans(host.surface):
                return dispatch.run_dry(name, args)  # where the environment plans, the call stops at ``Proposed``
            return (
                contextual(name, args, call_context=call_context)
                if contextual is not None
                else original(name, args)
            )
        finally:
            dispatch.supplied.reset(token)

    host.invoke = invoke_parent
    host.invoke_with_context = invoke_parent
    if host.surface.supports(ATOMIC_CAPABILITY):
        host.begin_read_only_feedback_batch = lambda name, calls: begin_mark_feedback(host, name, calls)
        host.finish_read_only_feedback_batch = lambda name, results: finish_mark_feedback(host, dispatch, name, results)
    return host


def begin_mark_feedback(host, name, calls):
    """Hold back the images of several parallel single-pixel ``mark_point`` calls; True when this batch shares one."""
    if name != "mark_point" or not host.surface.supports(MARK_AGGREGATION_CAPABILITY):
        return False
    if host._active_mark_point_batch is not None:
        raise ToolError("mark_point feedback batch is already active")
    batch = tuple(dict(call) for call in calls)
    if len(batch) < 2 or any(set(call) != {"point_id", "view", "x", "y"} for call in batch):
        return False
    host._active_mark_point_batch = batch
    return True


def finish_mark_feedback(host, dispatch, name, results):
    """At most one composite image for the batch ``begin_mark_feedback`` started."""
    if name != "mark_point":
        raise ToolError("active read-only feedback batch changed tool identity")
    calls, host._active_mark_point_batch = host._active_mark_point_batch, None
    if calls is None:
        return {}
    if len(calls) != len(results):
        raise ToolError("mark_point feedback batch result count drift")
    return dispatch.mark_feedback(calls, results)
