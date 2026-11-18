"""The interface a code-mode script's queries run over: the tools' own reads, nothing recorded.

A query (``robot.query``, ``ora/agent/code_mode/query.py``) is no tool call: it is read-only, so it reaches the
environment's read-only bridges and the published observation exactly as a toolkit tool's ``DataInterface`` does
(``adapter.bind_runtime`` with the adapters ``Dispatch._run`` gives a tool: the kinematics, alignment and motion
probe ones), but its requests enter no I/O journal, no gap file and no tool-call record. Where the host runs no toolkit
tools (a legacy surface) there is no such interface and the query is unavailable, and so it is where the environment
plans the moves: such a surface serves what it served on the handlers (``contract.with_reads``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.projections import alignment, motion_probe, reach
from ora.agent.toolkit.interface.runtime import Runtime, envelope, public_failure, request_wire
from ora.agent.toolkit.interface.types import CallContext
from ora.agent.error_classes import ToolError
from ora.contracts.refusal import EnvironmentRefusal

from .adapter import bind_runtime
from .failures import control_adapter, service_adapter
from .motion import environment_plans, kinematics_adapter, planned_motion
from .state import current

#: The context's parent identity: queries are no tool call, so none is borrowed from one.
QUERY_CALL = "code-mode-query"


class UnrecordedRuntime(Runtime):
    """The interface with every method run once through its adapter and nothing journaled."""

    def _invoke(self, method: str, parameters: dict, inputs: dict | None = None):
        request = request_wire(method, parameters, inputs or {}, context=self.current_context(),
                               bindings=self.bindings)
        try:
            return envelope(method, request, public=self.adapters[method](
                request["identity"]["parameters"], request["inputs"], self.store))
        except Exception as error:  # noqa: BLE001 - what the interface answers for any failed adapter
            status, failure = public_failure(method, error)
            return envelope(method, request, status=status, error=failure)


def _bridge_adapter(host: Any, projection: Any):
    """A read over the environment's read-only IK bridge with the request's argument refusals named
    (``alignment_adapter`` and ``reach_adapter`` of ``motion`` serve a tool, which reports the check as a
    reference or never meets a missing bridge)."""

    def query(parameters: dict, inputs: dict, store: Any):
        host._current()
        if host.align_plan_callback is None:
            raise ToolError("the environment serves no read-only IK bridge", code="tool_unavailable", retryable=False)
        try:
            return projection(host.align_plan_callback(parameters["request"]))
        except EnvironmentRefusal as error:
            raise ToolError(str(error), code=error.code, retryable=error.retryable, field=error.field,
                            received=error.received, constraints={"allowed": error.allowed}) from error

    return query


def query_api(host: Any) -> UnrecordedRuntime:
    """A ``DataInterface`` over the host's current observation for one query.

    Raises ``ToolError`` where there is none: ``query_unavailable`` on a host that runs no toolkit tools or whose
    environment plans the moves, the host's own refusal (``action_pending_fresh_observation``) while the observation is
    closed."""

    if not getattr(host, "_toolkit_tools_enabled", False) or environment_plans(host.surface):
        raise ToolError("this tool surface runs no toolkit tools, so there is no data interface to query through",
                        code="query_unavailable", retryable=False)
    identity, _root, manifest = current(host)
    context = CallContext.create(run_id=host.control_root.parent.name, round_index=int(manifest.get("round_index", 0)),
                                 tool_call_id=QUERY_CALL, boundary=dict(kind="before_call", call=0),
                                 state=dict(observation=identity), check_task=False)
    api = bind_runtime(runtime=host, context=context, store=BlobStore(host.control_root.parent), bindings={},
                       journal=SimpleNamespace(events=[], _closed=True), runtime_type=UnrecordedRuntime)
    api.planned_motion = planned_motion(host)  # the query.py probe plans with it, as the host's motion tools do
    api.adapters["kinematics"] = control_adapter(kinematics_adapter(host))
    api.adapters["align_plan"] = control_adapter(_bridge_adapter(host, alignment))
    api.adapters["reach_check"] = control_adapter(_bridge_adapter(host, reach))
    api.adapters["probe_motion"] = service_adapter(
        lambda parameters, inputs, store: motion_probe(host._motion_probe(
            {key: parameters["request"][key] for key in ("queries", "variant_policy")})), "grasp_proposal_failed")
    return api


__all__ = ["UnrecordedRuntime", "query_api"]
