"""The tool context a requested tool gets at a completed-snapshot boundary.

The live dispatch (``Dispatch.run`` in ``ora.agent.toolkit.host.runtime.dispatch``)
gives a tool the boundary's frozen configuration (``inputs.freeze``: the
same for every tool at one boundary, whatever its arguments) plus, for four
tools, a section built from that tool's own arguments and the runtime's current
observation. The parent index records the result for the tool the model actually
called, with its arguments. Replaying another tool, or the same tool with other
arguments, therefore keeps the frozen part and rebuilds the per-call section with
the live builders, over a read-only host that sees the run as it was at the
boundary. What the boundary's evidence cannot supply is refused with the
boundaries where it is recorded.
"""

import json
from types import SimpleNamespace

from ora.agent.composition.surface.tool_names import canonical as tool_name
from ora.agent.evidence.policy_state import json_sha256, verify_policy_state
from ora.agent.toolkit.interface.blobs import IO_ROOT
from ora.agent.toolkit.interface.missing import missing_input
from ora.agent.toolkit.interface.types import CorruptRecord, NotRecorded, canonical, decode, digest
from ora.agent.toolkit.host.runtime.base import RobotToolRuntime
from ora.agent.toolkit.host.runtime.observe import ObserveRuntimeMixin
from ora.agent.error_classes import ToolError
from .boundary import run_surface
from .selection import read_local, rows

# Dispatch.run's first branch: these tools get no frozen boundary configuration.
UNFROZEN = frozenset({"compute", "finish_episode", "observe", "report_issue"})
# The section Dispatch.run adds for one tool only.
CALL_SECTIONS = {"observe": "observe", "vision_segment": "vision_segment",
                 "geometry_query": "geometry_query", "finish_episode": "lifecycle"}
# Tools a snapshot replays without any recorded configuration (snapshot_replay).
NO_CONFIGURATION = frozenset({"compute", "check_task"})


def chain_view(source, root, step):
    """The policy-state chain as it stood when ``step`` was the latest publication.

    Steps up to ``step`` and their observation manifests are copied byte for byte;
    the manifest is the unfinalized one the store wrote then (context calls left
    out: only step selection reads this view). Later steps do not exist in it.
    """
    verified = verify_policy_state(source, verify_artifacts=False)
    if verified["status"] != "valid":
        raise CorruptRecord("policy-state chain is invalid: " + ", ".join(verified["errors"]))
    manifest = verified["manifest"]
    records = manifest["steps"][: step + 1]
    if len(records) != step + 1:
        raise CorruptRecord("boundary policy-state step is not in the chain")
    for record, state in zip(records, verified["steps"]):
        for name in (record["path"], state["observation"]["manifest"]["path"]):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(read_local(source, name))
    body = dict(schema=manifest["schema"], run_id=manifest["run_id"], plan_sha256=manifest["plan_sha256"],
                steps=records, context_calls=[], finalized=False, finalization=None)
    (root / "policy-state/manifest.json").write_bytes(canonical({**body, "manifest_sha256": json_sha256(body)}))
    if verify_policy_state(root, verify_artifacts=False)["status"] != "valid":
        raise CorruptRecord("boundary policy-state view does not verify")
    return root


class BoundaryHost:
    """The tool runtime as the dispatch saw it before the selected call, read-only.

    The selection methods are the live runtime's own; only the current
    observation pointer and the policy-state chain are the boundary's.
    """

    _select_observation = RobotToolRuntime._select_observation
    _retention_observation = ObserveRuntimeMixin._retention_observation
    _current_grip_state = ObserveRuntimeMixin._current_grip_state
    _physical_policy_cameras = ObserveRuntimeMixin._physical_policy_cameras
    _robotwin_policy_cameras = ObserveRuntimeMixin._robotwin_policy_cameras
    _derived_resolution = ObserveRuntimeMixin._derived_resolution

    def __init__(self, boundary, root, surface):
        capabilities = frozenset(surface["capabilities"])
        self.surface = SimpleNamespace(profile_id=surface.get("profile_id"), view_config=surface.get("view_config"),
                                       supports=capabilities.__contains__)
        self.view_config = dict(surface.get("view_config") or {})
        self.control_root = boundary.source / "control"  # tool-execution ledger for step visibility
        self.state_root = chain_view(boundary.source, root, boundary.policy_state["step"]) / "policy-state"
        self.toolkit_state_access = boundary.state.access(reader="rsi")
        observed = boundary.context.json()["state"]["observation"]
        self._control = dict(open=True, **observed, policy_state=dict(boundary.policy_state))

    @property
    def workspace_bounds_m(self):
        # Read only on a surface with point-cloud views: the frozen run plan holds the bounds.
        return decode(read_local(self.control_root.parent, "plan.json"))["runtime"]["workspace_bounds_m"]

    def _current(self):
        state = json.loads((self.state_root / "steps" / f"step-{self._control['policy_state']['step']:04d}.json")
                           .read_bytes())
        manifest = self.state_root.parent / state["observation"]["manifest"]["path"]
        return dict(self._control), manifest.parent, json.loads(manifest.read_bytes())

    def _load_points(self):
        # Pre-synchronous surfaces keep marks in control files that later calls rewrite.
        raise ToolError("the mark ledger of this surface is not frozen at a snapshot boundary",
                        code="artifact_not_found", retryable=False)


def _observe(host, arguments):
    from ora.agent.toolkit.host.observe import configuration
    if not host.surface.supports("toolkit_synchronous_execution_v1"):
        host._load_points()
    return configuration(host)[0]


def _segment(host, arguments):
    from ora.agent.toolkit.host.segment import selection
    return selection(host, arguments)[0]


def _geometry(host, arguments):
    from ora.agent.toolkit.host.geometry import selection
    return selection(host, arguments)


BUILDERS = {"observe": _observe, "vision_segment": _segment, "geometry_query": _geometry}


def usable_boundaries(source, tool):
    """Parent calls (``--at`` ordinals) whose recorded configuration serves ``tool``."""
    recorded = {row.get("context", {}).get("tool_call_id") for row in rows(source, f"{IO_ROOT}/tool-inputs.jsonl")}
    proposals = [row for row in rows(source, "agent-events.jsonl") if row.get("type") == "tool_call_proposed"]
    usable = []
    for index, proposal in enumerate(proposals):
        original = tool_name(proposal["payload"]["name"])
        if proposal["tool_call_id"] not in recorded:
            continue
        if (original == "finish_episode" if tool == "finish_episode"
                else tool in UNFROZEN or original not in UNFROZEN):
            usable.append(dict(at=index, tool=original))
    return usable


def refusal(boundary, tool, section, reason):
    usable = usable_boundaries(boundary.source, tool)
    error = missing_input("tool_context", {"tool": tool, "section": section}, boundary.context, reason=reason)
    shown = ", ".join(str(row["at"]) for row in usable[:20]) + (" …" if len(usable) > 20 else "")
    message = (f"未记录: tool_context for {tool}: {reason}; "
               + (f"boundaries with it: --at {shown}" if usable else "no boundary of this run records it"))
    error = NotRecorded(message, missing_requests=error.missing_requests)
    error.refusal = dict(status="refused", reason=reason, tool=tool, section=section, usable_boundaries=usable)
    return error


def tool_configuration(boundary, root, configuration, tool, arguments):
    """The configuration the live dispatch would hand ``tool`` with ``arguments`` here.

    ``configuration`` is the parent's recorded one (None if the boundary has no
    index row); ``root`` is a scratch directory in the replay output.
    """
    parent = boundary.selected
    original = None if parent is None else tool_name(parent["tool"])
    if original == tool and digest(canonical(parent["proposal"]["payload"]["arguments"])) == digest(canonical(arguments)):
        return configuration  # the recorded call itself
    if configuration is None:
        if tool in NO_CONFIGURATION:
            return None
        raise refusal(boundary, tool, "boundary", "no_tool_configuration_at_snapshot_boundary")
    if tool in UNFROZEN:
        config = {}
    elif original in UNFROZEN:
        raise refusal(boundary, tool, "frozen", f"boundary_configuration_not_frozen_before_{original}")
    else:
        config = {key: value for key, value in configuration.items() if key not in CALL_SECTIONS.values()}
    for key in ("synchronous_execution", "legacy_visible"):
        if key in configuration:
            config[key] = configuration[key]  # surface-wide, the same for every tool
    section = CALL_SECTIONS.get(tool)
    if section == "lifecycle":
        # Budget, pending action and abort signals at the boundary are recorded only for finish_episode.
        if original != "finish_episode":
            raise refusal(boundary, tool, section, "lifecycle_configuration_recorded_only_before_finish_episode")
        config[section] = configuration[section]
    elif section is not None:
        surface, _ = run_surface(boundary.source)
        if surface is None or boundary.policy_state is None:
            raise refusal(boundary, tool, section, "frozen_surface_or_policy_state_reference_missing")
        try:
            config[section] = BUILDERS[tool](BoundaryHost(boundary, root, surface), arguments)
        except ToolError as error:
            raise refusal(boundary, tool, section, f"{error.code}: {error}") from error
    return config
