"""Run explicitly supplied Python at a historical boundary, writing separate evidence."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time

from ora.agent.toolkit.interface.blobs import BlobStore, IO_ROOT
from ora.agent.toolkit.interface.gaps import require_no_gaps
from ora.agent.toolkit.interface.journal import Journal
from ora.agent.toolkit.interface.replay import Replay
from ora.agent.toolkit.interface.experiment import RecordedExperiment
from ora.agent.toolkit.interface.missing import missing_input, missing_request
from ora.agent.toolkit.interface.runtime import Runtime
from ora.agent.toolkit.interface.snapshot import open_replay
from ora.agent.toolkit.interface.types import (
    AccessDenied,
    CallContext,
    CorruptRecord,
    canonical,
    decode,
    digest,
)
from ora.agent.toolkit.lib.manifest import VERSION, CONTENT_SHA256

from .selection import select, recorded_entries, legacy_snapshot
from .services import service_session
from .progress import Progress


def live_state():
    """The state a live tool reads and writes: empty marks and holding intentions.

    The legacy boundary restores physics, not tool state, so a tool that stages a holding
    intention gets a store to stage it in. Whether the jaw holds is then the gripper
    command's to say (``move_pose._holding``).
    """
    from ora.agent.toolkit.state.store import StateStore

    access = StateStore().access(reader="rsi", writer="tool")
    access.write("tool.marks", {})
    for arm in ("left", "right"):
        access.write(f"tool.held.{arm}", None)
    return access


def recorded_state(api, source, selected):
    """The tool state a recorded replay runs on: none, unless the environment planned the run's moves.

    Such a run has no snapshot chain, so ``--snapshot`` cannot restore its state, and its tool context alone says so
    (``motion.environment_plans``). Its journal holds the state before the call, on a private copy for the tool's own
    writes. Every other run keeps the empty store it always had, which is what a sealed RoboDojo run replays against.
    """
    from ora.agent.toolkit.interface.types import NotRecorded
    from ora.agent.toolkit.state.deltas import journal_state
    from ora.agent.toolkit.state.keys import StateError
    from ora.agent.toolkit.state.store import StateStore

    tool_data = getattr(api, "tool_data", None)
    if tool_data is None or not tool_data.context().get("motion", {}).get("environment_plans"):
        return StateStore(read_only=True).access(reader="rsi")
    proposal = selected["proposal"]
    try:
        state = journal_state(source, model_call_id=proposal["model_call_id"], tool_call_id=proposal["tool_call_id"])
    except StateError as error:
        if str(error).startswith("未记录"):
            raise NotRecorded(str(error)) from error
        raise CorruptRecord(str(error)) from error
    return state.access(reader="rsi", writer="tool")


def output_directory(source: Path, requested: Path | None) -> Path:
    """A fresh disjoint directory; never reuse results or write inside the source."""
    if requested is None:
        # Keep non-run evidence out of the directory scanned by run tooling.
        workspace = (
            source.parent.parent if source.parent.name == "runs" else source.parent
        )
        root = (workspace / "toolkit-replay" / source.name).resolve()
        if root.is_relative_to(source) or source.is_relative_to(root):
            raise AccessDenied("source run and replay output must be disjoint")
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        return Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=root))
    output = requested.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise AccessDenied("source run and replay output must be disjoint")
    output.mkdir(parents=True, exist_ok=False)
    return output


def build_api(source, output, selected, events, snapshot, services, bindings):
    require_no_gaps(source, selected["tool_call_id"])
    cached_path = output / "source-calls.jsonl"
    entries = recorded_entries(source, selected, cached_path)
    if entries:
        first = entries[0]
        identity = first["identity"]
        context = CallContext.create(
            run_id=identity["run_id"],
            round_index=identity["round_index"],
            tool_call_id=identity["tool_call_id"],
            boundary=identity["boundary"],
            state=first["state_before"],
            check_task=first["request"]["identity"]["capabilities"]["check_task"],
        )
    else:
        from .selection import rows

        indices = [
            row
            for row in rows(source, "native/toolkit-io/tool-inputs.jsonl")
            if row.get("tool_call_id") == selected["tool_call_id"]
        ]
        if len(indices) > 1:
            raise CorruptRecord("ambiguous parent tool input index")
        if not indices:
            snapshot = snapshot or legacy_snapshot(source, selected, events)
            return open_replay(
                run_root=source,
                snapshot=snapshot,
                output_root=output,
                services=services,
                bindings=bindings,
            )
        context = CallContext.create(**indices[0]["context"])
        identity = context.json()
        cached_path.write_bytes(b"")
    boundary = context.json()["boundary"]
    proposal = selected["proposal"]
    if (
        boundary.get("kind") != "before_call"
        or boundary.get("call") != selected["call"]
        or any(
            boundary.get(key) != proposal.get(key)
            for key in ("seq", "tool_call_id", "model_call_id")
        )
    ):
        raise CorruptRecord(
            "recorded CallContext boundary differs from parent proposal"
        )
    # Preserve recorded simulator binding. External identity always comes from
    # the current report, never from history masquerading as a live service.
    from ora.agent.toolkit.interface.api import EXTERNAL

    bindings = dict(bindings)
    for event in entries:
        request = event["request"]["identity"]
        if request["method"] not in EXTERNAL:
            previous = bindings.setdefault(request["method"], request["binding"])
            if previous != request["binding"]:
                raise CorruptRecord("method binding changed inside parent invocation")
    evaluation_context = CallContext.create(
        **dict(context.json(), run_id=f"replay:{identity['run_id']}")
    )
    evaluation = Runtime(
        context=evaluation_context,
        store=BlobStore(output, writable=True),
        journal=Journal(output),
        adapters=services,
        bindings=bindings,
    )
    return Replay(
        context=context,
        store=BlobStore(source),
        services=services,
        bindings=bindings,
        evaluation=evaluation,
        journal_path=cached_path,
    )


def experiment(
    *,
    source: Path,
    at: str,
    script: Path | None = None,
    tool: str | None = None,
    arguments: dict | None = None,
    report: Path | None = None,
    output: Path | None = None,
    live: bool = False,
    resource_profile: str | None = None,
    execute_action: bool = False,
    motion_stack: dict | None = None,
) -> dict:
    """Execute a trusted local run(api,args) function, never a source-run program.

    This is same-process execution, not a sandbox. Scripts are responsible for
    respecting source immutability; the supplied API has no online action path.
    Errors become result.json with nonzero CLI exit status, not fake results.

    ``motion_stack`` (``--live`` only) runs the tool on another motion stack than the run's:
    ``{"tool_surface": profile, "toolkit_grants": grants, "runtime": {field: value}}``, over a copy
    of the plan's runtime.
    """
    source = source.resolve(strict=True)
    output = output_directory(source, output)
    io = output / IO_ROOT
    io.mkdir(parents=True)
    (io / "comparisons.jsonl").touch()
    result = dict(
        schema="ora.toolkit.replay-result.v1",
        status="error",
        source_run=str(source),
        output=str(output),
        mode="live" if live else "recorded",
        evidence_mode=None,
        lib=dict(version=VERSION, sha256=CONTENT_SHA256),
        started_unix_s=time.time(),
    )
    from ora.agent.toolkit.interface.protocol import audit
    result["protocol_compatibility"] = audit(source)
    result["warnings"] = list(result["protocol_compatibility"]["warnings"])
    progress = Progress(output)
    progress.start(
        "resolve", "Resolve the selected tool-call boundary and public evidence"
    )
    api = None
    alternative = False
    try:
        if live and not resource_profile:
            raise ValueError("--live requires --resource-profile")
        if not live and (resource_profile or execute_action or motion_stack):
            raise ValueError("--resource-profile, --execute-action and a stack require --live")
        selected, events, snapshot = select(source, at)
        result["selected"] = {k: v for k, v in selected.items() if k != "proposal"}
        require_no_gaps(source, selected["tool_call_id"])
        if (script is None) == (tool is None):
            raise ValueError("choose exactly one of --script or --tool")
        from ora.agent.composition.surface.tool_names import canonical as canonical_tool

        alternative = tool is not None and canonical_tool(tool) != canonical_tool(
            selected["tool"]
        )
        if alternative and arguments is None:
            raise ValueError("--args is required when selecting a different tool")
        arguments = (
            selected["proposal"]["payload"]["arguments"]
            if arguments is None
            else arguments
        )
        if not isinstance(arguments, dict):
            raise ValueError("--args / recorded arguments must be a JSON object")
        result["arguments"] = decode(canonical(arguments))
        surface = None
        if tool is None:
            code = script.read_bytes()
            result["script"] = dict(path=str(script.resolve()), sha256=digest(code))
        else:
            from ora.agent.composition.surface.tool_names import (
                canonical as canonical_tool,
            )
            from ora.agent.toolkit.host.runtime.dispatch import TOOLS

            tool = canonical_tool(tool)
            if tool not in TOOLS:
                raise ValueError("--tool supports " + ", ".join(sorted(TOOLS)))
            from .boundary import run_surface

            surface, names = run_surface(source)
            if alternative and tool != "check_task":
                if names is None:
                    from ora.agent.toolkit.interface.types import NotRecorded

                    raise NotRecorded("未记录: frozen run tool surface unavailable")
                if tool not in names:
                    raise AccessDenied(
                        "tool was not available in this run tool surface"
                    )
            result["tool"] = tool
        with ExitStack() as stack:
            services, bindings = stack.enter_context(service_session(report))
            api = build_api(
                source, output, selected, events, snapshot, services, bindings
            )
            if (
                live
                or tool is not None
                or (source / "native/toolkit-io/tool-inputs.jsonl").is_file()
            ):
                from .tool_inputs import attach

                result["tool_inputs"] = attach(api, source, output, selected, bindings)
            publications = None
            if live and surface is None:
                from .boundary import run_surface

                surface, _ = run_surface(source)
            if alternative or live:
                from .boundary import complete_configuration

                publications = complete_configuration(
                    api, source, selected, events, surface, tool, arguments
                )
            original = api
            if live:
                from ora.orchestrator.backends.registry import backend
                from .live_boundary import freeze
                from .live_observation import prepare
                from .live_host import assemble, interruptible
                from .live_api import Live

                if surface is None:
                    from .boundary import run_surface

                    surface, _ = run_surface(source)
                if surface is None:
                    from ora.agent.toolkit.interface.types import NotRecorded

                    raise NotRecorded("live requires the frozen tool surface")
                boundary = freeze(source, output, selected, events)
                plan = decode((source / "plan.json").read_bytes())
                factory = backend(plan["runtime"]["kind"]).toolkit_live
                if factory is None:
                    raise AccessDenied(
                        "backend does not support toolkit live diagnostics"
                    )
                result["verification"] = "verification.json"
                result["sensor_verification"] = "sensor-verification.json"
                stack.enter_context(interruptible())
                session = stack.enter_context(
                    factory(
                        source=source,
                        output=output,
                        boundary=boundary,
                        profile=resource_profile,
                        execute_action=execute_action,
                        progress=progress,
                        stack=motion_stack,
                    )
                )
                result["sensor_report"] = prepare(session, source, output, boundary)
                result["surface_compatibility"] = "surface-compatibility.json"
                host = assemble(
                    original,
                    source,
                    output,
                    boundary,
                    session,
                    surface,
                    tool,
                    warnings=result.setdefault("warnings", []),
                    stack=motion_stack,
                )
                api = Live(
                    original, session, source, output, selected, events, surface, host,
                    execute_requested=execute_action,
                )
            else:
                api = RecordedExperiment(
                    original,
                    alternative=alternative,
                    publication=None if publications is None else publications.read,
                    reconstruct_observation=alternative and tool == "observe",
                    allow_check_task=True,
                )
                api.tool_data = original.tool_data
            from .boundary import run_surface

            _, available_names = run_surface(source)
            policy_check = original.context.json()["check_task"] and (
                available_names is None or "check_task" in available_names
            )
            result["capability_override"] = dict(
                check_task=not policy_check,
                original_policy_available=bool(policy_check),
                reason=None if policy_check else "原局策略不可用",
            )
            from ora.agent.toolkit.host.numeric import evaluate

            api.numeric_evaluator = evaluate
            result["replay_semantics"] = (
                "restored_runtime"
                if live
                else ("alternative_tool" if alternative else "exact_sequence")
            )
            result["context"] = api.context.json()
            result["evidence_mode"] = (
                "recorded" if isinstance(original, Replay) else "legacy"
            )
            if isinstance(original, Replay):
                evidence = (output / "source-calls.jsonl").read_bytes()
                result["source_calls"] = dict(
                    path="source-calls.jsonl",
                    sha256=digest(evidence),
                    bytes=len(evidence),
                )
            progress.finish()
            if not live:
                for stage in ("boot_worker", "replay_controls", "verify_state"):
                    progress.skip(
                        stage,
                        "Skipped: recorded mode does not restore or run a simulator",
                    )
            progress.start(
                "run_tool",
                "Run tool code against the selected data interface and explicit perception services",
            )
            with (
                (output / "script.log").open("w") as log,
                redirect_stdout(log),
                redirect_stderr(log),
            ):
                # compile/exec avoids pycache beside user code or source evidence.
                if tool is None:
                    namespace = dict(
                        __name__="__toolkit_replay__", __file__=str(script.resolve())
                    )
                    exec(compile(code, str(script), "exec"), namespace)
                    function = namespace.get("run")
                else:
                    function = TOOLS[tool]
                if not callable(function):
                    raise ValueError("script must define run(api, args)")
                from ora.agent.toolkit.interface.assemble import assemble_rsi
                # Transport.context remains a CallContext for record identities;
                # scripts and tools always receive the classified context capability.
                public_api = assemble_rsi(
                    backend=api, state=live_state() if live else recorded_state(original, source, selected))
                result["result"] = decode(
                    canonical(function(public_api, decode(canonical(arguments))))
                )
            if live:
                result["execution"] = (
                    api.execute(output)
                    if execute_action
                    else dict(executed=False, reason="execute_action_not_requested")
                )
            progress.start(
                "compare", "Collect service comparisons computed during run_tool"
            )
            if tool is not None:
                from .visible import project

                result["model_visible"] = project(
                    source,
                    output,
                    selected,
                    events,
                    tool,
                    result["result"],
                    same_tool=not alternative,
                )
            from .selection import rows

            comparisons = rows(output, f"{IO_ROOT}/comparisons.jsonl")
            result["comparison_count"] = len(comparisons)
            progress.emit(
                "compare",
                "Comparisons saved",
                done=len(comparisons),
                total=len(comparisons),
            )
            result["next_interface_ordinal"] = getattr(api, "cursor", None)
            result["status"] = "ok"
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        from ora.agent.toolkit.interface.types import NotRecorded

        if isinstance(error, NotRecorded):
            missing = error.missing_requests
            if not missing and api is not None:
                missing = missing_input(
                    "tool_input",
                    dict(tool=tool, arguments=arguments),
                    api.context,
                    reason=str(error),
                ).missing_requests
            if not missing:
                request = dict(
                    identity=dict(
                        schema="ora.toolkit.input-request.v1",
                        method="replay_boundary",
                        parameters=dict(at=at, tool=tool),
                        observation=None,
                    ),
                    inputs={},
                )
                missing = missing_request(
                    request, kind="boundary", reason=str(error)
                ).missing_requests
            result["missing_requests"] = missing
    if getattr(api, "execution_report", None) is not None:
        result["execution"] = api.execution_report
    result.setdefault("missing_requests", getattr(getattr(api, "replay_state", api), "missing_requests", []))
    result["missing_requests_complete"] = result["status"] == "ok"
    result["proposed_actions"] = getattr(getattr(api, "replay_state", api), "proposed_actions", [])
    result["proposed_controls"] = getattr(getattr(api, "replay_state", api), "proposed_controls", [])
    # Collect diagnostics on failed physical verification as well as success.
    for name in ("live-worker/launch.json", "verification.json"):
        path = output / name
        if path.is_file():
            evidence = decode(path.read_bytes())
            result["warnings"].extend(evidence.get("warnings", []))
            result["warnings"].extend(evidence.get("replay", {}).get("warnings", []))
    progress.close(result["status"])
    result["progress"] = "progress.jsonl"
    result["timing"] = progress.timing
    result["comparisons"] = f"{IO_ROOT}/comparisons.jsonl"
    result["finished_unix_s"] = time.time()
    (output / "result.json").write_bytes(canonical(result) + b"\n")
    return result
