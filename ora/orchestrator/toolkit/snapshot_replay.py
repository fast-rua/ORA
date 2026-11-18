"""Phase-one completed-snapshot experiments, separate from legacy diagnostics."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
import time

from ora.agent.toolkit.interface.blobs import IO_ROOT
from ora.agent.toolkit.interface.journal import read_events
from ora.agent.toolkit.interface.types import CallContext, CorruptRecord, NotRecorded, canonical, decode, digest
from ora.agent.toolkit.interface.missing import missing_input, missing_request
from ora.agent.toolkit.lib.manifest import VERSION, CONTENT_SHA256
from .progress import Progress
from .replay import output_directory
from .selection import read_local, rows
from .services import service_session
from .snapshot_api import build
from .snapshot_boundary import select_snapshot


def inputs_at(boundary, output):
    """Only the exact parent index; no future table or eventual artifact scan."""
    if boundary.selected is None:
        return None, None, []
    expected = boundary.context.json()
    matches = [row for row in rows(boundary.source, f"{IO_ROOT}/tool-inputs.jsonl")
               if row.get("context", {}).get("boundary") == expected["boundary"]]
    if len(matches) > 1:
        raise CorruptRecord("ambiguous snapshot tool input index")
    config, inputs = None, None
    if matches:
        record = matches[0]
        if record.get("schema") != "ora.toolkit.tool-inputs.v1" or any(
                record["context"].get(key) != expected[key]
                for key in ("run_id", "round_index", "tool_call_id")) or record["context"].get("state", {}).get("observation") != expected["state"]["observation"]:
            raise CorruptRecord("snapshot tool input identity differs from selected boundary")
        # Use the original request identity for service comparisons; RSI's
        # separate privileged capability is not a live policy grant.
        boundary.context = CallContext.create(**record["context"])
        config, inputs = record["configuration"], record["inputs"]
        (output / "source-tool-inputs.json").write_bytes(canonical(record) + b"\n")
    path = boundary.source / IO_ROOT / "calls.jsonl"
    history = []
    if path.is_file():
        cached = output / "source-calls.jsonl"
        cached.write_bytes(read_local(boundary.source, f"{IO_ROOT}/calls.jsonl"))
        history = [row for row in read_events(cached)
                   if row["identity"]["boundary"] == expected["boundary"]
                   and row["identity"]["run_id"] == expected["run_id"]]
    return config, inputs, history


def experiment(*, source: Path, at=None, step=None, script=None, tool=None,
               arguments=None, report=None, output=None, native_action=None, resource_profile=None):
    """Read a completed snapshot and write independent replay evidence.

    No action proposal is accepted. Only native-action selection starts a
    diagnostic simulator, closed before script execution. Scripts
    run in this process; the API is a capability contract, not a Python sandbox.
    """
    source = source.resolve(strict=True)
    output = output_directory(source, output)
    io = output / IO_ROOT
    io.mkdir(parents=True)
    (io / "comparisons.jsonl").touch()
    progress = Progress(output)
    result = dict(schema="ora.toolkit.replay-result.v1", status="error", mode="snapshot",
                  source_run=str(source), output=str(output), started_unix_s=time.time(),
                  lib=dict(version=VERSION, sha256=CONTENT_SHA256),
                  replay_semantics="completed_snapshot", proposed_actions=[], proposed_controls=[])
    from ora.agent.toolkit.interface.protocol import audit
    result["protocol_compatibility"] = audit(source)
    result["warnings"] = list(result["protocol_compatibility"]["warnings"])
    boundary = None
    try:
        progress.start("resolve", "Verify completed snapshot and exact before-parent tool state")
        if (script is None) == (tool is None):
            raise ValueError("choose exactly one --script or --tool")
        from ora.agent.composition.surface.tool_names import canonical as tool_name
        if tool is not None:
            tool = tool_name(tool)
            from ora.agent.toolkit.host.runtime.dispatch import TOOLS
            if tool not in TOOLS:
                raise ValueError("unsupported toolkit tool")
        if arguments is not None and not isinstance(arguments, dict):
            raise ValueError("--args / original arguments must be a JSON object")
        if native_action is not None:
            if step is not None or at is not None:
                raise ValueError("native-action cannot mix with another boundary selector")
            if arguments is None and tool is not None:
                raise ValueError("--args is required without original parent arguments")
            if tool is not None and tool not in {"compute", "check_task"}:
                request = dict(identity=dict(schema="ora.toolkit.input-request.v1",
                                             method="tool_context", parameters={"tool": tool},
                                             observation=None), inputs={})
                raise missing_request(request, kind="tool_input",
                                      boundary=dict(kind="after_native_action", native_action=native_action),
                                      reason="no_tool_configuration_at_snapshot_boundary")
            from .native_snapshot import restore
            boundary = restore(source, output, native_action, resource_profile, progress)
            result["replay_semantics"] = "native_tick_fresh_render"
            result["native_action"] = native_action
            result["verification"] = "verification.json"
            result["state_evidence"] = "native-state.json"
            result["missing_state_fields"] = decode((output / "native-state.json").read_bytes())["missing_state_fields"]
        else:
            if resource_profile:
                raise ValueError("completed snapshots do not use a resource profile")
            boundary = select_snapshot(source, step=step, at=at)
            result["missing_state_fields"] = boundary.missing_state_fields
        result["selected"] = dict(step=boundary.step, boundary=boundary.context.json()["boundary"],
                                  snapshot_sha256=boundary.snapshot_sha256)
        parent = boundary.selected
        if arguments is None:
            if tool is not None and (parent is None or tool != tool_name(parent["tool"])):
                raise ValueError("--args is required without matching original parent arguments")
            arguments = {} if parent is None else parent["proposal"]["payload"]["arguments"]
        if not isinstance(arguments, dict):
            raise ValueError("--args / original arguments must be a JSON object")
        result["arguments"] = decode(canonical(arguments))
        configuration, inputs, history = inputs_at(boundary, output)
        result["context"] = boundary.context.json()
        if tool is not None:
            # The recorded configuration is the parent's; another tool or other arguments get their own.
            from .snapshot_config import tool_configuration
            configuration = tool_configuration(boundary, output / "boundary-host", configuration, tool, arguments)
        result["capability_override"] = dict(check_task=True, reason=("RSI restored simulator privilege" if native_action is not None
                                                                         else "RSI snapshot privilege; no live check"))
        with ExitStack() as stack:
            services, bindings = stack.enter_context(service_session(report))
            api, backend = build(boundary, output, services=services, bindings=bindings,
                                 configuration=configuration, inputs=inputs, history=history)
            progress.finish()
            for name in (() if native_action is not None else ("boot_worker", "replay_controls", "verify_state")):
                progress.skip(name, "Completed snapshot: no simulator startup or physical replay")
            progress.start("run_tool", "Run against snapshot state, fresh services and private tool state")
            with (output / "script.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                if script is not None:
                    code = script.read_bytes()
                    result["script"] = dict(path=str(script.resolve()), sha256=digest(code))
                    namespace = dict(__name__="__toolkit_snapshot__", __file__=str(script.resolve()))
                    exec(compile(code, str(script), "exec"), namespace)
                    function = namespace.get("run")
                else:
                    from ora.agent.toolkit.host.runtime.dispatch import TOOLS
                    if tool not in TOOLS:
                        raise ValueError("unsupported toolkit tool")
                    if configuration is None and tool not in {"compute", "check_task"}:
                        raise missing_input("tool_context", {"tool": tool}, boundary.context,
                                            reason="no_tool_configuration_at_snapshot_boundary")
                    result["tool"] = tool
                    function = TOOLS[tool]
                if not callable(function):
                    raise ValueError("script must define run(api, args)")
                result["result"] = decode(canonical(function(api, decode(canonical(arguments)))))
            progress.start("compare", "Collect external-service comparisons")
            result["comparison_count"] = len(rows(output, f"{IO_ROOT}/comparisons.jsonl"))
            result["tool_state"] = {key: value for key, value in boundary.state.access(reader="rsi").values().items()
                                     if key.startswith("tool.")}
            result["status"] = "ok"
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        if getattr(error, "refusal", None) is not None:
            result["refusal"] = error.refusal
        if isinstance(error, NotRecorded):
            result["missing_requests"] = error.missing_requests
            if not error.missing_requests and boundary is not None:
                result["missing_requests"] = missing_input(
                    "snapshot_input", {"tool": tool, "step": step, "at": at, "native_action": native_action}, boundary.context,
                    reason=str(error)).missing_requests
            elif not error.missing_requests:
                request = dict(identity=dict(schema="ora.toolkit.input-request.v1", method="snapshot_boundary",
                                             parameters=dict(at=at, step=step, native_action=native_action), observation=None), inputs={})
                result["missing_requests"] = missing_request(request, kind="boundary", reason=str(error)).missing_requests
    result.setdefault("missing_requests", [])
    result["missing_requests_complete"] = result["status"] == "ok"
    result["proposed_controls"] = rows(output, "episode-controls.jsonl")
    # Collect diagnostics on failed physical verification as well as success.
    for name in ("live-worker/launch.json", "verification.json"):
        path = output / name
        if path.is_file():
            evidence = decode(path.read_bytes())
            result["warnings"].extend(evidence.get("warnings", []))
            result["warnings"].extend(evidence.get("replay", {}).get("warnings", []))
    progress.close(result["status"])
    result.update(progress="progress.jsonl", timing=progress.timing,
                  comparisons=f"{IO_ROOT}/comparisons.jsonl", finished_unix_s=time.time())
    (output / "result.json").write_bytes(canonical(result) + b"\n")
    return result
