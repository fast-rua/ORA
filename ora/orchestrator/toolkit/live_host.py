"""Own output-only compatibility stores for the unchanged skill-sequence driver."""

from collections.abc import Mapping
from contextlib import contextmanager
from types import SimpleNamespace

from ora.agent.toolkit.interface.tool_data import ToolData
from ora.agent.toolkit.interface.types import (
    NotRecorded,
    canonical,
    decode,
)
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.agent.toolkit.host.motion import configuration as motion_configuration, profile_entry
from ora.orchestrator.paths import PROJECT_ROOT
from ora.agent.toolkit.host.runtime.tool_data import publish
from .live_observation import copy_publication
from .selection import read_local
from .surface_audit import audit


def current_motion(config, host):
    """The recorded tool configuration with this host's surface and motion in it.

    A tool reads its routes from the configuration frozen at the call, which names the run's
    own; a host on another stack has to replace them or its planner is never asked.
    """
    surface = host.surface
    fresh = dict(config, surface_id=surface.profile_id, motion=motion_configuration(host),
                 synchronous_execution=surface.supports("toolkit_synchronous_execution_v1"))
    fresh.pop("legacy_visible", None)
    if surface.supports("toolkit_legacy_projection_v1"):
        fresh["legacy_visible"] = True
    return fresh


def assemble(original, source, output, boundary, session, surface, tool, *, warnings, stack=None):
    """The tool host at the boundary; ``stack`` puts it on another motion stack than the run's."""
    config = original.tool_data.context()
    plan = decode(read_local(source, "plan.json"))
    runtime = {**plan.get("runtime", {}), **(stack["runtime"] if stack else {})}
    bounds = plan.get("runtime", {}).get("workspace_bounds_m") or config.get(
        "motion", {}
    ).get("workspace_bounds_m")
    if bounds is None:
        raise NotRecorded("live tool host requires frozen public workspace bounds")
    root, observation = copy_publication(
        source,
        output,
        boundary["snapshot"]["available_at_boundary"]["observation"]["manifest"],
    )
    control = output / "control"
    control.mkdir()
    manual = output / "diagnostic-manual.md"
    manual.write_text(
        "# Toolkit live diagnostic\n\n## Scope\nOne selected tool, no policy continuation.\n"
    )
    # Services belong exclusively to explicit service_session adapters. These
    # placeholders keep the compatibility driver from constructing any client.
    unavailable = SimpleNamespace()
    host = ManipulationToolRuntime(
        control_root=control,
        manual_path=manual,
        event_path=output / "events.jsonl",
        workspace_bounds_m=bounds,
        tool_surface_profile=stack["tool_surface"] if stack else surface["profile_id"],
        tool_surface_overrides=None if stack else plan.get("policy", {})
        .get("tool_surface", {})
        .get("component_overrides"),
        toolkit_grants=stack["toolkit_grants"] if stack else plan.get("toolkit_grants"),
        segmentation_backend=unavailable,
        grounding_backend=unavailable,
        grasp_backend=unavailable,
        align_plan_callback=session.align_plan,
        motion_probe_callback=session.probe_motion,
        motion_planner_available=isinstance(runtime.get("motion_planner"), Mapping),
        motion_profile=profile_entry(runtime.get("motion_profile"), root=PROJECT_ROOT),
    )
    compatibility = audit(surface, host.surface)
    (output / "surface-compatibility.json").write_bytes(
        canonical(compatibility) + b"\n"
    )
    if compatibility["warning"] is not None:
        warnings.append(compatibility["warning"])
    observed = original.context.json()["state"]["observation"]
    state = decode(read_local(source, boundary["state_ref"]["path"]))
    pointer = dict(
        observed,
        open=True,
        source_root=str(root),
        observation_manifest=str(root / "observation.json"),
        generation=config.get("execution", {}).get("generation", state["step"]),
    )
    (control / "current-observation.json").write_bytes(canonical(pointer))
    if stack:
        config = current_motion(config, host)
    inputs = decode(original.tool_data.inputs)
    # Only frozen input artifacts are materialized; never scan the source's
    # final cache. Perception queries will publish fresh outputs of their own.
    for identity, item in inputs.items():
        kind, key = identity.split(":", 1)
        if "error" in item or kind not in {"target", "grasp_query"}:
            continue
        if kind == "grasp_query" and tool != "grasp_execute":
            continue
        publish(
            host,
            kind,
            key,
            {name: original.read_blob(ref) for name, ref in item["files"].items()},
            item["metadata"],
        )
    original.tool_data = ToolData(
        store=original.evaluation.store,
        configuration=config,
        inputs=inputs,
        publisher=lambda *args: publish(host, *args),
    )
    return host


@contextmanager
def interruptible():
    """Turn process termination into normal unwinding of the diagnostic lease."""
    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
