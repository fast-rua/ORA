"""Registered construction of canonical policy-visible tool runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ora.agent.composition import resolve_tool_surface
from .core import ManipulationToolRuntime
from .base import RobotToolRuntime

from .protocol import ToolRuntime


def build_tool_runtime(
    *,
    control_root: Path,
    manual_path: Path,
    event_path: Path,
    workspace_bounds_m: Mapping[str, list[float]],
    allowed_tools: Sequence[str] | None,
    state_root: Path | None,
    profile_id: str,
    component_overrides: Mapping[str, Any] | None,
    prerequisite_binding: Mapping[str, Any] | None = None,
    controlled_end_effector: str | None = None,
    motion_probe_callback: (
        Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
    ) = None,
    align_plan_callback: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    motion_planner_available: bool = False,
    motion_profile: Mapping[str, Any] | None = None,
    public_facts_projector: Callable[..., Mapping[str, Any]] | None = None,
    record_toolkit_io: bool = False,
    toolkit_state_store: Any = None,
    toolkit_grants: Mapping[str, Any] | None = None,
    toolkit_receipt_delivery: str | None = None,
    latency_switches: Mapping[str, bool] | None = None,
) -> ToolRuntime:
    surface = resolve_tool_surface(profile_id, component_overrides=component_overrides,
                                   toolkit_grants=toolkit_grants,
                                   toolkit_receipt_delivery=toolkit_receipt_delivery)
    runtime_type = {
        "robot_runtime": RobotToolRuntime,
        "native_atomic_causal": ManipulationToolRuntime,
    }.get(surface.runtime_id)
    if runtime_type is None:
        raise ValueError(f"unregistered tool runtime: {surface.runtime_id}")
    runtime_arguments: dict[str, Any] = dict(
        control_root=control_root,
        manual_path=manual_path,
        event_path=event_path,
        workspace_bounds_m=workspace_bounds_m,
        allowed_tools=allowed_tools,
        state_root=state_root,
        tool_surface_profile=profile_id,
        tool_surface_overrides=component_overrides,
        toolkit_grants=toolkit_grants,
        toolkit_receipt_delivery=toolkit_receipt_delivery,
        controlled_end_effector=controlled_end_effector,
        motion_probe_callback=motion_probe_callback,
        align_plan_callback=align_plan_callback,
        latency_switches=latency_switches,
    )
    if runtime_type is ManipulationToolRuntime:
        runtime_arguments["motion_planner_available"] = motion_planner_available
        runtime_arguments["motion_profile"] = motion_profile
        runtime_arguments["prerequisite_binding"] = prerequisite_binding
        runtime_arguments["public_facts_projector"] = public_facts_projector
    if type(record_toolkit_io) is not bool:
        raise ValueError("record_toolkit_io must be true or false")
    if motion_profile is not None and runtime_type is not ManipulationToolRuntime:
        raise ValueError("a motion profile requires the manipulation tool runtime")
    if record_toolkit_io and runtime_type is not ManipulationToolRuntime:
        raise ValueError("record_toolkit_io requires the manipulation tool runtime")
    runtime = runtime_type(**runtime_arguments)
    if toolkit_state_store is not None:
        runtime.toolkit_state_store = toolkit_state_store
    if record_toolkit_io:
        from ora.agent.toolkit.host.recording import install

        install(runtime)
    from .dispatch import served

    if served(surface):
        from .dispatch import install as install_tools

        install_tools(runtime)
    return runtime
