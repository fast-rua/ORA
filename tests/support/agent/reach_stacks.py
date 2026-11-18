"""The toolkit's reach code under the names the reach tests call, in the shape the tests were written for.

A reach test hands a function a host with a resolved surface, and the environment's ``align_plan_callback``. A tool sees
neither. It has the frozen context (``tools/move_pose/_context.py``) and reaches the bridge through the host's
``reach_adapter``. Each name below puts a call into that shape and leaves every other name as the toolkit module has
it; the reach tests (``tests/ora/agent/toolkit/host/reach/``) import them.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from ora.agent.toolkit.tools.grasp_plan import _jaw, _reach as grasp_reach
from ora.agent.toolkit.tools.grasp_plan import tool as grasp_tool
from ora.agent.toolkit.tools.move_pose import _context, _orientation, _reach as move_reach, _reference
from ora.agent.toolkit.host.reach.alignment import frame_local_axes
from ora.agent.toolkit.host.reach.jaw_center import jaw_center_offsets
from ora.agent.toolkit.host.motion import reach_adapter


def through_adapter(bridge: Any) -> Any:
    """``bridge`` as a tool reaches it: what the host's adapter passes on, and what it raises."""

    if bridge is None:
        return None
    adapter = reach_adapter(SimpleNamespace(_current=lambda: None, align_plan_callback=bridge))
    return lambda request: json.loads(adapter({"request": request}, {}, None).data)


def _context_of(host: Any) -> SimpleNamespace:
    surface, view = host.surface, host.surface.view_config or {}
    return SimpleNamespace(
        surface=_context.Surface(tuple(surface.capabilities), dict(view),
                                 frame_local_axes(str(view.get("controlled_frame_convention"))),
                                 jaw_center_offsets(view)),
        workspace_bounds_m=host.workspace_bounds_m, align_plan_callback=through_adapter(host.align_plan_callback))


class _Module:
    """One toolkit module; a subclass's methods take a call in the handlers' shape."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


class _MoveReach(_Module):
    def plan_target(self, host: Any, *args: Any, **kwargs: Any) -> Any:
        return move_reach.plan_target(_context_of(host), *args, **kwargs)

    def primitive_reference(self, host: Any, *args: Any, **kwargs: Any) -> Any:
        return move_reach.primitive_reference(_context_of(host), *args, **kwargs)


class _Orientation(_Module):
    def nearest_orientation(self, *args: Any, bridge: Any, **kwargs: Any) -> Any:
        return _orientation.nearest_orientation(*args, bridge=through_adapter(bridge), **kwargs)


class _GraspReach(_Module):
    def check_queries(self, *args: Any, bridge: Any, **kwargs: Any) -> Any:
        return grasp_reach.check_queries(*args, bridge=through_adapter(bridge), **kwargs)

    def select_grasp_candidates(self, *args: Any, bridge: Any, **kwargs: Any) -> Any:
        return grasp_reach.select_grasp_candidates(*args, bridge=through_adapter(bridge), **kwargs)

    def grasp_candidate_selection(self, host: Any, arm: str, observation: Any = None) -> Any:
        surface, view = host.surface, host.surface.view_config or {}
        convention = view.get("controlled_frame_convention")
        config = dict(reach_annotation=surface.supports(grasp_reach.GRASP_ANNOTATION_CAPABILITY),
                      jaw_symmetry=surface.supports(_jaw.CAPABILITY), controlled_frame=convention,
                      workspace_bounds_m=host.workspace_bounds_m, approach_axis=frame_local_axes(str(convention))["approach"])
        return grasp_tool.selection(None, config, arm, SimpleNamespace(manifest=SimpleNamespace(json=lambda: observation)))


#: Each name a reach test module binds, and the toolkit module behind it.
move_pose_reach = _MoveReach(move_reach)
reach_orientation = _Orientation(_orientation)
reach_reference = _reference
policy_reach = _GraspReach(grasp_reach)
symmetry = _jaw
