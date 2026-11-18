"""Registered toolkit-only construction must dispatch every tool of its cohort and refuse every other name.

On RoboDojo's file-backed toolkit surface v5 with its required grants (20 tools); the nine-tool v1 these tests were
written for was retired on 2026-10-31 and the layered 18-tool v2 on 2026-11-01.
"""

import pytest

from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.agent.toolkit.host.runtime.base import ToolError
from ora.agent.toolkit.host.runtime.factory import build_tool_runtime
from ora.agent.toolkit.host.runtime.dispatch import Dispatch
from ora.agent.memory.tool_runtime import build_recoverable_artifact_runtime
from tests.support.agent.toolkit_tools import V5_GRANTS, make_host
from tests.support.agent.toolkit_recording import proposal

PROFILE = "robodojo_placement_toolkit_v5"
TOOLS = tuple(d["name"] for d in resolve_tool_surface(PROFILE, toolkit_grants=V5_GRANTS).definitions)
#: Not in v5's cohort: the Context reader and two released tools the toolkit dropped.
FORBIDDEN = ("artifact_read", "place_execute", "servo")


@pytest.fixture
def host(tmp_path, monkeypatch):
    old, client, _ = make_host(tmp_path / "run", monkeypatch)
    initialize = ManipulationToolRuntime.__post_init__

    def fake_services(self):
        self.segmentation_backend = old.segmentation_backend
        self.grounding_backend = old.grounding_backend
        self.grasp_backend = old.grasp_backend
        initialize(self)

    monkeypatch.setattr(ManipulationToolRuntime, "__post_init__", fake_services)
    runtime = build_tool_runtime(
        control_root=old.control_root,
        manual_path=old.manual_path,
        event_path=old.event_path,
        workspace_bounds_m=old.workspace_bounds_m,
        allowed_tools=None,
        state_root=None,
        profile_id=PROFILE,
        component_overrides=None,
        record_toolkit_io=True,
        toolkit_grants=V5_GRANTS,
        align_plan_callback=old.align_plan_callback,
    )
    try:
        yield runtime
    finally:
        client.close()


@pytest.mark.parametrize("name", TOOLS)
def test_every_public_tool_enters_toolkit_dispatch(host, monkeypatch, name):
    reached = []

    def run(self, tool, args):
        reached.append(tool)
        return dict(success=True, candidates=[])

    monkeypatch.setattr(Dispatch, "run", run)
    monkeypatch.setattr(host, "_validate_public_arguments", lambda *a: None)
    monkeypatch.setattr("ora.agent.toolkit.host.runtime.core.validate_schema_instance", lambda *a: None)
    assert host._toolkit_tools_enabled is True
    assert tuple(host.allowed_tools) == TOOLS
    host.invoke_with_context(name, {}, call_context={"tool_call_id": "test"})
    assert reached == [name]


@pytest.mark.parametrize("name", FORBIDDEN)
def test_removed_tools_are_unknown_before_legacy_dispatch(host, monkeypatch, name):
    monkeypatch.setattr(Dispatch, "run", lambda *a: pytest.fail("dispatch reached"))
    before = list(host.control_root.glob("submitted-action.json"))
    with pytest.raises(ToolError) as error:
        host.invoke_with_context(name, {}, call_context={"tool_call_id": "test"})
    assert error.value.code == "unknown_tool"
    assert list(host.control_root.glob("submitted-action.json")) == before


def test_lossless_context_adds_only_its_reader_as_a_context_tool(host):
    wrapped = build_recoverable_artifact_runtime(
        {"policy": {"context_retention": "action_step_lossless_v4"}},
        run_root=host.control_root.parent,
        event_path=host.event_path,
        base_runtime=host,
    )
    names = [d["name"] for d in wrapped.definitions()]
    assert names == [d["name"] for d in host.definitions()] + ["artifact_read"]
    assert wrapped.allowed_tools == tuple(host.allowed_tools) + ("artifact_read",)
    assert set(host.allowed_tools) <= set(TOOLS)
    assert wrapped.classify_call_effect("artifact_read", {"ref": "x"}) == "read_only"


def test_uninstalled_runtime_cannot_fall_back(host):
    host._toolkit_tools_enabled = False
    with pytest.raises(ToolError, match="requires the toolkit runtime factory"):
        ManipulationToolRuntime.invoke(host, "compute", {})
