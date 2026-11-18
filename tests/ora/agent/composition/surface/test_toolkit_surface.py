"""File-backed toolkit surfaces: per-tool description files, receipts in the tool result.

Each tool's description.json describes its native tool_output receipt, and v5
delivers it so: it was written out from the layered v3 (v3's tool set plus
return_home, its action tools' descriptions rewritten for tool_output receipts,
some inputs extended or replaced, changed in place 2026-10-28). v3 and the
observation-delivery v4 were retired on 2026-10-31; v5's files are the source.

A layered surface written out as files keeps the same text: its own ``<tool>.json`` where that text is not the tool's
own, without ``resultSchema`` where the tool's own result applies, with ``"resultSchema": null`` where it has none.
"""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from ora.agent.composition.schemas.toolkit_execution import EXECUTION_TOOLS
from ora.agent.composition.surface import profile_catalog
from ora.agent.composition.surface.schema_validation import ToolSurfaceError
from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from tests.support.agent.surface_catalog import surface_of
from ora.agent.composition.surface.toolkit_surface import surface_prompt_sources
from ora.agent.toolkit.registry import tool_names
from ora.orchestrator.evaluation.model_visible import model_visible_items
from ora.orchestrator.spec.run_spec import resolve_run_spec
from tests.support.orchestrator.run_specs import robodojo_native_spec

ROOT = Path(__file__).resolve().parents[5]
GRANTS = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
FILES = "robodojo_placement_toolkit_v5"
V6 = "robodojo_placement_toolkit_v6"


def test_v5_reads_one_description_file_per_tool_and_its_prompt():
    files = resolve_tool_surface(FILES, toolkit_grants=GRANTS)
    assert files.result_schemas["return_home"] == files.result_schemas["recover_pose"]
    assert [c["slot"] for c in files.components] == [f"tool:{d['name']}" for d in files.definitions] + ["surface_prompt"]
    for name in EXECUTION_TOOLS & {d["name"] for d in files.definitions}:
        description = next(d["description"] for d in files.definitions if d["name"] == name)
        for legacy in ("status=submitted", "explicitly selects", "on the next turn", "comparison images"):
            assert legacy not in description, (name, legacy)
        assert "remaining_budget" in description


def test_every_advertised_tool_has_an_entry_and_its_description_beside_it():
    surface = resolve_tool_surface(FILES, toolkit_grants=GRANTS)
    names = {d["name"] for d in surface.definitions}
    assert names <= set(tool_names())
    refs = {c["slot"]: c["ref"] for c in surface.components}
    for name in names:
        assert refs[f"tool:{name}"] == f"ora/agent/toolkit/tools/{name}/description.json"


def test_v6_is_v5_without_the_official_check():
    previous = resolve_tool_surface(FILES, toolkit_grants=GRANTS)
    current = resolve_tool_surface(V6)
    assert [d["name"] for d in previous.definitions if d["name"] != "check_task"] == [
        d["name"] for d in current.definitions]
    assert set(previous.capabilities) - set(current.capabilities) == {"faithful_check_task"}
    with pytest.raises(ToolSurfaceError, match="requires toolkit_grants"):
        resolve_tool_surface(V6, toolkit_grants=GRANTS)


def test_a_surface_can_name_its_own_description_for_a_tool():
    refs = {c["slot"]: c["ref"] for c in resolve_tool_surface(V6).components}
    assert refs["tool:finish_episode"] == f"ora/policy/tool_surfaces/{V6}/finish_episode.json"
    assert refs["tool:move_pose"] == "ora/agent/toolkit/tools/move_pose/description.json"


def test_v6_tells_the_model_of_no_check_and_offers_it_only_giving_up():
    surface = resolve_tool_surface(V6)
    finish = next(d for d in surface.definitions if d["name"] == "finish_episode")
    assert finish["inputSchema"]["properties"]["outcome"]["enum"] == ["give_up", "abort"]
    assert "check_task" not in json.dumps(surface.definitions)
    prompt = (ROOT / "ora/policy/tool_surfaces" / V6 / "prompt.md").read_text(encoding="utf-8")
    assert "check_task" not in prompt and "native_actions" not in prompt


def test_the_default_plan_delivers_the_boundary_without_a_verdict_query():
    plan = resolve_run_spec(robodojo_native_spec("general_pickup"))
    layers = plan["policy_composition"]["prompt_layers"]["project_instructions"]
    assert "ora/policy/environments/robodojo/boundary-v3.md" in [layer["path"] for layer in layers]


def spec(baseline):
    value = robodojo_native_spec("general_pickup")
    value["policy"]["baseline"] = baseline
    value["policy"].pop("tool_surface", None)
    value["toolkit_grants"] = deepcopy(GRANTS)
    return value


def test_the_file_backed_surface_delivers_receipts_in_the_tool_output():
    files = resolve_run_spec(spec("robodojo_toolkit_v6"))
    assert files["toolkit_receipt_delivery"] == "tool_output"
    assert files["toolkit_observation_receipts"] is False
    assert files["policy"]["tool_surface"]["profile"] == FILES
    # The baseline brings keep-written context (management v10): camera images in the exchange window, and the
    # artifact reader its references need.
    assert files["policy"]["context_management"]["camera_images"] == "exchange_window_v1"
    assert files["policy"]["context_tools"] == ["artifact_read"]
    assert model_visible_items(files, root=ROOT)["embodiment"]


def test_grants_and_receipt_delivery_are_the_surfaces_own():
    with pytest.raises(ToolSurfaceError):
        resolve_tool_surface(FILES)
    with pytest.raises(ToolSurfaceError, match="tool_output only"):
        resolve_tool_surface(FILES, toolkit_grants=GRANTS, toolkit_receipt_delivery="observation")
    with pytest.raises(ToolSurfaceError, match="component overrides"):
        resolve_tool_surface(FILES, toolkit_grants=GRANTS, component_overrides={"results": "x.json"})


def _copy_surface(tmp_path, monkeypatch, prompt: str) -> Path:
    """A copy of v5 inside this tree with its own prompt (the lock is not consulted for tmp files)."""
    import shutil

    target = tmp_path / FILES
    shutil.copytree(profile_catalog.PROFILE_CATALOG[FILES].parent, target)
    (target / "prompt.md").write_text(prompt, encoding="utf-8")
    monkeypatch.setitem(profile_catalog.PROFILE_CATALOG, FILES, target / "surface.json")
    return target


def test_an_empty_surface_prompt_adds_no_layer(tmp_path, monkeypatch):
    import ora.agent.composition.surface.toolkit_surface as module

    _copy_surface(tmp_path, monkeypatch, "\n")
    monkeypatch.setattr(module, "_relative", lambda root, path: str(path))
    monkeypatch.setattr(module, "_asset", lambda root, relative, label: type(
        "Asset", (), {"data": Path(relative).read_bytes(), "sha256": "0" * 64})())
    assert surface_prompt_sources(FILES, ROOT) == []


def test_a_surface_prompt_with_text_is_a_layer(tmp_path, monkeypatch):
    import ora.agent.composition.surface.toolkit_surface as module

    _copy_surface(tmp_path, monkeypatch, "Prefer observe before any motion.\n")
    monkeypatch.setattr(module, "_relative", lambda root, path: str(path))
    monkeypatch.setattr(module, "_asset", lambda root, relative, label: type(
        "Asset", (), {"data": Path(relative).read_bytes(), "sha256": "0" * 64})())
    assert surface_prompt_sources(FILES, ROOT) == [str(tmp_path / FILES / "prompt.md")]


def _layer_paths(plan, channel):
    return [layer["path"] for layer in plan["policy_context"][channel]]


def test_v6_delivers_the_surface_and_context_prompts_after_the_planner():
    """D4: the planner and core carry no tool or context rule; the surface and the scheme bring their own."""
    plan = resolve_run_spec(spec("robodojo_toolkit_v6"))
    assert plan["policy"]["tool_surface"]["profile"] == FILES
    assert plan["policy"]["context_scheme"]["id"] == "context_keep_written"
    assert plan["policy"]["context_management"]["camera_images"] == "exchange_window_v1"
    assert _layer_paths(plan, "harness_system") == [
        "ora/policy/planners/native/policy-v6.md",
        f"ora/policy/tool_surfaces/{FILES}/prompt.md",
        "ora/policy/context_schemes/keep-written.md",
    ]
    assert f"ora/policy/tool_surfaces/{FILES}/prompt.md" not in _layer_paths(plan, "tool_manual")


def test_the_v5_surface_defaults_to_the_v6_baseline():
    value = robodojo_native_spec("general_pickup")
    value["policy"]["tool_surface"] = {"profile": FILES, "component_overrides": {}}
    value["policy"].pop("baseline", None)
    value["toolkit_grants"] = deepcopy(GRANTS)
    assert resolve_run_spec(value)["policy"]["baseline"]["id"] == "robodojo_toolkit_v6"


def test_a_bundle_with_a_context_prompt_needs_a_scheme_that_has_one(monkeypatch):
    """Every registered scheme has a prompt today, so a scheme without one is stated here."""
    from ora.agent.composition.context_scheme import load_context_scheme

    monkeypatch.setattr("ora.orchestrator.spec.defaults.load_context_scheme",
                        lambda root, name: {**load_context_scheme(root, name), "prompt": None})
    value = spec("robodojo_toolkit_v6")
    value["policy"]["context_scheme"] = "context_keep_written"
    with pytest.raises(Exception, match="select a context scheme that has one"):
        resolve_run_spec(value)


# --- flattened surfaces ------------------------------------------------------------------------------------------

def _own(name):
    return json.loads((ROOT / f"ora/agent/toolkit/tools/{name}/description.json").read_text(encoding="utf-8"))


def test_a_surface_description_may_take_the_tools_own_result_schema():
    """A flattened surface keeps its own text and leaves out a result schema the tool's own file already is."""
    surface = resolve_tool_surface("yam_direct_v1")
    refs = {c["slot"]: c["ref"] for c in surface.components}
    assert refs["tool:mark_point"] == "ora/policy/tool_surfaces/yam_direct_v1/mark_point.json"
    assert refs["result:mark_point"] == "ora/agent/toolkit/tools/mark_point/description.json"
    assert surface.result_schemas["mark_point"] == _own("mark_point")["resultSchema"]
    assert "result:observe" not in refs, "a file with its own result schema names no other"
