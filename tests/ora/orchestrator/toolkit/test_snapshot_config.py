"""Another tool, or other arguments, at a snapshot boundary get the live per-call configuration."""
import json

from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.agent.toolkit.interface.types import canonical
from ora.agent.error_classes import ToolError
from ora.orchestrator.toolkit.snapshot_boundary import select_snapshot
from ora.orchestrator.toolkit.snapshot_config import tool_configuration
from ora.orchestrator.toolkit.snapshot_replay import experiment
from tests.support.agent.toolkit_atomic import timeline
from tests.support.agent.toolkit_snapshot import make_source
from tests.support.agent.toolkit_tools import V5_GRANTS, make_host

FROZEN = {"marker": "frozen-at-boundary", "synchronous_execution": True}
QUERY = {"camera": "head", "points": [[300, 190]]}


def recorded_source(root, monkeypatch, calls):
    """make_source with its two parent calls renamed and each one's recorded configuration."""
    make_source(root, monkeypatch)
    surface = resolve_tool_surface("robodojo_placement_toolkit_v5", toolkit_grants=V5_GRANTS).snapshot()
    (root / "plan.json").write_bytes(canonical({"policy_composition": {"tool_surface": surface}}))
    path = root / "agent-events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    proposals = [row for row in events if row["type"] == "tool_call_proposed"]
    for row, (name, arguments, _) in zip(proposals, calls):
        row["payload"] = dict(name=name, arguments=arguments)
    path.write_bytes(b"".join(canonical(row) + b"\n" for row in events))
    index = root / "native/toolkit-io/tool-inputs.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("wb") as stream:
        for at, (_, _, configuration) in enumerate(calls):
            context = select_snapshot(root, at=str(at)).context.json()
            stream.write(canonical(dict(schema="ora.toolkit.tool-inputs.v1", context=context,
                                        configuration=configuration, inputs={})) + b"\n")


def live(tmp_path, monkeypatch, arguments):
    """geometry_query on a live host whose current observation is step 1, served as a run serves it."""
    from tests.support.agent.toolkit_run import on_surface, served

    host, client, _ = make_host(tmp_path / "live", monkeypatch)
    timeline(host)
    served(on_surface(host, "robodojo_placement_toolkit_v5"))
    try:
        return host.invoke("geometry_query", arguments)
    except ToolError as error:
        return error.as_dict()
    finally:
        client.close()


def test_mark_point_boundary_replays_geometry_query_with_its_own_selection(tmp_path, monkeypatch):
    source = tmp_path / "source"
    recorded_source(source, monkeypatch, [("mark_point", {"fixture": 0}, FROZEN),
                                          ("mark_point", {"fixture": 1}, FROZEN)])
    boundary = select_snapshot(source, at="1")
    config = tool_configuration(boundary, tmp_path / "host", FROZEN, "geometry_query", QUERY)
    assert {key: config[key] for key in FROZEN} == FROZEN
    assert config["geometry_query"]["step"] == 1
    result = experiment(source=source, at="1", tool="geometry_query", arguments=QUERY)
    assert result["status"] == "ok", result
    assert result["result"] == live(tmp_path, monkeypatch, QUERY)
    # Step 1 did not exist yet before the first call: refused as the live runtime refused it then.
    early = experiment(source=source, at="0", tool="geometry_query", arguments={**QUERY, "step": 1})
    assert early["error"]["type"] == "ToolRefusal"
    assert "[0, 0]" in early["error"]["message"]
    assert experiment(source=source, at="0", tool="geometry_query", arguments=QUERY)["result"]["step"] == 0


def test_geometry_query_boundary_with_changed_step_rebuilds_the_selection(tmp_path, monkeypatch):
    source = tmp_path / "source"
    make_source(tmp_path / "probe", monkeypatch)
    selected = select_snapshot(tmp_path / "probe", at="1")
    recorded = {**FROZEN, "geometry_query": dict(
        step=1, identity=selected.context.json()["state"]["observation"],
        source_state_sha256=selected.policy_state["state_sha256"])}
    recorded_source(source, monkeypatch, [("mark_point", {"fixture": 0}, FROZEN),
                                          ("geometry_query", QUERY, recorded)])
    boundary = select_snapshot(source, at="1")
    assert tool_configuration(boundary, tmp_path / "same", recorded, "geometry_query", QUERY) is recorded
    rebuilt = tool_configuration(boundary, tmp_path / "rebuilt", recorded, "geometry_query", {**QUERY, "window_px": 1})
    assert rebuilt == recorded  # same selection, rebuilt by the live builder
    changed = {**QUERY, "step": 0}
    result = experiment(source=source, at="1", tool="geometry_query", arguments=changed)
    assert result["status"] == "ok", result
    assert result["result"]["step"] == 0
    assert result["result"] == live(tmp_path, monkeypatch, changed)


def test_boundary_without_frozen_configuration_is_a_structured_refusal(tmp_path, monkeypatch):
    source = tmp_path / "source"
    recorded_source(source, monkeypatch, [("observe", {}, {"observe": {}, "synchronous_execution": True}),
                                          ("mark_point", {"fixture": 1}, FROZEN)])
    result = experiment(source=source, at="0", tool="geometry_query", arguments=QUERY)
    assert result["status"] == "error"
    assert result["error"]["type"] == "NotRecorded"
    assert result["refusal"]["status"] == "refused"
    assert result["refusal"]["usable_boundaries"] == [{"at": 1, "tool": "mark_point"}]
    assert result["missing_requests"][0]["method"] == "tool_context"
    assert "--at 1" in result["error"]["message"]


def test_a_surface_with_point_cloud_views_finds_the_bounds_in_the_frozen_plan(tmp_path, monkeypatch):
    source = tmp_path / "source"
    recorded_source(source, monkeypatch, [("mark_point", {"fixture": 0}, FROZEN),
                                          ("mark_point", {"fixture": 1}, FROZEN)])
    plan = json.loads((source / "plan.json").read_bytes())
    plan["policy_composition"]["tool_surface"]["capabilities"].append("two_view_pointcloud_v1")
    bounds = {"x": [-.5, .5], "y": [-.4, .4], "z": [.7, 1.4]}
    plan["runtime"] = {"workspace_bounds_m": bounds}
    (source / "plan.json").write_bytes(canonical(plan))
    config = tool_configuration(select_snapshot(source, at="1"), tmp_path / "host", FROZEN, "observe", {})
    assert config["observe"]["pointcloud"] == {"cameras": ["agentview", "wrist"], "workspace_bounds_m": bounds,
                                               "resolution": [256, 256]}
