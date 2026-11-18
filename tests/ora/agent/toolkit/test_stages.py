"""``stage(name, **fields)`` (``ora/agent/toolkit/interface/stages.py``): a tool's note of a decision its result omits.

Nobody collecting, which is every tool-call-mode run, makes ``stage`` a no-op: the toolkit tools return what they
returned, to the byte, with their ledger and journal as before. A caller that collects (code mode, ``library:
extended``) gets the events in order, and never in a result.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.agent.toolkit.interface import stages
from ora.agent.toolkit.tools.grasp_plan import _reach, _selection as grasp_selection
from ora.agent.toolkit.tools.vision_segment._selection import select_masks
from tests.support.agent.toolkit_tools import install
from tests.support.agent.toolkit_atomic import timeline
from tests.support.agent.toolkit_recording import proposal
from tests.support.agent.toolkit_tools import make_host, use_toolkit_v5


def test_stage_is_a_no_op_unless_someone_collects() -> None:
    assert not stages.active()
    stages.stage("lost", value=1)  # nobody collects: nothing is kept, nothing raises
    with stages.collecting() as events:
        assert stages.active()
        stages.stage("first", count=2, ratio=0.5)
        stages.stage("second")
    assert events == [{"stage": "first", "fields": {"count": 2, "ratio": 0.5}}, {"stage": "second", "fields": {}}]
    assert not stages.active()
    stages.stage("late")
    assert events[-1]["stage"] == "second", "a block's list is closed when it ends"


def test_collecting_nests_resets_after_an_error_and_is_bounded() -> None:
    with stages.collecting() as outer:
        stages.stage("outer")
        with stages.collecting() as inner:
            stages.stage("inner")
        stages.stage("outer again")
        with pytest.raises(RuntimeError):
            with stages.collecting():
                raise RuntimeError("tool failed")
        assert stages.active()
    assert [event["stage"] for event in outer] == ["outer", "outer again"] and [event["stage"] for event in inner] == [
        "inner"] and not stages.active()
    with stages.collecting() as events:
        for number in range(stages.MAX_EVENTS + 10):
            stages.stage("many", number=number)
    assert len(events) == stages.MAX_EVENTS


def test_fields_are_plain_json_values() -> None:
    with stages.collecting() as events:
        stages.stage("values", array=np.array([1.5, 2.5]), scalar=np.float64(0.25), count=np.int64(3), pair=(1, 2),
                     members={"b", "a"}, bad=float("nan"), nested={"rows": [np.array([1, 2])]}, other=object)
    (event,) = events
    fields = json.loads(json.dumps(event["fields"], allow_nan=False))
    assert fields["array"] == [1.5, 2.5] and fields["scalar"] == 0.25 and fields["count"] == 3
    assert fields["pair"] == [1, 2] and fields["members"] == ["a", "b"] and fields["bad"] is None
    assert fields["nested"] == {"rows": [[1, 2]]} and fields["other"].startswith("<class")


def _pool(count: int) -> tuple[list[dict], list[dict]]:
    raw = [{"candidate_id": f"grasp-{n}", "rank": n} for n in range(count)]
    compiled = [{**item, "pregrasp_pose": {"position_m": [0, 0, 1], "orientation_xyzw": [0, 1, 0, 0]},
                 "contact_pose": {"position_m": [0, 0, 1], "orientation_xyzw": [0, 1, 0, 0]}} for item in raw]
    return raw, compiled


def _selection(order: list[str], reachable: dict[str, bool | None]) -> dict:
    def select(compiled):
        ordered = [{**next(item for item in compiled if item["candidate_id"] == name),
                    "ik_reachability": {"reachable": reachable[name]}} for name in order]
        return {"ordered": ordered, "summary": {"status": "annotated"}}

    return {"identity": {"pool_limit": 32}, "select": select}


def test_the_grasp_display_cap_records_what_it_cut_and_returns_what_it_did() -> None:
    raw, compiled = _pool(5)
    selection = _selection(["grasp-3", "grasp-0", "grasp-1", "grasp-4", "grasp-2"],
                           {"grasp-3": True, "grasp-0": True, "grasp-1": False, "grasp-4": None, "grasp-2": False})
    plain = grasp_selection.apply_candidate_selection(raw, compiled, selection, 3)
    with stages.collecting() as events:
        collected = grasp_selection.apply_candidate_selection(raw, compiled, selection, 3)
    assert collected == plain
    assert events == [
        {"stage": "display_cap", "fields": {"compiled": 5, "shown": 3}},
        {"stage": "cut_candidate", "fields": {"candidate_id": "grasp-4", "backend_rank": 4, "reachable": None}},
        {"stage": "cut_candidate", "fields": {"candidate_id": "grasp-2", "backend_rank": 2, "reachable": False}}]


def test_an_unavailable_ik_check_records_why_and_returns_what_it_did() -> None:
    _raw, compiled = _pool(2)
    arguments = dict(arm="right", bridge=None, workspace_bounds_m={"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
                     approach_axis_local=[0, 0, 1])
    plain = _reach.select_grasp_candidates(compiled, **arguments)
    with stages.collecting() as events:
        collected = _reach.select_grasp_candidates(compiled, **arguments)
    assert {key: value for key, value in collected.items() if key != "evidence"} == {
        key: value for key, value in plain.items() if key != "evidence"}
    assert collected["summary"]["status"] == "unavailable"
    assert events == [{"stage": "ik_unavailable", "fields": {"failure": "tool_unavailable"}}]


def test_the_segment_selection_records_each_mask_a_limit_left_out_and_returns_what_it_did() -> None:
    points = [[20 + 60 * i, 40] for i in range(8)]
    raw = [{"mask": np.pad(np.ones((2 * radius, 2 * radius), bool), ((40 - radius, 40 - radius), (u - radius, 500 - u - radius))),
            "query_index": query, "source_index": index, "point_uv": [u, v], "backend_score": 0.9 - 0.1 * index - 0.01 * query}
           for query, (u, v) in enumerate(points) for index, radius in enumerate((4, 8, 12))]
    plain = select_masks(raw, points_uv=points, shape=(80, 500), max_candidates=2)
    with stages.collecting() as events:
        collected = select_masks(raw, points_uv=points, shape=(80, 500), max_candidates=2)
    assert plain[1] == collected[1]
    assert [item["backend_score"] for item in plain[0]] == [item["backend_score"] for item in collected[0]]
    reasons = [event["fields"]["reason"] for event in events]
    assert {event["stage"] for event in events} == {"mask_omitted"} and reasons == ["point_limit"] * 8 + ["call_cap"] * 4
    assert collected[1]["omitted_over_point_limit_total"] == 8 and collected[1]["omitted_over_call_cap_total"] == 4
    assert all(set(event["fields"]) == {"query_index", "reason", "backend_score"} for event in events)


class Segmenter:
    backend_id, backend_version, calls = "sam3", "fixture", 0

    def configuration(self):
        return {"endpoint": "http://fake.invalid", "fixture": "stages"}

    def segment_points(self, *, rgb_u8, depth_m, points_uv):
        candidates, queries = [], []
        for query, (u, v) in enumerate(points_uv):
            queries.append({"query_index": query, "point_uv": [u, v], "raw_candidate_count": 2, "error_code": None})
            for index, radius in enumerate((12, 5)):
                mask = np.zeros(rgb_u8.shape[:2], dtype=bool)
                mask[v - radius:v + radius + 1, u - radius:u + radius + 1] = True
                candidates.append({"query_index": query, "source_index": index, "point_uv": [u, v],
                                   "backend_score": 0.9 - 0.1 * index - 0.01 * query, "mask": mask})
        return candidates, {"request_id": "fixture", "query_results": queries, "latency_ms": {"roundtrip": 1.},
                            "_service_identity": {"fixture": True}}


@contextmanager
def read_only(name, context, requested):
    assert requested == 0
    yield SimpleNamespace(recovered=False)


def _host(tmp_path, monkeypatch, name):
    host, client, _ = make_host(tmp_path / name, monkeypatch)
    timeline(host)
    host.segmentation_backend = Segmenter()
    use_toolkit_v5(host)
    host.toolkit_executor = SimpleNamespace(invocation=read_only)
    install(host)
    return host, client


def test_collecting_stage_events_changes_no_toolkit_result_and_the_events_are_the_tools_own(tmp_path, monkeypatch):
    """The same calls on two identical hosts, one inside ``collecting()``: every result equal to the byte, and the
    events only on the collecting side."""

    plain, plain_client = _host(tmp_path, monkeypatch, "plain")
    seen, seen_client = _host(tmp_path, monkeypatch, "collected")
    calls = [("vision_ground", {"camera": "head", "text": "object", "max_candidates": 1}),
             ("vision_ground", {"camera": "head", "text": "object", "max_candidates": 1}),
             ("geometry_query", {"camera": "head", "points": [[300, 190], [20, 20]], "window_px": 2}),
             ("geometry_query", {"camera": "head", "points": [[300, 190]]}),
             ("geometry_query", {"camera": "head", "region": [290, 180, 310, 200]}),
             ("vision_segment", {"camera": "head", "points_uv": [[300, 195], [302, 195]], "max_candidates": 1}),
             ("vision_segment", {"camera": "head", "points_uv": [[300, 195]]})]
    recorded: list[list[dict]] = []
    try:
        for index, (name, arguments) in enumerate(calls):
            expected = plain.invoke_with_context(name, arguments, call_context=proposal(plain, name, arguments, index=index))
            with stages.collecting() as events:
                result = seen.invoke_with_context(name, arguments, call_context=proposal(seen, name, arguments, index=index))
            assert json.dumps(result, sort_keys=True) == json.dumps(expected, sort_keys=True), (index, name)
            assert not any(key in result for key in ("stage", "stages", "events", "explain"))
            recorded.append(events)
        assert not stages.active()
        names = [[event["stage"] for event in events] for events in recorded]
        assert names[0] == ["query_cache"] and recorded[0][0]["fields"] == {"hit": False}
        assert recorded[1][0]["fields"] == {"hit": True}, "the second identical query is the kept result"
        assert recorded[2] == [{"stage": "depth_window", "fields": {"window_px": 2}}]
        assert recorded[3] == [{"stage": "depth_window", "fields": {"window_px": 3}}]
        assert recorded[4] == [], "a region has no window"
        assert {event["fields"]["reason"] for event in recorded[5]} == {"point_limit"} and len(recorded[5]) >= 2
        assert recorded[6] == []
        assert (seen.control_root / "submitted-action.json").exists() == (plain.control_root / "submitted-action.json").exists()
    finally:
        plain_client.close()
        seen_client.close()
