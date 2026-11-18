"""``latency_profile`` (``ora/orchestrator/spec/latency.py``): two named profiles over explicit switches.

``sim_benchmark``, the default, records its switches and changes nothing a benchmark plan resolves to: its runtime and
what the model reads are those of a plan without the record. ``realtime`` turns execution capture, the reach reference
and the preview pictures off; the two model-facing ones move ``model_visible_sha256``. Resolution only.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ora.orchestrator.evaluation.model_visible import model_visible_items, model_visible_sha256
from ora.orchestrator.spec import latency
from ora.orchestrator.spec.run_spec import resolve_run_spec
from ora.orchestrator.spec.values import RunSpecError
from tests.support.orchestrator.run_specs import robodojo_native_spec

ROOT = Path(__file__).resolve().parents[4]
ACTIVATION = json.loads((ROOT / "tests/data/default-expectations.json").read_text(encoding="utf-8"))
ALL_ON = {"execution_capture": True, "reach_reference": True, "preview_images": True}
ALL_OFF = {"execution_capture": False, "reach_reference": False, "preview_images": False}


def _plan(value=None, **runtime):
    spec = robodojo_native_spec("general_pickup")
    if value is not None:
        spec["latency_profile"] = value
    spec["runtime"].update(runtime)
    return resolve_run_spec(spec, root=ROOT)


@pytest.fixture(scope="module")
def benchmark_plan():
    return _plan()


@pytest.fixture(scope="module")
def realtime_plan():
    return _plan("realtime")


def test_the_selection_names_a_profile_and_its_switches():
    assert latency.select(None) == {"profile": "sim_benchmark", "switches": ALL_ON, "set_by_spec": []}
    assert latency.select("realtime") == {"profile": "realtime", "switches": ALL_OFF, "set_by_spec": []}
    chosen = latency.select({"profile": "realtime", "preview_images": True})
    assert chosen["switches"] == {**ALL_OFF, "preview_images": True} and chosen["set_by_spec"] == ["preview_images"]
    assert latency.select({"reach_reference": False})["profile"] == "sim_benchmark"


@pytest.mark.parametrize("value, message", [
    ("fast", "profile must be one of"), ({"profile": "realtime", "journal": "async"}, "unknown fields: journal"),
    ({"preview_images": "no"}, "must be true or false"), (3, "profile name or an object"),
])
def test_a_malformed_selection_is_refused(value, message):
    with pytest.raises(RunSpecError, match=message):
        latency.select(value)


def test_the_benchmark_default_is_recorded_and_changes_nothing_the_model_reads(benchmark_plan):
    assert ACTIVATION["default_latency_profile"] == "sim_benchmark"
    assert benchmark_plan["latency_profile"] == {
        "schema": latency.SCHEMA, "profile": ACTIVATION["default_latency_profile"], "switches": ALL_ON,
        "set_by_spec": [], "model_facing_off": []}
    assert benchmark_plan["runtime"]["execution_capture"]["enabled"] is True
    assert benchmark_plan["runtime"]["execution_capture"]["fps"] == 10
    without = {key: value for key, value in benchmark_plan.items() if key != "latency_profile"}
    assert model_visible_items(benchmark_plan, root=ROOT) == model_visible_items(without, root=ROOT)
    assert latency.tool_switches(benchmark_plan) is None, "a benchmark run's tool calls carry no latency_switches"


def test_naming_the_default_resolves_the_same_plan(benchmark_plan):
    named = _plan("sim_benchmark")
    assert named["latency_profile"] == benchmark_plan["latency_profile"]
    assert named["runtime"] == benchmark_plan["runtime"]
    assert named["policy_identity"] == benchmark_plan["policy_identity"]


def test_a_realtime_plan_turns_its_switches_off(realtime_plan, benchmark_plan):
    record = realtime_plan["latency_profile"]
    assert record["profile"] == "realtime" and record["switches"] == ALL_OFF
    assert record["model_facing_off"] == ["reach_reference", "preview_images"]
    capture = realtime_plan["runtime"]["execution_capture"]
    assert capture["enabled"] is False and capture["fps"] == 10, "only enabled changes"
    assert latency.tool_switches(realtime_plan) == {"reach_reference": False, "preview_images": False}
    assert (realtime_plan["policy_identity"]["model_visible_sha256"]
            != benchmark_plan["policy_identity"]["model_visible_sha256"])
    settings = model_visible_items(realtime_plan, root=ROOT)["settings"]
    assert settings["latency_model_facing_off"] == ["preview_images", "reach_reference"]


def test_capture_alone_off_keeps_the_benchmark_digest(benchmark_plan):
    """Execution capture is for people: a plan that only stops recording lets the model read the same thing."""
    plan = _plan({"profile": "realtime", "reach_reference": True, "preview_images": True})
    assert plan["runtime"]["execution_capture"]["enabled"] is False
    assert plan["latency_profile"]["model_facing_off"] == []
    assert (plan["policy_identity"]["model_visible_sha256"]
            == benchmark_plan["policy_identity"]["model_visible_sha256"])


def test_the_model_facing_switches_move_the_digest_even_with_capture_left_on(benchmark_plan):
    plan = copy.deepcopy(benchmark_plan)
    plan["latency_profile"]["model_facing_off"] = ["reach_reference"]
    assert model_visible_sha256(plan, root=ROOT) != model_visible_sha256(benchmark_plan, root=ROOT)


def test_a_runtime_that_names_capture_keeps_it_and_says_so():
    plan = _plan("realtime", execution_capture={"enabled": True, "fps": 30})
    assert plan["runtime"]["execution_capture"]["enabled"] is True
    assert plan["latency_profile"]["switches"]["execution_capture"] is True
    assert plan["latency_profile"]["set_by_spec"] == ["execution_capture"]
    off = _plan(None, execution_capture={"fps": 5, "enabled": False})
    assert off["latency_profile"]["switches"]["execution_capture"] is False
    assert off["latency_profile"]["profile"] == "sim_benchmark" and off["latency_profile"]["model_facing_off"] == []


def test_two_answers_for_capture_are_refused():
    with pytest.raises(RunSpecError, match="differ"):
        _plan({"profile": "sim_benchmark", "execution_capture": False}, execution_capture={"enabled": True})


def test_a_realtime_fps_stays_what_the_runspec_asked():
    plan = _plan("realtime", execution_capture={"fps": 5})
    assert plan["runtime"]["execution_capture"]["enabled"] is False
    assert plan["runtime"]["execution_capture"]["fps"] == 5
    assert plan["latency_profile"]["set_by_spec"] == []


def test_a_backend_that_records_nothing_unasked_keeps_its_runtime():
    spec = {"runtime": {"kind": "synthetic"}}
    assert latency.with_capture(spec, latency.select("realtime"), captures=False) is spec
    asked = {"runtime": {"kind": "synthetic", "execution_capture": {"fps": 10}}}
    assert latency.with_capture(asked, latency.select("realtime"), captures=False)["runtime"][
        "execution_capture"] == {"fps": 10, "enabled": False}
    assert latency.with_capture(asked, latency.select(None), captures=True) is asked
