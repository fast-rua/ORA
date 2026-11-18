"""Batched ``mark_point``: which surfaces declare it, the parallel-call aggregation and the depth estimator.

A measurement pass over 222 sealed runs found 379 model turns that issued several
``mark_point`` calls against one unchanged observation; the batched input form
(``mark_point_batch_v1``) and the parallel-call aggregation
(``batched_mark_overlay_v1``) collapse them into one turn. The aggregation test
drives the path end to end, through the Native session, on the Synthetic Native
surface ``native_atomic_causal_v1``, which declares it. The estimator tests hold
the toolkit's local-depth acceptance (``toolkit/tools/mark_point/_measurement.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG
from ora.agent.providers.fake import FakeProvider
from ora.agent.toolkit.tools.errors import ToolRefusal


ROOT = Path(__file__).resolve().parents[5]
# The aggregation path is only armed on a surface whose hash-bound manifest
# declares it, so a reachability test must name one of those surfaces.
AGGREGATION_CAPABILITY = "batched_mark_overlay_v1"
AGGREGATING_PROFILE = "native_atomic_causal_v1"
#: The batched input form (``point_ids`` / ``points_uv``) a manifest opts into.
BATCH_CAPABILITY = "mark_point_batch_v1"


def _assistant_tools(calls: Sequence[tuple[str, str, dict]]) -> dict:
    return {
        "role": "assistant", "content": "", "tool_calls": [
            {
                "id": call_id, "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, sort_keys=True),
                },
            }
            for call_id, name, arguments in calls
        ],
    }


def test_a_registered_surface_still_declares_the_aggregation_capability() -> None:
    """The gate is a manifest capability, so the manifests are the evidence."""

    declaring = sorted(
        profile
        for profile, path in PROFILE_CATALOG.items()
        if AGGREGATION_CAPABILITY in json.loads(
            path.read_text(encoding="utf-8"),
        ).get("capabilities", [])
    )
    assert declaring, (
        "no registered profile declares "
        f"{AGGREGATION_CAPABILITY}; only then is the aggregation path dead"
    )
    assert AGGREGATING_PROFILE in declaring


def test_the_native_session_reaches_the_aggregation_from_one_response(
    tmp_path: Path,
) -> None:
    """Two parallel read-only calls in one response drive the generic path."""

    from tests.support.orchestrator.native_synthetic_spec import synthetic_spec
    from ora.orchestrator.episode.coordinator import run_episode
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    spec = synthetic_spec(seed=0)
    spec["policy"]["context_retention"] = "rewrite_compact_v1"
    assert spec["policy"]["tool_surface"]["profile"] == AGGREGATING_PROFILE
    plan = resolve_run_spec(spec, root=ROOT)
    root = tmp_path / "run"
    root.mkdir()
    (root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    provider = FakeProvider([
        {"message": _assistant_tools([
            ("mark-a", "mark_point", {
                "point_id": "P0", "view": "wrist", "x": 2, "y": 1,
            }),
            ("mark-b", "mark_point", {
                "point_id": "P1", "view": "wrist", "x": 1, "y": 1,
            }),
        ])},
        {"message": _assistant_tools([
            ("finish-1", "finish_episode", {
                "outcome": "failure", "reason": "aggregation probe complete",
                "attempts": ["marked two points in one response"],
            }),
        ])},
    ])
    run_episode(run_root=root, plan=plan, run_id="mark-point-aggregation",
                provider_override=provider)

    events = [
        json.loads(line)
        for line in (root / "agent-events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    combined = [
        event for event in events
        if event.get("type") == "tool_call_proposed"
        and event.get("payload", {}).get("visual_feedback") == "combined"
    ]
    assert len(combined) == 2, (
        "both parallel mark_point calls must be routed through the "
        "aggregated read-only batch"
    )


def test_a_far_plane_pixel_reports_what_the_estimator_saw(
    tmp_path: Path,
) -> None:
    """A rejected depth still has diagnostics; the single form discards them."""

    import numpy as np

    from ora.agent.toolkit.tools.mark_point._measurement import robust_depth_measurement

    calibration = {"valid_depth_range_m": {"near": 0.1, "far": 10.0}}
    sink: dict[str, Any] = {}
    with pytest.raises(ToolRefusal) as far_plane:
        robust_depth_measurement(
            np.full((5, 5), 9.9, dtype=np.float32),
            source_x=2, source_y=2, calibration=calibration,
            quality_sink=sink,
        )
    assert far_plane.value.code == "invalid_depth"
    assert sink["quality_status"] == "partial"
    assert sink["quality_depth_m"] == pytest.approx(9.9, abs=1e-6)

    edge = np.full((5, 5), 3.0, dtype=np.float32)
    edge[1:4, 1:4] = 1.0
    sink = {}
    with pytest.raises(ToolRefusal) as discontinuity:
        robust_depth_measurement(
            edge, source_x=2, source_y=2, calibration=calibration,
            quality_sink=sink,
        )
    assert discontinuity.value.code == "depth_discontinuity"
    assert sink["quality_rejected_samples"] > 0, (
        "the counter is what tells the model to aim further inside"
    )


def test_omitting_the_sink_leaves_the_estimator_exactly_as_it_was() -> None:
    import numpy as np

    from ora.agent.toolkit.tools.mark_point._measurement import robust_depth_measurement

    calibration = {"valid_depth_range_m": {"near": 0.1, "far": 10.0}}
    measured, quality = robust_depth_measurement(
        np.ones((5, 5), dtype=np.float32), source_x=2, source_y=2,
        calibration=calibration,
    )
    assert measured == pytest.approx(1.0)
    assert quality["supporting_samples"] == 25
    assert quality["method"] == "local_supported_depth_v1"


#: The active default surfaces, and the only ones that may publish the batch.
def _maintained_generation() -> set[str]:
    """Every profile composed from the `explicit_motion_v3` components.

    Derived, not listed. The closed manipulation arm hash-reuses these exact
    files, so it inherits whatever the generation publishes without appearing
    in any literal -- and a name list is edited by whoever adds a sibling,
    which is the edit that gets forgotten.
    """
    import json as _json

    from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG as _CATALOG

    found: set[str] = set()
    for profile, path in _CATALOG.items():
        if path.parent.name == "explicit_motion_v3":
            found.add(profile)
            continue
        manifest = _json.loads(path.read_text(encoding="utf-8"))
        refs = [manifest.get("base_definitions") or "", manifest.get("additional_definitions") or ""]
        refs += list((manifest.get("components") or {}).values())
        if any("explicit_motion_v3/" in str(ref) for ref in refs):
            found.add(profile)
    # A file-backed toolkit surface is its source profile written out per tool. v5's source, the file-backed v4,
    # was v3 (a manifest of these components) written out; both were retired on 2026-10-31. A backend default was its
    # own layered manifest of these components, written out as files (flattened_from); the Synthetic, ora local and
    # robot_runtime_v2 surfaces were written out too but never composed these components.
    retired_sources = {"robodojo_placement_toolkit_v4"}
    for profile, path in _CATALOG.items():
        manifest = _json.loads(path.read_text(encoding="utf-8"))
        backend = manifest.get("runtime_id") != "robot_runtime" and "atomic_pose_gripper_v1" not in manifest["capabilities"]
        if manifest["schema"] == "ora.toolkit-surface.v1" and backend and (
                "flattened_from" in manifest["provenance"]
                or manifest["provenance"].get("source_profile") in found | retired_sources):
            found.add(profile)
    return found


# Component capability inheritance does not require publishing the tool.
# Toolkit-only v1 deliberately excludes it; its public-cohort tests enforce that.

def test_exactly_the_active_default_surfaces_declare_the_batch() -> None:
    """The surface JSON is owned elsewhere and binds components by sha256.

    The maintained generation inherits the batch capability. Profiles that
    publish the tool bind its batch form; restricted successors may withhold
    it. Frozen profiles keep the form their original runs were shown.
    """

    declaring = {
        profile
        for profile, path in PROFILE_CATALOG.items()
        if BATCH_CAPABILITY in json.loads(
            path.read_text(encoding="utf-8"),
        ).get("capabilities", [])
    }
    assert declaring == _maintained_generation()
