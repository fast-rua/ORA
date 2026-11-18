"""A published renderer reference selects a prefix, including intermediate captures."""

from ora.agent.toolkit.interface.types import canonical, digest
from ora.orchestrator.toolkit.render_history import schedule


def test_render_schedule_stops_before_selected_capture_and_includes_unpublished(
    tmp_path,
):
    (tmp_path / "obs").mkdir()
    (tmp_path / "native").mkdir()
    calibration = canonical(
        dict(
            renderer_calibration=dict(
                render_product="camera",
                reference_time=dict(
                    referenceTimeNumerator=153, referenceTimeDenominator=30
                ),
            )
        )
    )
    (tmp_path / "obs/calibration.json").write_bytes(calibration)
    manifest = dict(
        cameras=dict(
            head=dict(
                files=dict(
                    calibration=dict(
                        path="calibration.json", sha256=digest(calibration)
                    )
                )
            )
        )
    )
    events = [
        dict(
            physics_step=clock,
            matched=True,
            samples=[
                dict(
                    cameras=[dict(render_product="camera", reference_time=[frame, 30])]
                )
            ],
        )
        for clock, frame in (
            (844, 117),
            (879, 119),
            (1064, 151),
            (1064, 153),
            (1064, 155),
        )
    ]
    (tmp_path / "native/render-synchronization.jsonl").write_bytes(
        b"".join(canonical(e) + b"\n" for e in events)
    )
    result = schedule(tmp_path, manifest, dict(path="obs/observation.json"), 1064, 0)
    assert result["status"] == "recorded"
    assert [r["physics_clock"] for r in result["events"]] == [844, 879, 1064]
    assert result["final_capture"]["index"] == 3
    # A missing/ambiguous reference is diagnostic unavailability, never a guessed prefix.
    with (tmp_path / "native/render-synchronization.jsonl").open("ab") as f:
        f.write(canonical(events[3]) + b"\n")
    assert (
        schedule(tmp_path, manifest, dict(path="obs/observation.json"), 1064, 0)[
            "status"
        ]
        == "unavailable"
    )
