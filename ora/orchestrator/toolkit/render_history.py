"""Locate render synchronizations preceding a published observation, without guessing."""

from .selection import rows
from ora.agent.toolkit.interface.types import decode


def schedule(source, manifest, manifest_ref, clock, round_index):
    """Include execution captures as well as published observations; exclude final capture."""
    from .live_boundary import verified

    try:
        root = (source / manifest_ref["path"]).parent
        references = {}
        for camera in manifest["cameras"].values():
            calibration = decode(verified(root, camera["files"]["calibration"]))
            renderer = calibration["renderer_calibration"]
            reference = renderer["reference_time"]
            references[renderer["render_product"]] = [
                reference["referenceTimeNumerator"],
                reference["referenceTimeDenominator"],
            ]
        history = rows(source, "native/render-synchronization.jsonl")
        matches = []
        for index, event in enumerate(history):
            samples = event.get("samples") or []
            found = (
                {
                    v["render_product"]: v["reference_time"]
                    for v in samples[-1]["cameras"]
                }
                if samples
                else {}
            )
            if event.get("physics_step") == clock and found == references:
                matches.append(index)
        if len(matches) != 1:
            raise ValueError(
                "published renderer reference does not select a unique synchronization"
            )
        selected = matches[0]
        events, current_round, offset = [], 0, None
        previous_clock = None
        for index, event in enumerate(history[: selected + 1]):
            current_offset = event.get("simulation_time_offset_s", 0.0)
            if offset is not None and current_offset != offset:
                if current_offset < offset:
                    raise ValueError("render round offset went backwards")
                current_round += 1
                previous_clock = None
            offset = current_offset
            step = event["physics_step"]
            if (
                type(step) is not int
                or previous_clock is not None
                and step < previous_clock
            ):
                raise ValueError("render history has no monotonic physical position")
            previous_clock = step
            events.append(
                dict(
                    index=index,
                    round_index=current_round,
                    physics_clock=step,
                    recorded_render_calls=len(event["samples"]),
                    matched=event.get("matched"),
                )
            )
        if current_round != round_index:
            raise ValueError("render rounds do not identify the selected boundary")
        return dict(
            status="recorded",
            events=events[:-1],
            final_capture=events[-1],
            final_reference_times=references,
            source="native/render-synchronization.jsonl",
        )
    except Exception as error:
        return dict(
            status="unavailable",
            events=[],
            reason=f"{type(error).__name__}: {error}",
            source="native/render-synchronization.jsonl",
        )
