"""Replay recorded get_obs render synchronizations at their original physics clocks.

Every restored round is also rendered once at its start, before its first replayed
control, unless the history already renders there. The renderer's first transform
update after the reset's wrist-camera pose initialization fixes the camera's mount on
link6 from the camera's reset world pose and link6's pose *at that render*: rendered
first at a moved boundary (where the render history is unavailable), the
wrist cameras keep their reset world pose and then follow link6 rigidly with that wrong
offset. A recorded history starts with the reset observation, so the anchor only renders
when the history does not.
"""

import json
import time
from .control_replay import ReplayError


class RenderHistory:
    def __init__(self, session, history):
        self.session, self.history = session, history
        self.events = history.get("events", [])
        self.cursor = 0
        self.results = []
        self.callbacks = []
        self.rendered = set()
        self.anchored_rounds = set()
        self.anchors = []

    def attach(self):
        environment = self.session.environment
        if not any(previous is environment for previous, _ in self.callbacks):
            previous = getattr(environment, "execution_capture_callback", None)
            if previous is not None:
                raise ReplayError(
                    "diagnostic rendering requires execution capture disabled"
                )
            self.callbacks.append((environment, previous))
            environment.execution_capture_callback = self.synchronize
        self.synchronize()
        self._anchor_round()

    def _anchor_round(self):
        """Render once at a round's first attach (its start) if the history did not render there."""
        current = (self.session.round_index, self.session._clock())
        if current[0] in self.anchored_rounds:
            return
        self.anchored_rounds.add(current[0])
        if current in self.rendered:
            return
        self._render(dict(round_index=current[0], physics_clock=current[1], reason="round_start_camera_mount"),
                     self.anchors)

    def synchronize(self):
        current = (self.session.round_index, self.session._clock())
        while self.cursor < len(self.events):
            event = self.events[self.cursor]
            target = (event["round_index"], event["physics_clock"])
            if target > current:
                break
            self.cursor += 1
            if target < current:
                self.results.append(dict(event, status="missed_position"))
                continue
            self._render(event, self.results)

    def _render(self, event, into):
        started = time.monotonic()
        result = dict(event, status="ok")
        before = self.session._clock()
        self.rendered.add((self.session.round_index, before))
        try:
            self.session.environment.get_obs()
        except Exception as error:
            result.update(
                status="render_error", error=f"{type(error).__name__}: {error}"
            )
        result["duration_s"] = time.monotonic() - started
        into.append(result)
        if self.session._clock() != before:
            result["status"] = "physical_divergence"
            raise ReplayError("historical render synchronization advanced physics")

    def close(self):
        for environment, previous in self.callbacks:
            environment.execution_capture_callback = previous
        report = dict(
            schema="ora.toolkit.render-history.v1",
            source_status=self.history.get("status"),
            source_reason=self.history.get("reason"),
            scheduled=len(self.events),
            consumed=self.cursor,
            results=self.results,
            round_start_renders=self.anchors,
            pending=self.events[self.cursor :],
            final_capture=self.history.get("final_capture"),
            complete=self.history.get("status") == "recorded"
            and self.cursor == len(self.events)
            and all(row["status"] == "ok" for row in self.results),
        )
        path = self.session.output_root / "render-history-replay.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
