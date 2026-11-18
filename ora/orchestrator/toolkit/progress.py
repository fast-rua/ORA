"""Flush replay stage progress as it happens, with monotonic stage timings."""

import time

from ora.agent.toolkit.interface.types import canonical

STAGES = (
    "resolve",
    "boot_worker",
    "replay_controls",
    "verify_state",
    "run_tool",
    "compare",
    "done",
)


class Progress:
    def __init__(self, output):
        self.path = output / "progress.jsonl"
        self.timing = {stage: 0.0 for stage in STAGES}
        self.active = None
        self.started = None

    def emit(self, stage, message, *, done=None, total=None):
        with self.path.open("ab") as stream:
            stream.write(
                canonical(
                    dict(
                        t=time.time(),
                        stage=stage,
                        done=done,
                        total=total,
                        message=message,
                    )
                )
                + b"\n"
            )
            stream.flush()

    def finish(self):
        if self.active is not None:
            self.timing[self.active] += max(0.0, time.monotonic() - self.started)
            self.active = None

    def start(self, stage, message):
        self.finish()
        self.active, self.started = stage, time.monotonic()
        self.emit(stage, message)

    def skip(self, stage, message):
        self.emit(stage, message, done=0, total=0)

    def close(self, status):
        self.finish()
        self.emit("done", status, done=1, total=1)
