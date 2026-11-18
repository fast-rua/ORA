"""What the motion workers' clients and contracts share: the JSON-lines process, the value checks.

A motion worker (a planner, the timing worker) is one process speaking one JSON
object per line. ``motion_planner.py`` and ``motion_retime.py`` hold the
contracts on top of this module; it knows neither.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import select
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

#: v2 adds the ``retime`` command (v1: ``ready``, ``plan``, ``ik``, ``close``).
PROTOCOL = "ora.motion-planner-protocol.v2"
#: An error is the worker failing (or running out of time) and says nothing about the request.
ERROR_REASONS = ("timeout", "planner_error")


class MotionPlannerError(RuntimeError):
    """A request or answer broke the contract, or the worker was absent, drifted or timed out."""


_HEX = frozenset("0123456789abcdef")


def sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def vector(value: Any, length: int | None = None) -> bool:
    return (isinstance(value, list) and (length is None or len(value) == length)
            and all(not isinstance(item, bool) and isinstance(item, (int, float))
                    and math.isfinite(float(item)) for item in value))


def unit_pose(value: Any) -> bool:
    return vector(value, 7) and math.isclose(sum(float(item) ** 2 for item in value[3:]), 1.0, abs_tol=1e-5)


def number(value: Any, *, positive: bool = False) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
            and (float(value) > 0 if positive else float(value) >= 0))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise MotionPlannerError(message)


def validate_trajectory(trajectory: Any, joint_names: list[str], *, timed: bool = False) -> None:
    """A planner's trajectory; ``timed`` (a retime answer's rows) adds one acceleration per row, all required."""
    fields = {"joint_names", "positions", "velocities", "dt_s"} | ({"accelerations"} if timed else set())
    check(isinstance(trajectory, Mapping) and set(trajectory) == fields
          and trajectory["joint_names"] == joint_names, "trajectory fields or joint order drifted")
    positions = trajectory["positions"]
    check(isinstance(positions, list) and positions and all(vector(row, len(joint_names)) for row in positions),
          "trajectory positions are malformed")
    for key in ("velocities", "accelerations") if timed else ("velocities",):
        rows = trajectory[key]
        check((rows is None and not timed) or (isinstance(rows, list) and len(rows) == len(positions)
                                               and all(vector(row, len(joint_names)) for row in rows)),
              f"trajectory {key} are malformed")
    check(number(trajectory["dt_s"], positive=True) and float(trajectory["dt_s"]) <= 1, "trajectory dt is malformed")


def validate_common(value: Any, schema: str, request: Mapping[str, Any] | None, fields: set[str]) -> None:
    check(isinstance(value, Mapping) and value.get("schema") == schema, f"answer schema is not {schema}")
    check(not set(value) - fields, "answer fields drifted")
    check(isinstance(value.get("planner"), Mapping) and value["planner"].get("implementation")
          and isinstance(value["planner"]["implementation"], str), "answer names no planner implementation")
    timing = value.get("timing_s")
    check(isinstance(timing, Mapping) and number(timing.get("total"))
          and all(number(item) for item in timing.values()), "answer timing is malformed")
    check(isinstance(value.get("detail"), Mapping), "answer detail must be an object")
    if request is not None:
        check(value.get("request") == request, "answer request evidence differs from the request")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class JsonLinesWorker:
    """One worker process spoken to one JSON object per line; a failed exchange retires it.

    Subclasses hold ``_process``, ``_sequence``, ``binding``, ``request_timeout_s`` and ``_label``."""

    _label = "motion planner"

    def _exchange(self, command: str, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        process = self._process
        label = self._label
        if process is None or process.stdin is None or process.stdout is None:
            raise MotionPlannerError(f"{label} worker is not running")
        request_id = f"plan-{self._sequence:06d}-{uuid.uuid4().hex[:8]}"
        self._sequence += 1
        envelope = {"protocol": PROTOCOL, "command": command, "request_id": request_id,
                    **({"request": dict(request)} if request is not None else {})}
        try:
            process.stdin.write(json.dumps(envelope, separators=(",", ":")) + "\n")
            process.stdin.flush()
            deadline = time.monotonic() + self.request_timeout_s
            received = bytearray()
            while b"\n" not in received:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                    raise MotionPlannerError(f"{label} request timed out")
                block = os.read(process.stdout.fileno(), 65536)
                if not block:
                    raise MotionPlannerError(f"{label} worker closed without an answer")
                received.extend(block)
                if len(received) > 64 * 1024 * 1024:
                    raise MotionPlannerError(f"{label} answer exceeds its bound")
            line, trailing = bytes(received).split(b"\n", 1)
            if trailing.strip():
                raise MotionPlannerError(f"{label} worker sent unsolicited data")
            try:
                response = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise MotionPlannerError(f"{label} worker returned invalid JSON") from exc
            if (not isinstance(response, dict) or response.get("protocol") != PROTOCOL
                    or response.get("request_id") != request_id or response.get("command") != command):
                raise MotionPlannerError(f"{label} answer identity drift")
            if "error" in response:
                raise MotionPlannerError(f"{label} worker refused the {command} command")
            return response
        except (MotionPlannerError, OSError, ValueError) as exc:
            self._invalidate()
            if isinstance(exc, MotionPlannerError):
                raise
            raise MotionPlannerError(f"{label} transport failed") from exc

    def _invalidate(self) -> None:
        process, self._process = self._process, None
        self.binding.pop("worker", None)
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
        finally:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()

    def close(self) -> None:
        process, self._process = self._process, None
        self.binding.pop("worker", None)
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"protocol": PROTOCOL, "command": "close",
                                                "request_id": "close"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()


__all__ = ["ERROR_REASONS", "JsonLinesWorker", "MotionPlannerError", "PROTOCOL", "check", "file_sha256", "number",
           "project_root", "sha256", "unit_pose", "validate_common", "validate_trajectory", "vector"]
