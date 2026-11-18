"""Golden agent-core runs: what the agent core sent and recorded on a scripted episode, frozen.

Each golden (``tests/data/agent_core/golden/<name>.json.gz``) holds one scripted
Synthetic episode or wire capture of ``tests/agent_core``. They were recorded
from the Python Native core before ORA removed it, while the Pi core still
matched it byte for byte, so they keep that differential's reference. A test
runs the Pi core on the same scenario and compares, with ``differences``:

* every request: the sha256 and size of its exact bytes (with the remaining
  wall-time budget, a clock reading, pinned or normalized), and on a difference
  the first differing JSON path against the stored request (inline data URLs
  stored by sha256 and length);
* the agent trace: the event types in order and, per event, a sha256 of its
  fields without clocks and the hash chain over them (``episodes.CLOCK_FIELDS``;
  an artifact reference names its bytes by sha256);
* whatever else the scenario records under ``extra``.

A deliberate, reviewed change of what the agent core sends or records
re-records the goldens it touches from the Pi core: run the tests with
``ORA_AGENT_CORE_GOLDEN_BLESS=1``, then review the diff of
``python -m tests.support.agent_core.golden show NAME`` before and after.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests/data/agent_core/golden"
SCHEMA = "ora.agent-core-golden.v1"
BLESS_ENV = "ORA_AGENT_CORE_GOLDEN_BLESS"
BASE64 = re.compile(r"[A-Za-z0-9+/]+={0,2}")
#: Event fields that read a clock, and the hash chain over them.
CLOCK_FIELDS = frozenset({"unix_s", "monotonic_ns", "event_sha256", "previous_event_sha256", "started_unix_s",
                          "finished_unix_s", "duration_s", "latency_s", "duration_ms"})


def skeleton(value: Any) -> Any:
    """``value`` with each inline image of 1 KiB or more (a data URL, or a bare base64
    block as Anthropic's wire sends it) replaced by its sha256 and length."""

    if isinstance(value, str):
        if len(value) >= 1024 and (value.startswith("data:") or BASE64.fullmatch(value)):
            return f"<data {hashlib.sha256(value.encode()).hexdigest()} {len(value)}>"
        return value
    if isinstance(value, list):
        return [skeleton(item) for item in value]
    if isinstance(value, dict):
        return {key: skeleton(item) for key, item in value.items()}
    return value


def _without_clocks(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_clocks(item) for key, item in value.items() if key not in CLOCK_FIELDS}
    if isinstance(value, list):
        return [_without_clocks(item) for item in value]
    return value


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def summarize(requests: Iterable[bytes], events: Iterable[Mapping[str, Any]] | None = None,
              **extra: Any) -> dict[str, Any]:
    """The golden record of one run: its requests (exact bytes), its trace and ``extra``."""

    requests = list(requests)
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "requests": [{"sha256": _sha(body), "bytes": len(body)} for body in requests],
        "request_bodies": [skeleton(json.loads(body)) for body in requests],
    }
    if events is not None:
        normalized = [_without_clocks(dict(event)) for event in events]
        summary["event_types"] = [event["type"] for event in normalized]
        summary["events"] = [_sha(json.dumps(event, sort_keys=True, ensure_ascii=False).encode())
                             for event in normalized]
    if extra:
        summary["extra"] = json.loads(json.dumps(extra, sort_keys=True))
    return summary


def path(name: str) -> Path:
    return GOLDEN / f"{name}.json.gz"


def load(name: str) -> dict[str, Any]:
    return json.loads(gzip.decompress(path(name).read_bytes()))


def save(name: str, summary: Mapping[str, Any], *, recorded_from: str) -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    record = {**summary, "name": name, "recorded_from": recorded_from}
    data = json.dumps(record, sort_keys=True, ensure_ascii=False, indent=1).encode()
    path(name).write_bytes(gzip.compress(data, mtime=0))


def first_difference(a: Any, b: Any, where: str = "$") -> str | None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{where}.{key}: only in {'golden' if key in a else 'this run'}"
            found = first_difference(a[key], b[key], f"{where}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{where}: length {len(a)} (golden) vs {len(b)}"
        for index, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{where}[{index}]")
            if found:
                return found
        return None
    if a != b:
        return f"{where}: golden {str(a)[:160]!r} vs {str(b)[:160]!r}"
    return None


def differences(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    """Every way ``actual`` departs from the golden ``expected``; empty when they agree."""

    problems: list[str] = []
    golden_requests, requests = expected["requests"], actual["requests"]
    if len(golden_requests) != len(requests):
        problems.append(f"requests: {len(golden_requests)} in the golden, {len(requests)} in this run")
    for index, (left, right) in enumerate(zip(golden_requests, requests)):
        if left != right:
            where = first_difference(expected["request_bodies"][index], actual["request_bodies"][index])
            problems.append(f"request {index}: {where or 'same JSON, different bytes'}")
    if "events" in expected:
        if expected["event_types"] != actual.get("event_types"):
            problems.append(f"event types: {first_difference(expected['event_types'], actual.get('event_types'))}")
        else:
            problems += [f"event {index} ({kind}) differs"
                         for index, (kind, left, right) in enumerate(zip(
                             expected["event_types"], expected["events"], actual["events"])) if left != right]
    if expected.get("extra") != actual.get("extra"):
        problems.append(f"extra: {first_difference(expected.get('extra'), actual.get('extra'))}")
    return problems


def check(name: str, actual: Mapping[str, Any]) -> list[str]:
    """``differences`` from golden ``name``; with ``ORA_AGENT_CORE_GOLDEN_BLESS=1`` re-record it instead."""

    if os.environ.get(BLESS_ENV) == "1":
        save(name, actual, recorded_from="pi")
        return []
    if not path(name).is_file():
        return [f"no golden {path(name).relative_to(ROOT)}; record it with {BLESS_ENV}=1 and review it"]
    return differences(load(name), json.loads(json.dumps(actual, sort_keys=True)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("list", "show"))
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    names = args.names or sorted(item.name[:-len(".json.gz")] for item in GOLDEN.glob("*.json.gz"))
    for name in names:
        record = load(name)
        if args.command == "list":
            print(name, record["recorded_from"], len(record["requests"]), "requests",
                  len(record.get("events", [])), "events")
        else:
            print(json.dumps(record, sort_keys=True, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
