"""Python/TypeScript serialization parity for the Pi core spike.

The TypeScript side must reproduce Python ``json.dumps`` (model-visible text,
ORA ``canonical_bytes``) for values it received from Python, and both sides
must agree on RFC 8785 JCS for agent-event v2 hashes.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from ora.agent.evidence.canonical import canonical_bytes
from tests.support.agent.jcs import JcsError, es_number, jcs_string

ROOT = Path(__file__).resolve().parents[2]


def _floats(rng: random.Random) -> list[float]:
    fixed = [0.0, -0.0, 1.0, -1.0, 2.0, 0.1, 0.5, 1e-4, 1e-5, 1e-7, 1.5e-7, 1e16, 1e15,
             1234567890123456.0, 12345678901234567.0, 1e21, 1e22, 1.7976931348623157e308,
             5e-324, 0.30000000000000004, 3.141592653589793, 100.0, 1e-6, 1e-07, 123e-9]
    values = list(fixed)
    for _ in range(400):
        mantissa = rng.uniform(-10, 10)
        values.append(mantissa * 10 ** rng.randint(-30, 30))
        values.append(float(rng.randint(-10**6, 10**6)))
        values.append(round(rng.uniform(-1, 1), rng.randint(0, 8)))
    return values


def _strings(rng: random.Random) -> list[str]:
    pool = ['"', "\\", "\n", "\r", "\t", "\b", "\f", "\x00", "\x1f", "\x7f", "é", "中",
            " ", "\u0085", "😀", "a", "Z", " ", "/", "<", ">", "&", "﻿"]
    return ["".join(rng.choice(pool) for _ in range(rng.randint(0, 12))) for _ in range(200)]


def _value(rng: random.Random, depth: int = 0):
    kind = rng.randint(0, 9 if depth < 3 else 5)
    if kind == 0:
        return rng.choice([None, True, False])
    if kind == 1:
        return rng.choice([0, 1, -1, 2**53 - 1, -(2**53 - 1), rng.randint(-10**9, 10**9)])
    if kind in (2, 3):
        return rng.choice(_floats(rng))
    if kind in (4, 5):
        return rng.choice(_strings(rng))
    if kind in (6, 7):
        return [_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    keys = ["a", "b", "B", "_x", "é", "😀", "ﬁ", "z1", "z10", "z2", "", "ä"]
    return {rng.choice(keys) + rng.choice(["", "k", "😀"]): _value(rng, depth + 1)
            for _ in range(rng.randint(0, 5))}


@pytest.mark.usefixtures("node_runtime")
def test_typescript_reproduces_python_json_and_jcs(tmp_path: Path) -> None:
    rng = random.Random(20261015)
    option_sets = [
        {"py": dict(ensure_ascii=False, sort_keys=True), "ts": {"sortKeys": True, "ensureAscii": False}},
        {"py": dict(ensure_ascii=False, sort_keys=True, separators=(",", ":")),
         "ts": {"sortKeys": True, "ensureAscii": False, "separators": [",", ":"]}},
        {"py": dict(ensure_ascii=True, sort_keys=False), "ts": {"sortKeys": False, "ensureAscii": True}},
        {"py": dict(ensure_ascii=False, sort_keys=True, indent=2),
         "ts": {"sortKeys": True, "ensureAscii": False, "indent": 2}},
    ]
    cases = []
    for index in range(1500):
        value = _value(rng)
        option = option_sets[index % len(option_sets)]
        text = json.dumps(value, ensure_ascii=False)
        try:
            jcs = jcs_string(value)
        except JcsError:
            jcs = None
        cases.append({
            "text": text, "options": option["ts"],
            "expected": json.dumps(value, **option["py"]),
            "canonical": canonical_bytes(value).decode("utf-8"),
            "jcs": jcs,
        })
    fixture = tmp_path / "cases.json"
    fixture.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [shutil.which("node"), str(ROOT / "ora-pi/test/pyjson_parity.ts"), str(fixture)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_es_number_layout() -> None:
    assert es_number(1.0) == "1"
    assert es_number(-0.0) == "0"
    assert es_number(1e-7) == "1e-7"
    assert es_number(1e-6) == "0.000001"
    assert es_number(1e21) == "1e+21"
    assert es_number(1e20) == "100000000000000000000"
    assert es_number(123.456) == "123.456"
    assert es_number(1.5e-7) == "1.5e-7"
    with pytest.raises(JcsError):
        jcs_string(2**60)
