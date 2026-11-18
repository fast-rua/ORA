"""The station's switch table (``backends/yam/switches.py``) and the runtime values its keys may hold
(``benchmarks.yam.RUNTIME_SWITCHES``) name the same switches; ``ora play`` offers each flag, off by default."""

from __future__ import annotations

from pathlib import Path

import pytest

from ora.app.main import build_parser
from ora.environment.benchmarks.yam import RUNTIME_SWITCHES, YamBenchmarkError, validate_runtime_extension
from ora.orchestrator.backends.yam import switches

ROOT = Path(__file__).resolve().parents[5]


def test_every_runtime_key_a_switch_sets_has_one_value_and_every_value_a_switch() -> None:
    keys = [switch.runtime_key for switch in switches.SWITCHES if switch.runtime_key is not None]
    assert sorted(keys) == sorted(RUNTIME_SWITCHES) and len(set(keys)) == len(keys)
    assert len({switch.name for switch in switches.SWITCHES}) == len(switches.SWITCHES)
    for switch in switches.SWITCHES:
        assert switch.flag == "--" + switch.name.replace("_", "-")
        if switch.layer is not None:
            assert (ROOT / switch.layer).is_file(), switch.layer


def test_ora_play_offers_every_switch_off_by_default() -> None:
    args = build_parser().parse_args(["play", "--env", "yam"])
    assert all(getattr(args, switch.name) is False for switch in switches.SWITCHES)
    for switch in switches.SWITCHES:
        assert getattr(build_parser().parse_args(["play", "--env", "yam", switch.flag]), switch.name) is True


def test_the_runtime_keeps_each_switch_off_unless_it_holds_exactly_its_value() -> None:
    base = {"task": "operator_task", "rig_mode": "sim", "task_instruction": "stack the blocks"}
    plain = validate_runtime_extension(base)
    assert {key: plain[key] for key in RUNTIME_SWITCHES} == {
        key: (False if value is True else None) for key, value in RUNTIME_SWITCHES.items()}
    on = validate_runtime_extension({**base, **{key: (True if value is True else dict(value))
                                                for key, value in RUNTIME_SWITCHES.items()}})
    assert all(on[key] == (True if value is True else dict(value)) for key, value in RUNTIME_SWITCHES.items())
    for key, value in RUNTIME_SWITCHES.items():
        with pytest.raises(YamBenchmarkError, match=key):
            validate_runtime_extension({**base, key: "yes" if value is True else {"other": 1}})
