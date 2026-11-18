"""The Supervisor takes the controller binding and, with motion timing, the interpreter preflight started."""

from __future__ import annotations

import json

import pytest

from ora.orchestrator.backends.libero_pro.supervision import supervisor_binding
from ora.orchestrator.faults import SupervisorError

CONTROLLER = {"check": "libero_controller_action_contract", "passed": True,
              "detail": {"schema": "ora.libero-controller-qualification.v1", "contract_sha256": "c" * 64,
                         "binding": {}}}
TIMED = {"motion_planner": {"implementation": "curobo_v2",
                            "timing": {"implementation": "toppra_ruckig", "python_env": "ORA_TIMING_PYTHON"}}}


def _preflight(tmp_path, *checks):
    (tmp_path / "preflight.json").write_text(json.dumps({"checks": [CONTROLLER, *checks]}))


def test_the_binding_names_the_timing_interpreter_preflight_started(tmp_path):
    timing = {"check": "motion_timing_worker", "passed": True,
              "detail": {"implementation": "toppra_ruckig", "python": "/envs/timing/bin/python",
                         "python_sha256": "d" * 64, "worker": {"pid": 1}}}
    _preflight(tmp_path, timing)
    binding = supervisor_binding(tmp_path, TIMED)
    assert binding["schema"] == "ora.libero-controller-qualification.v1"
    assert binding["motion_timing"] == {"python": "/envs/timing/bin/python", "python_sha256": "d" * 64}
    # Without timing in the runtime, the binding is the controller's alone.
    assert "motion_timing" not in supervisor_binding(tmp_path, {"motion_planner": {"implementation": "curobo_v2"}})


@pytest.mark.parametrize("timing", [None, {"check": "motion_timing_worker", "passed": False, "detail": "no python"}])
def test_a_timed_run_without_a_passing_timing_preflight_does_not_start(tmp_path, timing):
    _preflight(tmp_path, *([timing] if timing else []))
    with pytest.raises(SupervisorError, match="motion timing preflight"):
        supervisor_binding(tmp_path, TIMED)
