"""Fixtures shared by the orchestrator backend tests: ``simulator_host``, a host with RoboCasa and RoboLab slots.

The fixture hosts in ``tests/data/hosts`` register no RoboCasa365 or RoboLab
slot, so the RunSpec builders of those backends compose against this one.
"""

from __future__ import annotations

import pytest

#: The host's id and its slots: ``profile -> (runtime kind, CUDA_VISIBLE_DEVICES)``.
SIMULATOR_HOST = "sim-host"
SIMULATOR_SLOTS = {"robocasa-slot0": ("robocasa", "0"), "robolab-slot2": ("robolab", "2"),
                   "robolab-two-cards": ("robolab", "2,3")}


@pytest.fixture
def simulator_host(workspace, monkeypatch):
    """``sim-host`` registered beside the fixture hosts and declared by this process's host marker."""

    from ora.orchestrator.preflight import resources

    hosts = workspace / "hosts"
    hosts.mkdir()
    lines = ["schema: ora.execution-host.v1", f"id: {SIMULATOR_HOST}", "execution_marker:",
             "  environment_variable: ORA_EXECUTION_HOST", f"  required_value: {SIMULATOR_HOST}",
             "execution_resources:"]
    for profile, (kind, devices) in SIMULATOR_SLOTS.items():
        lines += [f"  {profile}:", f"    runtime_kind: {kind}", f"    resource_class: {kind}_sim",
                  f"    slot_id: {profile}", "    worker_environment:", f"      CUDA_VISIBLE_DEVICES: '{devices}'",
                  "    minimum_free_memory_mib: 8192"]
    (hosts / f"{SIMULATOR_HOST}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(resources, "HOSTS_ROOT", hosts.resolve())
    monkeypatch.setenv("ORA_EXECUTION_HOST", SIMULATOR_HOST)
    return SIMULATOR_HOST
