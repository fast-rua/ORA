"""Resident perception services: readiness reports, binding and lifecycle."""
from __future__ import annotations

import copy
import json
import time

import pytest

from ora.agent.model_services import operator as services


def test_report_binds_content_adapter_endpoint_and_age(tmp_path, monkeypatch):
    monkeypatch.setenv("ORA_SAM_PORT", "18114")
    monkeypatch.setenv("ORA_GRASP_PORT", "18115")
    health = {**{key: "fixture" for key in services.FIELDS},
              "protocol_version": services.GROUNDING_GRASP_PROTOCOL_VERSION}
    report = {"schema": services.REPORT_SCHEMA, "created_unix_s": time.time(), "passed": True,
              "protocol_version": services.GROUNDING_GRASP_PROTOCOL_VERSION,
              "adapter_sha256": services.grounding_grasp_adapter_sha256(),
              "checks": {k: True for k in ("single_point_wire", "batch_point_wire", "text_wire", "grasp_warmup")},
              "services": {name: {"endpoint": endpoint, "health": health}
                           for name, endpoint in services.endpoints().items()}}
    path = tmp_path / (services.digest(report) + ".json")
    path.write_text(json.dumps(report))
    bound = services.binding(path, "test-surface", ["test-capability"])
    assert len(bound["services"]) == 2
    assert bound["services"]["segmentation"]["tool_surface"] == "test-surface"
    monkeypatch.setenv("ORA_SAM_PORT", "18116")
    with pytest.raises(ValueError, match="endpoint"):
        services.binding(path, "test-surface", [])
    monkeypatch.setenv("ORA_SAM_PORT", "18114")
    changed = copy.deepcopy(report)
    changed["created_unix_s"] -= 86401
    stale = tmp_path / (services.digest(changed) + ".json")
    stale.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="stale"):
        services.binding(stale, "test-surface", [])
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed"):
        services.binding(path, "test-surface", [])


def _report(tmp_path, *, protocol_version, adapter_sha256):
    health = {**{key: "fixture" for key in services.FIELDS}, "adapter_sha256": adapter_sha256}
    if protocol_version is not None:
        health["protocol_version"] = protocol_version
    report = {"schema": services.REPORT_SCHEMA, "created_unix_s": time.time(), "passed": True,
              "adapter_sha256": adapter_sha256,
              "checks": {k: True for k in ("single_point_wire", "batch_point_wire", "text_wire", "grasp_warmup")},
              "services": {name: {"endpoint": endpoint, "health": health}
                           for name, endpoint in services.endpoints().items()}}
    if protocol_version is not None:
        report["protocol_version"] = protocol_version
    path = tmp_path / (services.digest(report) + ".json")
    path.write_text(json.dumps(report))
    return path


def test_a_report_binds_on_its_protocol_version_and_records_the_adapter_hashes(tmp_path, monkeypatch):
    monkeypatch.setenv("ORA_SAM_PORT", "18114")
    monkeypatch.setenv("ORA_GRASP_PORT", "18115")
    monkeypatch.delenv(services.PROTOCOL_CHECK_VARIABLE, raising=False)
    version = services.GROUNDING_GRASP_PROTOCOL_VERSION
    # Another checkout's adapter bytes with the same protocol version bind; the hashes are evidence.
    other = _report(tmp_path, protocol_version=version, adapter_sha256="a" * 64)
    protocol = services.binding(other, "s", [])["services"]["segmentation"]["qualification"]["protocol"]
    assert protocol == {"check": "protocol_version", "service_version": version, "checkout_version": version,
                        "service_adapter_sha256": "a" * 64,
                        "checkout_adapter_sha256": services.grounding_grasp_adapter_sha256()}
    # A service of another version, or one that predates the version, is refused by default ...
    for stale in (_report(tmp_path, protocol_version=version + 1, adapter_sha256="b" * 64),
                  _report(tmp_path, protocol_version=None, adapter_sha256="c" * 64)):
        with pytest.raises(ValueError, match="protocol"):
            services.binding(stale, "s", [])
    # ... and bound, saying so, with the check off (the transition aid).
    monkeypatch.setenv(services.PROTOCOL_CHECK_VARIABLE, "off")
    old = _report(tmp_path, protocol_version=None, adapter_sha256="c" * 64)
    protocol = services.binding(old, "s", [])["services"]["grasp_proposals"]["qualification"]["protocol"]
    assert protocol["check"] == "off" and protocol["service_version"] is None
    monkeypatch.setenv(services.PROTOCOL_CHECK_VARIABLE, "maybe")
    with pytest.raises(ValueError, match="on or off"):
        services.binding(old, "s", [])


def test_service_settings_read_ora_names_then_pre_1_0_names_with_one_warning(monkeypatch):
    import warnings

    from ora.agent.model_services import service_env

    monkeypatch.setattr(service_env, "_deprecation_warned", False)
    monkeypatch.setattr(service_env, "_conflicts_warned", set())
    for service in ("SAM", "GRASP"):
        for name in (service_env.service_key, service_env.legacy_service_key):
            monkeypatch.delenv(name(service, "PORT"), raising=False)
    assert services.endpoints() == {"segmentation": "http://127.0.0.1:8114",
                                    "grasp_proposals": "http://127.0.0.1:8115"}
    monkeypatch.setenv(service_env.legacy_service_key("SAM", "PORT"), "18124")
    with pytest.warns(FutureWarning, match="rename it to ORA_SAM_PORT"):
        assert services.endpoints()["segmentation"] == "http://127.0.0.1:18124"
    monkeypatch.setenv(service_env.legacy_service_key("GRASP", "PORT"), "18125")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # one warning per process, not one per read or name
        assert services.endpoints()["grasp_proposals"] == "http://127.0.0.1:18125"
        monkeypatch.setenv("ORA_GRASP_PORT", "")  # an empty value counts as unset
        monkeypatch.setenv("ORA_SAM_PORT", "18124")  # both names, one value: nothing to say
        assert services.endpoints() == {"segmentation": "http://127.0.0.1:18124",
                                        "grasp_proposals": "http://127.0.0.1:18125"}
    # Both names with different values: the ORA_ name wins and says once that the other is ignored.
    monkeypatch.setenv("ORA_SAM_PORT", "18134")
    legacy = service_env.legacy_service_key("SAM", "PORT")
    with pytest.warns(UserWarning, match=f"ORA_SAM_PORT and {legacy} are both set and differ .*; using "
                                         f"ORA_SAM_PORT='18134' and ignoring {legacy}") as warned:
        assert services.endpoints()["segmentation"] == "http://127.0.0.1:18134"
    assert len(warned) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert services.endpoints()["segmentation"] == "http://127.0.0.1:18134"


def test_services_start_from_pre_1_0_names(tmp_path, monkeypatch):
    from ora.agent.model_services import service_env

    monkeypatch.setattr(service_env, "_deprecation_warned", False)
    values = {"PYTHON": "/fake/python", "ROOT": "/fake/source", "CHECKPOINT": "/fake/checkpoint", "GPU": "3",
              "CONFIG": "/fake/config"}
    for service, port in (("SAM", "18114"), ("GRASP", "18115")):
        for field, value in {**values, "PORT": port}.items():
            monkeypatch.delenv(service_env.service_key(service, field), raising=False)
            monkeypatch.setenv(service_env.legacy_service_key(service, field), value)
    commands = []

    class Process:
        pid = 111
        def __init__(self, command, **kwargs):
            commands.append(command)
    monkeypatch.setattr(services.subprocess, "Popen", Process)
    monkeypatch.setattr(services, "capture_service_process_identity", lambda *a: {"fixture": True})
    with pytest.warns(FutureWarning):
        services.lifecycle("start", root=tmp_path, state=tmp_path / "state")
    assert [command[0] for command in commands] == ["/fake/python", "/fake/python"]
    assert commands[1][-2:] == ["--config", "/fake/config"]
    assert commands[0][commands[0].index("--port") + 1] == "18114"


@pytest.mark.parametrize("marker", ["leases/robotwin--slot0.json", "active-run.json"])
def test_stop_refuses_resource_lease_before_touching_service(tmp_path, monkeypatch, marker):
    runtime_state = tmp_path / "runtime"
    lease = runtime_state / marker
    lease.parent.mkdir(parents=True)
    lease.write_text('{}')
    monkeypatch.setattr(services, "terminate_service_process", lambda record: pytest.fail("must not signal"))
    with pytest.raises(ValueError, match="leases exist"):
        services.lifecycle("stop", root=tmp_path, state=tmp_path / "services",
                           runtime_state_root=runtime_state)


def test_stop_requires_explicit_shared_runtime_state(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "terminate_service_process", lambda record: pytest.fail("must not signal"))
    with pytest.raises(ValueError, match="shared Runtime state root"):
        services.lifecycle("stop", root=tmp_path, state=tmp_path / "services")


def test_readiness_allows_multiple_candidates_for_one_point(tmp_path, monkeypatch):
    import numpy as np
    model_health = {**{key: "fixture" for key in services.FIELDS},
                    "protocol_version": services.GROUNDING_GRASP_PROTOCOL_VERSION}
    model_health.update(adapter_sha256=services.grounding_grasp_adapter_sha256(),
                        warmup_complete=True, load_count=1)
    closed = []

    class Client:
        def require_ready(self):
            return dict(model_health)
        def segment(self, **kwargs):
            return np.ones((128, 128), dtype=bool), {}
        def segment_points(self, **kwargs):
            assert kwargs["points_uv"] == [[64, 64]]
            return [{"candidate": i} for i in range(3)], {"query_results": [{"error_code": None}]}
        def ground(self, **kwargs):
            return [], {}
        def close(self):
            closed.append(self)
    monkeypatch.setattr(services, "clients", lambda: {name: Client() for name in services.NAMES})
    path = services.qualify(tmp_path)
    record = json.loads(path.read_text())
    assert record["checks"]["batch_point_wire"] is True
    assert len(closed) == 2
    assert path.name == services.digest(record) + ".json"


def test_services_start_passes_gpu_and_strips_credentials(tmp_path, monkeypatch):
    commands = []
    for prefix in ("SAM", "GRASP"):
        for field, value in {"PYTHON": "/fake/python", "ROOT": "/fake/source", "CHECKPOINT": "/fake/checkpoint", "GPU": "3"}.items():
            monkeypatch.setenv(f"ORA_{prefix}_{field}", value)
    monkeypatch.setenv("ORA_GRASP_CONFIG", "/fake/config")
    monkeypatch.setenv("DEEPSEEK_LOW_API_KEY", "test-secret-must-not-leak")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("ORA_SAM_PORT", "18114")
    monkeypatch.setenv("ORA_GRASP_PORT", "18115")

    class Process:
        pid = 111
        def __init__(self, command, **kwargs):
            commands.append((command, kwargs))
    monkeypatch.setattr(services.subprocess, "Popen", Process)
    monkeypatch.setattr(services, "capture_service_process_identity", lambda *a: {"fixture": True})
    result = services.lifecycle("start", root=tmp_path, state=tmp_path / "state")
    assert set(result) == set(services.NAMES)
    assert len(commands) == 2
    for command, kw in commands:
        assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        assert "DEEPSEEK_LOW_API_KEY" not in kw["env"]
        assert kw["env"]["HF_HUB_OFFLINE"] == "1"
        assert kw["start_new_session"] is True


def test_services_reject_changed_configuration_for_live_owned_process(tmp_path, monkeypatch):
    for field, value in {"PYTHON": "/fake/python", "ROOT": "/fake/source",
                         "CHECKPOINT": "/fake/checkpoint", "GPU": "3"}.items():
        monkeypatch.setenv(f"ORA_SAM_{field}", value)
    state = tmp_path / "state"
    state.mkdir()
    (state / "segmentation.json").write_text(json.dumps({"pid": 111,
        "command": ["old-command"], "gpu": "2", "source_root": str(tmp_path)}))
    monkeypatch.setattr(services, "service_process_identity_matches", lambda record: True)
    monkeypatch.setattr(services.subprocess, "Popen", lambda *a, **k: pytest.fail("must not start"))
    with pytest.raises(ValueError, match="configuration changed"):
        services.lifecycle("start", root=tmp_path, state=state)


def _record(pid=4242):
    return {"pid": pid, "command": ["python", "-m", "service"], "log": "service.log",
            "process_identity": {"process_group_id": pid, "boot_id": "boot"}}


def test_identity_state_separates_uninspectable_from_drift(monkeypatch):
    from ora.agent.model_services import process

    monkeypatch.setattr(process, "process_alive", lambda pid: True)

    def denied(pid, command, **kwargs):
        raise PermissionError(13, "Permission denied", f"/proc/{pid}/exe")

    monkeypatch.setattr(process, "capture_service_process_identity", denied)
    assert process.service_process_identity_state(_record()) == process.IDENTITY_UNINSPECTABLE
    assert process.service_process_identity_matches(_record()) is False
    with pytest.raises(RuntimeError, match="another container, sandbox or user namespace") as refused:
        process.terminate_service_process(_record())
    assert "process group 4242" in str(refused.value)
    monkeypatch.setattr(process, "capture_service_process_identity",
                        lambda pid, command, **kwargs: {"process_group_id": pid, "boot_id": "other"})
    assert process.service_process_identity_state(_record()) == process.IDENTITY_DRIFTED
    with pytest.raises(RuntimeError, match="drifted identity"):
        process.terminate_service_process(_record())
    monkeypatch.setattr(process, "process_alive", lambda pid: False)
    assert process.service_process_identity_state(_record()) == process.IDENTITY_EXITED
    assert process.terminate_service_process(_record()) is True


def test_stop_reports_a_refused_service_and_still_stops_the_other(tmp_path, monkeypatch):
    state = tmp_path / "services"
    state.mkdir()
    for name, pid in (("segmentation", 11), ("grasp_proposals", 12)):
        (state / f"{name}.json").write_text(json.dumps(_record(pid)))

    def terminate(record):
        if record["pid"] == 11:
            raise RuntimeError("cannot inspect")
        return True

    monkeypatch.setattr(services, "terminate_service_process", terminate)
    result = services.lifecycle("stop", root=tmp_path, state=state, runtime_state_root=tmp_path / "runtime")
    assert result["segmentation"] == {"stopped": False, "pid": 11, "refused": "cannot inspect"}
    assert result["grasp_proposals"] == {"stopped": True}


def test_start_names_an_uninspectable_running_service_instead_of_a_port_error(tmp_path, monkeypatch):
    state = tmp_path / "services"
    state.mkdir()
    (state / "segmentation.json").write_text(json.dumps(_record()))
    monkeypatch.setattr(services, "service_process_identity_state", lambda record: services.IDENTITY_UNINSPECTABLE)
    for prefix in ("SAM", "GRASP"):
        for field in ("PYTHON", "ROOT", "CHECKPOINT", "GPU"):
            monkeypatch.setenv(f"ORA_{prefix}_{field}", "x")
    with pytest.raises(ValueError, match="segmentation: resident service pid 4242 is alive"):
        services.lifecycle("start", root=tmp_path, state=state)
