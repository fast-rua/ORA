"""Readiness-report composition uses real clients and fake HTTP transport only."""
import json
import time

import httpx
import pytest

from ora.agent.model_services import operator
from ora.agent.model_services.clients import SAM3_MODEL_IDENTITY, CONTACT_GRASPNET_MODEL_IDENTITY
from ora.orchestrator.toolkit.services import service_session


def report(tmp_path, monkeypatch):
    monkeypatch.setenv("ORA_SAM_PORT", "18114")
    monkeypatch.setenv("ORA_GRASP_PORT", "18115")
    monkeypatch.delenv("ORA_SERVICE_PROTOCOL_CHECK", raising=False)
    services = {}
    for name, model in (("segmentation", SAM3_MODEL_IDENTITY), ("grasp_proposals", CONTACT_GRASPNET_MODEL_IDENTITY)):
        health = dict(instance_id=f"fake-{name}", pid=42, started_unix_s=1., adapter_sha256="a" * 64,
                      model=dict(model), load_count=1, warmup_complete=True,
                      protocol_version=operator.GROUNDING_GRASP_PROTOCOL_VERSION)
        services[name] = dict(endpoint=operator.endpoints()[name], health=health)
    value = dict(schema=operator.REPORT_SCHEMA, created_unix_s=time.time(), passed=True,
                 protocol_version=operator.GROUNDING_GRASP_PROTOCOL_VERSION, services=services,
                 checks={key: True for key in ("single_point_wire", "batch_point_wire", "text_wire", "grasp_warmup")})
    path = tmp_path / (operator.digest(value) + ".json")
    path.write_text(json.dumps(value))
    return path


def test_report_identity_client_defaults_and_cleanup_without_network(tmp_path, monkeypatch):
    path = report(tmp_path, monkeypatch)
    handles = []
    original = httpx.Client
    def client(**kwargs):
        def never(request):
            raise AssertionError("constructing replay must not contact services")
        handle = original(transport=httpx.MockTransport(never), **kwargs)
        handles.append(handle)
        return handle
    monkeypatch.setattr(httpx, "Client", client)
    with service_session(path) as (adapters, bindings):
        assert set(adapters) == {"ground", "segment", "segment_points", "grasp_candidates"}
        sam = bindings["ground"]
        assert sam["service_identity"]["instance_id"] == "fake-segmentation"
        assert sam["configuration_identity"]["timeout_s"] == 30
        assert "endpoint" not in sam["configuration_identity"]
        assert bindings["grasp_candidates"]["configuration_identity"]["hover_offset_m"] == .04
        assert bindings["grasp_candidates"]["configuration_identity"]["raw_candidate_limit"] == 32
        assert not any(h.is_closed for h in handles)
    assert len(handles) == 2 and all(h.is_closed for h in handles)


def test_missing_report_means_no_implicit_service(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no implicit client construction")
    monkeypatch.setattr(httpx, "Client", forbidden)
    with service_session(None) as pair:
        assert pair == ({}, {})


def test_report_endpoint_or_hash_drift_is_refused(tmp_path, monkeypatch):
    path = report(tmp_path, monkeypatch)
    monkeypatch.setenv("ORA_SAM_PORT", "18116")
    with pytest.raises(ValueError, match="endpoint"):
        with service_session(path):
            pass
    monkeypatch.setenv("ORA_SAM_PORT", "18114")
    path.write_text(path.read_text().replace('"passed": true', '"passed": false'))
    with pytest.raises(ValueError, match="changed"):
        with service_session(path):
            pass
