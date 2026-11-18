"""Run bindings: from a file or an allocation, the model check and the run environment."""
import json

import pytest

from ora.orchestrator.task.binding import (
    BindingError, check_model, load_binding, run_environment, validate_binding,
)
from ora.orchestrator.task.run import prepare_environment

ALLOCATION = {"allocation": {
    "build_spec": {"provider_catalog": "/c.yaml", "provider_profile": "p", "resource_profile": "robodojo-gpu3",
                   "service_report": "/s.json"},
    "env": {"ORA_SAM_PORT": "8214", "ORA_EXECUTION_HOST": "h"},
    "lane": {"card": 3, "env": {"ORA_HOSTS_DIR": "/hosts"}},
    "llm": {"model": "short-name", "served_model": "org/Model"},
}}
PROFILE = {"model": "org/Model", "endpoint": "https://replica:1/v1", "base_url_env": "M_BASE_URL",
           "model_env": "M_MODEL_ID"}


def test_an_allocation_file_is_a_binding(tmp_path, monkeypatch):
    path = tmp_path / "allocation.json"
    path.write_text(json.dumps(ALLOCATION))
    monkeypatch.setenv("ORA_ROUTER_ALLOCATION", str(path))
    binding = load_binding()
    assert (binding["gpu"], binding["resource_profile"], binding["provider_profile"]) == ("3", "robodojo-gpu3", "p")
    assert binding["env"] == {"ORA_HOSTS_DIR": "/hosts", "ORA_SAM_PORT": "8214", "ORA_EXECUTION_HOST": "h"}
    assert binding["models"] == ["short-name", "org/Model"]
    monkeypatch.delenv("ORA_ROUTER_ALLOCATION")
    with pytest.raises(BindingError):
        load_binding()


def test_a_binding_file_is_validated():
    with pytest.raises(BindingError):
        validate_binding({"schema": "ora.run-binding.v1", "gpu": 0, "provider_catalog": "/c.yaml"})
    with pytest.raises(BindingError):
        validate_binding({"schema": "ora.run-binding.v1", "gpu": 0, "provider_catalog": "/c", "provider_profile": "p",
                          "card": 1})


def test_the_replica_must_serve_the_tasks_model():
    binding = validate_binding({"schema": "ora.run-binding.v1", "gpu": 0, "provider_catalog": "/c",
                                "provider_profile": "p", "models": ["short-name"]})
    for name in ("short-name", "org/Model"):
        check_model({"model": name}, binding, PROFILE)
    with pytest.raises(BindingError):
        check_model({"model": "other"}, binding, PROFILE)


def test_the_run_environment_sets_the_profiles_variables_without_overriding():
    binding = {"env": {"ORA_SAM_PORT": "8214"}}
    env = run_environment(binding, PROFILE, {"M_MODEL_ID": "kept"})
    assert env == {"ORA_SAM_PORT": "8214", "M_BASE_URL": "https://replica:1/v1", "M_MODEL_ID": "kept"}


def test_startup_variables_cannot_change_after_ora_starts(tmp_path, monkeypatch):
    catalog = tmp_path / "c.yaml"
    catalog.write_text(json.dumps({"profiles": {"p": PROFILE}}))
    binding = validate_binding({"schema": "ora.run-binding.v1", "gpu": 0, "provider_catalog": str(catalog),
                                "provider_profile": "p", "env": {"ORA_WORKSPACE_ROOT": "/other"}})
    with pytest.raises(BindingError, match="ORA_WORKSPACE_ROOT"):
        prepare_environment({"model": "org/Model"}, binding, {"ORA_WORKSPACE_ROOT": "/here"})
    assert prepare_environment({"model": "org/Model"}, binding, {"ORA_WORKSPACE_ROOT": "/other"})[
        "M_BASE_URL"] == "https://replica:1/v1"
