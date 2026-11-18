"""Fixtures shared by the orchestrator tests: ``workspace``, a private workspace for composed RunSpecs."""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A private workspace: compose writes and the resolver reads outputs/<experiment> there."""
    from ora.orchestrator.spec import benchmark as benchmark_spec
    from ora.orchestrator.spec import run_spec as spec_module
    from ora.orchestrator.spec.inputs import SERVICE_VARIABLES

    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    monkeypatch.setattr(spec_module, "resolve_workspace_root", lambda root: tmp_path)

    def experiment_root(experiment):
        path = tmp_path / "outputs" / experiment
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(benchmark_spec, "experiment_root", experiment_root)
    # compose binds the tool-client endpoints into os.environ itself, which
    # monkeypatch does not record for an absent variable: restore them here so
    # the fixture endpoints never reach a later test module.
    saved = {variable: os.environ.pop(variable, None) for _, variable in SERVICE_VARIABLES}
    yield tmp_path
    for variable, value in saved.items():
        if value is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = value
