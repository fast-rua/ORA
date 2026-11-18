"""The record-only switch belongs to orchestration, not the worker configuration."""
import pytest

from ora.orchestrator.spec.run_spec import resolve_run_spec, RunSpecError
from tests.support.orchestrator.native_synthetic_spec import synthetic_spec
from tests.support.orchestrator.run_specs import robodojo_native_spec


def test_recording_is_default_off_and_not_worker_configuration():
    spec=robodojo_native_spec()
    default=resolve_run_spec(spec)
    disabled=resolve_run_spec(dict(spec,record_toolkit_io=False))
    enabled=resolve_run_spec(dict(spec,record_toolkit_io=True))
    assert 'record_toolkit_io' not in default and 'record_toolkit_io' not in disabled
    assert enabled['record_toolkit_io'] is True
    for key in ('runtime','policy','policy_composition','policy_context','policy_identity'):
        assert default[key]==disabled[key]==enabled[key]


@pytest.mark.parametrize('invalid',['true',1,None,{}])
def test_recording_requires_a_boolean(invalid):
    with pytest.raises(RunSpecError,match='record_toolkit_io'):
        resolve_run_spec(dict(synthetic_spec(),record_toolkit_io=invalid))


def test_save_sandbox_is_default_on_and_only_an_opt_out_enters_the_plan():
    spec=robodojo_native_spec()
    default=resolve_run_spec(spec)
    kept=resolve_run_spec(dict(spec,save_sandbox=True))
    hashes=resolve_run_spec(dict(spec,save_sandbox=False))
    assert 'save_sandbox' not in default and default['plan_sha256']==kept['plan_sha256']
    assert hashes.pop('save_sandbox') is False
    for plan in (default,hashes):
        for identity in ('plan_id','plan_sha256','confirmation_code'):plan.pop(identity)
    assert hashes==default


@pytest.mark.parametrize('invalid',['false',0,None,{}])
def test_save_sandbox_requires_a_boolean(invalid):
    with pytest.raises(RunSpecError,match='save_sandbox'):
        resolve_run_spec(dict(synthetic_spec(),save_sandbox=invalid))
