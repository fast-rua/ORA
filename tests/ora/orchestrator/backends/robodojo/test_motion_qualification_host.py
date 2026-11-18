"""Host migration keeps gate semantics and starts no simulator during imports."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import pytest
from ora.orchestrator.backends.robodojo import qualification
from ora.orchestrator.backends.robodojo.backend import BACKEND
from ora.orchestrator.episode.execution_state import ExecutionStateOwner

TIMING = {'check': 'motion_timing_worker', 'passed': True,
          'detail': {'implementation': 'toppra_ruckig', 'python': '/timing/python', 'python_sha256': 'b' * 64}}


@pytest.fixture(autouse=True)
def _timing_worker(monkeypatch):
    """The timing preflight starts a real worker; these tests stand in for its passing check."""
    monkeypatch.setattr(qualification, 'motion_timing_check', lambda timing, environment_overrides=None: TIMING)


def test_numeric_suite_import_is_host_safe():
    code = '''
import sys
import ora.orchestrator.backends.robodojo.qualification
for name in sys.modules:
    assert not any(name == p or name.startswith(p + '.') for p in
                   ('isaac', 'isaacsim', 'isaaclab', 'omni', 'pxr', 'carb', 'robodojo')), name
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
    assert BACKEND.qualify_tools['motion'].target == 'ora.orchestrator.backends.robodojo.qualification:main'


@pytest.mark.parametrize('passed', [True, False])
def test_host_binds_owner_preserves_record_and_writes_only_passing_gate(tmp_path, monkeypatch, passed):
    from ora.environment import factory
    runtime = {'required_host': 'test-host', 'cuda_visible_devices': '0', 'task_variant': 'general_pickup'}
    config, output, gate = tmp_path/'runtime.json', tmp_path/'out', tmp_path/'gate.json'
    config.write_text(json.dumps(runtime))
    gate.write_text('previous passing gate')
    monkeypatch.setenv('ORA_EXECUTION_HOST', 'test-host')
    monkeypatch.setattr(qualification, '_free_memory_mib', lambda device: 30000)
    calls = []
    class Session:
        def start(self):
            assert isinstance(self.execution_state_owner, ExecutionStateOwner)
            calls.append('start')
            return {}
        def close(self):
            calls.append('close')
        def __getattr__(self, name):
            if name == 'execution_state_owner':
                return None
            if name == 'delegate':
                return self
            return lambda *a, **kw: {}
    session = Session()
    monkeypatch.setattr(qualification, 'run_cases', lambda **kw: [{'case_id': 'fixture', 'passed': passed}])
    def record(**kwargs):
        identity = kwargs['session_identity']
        assert identity['action_codec'] == 'motion_action_v3' and identity['rounds'] == 2
        # The runtime named no planner, timing or motion profile, so the default baseline's were bound and recorded.
        assert identity['motion_stack_source'] == {'motion_planner': 'baseline:robodojo_toolkit_v7',
                                                   'motion_timing': 'baseline:robodojo_toolkit_v7',
                                                   'motion_profile': 'baseline:robodojo_toolkit_v7'}
        assert identity['motion_planner']['implementation'] == 'curobo_v2' and identity['free_route'] == 'global'
        assert identity['motion_timing'] == {'implementation': 'toppra_ruckig', 'python_sha256': 'b' * 64}
        assert identity['motion_profile']['profile_id'] == 'arx_x5_motion_profile_v2'
        assert identity['motion_profile']['path'] == 'ora/policy/embodiments/arx-x5/motion-profile-v2.json'
        return dict(schema='ora.test-record.v1', passed=passed, cases=kwargs['case_results'], missing_cases=[])
    monkeypatch.setattr(qualification, 'build_qualification_record', record)
    booted = {}
    monkeypatch.setattr(factory, 'build_robodojo_worker_session', lambda **kw: booted.update(kw) or session)
    code = qualification.main(['--runtime-config', str(config), '--output', str(output), '--gate', str(gate)])
    assert code == (0 if passed else 1)
    assert calls == ['start', 'close']
    assert booted['runtime']['motion_planner']['implementation'] == 'curobo_v2'
    # The worker binds its timing from the preflight record beside its output root (motion_timing.bind_preflight).
    assert booted['output_root'] == output / 'environment'
    assert json.loads((output / 'preflight.json').read_text()) == {'checks': [TIMING]}
    result = (output/'robodojo-sparse-motion-qualification.json').read_bytes()
    assert (gate.read_bytes() == result) if passed else (gate.read_text() == 'previous passing gate')


def test_boot_failure_still_closes_worker(tmp_path, monkeypatch):
    from ora.environment import factory
    config = tmp_path/'runtime.json'
    config.write_text(json.dumps({'required_host': 'test-host', 'cuda_visible_devices': '0',
                                 'task_variant': 'general_pickup'}))
    monkeypatch.setenv('ORA_EXECUTION_HOST', 'test-host')
    monkeypatch.setattr(qualification, '_free_memory_mib', lambda device: 30000)
    closed = []
    def boot():
        raise RuntimeError('fixture boot failure')
    session = SimpleNamespace(start=boot, close=lambda: closed.append(True))
    monkeypatch.setattr(factory, 'build_robodojo_worker_session', lambda **kw: session)
    with pytest.raises(RuntimeError, match='fixture boot failure'):
        qualification.main(['--runtime-config', str(config), '--output', str(tmp_path/'out')])
    assert closed == [True]


def test_owner_binding_failure_still_closes_worker(tmp_path, monkeypatch):
    from ora.environment import factory
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps({"required_host": "test-host", "cuda_visible_devices": "0",
                                 "task_variant": "general_pickup"}))
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    monkeypatch.setattr(qualification, "_free_memory_mib", lambda device: 30000)
    closed = []
    session = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(factory, "build_robodojo_worker_session", lambda **kw: session)
    def fail(session):
        raise RuntimeError("fixture binding failure")
    monkeypatch.setattr(qualification, "bind_execution_state", fail)
    with pytest.raises(RuntimeError, match="fixture binding failure"):
        qualification.main(["--runtime-config", str(config), "--output", str(tmp_path / "out")])
    assert closed == [True]


def test_the_motion_stack_is_the_runtimes_own_or_the_default_baselines(tmp_path):
    bound, source = qualification.motion_stack({'motion_planner': {'implementation': 'custom'}})
    assert bound['motion_planner'] == {'implementation': 'custom'} and source['motion_planner'] == 'runtime_config'
    assert bound['motion_timing']['implementation'] == 'toppra_ruckig'
    assert source['motion_timing'] == 'baseline:robodojo_toolkit_v7'
    with pytest.raises(SystemExit, match='runtime.motion_planner must be an object'):
        qualification.motion_stack({'motion_planner': None})


def test_a_timing_worker_that_does_not_start_stops_before_the_simulator(tmp_path, monkeypatch):
    from ora.environment import factory
    config = tmp_path / 'runtime.json'
    config.write_text(json.dumps({'required_host': 'test-host', 'cuda_visible_devices': '0',
                                  'task_variant': 'general_pickup'}))
    monkeypatch.setenv('ORA_EXECUTION_HOST', 'test-host')
    monkeypatch.setattr(qualification, '_free_memory_mib', lambda device: 30000)
    monkeypatch.setattr(qualification, 'motion_timing_check', lambda timing, environment_overrides=None: {
        'check': 'motion_timing_worker', 'passed': False, 'detail': 'ORA_TIMING_PYTHON is missing'})
    monkeypatch.setattr(factory, 'build_robodojo_worker_session',
                        lambda **kw: pytest.fail('a qualification without its timing worker booted a simulator'))
    with pytest.raises(SystemExit, match='timing worker did not start'):
        qualification.main(['--runtime-config', str(config), '--output', str(tmp_path / 'out')])
