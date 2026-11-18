"""Public publication/proposal fixtures shared by recorder and fault tests."""
import json
import shutil

from ora.agent.toolkit.interface.types import canonical, digest


def publication(host):
    control, source, manifest = host._current()
    root = host.control_root.parent
    target = root / 'policy-observations' / control['observation_id']
    shutil.copytree(source, target)
    (target / 'observation.json').chmod(0o644)
    manifest['round_index'] = 0
    (target / 'observation.json').write_bytes(canonical(manifest))
    if not (target / 'proprioception.json').exists():
        (target / 'proprioception.json').write_bytes(canonical(manifest['proprioception']))
    control.update(source_root=str(target), observation_sha256=digest((target / 'observation.json').read_bytes()))
    (host.control_root / 'current-observation.json').write_bytes(canonical(control))
    from ora.agent.toolkit.host.runtime.base import RobotToolRuntime
    host._current = RobotToolRuntime._current.__get__(host)
    event = dict(type='fresh_observation_published',seq=1,run_id='run',payload={'observation_id':control['observation_id']})
    (root / 'agent-events.jsonl').write_bytes(canonical(event)+b'\n')
    return control


def proposal(host, name, args, index=0):
    e = dict(type='tool_call_proposed',seq=index+2,run_id='run',tool_call_id=f'call-{index}',model_call_id='model-0',
             payload=dict(name=name,arguments=args))
    with (host.control_root.parent/'agent-events.jsonl').open('ab') as f:
        f.write(canonical(e)+b'\n')
    return {'tool_call_id': e['tool_call_id'], 'model_call_id': e['model_call_id']}


