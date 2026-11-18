"""Synthetic completions of the cropped public fixture for atomic-tool contracts.

Missing cameras deliberately reuse head pixels/calibration; these are fixture
inputs, not recovered historical sensor evidence or a simulator run.
"""
import json
import shutil
from ora.agent.toolkit.interface.types import canonical, digest
from ora.agent.evidence.policy_state import PolicyStateStore, verify_policy_state


def complete_cameras(host):
    control, root, manifest = host._current()
    head = manifest['cameras']['head']
    for name, camera in manifest['cameras'].items():
        camera.update(width=head['width'], height=head['height'])
        for role, ref in camera['files'].items():
            path = root / ref['path']
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((root / head['files'][role]['path']).read_bytes())
                ref.update(sha256=digest(path.read_bytes()), bytes=path.stat().st_size)
    (root / 'observation.json').write_bytes(canonical(manifest))
    control['observation_sha256'] = digest((root / 'observation.json').read_bytes())
    (host.control_root / 'current-observation.json').write_bytes(canonical(control))


def timeline(host):
    """Publish two distinct real policy-state entries over synthetic camera fixtures."""
    complete_cameras(host)
    control, root, manifest = host._current()
    store = PolicyStateStore(host.control_root.parent, 'run', 'a' * 64)
    first = store.publish(observation=manifest, observation_root=root, cause='reset',
        task_instruction='fixture task', previous_public_action=None,
        remaining_budget={'robot_decisions':10,'model_calls':20})
    second_root = root.parent / 'observation-0001'
    shutil.copytree(root, second_root)
    manifest = json.loads(canonical(manifest))
    manifest.update(observation_id='observation-0001', sequence=1)
    manifest['proprioception']['end_effectors']['left']['pose_xyz_xyzw'][0] += .01
    (second_root / 'observation.json').write_bytes(canonical(manifest))
    (second_root / 'proprioception.json').write_bytes(canonical(manifest['proprioception']))
    second = store.publish(observation=manifest, observation_root=second_root, cause='after_action',
        task_instruction='fixture task', previous_public_action=None,
        remaining_budget={'robot_decisions':9,'model_calls':19})
    control.update(source_root=str(second_root), observation_id=manifest['observation_id'],
        observation_sha256=digest((second_root/'observation.json').read_bytes()),
        policy_state={'step':1,'state_sha256':second['state_sha256']})
    (host.control_root / 'current-observation.json').write_bytes(canonical(control))
    host.state_root = store.root
    assert verify_policy_state(store.run_root, verify_artifacts=True)['status'] == 'valid'
    return first, second
