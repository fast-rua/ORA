"""Trusted host binding of categorized reads to the Supervisor state owner.

Standalone tool fixtures/hosts without an episode owner import their existing
public publication on demand. Production episodes supply the existing owner;
there is never a second mutable copy of episode state in the data interface.
"""
import json
from ora.agent.toolkit.state.store import StateStore
from ora.agent.toolkit.interface.types import AccessDenied

#: A surface of the one-gripper atomic tools, which keep each ``preview_move`` candidate in ``tool.previews``.
ATOMIC_CAPABILITY = "atomic_pose_gripper_v1"


def seed(host):
    """Every key a tool reads exists before its first call: no marks, nothing held."""
    existing = host.toolkit_state_access.values()
    if "tool.marks" not in existing:
        host.toolkit_state_access.write("tool.marks", {})
    for arm in ("left", "right"):
        key = f"tool.held.{arm}"
        if key not in existing:
            host.toolkit_state_access.write(key, None)
    if host.surface.supports(ATOMIC_CAPABILITY) and "tool.previews" not in existing:
        host.toolkit_state_access.write("tool.previews", {})


def bind(host):
    store = getattr(host, 'toolkit_state_store', None)
    host._toolkit_import_publication = store is None
    if store is None:
        store = StateStore()
        host.toolkit_state_store = store
    tool_grant = 'check_task' in host.allowed_tools
    synchronous = host.surface.supports("toolkit_synchronous_execution_v1")
    if synchronous:
        from ora.agent.composition.surface.toolkit_grants import validate_grants
        grants = validate_grants(host.toolkit_grants, host.surface.capabilities)
        tool_grant = tool_grant and "check_task" in grants["tools"]
        key_grant = "privileged.task_check" in grants["state_keys"]
    else:
        key_grant = host.surface.supports("faithful_check_task")
    host.toolkit_state_access = store.access(
        writer='tool', privileged_tools=('check_task',) if tool_grant else (),
        privileged_keys=('privileged.task_check',) if tool_grant and key_grant else (),
    )
    if synchronous or host.surface.supports("toolkit_only_v1"):
        from ora.agent.toolkit.state.deltas import bind_journal
        from ora.agent.toolkit.state.exec_ledger import bind_exec_ledger
        journal = bind_journal(store, host.control_root.parent)
        if synchronous:  # where the environment plans the moves, no worker reports an execution state
            bind_exec_ledger(store, host.control_root.parent)
        with journal.boundary({"kind": "owner", "operation": "initialize"}):
            seed(host)
    return dict(schema='ora.toolkit.state-grants.v1', reader='robot', privileged_category=False,
                tool_grants=['check_task'] if tool_grant else [],
                key_grants=['privileged.task_check'] if key_grant else [],
                effective_keys=['privileged.task_check'] if tool_grant and key_grant else [],
                source="independent_run_grants" if synchronous else "tool_surface_and_faithful_check_task",
                transitional=True)


def refresh_standalone(host, control, manifest):
    if not hasattr(host, '_toolkit_import_publication') or not host._toolkit_import_publication:
        return
    values = {'env.observation': {key: control[key] for key in ('observation_id', 'observation_sha256')}}
    for arm, value in manifest.get('proprioception', {}).get('end_effectors', {}).items():
        if arm in {'left', 'right'}:
            values[f'env.proprioception.{arm}'] = value
    host.toolkit_state_store.access(reader='supervisor', writer='environment').update(values)


def current(host):
    """Preserve closed-publication refusal, then select identity from current state."""
    control, root, manifest = host._current()
    if not hasattr(host, 'toolkit_state_store'):
        bind(host)
    refresh_standalone(host, control, manifest)
    observed = host.toolkit_state_access.read('env.observation')
    identity = {key: observed[key] for key in ('observation_id', 'observation_sha256')}
    if identity != {key: control[key] for key in identity}:
        raise AccessDenied('state layer and public observation boundary differ')
    return identity, root, manifest


def task_check(host):
    """Read exactly the granted key; standalone compatibility import stays host-owned."""
    if not hasattr(host, 'toolkit_state_store'):
        bind(host)
    if hasattr(host, '_toolkit_import_publication') and host._toolkit_import_publication:
        path = host.control_root / 'policy-task-check.json'
        value = json.loads(path.read_bytes()) if path.is_file() else None
        host.toolkit_state_store.access(reader='supervisor', writer='environment').write(
            'privileged.task_check', value)
    return host.toolkit_state_access.read('privileged.task_check')
