"""Restore/render one native tick, then detach read-only snapshot capabilities."""
from dataclasses import replace

from ora.agent.toolkit.interface.types import CallContext, CorruptRecord, Divergence, NotRecorded, canonical, decode
from ora.agent.toolkit.state.snapshot import LoadedState, file_ref
from ora.agent.toolkit.state.store import StateStore, encode
from ora.orchestrator.backends.registry import backend
from .live_boundary import verified
from .native_boundary import freeze
from .selection import read_local


def prepare(session, source, output, selected, boundary):
    """Fresh pixels are the new observation. Rendering must not advance physics."""
    expected = boundary["expected"]
    response = session.diagnostic("prepare", destination=str(output / "policy-observations"))
    manifest = response["observation"]
    oid = manifest["observation_id"]
    if not isinstance(oid, str) or "/" in oid or "\\" in oid or oid in (".", ".."):
        raise CorruptRecord("invalid fresh observation identity")
    name = f"policy-observations/{oid}/observation.json"
    ref = file_ref(output, name)
    if decode(verified(output, ref)) != manifest:
        raise CorruptRecord("rendered manifest differs from its persisted bytes")
    after = session.diagnostic("verify_state")
    for key in ("round_index", "physics_clock", "native_actions", "sim_steps", "proprioception"):
        if after.get(key) != expected[key]:
            raise Divergence(f"fresh rendering changed physical boundary: {key}")
    if manifest["proprioception"] != expected["proprioception"] or manifest["round_index"] != expected["round_index"]:
        raise Divergence("fresh observation robot state differs from verified native tick")
    session.diagnostic("authorize")
    status = session.status()
    reason = status.get("reason")
    observation = dict(observation_id=oid, observation_sha256=ref["sha256"],
                       step=selected.step, round_index=expected["round_index"], manifest=ref,
                       proprioception_file=file_ref(output, f"policy-observations/{oid}/proprioception.json"))
    if decode(verified(output, observation["proprioception_file"])) != manifest["proprioception"]:
        raise CorruptRecord("fresh proprioception file differs from rendered manifest")
    store = StateStore()
    # Only tool state and last-completed execution evidence are carried across.
    values = selected.state.access(reader="rsi").values()
    store._values = {key: encode(value) for key, value in values.items()
                     if key.startswith(("tool.", "exec."))}
    owner_values = session.execution_state_owner.store.access(reader="rsi").values()
    missing = []
    if expected["native_actions"]:
        for arm in ("left", "right"):
            key = f"exec.command.{arm}"
            if key in owner_values:
                owner_values[key] = dict(owner_values[key])
                owner_values[key].pop("pose_target_xyz_xyzw", None)
                missing.append(dict(field=f"{key}.pose_target_xyz_xyzw",
                                    reason="not_recorded_per_tick; no final-action target fallback"))
    # The diagnostic owner has no parent invocation counter of its own.
    owner_values.pop("exec.tool_calls_with_actions", None)
    store.access(reader="supervisor", writer="execution").update(owner_values)
    environment = {
        "env.observation": observation,
        "env.terminal": dict(terminal=status["terminal"], reason=(
            "official_boundary" if isinstance(reason, str) and reason.startswith("official_") else reason)),
        "privileged.task_check": dict(terminal=status["terminal"], reason=reason,
                                       task_success=bool(status["terminal"] and reason == "official_success"),
                                       source="restored_simulator"),
    }
    for arm, value in manifest["proprioception"]["end_effectors"].items():
        environment[f"env.proprioception.{arm}"] = value
    refs = [ref, observation["proprioception_file"]]
    root = output / "policy-observations" / oid
    for camera, record in manifest["cameras"].items():
        files = {}
        for role, original in record["files"].items():
            verified(root, original)
            files[role] = file_ref(output, str((root / original["path"]).relative_to(output)))
        environment[f"env.camera.{camera}"] = {**record, "files": files}
        refs.extend(files.values())
    if boundary.get("ground_truth") is not None:
        environment["privileged.ground_truth"] = boundary["ground_truth"]
    store.access(reader="supervisor", writer="environment").update(environment)
    store.complete_boundary = True
    count = expected["native_actions"]
    context = CallContext.create(run_id=selected.context.json()["run_id"], round_index=expected["round_index"],
                                 tool_call_id=f"snapshot-native-{count}",
                                 boundary=dict(kind="after_native_action", native_action=count),
                                 state=dict(observation={key: observation[key] for key in
                                                         ("observation_id", "observation_sha256")},
                                            native_action=count, physics_clock=expected["physics_clock"]), check_task=True)
    loaded = LoadedState(output, store, tuple(encode(ref) for ref in refs))
    result = replace(selected, state=store, context=context, selected=None, publications=[loaded])
    (output / "native-state.json").write_bytes(canonical(dict(
        schema="ora.toolkit.native-state.v1", context=context.json(), values=store.access(reader="rsi").values(),
        source_snapshot_sha256=selected.snapshot_sha256, action_mapping=boundary["action_mapping"],
        sensors="fresh_render_not_recorded_bytes", complete_action_boundary=False,
        missing_state_fields=missing)) + b"\n")
    return result


def restore(source, output, number, profile, progress):
    """Release the diagnostic worker before any user script executes."""
    if not profile:
        raise ValueError("--native-action requires an already allocated --resource-profile")
    selected, frozen = freeze(source, output, number)
    plan = decode(read_local(source, "plan.json"))
    adapter = backend(plan["runtime"]["kind"])
    if adapter.toolkit_live is None:
        raise NotRecorded("未记录: backend has no native snapshot restoration")
    progress.finish()
    from .live_host import interruptible
    with interruptible(), adapter.toolkit_live(source=source, output=output, boundary=frozen, profile=profile,
                              execute_action=False, progress=progress) as session:
        return prepare(session, source, output, selected, frozen)
