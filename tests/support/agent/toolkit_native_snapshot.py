"""Synthetic control/physics and real public pixels; never starts a simulator."""
from copy import deepcopy

from ora.contracts import validate_runtime_action
from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.journal import Journal
from ora.agent.toolkit.interface.runtime import Runtime
from ora.agent.toolkit.interface.types import PublicData, canonical, decode, digest
from ora.agent.toolkit.state.snapshot import file_ref
from ora.orchestrator.episode.tool_execution_log import ToolExecutionLog
from ora.orchestrator.toolkit.snapshot_boundary import select_snapshot
from tests.support.agent.toolkit_snapshot import make_source


def make_native(source, monkeypatch, *, action_kind="move_pose", request_arguments=None, ledger_transform=None):
    make_source(source, monkeypatch)
    (source / "plan.json").write_bytes(canonical(dict(runtime=dict(kind="robodojo"))))
    native = source / "native"
    native.mkdir(exist_ok=True)
    (native / "reset.json").write_bytes(canonical(dict(physics_steps_before_policy=10)))
    (native / "source-manifest.json").write_bytes(canonical({}))
    previous = None
    for step in range(2):
        path = source / f"state/steps/step-{step:04d}/snapshot.json"
        value = decode(path.read_bytes())
        manifest = decode((source / value["values"]["env.observation"]["manifest"]["path"]).read_bytes())
        arms = manifest["proprioception"]["end_effectors"]
        values = value["values"]
        values.update({"exec.native_actions": dict(used=step * 2, limit=1000),
                       "exec.world_mutation_sequence": step,
                       "exec.tool_calls_with_actions": step,
                       "exec.round": dict(index=0, task_instruction="fixture")})
        for arm, measured in arms.items():
            anchor = {key: measured[key] for key in ("arm_joint_qpos", "pose_xyz_xyzw")}
            values[f"exec.command.{arm}"] = dict(pose_target_xyz_xyzw=measured["pose_xyz_xyzw"],
                                                 gripper_directive=measured["gripper_command"])
            values[f"exec.recovery.{arm}.home"] = anchor
            values[f"exec.recovery.{arm}.previous"] = dict(anchor, arm_joint_qpos=[-99] * 6)
            values[f"exec.session_home.{arm}"] = anchor
        value["previous_snapshot_sha256"] = previous
        value.pop("snapshot_sha256")
        previous = value["snapshot_sha256"] = digest(canonical(value))
        path.chmod(0o644)
        path.write_bytes(canonical(value) + b"\n")
    clocks = []
    journal = Journal(source)
    for index in range(2):
        boundary = select_snapshot(source, at=str(index))
        observed = boundary.state.access(reader="rsi").read("env.observation")
        manifest = decode((source / observed["manifest"]["path"]).read_bytes())
        arms = manifest["proprioception"]["end_effectors"]
        action_id = f"native-{index}"
        pose = dict(position_m=[0.1, 0.2, 1.0], orientation_xyzw=[0.0, 0.0, 0.0, 1.0])
        arguments = (dict(targets={"left": pose}, interactions={"left": "free_space"})
                     if action_kind in {"move_pose", "move_free_space"} else
                     dict(targets={"left": "open"}) if action_kind == "set_gripper" else
                     dict(targets={"left": "home"}) if action_kind == "recover_motion" else
                     dict(ticks=2))
        action = dict(schema="ora-motion-action.v3", action_id=action_id, action=action_kind,
                      arguments=deepcopy(request_arguments if request_arguments is not None else arguments),
                      observation_id=observed["observation_id"], bound_observation_sha256=observed["observation_sha256"])
        traces, lines = [], []
        for number in (index * 2 + 1, index * 2 + 2):
            control = dict(left_arm_joint_state=dict(position=[number / 100] * 6))
            traces.append(dict(native_action=number, physics_steps=number * 10,
                               segment="move", control=control, measured=deepcopy(arms)))
            for tick in range(10):
                lines.append(dict(action_id=action_id, native_action=number,
                                  physics_clock=10 + (number - 1) * 10 + tick + 1,
                                  segment="move", measured=deepcopy(arms), command=control,
                                  applied_position_targets=dict(left_arm_joint_state=[number / 100] * 6)))
        path = native / f"controller/action-{index:04d}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"".join(canonical(row) + b"\n" for row in lines))
        trace = dict(schema="ora.robodojo-controller-trace.v1", action_id=action_id, samples=traces,
                     physics_journal=dict(path=str(path.relative_to(native)), sha256=digest(path.read_bytes())))
        trace_path = native / f"controller/action-{index:04d}.json"
        trace_path.write_bytes(canonical(trace))
        receipt = dict(action_id=action_id, action=action_kind, round_index=0, native_actions=2,
                       motion_strategy="global", target_arms=["left"], before=deepcopy(arms),
                       pose_targets_after={"left": [999] * 7}, command_targets_after={"future": True},
                       execution_status="completed", control_trace=dict(path=str(trace_path.relative_to(native)),
                                                                         sha256=digest(trace_path.read_bytes())))
        receipt_path = native / f"receipts/action-{index:04d}.json"
        receipt_path.parent.mkdir(exist_ok=True)
        receipt_path.write_bytes(canonical(receipt))
        public = dict(action_id=action_id, native_actions=2)
        public_path = native / f"receipt-event-{index}.json"
        public_path.write_bytes(canonical(dict(receipt=receipt, path=str(receipt_path.relative_to(native)),
                                               sha256=digest(receipt_path.read_bytes()))))
        log = ToolExecutionLog(source, f"t-{index}", f"m-{index}")
        before = dict(step=index, observation_id=observed["observation_id"])
        log.create(requested=20, admitted=20, observation=before)
        owner_action = validate_runtime_action(action)
        if ledger_transform is not None:
            ledger_transform(owner_action)
        log.begin(action=owner_action, admitted=20, observation=before)
        log.complete(used=2, observation=before, receipt=public, receipt_event_id=f"e-{index}",
                     receipt_ref=dict(path=str(public_path.relative_to(source)), sha256=digest(public_path.read_bytes())))
        log.finish()
        with (source / "agent-events.jsonl").open("ab") as stream:
            stream.write(canonical(dict(seq=4 + index, event_id=f"e-{index}", type="environment_receipt",
                                       payload=dict(action_id=action_id), artifact_ref=log.value["actions"][0]["receipt_ref"])) + b"\n")
        runtime = Runtime(context=boundary.context, store=BlobStore(source, writable=True), journal=journal, bindings={},
                          adapters={"execute_action": lambda p, i, s: PublicData.from_json({"receipt": public})})
        runtime.execute_action(action=action, budget=20)
        clocks.append(dict(observation_sequence=manifest["sequence"], before_physics_steps=10 + index * 20,
                           after_physics_steps=10 + index * 20, read_only=True))
    (native / "observation-clocks.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in clocks))
    return source
