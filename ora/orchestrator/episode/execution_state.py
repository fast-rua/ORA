"""Supervisor ownership of RoboDojo execution state and publication snapshots."""

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from ora.agent.toolkit.state.store import StateStore
from ora.agent.toolkit.state.snapshot import export_snapshot, file_ref
from ora.contracts.execution_state import SCHEMA, recovery_goals
from ora.contracts.joint_trajectory import MAX_TRAJECTORY_NATIVE_ACTIONS
from .native_allowance import receipt_native_actions

#: Backends whose worker takes Supervisor-owned execution state and so caps each action at the budget the
#: Supervisor lends it (``limited_action``, at most one primitive's). The coordinator binds that state for exactly
#: these; code mode refuses a committed skill sequence on them at resolution (``spec/native_policy.py``), since its
#: stages cannot fit in one capped action.
PER_ACTION_CAPPED_BACKENDS = frozenset({"robodojo"})


def per_action_capped(kind: Any) -> bool:
    """Whether a backend's worker caps each action, i.e. whether the coordinator binds execution state for it."""
    return kind in PER_ACTION_CAPPED_BACKENDS


class ExecutionStateOwner:
    """RPC callbacks carry values; the environment transport owns no controller state."""

    def __init__(self, store: StateStore | None = None):
        self.store = store if store is not None else StateStore()
        self.writer = self.store.access(reader="supervisor", writer="execution")
        self.writer.update({"exec.tool_calls_with_actions": 0})
        self.action_budget = None
        self.tool_call = None

    def finish_tool_call(self, receipt: Mapping[str, Any]) -> None:
        if (receipt_native_actions(receipt) or 0) > 0:
            count = self.writer.read("exec.tool_calls_with_actions")
            self.writer.write("exec.tool_calls_with_actions", count + 1)

    @contextmanager
    def limited_action(self, budget: int, tool_call: Mapping[str, Any] | None = None):
        """Borrow a per-action limit for one synchronous request, never persist it.

        ``tool_call`` (``tool_call_id``, ``tool``) names the tool invocation the action belongs to (inside a
        code-mode script, the script's own call of the tool, ``<call>#NNN``); the request lends it with the
        ordinal of the robot decision it belongs to, so the worker can keep one undo entry per arm-moving tool
        call. That is this round's count of tool calls with actions, plus one until the parent's first spend
        counted it (``decision_counted``: a later action of grasp_execute or of a script keeps the ordinal)."""
        if type(budget) is not int or not 1 <= budget <= MAX_TRAJECTORY_NATIVE_ACTIONS:
            raise ValueError(f"action budget must be an integer in 1..{MAX_TRAJECTORY_NATIVE_ACTIONS}")
        if self.action_budget is not None:
            raise RuntimeError("an execution budget is already active")
        self.action_budget = budget
        self.tool_call = None if tool_call is None else {
            "tool_call_id": str(tool_call["tool_call_id"]), "tool": str(tool_call["tool"]),
            "ordinal": int(self.writer.read("exec.tool_calls_with_actions"))
            + (0 if tool_call.get("decision_counted") else 1)}
        try:
            yield
        finally:
            self.action_budget = None
            self.tool_call = None

    def recovery_refusal(self, action: Mapping[str, Any]) -> str | None:
        """Reject changed or invented explicit anchors before recording physical intent."""
        arguments = action.get("arguments", {})
        if "recovery_targets" not in arguments and "recovery" not in arguments:
            return None
        try:
            recovery_goals(action, self.writer.values())
        except ValueError:
            return "stale_or_mismatched_recovery_target"
        return None

    def request(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        values = {k: v for k, v in self.writer.values().items() if k.startswith("exec.")}
        if method in {"execute", "restart_round", "toolkit_restore"}:
            self.store.action_started()
        return {"schema": SCHEMA, "values": values,
                "recovery_targets": recovery_goals(params.get("action", {}), values),
                **({"action_budget": self.action_budget}
                   if method == "execute" and self.action_budget is not None else {}),
                **({"tool_call": dict(self.tool_call)}
                   if method == "execute" and self.tool_call is not None else {})}

    def accept(self, method: str, params: Mapping[str, Any], update: Mapping[str, Any], result: Any) -> None:
        if update.get("schema") != SCHEMA:
            raise ValueError("worker execution-state schema mismatch")
        ledger = self.store.exec_ledger
        previous = dict(self.store._values) if ledger is not None else {}
        self.writer.update(update["values"])
        if ledger is not None:
            # Every IK the request solved (any method) advanced the generator: the exec ledger records each change.
            ledger.accepted(method, previous)
        if method in {"start", "restart_round"}:
            self.writer.write("exec.tool_calls_with_actions", 0)


def bind_execution_state(runtime: Any) -> ExecutionStateOwner:
    """Bind once before worker start; wrappers share the same owner."""
    transport = getattr(runtime, "delegate", runtime)
    owner = getattr(transport, "execution_state_owner", None)
    if owner is None:
        owner = ExecutionStateOwner()
        transport.execution_state_owner = owner
    return owner


def _last_json_line(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(0, 2)
        position, data = stream.tell(), b""
        while position:
            size = min(position, 4096)
            position -= size
            stream.seek(position)
            data = stream.read(size) + data
            rows = data.rstrip(b"\n").split(b"\n")
            if len(rows) > 1 or position == 0:
                return json.loads(rows[-1]) if rows[-1] else None
    return None


def publish_state(loop: Any, policy: Mapping[str, Any], record: Mapping[str, Any], status: Mapping[str, Any]) -> None:
    """Use existing captured files; never trigger rendering, queries or extra policy steps."""
    owner = getattr(loop, "execution_state_owner", None)
    if owner is None:
        return
    store = owner.store
    from ora.agent.toolkit.state.deltas import bind_journal
    from ora.agent.toolkit.state.exec_ledger import bind_exec_ledger
    bind_journal(store, loop.run_root)
    bind_exec_ledger(store, loop.run_root)
    manifest = policy["observation"]["manifest"]
    reason = status["reason"]
    public_reason = (
        "official_boundary" if isinstance(reason, str) and reason.startswith("official_") else reason
    )
    values = {
        "env.observation": {
            "observation_id": policy["observation"]["id"], "observation_sha256": manifest["sha256"],
            "step": policy["step"], "round_index": loop.round_index, "manifest": dict(manifest),
        },
        "env.terminal": {"terminal": status["terminal"], "reason": public_reason},
        "privileged.task_check": {
            "terminal": status["terminal"], "reason": reason,
            "task_success": bool(status["terminal"] and status["reason"] == "official_success"),
            "source": status["source"],
        },
    }
    source = (loop.run_root / manifest["path"]).parent
    proprio = source / "proprioception.json"
    if proprio.is_file():
        values["env.observation"]["proprioception_file"] = file_ref(loop.run_root, proprio.relative_to(loop.run_root).as_posix())
    for camera, value in policy["cameras"].items():
        values[f"env.camera.{camera}"] = dict(value)
    for arm, value in (policy.get("proprioception") or {}).get("end_effectors", {}).items():
        values[f"env.proprioception.{arm}"] = dict(value)
    if loop.plan["runtime"].get("ground_truth_capture") is True:
        path = loop.run_root.parent.parent / "ground-truth" / loop.run_root.name / "ticks.jsonl"
        latest = _last_json_line(path)
        # The worker writes the tick synchronously. Never import a future round.
        count = owner.writer.values().get("exec.native_actions", {}).get("used")
        values["privileged.ground_truth"] = (
            latest if latest is not None and latest.get("round_index") == loop.round_index
            and latest.get("native_action") == count else None
        )
    store.access(reader="supervisor", writer="environment").update(values)
    owner.writer.update({
        "exec.last_receipt": policy.get("previous_execution_receipt"),
        "exec.round": {"index": loop.round_index, "task_instruction": loop.task_instruction},
    })
    store.publication_completed()
    ref = {**record, "bytes": (loop.run_root / record["path"]).stat().st_size}
    export_snapshot(store, loop.run_root, ref)
