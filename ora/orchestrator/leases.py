"""Runtime resource leases: one claim per execution slot, held by the run that owns it.

A launcher claims the slot its RunPlan binds (``claim_resource_lease``), hands
the claim to its long-lived owner process (``assign_resource_lease``), and that
owner releases exactly its own claim (``release_resource_lease``). Claims are
atomic, with GPU probes outside the shared registry lock; a lease read never
deletes a lease another node holds (``lease_off_node``).
"""
from __future__ import annotations
import fcntl
import json
import os
import secrets
import socket
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from . import paths
from .preflight.resources import ExecutionResourceError, resolve_execution_resource, resource_lease_key
from .preflight.resource_admission import _resource_refusal as _resource_admission_refusal, _visible_gpu_ids
from .preflight.resource_inventory import _gpu_memory_inventory, lease_owner_pids


LEASE_PATH = paths.STATE_ROOT / "active-run.json"
LEASE_ROOT = paths.STATE_ROOT / "leases"
LEASE_LOCK_PATH = LEASE_ROOT / ".registry.lock"
LEASE_SCHEMA = "ora.runtime-resource-lease.v1"


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _legacy_lease() -> dict[str, Any] | None:
    if not LEASE_PATH.is_file():
        return None
    try:
        value = _load(LEASE_PATH)
        if not _pid_alive(value["pid"]):
            raise ValueError("stale legacy lease")
        return {
            **value,
            "schema": "ora.legacy-active-run-lease.v1",
            "resource_class": "exclusive_host",
            "slot_id": "legacy",
            "lease_path": str(LEASE_PATH),
        }
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        LEASE_PATH.unlink(missing_ok=True)
        return None


def _plan_resource_binding(plan: Mapping[str, Any]) -> dict[str, Any]:
    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("RunPlan runtime is unavailable for resource leasing")
    binding = runtime.get("resource_binding")
    if isinstance(binding, Mapping):
        return dict(binding)
    return resolve_execution_resource(
        None,
        runtime_kind=str(runtime.get("kind", "synthetic")),
        required_host=(
            str(runtime["required_host"])
            if isinstance(runtime.get("required_host"), str)
            else None
        ),
    )


def _read_resource_lease(path: Path) -> dict[str, Any] | None:
    try:
        value = _load(path)
        if value.get("schema") != LEASE_SCHEMA:
            raise ValueError("invalid resource lease schema")
        if lease_off_node(value):  # kept unverified: its PIDs live on another node
            return {**value, "lease_path": str(path)}
        owner_pid = value.get("supervisor_pid") or value.get("launcher_pid")
        if not _pid_alive(owner_pid):
            raise ValueError("stale resource lease")
        return {**value, "lease_path": str(path)}
    except (OSError, ValueError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None


def _active_resource_leases_unlocked() -> list[dict[str, Any]]:
    if not LEASE_ROOT.is_dir():
        return []
    active = []
    for path in sorted(LEASE_ROOT.glob("*.json")):
        lease = _read_resource_lease(path)
        if lease is not None:
            active.append(lease)
    return active


def _active_leases() -> list[dict[str, Any]]:
    active = []
    legacy = _legacy_lease()
    if legacy is not None:
        active.append(legacy)
    active.extend(_active_resource_leases_unlocked())
    return active


def _resource_refusal(binding, active, *, inventory=None, probe_error=None):
    # Retained operator queues preview the same capacity rule as atomic claims.
    error = probe_error
    if inventory is None and error is None and any(
            _visible_gpu_ids(binding) & _visible_gpu_ids(lease) for lease in active):
        try:
            inventory = _gpu_memory_inventory(lease_owner_pids(LEASE_ROOT))
        except ExecutionResourceError as exc:
            error = str(exc)
    return _resource_admission_refusal(binding, active, inventory=inventory, probe_error=error)


# Retained source-bound operator queues use the previous helper name.
_resource_conflict = _resource_refusal


def _claim_binding_lease(
    binding: Mapping[str, Any], *, run_id: str, plan_sha256: str, launcher_pid: int,
    lease_root: Path, lock_path: Path, schema: str,
    active_leases: Callable[[], list[Mapping[str, Any]]],
    memory_inventory: Callable[..., Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Recheck active claims under the lock after one bounded device probe.

    The probe is a snapshot, never an admission decision. Only keys still in
    the locked active set receive memory credit, and a new neighbour is charged
    its entire declared floor. Competing claims cannot bypass each other.
    """
    lease_root.mkdir(parents=True, exist_ok=True)
    inventory = None
    probe_error = None
    probed = False
    while True:
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            active = active_leases()
            shared = bool(_visible_gpu_ids(binding)) and any(
                _visible_gpu_ids(lease) & _visible_gpu_ids(binding) for lease in active)
            if probed or not shared:
                refusal = _resource_admission_refusal(binding, active, inventory=inventory,
                                                      probe_error=probe_error)
                if refusal is not None:
                    raise RuntimeError(refusal)
                return _write_lease(binding, run_id=run_id, plan_sha256=plan_sha256,
                                    launcher_pid=launcher_pid, lease_root=lease_root,
                                    schema=schema)
        probed = True
        try:
            inventory = memory_inventory(lease_owner_pids(lease_root))
        except ExecutionResourceError as exc:
            probe_error = str(exc)


def _write_lease(binding: Mapping[str, Any], *, run_id: str, plan_sha256: str,
                 launcher_pid: int, lease_root: Path, schema: str) -> dict[str, Any]:
    """Write one admitted claim; caller holds the registry lock."""
    path = lease_root / (resource_lease_key(binding) + ".json")
    lease = {
        "schema": schema, "run_id": run_id, "plan_sha256": plan_sha256,
        "resource_class": binding["resource_class"], "slot_id": binding["slot_id"],
        "profile_id": binding["profile_id"], "host_id": binding.get("host_id"),
        "runtime_kind": binding.get("runtime_kind"),
        "worker_environment": dict(binding.get("worker_environment", {})),
        "minimum_free_memory_mib": binding.get("minimum_free_memory_mib"),
        "profile_sha256": binding.get("profile_sha256"),
        "claim_token": secrets.token_hex(16), "launcher_pid": int(launcher_pid),
        "supervisor_pid": None, "created_unix_s": time.time(),
        # The launcher and its supervisor run here. Only this node can tell
        # whether their PIDs are alive; see lease_off_node.
        "node": socket.gethostname(),
    }
    # Publish the claim whole. A reader outside the registry lock (preflight,
    # status, a queue waiting on its runs) prunes a lease it cannot parse, and
    # once pruned a claim between its O_EXCL create and its JSON write, so the
    # launcher found its own lease gone. The
    # temporary name is not *.json; the hard link refuses an existing claim
    # exactly as O_EXCL did.
    temporary = lease_root / f".{path.name}.{lease['claim_token']}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(lease, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**lease, "lease_path": str(path)}


def lease_off_node(lease: Mapping[str, Any]) -> str | None:
    """The node a lease was claimed on, when that is not this node.

    A PID is only meaningful on the node that owns it: a lease scan from a login
    or scan node once judged every live cluster lease stale by a local PID and
    deleted it, and a queue then abandoned 22 running episodes.
    """
    node = lease.get("node")
    return node if isinstance(node, str) and node and node != socket.gethostname() else None


def _create_lease(
    plan: Mapping[str, Any], run_id: str, launcher_pid: int,
) -> dict[str, Any]:
    paths.STATE_ROOT.mkdir(parents=True, exist_ok=True)
    return _claim_binding_lease(_plan_resource_binding(plan), run_id=run_id,
        plan_sha256=plan["plan_sha256"], launcher_pid=launcher_pid,
        lease_root=LEASE_ROOT, lock_path=LEASE_LOCK_PATH, schema=LEASE_SCHEMA,
        active_leases=_active_leases, memory_inventory=_gpu_memory_inventory)


def _assign_lease(lease: Mapping[str, Any], supervisor_pid: int) -> dict[str, Any]:
    path = Path(str(lease["lease_path"]))
    with LEASE_LOCK_PATH.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = _load(path)
        if (
            current.get("run_id") != lease.get("run_id")
            or current.get("claim_token") != lease.get("claim_token")
        ):
            raise RuntimeError("resource lease changed before supervisor assignment")
        assigned = {**current, "supervisor_pid": int(supervisor_pid)}
        _atomic(path, assigned)
        return {**assigned, "lease_path": str(path)}


def _lease_for_run(run_id: str) -> dict[str, Any] | None:
    return next(
        (lease for lease in _active_leases() if lease.get("run_id") == run_id),
        None,
    )


def _release_lease(
    run_id: str, *, supervisor_pid: int | None = None,
    claim_token: str | None = None,
) -> None:
    paths.STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LEASE_ROOT.mkdir(parents=True, exist_ok=True)
    with LEASE_LOCK_PATH.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        lease = _lease_for_run(run_id)
        if lease is None:
            return
        if lease.get("schema") == "ora.legacy-active-run-lease.v1":
            if (
                supervisor_pid is not None
                and int(lease.get("pid", -1)) == supervisor_pid
            ):
                LEASE_PATH.unlink(missing_ok=True)
            return
        if supervisor_pid is not None and int(
            lease.get("supervisor_pid", -1)
        ) != supervisor_pid:
            return
        if claim_token is not None and lease.get("claim_token") != claim_token:
            return
        Path(str(lease["lease_path"])).unlink(missing_ok=True)


def claim_resource_lease(
    plan: Mapping[str, Any], owner_id: str, launcher_pid: int,
) -> dict[str, Any]:
    """Atomically claim a source-resolved Runtime or resident-service resource."""

    return _create_lease(plan, owner_id, launcher_pid)


def assign_resource_lease(
    lease: Mapping[str, Any], owner_pid: int,
) -> dict[str, Any]:
    """Transfer a claimed resource lease to its long-lived owner process."""

    return _assign_lease(lease, owner_pid)


def release_resource_lease(
    owner_id: str, *, owner_pid: int | None = None,
    claim_token: str | None = None,
) -> None:
    """Release exactly one matching resource lease without broad cleanup."""

    _release_lease(
        owner_id, supervisor_pid=owner_pid, claim_token=claim_token,
    )


def resource_lease_for_owner(owner_id: str) -> dict[str, Any] | None:
    """Return the active exact-owner lease, including its internal claim token."""

    return _lease_for_run(owner_id)


def _public_lease(lease: Mapping[str, Any]) -> dict[str, Any]:
    """Return operator-visible ownership without exposing the claim token."""
    return {
        key: value for key, value in lease.items()
        if key not in {"claim_token", "lease_path"}
    }
