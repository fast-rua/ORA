"""Complete publication snapshots, linked to the existing policy-state chain.

Loading reads only recorded files; it never starts a simulator or service.
Images and measurements remain in policy-observations, in their original bytes.
"""

from dataclasses import dataclass
import hashlib
import json
import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .keys import StateError, key_definition
from .store import StateAccess, StateStore, encode

SCHEMA = "ora.toolkit-state.v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path(root: Path, name: str, prefix: str) -> Path:
    parts = PurePosixPath(name)
    if (parts.is_absolute() or ".." in parts.parts or not name.startswith(prefix)
            or "\\" in name):
        raise StateError(f"state artifact outside {prefix}: {name}")
    target = root / name
    if any(part.is_symlink() for part in (target, *target.parents) if part != root):
        raise StateError(f"state artifact cannot be a symlink: {name}")
    if not target.resolve().is_relative_to(root.resolve()):
        raise StateError(f"state artifact escapes run: {name}")
    return target


def file_ref(root: Path, name: str, *, prefix: str = "policy-observations/") -> dict[str, Any]:
    """Supervisor-only publication helper; no arbitrary tool filesystem API."""
    data = _path(root, name, prefix).read_bytes()
    return {"path": name, "sha256": _sha(data), "bytes": len(data)}


def _read_ref(root: Path, ref: Mapping[str, Any], prefix: str) -> bytes:
    data = _path(root, ref["path"], prefix).read_bytes()
    if _sha(data) != ref["sha256"] or len(data) != ref["bytes"]:
        raise StateError(f"state artifact hash/size drift: {ref['path']}")
    return data


def _identity(value: Mapping[str, Any], field: str) -> None:
    if value.get(field) != _sha(encode({k: v for k, v in value.items() if k != field})):
        raise StateError(f"state identity mismatch: {field}")


def _snapshot_path(step: int) -> str:
    if type(step) is not int or step < 0:
        raise StateError("state step must be a nonnegative integer")
    return f"state/steps/step-{step:04d}/snapshot.json"


def _artifacts(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs = []
    observation = values.get("env.observation", {})
    for field in ("manifest", "proprioception_file"):
        if field in observation:
            refs.append(observation[field])
    for key, value in values.items():
        if key.startswith("env.camera."):
            refs.extend(value.get("files", {}).values())
    # tool.* and privileged.* JSON are never interpreted as filesystem paths.
    return sorted({ref["path"]: ref for ref in refs}.values(), key=lambda ref: ref["path"])


def _boundary(values: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    observed = values.get("env.observation", {})
    reference = policy["observation"]
    if (policy.get("schema") != "ora.policy-state.v1"
            or observed.get("step") != policy["step"]
            or observed.get("observation_id") != reference["id"]
            or observed.get("observation_sha256") != reference["manifest"]["sha256"]
            or observed.get("manifest") != reference["manifest"]):
        raise StateError("environment state is not bound to this policy publication")
    for camera, record in policy.get("cameras", {}).items():
        if values.get(f"env.camera.{camera}") != record:
            raise StateError("camera state differs from the policy publication")


def export_snapshot(store: StateStore, run_root: Path, policy_ref: Mapping[str, Any]) -> dict[str, Any]:
    """Persist all namespaces at an existing complete publication boundary."""
    if not store.complete_boundary:
        raise StateError("state is incomplete between action and publication")
    step = policy_ref["step"]
    name = _snapshot_path(step)
    policy = json.loads(_read_ref(run_root, policy_ref, "policy-state/steps/"))
    _identity(policy, "state_sha256")
    if policy["step"] != step or policy_ref["state_sha256"] != policy["state_sha256"]:
        raise StateError("policy-state boundary mismatch")
    previous = None
    if step:
        previous = json.loads((run_root / _snapshot_path(step - 1)).read_bytes())
        _identity(previous, "snapshot_sha256")
    if policy.get("previous_state_sha256") != (previous["policy_state"]["state_sha256"] if previous else None):
        raise StateError("snapshot/policy-state chains are not aligned")
    values = store.access(reader="supervisor").values()
    _boundary(values, policy)
    artifacts = _artifacts(values)
    for ref in artifacts:
        _read_ref(run_root, ref, "policy-observations/")
    body = {
        "schema": SCHEMA, "step": step, "run_id": policy["run_id"],
        "policy_state": dict(policy_ref), "values": values, "artifacts": artifacts,
        "previous_snapshot_sha256": previous["snapshot_sha256"] if previous else None,
    }
    if store.tool_journal is not None:
        body["tool_state_watermark"] = store.tool_journal.watermark()
    result = {**body, "snapshot_sha256": _sha(encode(body))}
    target = _path(run_root, name, "state/steps/")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".snapshot-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encode(result) + b"\n")
        temporary.chmod(0o444)
        os.link(temporary, target)  # Atomic publication; an existing snapshot is never overwritten.
    finally:
        temporary.unlink(missing_ok=True)
    return result


@dataclass(frozen=True)
class LoadedState:
    """Read-only state and hash-checked published bytes, without a simulator."""

    root: Path
    store: StateStore
    artifacts: tuple[bytes, ...]

    def access(self, *, reader: str = "robot", privileged_keys: tuple[str, ...] = (),
               privileged_tools: tuple[str, ...] = ()) -> StateAccess:
        return self.store.access(reader=reader, privileged_keys=privileged_keys,
                                 privileged_tools=privileged_tools)

    def read_artifact(self, name: str) -> bytes:
        """Return original bytes only for a registered public observation file."""
        for encoded in self.artifacts:
            ref = json.loads(encoded)
            if ref["path"] == name:
                return _read_ref(self.root, ref, "policy-observations/")
        raise StateError(f"not a published state artifact: {name}")


def _chain(run_root: Path, step: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Validate both chains from step 0 to ``step`` once; the (snapshot, policy-state) documents in order."""
    _snapshot_path(step)
    previous_hash = previous_policy = None
    documents = []
    for index in range(step + 1):
        snapshot = json.loads(_path(run_root, _snapshot_path(index), "state/steps/").read_bytes())
        _identity(snapshot, "snapshot_sha256")
        if snapshot["schema"] != SCHEMA or snapshot["step"] != index:
            raise StateError("unsupported state schema or wrong step")
        policy = json.loads(_read_ref(run_root, snapshot["policy_state"], "policy-state/steps/"))
        _identity(policy, "state_sha256")
        if (snapshot["previous_snapshot_sha256"] != previous_hash
                or policy["previous_state_sha256"] != previous_policy
                or policy["state_sha256"] != snapshot["policy_state"]["state_sha256"]
                or policy["step"] != index or policy["run_id"] != snapshot["run_id"]):
            raise StateError("snapshot/policy-state chain mismatch")
        previous_hash, previous_policy = snapshot["snapshot_sha256"], policy["state_sha256"]
        documents.append((snapshot, policy))
    return documents


def _loaded(run_root: Path, snapshot: Mapping[str, Any], policy: Mapping[str, Any], *,
            keys: Callable[[str], bool] | None = None, read_artifacts: bool = True) -> LoadedState:
    _boundary(snapshot["values"], policy)
    if _artifacts(snapshot["values"]) != snapshot["artifacts"]:
        raise StateError("snapshot artifact registry mismatch")
    store = StateStore(read_only=True)
    for key, value in snapshot["values"].items():
        key_definition(key)
        if keys is None or keys(key):
            store._values[key] = encode(value)
    if read_artifacts:
        for ref in snapshot["artifacts"]:
            _read_ref(run_root, ref, "policy-observations/")
    store.complete_boundary = True
    store.tool_watermark = snapshot.get("tool_state_watermark")
    return LoadedState(run_root.resolve(), store, tuple(encode(ref) for ref in snapshot["artifacts"]))


def _observation_key(key: str) -> bool:
    return key == "env.observation" or key.startswith("env.camera.")


def load_snapshot(run_root: Path, step: int) -> LoadedState:
    """Validate both chains and the selected boundary's files; return read-only state."""
    snapshot, policy = _chain(run_root, step)[-1]
    return _loaded(run_root, snapshot, policy)


def load_publications(run_root: Path, step: int) -> list[LoadedState]:
    """Every publication from step 0 to ``step`` from one validation of both chains.

    The last is ``load_snapshot(run_root, step)``. The earlier ones keep only their published observation
    (``env.observation``, ``env.camera.*``), never their tool values, and their observation files are not read
    here: a tool that reads one gets it through ``read_artifact`` or the blob store, which check its hash then.
    Loading each step with ``load_snapshot`` instead re-validated the chain from step 0 and re-read every file
    for every step, so a replay late in an episode spent up to ten minutes before running the tool.
    """
    documents = _chain(run_root, step)
    earlier = [_loaded(run_root, snapshot, policy, keys=_observation_key, read_artifacts=False)
               for snapshot, policy in documents[:-1]]
    return [*earlier, _loaded(run_root, *documents[-1])]
