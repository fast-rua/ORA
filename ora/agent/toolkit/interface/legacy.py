"""Conservative snapshot.v0 adapter: explicit public references, no guessed probes."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .api import EXTERNAL, Methods
from .blobs import BlobStore
from .comparison_log import append_comparison
from .gaps import require_no_gaps
from .projections import control_result, receipt
from .runtime import envelope, request_wire
from .missing import missing_request
from .types import AccessDenied, CallContext, CorruptRecord, Divergence, NotRecorded, PublicData, Response, ServiceUnavailable, canonical, digest


def read_snapshot_reference(root: Path, ref: dict) -> bytes:
    """Trusted importer reads one explicit evidence ref, never exposes a file API."""
    if not ref or ref.get("recorded") is False or not ref.get("sha256"):
        raise NotRecorded("未记录: missing snapshot reference")
    root = root.resolve()
    raw = Path(ref["path"])
    if ".." in raw.parts:
        raise AccessDenied("snapshot reference contains parent traversal")
    path = (root / raw).resolve()
    if not path.is_relative_to(root):
        raise AccessDenied("snapshot evidence escapes its source run")
    data = path.read_bytes()
    if digest(data) != ref["sha256"] or len(data) != ref["bytes"]:
        raise CorruptRecord("snapshot reference hash/length drift")
    return data


def public_reference(store: BlobStore, ref: dict) -> dict:
    read_snapshot_reference(store.root, ref)
    name = (store.root / ref["path"]).resolve().relative_to(store.root).as_posix()
    media = "image/png" if name.endswith(".png") else "application/x-npy" if name.endswith(".npy") else "application/json"
    return store.register(dict(path=name, sha256=ref["sha256"], bytes=ref["bytes"], media_type=media))


class LegacyReplay(Methods):
    """Recover published files; unsupported historical requests stay NotRecorded.

    External computation uses the same verified inputs through an independent
    service Runtime; old tool/manifest responses are not bare client envelopes.
    """

    def __init__(self, *, snapshot: dict, store: BlobStore, context: CallContext,
                 evaluation, bindings: dict) -> None:
        if snapshot.get("schema") != "rsi.traj.snapshot.v0":
            raise CorruptRecord("unsupported snapshot schema")
        if snapshot["boundary"]["kind"] != "before_call":
            raise NotRecorded("未记录: native-action snapshot has no interface cursor")
        if set(evaluation.adapters) - EXTERNAL or evaluation.store.root == store.root:
            raise AccessDenied("legacy replay can only recompute external services in a separate output")
        self.snapshot = json.loads(canonical(snapshot))
        require_no_gaps(store.root, context.json()["tool_call_id"])
        self.store, self.context, self.evaluation, self.bindings = store, context, evaluation, bindings
        evaluation.store.input_store = store
        observation = snapshot["available_at_boundary"]["observation"]
        self.observation_ref = dict(observation_id=observation["id"],
                                    observation_sha256=observation["manifest"]["sha256"])
        if context.json()["state"].get("observation") != self.observation_ref:
            raise CorruptRecord("snapshot/context observation mismatch")
        self.manifest_ref = public_reference(store, observation["manifest"])
        manifest = json.loads(store.read(self.manifest_ref))
        if (manifest.get("observation_id") != observation["id"]
                or manifest.get("round_index") != context.json()["round_index"]):
            raise CorruptRecord("manifest identity/round mismatch")
        base = str(Path(self.manifest_ref["path"]).parent)
        declared = {f"{base}/proprioception.json": None}
        for camera in manifest.get("cameras", {}).values():
            for source in camera.get("files", {}).values():
                declared[f"{base}/{source['path']}"] = source
        self.files = []
        for ref in observation.get("files", []):
            if ref.get("recorded") is False:
                continue
            path = (store.root / ref["path"]).resolve().relative_to(store.root).as_posix()
            if path not in declared:
                continue  # Display-only snapshot evidence is not a readable blob.
            if declared[path] is not None and any(ref[key] != declared[path][key] for key in ("sha256", "bytes")):
                raise CorruptRecord("published camera file differs from manifest")
            self.files.append(public_reference(store, ref))

    def read_blob(self, ref: dict) -> bytes:
        """Read granted historical/evaluation bytes, preserving missing evidence errors."""
        return self.evaluation.store.read(ref)

    def save_array(self, value: object) -> dict:
        """Persist derived inputs only to the evaluation root; no source modification."""
        return self.evaluation.store.put_array(value)

    def _invoke(self, method: str, parameters: dict, inputs: dict | None = None) -> Response:
        inputs = inputs or {}
        if method == "check_task" and not self.context.json()["check_task"]:
            raise AccessDenied("check_task capability disabled")
        if method in EXTERNAL:
            if method not in self.evaluation.adapters:
                raise ServiceUnavailable(f"service unavailable: {method}; supply --service-report")
            actual = self.evaluation._invoke(method, parameters, inputs)
            comparison = dict(status="unavailable", historical_invocation_id=None,
                              reference_sha256=None, actual_sha256=digest(actual.payload),
                              scope="semantic_envelope_bytes", reason="legacy_response_scope_not_equivalent")
            append_comparison(self.evaluation, comparison)
            return Response(actual.payload, canonical(comparison))
        wire = request_wire(method, parameters, inputs, context=self.context, bindings=self.bindings)
        public = None
        if method in ("observation", "proprioception"):
            if wire["identity"]["parameters"]["observation"] != self.observation_ref:
                raise NotRecorded("未记录: another publication is not this boundary")
            if method == "observation":
                public = PublicData.from_json(dict(observation=self.observation_ref,
                                                   manifest=self.manifest_ref, files=self.files), sources=[self.manifest_ref])
            else:
                refs = [ref for ref in self.files if ref["path"].endswith("/proprioception.json")]
                if len(refs) != 1:
                    raise NotRecorded("未记录: no unique published proprioception")
                public = PublicData.from_json(dict(proprioception=refs[0]), sources=refs)
        elif method == "check_task":
            evidence = self.snapshot["replay_evidence"]
            if evidence["tool_request"]["tool"] == "check_task" and evidence.get("tool_response"):
                ref = evidence["tool_response"]
                path = (self.store.root / ref["path"]).resolve().relative_to(self.store.root).as_posix()
                if not re.fullmatch(r"native/agent-artifacts/[0-9a-f]{64}\.json", path):
                    raise NotRecorded("未记录: checker result is not an explicit tool artifact")
                raw = json.loads(read_snapshot_reference(self.store.root, ref))
                public = control_result(raw)
        elif method == "receipt":
            matches = []
            for item in self.snapshot["available_at_boundary"].get("previous_receipts", []):
                for ref in item.get("native_receipts", []):
                    path = (self.store.root / ref["path"]).resolve().relative_to(self.store.root).as_posix()
                    if not re.fullmatch(r"native/receipts/action-[0-9]+\.json", path):
                        raise NotRecorded("未记录: receipt is not a bound native receipt")
                    raw = json.loads(read_snapshot_reference(self.store.root, ref))
                    if raw.get("action_id") == parameters["action_id"]:
                        matches.append(receipt(raw))
            if len(matches) == 1:
                public = matches[0]
            elif len(matches) > 1:
                from .types import AmbiguousRecord
                raise AmbiguousRecord("legacy action id has multiple receipts")
        elif method in ("submit_action", "report_issue", "finish_episode"):
            raise Divergence("分歧: legacy evidence cannot establish exact control invocation")
        if public is None:
            raise missing_request(wire, reason="exact_request_not_recorded", boundary=self.context.json()["boundary"])
        response = envelope(method, wire, public=public)
        value = response.json()
        value["provenance"]["reconstruction"] = "legacy_artifact"
        value["provenance"]["exact_interface_bytes"] = False
        return Response(canonical(value))
