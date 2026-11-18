"""Sequential replay has no online simulator/action adapter or fallback."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ..call_scope import CallScope
from .api import EFFECTS, EXTERNAL, Methods
from .blobs import BlobStore, IO_ROOT, input_identity
from .journal import read_events
from .gaps import require_no_gaps
from .comparison import compare
from .comparison_log import append_comparison
from .runtime import Adapter, Runtime, request_wire
from .missing import missing_request
from .protocol import audit, same_request
from .types import (
    AccessDenied, AmbiguousRecord, CallContext, CorruptRecord, Divergence,
    NotRecorded, Response, ServiceUnavailable, canonical, digest,
)


from ..replay_session import ReplaySession


class Replay(Methods):
    """Select one parent call; new external queries never skip historical entries.

    ``services`` is deliberately limited to EXTERNAL. A replay instance cannot
    be configured with submit/probe/simulator adapters, even accidentally.
    """

    def __init__(self, *, context: CallContext, store: BlobStore,
                 services: Mapping[str, Adapter], bindings: Mapping[str, dict],
                 evaluation: Runtime, journal_path: Path | None = None) -> None:
        if set(services) - EXTERNAL or set(evaluation.adapters) - EXTERNAL:
            raise AccessDenied("replay accepts only external service capabilities")
        if context.json()["boundary"]["kind"] != "before_call":
            raise NotRecorded("未记录: native-action boundary has no interface cursor")
        if evaluation.store.root == store.root:
            raise AccessDenied("replay output must not modify the source run")
        require_no_gaps(store.root, context.json()['tool_call_id'])
        self.context, self.store = context, store
        self.bindings = json.loads(canonical(dict(bindings)))
        self.evaluation = evaluation
        self.evaluation.adapters = dict(services)
        self.evaluation.store.input_store = store
        if (evaluation.context.json()["state"] != context.json()["state"]
                or evaluation.context.json()["check_task"] != context.json()["check_task"]
                or evaluation.bindings != self.bindings):
            raise AccessDenied("service evaluation must use the selected boundary and bindings")
        self.replay_state = ReplaySession()
        self.call_scope = CallScope()
        self.entries: list[dict] = []
        events = read_events(journal_path or store.root / IO_ROOT / "calls.jsonl")
        self.protocol_compatibility = audit(store.root, events=events)
        parent = context.json()
        selected = [event for event in events
                    if all(event["identity"][key] == parent[key]
                           for key in ("run_id", "round_index", "tool_call_id"))]
        grouped: dict[int, list] = {}
        for event in selected:
            ordinal = event["identity"]["interface_ordinal"]
            if event["identity"] != context.identity(ordinal):
                raise CorruptRecord("selected call boundary disagrees with evidence")
            grouped.setdefault(ordinal, []).append(event)
        if list(grouped) != list(range(len(grouped))):
            raise AmbiguousRecord("noncontiguous invocation sequence")
        for events in grouped.values():
            if events[0]["event"] != "begin" or events[-1]["event"] != "end":
                raise NotRecorded("未记录: incomplete invocation")
            if events[0]["attempt"] != 0:
                raise CorruptRecord("first attempt must be zero")
            request = events[0]["request"]
            if any(event["request"] != request for event in events):
                raise CorruptRecord("retry changed request identity")
            if {key: input_identity(ref) for key, ref in request["inputs"].items()} != request["identity"]["inputs"]:
                raise CorruptRecord("actual input references differ from request identity")
            if request["identity"]["observation"] != parent["state"].get("observation"):
                raise CorruptRecord("invocation refers to another publication")
            self.entries.append(events[-1])

    def read_blob(self, ref: dict) -> bytes:
        """Read granted source or evaluation blobs, never arbitrary run paths."""
        return self.evaluation.store.read(ref)

    def save_array(self, value: object) -> dict:
        """Save derived inputs only in the independent replay evaluation root."""
        return self.evaluation.store.put_array(value)

    def _invoke(self, method: str, parameters: dict, inputs: dict | None = None) -> Response:
        if method == "check_task" and not self.context.json()["check_task"]:
            raise AccessDenied("check_task capability disabled")
        if method == "submit_action":
            self.call_scope.admit_proposal(commit=False)
        inputs = inputs or {}
        request = request_wire(method, parameters, inputs, context=self.context, bindings=self.bindings)
        expected = self.entries[self.replay_state.cursor] if self.replay_state.cursor < len(self.entries) else None
        matches = expected is not None and same_request(expected["request"]["identity"], request["identity"])
        if expected is not None:
            for ref in expected["request"]["inputs"].values():
                self.store.register(ref)
        for ref in inputs.values():
            self.evaluation.store.read(ref)
        if method in EXTERNAL:
            if method not in self.evaluation.adapters:
                raise ServiceUnavailable(f"service unavailable: {method}; supply --service-report")
            # Service adapters read verified historical inputs through this store,
            # but write generated arrays/envelopes through the evaluation store.
            actual = self.evaluation._invoke(method, parameters, inputs)
            reference = None
            binding = request["identity"]["binding"]
            identity_known = all(binding.get(key) is not None for key in
                                 ("backend", "service_identity", "model_identity", "configuration_identity"))
            if matches and identity_known:
                reference = self.store.register(expected["response_ref"])
                self.store.read(reference)
            comparison = dict(status="unavailable",
                              historical_invocation_id=None if not matches else expected["invocation_id"],
                              reference_sha256=None if reference is None else reference["sha256"],
                              actual_sha256=digest(actual.payload), scope="semantic_envelope_bytes",
                              reason="no_matching_cursor_entry" if not matches else "service_identity_unknown" if not identity_known else None)
            if reference is not None:
                comparison.update(compare(method, self.store.read(reference), actual.payload,
                                          reference_store=self.store, actual_store=self.evaluation.store))
            if matches:
                self.replay_state.cursor += 1
            # Comparison is a separate artifact; never rewrite original response.
            append_comparison(self.evaluation, comparison)
            return Response(actual.payload, canonical(comparison))
        if not matches:
            if method in EFFECTS:
                raise Divergence("分歧: action/control request differs from cursor")
            raise missing_request(request, reason="no_matching_cursor_entry",
                                  boundary=self.context.json()["boundary"])
        ref = self.store.register(expected["response_ref"])
        payload = self.store.read(ref)
        response = json.loads(payload)
        if (response.get("schema") != "ora.toolkit.response.v1" or response.get("method") != method
                or response.get("status") != expected["status"]):
            raise CorruptRecord("response envelope disagrees with I/O event")
        self._register_response_refs(response)
        if method == "submit_action":
            self.call_scope.admit_proposal()
        self.replay_state.cursor += 1
        return Response(payload)

    def _register_response_refs(self, value):
        if isinstance(value, dict):
            if {"path", "sha256", "bytes", "media_type"} <= value.keys():
                self.store.register(value)
            else:
                for child in value.values():
                    self._register_response_refs(child)
        elif isinstance(value, list):
            for child in value:
                self._register_response_refs(child)
