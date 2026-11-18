"""Record each public I/O occurrence before and after its trusted adapter."""
from __future__ import annotations

import json
import re
from typing import Callable, Mapping

from ..call_scope import CallScope
from . import call_note
from .api import EFFECTS, EXTERNAL, METHODS, Methods
from .blobs import BlobStore, input_identity
from .journal import Journal
from .types import (
    AccessDenied, AmbiguousRecord, CallContext, InterfaceError, PublicData,
    Response, canonical, digest,
)

from ..recording import InvocationRecording

Adapter = Callable[[dict, dict, BlobStore], PublicData]


def _note() -> dict:
    """The ``note`` field of a call's events, none for a call without one (``call_note``)."""
    note = call_note.current()
    return {"note": note} if note else {}


def request_wire(method: str, parameters: dict, inputs: dict, *, context: CallContext,
                 bindings: Mapping[str, dict]) -> dict:
    """Canonical identity plus actual references; neither embeds arbitrary objects."""
    if method not in METHODS:
        raise InterfaceError("unknown interface method")
    parameters = json.loads(canonical(parameters))
    state = context.json()["state"]
    observation = state.get("observation")
    if method in ("observation", "proprioception"):
        parameters["observation"] = parameters["observation"] or observation
    if method == "probe_motion":
        parameters["request"].setdefault("variant_policy", "exact")
    if method == "ground" and (type(parameters["max_candidates"]) is not int
                               or not 1 <= parameters["max_candidates"] <= 3):
        raise InterfaceError("SAM3 ground supports 1..3 candidates")
    if method == "segment_points" and not 1 <= len(parameters["points_uv"]) <= 8:
        raise InterfaceError("SAM3 supports 1..8 independent positive points")
    if method == "grasp_candidates":
        if (parameters["forward_passes"] != 1 or type(parameters["max_candidates"]) is not int
                or not 1 <= parameters["max_candidates"] <= 32):
            raise InterfaceError("CGN supports one pass and 1..32 raw candidates")
    binding = dict(backend=None, service_identity=None, model_identity=None, configuration_identity=None)
    binding.update(bindings.get(method, {}))
    identity = dict(schema="ora.toolkit.request.v1", method=method, method_version=1,
                    parameters=parameters, inputs={key: input_identity(ref) for key, ref in inputs.items()},
                    observation=observation, binding=binding,
                    capabilities={"check_task": context.json()["check_task"]})
    return json.loads(canonical(dict(identity=identity, inputs=inputs)))


def envelope(method: str, request: dict, *, public: PublicData | None = None,
             status: str = "ok", error: dict | None = None) -> Response:
    source = "external_service" if method in EXTERNAL else "agent_control" if method in EFFECTS else "simulator"
    identity = request["identity"]
    return Response(canonical(dict(
        schema="ora.toolkit.response.v1", method=method, status=status,
        data=None if public is None else json.loads(public.data),
        provenance=dict(source=source, projection=f"{method}.v1", observation=identity["observation"],
                        binding=identity["binding"], sources=[] if public is None else json.loads(public.sources)),
        error=error,
    )))


def public_failure(method: str, error: Exception) -> tuple[str, dict]:
    """Keep public failure classes/status, never arbitrary exception detail."""
    from ora.agent.model_services.clients import ModelServiceError
    from .types import ActionNotStarted, PublicToolFailure, ReferenceFailure
    import httpx
    timeout = isinstance(error, (TimeoutError, httpx.TimeoutException)) or isinstance(
        error.__cause__, (TimeoutError, httpx.TimeoutException))
    status = "timeout" if timeout else "error"
    failure = dict(code="backend_timeout" if timeout else "backend_failure",
                   retryable=timeout and method in EXTERNAL)
    if method == "execute_action" and isinstance(error, ActionNotStarted):
        failure["execution"] = {"status": "not_started", "native_actions": 0,
                                "stage": error.stage}
    if isinstance(error, ReferenceFailure):
        failure["reference_failure"] = {"exception_type": error.exception_type}
    if isinstance(error, PublicToolFailure):
        failure.update(code=error.code, retryable=error.retryable and method in EXTERNAL,
                       tool_error=error.public())
    elif isinstance(error, ModelServiceError):
        failure.update(code=f"model_service_{error.failure}", failure_class=error.failure,
                       retryable=error.failure == "transient" and method in EXTERNAL)
        if type(error.status) is int and 100 <= error.status <= 599:
            failure["http_status"] = error.status
    else:
        code = getattr(error, "code", None)
        if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,99}", code):
            failure["code"] = code
            failure["retryable"] = bool(getattr(error, "retryable", False)) and method in EXTERNAL
        if isinstance(error, AccessDenied):
            failure.update(code="access_denied", retryable=False)
    return status, failure


class Runtime(Methods):
    """A per-parent-tool interface; injected adapters have public field projections.

    No retry is implicit. ``service_retries`` is an explicit trusted policy,
    applies only to external services, and every attempt has its own events.
    """

    def __init__(self, *, context: CallContext, store: BlobStore, journal: Journal,
                 adapters: Mapping[str, Adapter], bindings: Mapping[str, dict],
                 service_retries: int = 0, recording: InvocationRecording | None = None,
                 context_provider: Callable | None = None) -> None:
        if type(service_retries) is not int or not 0 <= service_retries <= 3:
            raise InterfaceError("service_retries must be in 0..3")
        self.context, self.store, self.journal = context, store, journal
        self.adapters = dict(adapters)
        self.bindings = json.loads(canonical(dict(bindings)))
        self.service_retries = service_retries
        self.call_scope = CallScope()
        self.context_provider = context_provider
        self.recording = recording if recording is not None else InvocationRecording()
        parent = context.json()
        if any(all(event["identity"][key] == parent[key] for key in
                   ("run_id", "round_index", "tool_call_id")) for event in journal.events):
            raise AmbiguousRecord("parent tool already has I/O; reuse its existing interface")

    def current_context(self) -> CallContext:
        """Read current binding from its owner; initial context remains the parent identity."""
        return self.context if self.context_provider is None else self.context_provider()

    def read_blob(self, ref: dict) -> bytes:
        """Read only previously registered public data, verifying its hash on every read."""
        return self.store.read(ref)

    def save_array(self, value: object) -> dict:
        """Save a derived array; its reference is bound into the next service request."""
        return self.store.put_array(value)

    def capture(self, method: str, parameters: dict, inputs: dict, *, call: Callable,
                project: Callable, on_gap: Callable | None = None) -> object:
        """Observe an existing call exactly once; recording can never refuse it.

        The original exception is re-raised outside all recording handlers.
        Unlike _invoke, capture is instrumentation, not execution authority.
        A projection/storage gap closes a known begin with recording_gap when
        possible. An uncertain append freezes the writer; no write is retried.
        """
        from .gaps import write_gap

        began = False
        append_attempted = False

        def gap(stage, error):
            self.recording.failed = True
            # Only a known committed begin can be closed. Never retry an
            # uncertain begin/end append or fabricate the original response.
            if began and stage in {"project", "store"} and not self.journal._closed:
                try:
                    failure = dict(code="recording_gap", retryable=False)
                    response = envelope(method, request, status="error", error=failure)
                    ref = self.store.put(response.payload, media_type="application/json")
                    self.journal.append(dict(
                        base, attempt=0, event="end", status="error", state_after=None,
                        response_ref=ref, source=response.json()["provenance"]["source"],
                        cache_hit=False, error=failure,
                    ))
                except Exception:
                    self.journal._closed = True
            elif stage == "end" or (stage == "begin" and append_attempted):
                self.journal._closed = True
            write_gap(self.store.root, tool_call_id=self.context.json()['tool_call_id'],
                      method=method, stage=stage, error=error,
                      interface_ordinal=self.recording.ordinal)
            if on_gap is not None:
                try:
                    on_gap()
                except Exception:
                    pass

        stage = 'begin'
        try:
            if self.recording.failed or self.journal._closed:
                raise InterfaceError('recording disabled after an earlier gap')
            stage = 'store'
            for ref in inputs.values():
                self.store.read(ref)
            stage = 'begin'
            context = self.current_context()
            request = request_wire(method, parameters, inputs, context=context, bindings=self.bindings)
            identity = context.identity(self.recording.ordinal)
            base = dict(invocation_id=digest(canonical(identity)), identity=identity,
                        request=request, semantic_request_hash=digest(canonical(request['identity'])),
                        state_before=context.json()['state'], **_note())
            append_attempted = True
            self.journal.append(dict(base, attempt=0, event='begin'))
            began = True
        except Exception as error:
            gap(stage, error)
            return call()

        def finish(value=None, original_error=None):
            stage = 'project'
            try:
                public = None
                if original_error is None:
                    public = project(value, self.store)
                    response = envelope(method, request, public=public)
                    status, failure = 'ok', None
                else:
                    status, failure = public_failure(method, original_error)
                    response = envelope(method, request, status=status, error=failure)
                stage = 'store'
                ref = self.store.put(response.payload, media_type='application/json')
                stage = 'end'
                self.journal.append(dict(base, attempt=0, event='end', status=status,
                                         state_after=None, response_ref=ref,
                                         source=response.json()['provenance']['source'],
                                         cache_hit=False if public is None else public.cache_hit, error=failure))
                self.recording.ordinal += 1
            except Exception as error:
                gap('store' if stage == 'project' and isinstance(error, OSError) else stage, error)

        try:
            value = call()
        except Exception as original_error:
            finish(original_error=original_error)
            raise
        finish(value=value)
        return value

    def _invoke(self, method: str, parameters: dict, inputs: dict | None = None) -> Response:
        if method == "check_task" and not self.context.json()["check_task"]:
            raise AccessDenied("check_task capability disabled")
        if method == "submit_action":
            self.call_scope.admit_proposal()
        inputs = inputs or {}
        for ref in inputs.values():
            self.store.read(ref)
        context = self.current_context()
        request = request_wire(method, parameters, inputs, context=context, bindings=self.bindings)
        identity = context.identity(self.recording.ordinal)
        base = dict(invocation_id=digest(canonical(identity)), identity=identity,
                    request=request, semantic_request_hash=digest(canonical(request["identity"])),
                    state_before=context.json()["state"], **_note())
        retries = self.service_retries if method in EXTERNAL else 0
        for attempt in range(retries + 1):
            self.journal.append(dict(base, attempt=attempt, event="begin"))
            public = None
            try:
                if method not in self.adapters:
                    raise InterfaceError("backend capability unavailable")
                public = self.adapters[method](request["identity"]["parameters"], request["inputs"], self.store)
                if not isinstance(public, PublicData):
                    raise InterfaceError("adapter must return a public projection")
                response = envelope(method, request, public=public)
                status, failure = "ok", None
            except Exception as error:
                status, failure = public_failure(method, error)
                response = envelope(method, request, status=status, error=failure)
            ref = self.store.put(response.payload, media_type="application/json")
            self.journal.append(dict(base, attempt=attempt, event="end", status=status,
                                     state_after=None, response_ref=ref,
                                     source=response.json()["provenance"]["source"],
                                     cache_hit=False if public is None else public.cache_hit, error=failure))
            if status == "ok" or not failure["retryable"]:
                break
        self.recording.ordinal += 1
        return response
