"""Best-effort evidence around the new tools' data calls; recording cannot alter execution."""

import json
from types import SimpleNamespace
from ora.agent.toolkit.interface.blobs import BlobStore, IO_ROOT
from ora.agent.toolkit.interface.runtime import (
    Runtime,
    request_wire,
    envelope,
    public_failure,
)
from ora.agent.toolkit.interface.gaps import write_gap
from ora.agent.toolkit.interface.types import (
    AccessDenied,
    AmbiguousRecord,
    canonical,
    digest,
)


class LiveStore(BlobStore):
    """Own derived bytes in memory so a recording disk fault cannot break tool computation."""

    def __init__(self, root, tool_call_id):
        super().__init__(root, writable=True)
        self.memory = {}
        self.tool_call_id = tool_call_id

    def put(self, data, *, media_type="application/octet-stream", array=None):
        ref = dict(
            path=f"{IO_ROOT}/blobs/sha256/{digest(data)}",
            sha256=digest(data),
            bytes=len(data),
            media_type=media_type,
        )
        if array is not None:
            ref["array"] = array
        self.memory[canonical(ref)] = bytes(data)
        try:
            super().put(data, media_type=media_type, array=array)
        except Exception as error:
            write_gap(
                self.root,
                tool_call_id=self.tool_call_id,
                method=None,
                stage="store",
                error=error,
            )
        return json.loads(canonical(ref))

    def read(self, ref):
        key = canonical(ref)
        if key in self.memory:
            return self.memory[key]
        return super().read(ref)


class LiveRuntime(Runtime):
    """Execute once through the interface while retaining M2.3 fail-open journal semantics."""

    def __init__(self, **kwargs):
        try:
            super().__init__(**kwargs)
        except AmbiguousRecord as error:
            # A recreated host may revisit a parent already present in evidence.
            # Reject replay through a gap, but never refuse the live tool call.
            write_gap(
                kwargs["store"].root,
                tool_call_id=kwargs["context"].json()["tool_call_id"],
                method=None,
                stage="context",
                error=error,
            )
            kwargs["journal"] = SimpleNamespace(events=[], _closed=True)
            super().__init__(**kwargs)

    def _invoke(self, method, parameters, inputs=None):
        if method == "check_task" and not self.context.json()["check_task"]:
            raise AccessDenied("check_task capability disabled")
        if method == "submit_action":
            self.call_scope.admit_proposal()
        inputs = inputs or {}
        request = request_wire(
            method, parameters, inputs, context=self.current_context(), bindings=self.bindings
        )
        response = None

        def call():
            nonlocal response
            try:
                public = self.adapters[method](
                    request["identity"]["parameters"], inputs, self.store
                )
            except Exception as error:
                status, failure = public_failure(method, error)
                response = envelope(method, request, status=status, error=failure)
                raise
            response = envelope(method, request, public=public)
            return public

        try:
            self.capture(
                method,
                parameters,
                inputs,
                call=call,
                project=lambda value, store: value,
            )
        except Exception:
            if response is None:
                raise
        return response
