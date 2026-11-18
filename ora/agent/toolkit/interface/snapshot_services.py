"""External-only transport for snapshot experiments; no simulator wire fallback.

State/privileged/lifecycle capabilities are assembled separately. Historical
service outputs are comparison evidence only, never answers to new requests.
"""

from types import SimpleNamespace

from .api import EXTERNAL, Methods
from .blobs import BlobStore
from .comparison import compare
from .comparison_log import append_comparison
from .protocol import same_request
from .runtime import request_wire
from .types import AccessDenied, Response, ServiceUnavailable, digest


class SnapshotServices(Methods):
    """Use injected service endpoints and output CAS, with explicit history."""

    def __init__(self, *, context, store, evaluation, history=()):
        if set(evaluation.adapters) - EXTERNAL:
            raise AccessDenied("snapshot transport admits only external services")
        if store.root == evaluation.store.root:
            raise AccessDenied("snapshot output must be separate from source")
        self.context, self.store = context, store
        self.evaluation, self.bindings = evaluation, evaluation.bindings
        evaluation.store.input_store = store
        self.history = tuple(history)
        # Historical responses can postdate the selected boundary. Keep their
        # grants entirely outside the script-facing artifact capability.
        self.reference_store = BlobStore(store.root)
        self.comparisons = SimpleNamespace(store=BlobStore(evaluation.store.root, writable=True),
                                           journal=evaluation.journal)

    def read_blob(self, ref):
        return self.evaluation.store.read(ref)

    def save_array(self, value):
        return self.evaluation.store.put_array(value)

    def _invoke(self, method, parameters, inputs=None):
        if method not in EXTERNAL:
            raise AccessDenied("snapshot transport has no simulator or action adapter")
        if method not in self.evaluation.adapters:
            raise ServiceUnavailable(f"service unavailable: {method}; supply --service-report")
        request = request_wire(method, parameters, inputs or {}, context=self.context,
                               bindings=self.bindings)
        matches = [row for row in self.history if row["event"] == "end"
                   and same_request(row["request"]["identity"], request["identity"])]
        actual = self.evaluation._invoke(method, parameters, inputs or {})
        comparison = dict(status="unavailable", historical_invocation_id=None,
                          reference_sha256=None, actual_sha256=digest(actual.payload),
                          scope="semantic_envelope_bytes", reason="no_matching_snapshot_service_request")
        responses = {row["response_ref"]["sha256"] for row in matches}
        known = all(request["identity"]["binding"].get(key) is not None for key in
                    ("backend", "service_identity", "model_identity", "configuration_identity"))
        if len(responses) > 1:
            comparison["reason"] = "ambiguous_historical_service_responses"
        elif matches and not known:
            comparison["reason"] = "service_identity_unknown"
        elif matches:
            reference = self.reference_store.register(matches[0]["response_ref"])
            comparison.update(historical_invocation_id=matches[0]["invocation_id"],
                              historical_invocation_ids=[row["invocation_id"] for row in matches],
                              reference_sha256=reference["sha256"], reason=None)
            comparison.update(compare(method, self.reference_store.read(reference), actual.payload,
                                      reference_store=self.reference_store, actual_store=self.evaluation.store))
        append_comparison(self.comparisons, comparison)
        return Response(actual.payload)
