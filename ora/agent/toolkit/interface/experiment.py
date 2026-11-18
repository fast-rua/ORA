"""Alternative tools at one recorded boundary, without online simulator authority."""

from ..call_scope import CallScope
from .api import EFFECTS, EXTERNAL, Methods
from .missing import missing_input, missing_request
from .protocol import same_request
from .runtime import envelope, request_wire
from .types import (
    AccessDenied,
    AmbiguousRecord,
    CallContext,
    NotRecorded,
    PublicData,
    Response,
    canonical,
    decode,
    digest,
)


from ..replay_session import ReplaySession


class RecordedExperiment(Methods):
    """Wrap exact replay; changed tools query only the selected pre-action evidence.

    Repeated requests with differing responses are ambiguous, never first-match.
    Effects from a changed tool are proposals only, stored outside the source.
    A trusted publication reader may reconstruct public bytes, never simulator
    queries. The wrapper has no live action/probe callback.
    """

    def __init__(
        self,
        replay,
        *,
        alternative=False,
        publication=None,
        reconstruct_observation=False,
        allow_check_task=False,
    ):
        self.replay = replay
        self.context, self.store = replay.context, replay.store
        self.evaluation, self.bindings = replay.evaluation, replay.bindings
        self.alternative = alternative
        self.allow_check_task = allow_check_task
        self.publication = publication
        self.reconstruct_observation = reconstruct_observation
        self.replay_state = ReplaySession()
        self.call_scope = CallScope()
        self.entries = []
        for entry in getattr(replay, "entries", []):
            if entry["request"]["identity"]["method"] in EFFECTS:
                break  # Never treat post-effect data as a before-call observation.
            self.entries.append(entry)

    @property
    def cursor(self):
        return None if self.alternative else getattr(getattr(self.replay, "replay_state", None), "cursor", None)

    def read_blob(self, ref):
        return self.replay.read_blob(ref)

    def save_array(self, value):
        return self.replay.save_array(value)

    def tool_artifact(self, *, kind, key, cache=False):
        value = super().tool_artifact(kind=kind, key=key, cache=cache)
        if value is None and self.alternative and not cache:
            error = missing_input(
                "tool_artifact",
                dict(kind=kind, key=key),
                self.context,
                reason="artifact_not_granted_at_boundary",
            )
            self.replay_state.missing_requests.extend(error.missing_requests)
            raise error
        return value

    def _invoke(self, method, parameters, inputs=None):
        try:
            return self._read(method, parameters, inputs or {})
        except NotRecorded as error:
            self.replay_state.missing_requests.extend(error.missing_requests)
            raise

    def _read(self, method, parameters, inputs):
        override = (
            method == "check_task"
            and self.allow_check_task
            and not self.context.json()["check_task"]
        )
        if not self.alternative and not override:
            return self.replay._invoke(method, parameters, inputs)
        if (
            method == "check_task"
            and not self.context.json()["check_task"]
            and not override
        ):
            raise AccessDenied("check_task capability disabled")
        context = (
            CallContext.create(**dict(self.context.json(), check_task=True))
            if override
            else self.context
        )
        request = request_wire(
            method, parameters, inputs, context=context, bindings=self.bindings
        )
        for ref in inputs.values():
            self.evaluation.store.read(ref)
        if method in EFFECTS:
            return self._propose(method, request)
        matches = [
            entry
            for entry in self.entries
            if same_request(entry["request"]["identity"], request["identity"])
        ]
        if method in EXTERNAL:
            # Reuse the established comparison implementation, without consuming
            # the original tool's sequence or changing the trusted service binding.
            old = getattr(getattr(self.replay, "replay_state", None), "cursor", None)
            if old is not None:
                try:
                    match = self._unique(matches)
                except AmbiguousRecord:
                    match = None  # Recompute perception; ambiguous history cannot be a comparison target.
                self.replay.replay_state.cursor = (
                    self.replay.entries.index(match)
                    if match is not None
                    else len(self.replay.entries)
                )
            try:
                return self.replay._invoke(method, parameters, inputs)
            finally:
                if old is not None:
                    self.replay.replay_state.cursor = old
        if method == "observation" and self.reconstruct_observation:
            return self.publication(method, request)
        match = self._unique(matches)
        if match is not None:
            ref = self.store.register(match["response_ref"])
            payload = self.store.read(ref)
            response = decode(payload)
            if (
                response.get("schema") != "ora.toolkit.response.v1"
                or response.get("method") != method
                or response.get("status") != match["status"]
            ):
                from .types import CorruptRecord

                raise CorruptRecord("response envelope disagrees with I/O event")
            self.replay._register_response_refs(response)
            return Response(payload)
        if method in {"observation", "proprioception"} and self.publication is not None:
            return self.publication(method, request)
        if method == "check_task" and hasattr(self.replay, "snapshot"):
            # Legacy only accepts an explicit same-boundary checker response;
            # never read final verdict/control state to supply this override.
            previous = self.replay.context
            self.replay.context = context
            try:
                return self.replay._invoke(method, parameters, inputs)
            finally:
                self.replay.context = previous
        raise missing_request(
            request,
            reason="no_matching_request_at_selected_boundary",
            boundary=self.context.json()["boundary"],
        )

    @staticmethod
    def _unique(matches):
        if len({(row["response_ref"]["sha256"], row["status"]) for row in matches}) > 1:
            raise AmbiguousRecord(
                "same-boundary repeated request has different responses"
            )
        return matches[0] if matches else None

    def _propose(self, method, request):
        if method == "submit_action":
            self.call_scope.admit_proposal()
        record = dict(
            method=method,
            request=request,
            semantic_request_hash=digest(canonical(request["identity"])),
            executed=False,
            compared=False,
        )
        if method == "submit_action":
            action = request["identity"]["parameters"]["action"]
            self.replay_state.proposed_actions.append(dict(record, action=action))
            data = dict(
                accepted=True, action_id=action.get("action_id"), proposal_only=True
            )
        else:
            self.replay_state.proposed_controls.append(record)
            data = dict(success=True, proposal_only=True)
        response = envelope(method, request, public=PublicData.from_json(data))
        value = response.json()
        value["provenance"]["execution"] = "proposal_only"
        return Response(canonical(value))
