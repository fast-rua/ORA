"""Machine-readable descriptions of actual missing requests, never guessed outcomes."""

from .types import NotRecorded, canonical, digest


def missing_request(request, *, reason, kind="interface", boundary=None):
    identity = request["identity"]
    item = dict(
        kind=kind,
        method=identity["method"],
        semantic_request_hash=digest(canonical(identity)),
        request=request,
        reason=reason,
    )
    if boundary is not None:
        item["boundary"] = boundary
    return NotRecorded(
        f"未记录: {identity['method']}: {reason}", missing_requests=[item]
    )


def missing_input(method, parameters, context, *, reason):
    identity = dict(
        schema="ora.toolkit.input-request.v1",
        method=method,
        parameters=parameters,
        observation=context.json()["state"].get("observation"),
    )
    return missing_request(
        dict(identity=identity, inputs={}),
        reason=reason,
        kind="tool_input",
        boundary=context.json()["boundary"],
    )
