"""Replay the runtime's delivery projection, keeping visible and audit bytes distinct."""

import json
from types import SimpleNamespace

from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.pi.hints import StaticFactsState, apply_hints
from ora.agent.pi.tool_rpc import PiToolRpc
from ora.agent.session_core.domain import PolicyDomainMixin
from ora.agent.toolkit.interface.types import canonical, decode, digest

from .boundary import run_surface
from .selection import read_local


# Explicit public result paths, never recursively remove user-authored compute keys.
VOLATILE_PATHS = {
    "grasp_plan": tuple(
        (*branch, field)
        for branch in (("diagnostics",), ("diagnostics", "learned"))
        for field in ("request_id", "latency_ms", "timings", "queue")
    ),
    "vision_ground": tuple(
        ("diagnostics", field)
        for field in ("request_id", "latency_ms", "timings", "queue")
    ),
}


class Projection(PiToolRpc, PolicyDomainMixin):
    """Only the existing pure projection methods; no planner or tool execution."""

    def _dump_hints(self, kind, record):
        pass  # Replay output must not follow a runtime debug path from the environment.


def normalized(value, tool):
    value = decode(canonical(value))
    for path in VOLATILE_PATHS.get(tool, ()):
        parent = value
        for key in path[:-1]:
            parent = parent.get(key, {}) if isinstance(parent, dict) else {}
        if isinstance(parent, dict):
            parent.pop(path[-1], None)
    return value


def compare(left, right, tool):
    return dict(
        byte_equal=left == right,
        content_equal=normalized(decode(left), tool) == normalized(decode(right), tool),
        volatile_paths=["/" + "/".join(path) for path in VOLATILE_PATHS.get(tool, ())],
    )


def project(source, output, selected, events, tool, result, *, same_tool):
    """Return exact UTF-8 visible text and stored-artifact projection references.

    Native causal sessions store a richer canonical result than the text sent to
    the model. Compare both domains explicitly; never label canonical bytes as
    model-visible. Missing frozen configuration produces unavailable evidence.
    """
    surface, _ = run_surface(source)
    plan = (
        decode(read_local(source, "plan.json"))
        if (source / "plan.json").is_file()
        else {}
    )
    policy = plan.get("policy", {})
    if surface is None or "context_composition" not in policy:
        return dict(
            status="unavailable", reason="frozen_projection_configuration_not_recorded"
        )
    host = Projection()
    host.tools = SimpleNamespace(
        surface=SimpleNamespace(
            capabilities=surface.get("capabilities", ()),
            definitions=surface.get("definitions", ()),
            result_schemas=surface.get("result_schemas", {}),
        )
    )
    host.context_composition = policy["context_composition"]
    host.supports_images = True
    host.artifacts = ArtifactStore(output)
    projected = host._project(tool, result, call_id=selected["tool_call_id"], step=0)
    visible, _ = apply_hints(
        projected["visible"], projected["visible_hints"], StaticFactsState()
    )
    stored, _ = apply_hints(
        projected["canonical"], projected["canonical_hints"], StaticFactsState()
    )
    # Exactly the provider text encoding used by Pi, separately from CAS JSON.
    text = json.dumps(visible, ensure_ascii=False, sort_keys=True)
    data = text.encode("utf-8")
    visible_ref = host.artifacts.put_bytes(
        data, kind="replay_model_visible", suffix=".txt", media_type="text/plain"
    )
    stored_ref = host.artifacts.put_json(stored, kind="replay_projected_tool_result")
    answer = dict(
        status="ok",
        text=text,
        sha256=digest(data),
        bytes=len(data),
        artifact_ref=visible_ref,
        stored_projection=stored_ref,
        images=projected["images"],
        image_text=projected["image_text"],
        comparison=dict(
            status="unavailable",
            reason=(
                "different_tool" if not same_tool else "original_result_not_recorded"
            ),
        ),
    )
    if not same_tool:
        return answer
    results = [
        event
        for event in events
        if event.get("type") == "tool_result"
        and event.get("tool_call_id") == selected["tool_call_id"]
    ]
    if len(results) != 1 or not isinstance(results[0].get("artifact_ref"), dict):
        return answer
    ref = results[0]["artifact_ref"]
    if not str(ref.get("path", "")).startswith("native/agent-artifacts/"):
        answer["comparison"] = dict(
            status="unavailable", reason="unsupported_original_result_path"
        )
        return answer
    original = read_local(source, ref["path"])
    if digest(original) != ref.get("sha256") or len(original) != ref.get("bytes"):
        answer["comparison"] = dict(
            status="unavailable", reason="original_result_hash_mismatch"
        )
        return answer
    # Stored canonical artifacts are already in delivered units. Do not project
    # them again (that would convert degrees twice). Causal filtering alone is
    # still needed to obtain the visible comparison domain.
    from ora.agent.session_core.constants import NATIVE_CAUSAL_SLOTS

    historical = decode(original)
    if host.context_composition in NATIVE_CAUSAL_SLOTS:
        historical = host._causal_tool_result(tool, historical)
    historical_text = json.dumps(historical, ensure_ascii=False, sort_keys=True).encode(
        "utf-8"
    )
    answer["comparison"] = dict(
        status="ok",
        original_artifact=ref,
        artifact=compare(canonical(stored), original, tool),
        visible=compare(data, historical_text, tool),
        original_visible_sha256=digest(historical_text),
        original_visible_scope="projected_from_stored_result; not a provider transcript",
    )
    return answer
