"""What policy-visible evidence may never carry: the forbidden key fragments, the
exact public exemptions, what a code-mode script wrote itself, and the recursive
path check.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Collection, Iterable, Mapping


FORBIDDEN_FRAGMENTS = {
    "actor_state",
    "object_gt",
    "segmentation",
    "contact",
    "success_predicate",
    "expert_program",
    "task_source",
}
# Forbidden terms match on word boundaries (start/end of the key or an
# adjacent non-alphanumeric character), with plural forms included.  A term
# embedded inside a larger word does not match: the controller phase noun
# "precontact" is public motion vocabulary, not contact-sensing evidence,
# and the bare substring rule invalidated nine sealed v13 runs over the
# policy-safe field `motion_strategy.precontact_public_tcp_poses`.
_FORBIDDEN_PATTERNS = tuple(
    re.compile(rf"(?<![a-z0-9]){re.escape(fragment)}(?:s|es)?(?![a-z0-9])")
    for fragment in sorted(FORBIDDEN_FRAGMENTS)
)


def _matches_forbidden_fragment(lowered: str) -> bool:
    return any(pattern.search(lowered) for pattern in _FORBIDDEN_PATTERNS)


def _is_forbidden_term(lowered: str) -> bool:
    """Whether a key is exactly a forbidden term (`contact`, `contacts`, `segmentation`), not just contains one."""

    return any(pattern.fullmatch(lowered) for pattern in _FORBIDDEN_PATTERNS)


# Exact policy-authored motion-parameter names that legitimately contain a
# forbidden term on a word boundary.  `contact_approach` is the public
# staged-approach strategy argument the policy itself writes
# (`staged_contact_approach_v1`); it cannot leak state the policy never saw.
# Values under these keys are still scanned recursively.
# `expect_contact` (2026-10-28) is the policy's own boolean that marks a move as a deliberate push; the
# receipt only echoes it back.
PUBLIC_MOTION_VOCABULARY_KEYS = frozenset({"contact_approach", "expect_contact"})


# Exact Panda grip-site compiler constants; the ALOHA dual-arm surfaces
# publish a restricted compiler projection instead, so their geometry never
# reaches provider-visible results.  The exemption stays value-pinned and
# fail-closed: an unregistered geometry value remains forbidden evidence.
PUBLIC_CONTACT_GEOMETRY_VALUES = {
    "contact_graspnet_gripper_depth_m": (frozenset({0.1034}), ".compiler"),
    "controlled_point_to_contact_baseline_m": (frozenset({0.0064}), ".compiler"),
    "contact_baseline_reconstruction": (frozenset({"passed"}), ".compiler_checks"),
}
PUBLIC_GRASP_CONTACT_POSE_CAPABILITY = "public_grasp_contact_pose_v1"
#: The two world directions a ``grasp_pose_world_axes_v1`` surface adds to each grasp-plan pose (RO-27). They
#: are the pose's own quaternion applied to the controlled point's public local axes, so they carry nothing
#: the pose does not; they come as a pair or not at all.
GRASP_POSE_WORLD_AXES = ("approach_world", "jaw_line_world")


def _is_public_grasp_contact_pose(value: Any) -> bool:
    """Recognize only the exact policy-visible grasp-plan pose contract, with or without its world axes."""

    if not isinstance(value, Mapping) or set(value) not in (
        {"position_m", "orientation_xyzw"}, {"position_m", "orientation_xyzw", *GRASP_POSE_WORLD_AXES},
    ):
        return False
    position = value.get("position_m")
    orientation = value.get("orientation_xyzw")
    if not isinstance(position, list) or len(position) != 3:
        return False
    if not isinstance(orientation, list) or len(orientation) != 4:
        return False
    axes = [value[key] for key in GRASP_POSE_WORLD_AXES if key in value]
    if not all(isinstance(axis, list) and len(axis) == 3 for axis in axes):
        return False
    return all(
        isinstance(component, (int, float))
        and not isinstance(component, bool)
        and math.isfinite(component)
        for component in [*position, *orientation, *(item for axis in axes for item in axis)]
    )


def _allows_public_grasp_contact_pose(plan: Mapping[str, Any]) -> bool:
    surface = plan.get("policy_composition", {}).get("tool_surface", {})
    capabilities = surface.get("capabilities", []) if isinstance(surface, Mapping) else []
    return (
        isinstance(capabilities, list)
        and PUBLIC_GRASP_CONTACT_POSE_CAPABILITY in capabilities
    )


# What a code-mode script wrote into its own result (``agent/pi/code_mode.py``,
# the model's ``run_python`` or ``bash`` call): the value it returned, what it
# printed, and the reason and data it handed back, under names the model chose.
# The script runs in a sandbox that sees no run, simulator or credential
# (``agent/code_mode/sandbox.py``) and reaches the world only through its tool
# calls, each recorded and audited as its own event (an action's receipt as its
# ``environment_receipt``; an action that did not start returns only the
# harness's error message), so these fields carry only the model's computation
# over what its calls returned; a handed-back point named `contact` is not
# contact sensing. The rest of the result (its calls, counts, limits, errors and
# workspace listing) is the harness's and stays audited.
SCRIPT_AUTHORED_FIELDS = frozenset({"value", "stdout", "stderr", "output"})
SCRIPT_HANDED_BACK_FIELDS = frozenset({"reason", "data"})
#: The script tools: ``bash``, and in a sealed run the retired ``run_python``.
SCRIPT_TOOLS = frozenset({"run_python", "bash"})
#: The ``origin`` kinds of the calls a script made. glue_code is code mode's name before the rename, kept so runs
#: sealed before it still verify.
SCRIPT_CALL_ORIGINS = ("code_mode", "glue_code")


def without_script_authored_fields(result: Any) -> Any:
    """A code-mode script's result without what the script wrote itself."""

    if not isinstance(result, Mapping):
        return result
    kept = {key: child for key, child in result.items() if key not in SCRIPT_AUTHORED_FIELDS}
    handed_back = result.get("handed_back")
    if isinstance(handed_back, Mapping):
        kept["handed_back"] = {
            key: child for key, child in handed_back.items() if key not in SCRIPT_HANDED_BACK_FIELDS
        }
    return kept


def _without_script_authored_text(content: Any) -> Any:
    """The JSON text of a script's result without what the script wrote; any other content as it is."""

    if not isinstance(content, str):
        return content
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return content
    return json.dumps(without_script_authored_fields(result)) if isinstance(result, Mapping) else content


def _is_script_call(call_id: Any, script_calls: Collection[str]) -> bool:
    return isinstance(call_id, str) and call_id in script_calls


def _heads_script_result(block: Mapping[str, Any], script_calls: Collection[str]) -> bool:
    """A compactor packet's text block naming a script's result: its text is the next block
    (``packetMessageBlocks`` in ``ora-pi/src/context/projection.ts``)."""

    text = block.get("text")
    if not isinstance(text, str) or '"tool_call_id"' not in text:
        return False
    try:
        head = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(head, Mapping) and head.get("role") == "tool" and _is_script_call(
        head.get("tool_call_id"), script_calls)


def _without_script_authored_blocks(blocks: list[Any], script_calls: Collection[str]) -> list[Any]:
    kept: list[Any] = []
    result_follows = False
    for block in blocks:
        if isinstance(block, Mapping) and block.get("type") == "tool_result":
            if _is_script_call(block.get("tool_use_id"), script_calls):
                block = {**block, "content": _without_script_authored_text(block.get("content"))}
        elif isinstance(block, Mapping) and block.get("type") == "text":
            if result_follows:
                block = {**block, "text": _without_script_authored_text(block.get("text"))}
                result_follows = False
            else:
                result_follows = _heads_script_result(block, script_calls)
        kept.append(block)
    return kept


#: Where a request or a compactor packet holds its messages: a provider or compactor request's ``messages``, and the
#: ``compactor_messages`` a context packet (``lossless_context_semantic_packet``, ``episode_context_capsule_packet``)
#: records before the compactor's request repeats them.
_MESSAGE_LISTS = ("messages", "compactor_messages")


def _without_script_authored_list(messages: list[Any], script_calls: Collection[str]) -> list[Any]:
    kept: list[Any] = []
    for message in messages:
        if isinstance(message, Mapping):
            content = message.get("content")
            if _is_script_call(message.get("tool_call_id"), script_calls):
                message = {**message, "content": _without_script_authored_text(content)}
            elif isinstance(content, list):
                message = {**message, "content": _without_script_authored_blocks(content, script_calls)}
        kept.append(message)
    return kept


def without_script_authored_messages(request: Any, script_calls: Collection[str]) -> Any:
    """A provider or compactor request, or a context packet, without what the scripts ``script_calls`` wrote.

    A request repeats each script's result, so the audit reads it as it reads that result: without what the
    script wrote, under whatever name the model gave it. The result is one JSON text, in a ``role: tool`` message
    (OpenAI, canonical), a ``tool_result`` block (Anthropic) or a compactor packet's text block, in ``messages`` or
    a packet's ``compactor_messages``. The rest of the request, other tools' results included, is kept."""

    if not script_calls or not isinstance(request, Mapping):
        return request
    lists = {field: request[field] for field in _MESSAGE_LISTS if isinstance(request.get(field), list)}
    if not lists:
        return request
    return {**request, **{field: _without_script_authored_list(messages, script_calls)
                          for field, messages in lists.items()}}


def script_call_ids(plan: Mapping[str, Any], events: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """The tool calls of the scripts the model itself called in a code-mode run; none in tool-call mode."""

    policy = plan.get("policy") or {}
    if not (policy.get("code_mode") or policy.get("glue_code")):
        return frozenset()
    calls: set[str] = set()
    for event in events:
        payload = event.get("payload")
        if event.get("type") == "tool_call_proposed" and isinstance(payload, Mapping):
            origin = payload.get("origin")
            nested = isinstance(origin, Mapping) and origin.get("kind") in SCRIPT_CALL_ORIGINS
            if payload.get("name") in SCRIPT_TOOLS and not nested and isinstance(event.get("tool_call_id"), str):
                calls.add(event["tool_call_id"])
    return frozenset(calls)


# The model's own tool-call arguments (2026-11-04). The audit asks whether privileged state reached the policy. It
# reaches the policy only through what the harness shows it (prompt, observations, tool results, receipts, packets),
# and all of that stays audited. The arguments of a tool call are what the model wrote: every key in them is a name
# it chose, computed from what it was shown, so a key holding a forbidden term is no evidence of a leak. The run
# ora-glue-code-robodojo-cb50f80ce61a--20261104T011111Z was refused for a malformed `write` call whose one key was
# `arguments=\n{"content": ...}`, a whole module in which a docstring said "pad contact". So the audit leaves out a
# call's arguments where the model's provenance is exact:
# - the reply that wrote them (`provider_response` `payload.message.tool_calls[].function.arguments`);
# - every later request or compactor packet that repeats that call: a tool call with the reply's id and the same
#   arguments text (OpenAI ``function.arguments``), or an Anthropic ``tool_use`` block with that id whose ``input``
#   is what the text parses to. A call the context rewrote (an aged or externalized arguments marker) is no longer the
#   model's text and stays audited;
# - the proposal that parsed them (`tool_call_proposed` `payload.arguments`, equal to that parse), and the arguments of
#   a call a code-mode script made (``origin`` naming a script the model called; the script runs in the sandbox
#   ``SCRIPT_AUTHORED_FIELDS`` describes, and the gateway only drops nulls and coerces integral floats).
# The rest of the reply, request and proposal stays audited, as does every tool result or receipt that echoes an
# argument (the harness wrote those; I-0011's declared-argument rule covers the echoes a surface declares). A tool
# schema that declares a forbidden name is the harness's too, and stays refused. The agent core's own check of a
# provider request (`causalMessageViolations` in ora-pi/src/context/projection.ts) already skips assistant messages.


def reply_tool_arguments(payload: Any) -> dict[str, str]:
    """The tool calls of one reply (a ``provider_response`` payload): each call's id to its exact arguments text."""

    message = payload.get("message") if isinstance(payload, Mapping) else None
    calls = message.get("tool_calls") if isinstance(message, Mapping) else None
    found: dict[str, str] = {}
    for call in calls if isinstance(calls, list) else []:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        if isinstance(call.get("id"), str) and isinstance(function, Mapping) and isinstance(
                function.get("arguments"), str):
            found[call["id"]] = function["arguments"]
    return found


def model_tool_arguments(events: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Every tool call the model wrote in the run's replies: its id to its exact arguments text."""

    found: dict[str, str] = {}
    for event in events:
        if isinstance(event, Mapping) and event.get("type") == "provider_response":
            found.update(reply_tool_arguments(event.get("payload")))
    return found


def _parses_to(text: str, value: Any) -> bool:
    try:
        return json.loads(text) == value
    except (TypeError, ValueError):
        return False


def _without_model_arguments(value: Mapping[str, Any], model_arguments: Mapping[str, str]) -> Mapping[str, Any]:
    """A tool call the model wrote, wherever a reply, request or packet holds it, without its arguments; any other
    mapping as it is."""

    call_id = value.get("id")
    written = model_arguments.get(call_id) if isinstance(call_id, str) else None
    if written is None:
        return value
    function = value.get("function")
    if isinstance(function, Mapping) and function.get("arguments") == written:
        return {**value, "function": {key: child for key, child in function.items() if key != "arguments"}}
    if value.get("type") == "tool_use" and "input" in value and _parses_to(written, value["input"]):
        return {key: child for key, child in value.items() if key != "input"}
    return value


def proposal_without_model_arguments(
    payload: Any, tool_call_id: Any, model_arguments: Mapping[str, str], script_calls: Collection[str],
) -> Any:
    """A ``tool_call_proposed`` payload without the arguments the model wrote: a call of its reply, whose
    arguments are what the reply's call of the same id parses to, or a call a script it called made (``origin`` of a
    script kind naming one of ``script_calls``). Any other payload as it is."""

    if not isinstance(payload, Mapping) or "arguments" not in payload:
        return payload
    origin = payload.get("origin")
    if origin is None:
        written = model_arguments.get(tool_call_id) if isinstance(tool_call_id, str) else None
        own = written is not None and _parses_to(written, payload["arguments"])
    else:
        own = (isinstance(origin, Mapping) and origin.get("kind") in SCRIPT_CALL_ORIGINS
               and _is_script_call(origin.get("parent_tool_call_id"), script_calls))
    return {key: child for key, child in payload.items() if key != "arguments"} if own else payload


# The run's own tool arguments (I-0011, narrowed 2026-10-30). A toolset may declare an argument whose name holds a
# forbidden term (`contact_speed_m_s`); the policy writes it and tools echo it. Such a name passes only where it holds
# the policy's own argument: a value the policy wrote for it in a tool call (`tool_call_proposed` `payload.arguments`,
# nested argument objects included) or the default the surface declares for it, wherever that exact value appears
# (the call itself, the call in the provider request and response, a tool result or receipt echoing it at whatever
# path the tool chose), and its declaration in a provider request's tool schema. The same key with any other value,
# in a tool result or anywhere else, is audited like any other key, so declaring an argument cannot blind the audit
# to state a tool returns under its name. A name that is exactly a forbidden term (`contact`, `segmentation`) is
# never exempted.
_ARGUMENT_DECLARATION_PATH = re.compile(r"\$\.tools\[\d+\]\.(?:function\.parameters|input_schema)\.properties")


def declared_argument_names(surface: Mapping[str, Any]) -> frozenset[str]:
    """The top-level argument names the run's tool surface declares that contain a forbidden term but are not
    exactly one (a toolset's `contact_speed_m_s`, I-0011): the names the policy-argument exemption applies to."""

    definitions = surface.get("definitions") if isinstance(surface, Mapping) else None
    names = set()
    for definition in definitions if isinstance(definitions, list) else []:
        schema = definition.get("inputSchema") if isinstance(definition, Mapping) else None
        properties = schema.get("properties") if isinstance(schema, Mapping) else None
        if isinstance(properties, Mapping):
            names.update(str(name).lower() for name in properties)
    return frozenset(name for name in names if _matches_forbidden_fragment(name) and not _is_forbidden_term(name))


def _argument_value(value: Any) -> str:
    """A value's identity for matching an echo to what the policy wrote: canonical JSON with numbers as floats, so
    a tool that echoes `3` as `3.0` still echoes it."""

    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): plain(child) for key, child in item.items()}
        if isinstance(item, list):
            return [plain(child) for child in item]
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return float(item)
        return item

    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"))


def policy_argument_values(surface: Mapping[str, Any], events: Any) -> dict[str, frozenset[str]]:
    """The values the policy wrote for each name ``declared_argument_names`` returns: in the arguments of every
    `tool_call_proposed` (a code-mode script's calls included) at any depth, and the default the surface declares
    for it, which a tool may echo when the call left the argument out."""

    names = declared_argument_names(surface)
    if not names:
        return {}
    found: dict[str, set[str]] = {}

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in names:
                    found.setdefault(str(key).lower(), set()).add(_argument_value(child))
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    for definition in surface.get("definitions") or []:
        schema = definition.get("inputSchema") if isinstance(definition, Mapping) else None
        properties = schema.get("properties") if isinstance(schema, Mapping) else None
        for name, declared in properties.items() if isinstance(properties, Mapping) else ():
            if str(name).lower() in names and isinstance(declared, Mapping) and "default" in declared:
                found.setdefault(str(name).lower(), set()).add(_argument_value(declared["default"]))
    for event in events:
        if isinstance(event, Mapping) and event.get("type") == "tool_call_proposed":
            payload = event.get("payload")
            collect(payload.get("arguments") if isinstance(payload, Mapping) else None)
    return {name: frozenset(values) for name, values in found.items()}


def _policy_argument(
    lowered: str, child: Any, path: str,
    argument_values: Mapping[str, frozenset[str]], argument_declarations: frozenset[str],
) -> bool:
    """Whether a key holds the policy's own argument: a value it wrote for it, or the request's declaration of it."""

    if lowered in argument_declarations and _ARGUMENT_DECLARATION_PATH.fullmatch(path):
        return True
    values = argument_values.get(lowered)
    return bool(values) and _argument_value(child) in values


def _forbidden_path(
    value: Any,
    *,
    path: str = "$",
    allow_public_grasp_contact_pose: bool = False,
    safe_field_names: frozenset[str] = frozenset(),
    argument_values: Mapping[str, frozenset[str]] | None = None,
    argument_declarations: frozenset[str] = frozenset(),
    model_arguments: Mapping[str, str] | None = None,
) -> str | None:
    """The first path in ``value`` whose key is forbidden evidence, or None. ``argument_values`` and
    ``argument_declarations`` are the run's own tool arguments (``policy_argument_values``,
    ``declared_argument_names``); a caller passes declarations only for a provider request. ``model_arguments``
    are the tool calls the model wrote (``model_tool_arguments``), whose arguments are left out wherever a call
    repeats them exactly."""

    from .motion_feedback import is_public_contact_feedback_field

    argument_values = argument_values or {}
    if isinstance(value, Mapping):
        if model_arguments:
            value = _without_model_arguments(value, model_arguments)
        for key, child in value.items():
            lowered = str(key).lower()
            aggregate_contact = (
                lowered in {"contact", "contact_estimated"}
                and (path.endswith(".feedback") or path.endswith(".motion_feedback"))
                and (isinstance(child, bool) or child is None)
            )
            geometry_rule = PUBLIC_CONTACT_GEOMETRY_VALUES.get(lowered)
            public_contact_geometry = False
            if allow_public_grasp_contact_pose and geometry_rule is not None:
                expected_values, required_parent = geometry_rule
                public_contact_geometry = (
                    path.endswith(required_parent)
                    and isinstance(child, (int, float, str))
                    and not isinstance(child, bool)
                    and child in expected_values
                )
            public_grasp_contact_pose = (
                allow_public_grasp_contact_pose
                and lowered == "contact_pose"
                and _is_public_grasp_contact_pose(child)
            )
            # The staged controller phase labels live under `phases`; the
            # `local_contact` label names the local contact-approach stage
            # and its value is public tracking evidence, not contact sensing.
            staged_phase_label = (
                lowered == "local_contact" and path.endswith(".phases")
            )
            if (
                lowered not in safe_field_names
                and lowered not in PUBLIC_MOTION_VOCABULARY_KEYS
                and not staged_phase_label
                and not is_public_contact_feedback_field(path, lowered, child)
                and not aggregate_contact
                and not public_contact_geometry
                and not public_grasp_contact_pose
                and _matches_forbidden_fragment(lowered)
                and not _policy_argument(lowered, child, path, argument_values, argument_declarations)
            ):
                # The explicit absence declaration is safe evidence, not a leak.
                if lowered != "forbidden_modalities_absent":
                    return f"{path}.{key}"
            found = _forbidden_path(
                child,
                path=f"{path}.{key}",
                allow_public_grasp_contact_pose=allow_public_grasp_contact_pose,
                safe_field_names=safe_field_names,
                argument_values=argument_values,
                argument_declarations=argument_declarations,
                model_arguments=model_arguments,
            )
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _forbidden_path(
                child,
                path=f"{path}[{index}]",
                allow_public_grasp_contact_pose=allow_public_grasp_contact_pose,
                safe_field_names=safe_field_names,
                argument_values=argument_values,
                argument_declarations=argument_declarations,
                model_arguments=model_arguments,
            )
            if found:
                return found
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            pass
        else:
            return _forbidden_path(
                decoded,
                path=f"{path}<json>",
                allow_public_grasp_contact_pose=allow_public_grasp_contact_pose,
                safe_field_names=safe_field_names,
                argument_values=argument_values,
                argument_declarations=argument_declarations,
                model_arguments=model_arguments,
            )
    return None
