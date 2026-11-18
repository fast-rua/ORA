"""Replay ToolRuntime, recorded provider responses and the session builder.

Nothing here reaches a simulator, a model service or a provider. The session
under replay is the one ``ora.orchestrator.episode.policy_factory.build_policy_session``
builds for the recorded RunPlan on the Pi agent core: the ``PiPlanner`` whose
agent core runs in TypeScript. The sealed runs were resolved for the Python
Native core ORA has since removed; their plan is replayed on Pi, which
reproduced Native's requests byte for byte. Only the external ports are
replaced:

* the ToolRuntime: :class:`ReplayToolRuntime` keeps the real resolved
  ToolSurface (allow-list, semantics, effect classifier, recoverable-Artifact
  reader), advertises the tool definitions the sealed run recorded in its
  plan, and returns the result the live ToolRuntime returned for each
  dispatched call, after checking the call's name, arguments and context;
* the provider transport: the recorded raw events go to the TypeScript
  ``ReplayProvider`` as a ``pi_provider_spec`` (``kind: replay``), which runs
  them through the provider's wire projection and SSE accumulation;
* the context compactor: scripted FakeProvider-style responses
  (:func:`compactor_responses`), the shape the TypeScript replay spec accepts;
  a faithful replay scripts none, so any compactor call fails.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import json
import shutil
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_SCHEMA = "ora.pi-spike.native-replay-fixture.v1"
#: A placeholder credential; it never leaves the process.
REPLAY_API_KEY = "ora-pi-spike-offline-replay"


class ReplayMismatch(AssertionError):
    """The replayed session asked for something the recording does not hold."""


def canonical_bytes(value: Any) -> bytes:
    from ora.agent.evidence.canonical import canonical_bytes as _canonical

    return _canonical(value)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8",
    )


# -- the fixture ------------------------------------------------------------


@dataclass
class Fixture:
    """A replay fixture directory written by :mod:`.extract`."""

    root: Path
    manifest: dict[str, Any]
    plan: dict[str, Any]
    script: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    provider_calls: list[dict[str, Any]]

    @classmethod
    def load(cls, root: Path) -> "Fixture":
        root = Path(root).resolve()
        manifest = read_json(root / "fixture.json")
        if manifest.get("schema") != FIXTURE_SCHEMA:
            raise ReplayMismatch(f"unsupported replay fixture schema: {manifest.get('schema')}")
        return cls(
            root=root,
            manifest=manifest,
            plan=read_json(root / "plan.json"),
            script=read_json(root / "script.json")["ops"],
            tool_calls=read_json(root / "tool-calls.json")["calls"],
            provider_calls=read_json(root / "provider-calls.json")["calls"],
        )

    def path(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ReplayMismatch(f"fixture path escapes the fixture: {relative}")
        return self.root / candidate

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def policy_session_id(self) -> str:
        return str(self.manifest["policy_session_id"])


# -- the real ToolRuntime, offline -----------------------------------------


class _OfflineServiceStub:
    """Stands in for the resident model-service clients.

    The real ``ManipulationToolRuntime`` probes service readiness when it
    builds its own clients; supplying these keeps construction offline. The
    replay never dispatches a real handler, so any call here is a bug.
    """

    def _refuse(self, *_args: Any, **_kwargs: Any) -> Any:
        raise ReplayMismatch("offline replay reached a model-service client")

    ground = segment = propose = segment_batch = _refuse


def recorded_tool_definitions(plan: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The tool definitions the sealed run advertised, as its RunPlan recorded them.

    ``policy_composition.tool_surface.definitions`` is what the live
    coordinator checked its ToolRuntime against when the run started, so it is
    exactly what the model was shown. ``None`` for a plan older than that
    snapshot.
    """

    planned = plan.get("policy_composition", {}).get("tool_surface", {}).get("definitions")
    return copy.deepcopy(planned) if isinstance(planned, list) else None


#: JSON Schema keywords whose value is one subschema, a list of them, or a map of them.
_SUBSCHEMA = frozenset({
    "items", "additionalItems", "additionalProperties", "unevaluatedItems",
    "unevaluatedProperties", "contains", "propertyNames", "not", "if", "then", "else",
})
_SUBSCHEMA_LISTS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SUBSCHEMA_MAPS = frozenset({
    "properties", "patternProperties", "dependentSchemas", "$defs", "definitions",
})


def schema_contract(schema: Any) -> Any:
    """``schema`` without its ``description`` annotations, at every subschema.

    What is left (types, properties, required members, enums, bounds) decides
    which arguments a tool accepts; the prose does not. A property that is
    itself named ``description`` is kept: only annotations are dropped.
    """

    if not isinstance(schema, Mapping):
        return copy.deepcopy(schema)
    contract: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "description":
            continue
        if key in _SUBSCHEMA:
            contract[key] = schema_contract(value)
        elif key in _SUBSCHEMA_LISTS and isinstance(value, list):
            contract[key] = [schema_contract(item) for item in value]
        elif key in _SUBSCHEMA_MAPS and isinstance(value, Mapping):
            contract[key] = {name: schema_contract(item) for name, item in value.items()}
        else:
            contract[key] = copy.deepcopy(value)
    return contract


def _without_added_optional_arguments(now: Any, then: Any) -> Any:
    """``now`` without the top-level properties ``then`` lacks and ``now`` does not require."""

    if not isinstance(now, Mapping) or not isinstance(then, Mapping):
        return now
    current, recorded = now.get("properties"), then.get("properties")
    if not isinstance(current, Mapping) or not isinstance(recorded, Mapping):
        return now
    required = set(now.get("required") or ())
    added = {name for name in current if name not in recorded and name not in required}
    if not added:
        return now
    return {**now, "properties": {name: value for name, value in current.items() if name not in added}}


def require_executable_definitions(
    current: Sequence[Mapping[str, Any]], recorded: Sequence[Mapping[str, Any]],
) -> None:
    """The current runtime runs every tool the sealed run advertised, with the same arguments.

    Tool and parameter descriptions are model-visible prose that a surface may
    revise in place; a sealed run carries the text it was shown in its plan,
    and the replay advertises that. The tools themselves must still be the
    same: the same names in the same order, and input schemas that agree
    apart from their descriptions and from optional top-level arguments the
    surface added since (the speed options, 2026-10-31): every call the
    sealed run made is still a valid call, and the replay advertises the
    recorded schema.
    """

    current_names = [item.get("name") for item in current]
    recorded_names = [item.get("name") for item in recorded]
    if current_names != recorded_names:
        raise ReplayMismatch(
            "offline ToolRuntime tools differ from the RunPlan composition: "
            f"{current_names} != recorded {recorded_names}"
        )
    for now, then in zip(current, recorded):
        extra = sorted(set(now) ^ set(then))
        if extra:
            raise ReplayMismatch(f"tool {now.get('name')}: definition keys differ from the recording: {extra}")
        for key in sorted(set(now) - {"name", "description"}):
            value = _without_added_optional_arguments(now[key], then[key]) if key == "inputSchema" else now[key]
            if canonical_bytes(schema_contract(value)) != canonical_bytes(schema_contract(then[key])):
                raise ReplayMismatch(
                    f"tool {now.get('name')}: {key} differs from the RunPlan composition "
                    "beyond its descriptions"
                )


def build_offline_tool_runtime(
    plan: Mapping[str, Any], *, run_root: Path, manual_path: Path,
) -> Any:
    """The RunPlan's ToolRuntime as ``coordinator._compose_tool_runtime`` builds it.

    Same profile, overrides, allow-list, workspace bounds and controlled end
    effector; the model-service clients are offline stubs and the motion
    callbacks are absent (only real handlers would use either). Its tools must
    be the ones the plan recorded (``require_executable_definitions``); their
    descriptions are today's text, which the replay does not advertise.
    """

    from ora.agent.composition import resolve_tool_surface
    from ora.agent.context.observation.contact_points import project_policy_public_facts
    from ora.agent.memory.tool_runtime import build_recoverable_artifact_runtime
    from ora.agent.toolkit.host.runtime.base import RobotToolRuntime
    from ora.agent.toolkit.host.runtime import ManipulationToolRuntime

    policy = plan["policy"]
    runtime_config = plan["runtime"]
    surface_binding = policy.get("tool_surface", {})
    profile = surface_binding.get("profile", "robot_runtime_v2")
    overrides = surface_binding.get("component_overrides", {})
    toolkit = {key: plan.get(key) for key in ("toolkit_grants", "toolkit_receipt_delivery")}
    surface = resolve_tool_surface(profile, component_overrides=overrides, **toolkit)
    events = run_root / "events.jsonl"
    arguments: dict[str, Any] = dict(
        control_root=run_root / "control",
        manual_path=manual_path,
        event_path=events,
        workspace_bounds_m=runtime_config["workspace_bounds_m"],
        allowed_tools=policy.get("tool_allowlist", ["robot.submit_action"]),
        state_root=run_root / "policy-state",
        tool_surface_profile=profile,
        tool_surface_overrides=overrides,
        controlled_end_effector=runtime_config.get("controlled_end_effector"),
        **toolkit,
    )
    if surface.runtime_id == "native_atomic_causal":
        stub = _OfflineServiceStub()
        base = ManipulationToolRuntime(
            **arguments,
            prerequisite_binding=(
                plan.get("prerequisite")
                if isinstance(plan.get("prerequisite"), Mapping) else None
            ),
            segmentation_backend=stub, grounding_backend=stub, grasp_backend=stub,
            public_facts_projector=project_policy_public_facts,
        )
    elif surface.runtime_id == "robot_runtime":
        base = RobotToolRuntime(**arguments)
    else:
        raise ReplayMismatch(f"unregistered tool runtime: {surface.runtime_id}")
    if surface.supports("toolkit_only_v1"):
        from ora.agent.toolkit.host.runtime.dispatch import install

        install(base)
    tools = build_recoverable_artifact_runtime(
        plan, run_root=run_root, event_path=events, base_runtime=base,
    )
    planned = recorded_tool_definitions(plan)
    if planned is not None:
        require_executable_definitions(tools.definitions(), planned)
    return tools


# -- the replay ToolRuntime ------------------------------------------------


def _recorded_tool_error(structured: Mapping[str, Any]) -> Exception:
    """A ToolError whose ``as_dict`` is exactly the recorded refusal."""

    from ora.agent.error_classes import ToolError

    class RecordedToolError(ToolError):
        def __init__(self, value: Mapping[str, Any]) -> None:
            super().__init__(
                str(value.get("message", "")),
                code=str(value.get("code", "invalid_arguments")),
                retryable=bool(value.get("retryable", True)),
            )
            self._recorded = copy.deepcopy(dict(value))

        def as_dict(self) -> dict[str, Any]:
            return copy.deepcopy(self._recorded)

    return RecordedToolError(structured)


class ReplayActionExecutor:
    """What ``PiToolRpc`` reads of the Supervisor's ``ToolActionExecutor``: the recorded ledger of the last call."""

    active = None
    blocked = False
    interruption = None

    def __init__(self) -> None:
        self.completed: dict[str, Any] | None = None
        self.loop = types.SimpleNamespace(environment_status={"terminal": False})


class ReplayToolRuntime:
    """What ``NativeToolGateway`` and ``PiPlanner`` use, served from a recording.

    ``surface``, ``allowed_tools``, ``classify_call_effect`` and
    ``bind_context_artifacts`` delegate to the real offline ToolRuntime of the
    plan. ``definitions()`` is what the sealed run advertised, from its plan
    (``recorded_tool_definitions``): the replay reproduces what the model saw,
    not the surface's current wording, and ``build_offline_tool_runtime``
    already required the real runtime to have the same tools and input
    schemas. ``invoke_with_context``/``invoke`` return the recorded result of
    the next dispatched call after checking its name, its arguments (as the
    gateway normalized them) and its call context; a mutation that the live
    tool persisted also writes ``control/submitted-action.json`` and closes the
    published observation exactly as ``RobotToolRuntime._persist_action`` did.
    On a synchronous surface the live tool ran its actions inside the call:
    ``run_tool_action`` (``run_script``) plays each of them where it ran, and
    ``executor`` hands the Pi session the call's recorded ledger.
    """

    def __init__(self, *, real: Any, fixture: Fixture, control_root: Path) -> None:
        self.real = real
        self.fixture = fixture
        self._control_root = Path(control_root).resolve()
        self._definitions = recorded_tool_definitions(fixture.plan)
        self._pending = [call for call in fixture.tool_calls if call["dispatched"]]
        self._cursor = 0
        self.executor = ReplayActionExecutor() if real.surface.supports("toolkit_synchronous_execution_v1") else None
        self.run_tool_action: Callable[..., None] | None = None
        self.invocations: list[dict[str, Any]] = []
        self.classifications: list[dict[str, Any]] = []

    # -- the surface, delegated ------------------------------------------

    @property
    def surface(self) -> Any:
        return self.real.surface

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return tuple(self.real.allowed_tools)

    @property
    def control_root(self) -> Path:
        return self._control_root

    @property
    def controlled_end_effector(self) -> str | None:
        return getattr(self.real, "controlled_end_effector", None)

    def definitions(self) -> list[dict[str, Any]]:
        if self._definitions is None:  # a plan older than the definitions snapshot
            return self.real.definitions()
        return copy.deepcopy(self._definitions)

    def classify_call_effect(self, name: str, arguments: Mapping[str, Any]) -> str:
        effect = str(self.real.classify_call_effect(name, arguments))
        self.classifications.append({"name": name, "effect": effect})
        return effect

    def bind_context_artifacts(self, catalog: Any, *, event_sink: Any | None = None) -> None:
        bind = getattr(self.real, "bind_context_artifacts", None)  # as PiPlanner: a plan without the Artifact reader
        if callable(bind):
            bind(catalog, event_sink=event_sink)

    # -- read-only feedback batches --------------------------------------

    def begin_read_only_feedback_batch(
        self, name: str, calls: Sequence[Mapping[str, Any]],
    ) -> bool:
        window = self._pending[self._cursor:self._cursor + len(calls)]
        if len(window) != len(calls) or any(
            record["name"] != name
            or canonical_bytes(record["runtime_arguments"]) != canonical_bytes(dict(call))
            for record, call in zip(window, calls)
        ):
            raise ReplayMismatch(f"read-only batch of {name} differs from the recording")
        combined = [record.get("visual_feedback") == "combined" for record in window]
        if any(combined):
            raise NotImplementedError(
                "combined read-only feedback batches are not modelled by this replay"
            )
        return False

    def finish_read_only_feedback_batch(
        self, name: str, results: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        raise ReplayMismatch("replay never begins a combined read-only feedback batch")

    # -- dispatch ----------------------------------------------------------

    def invoke_with_context(
        self, name: str, arguments: Mapping[str, Any], *, call_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._dispatch(name, arguments, call_context)

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._dispatch(name, arguments, None)

    def remaining(self) -> list[dict[str, Any]]:
        return self._pending[self._cursor:]

    def _dispatch(
        self, name: str, arguments: Mapping[str, Any],
        call_context: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if self._cursor >= len(self._pending):
            raise ReplayMismatch(f"unrecorded tool call {name}")
        record = self._pending[self._cursor]
        self._cursor += 1
        problems = []
        if name != record["name"]:
            problems.append(f"name {name!r} != recorded {record['name']!r}")
        if canonical_bytes(dict(arguments)) != canonical_bytes(record["runtime_arguments"]):
            problems.append("arguments differ from the recorded runtime arguments")
        if call_context is not None:
            for key in ("tool_call_id", "model_call_id", "observation_step"):
                if call_context.get(key) != record.get(key):
                    problems.append(
                        f"call_context.{key} {call_context.get(key)!r} != recorded {record.get(key)!r}"
                    )
        if problems:
            raise ReplayMismatch(f"tool call {record['tool_call_id']}: " + "; ".join(problems))
        self.invocations.append({
            "tool_call_id": record["tool_call_id"],
            "name": name,
            "call_context": None if call_context is None else {
                key: call_context.get(key) for key in sorted(call_context)
            },
        })
        if record["source"] == "context_artifact_reader":
            # The recoverable-Artifact reader is Harness-owned: it resolves
            # from the replay's own Context Artifact catalog.
            return self.real.invoke_with_context(
                name, arguments, call_context=call_context or {},
            )
        for action in record.get("inline_actions", ()):
            self.run_tool_action(action, call_context, name)
        if self.executor is not None:
            self.executor.completed = copy.deepcopy(record.get("execution"))
        outcome = record["outcome"]
        if "error" in outcome:
            raise _recorded_tool_error(outcome["error"])
        result = copy.deepcopy(outcome["result"])
        images = [
            base64.b64encode(self.fixture.path(image["path"]).read_bytes()).decode("ascii")
            for image in outcome.get("images", [])
        ]
        if images:
            key = outcome["image_key"]
            result[key] = images if key == "images_base64" else images[0]
        action = record.get("submitted_action")
        if action is not None:
            self._persist_action(action)
        return result

    def _persist_action(self, action: Mapping[str, Any]) -> None:
        """What ``RobotToolRuntime._persist_action`` writes before returning."""

        target = self._control_root / "submitted-action.json"
        if target.exists():
            raise ReplayMismatch("an action is already submitted for this observation")
        target.write_text(
            json.dumps(action, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pointer = self._control_root / "current-observation.json"
        control = read_json(pointer)
        control["open"] = False
        pointer.write_text(json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# -- provider transports ----------------------------------------------------


def inflate_usage(events: list[Any], factor: float) -> list[Any]:
    """Scale every reported ``prompt_tokens`` (and its ``total_tokens``)."""

    inflated = copy.deepcopy(events)
    for event in inflated:
        usage = event.get("usage") if isinstance(event, dict) else None
        if not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int:
            continue
        before = usage["prompt_tokens"]
        after = int(round(before * factor))
        usage["prompt_tokens"] = after
        if type(usage.get("total_tokens")) is int:
            usage["total_tokens"] += after - before
    return inflated


def _install(provider: Any, request_once: Callable[[Mapping[str, Any]], Any]) -> Any:
    """Answer a Python provider's requests with ``request_once`` (``extract`` checks recorded
    responses through the provider's own SSE accumulation)."""
    # The deadline recovery policy would run the transport in a worker
    # process; the legacy path calls ``_request_once`` in process. Both feed
    # the same SSE accumulation below it.
    provider.recovery_policy = "legacy_v1"
    provider.recovery_window_s = None
    provider._request_once = request_once
    return provider


#: The scripted L6 checkpoint (all seven v1 fields, qualitative only).
def default_checkpoint(revision: int) -> dict[str, Any]:
    return {
        "task_progress": [
            f"Checkpoint revision {revision}: the robot has been searching for and grasping "
            "a soup can to place in the basket; placement is not yet confirmed.",
        ],
        "confirmed_facts": [
            "The basket stands on the right side of the table and is open from above.",
        ],
        "invalidated_beliefs": [
            "The front red can was first assumed to be the alphabet soup; a later check did "
            "not support that belief.",
        ],
        "failed_attempts": [{
            "failure": "A steep side approach to the first can was refused by the planner.",
            "evidence": "The motion receipt reported a planning refusal and the arm stayed put.",
            "correction": "Approach from above with the jaws pointing down.",
        }],
        "current_constraints": [
            "Keep a held can above the basket rim before descending into it.",
        ],
        "do_not_repeat": ["Do not resend the refused side approach unchanged."],
        "re_ground_next": [
            "Re-ground the candidate can and the basket from the current images first.",
        ],
    }


def compactor_responses(count: int = 16) -> list[dict[str, Any]]:
    """Scripted compactor replies, FakeProvider-shaped, identical for both cores.

    Usage is fixed (not derived from the request) so the scripted reply cannot
    depend on which core produced the compactor request; it has the
    provider-neutral shape ``normalize_usage`` gives an OpenAI reply.
    """

    responses = []
    for revision in range(1, count + 1):
        content = json.dumps(default_checkpoint(revision), ensure_ascii=False, separators=(",", ":"))
        output_tokens = max(1, len(content.encode("utf-8")) // 3)
        responses.append({
            "message": {"role": "assistant", "content": content},
            "usage": {
                "input_tokens": 40000, "cached_input_tokens": 0, "output_tokens": output_tokens,
                "reasoning_output_tokens": 0, "total_tokens": 40000 + output_tokens,
                "cache_read_input_tokens": None, "cache_creation_input_tokens": None,
                "cache_read_status": "missing", "cache_creation_status": "missing",
            },
            "finish_reason": "stop",
        })
    return responses


def build_provider(policy: Mapping[str, Any], *, run_id: str) -> Any:
    """The provider ``build_policy_session`` builds, with a placeholder credential."""

    from ora.agent.providers import factory

    return factory.build_provider(
        policy["provider"], base_url=policy["endpoint"], api_key=REPLAY_API_KEY,
        standby_api_keys=(), credential_start_index=0,
        recovery_policy=policy.get("provider_recovery", "legacy_v1"),
        recovery_window_s=policy.get("provider_recovery_window_s"),
        timeout_seconds=float(policy.get("timeout_s", 900)),
        auth_scheme=policy.get("auth_scheme"),
        output_token_parameter=policy.get("output_token_parameter"),
        tool_strictness=policy.get("tool_strictness"),
        reasoning_field=policy.get("reasoning_field"),
        continuation_replay=policy.get("provider_continuation_replay", "exact_v1"),
        session_affinity_id=(
            f"ora:{run_id}:native"
            if policy.get("provider_session_affinity") == "x_session_id_run_v1"
            else None
        ),
    )


# -- the session -------------------------------------------------------------


@contextlib.contextmanager
def fixed_policy_session_id(session_id: str) -> Iterator[None]:
    """Make the session id the Pi core mints (``native-policy-<uuid4 hex>``) the recorded one.

    The id is part of every ``artifact://<session>/<sha>`` reference the
    Context shows the model, so the wire bytes depend on it. Patching each
    module's ``uuid`` lets the unmodified ``build_policy_session`` run.
    """

    import ora.agent.pi.policy_session as pi_module

    prefix = "native-policy-"
    if not session_id.startswith(prefix):
        raise ReplayMismatch(f"unexpected policy session id: {session_id}")
    value = session_id[len(prefix):]
    shim = types.SimpleNamespace(uuid4=lambda: types.SimpleNamespace(hex=value))
    with mock.patch.object(pi_module, "uuid", shim):
        yield


@contextlib.contextmanager
def agent_core(core: str) -> Iterator[None]:
    """Name the agent core ``build_policy_session`` builds (``ORA_AGENT_CORE``) explicitly: ``pi``."""

    import os

    if core != "pi":
        raise ReplayMismatch(f"unknown agent core: {core}")
    previous = os.environ.get("ORA_AGENT_CORE")
    os.environ["ORA_AGENT_CORE"] = core
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("ORA_AGENT_CORE", None)
        else:
            os.environ["ORA_AGENT_CORE"] = previous


def recorded_events(fixture: "Fixture", *, inflate_factor: float | None,
                    inflate_from_call: int) -> list[list[Any]]:
    """The recorded raw events of every model call, usage-inflated where asked."""

    recordings = []
    for index, call in enumerate(fixture.provider_calls):
        events = read_json(fixture.path(call["raw_events"]["path"]))
        if inflate_factor is not None and index >= inflate_from_call:
            events = inflate_usage(events, float(inflate_factor))
        recordings.append(events)
    return recordings


def pi_replay_spec(fixture: "Fixture", *, inflate_factor: float | None, inflate_from_call: int,
                   compactor: list[dict[str, Any]]) -> dict[str, Any]:
    """The TypeScript ``ReplayProvider`` spec, with the plan's wire options."""

    policy = fixture.plan["policy"]
    return {
        "kind": "replay",
        "recordings": recorded_events(
            fixture, inflate_factor=inflate_factor, inflate_from_call=inflate_from_call,
        ),
        "compactor_responses": copy.deepcopy(compactor),
        "output_token_parameter": policy.get("output_token_parameter"),
        "tool_strictness": policy.get("tool_strictness"),
        "reasoning_field": policy.get("reasoning_field"),
    }


def materialize_compiled_policy(fixture: Fixture, run_root: Path) -> tuple[dict[str, Path], str]:
    """Compile the plan's policy layers as the coordinator does; check the fixture copy."""

    from ora.agent.composition.prompt.policy_layers import PolicyCompositionError, materialize_policy

    destination = run_root / "compiled-policy"
    source = "materialized_from_plan_layers"
    try:
        compiled = materialize_policy(
            fixture.plan["policy_context"], project_root=REPO_ROOT, destination=destination,
        )
    except PolicyCompositionError:
        # Layer sources drifted after the run: fall back to the sealed text.
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(fixture.path("compiled-policy"), destination)
        compiled = {
            "harness_system": destination / "harness-system.md",
            "project_instructions": destination / "project-instructions.md",
            "episode_task": destination / "episode-task.md",
            "tool_manual": destination / "tool-manual.md",
        }
        compiled["codex_project"] = compiled["project_instructions"]
        compiled["episode_prefix"] = compiled["episode_task"]
        source = "fixture_copy"
    for expected in sorted(fixture.path("compiled-policy").iterdir()):
        if (destination / expected.name).read_bytes() != expected.read_bytes():
            raise ReplayMismatch(f"compiled policy {expected.name} differs from the sealed run")
    return {key: Path(value) for key, value in compiled.items()}, source


@dataclass
class ReplaySession:
    core: str
    planner: Any
    tools: ReplayToolRuntime
    #: The scripted compactor replies.
    compactor_responses: list[dict[str, Any]]
    run_root: Path
    compiled_policy: dict[str, Path]
    compiled_policy_source: str
    notes: list[str] = field(default_factory=list)

    @property
    def native(self) -> Any:
        """The ``PiPlanner`` behind the typed PolicySession (named for the native_api policy kind)."""

        return self.planner.delegate


def build_replay_session(
    fixture: Fixture, run_root: Path, *, core: str = "pi",
    inflate_factor: float | None = None, inflate_from_call: int = 0,
    compactor: list[dict[str, Any]] | None = None,
) -> ReplaySession:
    """Build the real PolicySession of ``core`` for the fixture's plan in a fresh run root."""

    from ora.orchestrator.episode.policy_factory import (
        build_policy_session, ensure_policy_session, policy_session_hints,
    )

    run_root = Path(run_root).resolve()
    if (run_root / "agent-events.jsonl").exists():
        raise ReplayMismatch(f"replay run root is not fresh: {run_root}")
    for directory in ("control", "native", "sandbox/workspace", "policy-observations", "policy-state"):
        (run_root / directory).mkdir(parents=True, exist_ok=True)
    # verify_agent_trace and the lossless evidence check read the run's plan.
    shutil.copyfile(fixture.path("plan.json"), run_root / "plan.json")
    # The sealed plan was resolved for the retired Native core (or before
    # policy.agent_core existed), which agent_core.selected refuses to start; the
    # replay runs it on Pi. The field is not model-visible.
    plan = copy.deepcopy(fixture.plan)
    plan["policy"]["agent_core"] = core
    policy = plan["policy"]
    compiled, compiled_source = materialize_compiled_policy(fixture, run_root)
    real = build_offline_tool_runtime(plan, run_root=run_root, manual_path=compiled["tool_manual"])
    tools = ReplayToolRuntime(real=real, fixture=fixture, control_root=run_root / "control")
    scripted = list(compactor or [])
    provider = types.SimpleNamespace(pi_provider_spec=pi_replay_spec(
        fixture, inflate_factor=inflate_factor, inflate_from_call=inflate_from_call,
        compactor=scripted,
    ))

    def _record_context_bundle(*_args: Any, **_kwargs: Any) -> None:
        raise ReplayMismatch("a native_api session does not record context bundles")

    with fixed_policy_session_id(fixture.policy_session_id), agent_core(core):
        planner = build_policy_session(
            plan=plan, run_root=run_root, run_id=fixture.run_id,
            workspace=run_root / "sandbox/workspace", control_root=run_root / "control",
            events_path=run_root / "events.jsonl", compiled_policy=compiled, tools=tools,
            state_root=run_root / "policy-state", record_context_bundle=_record_context_bundle,
            provider_override=provider,
        )
    if tools.executor is not None:
        planner.bind_tool_action_executor(
            tools.executor, observation_receipts=plan.get("toolkit_observation_receipts", False),
        )
    hints = policy_session_hints(policy)
    planner = ensure_policy_session(
        planner, kind=hints["session_kind"], supports_images=hints["supports_images"],
    )
    planner.start()
    if planner.delegate.policy_session_id != fixture.policy_session_id:
        raise ReplayMismatch("replayed policy session id differs from the recording")
    if type(planner.delegate).__name__ != "PiPlanner":
        raise ReplayMismatch(f"build_policy_session built {type(planner.delegate).__name__}, not PiPlanner")
    return ReplaySession(
        core=core, planner=planner, tools=tools,
        compactor_responses=scripted, run_root=run_root,
        compiled_policy=compiled, compiled_policy_source=compiled_source,
    )


__all__ = [
    "Fixture", "FIXTURE_SCHEMA", "ReplayMismatch", "ReplaySession",
    "ReplayToolRuntime", "agent_core", "build_offline_tool_runtime",
    "build_provider", "build_replay_session", "canonical_bytes", "canonical_sha256",
    "compactor_responses", "default_checkpoint", "fixed_policy_session_id", "inflate_usage",
    "pi_replay_spec", "read_json", "recorded_events", "recorded_tool_definitions",
    "require_executable_definitions", "schema_contract", "sha256_bytes", "write_json",
]
