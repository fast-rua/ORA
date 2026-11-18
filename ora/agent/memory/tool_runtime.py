"""The ToolRuntime wrapper that serves ``artifact_read`` and ``observe(ref)`` from
the catalog and forwards every other call to the base runtime.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ora.agent.composition.surface.tool_surface import validate_schema_instance
from ora.agent.error_classes import ToolError
from .artifact_openers import ARTIFACT_READ_NAMES, artifact_read_name
from .artifact_reader import (
    bounded_pointer_value,
    check_observe_ref_arguments,
    check_read_arguments,
    json_pointer,
    policy_resolve,
    read_range,
)
from .read_tool import (
    ARTIFACT_READ_LIMIT_BYTES,
    ARTIFACT_READ_RESULT_SCHEMA,
    ARTIFACT_READ_TOOL,
    HISTORICAL_OBSERVE_RESULT_SCHEMA,
    recoverable_artifact_definitions,
    recoverable_artifact_result_schemas,
)
from .catalog import ContextArtifactCatalog
from .markers import context_artifact_reader


_json_pointer = json_pointer  # moved to artifact_reader.py


@dataclass
class RecoverableArtifactToolRuntime:
    base: Any
    run_root: Path
    event_path: Path

    def __post_init__(self) -> None:
        # A toolkit-only profile takes the readers too: they are Context tools,
        # not robot tools, and lossless retention needs them to reopen what it
        # externalized.
        self.run_root = self.run_root.resolve()
        self.artifact_read_tool_name = artifact_read_name(self.base.definitions())
        self._surface = _RecoverableArtifactSurface(
            self.base.surface, self.artifact_read_tool_name,
        )
        self.catalog: ContextArtifactCatalog | None = None
        self.event_sink: Any | None = None

    @property
    def surface(self) -> Any:
        return self._surface

    @property
    def control_root(self) -> Path:
        return self.base.control_root

    @property
    def controlled_end_effector(self) -> str | None:
        return getattr(self.base, "controlled_end_effector", None)

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return tuple(self.base.allowed_tools) + (self.artifact_read_tool_name,)

    def bind_context_artifacts(
        self, catalog: ContextArtifactCatalog, *, event_sink: Any | None = None,
    ) -> None:
        if self.catalog is not None and self.catalog is not catalog:
            raise RuntimeError("recoverable Artifact catalog was already bound")
        catalog.reader_name = self.artifact_read_tool_name
        self.catalog = catalog
        if event_sink is not None:
            self.event_sink = event_sink

    def definitions(self) -> list[dict[str, Any]]:
        return recoverable_artifact_definitions(self.base.definitions())

    def classify_call_effect(self, name: str, arguments: Mapping[str, Any]) -> str:
        if name in ARTIFACT_READ_NAMES or (
            name == "observe" and arguments.get("ref") is not None
        ):
            return "read_only"
        return str(self.base.classify_call_effect(name, arguments))

    def _require_catalog(self) -> ContextArtifactCatalog:
        if self.catalog is None:
            raise RuntimeError("recoverable Artifact catalog is not bound")
        return self.catalog

    def _record(
        self, *, name: str, arguments: Mapping[str, Any], status: str,
        call_context: Mapping[str, Any], error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        if not callable(self.event_sink):
            raise RuntimeError("recoverable Artifact Trace sink is not bound")
        self.event_sink(
            "context_artifact_retrieval",
            visibility="derived",
            observation_step=call_context.get("observation_step"),
            model_call_id=call_context.get("model_call_id"),
            tool_call_id=call_context.get("tool_call_id"),
            payload={
                "tool": name,
                "ref": arguments.get("ref"),
                "status": status,
                "error": error,
                "duration_ms": duration_ms,
            },
        )

    def _observe_ref(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        check_observe_ref_arguments(arguments)
        catalog = self._require_catalog()
        entry, data = policy_resolve(catalog, str(arguments["ref"]))
        media_type = str(entry["media_type"])
        if not media_type.startswith("image/"):
            raise ToolError(
                f"observe(ref) requires an image artifact; use {self.artifact_read_tool_name} for text/JSON",
                code="invalid_arguments", field="ref", received=arguments["ref"],
                constraints={"media_type": media_type, "open": {
                    "tool": self.artifact_read_tool_name,
                    "arguments": {"ref": arguments["ref"]},
                }},
            )
        sources = entry.get("sources")
        source = (
            sources[0]
            if isinstance(sources, list) and sources
            and isinstance(sources[0], Mapping) else {}
        )
        result = {
            "status": "actual_historical_artifact",
            "ref": entry["ref"],
            "sha256": entry["sha256"],
            "media_type": media_type,
            "bytes": entry["bytes"],
            "round": source.get("round"),
            "action_step": source.get("action_step"),
            "views": ["historical_artifact"],
            "images_base64": [base64.b64encode(data).decode("ascii")],
        }
        validate_schema_instance(
            {key: value for key, value in result.items() if key != "images_base64"},
            HISTORICAL_OBSERVE_RESULT_SCHEMA,
        )
        return result

    def _read_ref(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        check_read_arguments(arguments, ARTIFACT_READ_TOOL["inputSchema"], ARTIFACT_READ_LIMIT_BYTES)
        catalog = self._require_catalog()
        entry, data = policy_resolve(catalog, str(arguments["ref"]))
        media_type = str(entry["media_type"])
        if media_type.startswith("image/"):
            raise ToolError(
                "ref names an image artifact; this reader returns text or JSON only",
                code="invalid_arguments", field="ref", received=arguments["ref"],
                constraints={"media_type": media_type},
            )
        pointer = arguments.get("json_pointer")
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", 4096)
        if pointer is not None:
            if media_type != "application/json":
                raise ToolError(
                    "json_pointer must be a string selecting a JSON artifact",
                    code="invalid_arguments", field="json_pointer", received=pointer,
                    constraints={"allowed": {
                        "type": "string", "media_type": "application/json",
                    }},
                )
            value = json.loads(data.decode("utf-8"))
            try:
                selected = _json_pointer(value, pointer)
            except (ValueError, KeyError) as error:
                root_keys = sorted(str(key) for key in value) if isinstance(value, Mapping) else []
                raise ToolError(
                    f"{error.args[0]}; select a path from this artifact's root",
                    code="invalid_arguments", field="json_pointer", received=pointer,
                    constraints={
                        "allowed": "an existing JSON Pointer; empty string selects the root",
                        "top_level_keys": root_keys,
                        "root_type": (
                            "object" if isinstance(value, Mapping) else
                            "array" if isinstance(value, list) else "scalar"
                        ),
                        **({"root_array_length": len(value)} if isinstance(value, list) else {}),
                    },
                ) from error
            return {
                "status": "ok", "ref": entry["ref"], "sha256": entry["sha256"],
                "media_type": media_type, "bytes": entry["bytes"], "offset": 0,
                "next_offset": len(data), "eof": True, "content": None,
                "json_value": bounded_pointer_value(selected, pointer, ARTIFACT_READ_LIMIT_BYTES),
            }
        offset = 0 if offset is None else offset
        text, next_offset = read_range(data, offset, 4096 if limit is None else limit)
        return {
            "status": "ok", "ref": entry["ref"], "sha256": entry["sha256"],
            "media_type": media_type, "bytes": entry["bytes"], "offset": offset,
            "next_offset": next_offset, "eof": next_offset >= len(data),
            "content": text, "json_value": None,
        }

    def invoke_with_context(
        self, name: str, arguments: Mapping[str, Any], *, call_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if name in ARTIFACT_READ_NAMES or (
            name == "observe" and arguments.get("ref") is not None
        ):
            started_ns = time.monotonic_ns()
            try:
                result = (
                    self._read_ref(arguments) if name in ARTIFACT_READ_NAMES
                    else self._observe_ref(arguments)
                )
                if name in ARTIFACT_READ_NAMES:
                    validate_schema_instance(result, ARTIFACT_READ_RESULT_SCHEMA)
            except Exception as exc:
                self._record(
                    name=name, arguments=arguments, status="rejected",
                    call_context=call_context, error=str(exc),
                    duration_ms=round(
                        (time.monotonic_ns() - started_ns) / 1_000_000, 3,
                    ),
                )
                raise
            self._record(
                name=name, arguments=arguments, status="ok",
                call_context=call_context,
                duration_ms=round(
                    (time.monotonic_ns() - started_ns) / 1_000_000, 3,
                ),
            )
            return result
        invoke = getattr(self.base, "invoke_with_context", None)
        return (
            invoke(name, arguments, call_context=call_context)
            if callable(invoke) else self.base.invoke(name, arguments)
        )

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if name in ARTIFACT_READ_NAMES or (
            name == "observe" and arguments.get("ref") is not None
        ):
            return self.invoke_with_context(name, arguments, call_context={})
        return self.base.invoke(name, arguments)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


@dataclass(frozen=True)
class _RecoverableArtifactSurface:
    base: Any
    reader_name: str

    @property
    def result_schemas(self) -> dict[str, Any]:
        return recoverable_artifact_result_schemas(
            self.base.result_schemas, reader_name=self.reader_name,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def build_recoverable_artifact_runtime(
    plan: Mapping[str, Any], *, run_root: Path, event_path: Path, base_runtime: Any,
) -> RecoverableArtifactToolRuntime | Any:
    if not context_artifact_reader(plan.get("policy", {})):
        return base_runtime
    return RecoverableArtifactToolRuntime(
        base=base_runtime, run_root=run_root, event_path=event_path,
    )
