"""The ``artifact_read`` tool's definition and result schemas, and the observe
result schemas a recoverable ref adds; the surface composer advertises them.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from .artifact_openers import artifact_read_name


ARTIFACT_READ_LIMIT_BYTES = 8192


ARTIFACT_READ_TOOL: dict[str, Any] = {
    "name": "artifact.read",
    "description": (
        "Retrieve exact text or JSON evidence from this Session's artifact store. "
        "Use when: exact historical text, JSON, a state dump, receipt, or diagnostic "
        "was replaced by an artifact:// reference. Effect: read-only and does not "
        "change the robot or Memory. Returns: one bounded byte range, or one JSON "
        "Pointer value, together with the verified content identity and continuation "
        "offset. Does not: expose host paths, cross Session scope, bypass artifact "
        "hash checks, or return images; reopen image references with observe(ref=...). "
        "On error: correct the reference/range or re-ground from current evidence; "
        "missing, tampered, wrong-media, and cross-scope references fail closed."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "minLength": 1,
                    "description": "Exact Session-scoped artifact:// reference supplied by a recoverable-artifact pointer."},
            "offset": {"type": ["integer", "null"], "minimum": 0,
                       "description": "Starting UTF-8 byte offset for a text range; omit or pass null to start at zero."},
            "limit": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": ARTIFACT_READ_LIMIT_BYTES,
                "description": "Maximum bytes to read in one text range; omit or pass null for the bounded default.",
            },
            "json_pointer": {"type": ["string", "null"],
                             "description": "JSON Pointer selecting one value from a JSON artifact; do not combine with a byte range. Omit or pass null for byte-range reading."},
        },
        "required": ["ref"],
        "additionalProperties": False,
    },
}

# Keep the mutually exclusive modes explicit on the model-visible schema.
# Null/default range values retain the existing decoder's accepted semantics.
_ARTIFACT_READ_PROPERTIES = ARTIFACT_READ_TOOL["inputSchema"]["properties"]
ARTIFACT_READ_TOOL["inputSchema"]["oneOf"] = [
    {
        "type": "object",
        "properties": {
            **copy.deepcopy(_ARTIFACT_READ_PROPERTIES),
            "json_pointer": {
                "type": "null", "description": "Omit or pass null for byte-range reading.",
            },
        },
        "required": ["ref"],
        "additionalProperties": False,
    },
    {
        "type": "object",
        "properties": {
            "ref": copy.deepcopy(_ARTIFACT_READ_PROPERTIES["ref"]),
            "json_pointer": {
                "type": "string", "description": "Existing JSON Pointer selecting one JSON value.",
            },
            "offset": {
                "enum": [None, 0], "description": "Omit in Pointer mode; null or the default zero is accepted.",
            },
            "limit": {
                "enum": [None, 4096], "description": "Omit in Pointer mode; null or the default 4096 is accepted.",
            },
        },
        "required": ["ref", "json_pointer"],
        "additionalProperties": False,
    },
]


ARTIFACT_READ_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "const": "ok"},
        "ref": {
            "type": "string",
            "pattern": (
                r"^artifact://[A-Za-z0-9][A-Za-z0-9._-]{0,159}/"
                r"[0-9a-f]{64}$"
            ),
        },
        "sha256": {"type": "string"},
        "media_type": {"type": "string"},
        "bytes": {"type": "integer"},
        "offset": {"type": "integer"},
        "next_offset": {"type": "integer"},
        "eof": {"type": "boolean"},
        "content": {"type": ["string", "null"]},
        "json_value": {},
    },
    "required": [
        "status", "ref", "sha256", "media_type", "bytes", "offset",
        "next_offset", "eof", "content", "json_value",
    ],
    "additionalProperties": False,
}


HISTORICAL_OBSERVE_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "const": "actual_historical_artifact"},
        "ref": {"type": "string"},
        "sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
        "media_type": {"type": "string", "pattern": r"^image/"},
        "bytes": {"type": "integer", "minimum": 0},
        "round": {"type": ["integer", "null"], "minimum": 0},
        "action_step": {"type": ["integer", "null"], "minimum": 0},
        "views": {
            "type": "array",
            "items": {"type": "string", "const": "historical_artifact"},
            "minItems": 1,
            "maxItems": 1,
        },
    },
    "required": [
        "status", "ref", "sha256", "media_type", "bytes", "round",
        "action_step", "views",
    ],
    "additionalProperties": False,
}


CURRENT_OBSERVE_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "const": "actual"},
        "views": {"type": "array", "items": {"type": "string"}},
        "view_dimensions": {"type": "object"},
    },
    "required": ["status", "views"],
    "additionalProperties": False,
}



def recoverable_artifact_definitions(
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add the orthogonal Context reader without changing a frozen ToolSurface."""

    values = [copy.deepcopy(dict(item)) for item in definitions]
    observe = next((item for item in values if item.get("name") == "observe"), None)
    if not isinstance(observe, dict):
        raise ValueError("lossless Context requires an observe tool")
    schema = observe.get("inputSchema")
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    if not isinstance(properties, Mapping):
        raise ValueError("observe input schema is unavailable")
    current_schema = copy.deepcopy(dict(schema))
    ref_schema = {
        "type": "string",
        "description": (
            "Exact Session-scoped artifact:// reference from a "
            "<recoverable_artifact> pointer."
        ),
        "pattern": (
            r"^artifact://[A-Za-z0-9][A-Za-z0-9._-]{0,159}/"
            r"[0-9a-f]{64}$"
        ),
    }
    historical_schema = {
        "type": "object",
        "properties": {"ref": copy.deepcopy(ref_schema)},
        "required": ["ref"],
        "additionalProperties": False,
    }
    root_properties = copy.deepcopy(dict(properties))
    root_properties["ref"] = copy.deepcopy(ref_schema)
    observe["inputSchema"] = {
        # Provider tool protocols require a top-level object declaration. The
        # branches remain authoritative for the two mutually exclusive modes.
        "type": "object",
        "properties": root_properties,
        "oneOf": [current_schema, historical_schema],
        "additionalProperties": False,
    }
    observe["description"] = (
        str(observe.get("description", "")).rstrip()
        + " A <recoverable_artifact> pointer gives the exact historical mode: "
        "pass its ref by itself to the named open tool to reopen that image. "
        "The Runtime verifies Session scope, media type and SHA-256 before returning it."
    )
    reader = copy.deepcopy(ARTIFACT_READ_TOOL)
    reader["name"] = artifact_read_name(definitions)
    values.append(reader)
    return values


def recoverable_artifact_result_schemas(
    result_schemas: Mapping[str, Mapping[str, Any]],
    *, reader_name: str | None = None,
) -> dict[str, Any]:
    values = copy.deepcopy(dict(result_schemas))
    observe = values.get("observe")
    if observe is None:
        observe = {"oneOf": [copy.deepcopy(CURRENT_OBSERVE_RESULT_SCHEMA)]}
        values["observe"] = observe
    if not isinstance(observe, dict):
        raise ValueError("observe result schema must be an object")
    branches = observe.get("oneOf")
    if not isinstance(branches, list):
        metadata = {
            key: copy.deepcopy(observe[key])
            for key in ("$schema", "$defs") if key in observe
        }
        current = {
            key: copy.deepcopy(value) for key, value in observe.items()
            if key not in metadata
        }
        observe = {**metadata, "oneOf": [current]}
        values["observe"] = observe
        branches = observe["oneOf"]
    branches.append(copy.deepcopy(HISTORICAL_OBSERVE_RESULT_SCHEMA))
    # Keyed the way this surface spells its tools; the schemas map is keyed by
    # advertised name and must agree with the definition appended beside it.
    values[reader_name or artifact_read_name([{"name": key} for key in result_schemas])] = (
        copy.deepcopy(ARTIFACT_READ_RESULT_SCHEMA)
    )
    return values
