"""Audit frozen policy versus current runtime surfaces without requiring equality."""

import json

from ora.agent.toolkit.interface.types import CorruptRecord, canonical, digest
from ora.orchestrator.spec.values import json_sha256


FIELDS = ("capabilities", "definitions", "view_config", "result_schemas")


def differences(source, current, path=""):
    """JSON-pointer differences, including presence so missing differs from null."""
    if canonical(source) == canonical(current):
        return []
    if isinstance(source, dict) and isinstance(current, dict):
        keys = sorted(source.keys() | current.keys())
    elif isinstance(source, list) and isinstance(current, list):
        source, current = dict(enumerate(source)), dict(enumerate(current))
        keys = sorted(source.keys() | current.keys())
    else:
        return [
            dict(
                path=path,
                source_present=True,
                current_present=True,
                source=source,
                current=current,
            )
        ]
    result = []
    for key in keys:
        child = path + "/" + str(key).replace("~", "~0").replace("/", "~1")
        if key in source and key in current:
            result.extend(differences(source[key], current[key], child))
        else:
            result.append(
                dict(
                    path=child,
                    source_present=key in source,
                    current_present=key in current,
                    source=source.get(key),
                    current=current.get(key),
                )
            )
    return result


def audit(frozen, resolved):
    """A different implementation/projection warns; malformed or corrupt evidence fails.

    The plan stores advertised definitions after context/allowlist overlays.
    ManipulationToolRuntime resolves the underlying execution surface. These
    can differ even at the same commit; neither replaces the other here.
    """
    for key, kind in zip(FIELDS, (list, list, dict, dict)):
        if not isinstance(frozen.get(key), kind):
            raise CorruptRecord("malformed frozen tool surface: " + key)
    names = [
        row.get("name") if isinstance(row, dict) else None
        for row in frozen["definitions"]
    ]
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(
        names
    ):
        raise CorruptRecord("malformed frozen tool surface: tool names")
    claimed = frozen.get("advertised_composition_sha256")
    if claimed is not None:
        keys = [
            "definitions",
            "result_schemas",
            "mutation_tools",
            "lifecycle_tools",
            "derived_layers",
        ]
        if "observation_tools" in frozen:
            keys += ["observation_tools", "call_effect_profile"]
        if any(key not in frozen for key in keys):
            raise CorruptRecord("incomplete frozen advertised surface evidence")
        if json_sha256({key: frozen[key] for key in keys}) != claimed:
            raise CorruptRecord(
                "frozen advertised surface hash differs from its contents"
            )
    source = {key: frozen[key] for key in FIELDS}
    current = {key: json.loads(json.dumps(getattr(resolved, key))) for key in FIELDS}
    changes = differences(source, current)
    source_tools = {row["name"]: row for row in source["definitions"]}
    current_tools = {row["name"]: row for row in current["definitions"]}
    affected = {
        name
        for name in source_tools.keys() | current_tools.keys()
        if source_tools.get(name) != current_tools.get(name)
        or source["result_schemas"].get(name) != current["result_schemas"].get(name)
    }
    if any(source[key] != current[key] for key in ("capabilities", "view_config")):
        affected |= source_tools.keys() | current_tools.keys()
    return dict(
        schema="ora.toolkit.surface-compatibility.v1",
        status="different" if changes else "equal",
        blocking=False,
        source=dict(
            kind="frozen_policy_surface",
            profile_id=frozen["profile_id"],
            sha256=digest(canonical(source)),
            advertised_composition_sha256=claimed,
        ),
        current=dict(
            kind="diagnostic_runtime_surface",
            profile_id=resolved.profile_id,
            sha256=digest(canonical(current)),
        ),
        affected_tools=sorted(affected),
        differences=changes,
        warning=(
            None
            if not changes
            else dict(
                code="tool_surface_mismatch",
                message="Frozen policy and current diagnostic runtime surfaces differ; continuing with current implementation.",
                evidence="surface-compatibility.json",
                affected_tools=sorted(affected),
            )
        ),
    )
