"""Named context schemes: the context half of a baseline, selectable on its own.

A baseline fixes the tool surface, prompt bundle, action codec, runtime and a
context scheme as one unit. A context scheme
(``ora/policy/context_schemes/<name>.json``, ``ora.context-scheme.v1``) names
the context fields alone; every baseline names its scheme by id
(``"context_scheme"`` in the catalog) instead of carrying the fields inline, so
a RunSpec can keep its baseline and select another scheme with
``policy.context_scheme``. The scheme then replaces every ``CONTEXT_FIELDS``
entry of the baseline: a field the scheme omits is not inherited.

Schemes are the files of ``SCHEMES_DIR``, each known by the ``id`` inside it
(``context_schemes``); two files with one id are refused, and so is a file that
is not a scheme. Adding a scheme is adding a file. A patch edits a scheme in place (a sealed run records the scheme's
path and sha256 and keeps its own copy); a new name is for a changed context
contract (``ora/policy/README.md``, "Versions, patches and retirement").

A scheme also owns ``context_tools``, the Context tools it provides (today only
``artifact_read``, the recoverable-artifact reader). A scheme without the field
keeps the old rule: lossless retention provides the reader.

A scheme may name its own global prompt, ``"prompt": "<file>.md"`` beside the
JSON: how context reaches the model under this scheme (what stays inline, what
is compacted, how observations are paired). A prompt bundle with a
``context_prompt`` layer delivers it in the system message after the planner;
other bundles never deliver it. A frozen scheme gains a prompt only under a new
name.

An ``ora.context-scheme.v2`` scheme lists the stages the agent core runs instead
of a management object: its ``policy`` holds the other context fields, and
``segments``, ``limits``, ``estimator``, ``references`` and ``stages`` the stage
list, validated with the parameter schemas of ``ora/policy/context_stages/``
(``schemas/context_stages.py``). It resolves to ``policy.context_stages`` and,
while the evidence reads one, the management object it is the stage list of
(``policy.context_management``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .locked_assets import locked_asset

SCHEMA = "ora.context-scheme.v1"
#: A scheme that lists its stages (``schemas/context_stages.py``).
SCHEMA_V2 = "ora.context-scheme.v2"
SCHEMAS = (SCHEMA, SCHEMA_V2)
#: A v2 scheme's keys besides its policy.
V2_STAGE_KEYS = ("segments", "limits", "estimator", "references", "stages")
#: The directory of the scheme files, relative to the source tree.
SCHEMES_DIR = "ora/policy/context_schemes"
#: The baseline policy fields a context scheme owns.
CONTEXT_FIELDS = (
    "context_profile", "context_composition", "context_retention", "context_token_estimator",
    "summary_policy", "memory_policy", "provider_continuation_replay", "initial_observation",
    "context_management", "context_tools", "context_stages",
)
#: The fields a v2 scheme's stages stand for; its ``policy`` does not name them.
STAGE_FIELDS = ("context_management", "context_stages")


def context_schemes(root: Path) -> dict[str, str]:
    """``{id: path}`` of every scheme file in ``root``'s ``SCHEMES_DIR``, by the ``id`` each file holds.

    A file that is not an ``ora.context-scheme.v1`` or ``v2`` object with a string id, or a second file with an id
    already seen, is refused: discovery never picks one of two definitions.
    """

    found: dict[str, str] = {}
    for file in sorted((Path(root) / SCHEMES_DIR).glob("*.json")):
        path = f"{SCHEMES_DIR}/{file.name}"
        try:
            value = json.loads(file.read_bytes().decode("utf-8"))
        except ValueError as error:
            raise ValueError(f"{path} is not a JSON context scheme: {error}") from None
        scheme_id = value.get("id") if isinstance(value, dict) and value.get("schema") in SCHEMAS else None
        if not isinstance(scheme_id, str) or not scheme_id:
            raise ValueError(f"{path} is not an {SCHEMA} or {SCHEMA_V2} scheme with a string id")
        if scheme_id in found:
            raise ValueError(f"context scheme {scheme_id} is defined twice: {found[scheme_id]} and {path}")
        found[scheme_id] = path
    return found


def load_context_scheme(root: Path, scheme_id: str) -> dict[str, Any]:
    """The scheme's policy fields and its ``{id, path, sha256}`` record."""

    schemes = context_schemes(root)
    if not isinstance(scheme_id, str) or scheme_id not in schemes:
        raise ValueError(f"policy.context_scheme must be one of {sorted(schemes)}")
    path = schemes[scheme_id]
    asset = locked_asset(root, path, label=f"context scheme {scheme_id}")
    value = json.loads(asset.data.decode("utf-8"))
    version = value.get("schema") if isinstance(value, dict) else None
    allowed = set(CONTEXT_FIELDS) - (set(STAGE_FIELDS) if version == SCHEMA_V2 else {"context_stages"})
    if (
        version not in SCHEMAS or value.get("id") != scheme_id
        or not isinstance(value.get("policy"), dict) or not value["policy"]
        or set(value["policy"]) - allowed
    ):
        raise ValueError(f"{path} is not an {SCHEMA} or {SCHEMA_V2} scheme of {scheme_id}")
    policy = dict(value["policy"])
    if version == SCHEMA_V2:
        policy.update(_resolve_stages(root, value, path))
    elif any(key in value for key in V2_STAGE_KEYS):
        raise ValueError(f"{path}: an {SCHEMA} scheme carries no stages")
    prompt = value.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, str) or "/" in prompt or not prompt.endswith(".md"):
            raise ValueError(f"{path}: prompt must be a Markdown file beside the scheme")
        prompt = (Path(path).parent / prompt).as_posix()
        if not locked_asset(root, prompt, label=f"context scheme {scheme_id} prompt").data.strip():
            raise ValueError(f"{prompt} is empty")
    return {"policy": policy, "prompt": prompt,
            "record": {"id": scheme_id, "path": path, "sha256": asset.sha256}}


def _resolve_stages(root: Path, value: dict[str, Any], path: str) -> dict[str, Any]:
    """A v2 scheme's management object and stage list, in the plan's field order."""

    from .schemas.context_stages import management_equivalent, resolve_stage_list

    stage_list = resolve_stage_list(root, value, label=path)
    return {"context_management": management_equivalent(stage_list, label=path), "context_stages": stage_list}
