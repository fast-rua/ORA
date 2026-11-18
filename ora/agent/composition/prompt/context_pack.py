"""External context packs: versioned delivery assets bound by their manifest hash.

A context pack is a directory owned outside this repository (for example by a
research project that trains a model on a compact context). Its
``manifest.json`` lists every file with its sha256; ORA reads the pack only
through that manifest and refuses it when the manifest or any listed file
differs from the hash the RunSpec names. ORA owns the mechanisms; the pack
owns the words and the parameters:

- a compact system message: the pack's core text plus named sections of the
  plan's compiled embodiment card (it replaces the harness system message and
  the project-instructions message);
- the episode task without the template prose (the task sentence and the
  public completion contract);
- a tool-description overlay looked up by the original function's hash, and
  tools withheld from the policy;
- the finish semantics (``official_verdict``: a success claim is judged by
  the environment's official boundary, not refused for want of check_task);
- a compact history view (``ora.compact-view.v1``), applied by the Pi agent
  core (``ora-pi/src/context/compact_view.ts``);
- switches that turn off lossless retention's layers acting on token usage
  (L4 pressure externalization, L5/L6 rollout compaction).

The delivery profile (schema ``ora.context-pack-profile.v1``) is read from the
manifest's ``ora_profile`` object. A manifest without one (an older pack) takes
the profile from the RunSpec instead; a RunSpec may never override a profile
the manifest states. See ``docs/reference/context-packs.md``.

This module holds the pure parts shared by plan resolution
(``ora.orchestrator.spec_context_pack``), the Pi core's Python half
(``ora.agent.pi.context_pack_rpc``) and the tests' replay of recorded runs.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ora.agent.contracts import PolicySessionError

MANIFEST_NAME = "manifest.json"
#: The manifest key that holds the delivery profile.
MANIFEST_PROFILE_KEY = "ora_profile"
PROFILE_SCHEMA = "ora.context-pack-profile.v1"
COMPACT_VIEW_SCHEMA = "ora.compact-view.v1"
BINDING_SCHEMA = "ora.context-pack-binding.v1"
#: ``episode_task`` modes: the bound template unchanged, or the task sentence
#: followed by the episode's public completion-contract layers.
EPISODE_TASK_MODES = ("template", "task_and_contract_v1")
#: ``finish.success`` modes. ``task_check`` keeps the surface's own rule;
#: ``official_verdict`` admits a success claim and leaves the verdict to the
#: environment's official boundary.
FINISH_SUCCESS_MODES = ("task_check", "official_verdict")
#: The ToolSurface capability ``official_verdict`` binds into the runtime surface.
OFFICIAL_VERDICT_FINISH = "finish_success_official_verdict_v1"
#: Fold-line grammars of the compact view (``turn_summary_v1``: see compact_view.ts).
FOLD_LINE_FORMATS = ("turn_summary_v1",)
#: ``context_management`` switches, each on unless the profile sets it false: lossless
#: retention's L4 pressure externalization and its L5/L6 rollout compaction.
CONTEXT_MANAGEMENT_SWITCHES = ("pressure_externalization", "rollout_compaction")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_OVERLAY_KEY = re.compile(r"[0-9a-f]{12}")


class ContextPackError(ValueError):
    """A context pack, its profile or its use in a plan is invalid."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8"))


# ---- the pack -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextPack:
    """A verified pack: every listed file matched its manifest hash when loaded."""

    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    files: Mapping[str, str]

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    def read_bytes(self, name: str) -> bytes:
        """A listed file's bytes, checked against the manifest again."""

        if name not in self.files:
            raise ContextPackError(f"context pack {self.name} does not list {name!r}")
        data = (self.root / name).read_bytes()
        if sha256_bytes(data) != self.files[name]:
            raise ContextPackError(f"context pack file {name} differs from its manifest sha256")
        return data

    def read_text(self, name: str) -> str:
        return self.read_bytes(name).decode("utf-8")


def _relative_name(name: Any) -> str:
    if not isinstance(name, str) or not name or name.startswith("/") or "\\" in name:
        raise ContextPackError(f"context pack file names must be relative paths: {name!r}")
    parts = PurePosixPath(name).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ContextPackError(f"context pack file names must stay inside the pack: {name!r}")
    return name


def load_context_pack(root: Path | str, manifest_sha256: str) -> ContextPack:
    """Load ``root/manifest.json`` and every file it lists; refuse any hash mismatch."""

    if not isinstance(manifest_sha256, str) or not _SHA256.fullmatch(manifest_sha256):
        raise ContextPackError("a context pack's manifest_sha256 must be 64 lowercase hex digits")
    root = Path(root)
    if not root.is_absolute():
        raise ContextPackError(f"a context pack path must be absolute: {root}")
    manifest_path = root / MANIFEST_NAME
    try:
        data = manifest_path.read_bytes()
    except OSError as exc:
        raise ContextPackError(f"context pack manifest is unreadable: {manifest_path} ({exc.strerror})") from None
    actual = sha256_bytes(data)
    if actual != manifest_sha256:
        raise ContextPackError(
            f"context pack manifest {manifest_path} has sha256 {actual}, the RunSpec names {manifest_sha256}"
        )
    try:
        manifest = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextPackError(f"context pack manifest is not JSON: {exc}") from None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("name"), str) or not manifest["name"]:
        raise ContextPackError("a context pack manifest is an object with a non-empty name")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ContextPackError("a context pack manifest lists its files with their sha256")
    verified: dict[str, str] = {}
    resolved_root = root.resolve()
    for name, digest in files.items():
        name = _relative_name(name)
        if name == MANIFEST_NAME:
            raise ContextPackError("a context pack manifest does not list itself")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ContextPackError(f"context pack file {name} needs a lowercase sha256")
        path = (root / name).resolve()
        if not path.is_relative_to(resolved_root) or not path.is_file():
            raise ContextPackError(f"context pack file {name} is missing from {root}")
        if sha256_bytes(path.read_bytes()) != digest:
            raise ContextPackError(f"context pack file {name} differs from its manifest sha256")
        verified[name] = digest
    return ContextPack(root=root, manifest=manifest, manifest_sha256=actual, files=verified)


def load_bound_pack(binding: Mapping[str, Any]) -> ContextPack:
    """The pack a plan's binding names, every file checked again against the plan."""

    if not isinstance(binding, Mapping) or binding.get("schema") != BINDING_SCHEMA:
        raise ContextPackError(f"policy.context_pack is not an {BINDING_SCHEMA} binding")
    manifest = binding["manifest"]
    pack = load_context_pack(Path(manifest["path"]).parent, manifest["sha256"])
    if dict(pack.files) != dict(binding["files"]):
        raise ContextPackError(f"context pack {pack.name} lists other files than the plan bound")
    return pack


def verify_context_pack_binding(binding: Mapping[str, Any], *, core: str) -> ContextPack:
    """Refuse to start a session whose pack changed since planning, or that names another core."""

    if core != "pi":
        raise PolicySessionError(
            f"this plan binds policy.context_pack, which only the Pi agent core delivers, not {core!r}"
        )
    try:
        return load_bound_pack(binding)
    except ContextPackError as exc:
        raise PolicySessionError(f"context pack refused: {exc}") from None


# ---- the delivery profile ------------------------------------------------------------------


def _object(value: Any, label: str, allowed: set[str], required: set[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContextPackError(f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ContextPackError(f"{label} has unknown fields: {', '.join(unknown)}")
    missing = sorted(required - set(value))
    if missing:
        raise ContextPackError(f"{label} needs {', '.join(missing)}")
    return dict(value)


def _count(value: Any, label: str, *, minimum: int, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContextPackError(f"{label} must be an integer >= {minimum}" + (" or null" if nullable else ""))
    return value


def _names(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value) \
            or len(set(value)) != len(value):
        raise ContextPackError(f"{label} must be a list of unique non-empty strings")
    return list(value)


def _template(value: Any, label: str, placeholder: str | None) -> str:
    if not isinstance(value, str) or not value:
        raise ContextPackError(f"{label} must be a non-empty string")
    if placeholder is not None and value.count(placeholder) != 1:
        raise ContextPackError(f"{label} must contain {placeholder} exactly once")
    return value


def validate_compact_view(value: Any) -> dict[str, Any]:
    """An ``ora.compact-view.v1`` object, validated and copied."""

    view = _object(value, "compact_view", {
        "schema", "recent_turns", "tool_text_cap_chars", "image_window", "image_long_edge_px", "fold", "texts",
    }, {"schema", "recent_turns", "tool_text_cap_chars", "image_window", "image_long_edge_px", "fold", "texts"})
    if view["schema"] != COMPACT_VIEW_SCHEMA:
        raise ContextPackError(f"compact_view.schema must be {COMPACT_VIEW_SCHEMA}")
    _count(view["recent_turns"], "compact_view.recent_turns", minimum=0)
    _count(view["tool_text_cap_chars"], "compact_view.tool_text_cap_chars", minimum=1, nullable=True)
    _count(view["image_window"], "compact_view.image_window", minimum=0, nullable=True)
    _count(view["image_long_edge_px"], "compact_view.image_long_edge_px", minimum=1, nullable=True)
    fold = _object(view["fold"], "compact_view.fold", {
        "line_format", "rationale_cap_chars", "arguments_cap_chars", "latest_result_tools",
    }, {"line_format", "rationale_cap_chars", "arguments_cap_chars"})
    if fold["line_format"] not in FOLD_LINE_FORMATS:
        raise ContextPackError(f"compact_view.fold.line_format must be one of {', '.join(FOLD_LINE_FORMATS)}")
    _count(fold["rationale_cap_chars"], "compact_view.fold.rationale_cap_chars", minimum=0, nullable=True)
    _count(fold["arguments_cap_chars"], "compact_view.fold.arguments_cap_chars", minimum=0, nullable=True)
    latest = _names(fold.get("latest_result_tools", []), "compact_view.fold.latest_result_tools")
    if latest:
        # Reserved for the dependency rule a later view version defines; refused, never ignored.
        raise ContextPackError(
            "compact_view.fold.latest_result_tools is reserved: ora.compact-view.v1 keeps no earlier "
            "result verbatim, so the list must be empty"
        )
    fold["latest_result_tools"] = latest
    texts = _object(view["texts"], "compact_view.texts", {
        "earlier_steps_header", "image_placeholder", "tool_text_omission",
    }, {"earlier_steps_header", "image_placeholder", "tool_text_omission"})
    _template(texts["earlier_steps_header"], "compact_view.texts.earlier_steps_header", None)
    _template(texts["image_placeholder"], "compact_view.texts.image_placeholder", "{label}")
    _template(texts["tool_text_omission"], "compact_view.texts.tool_text_omission", "{omitted}")
    view["fold"], view["texts"] = fold, texts
    return copy.deepcopy(view)


def validate_profile(value: Any, pack: ContextPack, *, label: str) -> dict[str, Any]:
    """An ``ora.context-pack-profile.v1`` object whose files the pack lists."""

    profile = _object(value, label, {
        "schema", "system", "episode_task", "tools", "finish", "compact_view", "context_management",
    }, {"schema"})
    if profile["schema"] != PROFILE_SCHEMA:
        raise ContextPackError(f"{label}.schema must be {PROFILE_SCHEMA}")
    if "system" in profile:
        system = _object(profile["system"], f"{label}.system", {"core_file", "embodiment_sections"},
                         {"core_file", "embodiment_sections"})
        if _relative_name(system["core_file"]) not in pack.files:
            raise ContextPackError(f"{label}.system.core_file {system['core_file']!r} is not in the manifest")
        if not _names(system["embodiment_sections"], f"{label}.system.embodiment_sections"):
            raise ContextPackError(f"{label}.system.embodiment_sections names at least one section")
        if not pack.read_text(system["core_file"]).strip():
            raise ContextPackError(f"{label}.system.core_file is empty")
        profile["system"] = system
    if "episode_task" in profile and profile["episode_task"] not in EPISODE_TASK_MODES:
        raise ContextPackError(f"{label}.episode_task must be one of {', '.join(EPISODE_TASK_MODES)}")
    if "tools" in profile:
        tools = _object(profile["tools"], f"{label}.tools", {"overlay_file", "drop"})
        if "overlay_file" in tools:
            if _relative_name(tools["overlay_file"]) not in pack.files:
                raise ContextPackError(f"{label}.tools.overlay_file {tools['overlay_file']!r} is not in the manifest")
            load_tool_overlay(pack, tools["overlay_file"])
        tools["drop"] = _names(tools.get("drop", []), f"{label}.tools.drop")
        profile["tools"] = tools
    if "finish" in profile:
        finish = _object(profile["finish"], f"{label}.finish", {"success"}, {"success"})
        if finish["success"] not in FINISH_SUCCESS_MODES:
            raise ContextPackError(f"{label}.finish.success must be one of {', '.join(FINISH_SUCCESS_MODES)}")
        profile["finish"] = finish
    if "compact_view" in profile:
        profile["compact_view"] = validate_compact_view(profile["compact_view"])
    if "context_management" in profile:
        management = _object(profile["context_management"], f"{label}.context_management",
                             set(CONTEXT_MANAGEMENT_SWITCHES))
        for key, item in management.items():
            if not isinstance(item, bool):
                raise ContextPackError(f"{label}.context_management.{key} must be true or false")
        profile["context_management"] = management
    return copy.deepcopy(profile)


def resolve_profile(pack: ContextPack, run_spec_profile: Any) -> tuple[dict[str, Any], str]:
    """The pack's delivery profile and its source (``manifest`` or ``run_spec``).

    A manifest that states ``ora_profile`` decides alone; a RunSpec profile
    beside it is refused rather than merged, so one manifest hash names the
    whole delivery.
    """

    stated = pack.manifest.get(MANIFEST_PROFILE_KEY)
    if stated is not None:
        if run_spec_profile is not None:
            raise ContextPackError(
                f"context pack {pack.name} states its own {MANIFEST_PROFILE_KEY}; "
                "policy.context_pack.profile cannot override it"
            )
        return validate_profile(stated, pack, label=f"manifest {MANIFEST_PROFILE_KEY}"), "manifest"
    if run_spec_profile is None:
        raise ContextPackError(
            f"context pack {pack.name} has no {MANIFEST_PROFILE_KEY}; "
            "policy.context_pack.profile must state the delivery profile"
        )
    return validate_profile(run_spec_profile, pack, label="policy.context_pack.profile"), "run_spec"


def dropped_tools(profile: Mapping[str, Any]) -> list[str]:
    return list((profile.get("tools") or {}).get("drop", []))


def context_management_off(profile: Mapping[str, Any]) -> list[str]:
    """The ``context_management`` switches the profile turns off."""

    switches = profile.get("context_management") or {}
    return [name for name in CONTEXT_MANAGEMENT_SWITCHES if switches.get(name) is False]


def finish_success(profile: Mapping[str, Any]) -> str:
    return str((profile.get("finish") or {}).get("success", "task_check"))


def bind_finish_semantics(surface: Any, binding: Mapping[str, Any] | None) -> Any:
    """The runtime ToolSurface with ``OFFICIAL_VERDICT_FINISH`` when the plan's pack asks for it."""

    if not binding or finish_success(binding["profile"]) != "official_verdict" \
            or OFFICIAL_VERDICT_FINISH in surface.capabilities:
        return surface
    from dataclasses import replace

    return replace(surface, capabilities=(*surface.capabilities, OFFICIAL_VERDICT_FINISH))


# ---- prompt texts ----------------------------------------------------------------------------


def embodiment_facts(text: str, sections: Sequence[str]) -> tuple[str, list[str]]:
    """The named ``##`` sections of the first compiled embodiment card in ``text``.

    The card is the ``# <Name> compiled embodiment vN`` document; the result is
    ``# Embodiment: <card title>``, a blank line, and the kept sections
    verbatim, in the card's order, separated by blank lines. Returns the text
    and the kept section names; an absent card or no kept section is refused.
    """

    start = text.find("compiled embodiment")
    if start < 0:
        raise ContextPackError("the embodiment layer holds no compiled embodiment card")
    start = text.rfind("\n#", 0, start) + 1
    manual = text[start:]
    following = re.search(r"\n# ", manual[1:])
    manual = manual[:following.start() + 1] if following else manual
    title = manual.splitlines()[0].lstrip("# ").strip()
    wanted = set(sections)
    kept, names = [], []
    for section in re.split(r"\n(?=## )", manual)[1:]:
        name = section.lstrip("# ").split("\n")[0].strip()
        if name in wanted:
            kept.append(section.strip())
            names.append(name)
    if not kept:
        raise ContextPackError(f"the compiled embodiment card {title!r} has none of the sections {list(sections)}")
    return f"# Embodiment: {title}\n\n" + "\n\n".join(kept), names


def compact_system_text(core_text: str, card_text: str, sections: Sequence[str]) -> tuple[str, list[str]]:
    """The pack's core text, a blank line, then the card's facts; and the kept section names."""

    facts, names = embodiment_facts(card_text, sections)
    return core_text.strip() + "\n\n" + facts, names


def episode_task_template(contract_texts: Sequence[str]) -> str:
    """``task_and_contract_v1``: ``{{TASK}}`` then each contract layer, as the template channel joins them."""

    return "\n\n".join(["{{TASK}}", *(text.strip() for text in contract_texts)])


def render_episode_task(template: str, task_instruction: str) -> str:
    """The bound template with the task, exactly as the default composition renders it."""

    return template.strip().replace("{{TASK}}", task_instruction.strip())


# ---- tools ---------------------------------------------------------------------------------------


def overlay_key(function_spec: Mapping[str, Any]) -> str:
    """The lookup key of one original function spec: ``sha256(json.dumps(spec, sort_keys=True))[:12]``."""

    return hashlib.sha256(json.dumps(function_spec, sort_keys=True).encode()).hexdigest()[:12]


def load_tool_overlay(pack: ContextPack, name: str) -> dict[str, dict[str, Any]]:
    """The overlay file: a list of ``{key, spec}`` entries; an entry with ``passed: false`` is not used."""

    try:
        entries = json.loads(pack.read_text(name))
    except json.JSONDecodeError as exc:
        raise ContextPackError(f"tool overlay {name} is not JSON: {exc}") from None
    if not isinstance(entries, list):
        raise ContextPackError(f"tool overlay {name} must be a list of entries")
    overlay: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("spec"), Mapping) \
                or not isinstance(entry.get("key"), str) or not _OVERLAY_KEY.fullmatch(entry["key"]):
            raise ContextPackError(f"tool overlay {name} entry {index} needs a 12-hex key and a spec object")
        if entry.get("passed", True) is False:
            continue
        if entry["key"] in overlay and overlay[entry["key"]] != entry["spec"]:
            raise ContextPackError(f"tool overlay {name} has two specs for key {entry['key']}")
        overlay[entry["key"]] = copy.deepcopy(dict(entry["spec"]))
    return overlay


def _without_descriptions(value: Any) -> Any:
    """``value`` with every string ``description`` field removed, at any depth."""

    if isinstance(value, Mapping):
        return {key: _without_descriptions(item) for key, item in value.items()
                if not (key == "description" and isinstance(item, str))}
    if isinstance(value, list):
        return [_without_descriptions(item) for item in value]
    return value


def overlay_tools(tools: Sequence[Mapping[str, Any]], overlay: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each wire tool with its function replaced by the overlay's; a tool without one is refused.

    An overlay may change description strings only: names, types, enums,
    required fields, ranges and defaults stay the ToolSurface's, so the
    runtime's argument validation is unchanged.
    """

    delivered = []
    for tool in tools:
        function = tool.get("function", tool)
        key = overlay_key(function)
        replacement = overlay.get(key)
        if replacement is None:
            raise ContextPackError(
                f"the context pack's tool overlay has no version of {function.get('name')} (key {key}); "
                "the pack must be regenerated for this ToolSurface"
            )
        if _without_descriptions(replacement) != _without_descriptions(function):
            raise ContextPackError(
                f"the tool overlay for {function.get('name')} (key {key}) changes more than descriptions"
            )
        delivered.append({**tool, "function": copy.deepcopy(dict(replacement))} if "function" in tool
                         else copy.deepcopy(dict(replacement)))
    return delivered


def drop_tools(tools: Sequence[Mapping[str, Any]], names: Sequence[str]) -> list[dict[str, Any]]:
    withheld = set(names)
    return [copy.deepcopy(dict(tool)) for tool in tools
            if tool.get("function", tool).get("name") not in withheld]


# ---- images -------------------------------------------------------------------------------------


def downscale_data_url(url: str, long_edge_px: int) -> str:
    """An image data URL whose long side exceeds ``long_edge_px``, resized to it (PIL bicubic, PNG).

    The scale is ``long_edge_px / max(width, height)`` and each side
    ``round(side * scale)``; the image is converted to RGB and written as PNG
    with Pillow's defaults. A smaller image is returned unchanged, bytes and
    media type included.
    """

    from PIL import Image

    header, separator, payload = url.partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        raise ContextPackError("compact view images must be base64 data URLs")
    source = base64.b64decode(payload)
    with Image.open(io.BytesIO(source)) as image:
        width, height = image.size
        if max(width, height) <= long_edge_px:
            return url
        scale = long_edge_px / max(width, height)
        resized = image.convert("RGB").resize((round(width * scale), round(height * scale)), Image.BICUBIC)
        buffer = io.BytesIO()
        resized.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


__all__ = [
    "BINDING_SCHEMA", "COMPACT_VIEW_SCHEMA", "CONTEXT_MANAGEMENT_SWITCHES", "bind_finish_semantics",
    "context_management_off", "ContextPack", "ContextPackError", "EPISODE_TASK_MODES",
    "FINISH_SUCCESS_MODES", "FOLD_LINE_FORMATS", "MANIFEST_NAME", "MANIFEST_PROFILE_KEY",
    "OFFICIAL_VERDICT_FINISH", "PROFILE_SCHEMA", "canonical_sha256", "compact_system_text",
    "downscale_data_url", "drop_tools", "dropped_tools", "embodiment_facts", "episode_task_template",
    "finish_success", "load_bound_pack", "load_context_pack", "load_tool_overlay", "overlay_key", "overlay_tools",
    "render_episode_task", "resolve_profile", "sha256_text", "validate_compact_view", "validate_profile",
    "verify_context_pack_binding",
]
