"""External context packs (``policy.context_pack``): loading, profile, composition, finish semantics.

A pack is read only through its manifest and refused on any hash mismatch; the
delivery profile comes from the manifest or, for a manifest without one, from
the RunSpec, never both. The plan binds the pack (manifest and file hashes,
profile, the composed system text and task template, the overlaid tools'
hash), withholds the profile's tools and stays on the Pi core; a default plan
carries nothing. The finish switch admits a success claim without check_task
and names no check_task in refusals. No provider, simulator or Node process.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ora.agent.composition.prompt import context_pack as cp
from ora.agent.contracts import PolicySessionError
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from tests.support.orchestrator.run_specs import native_spec

ROOT = Path(__file__).resolve().parents[4]
SYNTHETIC = ROOT / "tests/data/context_pack/synthetic-pack"
SYNTHETIC_MANIFEST = hashlib.sha256((SYNTHETIC / "manifest.json").read_bytes()).hexdigest()
CARD = ROOT / "ora/policy/embodiments/franka-panda/manual-public-geometry-v3.md"
SECTIONS = ["Identity and topology", "Controlled points", "Frames and units", "End-effector axes",
            "Gripper semantics", "Sensors and cameras"]


def write_pack(directory: Path, files: dict[str, str], **manifest_fields) -> str:
    """A pack with ``files`` and its manifest; returns the manifest's sha256."""

    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
    manifest = {"name": directory.name, "files": {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}, **manifest_fields}
    data = json.dumps(manifest, indent=1).encode()
    (directory / "manifest.json").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def synthetic_profile() -> dict:
    return copy.deepcopy(json.loads((SYNTHETIC / "manifest.json").read_text())["ora_profile"])


# ---- the pack and its profile -------------------------------------------------------------------


def test_a_pack_loads_only_with_its_manifest_hash_and_unchanged_files(tmp_path: Path) -> None:
    pack = cp.load_context_pack(SYNTHETIC, SYNTHETIC_MANIFEST)
    assert pack.name == "synthetic-compact-v1" and set(pack.files) == {"CORE.md", "tool-overlay.json"}
    with pytest.raises(cp.ContextPackError, match="the RunSpec names"):
        cp.load_context_pack(SYNTHETIC, "0" * 64)
    copied = tmp_path / "copied"
    shutil.copytree(SYNTHETIC, copied)
    (copied / "CORE.md").write_text("changed\n")
    with pytest.raises(cp.ContextPackError, match="CORE.md differs"):
        cp.load_context_pack(copied, SYNTHETIC_MANIFEST)
    (copied / "CORE.md").unlink()
    with pytest.raises(cp.ContextPackError, match="missing"):
        cp.load_context_pack(copied, SYNTHETIC_MANIFEST)
    write_pack(tmp_path / "escaping", {"core.md": "x"})
    manifest = json.loads((tmp_path / "escaping/manifest.json").read_text())
    manifest["files"] = {"../core.md": manifest["files"]["core.md"]}
    data = json.dumps(manifest).encode()
    (tmp_path / "escaping/manifest.json").write_bytes(data)
    with pytest.raises(cp.ContextPackError, match="inside the pack"):
        cp.load_context_pack(tmp_path / "escaping", hashlib.sha256(data).hexdigest())
    with pytest.raises(cp.ContextPackError, match="absolute"):
        cp.load_context_pack(Path("tests/data/context_pack/synthetic-pack"), SYNTHETIC_MANIFEST)


def test_the_manifest_profile_decides_alone_and_an_older_manifest_takes_the_run_spec_profile(tmp_path: Path) -> None:
    pack = cp.load_context_pack(SYNTHETIC, SYNTHETIC_MANIFEST)
    profile, source = cp.resolve_profile(pack, None)
    assert source == "manifest" and profile["compact_view"]["recent_turns"] == 1
    with pytest.raises(cp.ContextPackError, match="cannot override"):
        cp.resolve_profile(pack, synthetic_profile())
    files = {name: (SYNTHETIC / name).read_text() for name in ("CORE.md", "tool-overlay.json")}
    older = cp.load_context_pack(tmp_path / "older", write_pack(tmp_path / "older", files))
    with pytest.raises(cp.ContextPackError, match="must state the delivery profile"):
        cp.resolve_profile(older, None)
    profile, source = cp.resolve_profile(older, synthetic_profile())
    assert source == "run_spec" and profile == synthetic_profile()
    unlisted = synthetic_profile()
    unlisted["system"]["core_file"] = "OTHER.md"
    with pytest.raises(cp.ContextPackError, match="not in the manifest"):
        cp.resolve_profile(older, unlisted)


@pytest.mark.parametrize(("edit", "message"), [
    (lambda view: view.pop("image_window"), "needs image_window"),
    (lambda view: view.update(schema="ora.compact-view.v2"), "schema must be"),
    (lambda view: view["fold"].update(latest_result_tools=["grasp_plan"]), "reserved"),
    (lambda view: view["texts"].update(image_placeholder="[image omitted]"), "{label} exactly once"),
    (lambda view: view.update(recent_turns=-1), "integer >= 0"),
    (lambda view: view["fold"].update(line_format="turn_summary_v9"), "line_format"),
    (lambda view: view.update(extra=1), "unknown fields"),
])
def test_a_compact_view_states_every_parameter_and_reserves_the_dependency_slot(edit, message) -> None:
    view = synthetic_profile()["compact_view"]
    assert cp.validate_compact_view(view) == view
    view["fold"]["arguments_cap_chars"] = None  # null keeps the arguments whole
    cp.validate_compact_view(view)
    edit(view)
    with pytest.raises(cp.ContextPackError, match=message):
        cp.validate_compact_view(view)


# ---- texts and tools -----------------------------------------------------------------------------


def test_the_system_text_is_the_core_and_the_cards_named_sections_verbatim() -> None:
    card = CARD.read_text()
    text, kept = cp.compact_system_text("# Core\n\nBody.\n", card, list(reversed(SECTIONS)))
    title = card.splitlines()[0].lstrip("# ").strip()
    assert text.startswith(f"# Core\n\nBody.\n\n# Embodiment: {title}\n\n## Identity and topology\n")
    assert kept == SECTIONS  # the card's order, not the profile's
    for section in card.split("\n## ")[1:]:
        name = section.split("\n")[0].strip()
        assert (("## " + section.strip()) in text) == (name in SECTIONS)
    with pytest.raises(cp.ContextPackError, match="no compiled embodiment card"):
        cp.embodiment_facts("# Some manual\n\n## Identity and topology\n\nx", SECTIONS)
    with pytest.raises(cp.ContextPackError, match="none of the sections"):
        cp.embodiment_facts(card, ["Nothing like this"])


def test_the_bare_episode_task_keeps_the_task_and_the_contract() -> None:
    contract = "Public completion contract for this task:\n\n- Open the lid.\n"
    template = cp.episode_task_template([contract])
    assert template == "{{TASK}}\n\nPublic completion contract for this task:\n\n- Open the lid."
    assert cp.render_episode_task(template, "  Open the laptop. ") == (
        "Open the laptop.\n\nPublic completion contract for this task:\n\n- Open the lid.")
    assert cp.render_episode_task(cp.episode_task_template([]), "Open the laptop.") == "Open the laptop."


def test_the_tool_overlay_replaces_descriptions_by_hash_and_refuses_anything_else() -> None:
    pack = cp.load_context_pack(SYNTHETIC, SYNTHETIC_MANIFEST)
    overlay = cp.load_tool_overlay(pack, "tool-overlay.json")
    assert "0" * 12 not in overlay  # an entry that did not pass is never used
    entries = json.loads((SYNTHETIC / "tool-overlay.json").read_text())
    originals = {}
    for entry in entries[:3]:
        original = copy.deepcopy(entry["spec"])
        original["description"] = f"Original {original['name']}"
        originals[entry["key"]] = original
    fixture = json.loads((SYNTHETIC.parent / "synthetic-replay.json").read_text())
    wire = cp.drop_tools(fixture["tools"], ["check_task"])
    delivered = cp.overlay_tools(wire, overlay)
    assert [tool["function"]["description"] for tool in delivered] == [entry["spec"]["description"] for entry in entries[:3]]
    assert [tool["function"]["parameters"]["required"] for tool in delivered] == \
        [tool["function"]["parameters"]["required"] for tool in wire]
    unknown = copy.deepcopy(wire[0])
    unknown["function"]["description"] += " Changed."
    with pytest.raises(cp.ContextPackError, match="no version of observe"):
        cp.overlay_tools([unknown], overlay)
    widened = {key: copy.deepcopy(spec) for key, spec in overlay.items()}
    widened[cp.overlay_key(wire[1]["function"])]["parameters"]["properties"]["speed"]["maximum"] = 2.0
    with pytest.raises(cp.ContextPackError, match="more than descriptions"):
        cp.overlay_tools(wire, widened)


def _png(width: int, height: int, mode: str = "RGB", fmt: str = "PNG") -> str:
    image = Image.new(mode, (width, height))
    image.putdata([((x * 3) % 256, (y * 5) % 256, 7) if mode == "RGB" else ((x * 3) % 256, (y * 5) % 256, 7, 200)
                   for y in range(height) for x in range(width)])
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    media = "png" if fmt == "PNG" else "jpeg"
    return f"data:image/{media};base64," + base64.b64encode(buffer.getvalue()).decode()


def test_images_are_downscaled_bicubic_to_the_long_edge_as_png() -> None:
    small = _png(512, 300)
    assert cp.downscale_data_url(small, 512) == small
    large = _png(1024, 800, mode="RGBA")
    delivered = cp.downscale_data_url(large, 512)
    data = base64.b64decode(delivered.partition(",")[2])
    with Image.open(io.BytesIO(data)) as image:
        assert (image.size, image.format, image.mode) == ((512, 400), "PNG", "RGB")
    with Image.open(io.BytesIO(base64.b64decode(large.partition(",")[2]))) as source:
        expected = io.BytesIO()
        source.convert("RGB").resize((512, 400), Image.BICUBIC).save(expected, "PNG")
    assert data == expected.getvalue()
    jpeg = cp.downscale_data_url(_png(700, 100, fmt="JPEG"), 512)
    assert jpeg.startswith("data:image/png;base64,")


# ---- plans ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_libero_plan() -> dict:
    return resolve_run_spec(native_spec("libero_pro"), root=ROOT)


def _covering_pack(directory: Path, plan: dict, **profile_edits) -> dict:
    """A pack whose overlay covers ``plan``'s advertised tools; the RunSpec field naming it."""

    from ora.orchestrator.spec.context_pack import wire_tool_definitions

    surface = plan["policy_composition"]["tool_surface"]
    overlay = []
    for tool in wire_tool_definitions(surface["definitions"], surface["capabilities"]):
        compact = copy.deepcopy(tool["function"])
        compact["description"] = f"Compact {compact['name']}."
        overlay.append({"key": cp.overlay_key(tool["function"]), "spec": compact})
    profile = {**synthetic_profile(), "tools": {"overlay_file": "overlay.json", "drop": ["check_task", "report_issue"]}}
    profile["system"]["embodiment_sections"] = SECTIONS
    profile.update(profile_edits)
    sha = write_pack(directory, {"CORE.md": "# Core\n\nThe pack's own rules.\n", "overlay.json": json.dumps(overlay)},
                     ora_profile=profile)
    return {"path": str(directory), "manifest_sha256": sha}


def _pack_spec(field: dict, **spec_fields) -> dict:
    spec = native_spec("libero_pro")
    spec["policy"]["context_pack"] = field
    spec["result_tier"] = "zero_shot_no_check"
    spec.update(spec_fields)
    return spec


def test_a_plan_binds_the_pack_and_a_default_plan_is_unchanged(tmp_path: Path, default_libero_plan: dict) -> None:
    assert "context_pack" not in default_libero_plan["policy"]
    assert "check_task" in default_libero_plan["policy"]["tool_allowlist"]
    field = _covering_pack(tmp_path / "pack", default_libero_plan)
    plan = resolve_run_spec(_pack_spec(field), root=ROOT)
    binding = plan["policy"]["context_pack"]
    assert binding["schema"] == cp.BINDING_SCHEMA and binding["profile_source"] == "manifest"
    assert binding["manifest"] == {"path": str(tmp_path / "pack/manifest.json"), "sha256": field["manifest_sha256"]}
    assert plan["policy"]["tool_allowlist"] == [name for name in default_libero_plan["policy"]["tool_allowlist"]
                                                if name not in {"check_task", "report_issue"}]
    card = default_libero_plan["policy_context"]["project_instructions"][-1]
    assert binding["system"]["embodiment_card"] == card
    assert binding["system"]["text"] == cp.compact_system_text(
        "# Core\n\nThe pack's own rules.", (ROOT / card["path"]).read_text(), SECTIONS)[0]
    assert binding["episode_task"]["template"] == "{{TASK}}"  # this LIBERO task has no reviewed contract
    layer = plan["policy_composition"]["tool_surface"]["derived_layers"][-1]
    assert layer["kind"] == "context_pack_tool_overlay_v1"
    assert layer["delivered_tools_sha256"] == binding["delivered_tools_sha256"]
    assert "check_task" not in layer["tools"] and "artifact_read" in layer["tools"]
    assert plan["policy_identity"]["model_visible_sha256"] != default_libero_plan["policy_identity"]["model_visible_sha256"]


def test_a_pack_plan_declares_its_tier_runs_on_pi_and_covers_every_tool(tmp_path: Path,
                                                                     default_libero_plan: dict) -> None:
    field = _covering_pack(tmp_path / "pack", default_libero_plan)
    spec = _pack_spec(field)
    spec.pop("result_tier")
    with pytest.raises(RunSpecError, match="zero_shot_no_check"):
        resolve_run_spec(spec, root=ROOT)
    spec = _pack_spec(field)
    spec["policy"]["agent_core"] = "native"
    with pytest.raises(RunSpecError, match="Pi agent core only"):
        resolve_run_spec(spec, root=ROOT)
    with pytest.raises(RunSpecError, match="no version of observe"):
        resolve_run_spec(_pack_spec({"path": str(SYNTHETIC), "manifest_sha256": SYNTHETIC_MANIFEST}), root=ROOT)
    finish = _covering_pack(tmp_path / "drops-finish", default_libero_plan,
                            tools={"overlay_file": "overlay.json", "drop": ["check_task", "finish_episode"]})
    with pytest.raises(RunSpecError, match="mutation or lifecycle tools: finish_episode"):
        resolve_run_spec(_pack_spec(finish), root=ROOT)
    with pytest.raises(RunSpecError, match="is not set"):
        resolve_run_spec(_pack_spec({"path_env": "ORA_TEST_UNSET_PACK", "manifest_sha256": SYNTHETIC_MANIFEST}),
                         root=ROOT)


def test_a_plan_names_its_pack_through_an_environment_variable(tmp_path: Path, default_libero_plan: dict,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    field = _covering_pack(tmp_path / "pack", default_libero_plan)
    monkeypatch.setenv("ORA_TEST_PACK_DIR", field.pop("path"))
    plan = resolve_run_spec(_pack_spec({"path_env": "ORA_TEST_PACK_DIR", **field}), root=ROOT)
    assert plan["policy"]["context_pack"]["path_env"] == "ORA_TEST_PACK_DIR"


def test_a_session_refuses_a_changed_pack_and_the_native_core(tmp_path: Path, default_libero_plan: dict) -> None:
    field = _covering_pack(tmp_path / "pack", default_libero_plan)
    binding = resolve_run_spec(_pack_spec(field), root=ROOT)["policy"]["context_pack"]
    assert cp.verify_context_pack_binding(binding, core="pi").name == "pack"
    with pytest.raises(PolicySessionError, match="only the Pi agent core"):
        cp.verify_context_pack_binding(binding, core="native")
    (tmp_path / "pack/CORE.md").write_text("# Changed after planning\n")
    with pytest.raises(PolicySessionError, match="context pack refused: context pack file CORE.md differs"):
        cp.verify_context_pack_binding(binding, core="pi")


def test_the_session_rpc_serves_the_bound_replacements_tools_and_images(tmp_path: Path,
                                                                       default_libero_plan: dict) -> None:
    from ora.agent.session_core.composition import assemble_native_v3_static_context
    from ora.agent.pi.context_pack_rpc import context_pack_handlers
    from ora.orchestrator.spec.context_pack import wire_tool_definitions

    plan = resolve_run_spec(_pack_spec(_covering_pack(tmp_path / "pack", default_libero_plan)), root=ROOT)
    layers = {channel: "\n\n".join((ROOT / layer["path"]).read_text().strip() for layer in plan["policy_context"][channel])
              for channel in ("harness_system", "project_instructions", "episode_task")}
    surface = plan["policy_composition"]["tool_surface"]
    planner = SimpleNamespace(
        policy=plan["policy"], bound_task_instruction=plan["task_instruction"],
        harness_system_template=layers["harness_system"], project_template=layers["project_instructions"],
        episode_prefix_template=layers["episode_task"],
        gateway=SimpleNamespace(definitions=lambda: wire_tool_definitions(surface["definitions"], surface["capabilities"])),
    )
    handlers = context_pack_handlers(planner)
    assert set(handlers) == {"context_pack.delivery", "context_pack.images"}
    delivery = handlers["context_pack.delivery"]({})
    static = assemble_native_v3_static_context(
        harness_system=layers["harness_system"], project_instructions=layers["project_instructions"],
        episode_task_template=layers["episode_task"], task_instruction=plan["task_instruction"], task_id="t")
    assert [(item["role"], item["content_sha256"]) for item in delivery["messages"]] == [
        (message["role"], cp.sha256_text(message["content"])) for message in static["messages"]]
    assert [item["content"] for item in delivery["messages"]] == [
        plan["policy"]["context_pack"]["system"]["text"], None, plan["task_instruction"]]
    assert cp.canonical_sha256(delivery["tools"]) == plan["policy"]["context_pack"]["delivered_tools_sha256"]
    large, small = _png(1024, 800), _png(32, 16)
    assert handlers["context_pack.images"]({"images": [large, small]}) == {
        "images": [cp.downscale_data_url(large, 64), small]}
    planner.policy = {key: value for key, value in plan["policy"].items() if key != "context_pack"}
    assert context_pack_handlers(planner) == {}


# ---- finish semantics -----------------------------------------------------------------------------


class _Surface:
    def __init__(self, *capabilities: str) -> None:
        self.capabilities = tuple(capabilities)

    def supports(self, name: str) -> bool:
        return name in self.capabilities


def _host(tmp_path: Path, surface: _Surface, record: dict | None) -> SimpleNamespace:
    from ora.agent.toolkit.host.finish import task_check as finish_task_check

    if record is not None:
        (tmp_path / finish_task_check.RECORD).write_text(json.dumps(record))
    return SimpleNamespace(surface=surface, control_root=tmp_path)


def test_the_official_verdict_switch_admits_success_without_check_task(tmp_path: Path) -> None:
    from ora.agent.toolkit.host.finish import task_check as finish_task_check
    from ora.agent.error_classes import ToolError

    checked = _Surface(finish_task_check.CAPABILITY)
    switched = _Surface(finish_task_check.CAPABILITY, cp.OFFICIAL_VERDICT_FINISH)
    assert finish_task_check.OFFICIAL_VERDICT == cp.OFFICIAL_VERDICT_FINISH
    pending = _host(tmp_path, checked, {"task_success": False, "terminal": False})
    with pytest.raises(ToolError) as refused:
        finish_task_check.refuse_unconfirmed(pending, "success")
    assert "check_task" in str(refused.value)
    finish_task_check.refuse_unconfirmed(_host(tmp_path, switched, None), "success")
    latched = _host(tmp_path, switched, {"task_success": True, "terminal": True})
    finish_task_check.refuse_latched(latched, {"outcome": "give_up"})  # the hidden verdict stays hidden
    with pytest.raises(ToolError):
        finish_task_check.refuse_latched(_host(tmp_path, checked, None), {"outcome": "give_up"})


def test_a_refused_success_reason_names_no_check_task_under_the_switch() -> None:
    from ora.agent.toolkit.host.finish import outcomes as finish_outcomes

    plain = finish_outcomes._refuse_reason("success", "done", "too_short")
    switched = finish_outcomes._refuse_reason("success", "done", "too_short", without_check_task=True)
    assert "check_task" in plain.constraints["evidence_terms"]
    assert "check_task" not in json.dumps(switched.constraints) and "check_task" not in str(switched)
    assert switched.constraints["evidence_terms"] == [
        term for term in plain.constraints["evidence_terms"] if term != "check_task"]


def test_only_an_official_verdict_pack_binds_the_finish_capability() -> None:
    @dataclasses.dataclass(frozen=True)
    class Frozen:
        capabilities: tuple

    frozen = Frozen(capabilities=("faithful_check_task",))
    assert cp.bind_finish_semantics(frozen, None) is frozen
    assert cp.bind_finish_semantics(frozen, {"profile": {"finish": {"success": "task_check"}}}) is frozen
    bound = cp.bind_finish_semantics(frozen, {"profile": {"finish": {"success": "official_verdict"}}})
    assert bound.capabilities == ("faithful_check_task", cp.OFFICIAL_VERDICT_FINISH)
    assert cp.bind_finish_semantics(bound, {"profile": {"finish": {"success": "official_verdict"}}}) is bound


# ---- context management switches ------------------------------------------------------------------


def test_the_context_management_switches_are_booleans_and_default_on() -> None:
    pack = cp.load_context_pack(SYNTHETIC, SYNTHETIC_MANIFEST)
    profile = synthetic_profile()
    assert cp.context_management_off(profile) == []
    profile["context_management"] = {"pressure_externalization": False, "rollout_compaction": True}
    assert cp.validate_profile(profile, pack, label="p")["context_management"] == profile["context_management"]
    assert cp.context_management_off(profile) == ["pressure_externalization"]
    for wrong, message in (({"rollout_compaction": "off"}, "true or false"), ({"l4": False}, "unknown fields")):
        profile["context_management"] = wrong
        with pytest.raises(cp.ContextPackError, match=message):
            cp.validate_profile(profile, pack, label="p")


def test_switched_off_context_management_is_plan_identity_and_needs_lossless_retention(
        tmp_path: Path, default_libero_plan: dict) -> None:
    from ora.agent.composition import resolve_tool_surface
    from ora.orchestrator.spec.context_pack import resolve_context_pack

    on = resolve_run_spec(_pack_spec(_covering_pack(tmp_path / "on", default_libero_plan)), root=ROOT)
    off_field = _covering_pack(tmp_path / "off", default_libero_plan, context_management={
        "pressure_externalization": False, "rollout_compaction": False})
    off = resolve_run_spec(_pack_spec(off_field), root=ROOT)
    assert off["policy"]["context_pack"]["profile"]["context_management"] == {
        "pressure_externalization": False, "rollout_compaction": False}
    assert off["policy"]["context_management"] == on["policy"]["context_management"]  # the plan's layers, unchanged
    assert off["policy_identity"]["model_visible_sha256"] != on["policy_identity"]["model_visible_sha256"]
    surface = resolve_tool_surface(default_libero_plan["policy"]["tool_surface"]["profile"])
    policy = {"agent_core": "pi", "context_retention": "rewrite_compact_v1",
              "context_composition": "native_slots_v4", "context_pack": off_field,
              "tool_allowlist": [item["name"] for item in surface.definitions]}
    with pytest.raises(RunSpecError, match="act on action_step_lossless_v4 only"):
        resolve_context_pack(policy, surface)
