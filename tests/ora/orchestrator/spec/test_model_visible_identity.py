"""RunPlan policy identity: the policy.lock digest and the model-visible digest.

``model_visible_sha256`` must stay fixed when an asset edit only moves a pin,
a provenance note or a file path, and must move when a tool description, a
prompt layer or an embodiment card -- something the model reads -- changes.
The edits run on a scratch copy of ``ora/`` so the checkout is
never touched; ToolSurface manifests are read from the copy as well. Assets
name each other by path, so an edit is followed by regenerating the copy's
``policy.lock``, as it is in the repository.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ora.agent.composition.embodiment import geometry as embodiment_geometry
from ora.agent.composition.surface import profile_catalog, tool_surface
from ora.agent.composition.embodiment import load_embodiment_descriptor, render_embodiment_card
from ora.orchestrator.evaluation import model_visible
from ora.orchestrator.spec import policy_lock
from ora.orchestrator.spec import source as spec_source
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec, validate_run_plan
from ora.orchestrator.spec.values import json_sha256
from tests.support.orchestrator.run_specs import native_spec

ROOT = Path(__file__).resolve().parents[4]
HEX = re.compile(r"^[0-9a-f]{64}$")
# The surface and core layer the default (direct_v4, integration 0059 unit A0; its LIBERO surface
# since integration 0084) binds on LIBERO.
LIBERO_MANIFEST = "ora/policy/tool_surfaces/explicit_motion_v3/libero_placement_direct_overhead_v1-manifest.json"
CORE_LAYER = "ora/policy/core/evidence-action-loop-v8.md"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Fixture",
         "-c", "user.email=fixture@example.invalid", *args],
        check=True, capture_output=True, text=True,
    ).stdout


@pytest.fixture(scope="module")
def scratch_tree(tmp_path_factory) -> Path:
    """A committed copy of every tracked file under ora/ and its own policy.lock."""

    root = tmp_path_factory.mktemp("model-visible") / "tree"
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "ora"],
        check=True, capture_output=True, text=True,
    ).stdout
    for relative in filter(None, listed.split("\0")):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "scratch")
    policy_lock.write_lock(root)
    return root


@pytest.fixture
def tree(scratch_tree: Path, monkeypatch: pytest.MonkeyPatch):
    """Resolve ToolSurface manifests from the scratch copy; restore it afterwards.

    A manifest's geometry descriptor is read from the checkout that holds the
    manifest module, so that read is pointed at the copy as well: in the
    geometry module and where tool_surface imported its two functions.
    """

    profile_root = scratch_tree / "ora/policy/tool_surfaces"
    binding = embodiment_geometry.load_public_geometry_binding
    record = embodiment_geometry.bind_public_geometry_descriptor
    for module in (embodiment_geometry, tool_surface):
        monkeypatch.setattr(
            module, "load_public_geometry_binding",
            lambda value, *, project_root: binding(value, project_root=scratch_tree),
        )
        monkeypatch.setattr(
            module, "bind_public_geometry_descriptor",
            lambda value, *, project_root: record(value, project_root=scratch_tree),
        )
    monkeypatch.setattr(profile_catalog, "PROFILE_ROOT", profile_root)
    monkeypatch.setattr(profile_catalog, "PROFILE_CATALOG", {
        key: profile_root / path.relative_to(ROOT / "ora/policy/tool_surfaces")
        for key, path in profile_catalog.PROFILE_CATALOG.items()
    })
    yield scratch_tree
    _git(scratch_tree, "checkout", "-q", "--", ".")
    _git(scratch_tree, "clean", "-qfd", "-e", "policy.lock")
    _git(scratch_tree, "reset", "-q")
    policy_lock.write_lock(scratch_tree)


def _plan(root: Path, environment: str) -> dict:
    return resolve_run_spec(native_spec(environment), root=root)


def _replace_and_repin(root: Path, relative: str, edit) -> list[str]:
    """Edit one asset, then rewrite every hand-written pin that named its old bytes.

    This is the cascade the generated lock replaces: each rewritten pin holder
    has new bytes too, so its own pins are chased until nothing changes.
    Returns the holders it rewrote.
    """

    path = root / relative
    before = _sha(path.read_bytes())
    path.write_bytes(edit(path.read_bytes()))
    pending = {before: _sha(path.read_bytes())}
    rewritten: list[str] = []
    while pending:
        old, new = pending.popitem()
        if old == new:
            continue
        for holder in sorted((root / "ora/policy").rglob("*.json")):
            raw = holder.read_bytes()
            if old.encode() in raw:
                holder.write_bytes(raw.replace(old.encode(), new.encode()))
                rewritten.append(holder.relative_to(root).as_posix())
                pending[_sha(raw)] = _sha(holder.read_bytes())
    return rewritten


def _json_edit(mutate):
    def edit(raw: bytes) -> bytes:
        value = json.loads(raw)
        mutate(value)
        return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    return edit


# ---- the RunPlan fields --------------------------------------------------------

def _mv(plan: dict) -> str:
    return plan["policy_identity"]["model_visible_sha256"]


@pytest.mark.parametrize("environment", ["libero_pro", "robotwin"])
def test_run_plan_records_the_lock_and_model_visible_digests_beside_an_unchanged_source(environment):
    plan = _plan(ROOT, environment)
    assert set(plan["source"]) == {"commit", "dirty"}
    identity = plan["policy_identity"]
    assert set(identity) == {"schema", "policy_lock_sha256", "model_visible_sha256"}
    assert identity["schema"] == "ora.run-plan-policy-identity.v1"
    assert identity["policy_lock_sha256"] == _sha((ROOT / "policy.lock").read_bytes())
    assert identity["policy_lock_sha256"] == policy_lock.lock_sha256(ROOT)
    assert HEX.match(identity["model_visible_sha256"])
    assert identity["model_visible_sha256"] == model_visible.model_visible_sha256(plan, root=ROOT)


def test_a_tree_without_policy_lock_records_null_with_a_reason(tree):
    """Without a lock the tree's path references cannot be checked, so they do not resolve.

    The identity block still records the absence when it is bound.
    """
    plan = _plan(tree, "libero_pro")
    (tree / "policy.lock").unlink()
    with pytest.raises(ValueError, match="policy.lock is absent"):
        _plan(tree, "libero_pro")
    portable = {key: value for key, value in plan.items() if key != "policy_identity"}
    identity = spec_source.bind_policy_identity(portable, root=tree)
    assert identity["policy_lock_sha256"] is None
    assert identity["policy_lock_absent_reason"] == "policy.lock is absent from the resolving source tree"
    assert HEX.match(identity["model_visible_sha256"])


def test_execution_placement_does_not_move_the_model_visible_digest():
    """Lanes of one cohort, or two hosts, run the same cell on different slots."""

    plan = _plan(ROOT, "robotwin")
    moved = copy.deepcopy(plan)
    moved["runtime"].update(
        required_host="another-host", cuda_visible_devices="5", renderer_gpu=5,
        resource_binding={"profile_id": "robotwin-gpu5", "slot_id": "gpu5"},
    )
    assert model_visible.model_visible_sha256(moved, root=ROOT) == _mv(plan)
    moved["runtime"]["policy_cameras"] = ["head"]
    assert model_visible.model_visible_sha256(moved, root=ROOT) != _mv(plan)


# ---- execution capture: recorded for people, compared as the default -------------------

#: A plan with nothing but a capturing runtime and a budget, and the ``model_visible_sha256`` it had before execution
#: capture was compared as its default config (computed by the code before ``OBSERVER_KEYS``).
_CAPTURE_RUNTIME = {"kind": "robodojo", "task": "stack_bowls", "resource_binding": {"profile_id": "robodojo-gpu0"}}
_CAPTURE_PLAN_SHA256 = "1f0f5b47346795bcb8a8e93819d398c392e9d71d80265546ca132b27cec61a5b"


def _capture_plan(capture: dict) -> dict:
    return {"runtime": {**_CAPTURE_RUNTIME, "execution_capture": capture}, "budgets": {"max_robot_decisions": 80}}


def _digest_before_observer_keys(plan: dict) -> str:
    """``model_visible_sha256`` as computed before ``OBSERVER_KEYS``: the runtime only without placement."""

    items = model_visible.model_visible_items(plan, root=ROOT)
    items["settings"]["runtime"] = model_visible.mask_pins({
        key: value for key, value in plan["runtime"].items() if key not in model_visible.PLACEMENT_KEYS})
    return model_visible.canonical_sha256(items)


def test_the_compared_capture_is_the_recorders_default():
    from ora.environment.execution_capture import resolve_execution_capture

    assert model_visible.OBSERVER_KEYS["execution_capture"] == resolve_execution_capture(None)


def test_a_plan_that_records_at_the_default_keeps_its_digest():
    from ora.environment.execution_capture import resolve_execution_capture

    plan = _capture_plan(resolve_execution_capture(None))
    assert model_visible.model_visible_sha256(plan, root=ROOT) == _CAPTURE_PLAN_SHA256
    for environment in ("libero_pro", "robotwin"):
        resolved = _plan(ROOT, environment)
        assert resolved["runtime"]["execution_capture"] == resolve_execution_capture(None)
        assert _mv(resolved) == _digest_before_observer_keys(resolved), environment


@pytest.mark.parametrize("capture", [{"fps": 2}, {"enabled": False}, {"fps": 30, "queue_capacity": 8}])
@pytest.mark.parametrize("environment", ["libero_pro", "robotwin"])
def test_the_capture_setting_does_not_move_the_model_visible_digest(environment, capture):
    """A batch recorded at 2 fps, or not at all, compares by digest with the same cells at 10 fps."""

    spec = native_spec(environment)
    spec["runtime"]["execution_capture"] = capture
    plan = resolve_run_spec(spec, root=ROOT)
    for key, value in capture.items():
        assert plan["runtime"]["execution_capture"][key] == value
    assert _mv(plan) == _mv(_plan(ROOT, environment))
    assert _digest_before_observer_keys(plan) != _mv(plan), "the digest before OBSERVER_KEYS moved with it"
    assert model_visible.model_visible_sha256(
        _capture_plan({**model_visible.OBSERVER_KEYS["execution_capture"], **capture}), root=ROOT,
    ) == _CAPTURE_PLAN_SHA256


def test_a_plan_without_capture_stays_without_it():
    plan = {"runtime": dict(_CAPTURE_RUNTIME), "budgets": {"max_robot_decisions": 80}}
    assert "execution_capture" not in model_visible.model_visible_items(plan, root=ROOT)["settings"]["runtime"]
    assert model_visible.model_visible_sha256(plan, root=ROOT) != _CAPTURE_PLAN_SHA256


def _clean(plan: dict) -> dict:
    """Persisted plans must be clean; tests may run in a dirty checkout."""

    return {**plan, "source": {**plan["source"], "dirty": False}}


def test_persisted_plans_validate_with_and_without_the_policy_identity_block():
    plan = _clean(_plan(ROOT, "libero_pro"))
    assert validate_run_plan(plan) == plan

    tampered = copy.deepcopy(plan)
    tampered["policy_identity"]["model_visible_sha256"] = "0" * 64
    with pytest.raises(RunSpecError, match="hash does not match"):
        validate_run_plan(tampered)

    # A plan resolved before the block existed.
    legacy = {key: value for key, value in plan.items()
              if key not in {"plan_sha256", "plan_id", "confirmation_code", "policy_identity"}}
    legacy["source"] = {"commit": plan["source"]["commit"]}
    digest = json_sha256(legacy)
    legacy.update(
        plan_sha256=digest, confirmation_code=digest[-6:],
        plan_id=f"{plan['experiment']}-{plan['milestone'].lower()}-{digest[:12]}",
        source={"commit": plan["source"]["commit"], "dirty": False},
    )
    assert validate_run_plan(legacy) == legacy


SCHEMA = "ora.run-plan-policy-identity.v1"


@pytest.mark.parametrize("identity", [
    None,
    {"schema": SCHEMA, "policy_lock_sha256": None, "model_visible_sha256": "0" * 64},
    {"schema": SCHEMA, "policy_lock_sha256": "0" * 64},
    {"schema": SCHEMA, "policy_lock_sha256": "0" * 64, "model_visible_sha256": "0" * 64,
     "policy_lock_absent_reason": "absent"},
    {"schema": SCHEMA, "policy_lock_sha256": "ABC", "model_visible_sha256": "0" * 64},
    {"schema": "other", "policy_lock_sha256": "0" * 64, "model_visible_sha256": "0" * 64},
])
def test_a_malformed_policy_identity_is_refused(identity):
    plan = _clean(_plan(ROOT, "libero_pro"))
    with pytest.raises(RunSpecError, match="source identity is invalid"):
        validate_run_plan({**plan, "policy_identity": identity})


def test_the_source_block_keeps_its_exact_shape():
    plan = _clean(_plan(ROOT, "libero_pro"))
    with pytest.raises(RunSpecError, match="source identity is invalid"):
        validate_run_plan({**plan, "source": {**plan["source"], "model_visible_sha256": "0" * 64}})


def test_the_mask_drops_hashes_and_pin_only_paths_but_keeps_everything_else():
    value = {
        "profile": "p", "manifest_sha256": "a", "registry": "r.json",
        "baseline": {"id": "b", "path": "b.json", "sha256": "s",
                     "prompt_bundle": {"path": "x.yaml", "sha256": "t"}},
        "environment_prompt_profile": {"profile_id": "e", "profile_path": "e.json", "profile_sha256": "u"},
        "contract": {"record": "r.json", "record_sha256": "v", "layer_count": 2},
        "layers": [{"path": "l.md", "sha256": "w", "bytes": 3}],
        "lone": {"path": "kept.json", "bytes": 4},
    }
    assert model_visible.mask_pins(value) == {
        "profile": "p", "registry": "r.json",
        "baseline": {"id": "b", "prompt_bundle": {}},
        "environment_prompt_profile": {"profile_id": "e"},
        "contract": {"layer_count": 2},
        "layers": [{}],
        "lone": {"path": "kept.json", "bytes": 4},
    }


# ---- unchanged by pin-only edits -----------------------------------------------

def _identity(plan: dict) -> tuple[str, str, str]:
    return (
        _mv(plan),
        plan["policy"]["tool_surface"]["composition_sha256"],
        plan["runtime"]["environment_prompt_profile"]["profile_sha256"],
    )


def test_manifest_provenance_prose_does_not_move_the_model_visible_digest(tree):
    before = _identity(_plan(tree, "libero_pro"))

    def note(value):
        value["provenance"]["change"] += " Re-reviewed without changing delivery."
    _replace_and_repin(tree, LIBERO_MANIFEST, _json_edit(note))
    after = _identity(_plan(tree, "libero_pro"))
    assert after[1] != before[1]
    assert after[0] == before[0]


def test_a_component_moved_to_a_new_path_does_not_move_the_model_visible_digest(tree):
    before = _identity(_plan(tree, "libero_pro"))
    manifest = json.loads((tree / LIBERO_MANIFEST).read_text())
    # LIBERO's surface is file-backed: its components are its own tool files.
    component = next(iter(manifest["tool_descriptions"].values()))
    source = (tree / LIBERO_MANIFEST).parent / component
    moved = source.with_name("moved-" + source.name)
    shutil.copyfile(source, moved)

    def relocate(value):
        for tool, item in value["tool_descriptions"].items():
            if item == component:
                value["tool_descriptions"][tool] = str(Path(component).with_name(moved.name))
    _replace_and_repin(tree, LIBERO_MANIFEST, _json_edit(relocate))
    # A new asset is tracked and locked before anything may load it.
    _git(tree, "add", moved.relative_to(tree).as_posix())
    policy_lock.write_lock(tree)
    after = _identity(_plan(tree, "libero_pro"))
    assert after[1] != before[1]
    assert after[0] == before[0]


@pytest.mark.parametrize("environment", ["libero_pro", "robotwin"])
def test_a_descriptor_provenance_pin_and_its_cascade_do_not_move_the_model_visible_digest(tree, environment):
    plan = _plan(tree, environment)
    before = _identity(plan)
    descriptor = plan["runtime"]["public_geometry_descriptor"]["path"]

    def repin_external_source(value):
        source = next(item for item in value["provenance"]["sources"] if "local_path" not in item)
        source["sha256"] = _sha(source["sha256"].encode())
    # Nothing pins the descriptor's bytes any more: profiles and surfaces name it
    # by path (L2) and its wording review attests only the card (L3).
    assert _replace_and_repin(tree, descriptor, _json_edit(repin_external_source)) == []
    after_plan = _plan(tree, environment)
    after = _identity(after_plan)
    assert after_plan["runtime"]["public_geometry_descriptor"] != plan["runtime"]["public_geometry_descriptor"]
    # The surface records the descriptor's new hash; the context profile names
    # the descriptor by path, so the edit no longer cascades into its bytes.
    assert after[1] != before[1] and after[2] == before[2]
    assert after[0] == before[0]


# ---- moved by model-visible edits ---------------------------------------------

def test_a_tool_description_change_moves_the_model_visible_digest(tree):
    plan = _plan(tree, "libero_pro")
    manifest = json.loads((tree / LIBERO_MANIFEST).read_text())
    # A tool file of LIBERO's file-backed surface is the text it delivers.
    ref = manifest["tool_descriptions"]["move_pose"]
    component = os.path.normpath((Path(LIBERO_MANIFEST).parent / ref).as_posix())

    def describe(value):
        value["description"] += " Edited."
    _replace_and_repin(tree, component, _json_edit(describe))
    after_plan = _plan(tree, "libero_pro")
    assert _mv(after_plan) != _mv(plan)
    before_items = model_visible.model_visible_items(plan, root=tree)
    after_items = model_visible.model_visible_items(after_plan, root=tree)
    assert after_items["advertised"] != before_items["advertised"]
    assert after_items["prompt_layers"] == before_items["prompt_layers"]


def test_a_prompt_layer_text_change_moves_the_model_visible_digest(tree):
    before = _mv(_plan(tree, "libero_pro"))
    path = tree / CORE_LAYER
    path.write_text(path.read_text() + "\nOne more sentence.\n")
    assert _mv(_plan(tree, "libero_pro")) != before


def test_an_embodiment_card_change_moves_the_model_visible_digest(tree):
    plan = _plan(tree, "libero_pro")
    before = model_visible.model_visible_items(plan, root=tree)
    descriptor = plan["runtime"]["public_geometry_descriptor"]["path"]
    profile = json.loads((tree / plan["runtime"]["environment_prompt_profile"]["profile_path"]).read_text())
    card = tree / profile["sources"]["embodiment_policy"]
    old_card = card.read_bytes()

    def widen(value):
        value["geometry"]["grippers"][0]["maximum_aperture_m"] = 0.081
    _replace_and_repin(tree, descriptor, _json_edit(widen))
    new_card = render_embodiment_card(load_embodiment_descriptor(tree / descriptor)).encode()
    assert new_card != old_card
    _replace_and_repin(tree, card.relative_to(tree).as_posix(), lambda raw: new_card)

    after_plan = _plan(tree, "libero_pro")
    after = model_visible.model_visible_items(after_plan, root=tree)
    assert _mv(after_plan) != _mv(plan)
    assert after["embodiment"] != before["embodiment"]
    assert after["prompt_layers"] != before["prompt_layers"]


# ---- scripts/dev/compare-model-visible ----------------------------------------------

def _compare_script():
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.SourceFileLoader(
        "compare_model_visible_script", str(ROOT / "scripts/dev/compare-model-visible"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_moving_a_surface_to_the_toolkit_keeps_its_digest_where_the_environment_plans():
    capabilities = ["rgbd", "yam_rgbd_v1"]
    surface = {"view_config": None, "capabilities": capabilities}

    def digest(names):
        plan = {"policy_composition": {"tool_surface": {**surface, "capabilities": names}}}
        return model_visible.model_visible_items(plan, root=ROOT)["observation_projection"]["capabilities"]

    assert digest(["rgbd", "toolkit_only_v1", "yam_rgbd_v1"]) == capabilities
    synchronous = ["toolkit_only_v1", "toolkit_synchronous_execution_v1"]
    assert digest(synchronous) == synchronous, "a synchronous surface's tools execute: its digest says so"


def test_the_equivalence_matrix_is_the_fixed_cells():
    cells = _compare_script().matrix(ROOT)
    environments = ("libero_pro", "robotwin", "robodojo", "robocasa", "robolab")
    # A fresh default RunSpec on each of the five backends, and the Synthetic quickstart.
    defaults = {f"direct_v4/{environment}/native_api" for environment in environments}
    defaults.add("quickstart/synthetic/native_api")
    # Each of those in code mode, with each library, and the two consumers no default names.
    assert set(cells) == defaults | {
        f"{mode}/{environment}/native_api" for mode in ("code_mode", "code_mode_extended")
        for environment in (*environments, "synthetic")
    } | {"local/libero_pro/native_api", "play/yam/native_api", "play_perception/yam/native_api",
         # RoboDojo's default runs code mode (2026-11-05); its tool calls are the code_mode: null cell.
         "tool_calls/robodojo/native_api"}


def test_the_consumer_cells_resolve_the_composition_they_are_named_for():
    script = _compare_script()
    cells = script.matrix(ROOT)

    def plan(cell):
        return resolve_run_spec(cells[cell](), root=ROOT)

    local = plan("local/libero_pro/native_api")
    assert local["policy"]["tool_surface"]["profile"] == "local_atomic_cpu_v1"
    play = plan("play/yam/native_api")
    assert play["policy"]["tool_surface"]["profile"] == "yam_direct_v1"
    assert play["episode_mode"] == "play" and "finish_episode" not in play["policy"]["tool_allowlist"]
    perception = plan("play_perception/yam/native_api")
    assert perception["policy"]["tool_surface"]["profile"] == "yam_perception_v1"
    assert {"vision_segment", "vision_ground", "grasp_plan"} <= set(perception["policy"]["tool_allowlist"])
    base = plan("code_mode/libero_pro/native_api")["policy"]["code_mode"]
    assert "library" not in base and "script_reads" not in base
    extended = plan("code_mode_extended/libero_pro/native_api")["policy"]["code_mode"]
    assert extended["library"] == "extended" and extended["prompt"] == "v3"
    assert extended["script_reads"] == ["dry_run"]
    assert "query" in plan("code_mode_extended/robodojo/native_api")["policy"]["code_mode"]["script_reads"]
    assert plan("direct_v4/robodojo/native_api")["policy"]["code_mode"]["source"] is True
    assert "code_mode" not in plan("tool_calls/robodojo/native_api")["policy"]


def test_every_equivalence_cell_resolves_on_this_tree(monkeypatch):
    """A cell that no longer resolves would compare as unresolved at both commits, which passes."""

    for name in ("ORA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for cell, build in _compare_script().matrix(ROOT).items():
        spec = build()
        plan = resolve_run_spec(spec, root=ROOT)
        assert plan["runtime"]["kind"] == cell.split("/")[1], cell
        if plan["runtime"]["kind"] in {"robocasa", "robolab"}:
            # The episode identity these backends check: task, seed and official language.
            assert (plan["runtime"]["task"], plan["runtime"]["seed"]) == (spec["task"], spec["seed"]), cell
            assert plan["runtime"]["expected_task_language"] == spec["task_instruction"], cell
        assert _mv(plan) == model_visible.model_visible_sha256(plan, root=ROOT), cell


def test_the_comparison_names_what_differs_and_sets_the_exit_code(capsys):
    script = _compare_script()
    plan = _plan(ROOT, "libero_pro")
    items = model_visible.model_visible_items(plan, root=ROOT)
    changed = copy.deepcopy(items)
    changed["advertised"]["definitions"][1]["description"] += " Edited."
    changed["prompt_layers"]["channels"]["tool_manual"] = ["0" * 64]

    def entry(value):
        return {"status": "resolved", "items": value, "model_visible_sha256": model_visible.canonical_sha256(value)}

    base = {"commit": "a", "cells": {"same": entry(items), "edited": entry(items),
                                       "refused": {"status": "unresolved", "error": "E: x"},
                                       "lost": entry(items)}}
    candidate = {"commit": "b", "cells": {"same": entry(items), "edited": entry(changed),
                                            "refused": {"status": "unresolved", "error": "E: x"},
                                            "lost": {"status": "unresolved", "error": "E: y"}}}
    cells = script.compare(base, candidate)
    assert cells["same"]["result"] == "same"
    assert cells["refused"]["result"] == "unresolved_at_both"
    assert cells["lost"]["result"] == "different"
    name = items["advertised"]["definitions"][1]["name"]
    assert f"advertised: tool definition differs: {name}" in cells["edited"]["details"]
    assert "prompt_layers: $prompt_layers.channels.tool_manual[0]" in cells["edited"]["details"]
    assert script.report(base, candidate, cells) == 1
    assert "2 different" in capsys.readouterr().out
    assert script.report(base, base, script.compare(base, base)) == 0
    nothing = {"commit": "c", "cells": {"refused": {"status": "unresolved", "error": "E: x"}}}
    assert script.report(nothing, nothing, script.compare(nothing, nothing)) == 2
