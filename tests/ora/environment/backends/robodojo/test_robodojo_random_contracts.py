"""The reviewed contracts of the RoboDojo ``*_random`` generalization variants (registry-v10).

RoboDojo runs 54 settings: the 42 official base tasks and the ``*_random``
variant of each of its 12 generalization tasks, each its own upstream task
module (``task/RoboDojo/tasks/<task>_random.py``) with its own cluttered
layouts. A variant's record reviews its own layout 0 against its base task's
record at the same pins. Where the variant keeps its base task's instruction
and the base text holds on the variant's scene, the registry binds the base
task's completion layer itself, so the policy reads the same reviewed words;
``sweep_blocks_random`` scores bricks, earbuds and paper balls instead of
three blocks and binds its own layer. A variant the source cannot contract
honestly stays out of the registry, named here with the reason.

The cases marked ``needs_source`` re-read the pinned upstream source and
Assets (``ORA_TEST_ROBODOJO_RUNTIME``) and skip without them.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re

import pytest

from ora.agent.composition.task_contract.robodojo import (
    REGISTRY_PATH, load_task_contract, resolve_reviewed_contract,
)
from ora.environment.benchmarks.robodojo import NATIVE_ACTION_LIMITS, REVISIONS
from ora.orchestrator.paths import PROJECT_ROOT
from tests.support.robodojo import runtime as robodojo_runtime

REGISTRY = json.loads((PROJECT_ROOT / REGISTRY_PATH).read_text(encoding="utf-8"))["tasks"]
ACTIVATION = json.loads((PROJECT_ROOT / "tests/data/default-expectations.json").read_text(encoding="utf-8"))
#: The 12 generalization tasks of upstream scripts/internal/task_inventory.py at f57e163.
GENERALIZATION = (
    "arrange_largest_number", "fold_clothes", "hang_mugs", "make_toast", "pack_objects_into_box",
    "pour_liquid_into_cup", "push_T", "sort_nesting_dolls_by_size", "stack_blocks", "stack_bowls",
    "store_laptop_and_headphones", "sweep_blocks",
)
VARIANTS = tuple(f"{task}_random" for task in GENERALIZATION)
#: Variants the source review could not contract honestly, with the reason; they stay out of the registry.
LEFT_OUT: dict[str, str] = {}
CONTRACTED = tuple(variant for variant in VARIANTS if variant not in LEFT_OUT)
#: Variants whose delivered text differs from their base task's, and why.
OWN_TEXT = {"sweep_blocks_random": "scores every toy brick, earbud and paper ball and says 'sweep the objects'"}
RETURN_CLAUSE = (
    "Both arms must be back at the poses they held when the episode reset, at the "
    "same instant that everything else in this contract is true."
)
SOURCE = robodojo_runtime.SOURCE_ROOT
ASSETS = robodojo_runtime.ASSET_ROOT
needs_source = pytest.mark.skipif(
    not (SOURCE / "task/RoboDojo/tasks").is_dir() or not (ASSETS / "Eval_Layout/RoboDojo/arx_x5/0").is_dir(),
    reason="pinned RoboDojo source and Assets absent (ORA_TEST_ROBODOJO_RUNTIME)")


def _record(task: str) -> dict:
    return json.loads((PROJECT_ROOT / REGISTRY[task]["record"]).read_text(encoding="utf-8"))


def _layer(task: str) -> str:
    return (PROJECT_ROOT / REGISTRY[task]["layer"]).read_text(encoding="utf-8")


# --- registry coverage --------------------------------------------------------------------------------

def test_the_registry_covers_all_54_settings_but_the_named_exclusions():
    base_tasks = {task for task in NATIVE_ACTION_LIMITS if not task.endswith("_random")}
    assert len(base_tasks) == 42 and set(GENERALIZATION) <= base_tasks
    assert set(NATIVE_ACTION_LIMITS) == base_tasks | set(VARIANTS) and len(NATIVE_ACTION_LIMITS) == 54
    assert set(REGISTRY) == set(NATIVE_ACTION_LIMITS) - set(LEFT_OUT)
    assert len([task for task in REGISTRY if not task.endswith("_random")]) == 42
    assert ACTIVATION["task_contracts"]["robodojo"] == {
        **ACTIVATION["task_contracts"]["robodojo"], "registry": REGISTRY_PATH, "task_count": 54 - len(LEFT_OUT)}
    # registry-v10 keeps every registry-v9 entry byte for byte and only appends the variants.
    frozen = ACTIVATION["frozen_task_contracts"]["robodojo_registry_v9"]
    v9 = json.loads((PROJECT_ROOT / frozen["path"]).read_text(encoding="utf-8"))["tasks"]
    assert list(REGISTRY)[:len(v9)] == list(v9) and {task: REGISTRY[task] for task in v9} == v9
    assert list(REGISTRY)[len(v9):] == list(CONTRACTED)
    for variant, reason in LEFT_OUT.items():
        assert reason and variant not in REGISTRY


@pytest.mark.parametrize("variant", CONTRACTED)
def test_each_variant_keeps_its_base_tasks_official_limit(variant):
    base = variant.removesuffix("_random")
    record = _record(variant)
    assert record["native_action_limit"] == NATIVE_ACTION_LIMITS[variant] == NATIVE_ACTION_LIMITS[base]
    assert record["native_action_limit"] == _record(base)["native_action_limit"]


# --- record and registry consistency ---------------------------------------------------------------------

@pytest.mark.parametrize("variant", CONTRACTED)
def test_each_variant_record_reviews_its_own_module_and_layout_against_its_base(variant):
    base = variant.removesuffix("_random")
    record = _record(variant)
    assert record["schema"] == "ora.robodojo-task-contract.v1" and record["task"] == variant
    source, layout = record["source"], record["layout"]
    assert source["robodojo_revision"] == REVISIONS["robodojo"]
    assert source["source_path"] == f"task/RoboDojo/tasks/{variant}.py"
    assert source["config_path"] == f"task/RoboDojo/config/{variant}.yml"
    assert source["variant_of"]["task"] == base
    assert source["variant_of"]["source_path"] == _record(base)["source"]["source_path"]
    assert (layout["eval_seed"], layout["ordinal"], layout["assets_revision"]) == (0, 0, REVISIONS["assets"])
    assert layout["layout_file"] == f"Eval_Layout/RoboDojo/arx_x5/0/{variant}_0.json"
    assert record["variant"]["base_task"] == base
    assert record["variant"]["base_record"] == REGISTRY[base]["record"]
    # The user approved the twelve contracts on 2026-11-14, relayed by the coordinator.
    approval = record["review"]["human_approval"]
    assert (approval["status"], approval["by"], approval["date"]) == ("approved", "the user", "2026-11-14")
    assert "relayed" in approval["provenance"] and approval["user_words"]
    # The config the variant's layouts were drawn from is an evaluator dependency fresh preflight re-hashes.
    assert {"root": "robodojo", "path": source["config_path"], "sha256": source["config_sha256"]} in (
        record["evaluator_dependencies"])
    if variant in OWN_TEXT:
        assert record["instruction"]["text"] != _record(base)["instruction"]["text"]
    else:
        assert record["instruction"]["text"] == _record(base)["instruction"]["text"]


@pytest.mark.parametrize("variant", CONTRACTED)
def test_a_variant_delivers_its_base_layer_verbatim_unless_its_own_text_is_reviewed(variant):
    base = variant.removesuffix("_random")
    record = _record(variant)
    contract = load_task_contract(variant, bindings=REGISTRY, project_root=PROJECT_ROOT)
    assert contract["layer"] == REGISTRY[variant]["layer"] == record["contract_layer"]["path"]
    assert record["contract_layer"]["sha256"] == hashlib.sha256(
        (PROJECT_ROOT / REGISTRY[variant]["layer"]).read_bytes()).hexdigest()
    if variant in OWN_TEXT:
        assert REGISTRY[variant]["layer"] != REGISTRY[base]["layer"]
        assert record["semantic_contract"] != _record(base)["semantic_contract"]
        assert record["variant"]["relation"] == "different"
    else:
        assert REGISTRY[variant]["layer"] == REGISTRY[base]["layer"]
        assert record["semantic_contract"] == _record(base)["semantic_contract"]
    if _record(base)["semantic_contract"][-1] == " ".join(RETURN_CLAUSE.split()):
        assert record["semantic_contract"][-1] == " ".join(RETURN_CLAUSE.split())


@pytest.mark.parametrize("variant", CONTRACTED)
def test_a_variant_record_is_wellformed_and_keeps_private_detail_out_of_its_layer(variant):
    record = _record(variant)
    clauses = record["evaluator_clauses"]
    ids = [clause["id"] for clause in clauses]
    gap = record["gap"]
    assert ids and len(ids) == len(set(ids))
    assert sorted(gap["stated"] + gap["derivable"] + gap["hidden"]) == sorted(ids)
    assert gap["clause_count"] == len(ids) and gap["hidden_count"] == len(gap["hidden"])
    active = [clause["id"] for clause in clauses
              if clause["classification"] == "hidden" and clause.get("bite") == "active"]
    assert record["verdict"] == ("instruction_incomplete" if active else "instruction_complete")
    assert record["disclosure"]["private_numeric_thresholds_delivered"] is False
    assert record["disclosure"]["task_completion_contract"] == record["semantic_contract"]
    text = _layer(variant)
    assert "{{TASK}}" not in text
    assert re.search(r"\b\d+(\.\d+)?\s*(m|mm|cm|deg|degrees|metres|meters)\b", text) is None
    for token in ("is_lift", "is_axis_up", "is_stacked", "is_A_in_B", "is_axis_aligned",
                  "all_robot_back_to_origin", "reward_manager", "run_reward", "get_score", "target_",
                  "functional point", "bounding box", "quaternion", "threshold", "tolerance", "predicate"):
        assert token not in text, (variant, token)


@pytest.mark.parametrize("variant", CONTRACTED)
def test_the_default_resolver_delivers_each_variant_on_its_reviewed_layout(variant):
    record = _record(variant)
    runtime = {"task_variant": variant, "required_revision": record["source"]["robodojo_revision"],
               "required_asset_revision": record["layout"]["assets_revision"],
               "layout_file": record["layout"]["layout_file"], "layout_sha256": record["layout"]["layout_sha256"],
               "eval_seed": 0, "layout_id": 0, "expected_task_language": record["instruction"]["text"]}
    layer, disclosure = resolve_reviewed_contract(runtime, project_root=PROJECT_ROOT,
                                                  disclosure={"backend": "robodojo"})
    assert layer == REGISTRY[variant]["layer"] and disclosure["delivered"] is True
    assert disclosure["registry"] == REGISTRY_PATH and disclosure["record"] == REGISTRY[variant]["record"]
    base_record = _record(variant.removesuffix("_random"))
    # The base task's layout is another file: a variant's record is refused on it.
    with pytest.raises(RuntimeError, match="differs from runtime layout_(file|sha256)"):
        resolve_reviewed_contract({**runtime, "layout_file": base_record["layout"]["layout_file"],
                                   "layout_sha256": base_record["layout"]["layout_sha256"]},
                                  project_root=PROJECT_ROOT, disclosure={})


# --- the pinned upstream source --------------------------------------------------------------------------

def _span(path, name):
    data = path.read_bytes()
    lines = data.decode("utf-8").splitlines(keepends=True)
    for node in ast.parse(data).body:
        if isinstance(node, ast.ClassDef) and node.name.endswith("Common"):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    body = "".join(lines[item.lineno - 1:item.end_lineno]).encode("utf-8")
                    return [item.lineno, item.end_lineno], hashlib.sha256(body).hexdigest()
    raise AssertionError(f"{path} has no {name}")


@needs_source
@pytest.mark.parametrize("variant", CONTRACTED)
def test_each_variant_record_binds_the_pinned_upstream_bytes(variant):
    record = _record(variant)
    source = record["source"]
    path = SOURCE / source["source_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source["file_sha256"]
    assert _span(path, "run_reward") == (source["run_reward_lines"], source["run_reward_sha256"])
    assert _span(path, "gen_instruction") == (source["gen_instruction_lines"], source["gen_instruction_sha256"])
    assert f"self.step_lim = {record['native_action_limit']}" in path.read_text().splitlines()[
        source["step_lim_line"] - 1]
    assert record["instruction"]["text"] in path.read_text()
    layout = ASSETS / record["layout"]["layout_file"]
    robodojo_runtime.skip_lfs_pointers([layout] if robodojo_runtime.is_lfs_pointer(layout) else [],
                                       "the record's other bytes were not checked either")
    assert hashlib.sha256(layout.read_bytes()).hexdigest() == record["layout"]["layout_sha256"]
    for dependency in record["evaluator_dependencies"]:
        root = SOURCE if dependency["root"] == "robodojo" else ASSETS
        assert hashlib.sha256((root / dependency["path"]).read_bytes()).hexdigest() == dependency["sha256"], (
            dependency["path"])
    # Same-predicate variants carry their base task's run_reward bytes exactly.
    base = SOURCE / source["variant_of"]["source_path"]
    same = _span(base, "run_reward")[1] == source["run_reward_sha256"]
    assert same == record["variant"]["predicate_same_bytes_as_base"] or variant == "make_toast_random"
