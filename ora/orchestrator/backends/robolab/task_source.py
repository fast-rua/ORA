"""RoboLab-120 task facts read from the pinned source: task classes by AST, metadata and scene poses.

No simulator import: ``load_facts`` parses each benchmark task class, its termination terms
(every non-timeout DoneTerm: RoboLab's verdict is their OR) and its subtasks, checks them against
``task_metadata.json`` and reads the scene's authored top-level prims. ``task_catalog`` writes the
roster, contracts and reviews from these facts.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ora.environment.benchmarks.robolab import CONTROL_FREQ_HZ

HELPERS = {
    "conditionals": "robolab/core/task/conditionals.py",
    "predicate_logic": "robolab/core/task/predicate_logic.py",
    "hull_check": "robolab/core/task/hull_check.py",
    "world_state": "robolab/core/world/world_state.py",
    "env": "robolab/core/environments/env.py",
}

# ------------------------------------------------------------------------------------------------- facts
@dataclass
class Term:
    name: str
    func: str
    params: dict[str, Any]


@dataclass
class Facts:
    task: str
    file: str
    instruction: str
    episode_s: int
    scene: str
    contact_objects: list[str]
    attributes: list[str]
    terms: list[Term]
    subtasks_text: str
    subtask_calls: list[tuple[str, dict[str, Any]]]
    metadata: dict[str, Any]
    prims: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def native_limit(self) -> int:
        return self.episode_s * CONTROL_FREQ_HZ


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except ValueError:
        return ast.unparse(node)


def _partial_calls(node: ast.AST) -> list[tuple[str, dict[str, Any]]]:
    """Every ``partial(fn, k=v)`` or composite ``fn(k=v)`` call under a subtasks expression."""
    calls = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        name = item.func.id if isinstance(item.func, ast.Name) else None
        if name == "partial" and item.args and isinstance(item.args[0], ast.Name):
            calls.append((item.args[0].id, {k.arg: _literal(k.value) for k in item.keywords if k.arg}))
        elif name in ("pick_and_place", "pick_and_place_on_surface", "pick_and_place_grouped"):
            calls.append((name, {k.arg: _literal(k.value) for k in item.keywords if k.arg}))
    return calls


_PRIM = re.compile(r'^    def (?:\w+ )?"([^"]+)"')


def scene_prims(text: str) -> dict[str, dict[str, Any]]:
    """Top-level prims under the default prim of an ASCII scene: asset, authored translate and orient."""
    prims: dict[str, dict[str, Any]] = {}
    current = None
    for line in text.splitlines():
        match = _PRIM.match(line)
        if match:
            current = prims.setdefault(match.group(1), {})
            continue
        if re.match(r"^\S", line) or line.rstrip() == "    }":
            current = None
            continue
        if current is None or not line.startswith("        ") or line.startswith("         "):
            continue
        text_line = line.strip()
        asset = re.match(r"(?:prepend )?(?:payload|references) = @([^@]+)@", text_line)
        if asset and "asset" not in current:
            current["asset"] = asset.group(1)
        translate = re.match(r"double3 xformOp:translate = \(([^)]+)\)", text_line)
        if translate and "translate" not in current:
            current["translate"] = [float(value) for value in translate.group(1).split(",")]
        orient = re.match(r"quat[fd] xformOp:orient = \(([^)]+)\)", text_line)
        if orient and "orient" not in current:
            current["orient"] = [float(value) for value in orient.group(1).split(",")]
    return prims


def _asset_key(asset: str | None) -> str | None:
    if not asset:
        return None
    return asset.split("../", 1)[-1] if not asset.startswith("http") else asset


def load_facts(source: Path) -> dict[str, Facts]:
    """Every benchmark task class, read by AST and checked against ``task_metadata.json``."""
    metadata = {item["task_name"]: item for item in
                json.loads((source / "robolab/tasks/_metadata/task_metadata.json").read_text(encoding="utf-8"))}
    facts: dict[str, Facts] = {}
    for path in sorted((source / "robolab/tasks/benchmark").glob("*.py")):
        if path.name.startswith("_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        for node in classes.values():
            if not any(isinstance(base, ast.Name) and base.id == "Task" for base in node.bases):
                continue
            values: dict[str, ast.AST] = {}
            for statement in node.body:
                target = statement.targets[0] if isinstance(statement, ast.Assign) else (
                    statement.target if isinstance(statement, ast.AnnAssign) else None)
                if isinstance(target, ast.Name) and statement.value is not None:
                    values[target.id] = statement.value
            terminations = classes[values["terminations"].id]
            terms = []
            for statement in terminations.body:
                if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
                    continue
                keywords = {k.arg: k.value for k in statement.value.keywords}
                if _literal(keywords.get("time_out", ast.Constant(False))) is True:
                    continue
                func = keywords["func"]
                terms.append(Term(statement.targets[0].id, func.id if isinstance(func, ast.Name) else ast.unparse(func),
                                  _literal(keywords["params"]) if "params" in keywords else {}))
            instruction = _literal(values["instruction"])
            instruction = instruction["default"] if isinstance(instruction, dict) else instruction
            scene_call = values["scene"]
            scene = _literal(scene_call.args[0])
            meta = metadata.get(node.name)
            if meta is None:
                raise ValueError(f"{node.name} has no task_metadata.json entry")
            item = Facts(task=node.name, file=f"robolab/tasks/benchmark/{path.name}", instruction=instruction,
                         episode_s=int(_literal(values["episode_length_s"])), scene=scene,
                         contact_objects=list(_literal(values["contact_object_list"])),
                         attributes=list(_literal(values.get("attributes", ast.Constant([]))) or []),
                         terms=terms, subtasks_text=meta.get("subtasks", ""),
                         subtask_calls=_partial_calls(values["subtasks"]) if "subtasks" in values else [],
                         metadata=meta)
            mismatches = [name for name, ours, theirs in (
                ("instruction", item.instruction, meta["instruction_variants"]["default"]),
                ("episode_s", str(item.episode_s), str(meta["episode_s"])), ("scene", item.scene, meta["scene"]),
                ("filename", f"benchmark/{path.name}", meta["filename"]),
                ("contact_objects", ", ".join(item.contact_objects), meta["contact_objects"]),
            ) if ours != theirs]
            if mismatches:
                raise ValueError(f"{node.name}: task_metadata.json differs from the class on {mismatches}")
            if not item.terms:
                raise ValueError(f"{node.name} has no success termination")
            item.prims = scene_prims((source / "assets/scenes" / scene).read_text(encoding="utf-8"))
            facts[node.name] = item
    if set(facts) != set(metadata):
        raise ValueError(f"task classes and task_metadata.json differ: {sorted(set(facts) ^ set(metadata))}")
    return facts


# ------------------------------------------------------------------------------------------------ family
FAMILIES = {
    "object_in_container": "in_container", "object_inside": "in_container",
    "object_groups_in_containers": "groups_in_containers", "object_on_top": "on_surface", "stacked": "stacked",
    "object_outside_of": "out_of_container", "object_outside_of_and_on_surface": "out_onto_surface",
    "object_picked_up": "picked_up", "object_upright": "upright",
    "objects_placed_in_container_in_order": "ordered_in_container", "object_left_of": "relative_position",
    "object_right_of": "relative_position", "object_in_front_of": "relative_position",
    "object_behind": "relative_position", "object_on_center": "centered",
}
SKILLS = {"groups_in_containers": "sorting", "stacked": "stacking", "out_of_container": "take_out",
          "out_onto_surface": "take_out", "picked_up": "pick_up", "upright": "reorient",
          "ordered_in_container": "ordered_pick_and_place", "relative_position": "relative_placement",
          "centered": "relative_placement"}


def family(facts: Facts) -> str:
    funcs = {term.func for term in facts.terms}
    if len(funcs) != 1:
        raise ValueError(f"{facts.task} mixes success predicates {sorted(funcs)}")
    return FAMILIES[funcs.pop()]


def _as_list(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else list(value or [])


def targets(facts: Facts) -> list[str]:
    names: list[str] = []
    for term in facts.terms:
        params = term.params
        for key in ("object", "objects"):
            names += _as_list(params.get(key))
        for group in params.get("groups") or []:
            names += _as_list(group["object"])
    return names


def references(facts: Facts) -> list[str]:
    names: list[str] = []
    for term in facts.terms:
        for key in ("container", "reference_object", "surface"):
            if term.params.get(key):
                names.append(term.params[key])
        for group in term.params.get("groups") or []:
            names.append(group["container"])
    return list(dict.fromkeys(names))


def logical(facts: Facts) -> tuple[str, int]:
    if len(facts.terms) > 1:
        return "any", 1
    params = facts.terms[0].params
    return str(params.get("logical", "all")), int(params.get("K", 1))



__all__ = ["FAMILIES", "Facts", "HELPERS", "SKILLS", "Term", "family", "load_facts", "logical", "references",
           "scene_prims", "targets"]
