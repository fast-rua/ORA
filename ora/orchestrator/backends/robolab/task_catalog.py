"""RoboLab-120 admission: the roster, completion contracts and review drafts, generated from the pinned source.

    ora dev robolab-tasks generate --source-root ROOT [--check]   roster, contracts, reviews, registry, task page
    ora dev robolab-tasks regression --source-root ROOT           this generator against the three reviewed tasks
    ora dev robolab-tasks scan-assets --source-root ROOT --python PY   composed-scene asset scan (pxr only, no Kit)
    ora dev robolab-tasks promote [--task T ...]                   witnessed reviews into the delivered registry
    ora spec robolab tasks                                          the roster with contract and compatibility state

Every task comes from one reading of the pinned checkout: the task class (AST, no
simulator import), its termination terms, ``task_metadata.json`` (checked against the
class), the scene's authored poses and the asset scan. RoboLab's verdict is
``TerminationManager.terminated``, the OR of every non-timeout DoneTerm, so the
contract restates those terms, never the partial ``subtasks`` scores; where the two
differ the review lists the difference (``termination_vs_subtasks``).

A generated review is a draft: ``status`` stays ``needs_review``, the witness checks
stay ``pending`` and the registry lists the task under ``pending_review``, which the
resolver never delivers (``contract_review_pending``) and launch preflight refuses.
``ora qualify robolab contract-witness`` runs the seed-0 witness and fills the
witness checks; ``promote`` moves a review whose nine checks all passed into
``tasks``. The three tasks reviewed on 2026-10-18 keep their committed text.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.environment.benchmarks.robolab import REVISION
from ora.orchestrator.paths import PROJECT_ROOT

from .task_review import build_review, merge_witness, predicate_binding
from .task_source import SKILLS, Facts, family, load_facts, logical, targets
from .task_wording import Projection, project

CONTRACTS = "ora/policy/environments/robolab/task-contracts"
REGISTRY = f"{CONTRACTS}/registry-v1.json"
ASSET_SCAN = f"{CONTRACTS}/asset-scan-v1.json"
ROSTER_MODULE = "ora/environment/benchmarks/robolab.py"
TASK_PAGE = "docs/reference/robolab-tasks.md"
ROSTER_BEGIN = "    # --- generated roster: ora dev robolab-tasks generate ---\n"
ROSTER_END = "    # --- end of generated roster ---\n"
PAGE_BEGIN = "<!-- generated: ora dev robolab-tasks generate -->\n"
PAGE_END = "<!-- end of generated table -->\n"
SCAN_SCHEMA = "ora.robolab-asset-scan.v1"
PENDING_VERDICT = "source_reviewed_witness_pending"
REVIEWED_VERDICT = "reviewed_semantic_projection_with_explicit_scoring_gaps"

# ------------------------------------------------------------------------------------------------ render
def render_contract(clauses: Sequence[str]) -> str:
    return "Public completion contract for this task:\n\n" + "".join(f"- {item}\n" for item in clauses)


def _wrap(key: str, value: str, indent: int = 8) -> str:
    line = f'{" " * indent}"{key}": {json.dumps(value)},\n'
    if len(line) <= 130:
        return line
    words, chunks, current = value.split(" "), [], ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if len(candidate) > 88 and current:
            chunks.append(current + " ")
            current = word
        else:
            candidate_ok = candidate
            current = candidate_ok
    chunks.append(current)
    pad = " " * (indent + len(key) + 5)
    body = ("\n" + pad).join(json.dumps(chunk) for chunk in chunks)
    return f'{" " * indent}"{key}": ({body}),\n'


def roster_entry(facts: Facts, projection: Projection) -> str:
    kind = family(facts)
    objects = set(targets(facts))
    skill = SKILLS.get(kind) or ("multi_object_pick_and_place" if len(objects) > 1 and logical(facts)[0] == "all"
                                 else "pick_and_place")
    if kind == "on_surface":
        skill = "multi_object_place_on_surface" if len(objects) > 1 and logical(facts)[0] == "all" else "place_on_surface"
    if kind == "in_container" and logical(facts)[0] == "choose":
        skill = "counted_pick_and_place"
    return (f'    "{facts.task}": {{\n'
            f'        "episode_length_s": {facts.episode_s}, "native_action_limit": {facts.native_limit}, '
            f'"skill": "{skill}",\n'
            + _wrap("instruction", facts.instruction) + _wrap("official_success", projection.official_success)
            + "    },\n")


def _existing_roster(text: str) -> dict[str, str]:
    """The current roster's entry text by task, byte for byte."""
    entries, current, lines = {}, None, []
    for line in text.splitlines(keepends=True):
        match = re.match(r'^    "(\w+Task)": \{$', line.rstrip("\n"))
        if match:
            current, lines = match.group(1), [line]
            continue
        if current is not None:
            lines.append(line)
            if line == "    },\n":
                entries[current] = "".join(lines)
                current = None
    return entries


# ------------------------------------------------------------------------------------------------ driver
@dataclass
class Output:
    files: dict[str, str]
    reviews: dict[str, dict[str, Any]]
    regression: dict[str, Any]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=1, ensure_ascii=False) + "\n"


def verify_source(source: Path) -> None:
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=False)
    if head.stdout.strip() != REVISION or dirty.stdout.strip():
        raise SystemExit(f"RoboLab source must be the clean pin {REVISION}; found {head.stdout.strip()!r} "
                         f"dirty={bool(dirty.stdout.strip())}")


def generate(source: Path, root: Path = PROJECT_ROOT) -> Output:
    facts = load_facts(source)
    scan = _load_json(root / ASSET_SCAN)
    if scan.get("schema") != SCAN_SCHEMA or scan.get("revision") != REVISION:
        raise SystemExit(f"{ASSET_SCAN} is not a scan of {REVISION}; run scan-assets")
    catalog = {item["usd_path"]: item for item in _load_json(source / "assets/objects/object_catalog.json")}
    registry = _load_json(root / REGISTRY)
    reviewed = set(registry["tasks"])
    roster_text = (root / ROSTER_MODULE).read_text(encoding="utf-8")
    existing_entries = _existing_roster(roster_text)
    files: dict[str, str] = {}
    reviews: dict[str, dict[str, Any]] = {}
    pending: dict[str, Any] = {}
    entries: list[str] = []
    regression: dict[str, Any] = {}
    for task in sorted(facts):
        item = facts[task]
        projection = project(item)
        contract_path = f"{CONTRACTS}/{task}-completion-v1.md"
        review_path = f"{CONTRACTS}/reviews/{task}-review-v1.json"
        review = build_review(item, source, scan, catalog, projection)
        contract = render_contract(projection.clauses)
        if task in reviewed:
            regression[task] = {"contract": contract, "review": review, "roster": roster_entry(item, projection)}
            entries.append(existing_entries[task])
            # The reviewed record stays as committed; the page still shows this audit's compatibility.
            reviews[task] = dict(_load_json(root / review_path), family=review["family"],
                                 compatibility=review["compatibility"])
            continue
        existing = root / review_path
        review = merge_witness(review, _load_json(existing) if existing.is_file() else None)
        reviews[task] = review
        files[contract_path] = contract
        files[review_path] = _json_text(review)
        check_sha, method = predicate_binding(item, source)
        pending[task] = {
            "path": contract_path, "sha256": hashlib.sha256(contract.encode("utf-8")).hexdigest(),
            "clause_count": len(projection.clauses), "check_success_sha256": check_sha,
            "check_success_sha256_method": method, "verified_revisions": [REVISION], "verdict": PENDING_VERDICT,
            "semantic_review": review_path, "family": review["family"],
            "compatibility": review["compatibility"]["status"],
        }
        entries.append(roster_entry(item, projection))
    registry = dict(registry)
    registry["note"] = (
        "Generation v1 of the RoboLab-120 roster. `tasks` holds the reviewed contracts the resolver delivers (three "
        "admitted 2026-10-18); `pending_review` holds the contracts generated 2026-11-05 per predicate family, whose "
        "seed-0 witnesses are pending: the resolver withholds them (contract_review_pending) and launch preflight "
        "refuses the task until `ora dev robolab-tasks promote` moves a fully checked review into `tasks`. Public "
        "clauses state the termination's end conditions qualitatively; numeric limits, private targets and predicate "
        "weaknesses stay in the reviews.")
    registry["pending_review"] = {task: pending[task] for task in sorted(pending)}
    files[REGISTRY] = json.dumps(registry, indent=1, ensure_ascii=False) + "\n"
    begin, end = roster_text.index(ROSTER_BEGIN) + len(ROSTER_BEGIN), roster_text.index(ROSTER_END)
    files[ROSTER_MODULE] = roster_text[:begin] + "".join(entries) + roster_text[end:]
    files[TASK_PAGE] = render_page((root / TASK_PAGE).read_text(encoding="utf-8"), facts, reviews, reviewed)
    return Output(files=files, reviews=reviews, regression=regression)


def render_page(text: str, facts: Mapping[str, Facts], reviews: Mapping[str, Mapping[str, Any]],
                reviewed: set[str]) -> str:
    rows = ["| Task | Family | Episode (s) | Official limit | Contract | Compatibility | Notes |",
            "|---|---|---|---|---|---|---|"]
    for task in sorted(facts):
        item, review = facts[task], reviews[task]
        compat = review.get("compatibility") or {"status": "ready", "notes": []}
        codes = ", ".join(dict.fromkeys(note["code"] for note in compat.get("notes", [])
                                        + compat.get("blocked", []))) or "-"
        rows.append(f"| `{task}` | {review.get('family') or family(item)} | {item.episode_s} | {item.native_limit} | "
                    f"{'reviewed' if task in reviewed else 'pending witness'} | "
                    f"{compat['status'].replace('_', ' ')} | {codes} |")
    begin, end = text.index(PAGE_BEGIN) + len(PAGE_BEGIN), text.index(PAGE_END)
    return text[:begin] + "\n".join(rows) + "\n" + text[end:]


def _diff(label: str, ours: str, theirs: str) -> list[str]:
    import difflib

    return list(difflib.unified_diff(theirs.splitlines(), ours.splitlines(), f"committed/{label}",
                                     f"generated/{label}", lineterm=""))


def regression_report(output: Output, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for task, generated in output.regression.items():
        committed_contract = (root / f"{CONTRACTS}/{task}-completion-v1.md").read_text(encoding="utf-8")
        committed_review = _load_json(root / f"{CONTRACTS}/reviews/{task}-review-v1.json")
        roster = _existing_roster((root / ROSTER_MODULE).read_text(encoding="utf-8"))[task]
        review = generated["review"]
        report[task] = {
            "contract_identical": generated["contract"] == committed_contract,
            "contract_diff": _diff(f"{task}-completion-v1.md", generated["contract"], committed_contract),
            "roster_identical": generated["roster"] == roster,
            "roster_diff": _diff(f"{task} roster", generated["roster"], roster),
            "review_field_differences": {
                key: {"committed": committed_review.get(key), "generated": review.get(key)}
                for key in ("public_clauses", "conditions", "counterexamples", "source_dependencies")
                if committed_review.get(key) != review.get(key)},
            "check_basis_differences": {
                key: {"committed": committed_review["checks"][key]["basis"], "generated": review["checks"][key]["basis"]}
                for key in review["checks"] if committed_review["checks"].get(key, {}).get("basis")
                != review["checks"][key]["basis"]},
        }
    return report


# ------------------------------------------------------------------------------------------- asset scan
_SCAN_SCRIPT = r"""
import json, os, sys
from pxr import Sdf, Usd, UsdPhysics, UsdUtils
root, scenes = sys.argv[1], json.loads(sys.argv[2])
media = ('.mdl', '.png', '.jpg', '.jpeg', '.exr', '.hdr', '.tga', '.ktx', '.dds')
out = {}
for scene in scenes:
    path = os.path.join(root, 'assets/scenes', scene)
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(path))
    stage = Usd.Stage.Open(path)
    objects = {}
    for prim in stage.GetDefaultPrim().GetChildren():
        joints = rigid = kinematic = colliders = 0
        articulation = False
        for item in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            joints += item.IsA(UsdPhysics.Joint)
            articulation = articulation or item.HasAPI(UsdPhysics.ArticulationRootAPI)
            if item.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid += 1
                kinematic += bool(UsdPhysics.RigidBodyAPI(item).GetKinematicEnabledAttr().Get())
            colliders += item.HasAPI(UsdPhysics.CollisionAPI)
        objects[prim.GetName()] = {'joints': joints, 'articulation_root': articulation, 'rigid_bodies': rigid,
                                   'kinematic_bodies': kinematic, 'colliders': colliders}
    remote = [u for u in unresolved if u.startswith('http')]
    local = [os.path.relpath(u, root) if u.startswith(root) else u for u in unresolved if not u.startswith('http')]
    out[scene] = {
        'layers': len(layers),
        'remote_geometry': sorted(u for u in remote if not u.lower().endswith(media)),
        'remote_render_assets': len([u for u in remote if u.lower().endswith(media)]),
        'local_missing_geometry': sorted(u for u in local if not u.lower().endswith(media)),
        'unresolved_render_assets': sorted(u for u in local if u.lower().endswith(media)),
        'objects': objects}
print(json.dumps(out))
"""


def scan_assets(source: Path, python: Path, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Compose every roster scene with the interpreter's USD libraries (no Kit) and record what it needs."""
    facts = load_facts(source)
    scenes = sorted({item.scene for item in facts.values()})
    probe = subprocess.run([str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                           capture_output=True, text=True, check=True)
    libs = sorted(Path(probe.stdout.strip()).glob("isaacsim/extscache/omni.usd.libs-*"))
    if not libs:
        raise SystemExit(f"{python} has no omni.usd.libs extension")
    environment = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "LANG": "C.UTF-8",
                   "LD_LIBRARY_PATH": str(libs[-1] / "bin"), "PYTHONPATH": str(libs[-1])}
    result = subprocess.run([str(python), "-c", _SCAN_SCRIPT, str(source.resolve()), json.dumps(scenes)],
                            capture_output=True, text=True, env=environment, check=False)
    if result.returncode != 0:
        raise SystemExit(f"asset scan failed: {result.stderr[-2000:]}")
    record = {"schema": SCAN_SCHEMA, "revision": REVISION,
              "method": "pxr UsdUtils.ComputeAllDependencies and Usd.Stage.Open of each roster scene with the "
                        "interpreter's omni.usd.libs (no Kit, no Omniverse client: http(s) assets stay unresolved)",
              "scanned_at": datetime.date.today().isoformat(),
              "scenes": json.loads(result.stdout.strip().splitlines()[-1])}
    (root / ASSET_SCAN).write_text(_json_text(record), encoding="utf-8")
    return record


# ------------------------------------------------------------------------------------------------ promote
def promote(tasks: Sequence[str] | None, *, root: Path = PROJECT_ROOT, reviewer: str, today: str) -> list[str]:
    """Move pending reviews whose nine checks passed into ``tasks``; return the promoted tasks."""
    from ora.agent.composition.task_contract.review import render_public_contract, review_errors

    registry = _load_json(root / REGISTRY)
    pending = registry.get("pending_review") or {}
    chosen = list(tasks) if tasks else sorted(pending)
    promoted = []
    for task in chosen:
        entry = pending.get(task)
        if entry is None:
            raise SystemExit(f"{task} is not pending review")
        path = root / entry["semantic_review"]
        review = _load_json(path)
        if (review.get("compatibility") or {}).get("status") == "blocked":
            print(f"{task}: blocked ({review['compatibility'].get('blocked')}); not promoted")
            continue
        candidate = dict(review, status="reviewed", reviewer=reviewer, reviewed_at=today)
        errors = review_errors(candidate)
        if render_public_contract(candidate) != (root / entry["path"]).read_text(encoding="utf-8"):
            errors.append("public_projection_differs_from_review")
        if errors:
            print(f"{task}: not promoted: {', '.join(errors)}")
            continue
        path.write_text(_json_text(candidate), encoding="utf-8")
        registry["tasks"][task] = {key: value for key, value in entry.items()
                                   if key not in ("family", "compatibility")} | {"verdict": REVIEWED_VERDICT}
        del pending[task]
        promoted.append(task)
    registry["tasks"] = dict(sorted(registry["tasks"].items()))
    (root / REGISTRY).write_text(json.dumps(registry, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return promoted


# --------------------------------------------------------------------------------------------- list tasks
def roster_rows(root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    from ora.environment.benchmarks.robolab import TASKS

    registry = _load_json(root / REGISTRY)
    rows = []
    for task, entry in sorted(TASKS.items()):
        bound = registry["tasks"].get(task) or registry.get("pending_review", {}).get(task)
        review = _load_json(root / bound["semantic_review"]) if bound else {}
        compat = review.get("compatibility") or {}
        rows.append({"task": task, "instruction": entry["instruction"], "episode_length_s": entry["episode_length_s"],
                     "native_action_limit": entry["native_action_limit"], "skill": entry["skill"],
                     "contract": "reviewed" if task in registry["tasks"] else (
                         "pending_review" if bound else "none"),
                     "compatibility": compat.get("status", "not_recorded"),
                     "notes": [note["code"] for note in compat.get("notes", []) + compat.get("blocked", [])]})
    return rows


def _write(files: Mapping[str, str], root: Path, *, check: bool) -> list[str]:
    drift = []
    for relative, text in sorted(files.items()):
        path = root / relative
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            continue
        drift.append(relative)
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    return drift


def main(argv: list[str] | None = None) -> int:
    # No fixed prog: `ora dev robolab-tasks` and `ora spec robolab tasks` both name themselves (passthrough.call).
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("generate", "write (or --check) the roster, contracts, reviews, registry and task page"),
                       ("regression", "diff this generator against the reviewed tasks' committed text"),
                       ("scan-assets", "compose each roster scene with pxr and write the asset scan")):
        command = commands.add_parser(name, help=text)
        command.add_argument("--source-root", type=Path, default=os.environ.get("ROBOLAB_ROOT"),
                             help="the pinned RoboLab checkout (default: $ROBOLAB_ROOT)")
        if name == "generate":
            command.add_argument("--check", action="store_true", help="report drift and exit 1 instead of writing")
        if name == "scan-assets":
            command.add_argument("--python", type=Path, default=os.environ.get("ROBOLAB_PYTHON"),
                                 help="the RoboLab interpreter (default: $ROBOLAB_PYTHON)")
    promote_parser = commands.add_parser("promote", help="move fully checked reviews from pending_review to tasks")
    promote_parser.add_argument("--task", action="append", default=[])
    promote_parser.add_argument("--reviewer", default="Seed-0 witness review (ora qualify robolab contract-witness)")
    commands.add_parser("tasks", help="list the roster with contract and compatibility state")
    args = parser.parse_args(argv)
    root = PROJECT_ROOT
    if args.command == "tasks":
        for row in roster_rows(root):
            print(json.dumps(row, ensure_ascii=False))
        return 0
    if args.command == "promote":
        promoted = promote(args.task or None, root=root, reviewer=args.reviewer,
                           today=datetime.date.today().isoformat())
        print(json.dumps({"promoted": promoted}))
        if promoted:
            print("Next: ora dev robolab-tasks generate (the task page), ./scripts/ora lock, and "
                  "task_contracts.robolab.task_count in tests/data/default-expectations.json.", file=sys.stderr)
        return 0
    if args.source_root is None:
        parser.error("--source-root (or ROBOLAB_ROOT) is required")
    source = Path(args.source_root).expanduser().resolve()
    verify_source(source)
    if args.command == "scan-assets":
        if args.python is None:
            parser.error("--python (or ROBOLAB_PYTHON) is required")
        record = scan_assets(source, Path(args.python).expanduser(), root)
        print(json.dumps({"path": ASSET_SCAN, "scenes": len(record["scenes"])}))
        return 0
    output = generate(source, root)
    if args.command == "regression":
        print(json.dumps(regression_report(output, root), indent=1, ensure_ascii=False))
        return 0
    drift = _write(output.files, root, check=args.check)
    print(json.dumps({"drift" if args.check else "written": drift}, indent=1))
    return 1 if args.check and drift else 0


__all__ = ["generate", "load_facts", "main", "project", "promote", "regression_report", "roster_rows",
           "scan_assets", "witness_plan"]
