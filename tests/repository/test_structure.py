"""The repository structure contract as executable rules.

`docs/developer/repository-structure.md` is the target structure of the ORA
restructure: section 2 states nine import rules (L1-L9), section 8 the test
tree and section 9 the package READMEs. This module checks each of them on the
source as it stands, statically: it parses files with `ast` and imports no
`ora` module, so a broken package cannot hide its own violations.

An import edge is counted the way the structure review counted it: a
module-level import, an import inside a function or method (lazy) and an import
under `if TYPE_CHECKING:` are all edges; a module name in a string (`python -m`
arguments, `"module:attr"` entry points, `importlib.import_module("...")`) is
not. `from X import Y` points at `X.Y` when that is a module, else at `X`; a
module importing itself is not an edge.

The rules are written against the target package names (`ora.app`,
`ora.orchestrator.backends.registry`).
Where a target package does not exist yet the rule holds trivially for it and
becomes real the moment the package appears. A rule the current tree breaks is
marked `xfail(strict=True)` with the lane that fixes it; the lane flips the
marker when its last violation is gone. Every assertion prints the violations
as `file:line -> target`, so a failing rule is also that lane's worklist.
"""

from __future__ import annotations

import ast
import functools
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# The contract as data
# --------------------------------------------------------------------------

#: L1: rank of each layered top-level package of `ora`. A module may import a
#: package of lower rank, or its own package; `agent` and `environment` share a
#: rank and import nothing of each other.
L1_LAYERS = {"contracts": 0, "agent": 1, "environment": 1, "orchestrator": 2, "app": 3}
#: L1: packages beside the layers. They import no `ora` package themselves; the
#: value names who may import them (None: any package). `ora` is the root
#: package module. `robot_client` is installed into the sandbox by the
#: orchestrator (X1), never by the agent. `cell`, the real-cell kit, reaches a
#: robot's backend only by its `"ora.environment.backends.<env>.cell_adapter"`
#: string; the policy side never imports it.
L1_SIDE: dict[str, tuple[str, ...] | None] = {
    "ora": None, "policy": None, "version": None,
    "robot_client": ("orchestrator", "app"),
    "cell": ("environment", "orchestrator", "app"),
}

#: L3: orchestrator modules owned by one simulator backend: the
#: `ora.orchestrator.backends` package, and the RoboDojo files pinned in the
#: orchestrator root by their hash-bound paths (section 12), recognised by the
#: backend prefix of their name.
L3_BACKEND_PACKAGE = "ora.orchestrator.backends"
L3_BACKEND_PREFIXES = ("robodojo_", "robotwin_", "libero_", "robocasa_", "robolab_")
#: L3: the one backend module the orchestrator core may import: the registry of
#: `{runtime.kind: "module:attr"}` strings.
L3_REGISTRY = "ora.orchestrator.backends.registry"

#: L5: what the agent never imports.
L5_FORBIDDEN = ("ora.environment.backends", "ora.orchestrator", "ora.app")

#: L6: rank of each unit inside `ora.agent` (the first name below `ora.agent`).
#: A unit imports only units of lower rank; an edge to a unit of the same rank
#: is a peer edge and as forbidden as an upward one. `sessions` and `scripted`
#: are not named by the contract; they keep the ranks the structure review gave
#: them. `__init__` is the `ora.agent` package module itself. `code_mode` (code mode's
#: sandbox, contract and robot library) sits above the tools it calls, below Pi.
#: `toolkit.host` (`ora/agent/toolkit/host/`, the trusted host the toolkit's tools run
#: under, with the tool runtime frame in `host/runtime/`) is a unit of its own: it imports
#: the rest of the toolkit and `composition`, and nothing else in the toolkit imports it.
L6_AGENT_RANKS = {
    "contracts": 0, "error_classes": 0, "__init__": 0,
    "evidence": 1, "sessions": 1,
    "providers": 2, "model_services": 2,
    "composition": 3, "toolkit": 3,
    "toolkit.host": 4,
    "memory": 6, "code_mode": 6,
    "context": 7,
    "session_core": 8, "scripted": 8,
    "pi": 9, "tui": 9,
}
#: L6: same-rank edges the contract allows, with the reason.
L6_ALLOWED = {
    ("__init__", "contracts"): "A0: the agent package's public facade re-exports its contracts",
}

#: L8: what a module under tests/ may import from this repository: the packages
#: under test (ora, viewer, the shipped examples) and the shared support package.
#: Anything else first-party (another test module, a conftest, any other
#: top-level directory) is a violation; third-party imports are not checked.
L8_ALLOWED = ("ora", "viewer", "examples", "tests.support")

#: L9: package modules that keep a module-level import anyway, with the reason.
#: A parent package is one that has a sub-package.
L9_ALLOWED = {
    ("ora.agent", "ora.agent.contracts"):
        "A0: the agent package's public facade re-exports its zero-dependency contracts",
    ("ora.environment.backends.robodojo", "ora.environment.backends.robodojo.session"):
        "section 12: the file is a RoboDojo worker kernel member, hash-bound by its qualification",
}

#: Section 8: top-level entries of tests/ in the target tree. Test modules live
#: in the mirror roots below, in tests/agent_core/ (decision 5) and in
#: tests/repository/.
TEST_TREE = {
    "README.md", "conftest.py", "__init__.py",
    "support", "data", "ora", "viewer", "agent_core", "repository",
}
#: Section 8: mirror roots, tests directory -> code directory.
TEST_MIRRORS = {"tests/ora": "ora", "tests/viewer": "viewer"}

#: Section 9: the directories that each carry a README.md.
READMES = (
    "ora", "ora/contracts", "ora/environment",
    "ora/environment/backends/libero_pro", "ora/environment/backends/robotwin",
    "ora/environment/backends/robodojo", "ora/environment/backends/robocasa",
    "ora/environment/backends/robolab",
    "ora/environment/benchmarks", "ora/environment/controllers", "ora/environment/planners",
    "ora/agent", "ora/agent/composition", "ora/agent/context", "ora/agent/code_mode",
    "ora/orchestrator", "ora/orchestrator/backends", "ora/app", "ora/policy",
    "ora/robot_client", "ora/cell", "ora-pi", "viewer", "tests", "scripts", "examples", "internal",
)

# --------------------------------------------------------------------------
# The import graph
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    source: str  # importing module
    target: str  # imported module, resolved as described in the module docstring
    path: str  # repository-relative file of the importing module
    line: int
    kind: str  # "top", "lazy" or "typing"

    def __str__(self) -> str:
        return f"{self.path}:{self.line} -> {self.target} ({self.kind})"


def _module_name(relative: Path) -> str:
    parts = relative.with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _python_files(top: str) -> list[Path]:
    base = ROOT / top
    return sorted(
        path for path in base.rglob("*.py")
        if "__pycache__" not in path.parts and "node_modules" not in path.parts
    )


@functools.lru_cache(maxsize=None)
def _modules(top: str) -> dict[str, Path]:
    """Module name -> file for every Python file under one top-level directory."""

    return {_module_name(path.relative_to(ROOT)): path for path in _python_files(top)}


@functools.lru_cache(maxsize=None)
def _first_party() -> frozenset[str]:
    """Top-level importable names of this repository (directories and modules at its root)."""

    names = set()
    for entry in ROOT.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_dir() and entry.name.isidentifier():
            names.add(entry.name)
        elif entry.suffix == ".py":
            names.add(entry.stem)
    return frozenset(names)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _raw_imports(tree: ast.AST):
    """Yield (node, kind) for every import statement, with where it runs."""

    def visit(node: ast.AST, kind: str):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                yield child, kind
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                yield from visit(child, "lazy")
            elif isinstance(child, ast.If) and kind != "lazy" and _is_type_checking(child.test):
                for statement in child.body:
                    if isinstance(statement, (ast.Import, ast.ImportFrom)):
                        yield statement, "typing"
                    yield from visit(statement, "typing")
                for statement in child.orelse:
                    if isinstance(statement, (ast.Import, ast.ImportFrom)):
                        yield statement, kind
                    yield from visit(statement, kind)
            else:
                yield from visit(child, kind)

    yield from visit(tree, "top")


@functools.lru_cache(maxsize=None)
def _edges(top: str) -> tuple[Edge, ...]:
    """Every first-party import edge whose importing file lives under `top`.

    Targets inside `ora` and `viewer` resolve to the module `from X import Y`
    reaches; other first-party targets (tests, examples, ...) resolve the same
    way against the files under their own directory.
    """

    known: dict[str, Path] = {}
    for directory in ("ora", "viewer", "tests", "examples"):
        if (ROOT / directory).is_dir():
            known.update(_modules(directory))
    first_party = _first_party()
    found: list[Edge] = []
    for name, path in _modules(top).items():
        relative = path.relative_to(ROOT).as_posix()
        if top == "tests" and relative.startswith("tests/data/"):
            continue
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node, kind in _raw_imports(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                base = node.module or ""
                if node.level:
                    anchor = package.split(".")
                    anchor = anchor[: len(anchor) - node.level + 1]
                    base = ".".join([*anchor, *([base] if base else [])])
                targets = []
                for alias in node.names:
                    candidate = f"{base}.{alias.name}"
                    targets.append(candidate if candidate in known else base)
            for target in dict.fromkeys(targets):  # one edge per statement and target
                if not target or target.split(".")[0] not in first_party or target == name:
                    continue
                found.append(Edge(name, target, relative, node.lineno, kind))
    return tuple(found)


def _within(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def _report(violations: list, what: str) -> str:
    lines = sorted({str(item) for item in violations})
    return f"{len(lines)} {what}:\n" + "\n".join(lines)


# --------------------------------------------------------------------------
# L1-L9
# --------------------------------------------------------------------------


def _ora_package(module: str) -> str | None:
    parts = module.split(".")
    if parts[0] != "ora":
        return None
    return "ora" if len(parts) == 1 else parts[1]


def l1_violations() -> list[Edge]:
    found = []
    for edge in _edges("ora"):
        source, target = _ora_package(edge.source), _ora_package(edge.target)
        if target is None or source == target:
            continue  # viewer and other first-party targets belong to L2
        if source in L1_SIDE:
            found.append(edge)  # a side package imports no other ora package
        elif target in L1_SIDE:
            importers = L1_SIDE[target]
            if importers is not None and source not in importers:
                found.append(edge)
        elif source not in L1_LAYERS or target not in L1_LAYERS:
            found.append(edge)  # a package the contract does not place
        elif L1_LAYERS[target] >= L1_LAYERS[source]:
            found.append(edge)
    return found


def l2_violations() -> list[Edge]:
    return [edge for edge in _edges("ora") if _within(edge.target, "viewer")]


def _orchestrator_backend(module: str) -> bool:
    if _within(module, L3_BACKEND_PACKAGE):
        return True
    parts = module.split(".")
    return len(parts) == 3 and parts[1] == "orchestrator" and parts[2].startswith(L3_BACKEND_PREFIXES)


def l3_violations() -> list[Edge]:
    return [
        edge for edge in _edges("ora")
        if _within(edge.source, "ora.orchestrator") and not _orchestrator_backend(edge.source)
        and _orchestrator_backend(edge.target) and not _within(edge.target, L3_REGISTRY)
    ]


#: L4: the isolated-interpreter entry points are the one place environment core
#: loads a backend: they run inside the simulator's interpreter and are what the
#: definitions.py ``module:attr`` strings resolve to (worker.py is also a pinned
#: RoboDojo worker-kernel member, section 12).
L4_BACKEND_LOADERS = ("ora.environment.worker", "ora.environment.replay_worker")


#: L4: the motion planner workers, each started by module name under its own
#: interpreter (``controllers/motion_planner.py``): nothing outside a planner
#: imports one, and a planner imports no simulator backend.
L4_PLANNERS = "ora.environment.planners"


def l4_violations() -> list[Edge]:
    backends = "ora.environment.backends"
    return [
        edge for edge in _edges("ora")
        if (_within(edge.source, "ora.environment") and not _within(edge.source, backends)
            and not _within(edge.source, L4_PLANNERS) and edge.source not in L4_BACKEND_LOADERS
            and _within(edge.target, backends))
        or (not _within(edge.source, L4_PLANNERS) and _within(edge.target, L4_PLANNERS))
        or (_within(edge.source, L4_PLANNERS) and _within(edge.target, backends))
    ]


def l5_violations() -> list[Edge]:
    return [
        edge for edge in _edges("ora")
        if _within(edge.source, "ora.agent")
        and any(_within(edge.target, forbidden) for forbidden in L5_FORBIDDEN)
    ]


def _agent_unit(module: str) -> str:
    parts = module.split(".")
    if parts[2:4] == ["toolkit", "host"]:
        return "toolkit.host"
    return "__init__" if len(parts) == 2 else parts[2]


def l6_violations() -> list[str]:
    found = []
    for edge in _edges("ora"):
        if not (_within(edge.source, "ora.agent") and _within(edge.target, "ora.agent")):
            continue
        source, target = _agent_unit(edge.source), _agent_unit(edge.target)
        if source == target or (source, target) in L6_ALLOWED:
            continue
        if source not in L6_AGENT_RANKS or target not in L6_AGENT_RANKS:
            found.append(f"{edge}: agent unit not ranked by the contract")
        elif L6_AGENT_RANKS[target] >= L6_AGENT_RANKS[source]:
            found.append(f"{edge}: {source}({L6_AGENT_RANKS[source]}) -> {target}({L6_AGENT_RANKS[target]})")
    return found


def l8_violations() -> list[Edge]:
    return [
        edge for edge in _edges("tests")
        if not any(_within(edge.target, allowed) for allowed in L8_ALLOWED)
    ]


def cross_test_violations() -> list[Edge]:
    return [
        edge for edge in _edges("tests")
        if _within(edge.target, "tests") and not _within(edge.target, "tests.support")
    ]


def l9_violations() -> list[Edge]:
    modules = _modules("ora")
    packages = {name for name, path in modules.items() if path.name == "__init__.py"}
    parents = {
        name for name in packages
        if any(other != name and other.rpartition(".")[0] == name for other in packages)
    }
    return [
        edge for edge in _edges("ora")
        if edge.source in parents and edge.kind == "top" and _within(edge.target, "ora")
        and (edge.source, edge.target) not in L9_ALLOWED
    ]


def test_l1_layers_import_only_downward() -> None:
    """app > orchestrator > {agent, environment} > contracts; the side packages import no ora package."""

    found = l1_violations()
    assert not found, _report(found, "edges against the layer order")


def test_l2_ora_never_imports_viewer() -> None:
    found = l2_violations()
    assert not found, _report(found, "imports of viewer from ora")


def test_l3_orchestrator_core_reaches_backends_only_through_the_registry() -> None:
    found = l3_violations()
    assert not found, _report(found, "orchestrator-core imports of backend modules")


def test_l4_environment_core_never_imports_its_backends() -> None:
    found = l4_violations()
    assert not found, _report(found, "environment-core imports of environment.backends, or planner imports")


def test_l5_agent_never_imports_simulator_backends_orchestrator_or_app() -> None:
    found = l5_violations()
    assert not found, _report(found, "agent imports of environment.backends, orchestrator or app")


def test_l6_agent_units_import_only_lower_layers() -> None:
    """pi/tui > session_core > context > memory > toolkit.host > {composition, toolkit}
    > {providers, model_services} > evidence > contracts/error_classes."""

    found = l6_violations()
    assert not found, _report(found, "upward or peer edges between agent units")


def test_l8_tests_import_only_ora_viewer_and_support() -> None:
    found = l8_violations()
    assert not found, _report(found, "first-party imports outside ora, viewer, examples and tests.support")


def test_l9_parent_packages_import_nothing_eagerly() -> None:
    """A package with sub-packages is a lazy facade, so importing one leaf does not load its siblings."""

    found = l9_violations()
    assert not found, _report(found, "module-level imports in parent package modules")


def test_l9_agent_packages_are_lazy_facades() -> None:
    """Lane A's half of L9: no ora.agent package module imports its sub-packages eagerly.

    ``ora.agent.composition``, ``ora.agent.context`` and ``ora.agent.toolkit.host.runtime`` are
    lazy facades; the agent package itself keeps its one allowed re-export.
    """

    found = [edge for edge in l9_violations() if _within(edge.source, "ora.agent")]
    assert not found, _report(found, "module-level imports in ora.agent package modules")


# --------------------------------------------------------------------------
# Sections 8 and 9
# --------------------------------------------------------------------------


def test_no_cross_test_imports() -> None:
    """A module under tests/ imports another tests module only from tests/support/ (rule L8)."""

    found = cross_test_violations()
    assert not found, _report(found, "imports of test modules or conftest files")


def test_tests_mirror_code_tree() -> None:
    """tests/ora/<path>/ mirrors ora/<path>/ and tests/viewer/ mirrors viewer/; nothing else sits at the top."""

    found = []
    tests = ROOT / "tests"
    for entry in sorted(tests.iterdir()):
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        if entry.name not in TEST_TREE:
            found.append(f"tests/{entry.name}: not in the target test tree")
    for mirror, code in TEST_MIRRORS.items():
        base = ROOT / mirror
        if not base.is_dir():
            continue
        for directory in sorted(path for path in base.rglob("*") if path.is_dir()):
            if directory.name == "__pycache__":
                continue
            relative = directory.relative_to(base)
            if not (ROOT / code / relative).is_dir():
                where = directory.relative_to(ROOT).as_posix()
                found.append(f"{where}: no {code}/{relative.as_posix()}/ to mirror")
    assert not found, "\n".join(found)


def _is_fixture(expression: ast.expr, local_names: set[str]) -> bool:
    """``pytest.fixture``, or a name imported from pytest as ``fixture``, called or not."""

    target = expression.func if isinstance(expression, ast.Call) else expression
    return (
        isinstance(target, ast.Attribute) and target.attr == "fixture"
        and isinstance(target.value, ast.Name) and target.value.id == "pytest"
    ) or (isinstance(target, ast.Name) and target.id in local_names)


def support_fixtures() -> list[str]:
    found = []
    for path in _python_files("tests/support"):
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        local_names = {
            alias.asname or alias.name
            for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "pytest"
            for alias in node.names if alias.name == "fixture"
        }
        decorators = {
            id(decorator) for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for decorator in node.decorator_list
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                _is_fixture(decorator, local_names) for decorator in node.decorator_list
            ):
                found.append(f"{relative}:{node.lineno} {node.name}")
            elif isinstance(node, ast.Call) and id(node) not in decorators and _is_fixture(node.func, local_names):
                found.append(f"{relative}:{node.lineno} pytest.fixture(...) call")
    return found


def test_shared_fixtures_live_in_conftests() -> None:
    """tests/support holds plain helpers; a shared fixture is registered by the narrowest conftest.py.

    A fixture imported into a test module registers there under the name it is
    imported as, so an aliased autouse fixture turns autouse silently; the
    module asks for a conftest fixture by name or in ``pytestmark`` instead.
    """

    found = support_fixtures()
    assert not found, _report(found, "fixtures defined in tests/support")


def test_every_package_has_readme() -> None:
    missing = [
        f"{directory}/README.md" for directory in READMES
        if (ROOT / directory).is_dir() and not (ROOT / directory / "README.md").is_file()
    ]
    assert not missing, f"{len(missing)} READMEs missing:\n" + "\n".join(missing)
