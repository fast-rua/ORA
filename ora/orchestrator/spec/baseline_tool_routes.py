"""Where each default tool is served and how its proposal reaches the backend.

The tool half of `ora baselines --explain`, derived from code:

* the tool runtime is the class the tool-runtime factory registers for the
  surface's runtime id (its literal table, located with `ast`); on a surface the
  toolkit serves (`toolkit_dispatch.served`) every tool call runs through
  `toolkit_dispatch.Dispatch.run`;
* a tool's handler is its toolkit registry entry, `run` in
  `ora/agent/toolkit/tools/<name>/tool.py` (`ora.agent.toolkit.registry`);
* a state-changing tool's proposal comes from a static walk of that entry and of
  the toolkit functions it calls, which records the action envelopes the walk
  can build; the host adapter that admits and persists it is the
  `api.adapters["submit_action"]` assignment `Dispatch._run` makes for that tool;
* the Supervisor executor for each action schema is read from the branches of
  `episode.actions.dispatch_action`, the transport client from the environment
  factory, and the backend session class from the isolated worker.

Where the code admits no exact answer the record says so and gives the
generic route. Backend session modules are parsed, never imported. Nothing is
written or started; the only code run is imports of host-safe project modules.
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
import inspect
import re
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

from ora.agent.composition.surface.tool_surface import ResolvedToolSurface
from ora.agent.composition.surface import tool_names
from ora.agent.toolkit.registry import load_tools
from ora.agent.toolkit.host.runtime import factory as tool_factory
from ora.agent.toolkit.host.runtime import dispatch as toolkit_dispatch
from ora.environment import factory as environment_factory
from ora.environment import worker as environment_worker
from ora.environment.session import DescribedEnvironmentSession

from ..episode import actions as episode_actions
from ..episode.loop import EpisodeLoop

# Tool entries reach their proposal through helpers; past this depth a walk is
# exploring unrelated utilities, not the submission path.
MAX_WALK_DEPTH = 8
# Only toolkit code builds a proposal; the walk follows nothing else.
_WALKED_PACKAGE = "ora.agent.toolkit."
_PROJECT_PACKAGE = "ora."
_NEVER_IMPORTED = "ora.environment.backends."


class BaselineExplainError(RuntimeError):
    """The explanation cannot be derived, or the request is invalid."""


# -- small code-location helpers ------------------------------------------------

def _relative(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def dotted(value: Any) -> str:
    return f"{value.__module__}:{value.__qualname__}"


@functools.lru_cache(maxsize=None)
def _parsed(function: Callable[..., Any]) -> tuple[ast.AST, int]:
    """The function's own AST node and the file line its source starts on."""
    lines, start = inspect.getsourcelines(function)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    return tree.body[0], start


def location(function: Callable[..., Any], root: Path) -> str:
    """`path:line` of the `def` line itself, past any decorators."""
    node, start = _parsed(function)
    return f"{_relative(inspect.getsourcefile(function), root)}:{start + node.lineno - 1}"


def callable_record(function: Callable[..., Any], root: Path) -> dict[str, str]:
    return {"callable": dotted(function), "location": location(function, root)}


def _resolve_import(node: ast.ImportFrom, package: str | None) -> str:
    return importlib.util.resolve_name("." * node.level + (node.module or ""), package)


def _module_file(module: str, root: Path) -> Path:
    """Locate a backend module by name without importing it."""
    return root / (module.replace(".", "/") + ".py")


def _local_imports(function: Callable[..., Any], node: ast.AST) -> dict[str, tuple[str, str]]:
    """Function-local ``from x import y`` names, resolved only when used."""
    package = function.__module__.rpartition(".")[0] or None
    found: dict[str, tuple[str, str]] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom):
            base = _resolve_import(child, package)
            for alias in child.names:
                found[alias.asname or alias.name] = (base, alias.name)
    return found


def _import_member(base: str, name: str) -> Any:
    """Import a project member a function imports locally; never a backend module."""
    if not base.startswith(_PROJECT_PACKAGE) or base.startswith(_NEVER_IMPORTED):
        return None
    try:
        module = importlib.import_module(base)
        if hasattr(module, name):
            return getattr(module, name)
        return importlib.import_module(f"{base}.{name}")
    except Exception:  # noqa: BLE001 - an unimportable member resolves nothing
        return None


def _lookup(function: Callable[..., Any], imports: Mapping[str, tuple[str, str]], name: str) -> Any:
    if name in imports:
        return _import_member(*imports[name])
    return function.__globals__.get(name)


# -- the tool runtime, the toolkit entries and the host's submit adapters --------

def _tool_runtime_class(surface: ResolvedToolSurface) -> type:
    """The class the factory's literal ``{runtime_id: class}`` table registers for this surface."""
    node, _start = _parsed(tool_factory.build_tool_runtime)
    for child in ast.walk(node):
        if (isinstance(child, ast.Dict) and len(child.keys) > 1
                and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in child.keys)
                and all(isinstance(v, ast.Name) for v in child.values)):
            for key, value in zip(child.keys, child.values):
                if key.value == surface.runtime_id:
                    return tool_factory.build_tool_runtime.__globals__[value.id]
    raise BaselineExplainError(f"no tool runtime registered for {surface.runtime_id}")


def _entry_name(name: str) -> str:
    """The registry directory a surface tool name runs (``robot.submit_action`` -> ``robot_submit_action``)."""
    return toolkit_dispatch.ROBOT_TOOLS.get(name, tool_names.canonical(name))


def _truth(test: ast.AST, name: str, function: Callable[..., Any],
           imports: Mapping[str, tuple[str, str]]) -> bool | None:
    """``name == "x"``, ``name in NAMES`` and their ``not``/``and``/``or`` for one tool; None if not one."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _truth(test.operand, name, function, imports)
        return None if inner is None else not inner
    if isinstance(test, ast.BoolOp):
        values = [_truth(value, name, function, imports) for value in test.values]
        decisive = isinstance(test.op, ast.Or)
        if decisive in values:
            return decisive
        return (not decisive) if all(value is not None for value in values) else None
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.left, ast.Name) and test.left.id == "name"):
        return None
    operator, right = test.ops[0], test.comparators[0]
    if isinstance(operator, (ast.Eq, ast.NotEq)) and isinstance(right, ast.Constant):
        return (name == right.value) == isinstance(operator, ast.Eq)
    if isinstance(operator, (ast.In, ast.NotIn)):
        if isinstance(right, (ast.Set, ast.Tuple, ast.List)):
            members = {item.value for item in right.elts if isinstance(item, ast.Constant)}
        elif isinstance(right, ast.Name):
            members = _lookup(function, imports, right.id)
        else:
            return None
        if not hasattr(members, "__contains__"):
            return None
        return (name in members) == isinstance(operator, ast.In)
    return None


def _submit_adapter(name: str, root: Path) -> tuple[dict[str, str] | None, bool]:
    """The host adapter ``Dispatch._run`` binds as ``submit_action`` for tool ``name``, and whether that is exact.

    Assignments run in order; the last one whose guarding conditions hold wins.
    """
    function = toolkit_dispatch.Dispatch._run
    node, _start = _parsed(function)
    imports = _local_imports(function, node)
    chosen, exact = None, True

    def adapter(value: ast.AST) -> Any:
        """The innermost call of the assigned value that is not a wrapper (``control_adapter``)."""
        calls = [child for child in ast.walk(value) if isinstance(child, ast.Call)
                 and isinstance(child.func, ast.Name) and child.func.id != "control_adapter"]
        return _lookup(function, imports, calls[0].func.id) if calls else None

    def walk(statements: list[ast.stmt], holds: bool | None) -> None:
        nonlocal chosen, exact
        for statement in statements:
            if isinstance(statement, ast.If):
                verdict = _truth(statement.test, name, function, imports)
                for branch, taken in ((statement.body, verdict), (statement.orelse, None if verdict is None
                                                                  else not verdict)):
                    if taken is False:
                        continue
                    walk(branch, None if holds is None or taken is None else holds)
                continue
            for field in ("body", "orelse", "finalbody"):
                if isinstance(getattr(statement, field, None), list):
                    walk(getattr(statement, field), holds)
            for handler in getattr(statement, "handlers", []):
                walk(handler.body, holds)
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "adapters" and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "submit_action"):
                found = adapter(statement.value)
                if holds and inspect.isfunction(found):
                    chosen, exact = callable_record(found, root), True
                else:  # an assignment that may or may not run, or one this walk cannot resolve
                    exact = False

    walk(list(getattr(node, "body", [])), True)
    return chosen, exact


# -- static proposal scan of a toolkit entry -------------------------------------

def _walked(value: Any) -> bool:
    return inspect.isfunction(value) and str(getattr(value, "__module__", "")).startswith(_WALKED_PACKAGE)


class _ProposalScan:
    """Collect the action envelopes a toolkit entry and the toolkit code it calls can build."""

    def __init__(self) -> None:
        self.found: set[tuple[str, str]] = set()
        self.unresolved = False
        self._seen: set[Any] = set()

    def function(self, function: Callable[..., Any], depth: int = 0) -> None:
        function = inspect.unwrap(function)
        code = getattr(function, "__code__", None)
        if code is None or code in self._seen or depth > MAX_WALK_DEPTH:
            return
        self._seen.add(code)
        node, _start = _parsed(function)
        imports = _local_imports(function, node)
        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                keys = {key.value: value for key, value in zip(child.keys, child.values)
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)}
                self._envelope(keys, function, imports)
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == "dict":
                    self._envelope({item.arg: item.value for item in child.keywords if item.arg},
                                   function, imports)
                target = self._target(child.func, function, imports)
                if target is not None:
                    self.function(target, depth + 1)

    def _envelope(self, keys: Mapping[str, ast.AST], function: Callable[..., Any],
                  imports: Mapping[str, tuple[str, str]]) -> None:
        if not {"schema", "action"} <= set(keys) or not isinstance(keys["action"], ast.Constant):
            return
        schemas = self._strings(keys["schema"], function, imports)
        if not schemas:
            self.unresolved = True
        for schema in schemas:
            self.found.add((keys["action"].value, schema))

    @staticmethod
    def _strings(node: ast.AST, function: Callable[..., Any], imports: Mapping[str, tuple[str, str]]) -> set[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name) and isinstance(_lookup(function, imports, node.id), str):
            return {_lookup(function, imports, node.id)}
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            owner = _lookup(function, imports, node.value.id)
            if isinstance(owner, ModuleType) and isinstance(getattr(owner, node.attr, None), str):
                return {getattr(owner, node.attr)}
        return set()

    @staticmethod
    def _target(func: ast.AST, function: Callable[..., Any], imports: Mapping[str, tuple[str, str]]) -> Any:
        value = None
        if isinstance(func, ast.Name):
            value = _lookup(function, imports, func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            owner = _lookup(function, imports, func.value.id)
            if isinstance(owner, ModuleType):
                value = getattr(owner, func.attr, None)
        value = inspect.unwrap(value) if callable(value) else value
        return value if _walked(value) else None


# -- the Supervisor and environment route ---------------------------------------

def _dispatch_branches(root: Path) -> list[dict[str, Any]]:
    """Read `dispatch_action`'s schema branches: which executor runs which proposal."""
    function = episode_actions.dispatch_action
    namespace = function.__globals__
    node, _start = _parsed(function)
    statement = next((item for item in node.body if isinstance(item, ast.If)), None)
    branches: list[dict[str, Any]] = []
    while isinstance(statement, ast.If):
        schemas: set[str] = set()
        test = statement.test
        for child in ast.walk(test.comparators[0] if isinstance(test, ast.Compare) else test):
            if isinstance(child, ast.Name) and isinstance(namespace.get(child.id), str):
                schemas.add(namespace[child.id])
        if not schemas:
            raise BaselineExplainError("dispatch_action branch names no action schema")
        branches.append({"schemas": sorted(schemas),
                         **_first_call(statement.body, namespace, root)})
        orelse = statement.orelse
        statement = orelse[0] if len(orelse) == 1 and isinstance(orelse[0], ast.If) else None
        if statement is None and orelse:
            branches.append({"schemas": None, **_first_call(orelse, namespace, root)})
    if not branches or branches[-1]["schemas"] is not None:
        raise BaselineExplainError("dispatch_action has no default executor branch")
    return branches


def _first_call(body: list[ast.stmt], namespace: Mapping[str, Any], root: Path) -> dict[str, str]:
    for statement in body:
        for child in ast.walk(statement):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                value = namespace.get(child.func.id)
                if inspect.isfunction(value):
                    record = callable_record(value, root)
                    return {"executor": record["callable"], "location": record["location"]}
    raise BaselineExplainError("dispatch_action branch calls no executor")


def _executor_for(schema: str, branches: list[dict[str, Any]]) -> str:
    return next(branch["executor"] for branch in branches
                if branch["schemas"] is None or schema in branch["schemas"])


def _builder_facts(runtime: Mapping[str, Any]) -> tuple[type, str]:
    """The transport client class and worker backend id the factory builds."""
    key = (runtime["kind"], runtime.get("transport", "in_process_direct"))
    builder = environment_factory._BUILDERS.get(key)
    if builder is None:
        raise BaselineExplainError(f"no environment builder for {key}")
    node, _start = _parsed(builder)
    client, backend = None, None
    for child in ast.walk(node):
        if (isinstance(child, ast.Return) and isinstance(child.value, ast.Call)
                and isinstance(child.value.func, ast.Name)):
            client = builder.__globals__.get(child.value.func.id)
            # A builder that only delegates (RoboDojo's returns
            # build_robodojo_worker_session(...)) is read through its delegate.
            if inspect.isfunction(client):
                node, _start = _parsed(client)
                client = None
                for inner in ast.walk(node):
                    if (isinstance(inner, ast.Return) and isinstance(inner.value, ast.Call)
                            and isinstance(inner.value.func, ast.Name)):
                        client = builder.__globals__.get(inner.value.func.id)
                break
    for child in ast.walk(node):
        if isinstance(child, ast.Dict):
            for item, value in zip(child.keys, child.values):
                if (isinstance(item, ast.Constant) and item.value == "backend"
                        and isinstance(value, ast.Constant)):
                    backend = value.value
    if not isinstance(client, type) or backend is None:
        raise BaselineExplainError(f"environment builder {dotted(builder)} is not static")
    return client, backend


def _backend_session(backend: str, root: Path) -> dict[str, Any]:
    """Find the worker's session class for ``backend`` by parsing, never importing."""
    function = environment_worker._build_session
    node, _start = _parsed(function)
    package = function.__module__.rpartition(".")[0]
    for child in ast.walk(node):
        test = getattr(child, "test", None)
        if not (isinstance(child, ast.If) and isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name) and test.left.id == "backend"
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == backend):
            continue
        for statement in child.body:
            if isinstance(statement, ast.ImportFrom):
                module = _resolve_import(statement, package)
                name = statement.names[0].name
                path = _module_file(module, root)
                tree = ast.parse(path.read_text(encoding="utf-8"))
                cls = next(item for item in tree.body
                           if isinstance(item, ast.ClassDef) and item.name == name)
                execute = next((item for item in cls.body if isinstance(item, ast.FunctionDef)
                                and item.name == "execute"), None)
                return {
                    "class": f"{module}:{name}",
                    "location": f"{_relative(path, root)}:{cls.lineno}",
                    "execute": (f"{_relative(path, root)}:{execute.lineno}" if execute
                                else "inherited, resolved in the isolated worker"),
                }
    raise BaselineExplainError(f"environment worker builds no session for {backend!r}")


def _proposal_route(runtime: Mapping[str, Any], root: Path) -> dict[str, Any]:
    client, backend = _builder_facts(runtime)
    return {
        "persisted_to": "control/submitted-action.json",
        "supervisor": [
            callable_record(EpisodeLoop._execute_proposal, root),
            callable_record(EpisodeLoop._accept_submitted_action, root),
            callable_record(episode_actions.dispatch_action, root),
        ],
        "dispatch_branches": _dispatch_branches(root),
        "environment": [
            callable_record(DescribedEnvironmentSession.execute, root),
            callable_record(client.execute, root),
        ],
        "transport": runtime.get("transport"),
        "worker_backend": backend,
        "backend_session": _backend_session(backend, root),
    }


def _summary(description: str) -> str:
    """The model-visible description's opening statement, before its help markers."""
    head = description.split(" Use when:", 1)[0].strip()
    match = re.match(r"(.+?[.!?])(\s|$)", head)
    return match.group(1) if match else head


def _tool(definition: Mapping[str, Any], entries: Mapping[str, Callable[..., Any]],
          surface: ResolvedToolSurface, route: Mapping[str, Any], codec: str, root: Path) -> dict[str, Any]:
    name = definition["name"]
    effect = _effect(name, surface)
    entry = entries.get(_entry_name(name))
    if entry is None:
        raise BaselineExplainError(f"tool {name!r} has no toolkit registry entry")
    record: dict[str, Any] = {
        "name": name, "effect": effect, "summary": _summary(str(definition["description"])),
        "proposal": None, "handler": callable_record(entry, root),
    }
    if effect != "state_changing_proposal":
        return record
    scan = _ProposalScan()
    scan.function(entry)
    admitted, exact = _submit_adapter(tool_names.canonical(name), root)
    tail = f"EnvironmentSession.execute on {route['backend_session']['class'].split(':')[1]}"
    actions = [{"action": action, "schema": schema,
                "executor": _executor_for(schema, route["dispatch_branches"])}
               for action, schema in sorted(scan.found)]
    executors = " | ".join(sorted({item["executor"].split(":")[1] for item in actions}))
    generic = (f"submitted as a {codec} proposal (action type not resolved statically) -> "
               f"Supervisor dispatch_action -> {tail}")
    proposal: dict[str, Any] = {"admitted_by": admitted}
    if actions and admitted is not None and exact and not scan.unresolved:
        proposal.update(resolution="static", actions=actions, route=f"{executors} -> {tail}")
    else:
        # Whatever the walk did resolve is still shown; the rest takes the
        # generic route rather than a guessed one.
        proposal.update(resolution="partial" if actions else "generic", actions=actions, route=generic)
    record["proposal"] = proposal
    return record


def _effect(name: str, surface: ResolvedToolSurface) -> str:
    if name in surface.lifecycle_tools:
        return "lifecycle"
    if name in surface.mutation_tools:
        return "state_changing_proposal"
    if name in surface.observation_tools:
        return "read_only_observation"
    return "read_only"


def explain_tools(surface: ResolvedToolSurface, runtime: Mapping[str, Any], codec: str,
                  root: Path) -> dict[str, Any]:
    """The tool runtime, the shared proposal route and one record per surface tool."""
    if not toolkit_dispatch.served(surface):
        raise BaselineExplainError(f"the toolkit does not serve {surface.profile_id}")
    runtime_class = _tool_runtime_class(surface)
    entries = load_tools()
    route = _proposal_route(runtime, root)
    return {
        "tool_runtime": {"class": dotted(runtime_class),
                         "location": location(runtime_class, root),
                         "dispatch": location(toolkit_dispatch.Dispatch.run, root)},
        "proposal_route": route,
        "tools": [_tool(item, entries, surface, route, codec, root) for item in surface.definitions],
    }
