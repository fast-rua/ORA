"""The library card and the library's entries (``policy.code_mode.library: extended``), by introspection.

``card``, one line per public class and function of ``ora_lib`` and ``robot`` under the import that reaches it, follows
the functions in the script tool's description (``docs.listing``): it is what scripts otherwise learned with ``dir()``.
``fragment_parts`` is a name's entry, its signature and docstring (``robot.docs("Point")``, ``"target"``). Nothing here is
written by hand, so neither can drift from the classes they describe. The two sandbox modules are imported on the host
for this, and only inside these functions: they import nothing of ORA, ``ora_lib`` needs only NumPy and SciPy, which the
host has too, and ``robot`` opens no connection until a call is made.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from typing import Any, Collection

from ora.agent.toolkit.interface import dry_run

#: What a script can do only where its surface serves it: a resolved contract's ``script_reads`` names them.
READS = ("dry_run", "query", "source")
#: The card's first line where ORA's code is mounted (``library.source_files``).
SOURCE_LINE = ("ORA's code is readable at `/ora/src`, as it runs here, at its repository paths (`ora.a.b` is "
               "`ora/a/b.py`): the toolkit (`ora/agent/toolkit/tools/<name>/tool.py` is `tools.<name>`, `lib/README.md` "
               "the map), the motion planner (`ora/environment/planners`), what they import, and each robot's public "
               "geometry (`ora/policy/embodiments`). It is read-only, for `grep -r`, `sed` and `cat`, not for importing: "
               "read what the question needs.")
#: ``target``'s parameter a surface whose ``move_pose`` does not take it is not told of (``docs.takes_interaction``).
INTERACTION = "interaction"
#: ``robot``'s public names the card leaves out: what a script calls every line of, and the library's own plumbing.
ROBOT_UNLISTED = frozenset({"docs", "query_arguments", "tools", "typed"})
#: A ``Result``'s fields that only a dry run sets.
DRY_RUN_FIELDS = frozenset({"dry_run", "end_joints_deg", "feasible", "predicted_pose", "predicted_poses", "route",
                            "screening"})
#: ``robot``'s names of a plan, listed only where dry runs are served: a ``Plan`` holds the dry-run fields, and
#: ``Result`` lists a run's.
PLAN_NAMES = frozenset({"Plan", "PlanNotRun"})
#: A ``Plan``'s fields as the card lists them.
PLAN_MEMBERS = ("feasible", "failure_code", "failure_stage", "summary", "predicted_pose", "predicted_poses",
                "end_joints_deg", "route", "screening", "call_id", "explain")
#: The parameter of ``ora_lib``'s functions the card leaves out: it says once that they default to the current one.
IMPLIED_PARAMETERS = frozenset({"self", "observation"})


def _library(reads: Collection[str]) -> dict[str, tuple[str, str, Any]]:
    """Every public class and function of ``ora_lib`` and ``robot`` a script can use here, by the name it is looked up
    by: ``(source, label, value)``. ``source`` is the module (``ora_lib``, ``ora_lib.observation``, ``robot``), ``label``
    what the card calls it (``Point``, ``geometry.rigid``)."""

    from . import ora_lib, robot

    found: dict[str, tuple[str, str, Any]] = {}
    public = {name: getattr(ora_lib, name) for name in ora_lib.__all__}
    for name, value in public.items():
        if inspect.isclass(value) or inspect.isfunction(value):
            found[name] = ("ora_lib", name, value)
    listed = [value for _source, _label, value in found.values()]
    for module_name, module in public.items():
        for name in getattr(module, "__all__", ()) if inspect.ismodule(module) else ():
            value = getattr(module, name)
            if not (inspect.isclass(value) or inspect.isfunction(value)) or any(value is item for item in listed):
                continue
            label = name if inspect.isclass(value) else f"{module_name}.{name}"
            found[label] = (f"ora_lib.{module_name}", label, value)
            found.setdefault(name, found[label])
    plans = dry_run.ARGUMENT in reads
    for name in robot.__all__:
        if name not in ROBOT_UNLISTED and (name != "query" or "query" in reads) and (plans or name not in PLAN_NAMES):
            found[name] = ("robot", name, getattr(robot, name))
    return found


def _assigned(function: Any) -> list[str]:
    """The public attributes ``function`` assigns on ``self``, as written."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    stored = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
              and isinstance(node.value, ast.Name) and node.value.id == "self" and not node.attr.startswith("_")]
    return [node.attr for node in sorted(stored, key=lambda node: (node.lineno, node.col_offset))]


def _call(function: Any, *, skip: Collection[str] = IMPLIED_PARAMETERS) -> str:
    """``(a, b=1, *, c=None)``: a function's parameters without their annotations, as the card shows them; nothing for
    a class with no signature of its own (a builtin's constructor)."""

    try:
        declared = inspect.signature(function)
    except (TypeError, ValueError):
        return ""
    kept = [item.replace(annotation=inspect.Parameter.empty) for name, item in declared.parameters.items()
            if name not in skip]
    return str(declared.replace(parameters=kept, return_annotation=inspect.Signature.empty))


def _returned(function: Any) -> str:
    """`` -> Name`` for a function annotated to return one of the library's classes."""

    name = str(inspect.signature(function).return_annotation).strip("'\"").rsplit(".", 1)[-1]
    return f" -> {name}" if re.fullmatch(r"[A-Z]\w*", name) and name not in ("Any", "None") else ""


def members(cls: type) -> list[str]:
    """A class's public members as a script reads them: ``.name`` per attribute (what its ``__init__`` assigns, its
    annotations, its properties), ``.name(arguments)`` per method, the builtin bases' left out."""

    shown: dict[str, str] = {}
    for base in cls.__mro__:
        if base.__module__ == "builtins":
            continue
        if "__init__" in vars(base):
            shown.update({name: f".{name}" for name in _assigned(vars(base)["__init__"])})
        shown.update({name: f".{name}" for name in vars(base).get("__annotations__", ()) if not name.startswith("_")})
        for name, value in vars(base).items():
            if not name.startswith("_"):
                method = value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
                shown.setdefault(name, f".{name}{_call(method)}" if inspect.isfunction(method) else f".{name}")
    return list(shown.values())


def _result_fields(reads: Collection[str]) -> list[str]:
    """``.name`` per field of a call's ``Result`` (an instance's, since it sets some by name), those its repr shows
    first; a dry run's are a ``Plan``'s (``_plan_fields``)."""

    from . import robot

    fields = dict.fromkeys([*robot.FIELDS, *vars(robot.Result("tool", {"ok": True, "result": {}}))])
    return [f".{name}" for name in fields if name not in DRY_RUN_FIELDS]


def _plan_fields() -> list[str]:
    """``.name`` per field of a ``Plan``."""

    return [f".{name}" for name in PLAN_MEMBERS]


def _class_members(value: type, reads: Collection[str]) -> list[str]:
    """``members``, for the two classes that hold more than a class says: a call's ``Result`` and ``ObjectGeometry``,
    whose fields are keys of what ``localize_mask`` answers, every one of them, on an empty mask too."""

    import numpy as np

    from . import robot
    from .ora_lib import vision

    if value is robot.Plan:
        return _plan_fields()
    if value is robot.Result:
        return _result_fields(reads)
    if value is vision.ObjectGeometry:
        empty = vision.localize_mask(np.zeros((2, 2), bool), np.zeros((2, 2)), np.eye(3), np.eye(4))
        return [*members(value), *(f".{name}" for name in empty)]
    return members(value)


def _card_line(source: str, label: str, value: Any, reads: Collection[str], *, interaction: bool) -> str:
    """One line of the card: a class's constructor and members, a function's signature."""

    if not inspect.isclass(value):
        skip = IMPLIED_PARAMETERS if interaction else IMPLIED_PARAMETERS | {INTERACTION}
        return f"- {label}{_call(value, skip=skip)}{_returned(value)}"
    made_by_scripts = source == "ora_lib" and not issubclass(value, BaseException) and "__init__" in vars(value)
    return f"- {label}{_call(value) if made_by_scripts else ''}: {' '.join(_class_members(value, reads))}".rstrip(": ")


def card(reads: Collection[str] = READS, *, interaction: bool = True) -> str:
    """The library card, after the functions: one line per public class and function of ``ora_lib`` and ``robot``, under
    the import that reaches it; ``interaction`` False leaves that parameter of ``target`` out."""

    found = {label: (source, value) for source, label, value in _library(reads).values()}
    library = [_card_line(source, label, value, reads, interaction=interaction)
               for label, (source, value) in found.items() if source != "robot"]
    own = [_card_line(source, label, value, reads, interaction=interaction)
           for label, (source, value) in found.items() if source == "robot"]
    return ((f"{SOURCE_LINE}\n" if "source" in reads else "")
            + "`ora_lib` (`from ora_lib import <name>`; `<module>.<name>` after `from ora_lib import <module>`; a class "
            "with no call here is one you get from a function; where a function reads the scene it also takes "
            "`observation=`, by default the current one):\n" + "\n".join(library)
            + "\n`robot`, not `ora_lib` (`from robot import <name>`):\n" + "\n".join(own))


def _unserved(text: str, reads: Collection[str]) -> str:
    """A docstring without the paragraphs of a dry run, where none is served."""

    if dry_run.ARGUMENT in reads:
        return text
    return "\n\n".join(item for item in text.split("\n\n") if not re.search(r"dry[ _]run", item, re.I))


def library_names(reads: Collection[str] = READS) -> list[str]:
    """Every name ``fragment_parts`` answers."""

    return list(_library(reads))


def library_module(name: str, reads: Collection[str] = READS) -> str | None:
    """The module a script imports the library's ``name`` from (``ora_lib``, ``ora_lib.geometry``, ``robot``); None
    for a name that is not the library's."""

    found = _library(reads).get(name)
    return None if found is None else found[0]


def fragment_parts(name: str, *, reads: Collection[str] = READS, interaction: bool = True) -> dict[str, str] | None:
    """The entry of a public name of ``ora_lib`` or ``robot``, from its signature and docstring and, for a class, its
    members' (one ``description`` part); None for any other name. ``interaction`` False leaves that parameter of
    ``target`` out."""

    found = _library(reads).get(name)
    if found is None:
        return None
    source, label, value = found
    head = f"{source}.{label.rsplit('.', 1)[-1]}{_call(value, skip=('self',) if interaction else ('self', INTERACTION))}"
    text = _unserved(inspect.getdoc(value) or "", reads)
    if not interaction:
        text = "\n".join(line for line in text.split("\n") if INTERACTION not in line)
    if not inspect.isclass(value):
        return {"description": f"{head}{_returned(value)}\n\n{text}".rstrip()}
    listed = _class_members(value, reads)
    documented = [f"{shown}: {doc.split(chr(10) * 2)[0]}" for shown in listed
                  if (doc := inspect.getdoc(getattr(value, shown[1:].split("(")[0], None))) and "(" in shown]
    return {"description": "\n\n".join(item for item in (f"{head}\n{' '.join(listed)}", text, "\n".join(documented))
                                        if item)}


__all__ = ["DRY_RUN_FIELDS", "IMPLIED_PARAMETERS", "INTERACTION", "PLAN_MEMBERS", "PLAN_NAMES", "READS",
           "ROBOT_UNLISTED", "SOURCE_LINE", "card", "fragment_parts", "library_module", "library_names", "members"]
