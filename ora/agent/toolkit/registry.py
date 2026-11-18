"""The toolset's registry: one tool is one directory of ``tools/`` holding its entry.

``tools/<name>/tool.py`` defines ``run(api, args)``; ``tools/<name>/description.json``
is what a file-backed surface shows the model for it (read by
``ora.agent.composition.surface.toolkit_surface``, not here). A directory without ``tool.py`` (``__pycache__``,
helpers) is not a tool. Entries import by validated name only, never from a
path the caller supplies.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any, Callable

TOOLS_ROOT = Path(__file__).resolve().parent / "tools"
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def tool_names() -> tuple[str, ...]:
    """Every tool of this toolset, sorted by name."""
    return tuple(sorted(
        path.name for path in TOOLS_ROOT.iterdir()
        if path.is_dir() and _NAME.match(path.name) and (path / "tool.py").is_file()
    ))


def load_tool(name: str) -> Callable[[Any, Any], Any]:
    """The ``run`` entry of one tool; an unknown name raises ``KeyError``."""
    if not isinstance(name, str) or not _NAME.match(name) or not (TOOLS_ROOT / name / "tool.py").is_file():
        raise KeyError(name)
    return importlib.import_module(f"{__package__}.tools.{name}.tool").run


def load_tools() -> dict[str, Callable[[Any, Any], Any]]:
    """Every tool's entry by name."""
    return {name: load_tool(name) for name in tool_names()}
