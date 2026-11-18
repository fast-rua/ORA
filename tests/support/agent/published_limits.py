"""Test helpers for published limits, shared by several test modules.

They were defined in ``tests/ora/agent/composition/surface/test_published_limits.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from ora.agent.composition.surface.tool_surface import resolve_tool_surface


ROOT = Path(__file__).resolve().parents[3]
TABLE = json.loads((ROOT / "tests/data/published-limits.json").read_text(encoding="utf-8"))
SURFACES: dict[str, dict[str, Any]] = TABLE["surfaces"]


@functools.cache
def _surface(key: str) -> Any:
    return resolve_tool_surface(SURFACES[key]["profile"])
