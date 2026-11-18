"""Importing one host module loads that module, not the tool runtime.

``ora.agent.toolkit.host.runtime`` is a lazy facade (rule L9 of ``docs/developer/repository-structure.md``): its
public name resolves on first use, so a leaf such as the runtime protocol can be imported without loading the runtime
classes and the mixins they are built from.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]

_PROBE = """
import json, sys
import {module}
print(json.dumps(sorted(name for name in sys.modules if name == "ora" or name.startswith("ora."))))
"""


def _loaded(module: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return json.loads(completed.stdout)


def test_a_leaf_loads_fewer_than_twenty_ora_modules() -> None:
    loaded = _loaded("ora.agent.toolkit.host.runtime.protocol")
    assert len(loaded) < 20, loaded
    assert "ora.agent.toolkit.host.runtime.core" not in loaded


def test_the_facade_resolves_its_name_on_first_use() -> None:
    from ora.agent.toolkit.host import runtime

    assert runtime.ManipulationToolRuntime.__module__ == "ora.agent.toolkit.host.runtime.core"
    assert sorted(runtime.__all__) == ["ManipulationToolRuntime"]
