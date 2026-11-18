"""``ora --help`` and ``ora status`` stay cheap: building the parser loads the app modules, not the engine.

Each command module imports its implementation inside ``run``, so the parser of
every command loads fewer than ``BUDGET`` ``ora`` modules and neither the
evaluation, the episode coordinator, the launch path nor NumPy; ``ora status``
reads the run registry and leases without them. Each probe runs in a fresh
interpreter, where ``sys.modules`` says what one command loaded.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
#: ``ora`` modules one interpreter may load to build the whole parser.
BUDGET = 40
#: What only the commands that launch, evaluate or seal a run may load.
ENGINE = ("ora.orchestrator.evaluation", "ora.orchestrator.episode.coordinator", "ora.orchestrator.launch",
          "ora.orchestrator.spec.run_spec", "numpy")


def _loaded(code: str) -> list[str]:
    probe = (code + "\nimport json, sys\nprint(json.dumps(sorted(name for name in sys.modules "
             "if name == 'ora' or name.startswith('ora.') or name == 'numpy')))")
    result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True,
                            timeout=60)
    return json.loads(result.stdout.splitlines()[-1])


def _engine(loaded: list[str]) -> list[str]:
    return [name for name in loaded if any(name == item or name.startswith(item + ".") for item in ENGINE)]


def test_importing_the_entry_point_loads_almost_nothing() -> None:
    loaded = _loaded("import ora.app.main")
    assert len(loaded) < BUDGET, loaded
    assert loaded == ["ora", "ora.app", "ora.app.main", "ora.app.passthrough"]


def test_the_whole_parser_stays_within_the_budget_and_off_the_engine() -> None:
    loaded = _loaded("import ora.app.main as main\nmain.build_parser()")
    assert len(loaded) < BUDGET, f"{len(loaded)} modules: {loaded}"
    assert _engine(loaded) == []


def test_ora_status_loads_neither_the_evaluation_nor_the_coordinator() -> None:
    loaded = _loaded("import ora.app.main as main\nassert main.main(['status']) == 0")
    assert _engine(loaded) == []


def test_ora_help_completes_within_three_seconds() -> None:
    """A coarse wall-clock backstop; the module-count tests above are the real guard.

    They catch the parser loading the engine, which on a quiet host adds only
    about a second. The time measured here includes interpreter start-up from
    the virtual environment, which on a loaded shared host or a stalling network
    file system took seconds by itself: a 1 s budget failed there while the
    module counts passed. So one warm-up run fills the file-system caches, the
    best of three timed runs counts, and the budget is 3 s, about ten times what
    ``ora --help`` takes on a quiet host (0.3 s); it fails only when ``--help``
    starts doing seconds of work.
    """
    def run_help() -> float:
        started = time.perf_counter()
        result = subprocess.run([sys.executable, "-m", "ora.app", "--help"], cwd=ROOT, capture_output=True,
                                text=True, check=True, timeout=60)
        assert result.stdout.startswith("ora — Open Robot-Use Agents")
        return time.perf_counter() - started

    run_help()
    elapsed = [run_help() for _ in range(3)]
    assert min(elapsed) < 3.0, elapsed
