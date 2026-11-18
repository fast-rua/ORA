"""Test helpers for skill sequence executor, shared by several test modules.

They were defined in ``tests/ora/orchestrator/episode/test_skill_sequence_executor.py``;
test modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from pathlib import Path

from ora.environment import SyntheticSession


def _session(tmp_path: Path, *, max_actions: int = 8) -> SyntheticSession:
    session = SyntheticSession(
        output_root=tmp_path / "native", task="synthetic-0", seed=0,
        max_actions=max_actions,
    )
    session.start()
    return session
