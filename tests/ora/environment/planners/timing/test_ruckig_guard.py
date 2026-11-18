"""ORA never sets Ruckig's ``intermediate_positions``: the Community wheel would then call a remote API."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[5] / "ora/environment/planners/timing"
FORBIDDEN = {"intermediate_positions", "per_section_max_velocity", "per_section_max_acceleration",
             "per_section_max_jerk", "per_section_minimum_duration"}


def test_no_timing_module_names_a_waypoint_field_of_ruckig():
    # By the syntax tree, so the documentation may say why; the same check holds without the libraries in CI.
    modules = sorted(PACKAGE.glob("*.py"))
    assert {path.name for path in modules} >= {"retime.py", "worker.py"}
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names |= {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                  and isinstance(node.value, str) and node.value in FORBIDDEN}
        assert not names & FORBIDDEN, path
