"""Test helpers for robodojo ik continuity, shared by several test modules.

They were defined in
``tests/ora/environment/backends/robodojo/test_robodojo_ik_continuity.py``; test modules
import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from ora.environment.backends.robodojo import ik


TARGET = [0.2, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0]


def candidates(values, *, success=None, feasible=None):
    count, joints = len(values), len(values[0])
    return {"solutions": values, "joint_names": [f"joint_{i}" for i in range(joints)],
            "success": [True] * count if success is None else success,
            "feasible": [True] * count if feasible is None else feasible,
            "position_tolerance_m": 0.01, "orientation_tolerance_rad": 0.02}


def select(value, *, current=None, limits=None, fk=None, bound=1.0, path_clearance=None, canonical=None):
    joints = len(value["joint_names"])
    extra = {} if canonical is None else {"canonical_jump_limit_rad": canonical}
    return ik.select_candidate_set(candidates=value, current=[0.0] * joints if current is None else current,
        limits=[[-10.0, 10.0]] * joints if limits is None else limits,
        target=TARGET, fk=(lambda _: TARGET) if fk is None else fk, jump_limit_rad=bound,
        path_clearance=path_clearance, **extra)
