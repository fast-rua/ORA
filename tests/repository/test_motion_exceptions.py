"""A backend that plans its own motions says so beside its code, with a reason, an owner and an expiry.

docs/architecture.md: the simulator-independent motion base (the shared cuRobo V2 planner and the TOPP-RA + Ruckig
timing worker) is every backend's; "a backend-local IK solver or planner needs a written reason, an owner and an
expiry beside its code". The YAM station planned on its own for two weeks with none, unnoticed, while main moved
every simulator onto the shared base (2026-11-14). This test makes such a planner visible and dated.
"""

from __future__ import annotations

import ast
import datetime
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ROOT / "ora/environment/backends"
#: The exception: a module of the backend holding one literal (not imported: a backend loads under its own interpreter).
EXCEPTION_MODULE = "motion_exception.py"
#: What marks a backend-local planner: its own motion planning entry point or an IK library of its own.
LOCAL_PLANNER = re.compile(r"^def plan_motion\b|^\s*(import mink\b|from mink\b)", re.M)
FIELDS = {"schema", "backend", "what", "reason", "owner", "recorded", "expires", "ends_when"}


def _local_planners() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for backend in sorted(path for path in BACKENDS.iterdir() if path.is_dir() and not path.name.startswith("__")):
        hits = [str(module.relative_to(ROOT)) for module in sorted(backend.rglob("*.py"))
                if "__pycache__" not in module.parts and LOCAL_PLANNER.search(module.read_text(encoding="utf-8"))]
        if hits:
            found[backend.name] = hits
    return found


def _record(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    [value] = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and [target.id for target in node.targets if isinstance(target, ast.Name)] == ["MOTION_EXCEPTION"]]
    return ast.literal_eval(value)


def test_every_backend_local_planner_has_a_dated_written_exception():
    today = datetime.date.today()
    planners = _local_planners()
    assert planners, "the detector finds the YAM station's planner; if it finds none, it no longer detects anything"
    for backend, modules in planners.items():
        record_path = BACKENDS / backend / EXCEPTION_MODULE
        assert record_path.is_file(), (f"{backend} plans its own motions ({modules}) without "
                                       f"{record_path.relative_to(ROOT)}: write its reason, owner and expiry")
        record = _record(record_path)
        assert set(record) == FIELDS and record["schema"] == "ora.motion-exception.v1" and record["backend"] == backend
        assert all(isinstance(record[key], str) and record[key].strip() for key in FIELDS)
        expires = datetime.date.fromisoformat(record["expires"])
        assert expires >= today, (f"{backend}'s motion exception expired on {expires}: move it onto the shared "
                                  f"motion base, or renew the exception with a new reason")


def test_an_exception_names_a_backend_that_still_plans_its_own_motions():
    planners = _local_planners()
    for record_path in BACKENDS.glob(f"*/{EXCEPTION_MODULE}"):
        assert record_path.parent.name in planners, (f"{record_path.relative_to(ROOT)} remains after its backend "
                                                     "stopped planning on its own: remove it")
