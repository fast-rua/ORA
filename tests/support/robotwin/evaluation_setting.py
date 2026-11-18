"""Test helpers for robotwin evaluation setting, shared by several test modules.

They were defined in
``tests/ora/orchestrator/backends/robotwin/test_robotwin_evaluation_setting.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import yaml

from ora.environment.benchmarks.robotwin import RANDOMIZATION, TASKS


def _official_robotwin_root(tmp_path):
    from ora.environment.benchmarks.robotwin import CLEAN_RANDOMIZATION
    root = tmp_path / "robotwin"
    (root / "task_config").mkdir(parents=True)
    for name, randomization in (("demo_randomized", RANDOMIZATION), ("demo_clean", CLEAN_RANDOMIZATION)):
        (root / f"task_config/{name}.yml").write_text(yaml.safe_dump({"domain_randomization": dict(randomization), "embodiment": ["aloha-agilex"]}))
    tasks_dir = root / "description/task_instruction"
    tasks_dir.mkdir(parents=True)
    for task in TASKS:
        (tasks_dir / f"{task}.json").write_text("{}")
    textures = root / "assets/background_texture/unseen"
    textures.mkdir(parents=True)
    (textures / "0.png").write_bytes(b"fixture")
    return root


def _ledger_record(seed, *, accepted, classification="accepted", hold=None):
    """A worker ledger record in the shape ``backends.robotwin.qualification`` writes."""
    record = {"requested_seed": seed, "accepted": accepted, "classification": classification, "error": None,
              "reset_official_predicate": False, "stability": {"required_steps": 250}}
    if classification != "reset_predicate_true":
        record["expert"] = {"plan_success": accepted or classification != "oracle_failure",
                            "post_play_official_predicate": accepted, "eval_success_seen": accepted}
    if hold is not None:
        passed, checks, first_false = hold
        record["stability"].update({"passed": passed, "checks": checks, "first_false_step": first_false})
    return record
