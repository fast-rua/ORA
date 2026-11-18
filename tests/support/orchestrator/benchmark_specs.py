"""Test helpers for benchmark specs, shared by several test modules.

They were defined in ``tests/ora/orchestrator/spec/test_benchmark_spec.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ora.agent.composition.baseline import load_native_baseline
from ora.orchestrator.spec import benchmark as benchmark_spec
from ora.orchestrator.backends.robotwin import evaluation_seeds as evaluation


ROOT = Path(__file__).resolve().parents[3]
ACTIVATION = json.loads((ROOT / "tests/data/default-expectations.json").read_text())
CATALOG = load_native_baseline(ROOT)
EXPERIMENT = "benchmark-spec-fixture"
REVISION = CATALOG["instances"]["robotwin"]["runtime"]["required_revision"]


def _official_manifest(directory: Path, task="adjust_bottle", task_config="demo_randomized", seeds=(100002,),
                       revision=REVISION, **changes) -> Path:
    """A sealed manifest in the shape backends.robotwin.evaluation_seeds qualify writes."""
    records = {str(seed): {"reset_predicate_false": True, "oracle_success": True,
                           "post_success_physics_hold": True, "terminal_stability_steps": 250} for seed in seeds}
    body = {"schema": evaluation.SEED_MANIFEST_SCHEMA, "task": task, "task_config": task_config,
            "robotwin_revision": revision, "source_commit": "a" * 40, "plan_sha256": "b" * 64,
            "accepted_seeds": list(seeds), "complete": True, "seeds": records}
    body.update(changes)
    body["manifest_sha256"] = evaluation.digest(body)
    path = directory / f"{task}-{task_config}.json"
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


def _readiness_report(directory: Path, monkeypatch, *, age_s: float = 0.0) -> Path:
    from ora.agent.model_services import operator as services

    monkeypatch.setenv("ORA_SAM_PORT", "18124")
    monkeypatch.setenv("ORA_GRASP_PORT", "18135")
    health = {**{key: "fixture" for key in services.FIELDS},
              "protocol_version": services.GROUNDING_GRASP_PROTOCOL_VERSION}
    report = {"schema": services.REPORT_SCHEMA, "created_unix_s": time.time() - age_s, "passed": True,
              "protocol_version": services.GROUNDING_GRASP_PROTOCOL_VERSION,
              "adapter_sha256": services.grounding_grasp_adapter_sha256(),
              "checks": {k: True for k in ("single_point_wire", "batch_point_wire", "text_wire", "grasp_warmup")},
              "services": {name: {"endpoint": endpoint, "health": health}
                           for name, endpoint in services.endpoints().items()}}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (services.digest(report) + ".json")
    path.write_text(json.dumps(report))
    return path


def _robotwin(tmp_path, task="adjust_bottle", seed=100002, task_config="demo_randomized", manifest=None, **kwargs):
    manifest = manifest or _official_manifest(tmp_path / "manifests", task, task_config, (seed,))
    kwargs.setdefault("resource_profile", "robotwin-gpu4")
    spec = benchmark_spec.build_robotwin_spec(task, seed=seed, task_config=task_config, experiment=EXPERIMENT,
                                              seed_manifest=manifest, **kwargs)
    return spec, manifest


def _libero(suite="libero_10_swap", task_id=6, seed=0, **kwargs):
    kwargs.setdefault("resource_profile", "libero-gpu4b")
    return benchmark_spec.build_libero_pro_spec(suite, task_id, seed=seed, experiment=EXPERIMENT, **kwargs)
