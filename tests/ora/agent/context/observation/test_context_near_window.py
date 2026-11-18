"""Historical observation bodies and the coverage evidence of the near window.

The near-window projections (the fixed Episode media tail below and above the
soft image threshold, exact recovery of an externalized observation body, the
resolved L5 threshold) drove the Python Context pipeline, which ORA removed
with the Native core; they were recorded as Context fixtures
(``tests/data/context_differential/captured_near_window_*``) that the
TypeScript pipeline replays. Here: the recoverable-body invariant
(``tests/support/agent/historical_observation.py``, the Python mirror of the Pi core's) and the coverage check sealing
applies, on the projection manifest the pipeline recorded.
"""
import copy
import gzip
import json
from pathlib import Path

import pytest

from tests.support.agent.historical_observation import externalize, invariant_block, materialize
from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.memory.catalog import ContextArtifactCatalog
from ora.orchestrator.evaluation.context_coverage import coverage_violations
from tests.support.agent.lossless_context_runtime import _image, _messages

ROOT = Path(__file__).resolve().parents[5]
#: The near-window projection the seal-coverage case recorded from the Python pipeline.
RECORDED = ROOT / ("tests/data/context_differential/captured_near_window_06_test_seal_coverage_accepts_"
                   "target_retention_and_rejects_protected_eviction.json.gz")


def history(completed, images=4):
    messages = _messages(completed)
    for message in messages:
        if "_observation" in message:
            step = message["_observation"]["step"]
            message["content"][1:] = [_image(f"{step}-{i}") for i in range(images)]
    return messages


def test_observation_invariant_rejects_tampered_recovery_body(tmp_path):
    catalog = ContextArtifactCatalog(tmp_path, "policy-session-1", ArtifactStore(tmp_path))
    message = history(0)[-1]
    message["content"][0]["text"] = json.dumps({"current_actual": {"measured": [1] * 4000}})
    materialize(message, catalog=catalog, round_index=1, action_step=0)
    projected, changed, _ = externalize([message], eligible_indexes={0})
    assert changed == [0]
    assert invariant_block(projected[0]) == message["content"][0]
    tampered = copy.deepcopy(projected[0])
    tampered["content"][0]["text"] = tampered["content"][0]["text"].replace('"observation_externalized":true', '"observation_externalized":false')
    with pytest.raises(PolicySessionError, match="differs from recoverable"):
        invariant_block(tampered)


def test_seal_coverage_accepts_target_retention_and_rejects_protected_eviction():
    fixture = json.loads(gzip.decompress(RECORDED.read_bytes()))
    manifest = copy.deepcopy([op for op in fixture["ops"] if op["op"] == "project"][-1]["result"]["manifest"])
    assert coverage_violations(manifest, manifest["action_step_inventory"]) == []
    tampered = {**manifest, "lifecycle_media_externalized_through_action_step": 4}
    assert "lossless_context_lifecycle_coverage_invalid" in coverage_violations(
        tampered, manifest["action_step_inventory"])
    tampered = {**manifest, "active_inline_images": 4}
    assert "lossless_context_image_target_invalid" in coverage_violations(
        tampered, manifest["action_step_inventory"])
    tampered = {**manifest, "resolved_episode_checkpoint_trigger_tokens": 150000}
    assert "lossless_context_resolved_threshold_invalid" in coverage_violations(
        tampered, manifest["action_step_inventory"])
