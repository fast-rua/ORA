"""``policy.context_tools``: the recoverable-artifact reader belongs to the context scheme.

Without the field, lossless retention advertises ``artifact_read`` and
``observe(ref=...)`` as it always did. ``[]`` leaves both out; v8 management
needs the field and cannot provide the reader, v9 management needs it and must
provide the reader. Resolution runs on the Synthetic cell; nothing is launched.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ora.agent.composition.baseline import BASELINE_CATALOGS, BASELINE_ID
from ora.agent.composition.context_scheme import load_context_scheme
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from tests.support.orchestrator.native_synthetic_spec import synthetic_spec
from tests.support.agent.retired_context_schemes import retired_scheme

ROOT = Path(__file__).resolve().parents[4]
#: The default baseline's lossless context: the scheme it names (context_lossless_v6).
LOSSLESS = load_context_scheme(
    ROOT, json.loads((ROOT / BASELINE_CATALOGS[BASELINE_ID]).read_text(encoding="utf-8"))["context_scheme"],
)["policy"]
#: Management v8 and v9 as the retired schemes context_rolling_tail_v2 and _v3 configured them (test fixtures).
ROLLING = retired_scheme("context_rolling_tail_v2")["policy"]
AGING = retired_scheme("context_rolling_tail_v3")["policy"]


def _plan(**policy: Any) -> dict[str, Any]:
    spec = synthetic_spec()
    spec["policy"].update(policy)
    return resolve_run_spec(spec, root=ROOT)


def _tools(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The advertised tools; the reader is ``artifact.read`` on this dotted surface."""
    return {item["name"].replace("artifact.read", "artifact_read"): item
            for item in plan["policy_composition"]["tool_surface"]["definitions"]}


def test_lossless_retention_without_the_field_keeps_the_reader():
    plan = _plan(**LOSSLESS)
    tools = _tools(plan)
    assert "context_tools" not in plan["policy"]
    assert "artifact_read" in tools and "ref" in tools["observe"]["inputSchema"]["properties"]


def test_an_empty_list_leaves_the_reader_out():
    plan = _plan(**{**LOSSLESS, "context_tools": []})
    tools = _tools(plan)
    assert plan["policy"]["context_tools"] == []
    assert "artifact_read" not in tools and "ref" not in tools["observe"]["inputSchema"].get("properties", {})
    layers = [layer["kind"] for layer in plan["policy_composition"]["tool_surface"]["derived_layers"]]
    assert "recoverable_context_artifacts_v1" not in layers


def test_the_rolling_tail_scheme_resolves_without_the_reader():
    plan = _plan(**ROLLING)
    assert plan["policy"]["context_management"]["schema"] == "ora.native-context-management.v8"
    assert "artifact_read" not in _tools(plan)


def test_the_exchange_aging_scheme_resolves_with_the_reader():
    plan = _plan(**AGING)
    tools = _tools(plan)
    assert plan["policy"]["context_management"]["schema"] == "ora.native-context-management.v9"
    assert "artifact_read" in tools and "ref" in tools["observe"]["inputSchema"]["properties"]


@pytest.mark.parametrize("policy, match", [
    ({**ROLLING, "context_tools": ["artifact_read"]}, "cannot provide artifact_read"),
    ({**AGING, "context_tools": []}, "must provide artifact_read"),
    ({key: value for key, value in ROLLING.items() if key != "context_tools"}, "requires policy.context_tools"),
    ({**LOSSLESS, "context_tools": ["artifact_read", "artifact_read"]}, "distinct names"),
    ({**LOSSLESS, "context_tools": ["notes"]}, "distinct names"),
    ({**LOSSLESS, "context_tools": "artifact_read"}, "distinct names"),
])
def test_context_tools_refusals(policy, match):
    with pytest.raises(RunSpecError, match=match):
        _plan(**policy)


def test_the_reader_needs_lossless_retention():
    # The context half of the retired baseline robodojo_toolkit_v1 (scheme context_rewrite_compact_v1).
    rewrite = {"context_profile": "native_causal_context_v1", "context_composition": "native_slots_v3",
               "context_retention": "rewrite_compact_v1", "context_token_estimator": "image_aware_v1",
               "summary_policy": "disabled", "memory_policy": "disabled", "provider_continuation_replay": "exact_v1",
               "initial_observation": "automatic"}
    with pytest.raises(RunSpecError, match="requires context_retention"):
        _plan(**{**rewrite, "context_tools": ["artifact_read"]})
