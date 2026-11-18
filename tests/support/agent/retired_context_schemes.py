"""Retired context schemes, kept as test fixtures of the stages and management versions that remain.

On 2026-11-05 the v9 schemes (``context_rolling_tail_v3`` and its arms), ``context_rolling_tail_v2`` (v8) and
``context_age_scripts`` (arm B) left ``ora/policy/context_schemes``. Their mechanisms stay: management v8 and v9
(sealed runs and the v10 schemes build on them), the camera tail, exchange aging with its argument previews, the
reasoning binding. The files under ``tests/data/context_scheme_fixtures`` are those schemes without their prompts;
``retired_scheme`` resolves one as ``load_context_scheme`` resolved it (its policy with the stage list and the
management object the list is the stage list of).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ora.agent.composition.schemas.context_stages import management_equivalent, resolve_stage_list

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/data/context_scheme_fixtures"
#: Retired scheme id -> fixture file.
RETIRED = {
    "context_rolling_tail_v2": "rolling-tail-v2.json",
    "context_rolling_tail_v3": "rolling-tail-v3.json",
    "context_rolling_tail_v3_camera_tail": "rolling-tail-v3-camera-tail.json",
    "context_rolling_tail_v3_recent_reasoning": "rolling-tail-v3-recent-reasoning.json",
    "context_rolling_tail_v3_unaged_reasoning": "rolling-tail-v3-unaged-reasoning.json",
    "context_age_scripts": "age-scripts.json",
}


def retired_raw(scheme_id: str) -> dict[str, Any]:
    """The fixture file of a retired scheme, as the scheme file held it (without its prompt)."""

    return json.loads((FIXTURES / RETIRED[scheme_id]).read_text(encoding="utf-8"))


def retired_scheme(scheme_id: str) -> dict[str, Any]:
    """``{"policy": ...}`` of a retired scheme: its policy, stage list and management object."""

    raw = retired_raw(scheme_id)
    stage_list = resolve_stage_list(ROOT, raw, label=f"fixture {scheme_id}")
    policy = {**raw["policy"], "context_management": management_equivalent(stage_list, label=scheme_id),
              "context_stages": stage_list}
    return {"policy": policy, "prompt": None}
