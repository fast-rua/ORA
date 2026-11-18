"""Every selectable policy asset is reached by a default or maintained consumer, or has an owner and an expiry.

Selectable: the registered baselines (``BASELINE_CATALOGS``), ToolSurfaces (``PROFILE_CATALOG``) and context
schemes (the files of ``ora/policy/context_schemes``, ``context_schemes``). Reached: what the defaults select (``BASELINE_ID``, ``ENVIRONMENT_BASELINES``, their
instances' surfaces and named schemes) and what a maintained code path names (``ora local``, YAM play with and
without ``--no-perception``, and
Synthetic), closed over a reached surface's
``TOOLKIT_BASELINES`` baseline. Everything else needs an entry in ``tests/data/policy-retention.json`` with an owner,
a reason and an expiry date that has not passed; an entry for a reached or unregistered id is stale and fails too
(``ora/policy/README.md``, "Versions, patches and retirement").
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from ora.agent.composition.baseline import BASELINE_CATALOGS, BASELINE_ID, ENVIRONMENT_BASELINES
from ora.agent.composition.context_scheme import context_schemes
from ora.agent.composition.ora_local import DEFAULT_LOCAL_COMPOSITION
from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.surface.toolkit_profiles import TOOLKIT_BASELINES
from ora.orchestrator.backends.yam.play import NO_PERCEPTION_SURFACE as YAM_PLAY_NO_PERCEPTION_SURFACE
from ora.orchestrator.backends.yam.play import SURFACE as YAM_PLAY_SURFACE

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = json.loads((ROOT / "tests/data/policy-retention.json").read_text(encoding="utf-8"))
#: Surfaces a maintained code path selects by a literal, with the file that holds it. robot_runtime_v1 and the
#: M0-M5 ladder were retired by human decision on 2026-11-02, after 0 of 8,202 scanned sealed plans named either
#: surface. robot_runtime_v2 stays because it serves Synthetic, the quickstart and ``ora interactive``, which that
#: count could not see: those plans hold no Synthetic run (``ora/policy/README.md``).
CODE_DEFAULT_SURFACES = {
    "robot_runtime_v2": "ora/orchestrator/spec/quickstart.py",
}
KINDS = {"baselines": BASELINE_CATALOGS, "tool_surfaces": PROFILE_CATALOG, "context_schemes": context_schemes(ROOT)}


def _reached() -> dict[str, set[str]]:
    baselines = {BASELINE_ID, *ENVIRONMENT_BASELINES.values()}
    surfaces = {DEFAULT_LOCAL_COMPOSITION["tool_surface"], YAM_PLAY_SURFACE, YAM_PLAY_NO_PERCEPTION_SURFACE,
                *CODE_DEFAULT_SURFACES}
    schemes: set[str] = {DEFAULT_LOCAL_COMPOSITION["context_scheme"]}
    while True:
        before = len(baselines) + len(surfaces)
        for baseline in list(baselines):
            catalog = json.loads((ROOT / BASELINE_CATALOGS[baseline]).read_text(encoding="utf-8"))
            surfaces |= {instance["tool_surface"] for instance in catalog["instances"].values()}
            # A baseline's scheme and the one it names for code mode (``code_mode_context_scheme``).
            schemes |= {catalog[key] for key in ("context_scheme", "code_mode_context_scheme") if catalog.get(key)}
        baselines |= {TOOLKIT_BASELINES[surface] for surface in surfaces if surface in TOOLKIT_BASELINES}
        if len(baselines) + len(surfaces) == before:
            return {"baselines": baselines, "tool_surfaces": surfaces, "context_schemes": schemes}


def test_the_code_default_surfaces_are_still_named_where_cited():
    for surface, path in CODE_DEFAULT_SURFACES.items():
        assert f'"{surface}"' in (ROOT / path).read_text(encoding="utf-8"), (surface, path)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_unreached_selectable_asset_has_an_entry_and_no_entry_is_stale(kind):
    unreached = set(KINDS[kind]) - _reached()[kind]
    listed = set(REGISTRY[kind])
    assert unreached - listed == set(), f"{kind} with no default or maintained consumer and no retention entry"
    assert listed - unreached == set(), f"stale {kind} entries: reached by a consumer, or no longer registered"


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_entry_has_an_owner_a_reason_and_a_live_expiry(kind):
    today = datetime.date.today()
    for name, entry in REGISTRY[kind].items():
        assert entry["owner"].strip() and entry["reason"].strip(), name
        recorded, expires = (datetime.date.fromisoformat(entry[key]) for key in ("recorded", "expires"))
        assert recorded <= expires, name
        assert today <= expires, f"{kind} {name} expired on {expires}: retire it, or renew the entry with a reason"
