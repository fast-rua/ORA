"""Host-safe, provider-independent Native baseline catalog.

This tree composes the default baseline and the maintained ones registered in
``BASELINE_CATALOGS``; any other id is refused.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .context_scheme import CONTEXT_FIELDS, load_context_scheme
from .locked_assets import locked_asset


BASELINE_ID = "native_full_explicit_motion_direct_v4"
CATALOG_PATH = "ora/policy/baselines/native-full-explicit-motion-direct-v4.json"
#: Environments whose default is a baseline of their own instead of ``BASELINE_ID``'s instance: RoboDojo runs the
#: toolkit v6 surface on ``robodojo_toolkit_v7`` (2026-11-02), in code mode since 2026-11-05 (its ``code_mode``).
#: ``BASELINE_ID``'s RoboDojo instance (the closed skills) stays selectable by naming that baseline, and
#: ``robodojo_toolkit_v6`` (toolkit v5, with ``check_task``, tool calls) by naming it.
ENVIRONMENT_BASELINES = {"robodojo": "robodojo_toolkit_v7"}
# Selectable from this tree: the default and every maintained baseline.
BASELINE_CATALOGS = {
    "robodojo_toolkit_v7": "ora/policy/baselines/robodojo-toolkit-v7.json",
    "robodojo_toolkit_v6": "ora/policy/baselines/robodojo-toolkit-v6.json",
    BASELINE_ID: CATALOG_PATH,
}
#: General baselines carry LIBERO-Pro and RoboTwin; the toolkit-only catalogs
#: explicitly target RoboDojo. The toolkit baselines v1, v2, v4, v5,
#: matched-context and legacy-visible/-replay were retired on 2026-10-31, v3 on 2026-11-01.
TOOLKIT_BASELINE_IDS = frozenset({"robodojo_toolkit_v6", "robodojo_toolkit_v7"})
REQUIRED_INSTANCES = frozenset({"libero_pro", "robotwin"})
OPTIONAL_INSTANCES = frozenset({"robodojo", "robocasa", "robolab"})
#: Instance runtime fields that name a profile asset by path.
RUNTIME_PROFILE_FIELDS = ("motion_profile", "control_profile")


def default_baseline_id(environment: str | None) -> str:
    """The baseline a RunSpec for ``environment`` gets when it names none (``ENVIRONMENT_BASELINES``)."""
    return ENVIRONMENT_BASELINES.get(environment, BASELINE_ID)


def load_native_baseline(root: Path, baseline_id: str = BASELINE_ID) -> dict[str, Any]:
    """Return the tracked baseline and its exact catalog identity; no probes.

    An instance names its runtime motion and control profiles by path; the
    returned catalog carries the ``{path, sha256}`` record of each, hashed
    from the bytes read here, which is what the RunSpec runtime receives.

    A catalog may name the code mode a RunSpec that names no ``policy.code_mode`` gets (``code_mode``, a
    ``policy.code_mode`` value; ``spec/defaults.py`` applies it, ``agent/code_mode/contract.py`` resolves it).
    """
    path = BASELINE_CATALOGS[baseline_id]
    raw = (root / path).read_bytes()
    value = json.loads(raw)
    required = (
        frozenset({"robodojo"})
        if baseline_id in TOOLKIT_BASELINE_IDS else REQUIRED_INSTANCES
    )
    if (
        not isinstance(value, dict)
        or value.get("schema") != "ora.native-baseline.v1"
        or value.get("id") != baseline_id
        or not required <= set(value.get("instances", {})) <= required | OPTIONAL_INSTANCES
        or not isinstance(value.get("code_mode", {}), dict)
    ):
        raise ValueError("invalid Native baseline catalog")
    for name, instance in value["instances"].items():
        runtime = instance.get("runtime", {})
        for field in RUNTIME_PROFILE_FIELDS:
            if field in runtime:
                runtime[field] = locked_asset(
                    root, runtime[field], label=f"{baseline_id} {name} runtime.{field}",
                ).record()
    return {**value, "path": path, "sha256": hashlib.sha256(raw).hexdigest()}


def baseline_policy(root: Path, catalog: Mapping[str, Any]) -> dict[str, Any]:
    """A baseline's whole policy: its own fields, then those of the context scheme it names.

    The scheme replaces every ``CONTEXT_FIELDS`` entry, as ``expand_native_defaults`` applies it to a RunSpec. A
    consumer that composes its policy from a baseline outside that expansion (YAM free play) reads it here.
    """
    own = copy.deepcopy(dict(catalog["policy"]))
    named = catalog.get("context_scheme")
    if named is None:
        return own
    return {**{key: value for key, value in own.items() if key not in CONTEXT_FIELDS},
            **copy.deepcopy(load_context_scheme(root, named)["policy"])}
