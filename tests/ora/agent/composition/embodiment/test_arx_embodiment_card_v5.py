"""ARX X5 embodiment card v5: the public jaw reading is both fingers' mean (E11G12-001).

The v4 card said "The public encoder reading is normalized joint position",
which the backend published from the first finger alone. Now the
public opening is the mean of both finger joints on the same normalized command
scale, and the finger pair can shift to one side under load. Descriptor v5
changes only the card projection (``full_finger_reference_v3`` = v2 plus that
sentence); geometry, axes, controlled points and provenance are v4's, so the
loaded-geometry evidence stays reusable. Descriptor v6 is v5 with the gripper
module's provenance path at the toolkit host; it compiles the same card, so the
live Dojo bundles, environment profiles v6 and v7 and both toolkit surfaces pin
v6 with card v5, and v5 stays for the runs sealed with it. The v4 predecessor
left the open-source tree, so the v4-equivalence comparisons are no longer
pinned here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ora.agent.composition.embodiment import load_embodiment_descriptor, render_embodiment_card
from ora.agent.composition.embodiment.descriptor import FULL_FINGER_TOKEN_LIMIT

ROOT = Path(__file__).resolve().parents[5]
ARX = ROOT / "ora/policy/embodiments/arx-x5"
V5 = ARX / "descriptor-public-geometry-v5.json"
V6 = ARX / "descriptor-public-geometry-v6.json"
CARD_V5 = ARX / "manual-public-geometry-v5.md"
LIVE_BUNDLE = ROOT / "ora/policy/prompt_bundles/toolkit_v6/robodojo.yaml"
ENVIRONMENTS = ROOT / "ora/policy/environments/robodojo"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_compiled_card_states_the_two_finger_encoder_sentence():
    card = render_embodiment_card(load_embodiment_descriptor(V5))
    assert card == CARD_V5.read_text(encoding="utf-8")
    text = _flat(card)
    assert "normalized joint position" not in text
    assert ("The public encoder reading is the mean of both finger joints on the normalized command scale "
            "(empty-close reference about 0.185, full-open 1); it is not measured pad separation, and a pair "
            "shifted to one side is no evidence of holding or release.") in text
    assert (len(card.encode()) + 3) // 4 <= FULL_FINGER_TOKEN_LIMIT


def test_the_live_referrers_pin_v6():
    from tests.support.agent.surface_catalog import surface_of
    from ora.agent.composition.embodiment.geometry import PUBLIC_GEOMETRY_DESCRIPTORS
    from ora.agent.composition.embodiment.environment_context import resolve_environment_prompt_profile
    from ora.environment.benchmarks.robodojo import public_context

    bundle = yaml.safe_load(LIVE_BUNDLE.read_text(encoding="utf-8"))
    assert bundle["layers"]["embodiment"]["source"] == str(CARD_V5.relative_to(ROOT))
    assert bundle["layers"]["embodiment"]["descriptor"] == str(V6.relative_to(ROOT))
    profile_id = bundle["bindings"]["environment_prompt_profile"]["profile_id"]
    assert profile_id == "robodojo_public_explicit_v6"
    profile = json.loads((ENVIRONMENTS / "context-profile-public-explicit-v6.json").read_text())
    compiled = profile["compiled_embodiment"]
    # Named by path; the resolved bindings carry the sha256 of the bytes read.
    assert compiled["descriptor"] == str(V6.relative_to(ROOT))
    assert compiled["card_sha256"] == _sha(CARD_V5)
    review = json.loads((ROOT / compiled["review"]).read_text())
    # The review attests the card text; the descriptor's bytes are policy.lock's.
    assert "descriptor_sha256" not in review and review["card_sha256"] == _sha(CARD_V5)
    activation = json.loads((ROOT / "tests/data/default-expectations.json").read_text())["instances"]["robodojo"]
    assert activation["public_geometry_descriptor"] == str(V6.relative_to(ROOT))
    for profile in ("robodojo_placement_toolkit_v5", activation["tool_surface"]):
        surface = surface_of(profile)
        assert surface.view_config["public_geometry_descriptor"] == {"path": compiled["descriptor"], "sha256": _sha(V6)}
    assert PUBLIC_GEOMETRY_DESCRIPTORS["arx_x5_link6"] == "arx-x5/descriptor-public-geometry-v6.json"
    resolved = resolve_environment_prompt_profile(public_context(), project_root=ROOT, profile_id=profile_id)
    assert resolved["embodiment_policy"] == str(CARD_V5.relative_to(ROOT))


def test_v6_is_v5_with_the_gripper_source_at_the_toolkit_host():
    v5, v6 = json.loads(V5.read_text()), json.loads(V6.read_text())
    gripper = "arx_x5_reviewed_nominal_gripper_parameters"
    moved = {"uri": "repo://ora/agent/toolkit/host/grasp/arx_x5_gripper.py",
             "local_path": "ora/agent/toolkit/host/grasp/arx_x5_gripper.py"}
    expected = {**v5, "descriptor_id": "arx_x5_public_geometry_v6", "provenance": {**v5["provenance"], "sources": [
        {**source, **moved} if source["source_id"] == gripper else source for source in v5["provenance"]["sources"]]}}
    assert v6 == expected
    assert render_embodiment_card(load_embodiment_descriptor(V6)) == CARD_V5.read_text(encoding="utf-8")
