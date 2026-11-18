"""Surface projection/version differences warn; corrupt frozen evidence fails."""

from copy import deepcopy

import pytest

from tests.support.agent.surface_catalog import surface_of
from ora.agent.toolkit.interface.types import CorruptRecord
from ora.orchestrator.toolkit.surface_audit import audit


def test_runtime_version_changes_are_nonblocking_with_exact_diff_values():
    current = surface_of("robodojo_placement_toolkit_v5")
    frozen = deepcopy(current.snapshot())
    assert audit(frozen, current)["status"] == "equal"
    frozen["definitions"][0]["description"] = "Older implementation description"
    frozen["view_config"]["future_option"] = None
    report = audit(frozen, current)
    assert report["status"] == "different" and report["blocking"] is False
    assert report["source"]["sha256"] != report["current"]["sha256"]
    differences = {row["path"]: row for row in report["differences"]}
    assert (
        differences["/definitions/0/description"]["source"]
        == "Older implementation description"
    )
    assert differences["/view_config/future_option"]["source_present"] is True
    assert differences["/view_config/future_option"]["current_present"] is False


def test_malformed_frozen_surface_still_fails():
    current = surface_of("robodojo_placement_toolkit_v5")
    frozen = deepcopy(current.snapshot())
    frozen["definitions"] = "not a tool definition list"
    with pytest.raises(CorruptRecord, match="malformed frozen"):
        audit(frozen, current)
