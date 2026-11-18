"""A hardcoded palette must not silently decide how many candidates are shown.

``grasp.query`` validates ``max_candidates`` up to four. The camera overlay held
three colours and consumed them with ``zip(..., strict=False)``, so a fourth
candidate drew nothing at all — no wireframe, no label, no error — while the
global review panel, holding four colours, showed it. Two renderings of one
candidate list disagreed and nothing announced it.

Both renderers now read one table, and the drawable bound is derived from that
table rather than duplicated beside it.
"""

from __future__ import annotations

from pathlib import Path

import ora.agent.model_services.clients as service_clients
from ora.agent.toolkit.host.perception import grounding_palette


def test_grounding_request_and_rendering_bounds_are_independent():
    """A palette extension must not advertise unsupported service requests."""
    assert grounding_palette.GROUNDING_MAX_CANDIDATES == len(grounding_palette.GROUNDING_CANDIDATE_COLOURS)
    assert grounding_palette.GROUNDING_MAX_CANDIDATES >= service_clients.GROUNDING_REQUEST_MAX_CANDIDATES
    source = Path(service_clients.__file__).read_text(encoding="utf-8")
    assert "1 <= max_candidates <= GROUNDING_REQUEST_MAX_CANDIDATES" in source
