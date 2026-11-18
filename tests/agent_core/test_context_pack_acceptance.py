"""A context pack's reference examples, replayed through the Pi core's view (opt-in).

A pack owner's examples record, for sampled steps of sealed runs, the exact
messages and tools a model trained on the pack must receive
(``tests/support/context_pack/replay.py`` describes the files). The pack and the
examples live outside the repository; the check runs only when these variables
name them, and is skipped with the reason otherwise:

=====================================  ===================================================
``ORA_TEST_CONTEXT_PACK_DIR``          the pack directory (its ``manifest.json`` and files)
``ORA_TEST_CONTEXT_PACK_MANIFEST``     the sha256 of that ``manifest.json``
``ORA_TEST_CONTEXT_PACK_PROFILE``      the delivery profile JSON, for a manifest without one
``ORA_TEST_CONTEXT_PACK_EXAMPLES``     the examples directory
=====================================  ===================================================

Every sample whose reference view did not depend on its own target must match
message for message (images by sha256) and tool for tool.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ora.agent.composition.prompt import context_pack as cp
from tests.support.context_pack import replay

pytestmark = pytest.mark.usefixtures("pi_core")

VARIABLES = ("ORA_TEST_CONTEXT_PACK_DIR", "ORA_TEST_CONTEXT_PACK_MANIFEST", "ORA_TEST_CONTEXT_PACK_EXAMPLES")


def test_every_comparable_reference_example_is_reproduced(tmp_path: Path) -> None:
    missing = [name for name in VARIABLES if not os.environ.get(name)]
    if missing:
        pytest.skip(f"{', '.join(missing)} not set (the pack and its reference examples live outside the repository)")
    pack = cp.load_context_pack(Path(os.environ["ORA_TEST_CONTEXT_PACK_DIR"]).resolve(),
                                os.environ["ORA_TEST_CONTEXT_PACK_MANIFEST"])
    profile_path = os.environ.get("ORA_TEST_CONTEXT_PACK_PROFILE")
    profile, _source = cp.resolve_profile(
        pack, json.loads(Path(profile_path).read_text(encoding="utf-8")) if profile_path else None)
    examples = replay.Examples.load(Path(os.environ["ORA_TEST_CONTEXT_PACK_EXAMPLES"]))
    report = replay.replay(pack, profile, examples, scratch=tmp_path)
    print(json.dumps({key: value for key, value in report.as_dict().items() if key not in ("failures", "skipped")}))
    assert report.compared > 0
    assert report.failed == [], report.failed[:3]
    assert report.passed == report.compared == report.total - len(report.skipped)
