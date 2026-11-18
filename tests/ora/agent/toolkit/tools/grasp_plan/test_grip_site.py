"""grasp_plan on a grip-site surface: the toolkit answers as it did beside the handlers.

``tests/ora/agent/toolkit/test_stack_parity.py`` checks one call on the stand-in's learned pool. This checks what that
row does not reach: a pool from the learned service, an empty one, the arguments, and a target the fallback family
cannot use. Each answer is checked against its digest in ``tests/data/toolkit-goldens``, recorded while the handlers
answered it alike.
"""
import httpx
import numpy as np
import pytest

from ora.agent.toolkit.host.grasp.contact_graspnet_client import ContactGraspNetApiProposer
from ora.agent.toolkit.host.grasp.proposers import CONTACT_FIRST
from ora.agent.toolkit.host.runtime.tool_data import grasp_configuration
from ora.agent.toolkit.tools.grasp_plan._proposals import CONTACT_FIRST as TOOL_CONTACT_FIRST, hybrid
from ora.agent.toolkit.tools.grasp_plan._topdown import PcaTopDownGraspProposer as ToolProposer, TargetInsufficient
from tests.support.agent.stack_parity import (
    BLOCK, SIZE, Call, Perception, on_toolkit, outcome, prepare, stand_in,
)
from tests.support.agent.toolkit_goldens import Goldens
from tests.support.agent.toolkit_tools import grasp_service

GRIP_SITE = ("libero", "robocasa", "robolab")
GOLDEN = Goldens(__file__)


@pytest.fixture
def golden(request):
    return GOLDEN.checker(request)


@pytest.fixture(autouse=True)
def graspable_strip(monkeypatch):
    """The stand-in block is 22 cm wide, past any opening, so the target is a 5 cm strip of it, which a top-down grasp fits."""

    def mask(self):
        strip = np.zeros((SIZE, SIZE), dtype=bool)
        strip[BLOCK[0]:BLOCK[1], BLOCK[2] + 10:BLOCK[2] + 16] = True
        return strip

    monkeypatch.setattr(Perception, "_mask", mask)


def _serving(host, pool):
    """``host`` with the learned service answering ``pool`` candidates."""
    contact = getattr(host.grasp_backend, "contact_backend", host.grasp_backend)
    if isinstance(contact, ContactGraspNetApiProposer):
        contact._client.close()
        object.__setattr__(contact, "_client", httpx.Client(transport=httpx.MockTransport(grasp_service([], pool))))
    return host


def _answered(root, key, call, monkeypatch, *, pool=3):
    """``call`` on the toolkit after what it needs, the learned service answering ``pool`` candidates."""
    host = on_toolkit(_serving(stand_in(root / "toolkit", key, monkeypatch), pool))
    prepared, start = prepare(host, key, call)
    return outcome(host, prepared, start)




@pytest.mark.parametrize("key", GRIP_SITE)
@pytest.mark.parametrize("pool", [3, 0])
def test_a_grip_site_surface_plans_in_its_own_spelling(golden, tmp_path, monkeypatch, key, pool):
    new = _answered(tmp_path, key, Call("grasp_plan", {}, needs="target"), monkeypatch, pool=pool)
    golden("plan", new)
    result = new["result"]
    assert "arm" not in result and "source_origin_to_controlled_point_m" in result["compiler"]
    assert result["status"] == ("no_candidate" if pool == 0 and key != "libero" else "candidates")
    for row in result["candidates"]:  # what a move_pose call copies, in the surface's own spelling
        assert row["approach_delta"]["position_mode"] == "delta" and "right_handed_grip_site_axes" in row["compiler_checks"]


@pytest.mark.parametrize("pool", [3, 0])
def test_robotwin_plans_on_its_grip_centre_compiler(golden, tmp_path, monkeypatch, pool):
    new = _answered(tmp_path, "robotwin", Call("grasp_plan", {"arm": "right"}, needs="target"), monkeypatch, pool=pool)
    golden("plan", new)
    assert new["result"]["arm"] == "right" and new["result"]["compiler"]["arm"] == "right"


def test_the_fallback_family_answers_only_an_empty_pool(tmp_path, monkeypatch):
    learned = _answered(tmp_path / "learned", "libero", Call("grasp_plan", {}, needs="target"), monkeypatch)["result"]
    fallback = _answered(tmp_path / "empty", "libero", Call("grasp_plan", {}, needs="target"), monkeypatch, pool=0)["result"]
    assert learned["diagnostics"]["candidate_family_counts"]["selected"] == 1
    counts = fallback["diagnostics"]["candidate_family_counts"]
    assert counts["learned_available"] == 0 and counts["selected"] == counts["public_rgbd_topdown_available"] > 0


@pytest.mark.parametrize("arguments", [{"arm": "right"}, {"max_candidates": 4}, {"max_candidates": 1}])
def test_the_arguments_shape_the_plan(golden, tmp_path, monkeypatch, arguments):
    new = _answered(tmp_path, "libero", Call("grasp_plan", arguments, needs="target"), monkeypatch)
    golden("plan", new)


def test_a_target_the_fallback_cannot_use_costs_only_that_family(golden, tmp_path, monkeypatch):
    def insufficient(self, **_):
        raise TargetInsufficient("grasp proposal has too few valid target points")

    monkeypatch.setattr(ToolProposer, "propose", insufficient)
    new = _answered(tmp_path, "libero", Call("grasp_plan", {}, needs="target"), monkeypatch)
    golden("plan", new)
    assert new["result"]["diagnostics"]["public_rgbd_topdown"]["warning"] == (
        "grasp proposal has too few valid target points")


def test_a_target_a_top_down_first_hybrid_cannot_use_names_the_argument(golden, tmp_path, monkeypatch):
    """The refusal keeps its field and says only a different target can succeed."""
    def insufficient(self, **_):
        raise TargetInsufficient("grasp proposal has too few valid target points")

    monkeypatch.setattr(ToolProposer, "propose", insufficient)
    new = _answered(tmp_path, "robodojo", Call("grasp_plan", {"arm": "right"}, needs="target"), monkeypatch)
    golden("refusal", new)


def test_only_a_contact_first_surface_has_the_selection_key(tmp_path, monkeypatch):
    libero, dojo = (grasp_configuration(stand_in(tmp_path / key, key, monkeypatch)) for key in ("libero", "robodojo"))
    assert libero["selection"] == CONTACT_FIRST == TOOL_CONTACT_FIRST
    assert "selection" not in dojo


def test_a_hybrid_keeps_its_released_order_without_the_key():
    learned, geometric = [dict(candidate_id=f"l{i}") for i in range(3)], [dict(candidate_id=f"g{i}") for i in range(3)]
    assert [row["candidate_id"] for row in hybrid(learned, geometric, 3)] == ["g0", "g1", "l0"]
    assert [row["candidate_id"] for row in hybrid(learned, geometric, 3, contact_first=True)] == ["l0", "l1", "l2"]
    assert [row["candidate_id"] for row in hybrid([], geometric, 2, contact_first=True)] == ["g0", "g1"]
