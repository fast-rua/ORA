"""Test helpers for robodojo closed skills default, shared by several test modules.

They were defined in
``tests/ora/environment/backends/robodojo/test_robodojo_closed_skills_default.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from tests.support.agent.surface_catalog import catalog_grants, handlers_only
from tests.support.agent.toolkit_run import served

import copy
import json
from pathlib import Path
from unittest import mock

import httpx

from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.agent.toolkit.host.grasp.contact_graspnet_client import ContactGraspNetApiProposer
from tests.support.agent.toolkit_tools import grasp_service
from ora.agent.context.observation.contact_points import project_policy_public_facts
from ora.agent.composition.task_contract.robodojo import REGISTRY_PATH
from tests.support.orchestrator.run_specs import robodojo_native_spec
from tests.support.robodojo.perception import (
    BOUNDS, FakeHeadGrounding, FakeX5Proposer, _synthetic_robodojo_observation,
)


ROOT = Path(__file__).resolve().parents[3]
SURFACE = "robodojo_placement_toolkit_v6"
REGISTRY = json.loads((ROOT / REGISTRY_PATH).read_text())["tasks"]


def spec_for(task):
    spec = robodojo_native_spec(task)
    spec["toolkit_grants"] = catalog_grants(SURFACE)
    return spec


class BothArmsProposer(FakeX5Proposer):
    def propose(self, **kwargs):
        assert kwargs["compiler_identity"]["id"] in {
            f"contact_graspnet_to_robodojo_x5_{arm}_link6_v2" for arm in ("left", "right")}
        checked = copy.deepcopy(kwargs)
        checked["compiler_identity"]["id"] = "contact_graspnet_to_robodojo_x5_left_link6_v1"
        return super().propose(**checked)


def _grasp_service(runtime):
    """The surface's own grasp client, its resident service the in-process fake (``toolkit_tools.grasp_service``)."""
    contact = getattr(runtime.grasp_backend, "contact_backend", runtime.grasp_backend)
    contact._client.close()
    object.__setattr__(contact, "_client", httpx.Client(transport=httpx.MockTransport(grasp_service([]))))
    object.__setattr__(contact, "require_ready", lambda: {"fixture": True})


def tool_runtime(tmp_path, *, serve=True, handlers=False):
    """The fake Dojo plant's tool runtime over one synthetic observation, served the way a run serves it
    (``toolkit_run.served``): the surface's own grasp client against the in-process fake service. ``serve=False``
    leaves it constructed for the caller to change and serve. ``handlers=True`` serves it from the legacy tool
    handlers, with their proposer test double, for the tests that leave with them."""
    source, observation = _synthetic_robodojo_observation(tmp_path / "observations")
    control = tmp_path / "control"
    control.mkdir()
    with mock.patch.object(ContactGraspNetApiProposer, "require_ready", lambda self: {"fixture": True}):
        runtime = ManipulationToolRuntime(control_root=control,
            manual_path=ROOT / "ora/policy/prompt_bundles/templates/tool-schema-delivery-v1.md",
            event_path=tmp_path / "events.jsonl", workspace_bounds_m=BOUNDS, state_root=None,
            tool_surface_profile=SURFACE, toolkit_grants=catalog_grants(SURFACE), controlled_end_effector="right",
            segmentation_backend=FakeHeadGrounding(), grounding_backend=FakeHeadGrounding(),
            grasp_backend=BothArmsProposer() if handlers else None,
            public_facts_projector=project_policy_public_facts)
    if not handlers:
        _grasp_service(runtime)
    pointer = {"open": True, "observation_id": observation["observation_id"],
        "observation_path": str(source / "observation.json")}
    runtime._current = lambda: (pointer, source, observation)
    if handlers:
        return handlers_only(runtime)
    return served(runtime) if serve else runtime
