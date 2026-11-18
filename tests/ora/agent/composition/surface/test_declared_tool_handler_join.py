"""Every tool a registered surface advertises can actually be dispatched.

Nothing else in the suite joins these two halves. A surface manifest lists the
tools the model is shown; the toolkit decides what to do with a name when the
model calls it. The loader checks the manifest against itself -- order, schema
references, mutation subsets -- and never against the toolkit, so a tool can be
advertised to a policy by a profile no test invokes and fail only in a live
episode.

Every tool now runs on the toolkit: ``dispatch.install`` binds each
advertised name to its toolkit tool and refuses a surface that names one the
toolset does not have, and both runtimes' ``invoke`` refuse a toolkit-only
surface the factory did not install. So the join is two facts about every
profile in the catalog: the toolkit serves it (``served``), and each tool it
advertises is a toolkit tool.
"""

from __future__ import annotations

from tests.support.agent.surface_catalog import catalog_grants

import pytest

from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.agent.toolkit.host.runtime.dispatch import ROBOT_TOOLS, TOOLS, served


@pytest.mark.parametrize("profile_id", sorted(PROFILE_CATALOG))
def test_every_registered_profile_is_served_by_the_toolkit(profile_id: str) -> None:
    """A profile the toolkit does not serve has no handler left for any of its tools."""

    surface = resolve_tool_surface(profile_id, toolkit_grants=catalog_grants(profile_id))
    assert served(surface), (
        f"{profile_id} does not declare toolkit_only_v1, so the toolkit does not serve it "
        f"and no runtime has a handler for its tools"
    )


@pytest.mark.parametrize("profile_id", sorted(PROFILE_CATALOG))
def test_every_declared_tool_of_a_registered_profile_is_a_toolkit_tool(profile_id: str) -> None:
    """A profile may not advertise a tool the toolset does not have."""

    surface = resolve_tool_surface(profile_id, toolkit_grants=catalog_grants(profile_id))
    declared = {item["name"] for item in surface.definitions}
    missing = sorted(declared - (TOOLS.keys() | ROBOT_TOOLS.keys()))
    assert not missing, (
        f"{profile_id} advertises {missing} to the policy, but the toolset has no tool "
        f"of that name, so dispatch.install refuses the surface. Either add the "
        f"tool or stop declaring it."
    )
