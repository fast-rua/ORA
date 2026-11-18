"""Call-local configuration and read-only IK bridge; no runtime host object.

Two questions every moving tool asks of a call's context. Their answers are RoboDojo's for a context that does not say
otherwise, which is what lets a sealed run replay: only a surface whose environment plans its moves has the key, and
only a single-arm surface declares the capability.
"""

from dataclasses import dataclass

from ...lib.values import PublicResult
from ...interface.types import ReferenceFailure, PublicToolFailure
from .. import _latency

#: A surface that drives one right arm: its arguments are that arm's target, with no ``targets`` wrapper.
SINGLE_ARM = "single_arm_flat_targets_v1"
BOTH_ARMS = ("left", "right")


def environment_plans(config):
    """Whether the environment plans this call's move as it executes it: the tool proposes one action and returns."""
    return bool((config.get("motion") or {}).get("environment_plans"))


def arms(config):
    """The arms of this call's surface."""
    return ("right",) if SINGLE_ARM in (config.get("motion") or {}).get("capabilities", ()) else BOTH_ARMS


@dataclass(frozen=True)
class Surface:
    capabilities: tuple[str, ...]
    view_config: dict
    local_axes: dict
    jaw_offsets: dict

    def supports(self, name):
        return name in self.capabilities


@dataclass(frozen=True)
class Context:
    api: object
    surface: Surface
    workspace_bounds_m: dict

    reach_available: bool
    #: Whether this run asks the reach reference before a move (``_latency``): off, ``_reach`` asks the IK nothing for
    #: ``reach_along_request`` and says so.
    reach_reference: bool = True

    @property
    def align_plan_callback(self):
        return self.query if self.reach_available else None

    def query(self, request):
        response = self.api.planner.reach_check(request=request)
        failure = response.json().get("error") or {}
        if "reference_failure" in failure:
            raise ReferenceFailure(failure["reference_failure"]["exception_type"])
        if "tool_error" in failure:
            raise PublicToolFailure(failure["tool_error"])
        return PublicResult.of(response).data.json()


def from_configuration(api, config, *, reach_reference=True):
    """The context of one call from its ``motion`` configuration; ``reach_reference`` is the call's latency switch."""
    return Context(
        api,
        Surface(
            tuple(config["capabilities"]),
            config["view_config"],
            config["local_axes"],
            config["jaw_offsets"],
        ),
        config["workspace_bounds_m"],
        config["reach_available"],
        reach_reference,
    )
