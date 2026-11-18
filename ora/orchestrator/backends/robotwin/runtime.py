"""The RoboTwin runtime of a RunSpec: its own fields, after the checks every runtime shares (spec.runtime)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Mapping

from ora.environment.controllers.profile_capabilities import (
    OLDEST_COMPOSED_CONTROL_PROFILE, bound_control_profile_version,
)
from ora.orchestrator.spec.values import RunSpecError, _positive_int

if TYPE_CHECKING:
    from ora.agent.composition import ResolvedToolSurface


def resolve_runtime(
    runtime: dict[str, Any], policy: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface, *, task: str, seed: int,
    task_instruction: str,
) -> None:
    if runtime["transport"] != "local_rpc":
        raise RunSpecError("robotwin runtime.transport must be local_rpc")
    for field in ("root_env", "python_env", "required_revision", "task_config"):
        if not isinstance(runtime.get(field), str) or not runtime[field]:
            raise RunSpecError(f"runtime.{field} must be non-empty")
    runtime.setdefault("startup_timeout_s", 600)
    runtime.setdefault("request_timeout_s", 1800)
    for field in ("startup_timeout_s", "request_timeout_s"):
        runtime[field] = _positive_int(runtime[field], f"runtime.{field}")
    if policy["action_codec"] == "motion_action_v3":
        require_sparse_motion_topology(runtime)
    elif "control_profile" in runtime:
        raise RunSpecError(
            "runtime.control_profile is valid only with a sparse motion action"
        )
    if "reach_check_inter_arm_screen" in runtime:
        raise RunSpecError("runtime.reach_check_inter_arm_screen was retired: the other arm stands in the planner's world")


def require_sparse_motion_topology(runtime: Mapping[str, Any]) -> None:
    """Sparse motion binds a control profile and the qualified cameras."""

    control_profile = runtime.get("control_profile")
    if not isinstance(control_profile, Mapping) or set(control_profile) != {
        "path", "sha256",
    }:
        raise RunSpecError(
            "sparse motion action requires runtime.control_profile path and sha256"
        )
    if (
        not isinstance(control_profile.get("path"), str)
        or not control_profile["path"]
        or not isinstance(control_profile.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", control_profile["sha256"]) is None
    ):
        raise RunSpecError("runtime.control_profile binding is invalid")
    version = bound_control_profile_version(control_profile)
    if version is not None and version < OLDEST_COMPOSED_CONTROL_PROFILE:
        raise RunSpecError(
            f"runtime.control_profile v{version} is older than the oldest control "
            f"profile this tree composes (v{OLDEST_COMPOSED_CONTROL_PROFILE}); a run "
            "bound to it composes and evaluates only at the commits that composed it"
        )
    if runtime.get("policy_observer_camera", False) is not False:
        raise RunSpecError("runtime.policy_observer_camera is not supported")
    expected_cameras = ["head_camera", "front_camera", "left_camera", "right_camera"]
    if runtime.get("policy_cameras") != expected_cameras:
        raise RunSpecError(
            "sparse motion policy cameras must match the qualified RoboTwin order"
        )
    if runtime.get("main_camera") != "head_camera":
        raise RunSpecError("sparse motion main_camera must be head_camera")


def expand_template(runtime: Mapping[str, Any], template: dict[str, Any], requested: str | None) -> dict[str, Any]:
    """The baseline instance's runtime template for this RunSpec (defaults.expand_native_defaults).

    Both official scene configurations evaluate the same composition; the
    instance names the default one and the RunSpec may select the other. A
    frozen baseline keeps its exact prior contract, which for the ones listed
    in ``SCENE_CONFIGURATION_CHOICE_BASELINES`` included this choice.
    """

    from ora.environment.benchmarks.robotwin import TASK_CONFIGS
    from ora.orchestrator.spec.defaults import SCENE_CONFIGURATION_CHOICE_BASELINES

    task_config = runtime.get("task_config")
    if ((requested is None or requested in SCENE_CONFIGURATION_CHOICE_BASELINES)
            and isinstance(task_config, str) and task_config in TASK_CONFIGS):
        template.pop("task_config", None)
    return template


__all__ = ["expand_template", "require_sparse_motion_topology",
           "resolve_runtime"]
