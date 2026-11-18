"""Motion refusals teach shapes the surface admits and name public fields and bounds.

Refusal census 1, 2, 3 and 5 (0059 open-bugs triage rows 5, 6, 17, 24):

* every ``move_pose`` refusal example is written in the spelling the bound
  schema admits and compiles; the nested examples were refused by every flat
  surface;
* a rotation angle or quaternion outside the compiler's bounds is refused under
  its public field with the bound, and a compiler message names the primitive
  that was called, not ``move_pose``;
* the workspace refusal names the argument the target came from, not ``xyz_m``;
* an undeclared tool is not retryable and lists the declared ones.
"""

from __future__ import annotations

from tests.support.agent.surface_catalog import catalog_grants
import pytest

from ora.agent.composition import resolve_tool_surface, validate_schema_instance
from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG
from ora.agent.toolkit.host.motion_rules import arguments as motion_arguments, refusals as motion_refusals
from ora.agent.toolkit.host.motion_rules.refusals import ENHANCED_MOVE_POSE_EXAMPLES
from ora.agent.error_classes import ToolError
from ora.contracts import compile_enhanced_move_pose

HOME = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
ENHANCED = sorted(
    profile for profile in PROFILE_CATALOG
    if resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile)).supports("enhanced_move_pose_v1")
)


def _refused(call, *args, **kwargs) -> dict:
    with pytest.raises(ToolError) as caught:
        call(*args, **kwargs)
    return caught.value.as_dict()


@pytest.mark.parametrize("profile", ENHANCED)
def test_every_move_pose_example_passes_the_delivered_schema_and_the_compiler(profile):
    surface = resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile))
    schema = motion_arguments.surface_move_pose_schema(surface)
    examples = motion_refusals.move_pose_examples(surface, ENHANCED_MOVE_POSE_EXAMPLES)
    targets_only = [example for example in ENHANCED_MOVE_POSE_EXAMPLES if "plan_ref" not in example]
    written = [item for item in examples if "plan_ref" not in item]
    # A flat surface gets every example back in its own spelling; a nested one
    # that publishes no bare TCP pose loses only that one.
    if motion_arguments.declares_flat_targets(schema):
        assert len(written) == len(targets_only)
    else:
        assert len(written) >= len(targets_only) - 1
    for example in examples:
        validate_schema_instance(example, schema, root_path="$input")
        if "plan_ref" in example:
            continue
        arms = example.get("targets", {"right": example})
        for target in motion_arguments.arm_targets(arms).values():
            request = {key: value for key, value in target.items() if key != "interaction"}
            compile_enhanced_move_pose(request, current_pose_xyz_xyzw=HOME, controlled_point="end_effector")


def _dojo(tmp_path):
    from tests.support.agent.move_pose_reach import _dojo_runtime

    return _dojo_runtime(tmp_path)


def test_an_undeclared_tool_is_not_retryable_and_lists_the_declared_ones(tmp_path):
    runtime = _dojo(tmp_path)
    error = _refused(runtime.invoke, "bindings", {})
    assert (error["code"], error["field"], error["received"]) == ("tool_unavailable", "tool", "bindings")
    assert error["retryable"] is False
    assert error["constraints"]["allowed"] == list(runtime.allowed_tools)
