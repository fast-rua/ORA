"""Full fixed-data grasp results, compiled pools and review artifacts.

Each answer is checked against its digest in ``tests/data/toolkit-goldens``, recorded while the handlers answered it
alike.
"""
import json
import pytest
from ora.agent.toolkit.tools.vision_ground.tool import run as ground
from ora.agent.toolkit.tools.grasp_plan.tool import run
from ora.agent.toolkit.host.failures import invoke
from tests.support.agent.toolkit_tools import make_host, api_for
from tests.support.agent.toolkit_goldens import Goldens

GOLDEN = Goldens(__file__)


@pytest.fixture
def golden(request):
    return GOLDEN.checker(request)


@pytest.mark.parametrize('arm',['left','right'])
@pytest.mark.parametrize('count',[0,3])
def test_fixed_pool_result_and_review_artifacts(golden,tmp_path,monkeypatch,arm,count):
    host,client,requests=make_host(tmp_path/'run',monkeypatch,count)
    try:
        api=api_for(host)
        grounded=ground(api,dict(camera='head',text='object',max_candidates=1))
        args=dict(target_ref=grounded['candidates'][0]['target_ref'],arm=arm,max_candidates=3)
        actual=invoke(run,api,args)
        golden('result',actual,tmp_path/'run')
        query=actual['grasp_query_ref']
        record=json.loads(api.tool_data.outputs['grasp_query:'+query])
        golden('artifacts',{name:api.read_blob(ref) for name,ref in record['files'].items()},tmp_path/'run')
        golden('service_request',requests[-1],tmp_path/'run')
    finally:client.close()


# RO-27: on a grasp_pose_world_axes_v1 surface (every default with grasp_plan, user decision 2026-11-06) every
# published pose carries the world directions of the controlled point's approach axis and jaw line, so the policy no
# longer converts the quaternion by hand.
AXES = ("approach_world", "jaw_line_world")


def _planned(root, key, monkeypatch):
    """grasp_plan on ``key``'s stand-in, its learned pool answering, on a 5 cm strip of the block a grasp fits."""
    import numpy as np
    from tests.support.agent.stack_parity import BLOCK, SIZE, SURFACES, Call, Perception, answered

    def strip(self):
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        mask[BLOCK[0]:BLOCK[1], BLOCK[2] + 10:BLOCK[2] + 16] = True
        return mask

    monkeypatch.setattr(Perception, "_mask", strip)
    arm = {} if SURFACES[key].flat else {"arm": "right"}  # a dual-arm surface names the arm it plans for
    return answered(root / key, key, Call("grasp_plan", arm, needs="target"), monkeypatch)["result"]


def _poses(result):
    return [row[key] for row in result["candidates"] for key in ("pregrasp_pose", "contact_pose") if key in row]


#: Each default's stand-in (``tests/support/agent/stack_parity.SURFACES``) that has grasp_plan.
GRASP_DEFAULTS = ("libero", "robotwin", "robodojo", "robocasa", "robolab")


def _convention(key):
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from ora.agent.composition.surface.toolkit_surface import declared_required_grants
    from tests.support.agent.stack_parity import SURFACES

    profile = SURFACES[key].profile
    surface = resolve_tool_surface(profile, toolkit_grants=declared_required_grants(profile))
    assert surface.supports("grasp_pose_world_axes_v1"), profile
    return surface.view_config["controlled_frame_convention"]


@pytest.mark.parametrize("key", GRASP_DEFAULTS)
def test_every_default_s_poses_carry_their_world_axes(key, tmp_path, monkeypatch):
    from ora.agent.toolkit.host.reach.alignment import frame_local_axes
    from ora.agent.toolkit.lib._pose_math import quaternion_matrix
    from ora.agent.toolkit.tools.move_pose._axes import quaternion_from_world_axes

    local = frame_local_axes(_convention(key))
    poses = _poses(_planned(tmp_path, key, monkeypatch))
    assert len(poses) >= 2
    for pose in poses:
        assert set(pose) == {"position_m", "orientation_xyzw", *AXES}
        matrix = quaternion_matrix(pose["orientation_xyzw"])
        for name, axis in zip(AXES, (local["approach"], local["closing"])):
            expected = [sum(matrix[row][k] * axis[k] for k in range(3)) for row in range(3)]
            assert max(abs(a - b) for a, b in zip(pose[name], expected)) <= 1e-6, name
            assert abs(sum(a * a for a in pose[name]) - 1.0) <= 1e-5, name
        assert abs(sum(a * b for a, b in zip(*(pose[name] for name in AXES)))) <= 1e-5
        # Copied into a world_axes target they compile back to the published orientation.
        compiled = quaternion_from_world_axes(*(pose[name] for name in AXES), local_axes=local, arm="right")
        assert abs(abs(sum(a * b for a, b in zip(compiled, pose["orientation_xyzw"]))) - 1.0) <= 1e-6


def test_a_context_without_the_axes_publishes_its_poses_unchanged(tmp_path, monkeypatch):
    """A tool context without ``pose_world_axes`` (a run sealed before the capability) publishes the bare pose."""
    from ora.agent.toolkit.tools.grasp_plan import tool

    published, real = [], tool.public_result
    monkeypatch.setattr(tool, "public_result", lambda *args: published.append(args) or real(*args))
    assert _poses(_planned(tmp_path, "libero", monkeypatch))
    api, manifest, record, config, *rest = published[-1]
    bare = real(api, manifest, record, {key: value for key, value in config.items() if key != "pose_world_axes"}, *rest)
    assert _poses(bare) and all(set(pose) == {"position_m", "orientation_xyzw"} for pose in _poses(bare))


@pytest.mark.parametrize("key", GRASP_DEFAULTS)
def test_every_default_s_result_schema_requires_both_axes(key, tmp_path, monkeypatch):
    import copy
    from ora.agent.composition.surface.schema_validation import SchemaViolation, validate_schema_instance
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from ora.agent.composition.surface.toolkit_surface import declared_required_grants
    from tests.support.agent.stack_parity import SURFACES

    profile = SURFACES[key].profile
    schema = resolve_tool_surface(profile, toolkit_grants=declared_required_grants(profile)).result_schemas["grasp_plan"]
    candidate = {"$defs": schema["$defs"], **schema["$defs"]["candidate"]}
    row = _planned(tmp_path, key, monkeypatch)["candidates"][0]
    validate_schema_instance(row, candidate)
    for pose_key in ("pregrasp_pose", "contact_pose"):
        for name in AXES:
            damaged = copy.deepcopy(row)
            del damaged[pose_key][name]
            with pytest.raises(SchemaViolation, match=name):
                validate_schema_instance(damaged, candidate)
