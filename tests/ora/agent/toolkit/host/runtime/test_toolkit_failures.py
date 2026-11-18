"""New interface service failures retain the old model-visible tool RPC envelope."""
import numpy as np
import pytest

from ora.agent.model_services.clients import ModelServiceError
from ora.agent.pi.tool_rpc import _tool_error
from ora.agent.toolkit.lib.perception import ground
from ora.agent.toolkit.lib.values import JsonValue
from ora.agent.toolkit.interface.types import canonical
from ora.agent.error_classes import ToolError
from ora.agent.toolkit.host.shared.artifact_errors import ArtifactHashMismatch, TargetInsufficient
from ora.agent.toolkit.host.shared.service_refusals import refusal
from ora.agent.toolkit.host.failures import service_adapter, invoke
from tests.support.agent.toolkit_lib import make_runtime


@pytest.mark.parametrize('error', [ValueError('invalid mask'), TargetInsufficient('too few pixels'),
    ArtifactHashMismatch('source bytes drifted'), ToolError('camera unavailable', code='invalid_camera'),
    ModelServiceError('timeout', service='SAM3', failure='transient', status=503),
    ModelServiceError('identity changed', service='SAM3', failure='permanent'),
    ModelServiceError('unsupported prompt', service='SAM3', failure='request', status=422,
                      detail={'field':'text'})])
def test_original_public_error_and_rpc_projection_are_preserved(tmp_path, error):
    runtime, _, _ = make_runtime(tmp_path)
    def fail(*args):
        raise error
    runtime.adapters['ground'] = service_adapter(fail, 'grounding_failed', target_ref='target-old')
    ref = JsonValue.of(runtime.save_array(np.zeros((4,4,3), dtype=np.uint8)))
    def new_tool(api, args):
        return ground(api, rgb=ref, text='cup')
    expected = error if isinstance(error, ToolError) else refusal(error, code='grounding_failed', target_ref='target-old')
    with pytest.raises(ToolError) as actual:
        invoke(new_tool, runtime, {})
    assert actual.value.as_dict() == expected.as_dict()
    assert canonical(_tool_error(actual.value)) == canonical(_tool_error(expected))


@pytest.mark.parametrize('error',[OSError('publish failed'),ValueError('invalid persisted JSON'),ToolError('closed publication',code='stale_observation')])
def test_control_error_projection_matches_gateway(tmp_path,error):
    from ora.agent.toolkit.host.failures import control_adapter
    from ora.agent.toolkit.lib.control import finish_episode
    runtime, _, _ = make_runtime(tmp_path)
    def fail(*args):
        raise error
    runtime.adapters['finish_episode'] = control_adapter(fail)
    def tool(api,args):
        return finish_episode(api,JsonValue.of({})).data
    with pytest.raises(ToolError) as actual:
        invoke(tool,runtime,{})
    assert canonical(_tool_error(actual.value)) == canonical(_tool_error(error))
