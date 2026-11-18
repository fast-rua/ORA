"""Shared native_api causal/atomic Runtime contracts and frozen Synthetic bindings.

The request-assembly cases that drove the Python Native core's assembler are
ported to ``ora-pi/test/request_assembly.test.ts``; the episode runs on the Pi
core with a scripted model.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from ora.agent.composition import resolve_tool_surface
from ora.agent.context import ContextProjector
from ora.agent.session_core.domain import PolicyDomainMixin
from ora.agent.session_core.gateway import NativeToolGateway
from ora.agent.providers.anthropic_request import translate_anthropic_request
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.environment import SyntheticSession
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from tests.support.orchestrator import multi_round
from tests.support.orchestrator.multi_round import live_exp01b_spec
from tests.support.agent.atomic_causal_runtime import ROOT, _runtime


def _reopen_same_observation(runtime: ManipulationToolRuntime) -> None:
    """Test-only reset; production requires a newly published actual observation."""

    path = runtime.control_root / "current-observation.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["open"] = True
    path.write_text(json.dumps(value), encoding="utf-8")


def test_atomic_surface_cannot_construct_pose_gripper_composites(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    surface = resolve_tool_surface("native_atomic_causal_v1")
    assert runtime.definitions() == list(surface.definitions)
    assert NativeToolGateway(runtime).delivered() == runtime.definitions()
    assert surface.mutation_tools == ("move_pose", "close_gripper", "open_gripper")
    assert surface.call_effect_profile == "native_atomic_causal_effects_v1"
    assert runtime.classify_call_effect("preview_move", {}) == "read_only"
    assert runtime.classify_call_effect("move_pose", {}) == "mutation"
    assert runtime.classify_call_effect("close_gripper", {}) == "mutation"
    assert runtime.classify_call_effect("open_gripper", {}) == "mutation"

    preview = runtime.invoke("preview_move", {"delta_mm": [1.0, 0.0, 0.0]})
    assert preview["status"] == "candidate"
    assert not (runtime.control_root / "submitted-action.json").exists()
    moved = runtime.invoke("move_pose", {"preview_id": preview["preview_id"]})
    action = json.loads((runtime.control_root / "submitted-action.json").read_text())
    assert moved == {
        "status": "submitted", "action": "move_pose",
        "_runtime": {"directive": "action_submitted", "action_id": action["action_id"]},
    }
    assert action["schema"] == "ora-action.v3"
    assert action["action"] == "move_pose"
    assert "gripper" not in action["arguments"]

    (runtime.control_root / "submitted-action.json").unlink()
    _reopen_same_observation(runtime)
    runtime.invoke("close_gripper", {})
    close = json.loads((runtime.control_root / "submitted-action.json").read_text())
    assert close["action"] == "set_gripper"
    assert close["arguments"] == {"arm": "right", "gripper": 0.0}
    assert "position_m" not in close["arguments"]

    (runtime.control_root / "submitted-action.json").unlink()
    _reopen_same_observation(runtime)
    runtime.invoke("open_gripper", {})
    opened = json.loads((runtime.control_root / "submitted-action.json").read_text())
    assert opened["arguments"] == {"arm": "right", "gripper": 1.0}


#: The shared policy-view contract the robot defaults declare
#: (``shared_policy_view_config_v1``), lifted onto the Synthetic atomic runtime,
#: whose kept surface does not declare one.
ATOMIC_VIEW_CONFIG = {
    "schema": "ora.policy-view-config.v1",
    "rgb_resolution": "native",
    "derived_resolution": [256, 256],
    "automatic_actual_views": ["agentview"],
    "default_observe_views": ["agentview"],
    "coordinate_convention": "opencv_top_left_xy",
}


def test_atomic_policy_views_share_native_coordinates_and_one_camera_default(
    tmp_path: Path,
) -> None:
    from tests.support.agent.toolkit_run import served

    runtime = _runtime(tmp_path, serve=False)
    runtime.surface = dataclasses.replace(
        runtime.surface, view_config=ATOMIC_VIEW_CONFIG,
        capabilities=(*runtime.surface.capabilities, "shared_policy_view_config_v1"),
    )
    runtime.view_config = dict(ATOMIC_VIEW_CONFIG)
    served(runtime)
    preview = runtime.invoke("preview_move", {"delta_mm": [1.0, 0.0, 0.0]})
    assert preview["view_dimensions"] == {
        "candidate_agentview": {"width": 4, "height": 3},
    }
    observed = runtime.invoke("observe", {})
    assert observed["views"] == ["agentview"]
    assert observed["view_dimensions"] == {
        "agentview": {"width": 4, "height": 3},
    }
    image = Image.open(io.BytesIO(base64.b64decode(observed["images_base64"][0])))
    assert image.size == (4, 3)

    wrist = runtime.invoke("observe", {"views": ["wrist"]})
    assert wrist["views"] == ["wrist"]
    assert wrist["view_dimensions"]["wrist"] == {"width": 4, "height": 3}

    marked = runtime.invoke("mark_point", {
        "point_id": "native-center", "view": "agentview", "x": 2, "y": 1,
    })
    assert marked["status"] == "solved"
    point_store = runtime.toolkit_state_access.read("tool.marks")  # the toolkit keeps its marks in tool state
    source = point_store["native-center"]["sources"][0]
    assert {key: source[key] for key in (
        "view", "x", "y", "source_x", "source_y",
    )} == {
        "view": "agentview", "x": 2, "y": 1,
        "source_x": 2, "source_y": 1,
    }
    assert source["observation_id"] == observed["observation_id"]
    assert source["measurement_quality"]["method"] == "local_supported_depth_v1"
    feedback = Image.open(io.BytesIO(base64.b64decode(marked["images_base64"][0])))
    assert feedback.size == (4, 3)

    pointcloud = runtime.invoke("observe", {"views": ["pointcloud_top"]})
    assert pointcloud["view_dimensions"] == {
        "pointcloud_top": {"width": 256, "height": 256},
    }
    derived = Image.open(io.BytesIO(base64.b64decode(pointcloud["images_base64"][0])))
    assert derived.size == (256, 256)


def test_atomic_preview_is_bound_to_one_actual_observation(tmp_path: Path) -> None:
    from ora.agent.error_classes import ToolError
    from tests.support.agent.toolkit_run import republished

    runtime = _runtime(tmp_path)
    preview = runtime.invoke("preview_move", {"delta_mm": [1.0, 0.0, 0.0]})
    republished(runtime, "a-different-actual-observation")

    try:
        runtime.invoke("move_pose", {"preview_id": preview["preview_id"]})
    except ToolError as exc:
        assert exc.code == "stale_preview"
    else:
        raise AssertionError("stale preview was accepted")


def test_post_mutation_capture_failure_is_world_state_unknown(tmp_path: Path) -> None:
    class FailAfterMutation(SyntheticSession):
        def observe(self, output_root: Path):  # type: ignore[override]
            if self.action_sequence:
                raise RuntimeError("post-action capture failed")
            return super().observe(output_root)

    spec = live_exp01b_spec("synthetic-native-atomic-causal-cache-v1")
    plan = resolve_run_spec(spec, root=ROOT)
    run_root = tmp_path / "failed-run"
    session = FailAfterMutation(
        run_root / "external", plan["task"], 0, max_actions=6,
        main_camera="agentview", camera_names=("agentview", "wrist"),
    )
    # The first call is a direct move_pose: it mutates the world.
    result = multi_round.run(
        run_root, plan, [
            multi_round.tool_reply("lift-0", "move_pose", {"delta_mm": [0.0, 0.0, 10.0]}),
            multi_round.tool_reply("open-0", "open_gripper"),
            multi_round.tool_reply("finish-0", "finish_episode", {
                "outcome": "failure", "reason": "synthetic qualification complete",
                "attempts": ["atomic pose, close, lift, and open completed"],
            }),
        ],
        run_id="native-exp01b-capture-fault",
        environment_session_factory=lambda **_kwargs: session,
    )
    assert result["state"] == "incomplete"
    assert result["termination"] == "trusted_fault"
    interaction = json.loads(
        (run_root / "native/episode-interaction.json").read_text(encoding="utf-8")
    )
    assert interaction["world_state_certainty"] == "unknown"
    assert any(
        transition["to"] == "world_state_unknown"
        for transition in interaction["transitions"]
    )
    assert len(multi_round.provider_requests(run_root)) == 1


def test_causal_pose_effect_distinguishes_miss_from_unknown() -> None:
    planner = object.__new__(PolicyDomainMixin)
    missed = planner._minimal_action_effect({
        "action": {
            "action": "move_pose",
            "arguments": {"position_m": [0.1, 0.0, 1.0]},
        },
        "commanded_pose_xyz_xyzw": [0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "reached_pose_xyz_xyzw": [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    })
    assert missed == {
        "action": "move_pose",
        "status": "not_reached",
        "target_pose_xyz_xyzw": [0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "actual_pose_xyz_xyzw": [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "position_error_m": 0.1,
        "orientation_error_rad": 0.0,
    }
    assert planner._minimal_action_effect({
        "action": {"action": "move_pose", "arguments": {}},
    }) == {"action": "move_pose", "status": "world_state_unknown"}


def test_a_semantic_manifest_is_refused() -> None:
    """No capability manifest is registered, so policy.semantic_manifest is refused."""

    spec = live_exp01b_spec("synthetic-native-atomic-causal-cache-v1")
    spec["policy"]["semantic_manifest"] = {"id": "manipulation_core_draft_v0", "instance": "x"}
    with pytest.raises(RunSpecError, match="semantic_manifest is not supported"):
        resolve_run_spec(spec, root=ROOT)


def test_anthropic_wire_projection_states_object_type_for_oneof_tools() -> None:
    """Pool gateways require input_schema.type == "object"."""

    body = translate_anthropic_request({
        "model": "k3",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "move_pose",
                "description": "d",
                "parameters": {
                    "$defs": {},
                    "oneOf": [
                        {"type": "object", "properties": {},
                         "additionalProperties": False},
                    ],
                },
            },
        }],
    })
    schema = body["tools"][0]["input_schema"]
    assert schema["type"] == "object"
    assert "oneOf" in schema


def test_provider_rotates_host_local_credentials_on_exhaustion() -> None:
    """A quota rejection moves to the next key and retries (host-local pool)."""

    from ora.agent.providers.anthropic_messages import AnthropicMessagesProvider

    class Stream:
        def __init__(self, status: int, body: bytes) -> None:
            self.status_code = status
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def read(self) -> bytes:
            return self._body

    class Client:
        def __init__(self) -> None:
            self.keys: list[str] = []

        def stream(self, method, url, headers, content):
            self.keys.append(headers.get("x-api-key"))
            return Stream(429, b'{"error":"rate_limit"}')

    client = Client()
    provider = AnthropicMessagesProvider(
        base_url="https://api.kimi.com/coding",
        api_key="key-one",
        standby_api_keys=("key-two", "key-three"),
        client=client,
    )
    for _ in range(3):
        try:
            provider._request_once({"model": "k3", "messages": [], "stream": False})
        except Exception as error:  # noqa: BLE001 - contract under test
            assert getattr(error, "retryable", False) is True
    assert client.keys == ["key-one", "key-two", "key-three"]
    rotations = provider.credential_rotations()
    assert [item["credential_index"] for item in rotations] == [1, 2, 3]
    assert all("api_key" not in json.dumps(item) for item in rotations)
    # The ring is cyclic: an exhausted key is one to come back to once the
    # provider's usage window resets, not one to discard.
    try:
        provider._request_once({"model": "k3", "messages": [], "stream": False})
    except Exception as error:  # noqa: BLE001 - contract under test
        assert getattr(error, "retryable", False) is True
        assert getattr(error, "retry_delay_s", 0) >= 60.0
        assert getattr(error, "retry_budget", 0) >= 8
    assert client.keys[-1] == "key-one"
    assert provider.credential_rotations()[-1]["completed_laps"] == 1


def test_credential_ring_spreads_a_batch_by_slot_ordinal() -> None:
    """Each concurrent slot starts on a different key; rotation order holds."""

    from ora.agent.providers.credentials import credential_ring as _credential_ring

    keys = ["key-one", "key-two", "key-three"]
    assert _credential_ring(keys[0], keys[1:], "gpu1") == (
        ["key-two", "key-three", "key-one"], 1,
    )
    assert _credential_ring(keys[0], keys[1:], "gpu2") == (
        ["key-three", "key-one", "key-two"], 2,
    )
    # gpu3 wraps back onto the first key; gpu4 lands on the second.
    assert _credential_ring(keys[0], keys[1:], "gpu3")[0][0] == "key-one"
    assert _credential_ring(keys[0], keys[1:], "gpu4")[0][0] == "key-two"
    # No slot identity and a single key both leave the ring untouched.
    assert _credential_ring(keys[0], keys[1:], None) == (keys, 0)
    assert _credential_ring("only", [], "gpu5") == (["only"], 0)


def test_provider_throttle_gets_patient_retry_budget(monkeypatch) -> None:
    """A 429 storm is retried with long backoff instead of a fast fatal fault."""

    from ora.agent.providers.anthropic_messages import AnthropicMessagesProvider

    class Stream:
        def __init__(self, status: int, body: bytes) -> None:
            self.status_code = status
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def read(self) -> bytes:
            return self._body

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def stream(self, method, url, headers, content):
            self.calls += 1
            return Stream(429, b'{"error":{"type":"rate_limit_error"}}')

    sleeps: list[float] = []
    monkeypatch.setattr(
        "ora.agent.providers.anthropic_messages.time.sleep", sleeps.append
    )
    client = Client()
    provider = AnthropicMessagesProvider(
        base_url="https://api.kimi.com/coding",
        api_key="key-one",
        client=client,
    )
    with pytest.raises(Exception) as excinfo:
        provider.complete(
            request={
                "model": "k3", "messages": [], "tools": [],
                "max_tokens": 512,
            },
            model_call_id="call-throttle",
        )
    assert "429" in str(excinfo.value)
    # max_retries stays 2 for other failures, but a throttle earns at least
    # seven total attempts with 20s-and-up waits between them.
    assert client.calls == 7
    assert len(sleeps) == 6
    assert sleeps[0] == pytest.approx(20.0)
    assert all(delay >= 20.0 for delay in sleeps)
    assert max(sleeps) <= 60.0
