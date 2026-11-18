"""Dispatch intentions are explicit, reversible only with known zero spend."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ora.agent.toolkit.tools import _held
from ora.agent.toolkit.tools.grasp_execute import _stages
from ora.agent.toolkit.lib.values import JsonValue
from ora.agent.toolkit.interface.types import Response, canonical


def api_state():
    values = {"tool.held.left": _held.intent("prior", "old", [0, 0, 0, 0, 0, 0, 1]),
              "tool.held.right": None}
    return values, SimpleNamespace(state=SimpleNamespace(
        read=lambda key: deepcopy(values[key]), write=lambda key, value: values.__setitem__(key, deepcopy(value))))


@pytest.mark.parametrize("native", [0, 1, None])
def test_rollback_only_on_explicit_zero_spend(native):
    values, api = api_state()
    old = deepcopy(values)
    previous = _held.stage(api, {"left": None})
    _held.restore_if_unspent(api, previous, {"native_actions": native})
    assert values == (old if native == 0 else {**old, "tool.held.left": None})


def test_extents_use_state_and_current_artifact_without_cache():
    values, api = api_state()
    localization = {"aabb_min_xyz_m": [-.01, -.02, -.03], "aabb_max_xyz_m": [.01, .02, .03]}
    requests = []
    def artifact(**request):
        requests.append(request)
        return {"metadata": {"observation_id": "obs", "camera": "head"},
                "files": {"manifest.json": {"fixture": "manifest"}}}
    api.artifacts = SimpleNamespace(tool_artifact=artifact,
                                   read_blob=lambda ref: canonical({"localization": deepcopy(localization)}))
    jaw = {"left": {"contact_center_local": [0, 0, 0]}}
    first = _held.extents(api, jaw)
    assert first["left"]["target_ref"] == "old"
    assert len(first["left"]["local_corners"]) == 8
    localization["aabb_max_xyz_m"][2] = .06
    assert _held.extents(api, jaw) != first
    assert len(requests) == 2
    _held.stage(api, {"left": None})
    assert _held.extents(api, jaw) == {}


@pytest.mark.parametrize("zero_stage", [None, "open", "close"])
@pytest.mark.parametrize("verified", ["unknown", "unlikely"])
def test_grasp_intent_is_present_at_action_snapshot_and_verify_waits_for_next_boundary(monkeypatch, zero_stage, verified):
    values, api = api_state()
    before = deepcopy(values["tool.held.left"])
    snapshots = []
    pose = {"position_m": [0, 0, .3], "orientation_xyzw": [0, 0, 0, 1]}
    def observe(api):
        return SimpleNamespace(identity=JsonValue.of({"observation_id": "obs", "observation_sha256": "a"*64}),
                               manifest=JsonValue.of({}))
    def execute(action, budget):
        stage = action["arguments"]["targets"]["left"]
        snapshots.append((stage, deepcopy(values["tool.held.left"])))
        used = 0 if stage == zero_stage else 1
        return Response(canonical({"status": "ok", "data": {
            "remaining_budget": 20-used, "receipt": {"native_actions": used,
            "stage_evidence": {"consumed": bool(used), "command_invariant_verified": True,
                               "execution_status": "reached"}}}}))
    api.act = SimpleNamespace(execute_action=execute)
    monkeypatch.setattr(_stages, "observe", observe)
    monkeypatch.setattr(_stages._verify, "measure", lambda *args: {"verification": {"jaw_hold": {"held": verified}}})
    result = _stages.run(api, {"execution": {"generation": 0}},
        {"arm": "left", "candidate_id": "candidate", "source_ref": "new", "poses": {"approach": pose}},
        ["open", "close", "verify"])
    assert snapshots[0] == ("open", None)
    if zero_stage == "open":
        assert result["status"] == "stopped"
        assert values["tool.held.left"] == before
        assert len(snapshots) == 1
    else:
        intent = _held.intent("grasp_execute", "new", [0, 0, .3, 0, 0, 0, 1])
        assert snapshots[1] == ("close", intent)
        assert values["tool.held.left"] == (None if zero_stage is None and verified == "unlikely" else intent)
        assert len(snapshots) == 2  # verify never adds an action/publication


@pytest.mark.parametrize("explicit", [None, "before_intent", "begin"])
@pytest.mark.parametrize("wire", [False, True])
def test_execution_exception_rolls_back_only_with_owner_zero_spend_proof(explicit, wire):
    from ora.agent.toolkit.interface.runtime import public_failure
    from ora.agent.toolkit.interface.types import ActionNotStarted, InterfaceError
    values, api = api_state()
    before = deepcopy(values)
    previous = _held.stage(api, {"left": None})
    error = ActionNotStarted("invalid current binding", stage=explicit) if explicit else RuntimeError("connection lost")
    def execute(action, budget):
        if not wire:
            raise error
        status, failure = public_failure("execute_action", error)
        return Response(canonical({"method": "execute_action", "status": status, "data": None, "error": failure}))
    api.act = SimpleNamespace(execute_action=execute)
    with pytest.raises((RuntimeError, InterfaceError)):
        _held.execute(api, {}, previous)
    assert values == (before if explicit else {**before, "tool.held.left": None})


def test_grasp_stopped_after_open_retains_commitment_like_legacy_after_skill(monkeypatch):
    values, api = api_state()
    monkeypatch.setattr(_stages, "observe", lambda api: SimpleNamespace(
        identity=JsonValue.of({"observation_id": "obs", "observation_sha256": "a"*64}),
        manifest=JsonValue.of({})))
    api.act = SimpleNamespace(execute_action=lambda *args: Response(canonical({"status": "ok", "data": {
        "remaining_budget": 19, "receipt": {"native_actions": 1,
        "stage_evidence": {"consumed": True, "command_invariant_verified": True,
                           "execution_status": "not_reached"}}}})))
    pose = {"position_m": [0, 0, .3], "orientation_xyzw": [0, 0, 0, 1]}
    result = _stages.run(api, {"execution": {"generation": 0}},
                        {"arm": "left", "candidate_id": "candidate", "source_ref": "new",
                         "poses": {"approach": pose}}, ["open", "close", "verify"])
    assert result["status"] == "stopped"
    assert values["tool.held.left"]["target_ref"] == "new"
