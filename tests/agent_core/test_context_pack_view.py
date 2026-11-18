"""A context pack's delivery on the Pi agent core (``policy.context_pack``).

* The synthetic pack in ``tests/data/context_pack/`` and its replay fixture:
  four recorded requests of one short history and the views an independent
  reference implementation of ``ora.compact-view.v1`` produced for them. The
  Pi core's pack view (``ora-pi/src/context/context_pack.ts``, through
  ``tests/support/context_pack/view_driver.ts``) reproduces every message,
  images compared by sha256, and the overlaid tools.
* A Synthetic episode on the Pi core with a pack generated for its tools: the
  requests carry the bare task, the overlaid tools and the compact history
  with downscaled images, a success claim is admitted without check_task,
  and the run evaluates valid. A plan naming the retired Native core is refused.
* The pack's ``context_management`` switches: off, lossless retention's L4
  pressure externalization never advances and L5/L6 never call the compactor;
  a window the provider reports full ends the episode with context_exhausted.

The pack has no agent-core golden: those were recorded from the retired Native
core, which never delivered packs.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from ora.agent.composition.prompt import context_pack as cp
from tests.support.agent_core import episodes
from tests.support.agent_core.episodes import tool_reply
from tests.support.context_pack import replay

pytestmark = pytest.mark.usefixtures("pi_core")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/context_pack"
SYNTHETIC_MANIFEST_SHA256 = "03b07a0ddfa27c497b7cb0d4d15241dbedfbf7ec7a42eaeae391aa44eaddfb7c"


def test_the_pi_view_reproduces_the_reference_views_of_the_synthetic_fixture(tmp_path: Path) -> None:
    fixture = json.loads((DATA / "synthetic-replay.json").read_text(encoding="utf-8"))
    pack = cp.load_context_pack(DATA / fixture["pack"], SYNTHETIC_MANIFEST_SHA256)
    profile, source = cp.resolve_profile(pack, None)
    assert source == "manifest"
    compiled = fixture["compiled"]
    delivery = replay.static_delivery(
        pack, profile, harness_system=compiled["harness_system"],
        project_instructions=compiled["project_instructions"], episode_task=compiled["episode_task"],
        task_instruction=fixture["task_instruction"], task_template=fixture["episode_task_template"],
    )
    tools = cp.drop_tools(fixture["tools"], cp.dropped_tools(profile))
    delivery["tools"] = cp.overlay_tools(tools, cp.load_tool_overlay(pack, profile["tools"]["overlay_file"]))
    images = replay.ImageRefs(profile["compact_view"]["image_long_edge_px"])
    steps = [{"id": step["id"], "messages": images.swap(step["messages"]), "tools": tools}
             for step in fixture["steps"]]
    binding = {"schema": cp.BINDING_SCHEMA, "name": pack.name, "profile": profile}
    outputs = replay.run_view(binding, delivery, steps, images, scratch=tmp_path)
    assert [output["id"] for output in outputs] == [step["id"] for step in fixture["steps"]]
    for output in outputs:
        expected = fixture["expected"]["steps"][output["id"]]
        ours = replay.normalized_ours(output["messages"], images)
        assert replay.first_difference(ours, replay.normalized_expected(expected["messages"])) is None, output["id"]
        assert output["tools"] == fixture["expected"]["tools"]
        record = output["manifest"]["context_pack"]
        assert record["compact_view"]["folded_turns"] == expected["folded_turns"]
        assert record["compact_view"]["omitted_images"] == expected["omitted_images"]
        assert (record["replaced_messages"], record["dropped_messages"]) == (2, 1)


# ---- a Synthetic episode -----------------------------------------------------------------------------

VIEW = {
    "schema": "ora.compact-view.v1", "recent_turns": 1, "tool_text_cap_chars": 120, "image_window": 2,
    "image_long_edge_px": 3,
    "fold": {"line_format": "turn_summary_v1", "rationale_cap_chars": 80, "arguments_cap_chars": 40},
    "texts": {"earlier_steps_header": "EARLIER STEPS:", "image_placeholder": "[image omitted: {label}]",
              "tool_text_omission": "\n[{omitted} more characters omitted]"},
}


def _synthetic_pack(directory: Path) -> tuple[dict[str, Any], list[str]]:
    """A pack whose overlay covers the Synthetic surface's advertised tools; its RunSpec field."""

    from ora.orchestrator.spec.context_pack import wire_tool_definitions

    plan = episodes.synthetic_plan(actions=2)
    surface = plan["policy_composition"]["tool_surface"]
    wire = wire_tool_definitions(surface["definitions"], surface["capabilities"])
    overlay = []
    for tool in wire:
        compact = copy.deepcopy(tool["function"])
        compact["description"] = f"Overlay: {compact['name']}."
        overlay.append({"key": cp.overlay_key(tool["function"]), "spec": compact})
    directory.mkdir(parents=True)
    (directory / "CORE.md").write_text("# Unused core\n", encoding="utf-8")
    (directory / "overlay.json").write_text(json.dumps(overlay), encoding="utf-8")
    files = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ("CORE.md", "overlay.json")}
    manifest = {"name": "synthetic-episode-pack", "files": files, "ora_profile": {
        "schema": cp.PROFILE_SCHEMA, "episode_task": "task_and_contract_v1",
        "tools": {"overlay_file": "overlay.json"}, "finish": {"success": "official_verdict"},
        "compact_view": VIEW,
    }}
    data = json.dumps(manifest).encode()
    (directory / "manifest.json").write_bytes(data)
    names = [tool["function"]["name"] for tool in wire]
    return {"path": str(directory), "manifest_sha256": hashlib.sha256(data).hexdigest()}, names


def _image_sizes(message: dict[str, Any]) -> list[tuple[int, int]]:
    from PIL import Image

    sizes = []
    for part in message.get("content") if isinstance(message.get("content"), list) else []:
        if part.get("type") == "image_url":
            data = base64.b64decode(part["image_url"]["url"].partition(",")[2])
            with Image.open(io.BytesIO(data)) as image:
                sizes.append(image.size)
    return sizes


def test_a_pi_episode_delivers_the_pack_and_admits_success_without_check_task(tmp_path: Path) -> None:
    field, names = _synthetic_pack(tmp_path / "pack")
    script = episodes.action_script(3)
    script[-1] = tool_reply(("f0", "finish_episode", {
        "outcome": "success", "reason": "the camera image shows the scripted episode is complete",
        "attempts": ["scripted"]}))
    run = episodes.run_core("pi", tmp_path / "out", responses=script, actions=3,
                            policy_overrides={"context_pack": field})
    assert run["core"] == "pi"
    assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    binding = json.loads((run["root"] / "plan.json").read_text())["policy"]["context_pack"]
    assert binding["profile_source"] == "manifest" and binding["name"] == "synthetic-episode-pack"
    requests = [json.loads(data) for data in run["requests"]]
    task = json.loads((run["root"] / "plan.json").read_text())["task_instruction"]
    for request in requests:
        assert [tool["function"]["description"] for tool in request["tools"]] == [f"Overlay: {name}." for name in names]
        assert {"role": "user", "content": task} in request["messages"]  # the bare task, no template heading
        images = [size for message in request["messages"] for size in _image_sizes(message)]
        assert len(images) <= VIEW["image_window"] and all(max(size) <= 3 for size in images)
    assert any(message.get("content", "").startswith("EARLIER STEPS:\n1. ")
               for message in requests[-1]["messages"] if isinstance(message.get("content"), str))
    projections = [event["payload"] for event in run["events"] if event["type"] == "context_projected"]
    assert all(payload["context_pack"]["name"] == "synthetic-episode-pack" for payload in projections)
    assert "context_pack_view" in {payload.get("request_relation") for payload in projections}
    # The Synthetic cameras render 4x3 and its tool images 512x512: Python downscaled every kept image.
    assert all(payload["context_pack"]["compact_view"]["resized_images"]
               == payload["context_pack"]["compact_view"]["kept_images"] > 0 for payload in projections)
    finish = [event["payload"] for event in run["events"] if event["type"] == "tool_result"
              and event.get("payload", {}).get("name") == "finish_episode"]
    assert finish == [{"name": "finish_episode", "ok": True}]
    assert run["result"]["termination"] == "policy_finished_episode"


def test_the_native_core_refuses_a_plan_with_a_context_pack(tmp_path: Path) -> None:
    from ora.orchestrator.spec.run_spec import RunSpecError

    field, _names = _synthetic_pack(tmp_path / "pack")
    with episodes.environment(ORA_AGENT_CORE="native"), pytest.raises(RunSpecError, match="Pi agent core only"):
        episodes.synthetic_plan(actions=2, policy_overrides={"context_pack": field})


# ---- context_management: the layers that act on token usage ------------------------------------------


def _with_switches(field: dict[str, Any], switches: dict[str, bool] | None) -> dict[str, Any]:
    """The same pack with ``ora_profile.context_management`` set (``None``: absent)."""

    path = Path(field["path"]) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["ora_profile"].pop("context_management", None)
    if switches is not None:
        manifest["ora_profile"]["context_management"] = switches
    data = json.dumps(manifest).encode()
    path.write_bytes(data)
    return {"path": field["path"], "manifest_sha256": hashlib.sha256(data).hexdigest()}


@pytest.mark.parametrize("switched_off", [False, True])
def test_pressure_externalization_runs_only_while_the_pack_leaves_it_on(tmp_path: Path, switched_off: bool) -> None:
    field, _names = _synthetic_pack(tmp_path / "pack")
    field = _with_switches(field, {"pressure_externalization": False} if switched_off else None)
    management = episodes.management()
    management["trigger_inline_images"] = 1  # L4 fires as soon as a step has aged
    run = episodes.run_core("pi", tmp_path / "out", actions=8,
                            policy_overrides={"context_pack": field, "context_management": management})
    assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    projections = [event["payload"] for event in run["events"] if event["type"] == "context_projected"]
    watermarks = {payload.get("pressure_externalized_through_action_step") for payload in projections}
    if switched_off:
        assert watermarks == {None}
        assert projections[-1]["context_pack"]["context_management"] == {
            "pressure_externalization": False, "rollout_compaction": True}
    else:
        assert watermarks - {None}
        assert "context_management" not in projections[-1]["context_pack"]


@pytest.mark.parametrize("switched_off", [False, True])
def test_without_rollout_compaction_a_full_window_ends_in_context_exhausted(tmp_path: Path,
                                                                          switched_off: bool) -> None:
    """Provider usage crosses L6's watermark (0.8 of the hard limit), then the hard limit itself.

    On, L6 asks the compactor (none is scripted here, so each attempt fails and the
    projection is kept); off, nothing is summarized. Either way the turn after the
    provider reported more than the hard limit ends the episode with context_exhausted.
    """

    field, _names = _synthetic_pack(tmp_path / "pack")
    field = _with_switches(field, {"rollout_compaction": False} if switched_off else None)
    script = episodes.action_script(8)
    usage = [5000] * 10 + [13000] * 4 + [17000] * len(script)  # hard limit 20000 - 2048 - 2048 = 15904
    for response, tokens in zip(script, usage):
        response["usage"] = {"input_tokens": tokens, "output_tokens": 30}
    run = episodes.run_core("pi", tmp_path / "out", responses=script, actions=8, policy_overrides={
        "context_pack": field, "context_window_tokens": 20000, "max_output_tokens": 2048})
    assert run["result"]["termination"] == "context_exhausted"
    compactor = [event for event in run["events"] if event["type"].startswith("context_compactor")]
    reasons = {event["payload"]["emergency_compaction_admission"]["reason"]
               for event in run["events"] if event["type"] == "context_projected"}
    if switched_off:
        assert compactor == [] and reasons == {"rollout_compaction_disabled"}
    else:
        assert compactor and "summary_rejected_projection_retained" in reasons
