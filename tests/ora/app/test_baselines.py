"""`ora baselines --explain` describes the default composition from code.

The explanation is for a reader who has not opened `orchestrator/spec/defaults.py`
or the tool runtime. Its value depends on every statement being derived from
the code it describes, so these tests hold it to the code: each tool of each
default surface appears in surface order, each handler is the tool's toolkit
registry entry at the stated line, and each state-changing tool shows the host
adapter that admits its proposal and how the proposal reaches the backend. The pre-existing `baselines` and
`baselines --json` outputs must not change, and the command must stay
read-only.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ora.agent.composition.baseline import (
    BASELINE_CATALOGS, BASELINE_ID, default_baseline_id, load_native_baseline,
)
from ora.agent.composition.surface.tool_surface import resolve_tool_surface as _resolve_surface
from ora.agent.composition.surface.toolkit_surface import declared_required_grants
from ora.orchestrator.spec.baseline_explain import (
    EXPLANATION_SCHEMA, explain_baselines, render_explanation,
)
from ora.app.main import main

ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENTS = tuple(load_native_baseline(ROOT)["instances"])


def resolve_tool_surface(profile: str):
    """The surface as a default RunSpec binds it, with the grants a file-backed one requires (toolkit v5)."""
    return _resolve_surface(profile, toolkit_grants=declared_required_grants(profile))


def _default_instance(environment: str) -> dict:
    """The instance a RunSpec naming no baseline gets (RoboDojo: ``robodojo_toolkit_v6``'s)."""
    return load_native_baseline(ROOT, default_baseline_id(environment))["instances"][environment]


@pytest.fixture(scope="module")
def explanation() -> dict:
    return explain_baselines(ROOT)


@pytest.fixture(scope="module")
def text(explanation) -> str:
    return render_explanation(explanation)


def _legacy_plain() -> str:
    """The pre-`--explain` plain format, kept here as the snapshot; an environment whose default is a
    baseline of its own (RoboDojo, 2026-10-30) shows that baseline's instance and names it."""
    baseline = load_native_baseline(ROOT)
    lines = [f"{baseline['id']}\tsha256={baseline['sha256']}"]
    for environment in baseline["instances"]:
        instance, selected = _default_instance(environment), default_baseline_id(environment)
        lines.append(f"{environment}\ttools={instance['tool_surface']}"
                     f"\tprompt={instance['prompt_bundle']}"
                     + (f"\tbaseline={selected}" if selected != baseline["id"] else ""))
    lines.append("Provider, task, seed, budgets and live qualifications remain explicit.")
    return "\n".join(lines) + "\n"


def test_plain_and_json_outputs_are_unchanged(capsys):
    assert main(["baselines"]) == 0
    assert capsys.readouterr().out == _legacy_plain()
    assert main(["baselines", "--json"]) == 0
    expected = json.dumps(
        load_native_baseline(ROOT), ensure_ascii=False, indent=2, sort_keys=True,
    ) + "\n"
    assert capsys.readouterr().out == expected


def test_explain_lists_every_tool_of_every_default_surface_in_order(explanation, text):
    assert explanation["schema"] == EXPLANATION_SCHEMA
    assert tuple(explanation["instances"]) == ENVIRONMENTS
    for environment, instance in explanation["instances"].items():
        profile = _default_instance(environment)["tool_surface"]
        assert instance["baseline"] == default_baseline_id(environment)
        surface = resolve_tool_surface(profile)
        expected = [item["name"] for item in surface.definitions]
        assert [tool["name"] for tool in instance["tools"]] == expected
        assert instance["tool_surface"]["profile"] == profile
        assert instance["tool_surface"]["manifest_sha256"] == surface.manifest_sha256
        assert instance["tool_surface"]["composition_sha256"] == surface.composition_sha256
        # The rendered section lists the same tools in the same order.
        section = text.split(f"== {environment} ", 1)[1].split("\n== ", 1)[0]
        tool_lines = [
            line.split()[0] for line in section.split("tools (", 1)[1].splitlines()[1:]
            if line.startswith("  ") and not line.startswith("   ")
        ]
        assert tool_lines == expected


def _resolve(dotted: str):
    module_name, qualname = dotted.split(":", 1)
    value = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


def test_every_handler_resolves_to_a_callable_at_the_stated_line(explanation):
    for instance in explanation["instances"].values():
        for tool in instance["tools"]:
            handler = tool["handler"]
            target = inspect.unwrap(_resolve(handler["callable"]))
            assert callable(target), tool["name"]
            path, line = handler["location"].rsplit(":", 1)
            assert (ROOT / path).resolve() == Path(inspect.getsourcefile(target)).resolve()
            lines, start = inspect.getsourcelines(target)
            assert start <= int(line) < start + len(lines), tool["name"]
            source_line = (ROOT / path).read_text(encoding="utf-8").splitlines()[int(line) - 1]
            assert f"def {target.__name__}(" in source_line, tool["name"]


def test_effect_classes_come_from_the_surface(explanation):
    for environment, instance in explanation["instances"].items():
        surface = resolve_tool_surface(_default_instance(environment)["tool_surface"])
        for tool in instance["tools"]:
            name = tool["name"]
            expected = (
                "lifecycle" if name in surface.lifecycle_tools
                else "state_changing_proposal" if name in surface.mutation_tools
                else "read_only_observation" if name in surface.observation_tools
                else "read_only"
            )
            assert tool["effect"] == expected, name


def test_every_state_changing_tool_shows_a_proposal_route(explanation, text):
    for instance in explanation["instances"].values():
        backend = instance["proposal_route"]["backend_session"]["class"]
        assert backend.split(":")[1] in text
        for tool in instance["tools"]:
            if tool["effect"] != "state_changing_proposal":
                assert tool["proposal"] is None, tool["name"]
                continue
            proposal = tool["proposal"]
            assert proposal["route"], tool["name"]
            assert "EnvironmentSession.execute" in proposal["route"], tool["name"]
            assert backend.split(":")[1] in proposal["route"], tool["name"]
            assert proposal["resolution"] in {"static", "partial", "generic"}, tool["name"]
            if proposal["resolution"] != "generic":
                assert proposal["actions"], tool["name"]
                for action in proposal["actions"]:
                    assert action["action"] and action["schema"], tool["name"]
                    _resolve(action["executor"])


#: State-changing default tools the static walk cannot resolve, each with its reason.
GENERIC_PROPOSALS = {
    "return_home": "RoboDojo only: ora/agent/toolkit/tools/return_home/ plans its recovery trajectory with "
                   "lib.motion_plan at run time and runs it inside the call; it submits no proposal envelope",
}


def test_default_proposals_resolve_to_one_schema_per_action(explanation):
    """The walk follows only the branches the default surfaces select.

    Every state-changing default tool resolves statically today, except those in
    ``GENERIC_PROPOSALS``. A tool that stops resolving falls back to the generic route
    rather than to a guess; accept that here only with a reason, since it makes the
    explanation vaguer.
    """
    for instance in explanation["instances"].values():
        for tool in instance["tools"]:
            proposal = tool["proposal"]
            if proposal is None:
                continue
            if tool["name"] in GENERIC_PROPOSALS:
                assert proposal["resolution"] == "generic" and GENERIC_PROPOSALS[tool["name"]].strip()
                continue
            assert proposal["resolution"] == "static", tool["name"]
            schemas: dict[str, set[str]] = {}
            for action in proposal["actions"]:
                schemas.setdefault(action["action"], set()).add(action["schema"])
            assert all(len(found) == 1 for found in schemas.values()), (tool["name"], schemas)


def test_every_handler_is_the_toolkit_registry_entry_and_each_proposal_is_what_its_adapter_admits(explanation):
    from ora.agent.toolkit.host.motion import PROPOSED_ACTIONS
    from ora.agent.toolkit.registry import load_tools

    entries = load_tools()
    for instance in explanation["instances"].values():
        assert instance["tool_runtime"]["dispatch"].startswith("ora/agent/toolkit/host/runtime/dispatch.py:")
        for tool in instance["tools"]:
            entry = entries[tool["name"]]
            assert tool["handler"]["callable"] == f"{entry.__module__}:{entry.__qualname__}", tool["name"]
            proposal = tool["proposal"]
            if proposal is None or proposal["resolution"] == "generic":
                continue
            admitted = proposal["admitted_by"]["callable"]
            actions = {item["action"] for item in proposal["actions"]}
            if admitted == "ora.agent.toolkit.host.motion:submit_adapter":
                assert actions == {PROPOSED_ACTIONS[tool["name"]]}, tool["name"]
            else:
                assert (admitted, actions) == (
                    "ora.agent.toolkit.host.execution:submit_adapter", {"execute_skill_sequence"}), tool["name"]


def _known_entry(api, args):
    """A toolkit entry shape the proposal walk reads; never called."""
    return api.submit_action(action={"schema": "ora-motion-action.v3", "action": "move_pose", "arguments": {}})


def _unresolved_entry(api, args):
    return api.submit_action(action={"schema": args["schema"], "action": "move_pose", "arguments": {}})


def _dynamic_entry(api, args):
    return api.submit_action(action=dict(schema="ora-motion-action.v3", action=args["name"], arguments={}))


def _scanned_surface():
    from types import SimpleNamespace

    return SimpleNamespace(
        capabilities=(), profile_id="scanned", runtime_id="scanned",
        mutation_tools=("dynamic",), lifecycle_tools=(), observation_tools=(),
    )


def test_the_walk_records_the_envelopes_an_entry_and_the_toolkit_code_it_calls_build():
    from ora.agent.toolkit.tools.set_gripper.tool import run as set_gripper
    from ora.orchestrator.spec import baseline_tool_routes as routes

    scan = routes._ProposalScan()
    scan.function(_known_entry)
    assert scan.found == {("move_pose", "ora-motion-action.v3")} and not scan.unresolved
    scan = routes._ProposalScan()
    scan.function(_unresolved_entry)
    assert scan.found == set() and scan.unresolved
    # set_gripper builds its envelope in lib.motion_plan.execute.gripper_action, which the walk follows.
    scan = routes._ProposalScan()
    scan.function(set_gripper)
    assert scan.found == {("set_gripper", "ora-motion-action.v3")} and not scan.unresolved


def test_the_adapter_is_the_last_submit_action_binding_dispatch_makes_for_the_tool():
    from ora.orchestrator.spec import baseline_tool_routes as routes

    assert routes._submit_adapter("move_pose", ROOT)[0]["callable"] == "ora.agent.toolkit.host.motion:submit_adapter"
    assert routes._submit_adapter("grasp_execute", ROOT)[0]["callable"] == (
        "ora.agent.toolkit.host.execution:submit_adapter")
    assert routes._submit_adapter("move_pose", ROOT)[1] is True


def test_an_unresolvable_proposal_takes_the_generic_route():
    from ora.orchestrator.spec import baseline_tool_routes as routes

    route = {"backend_session": {"class": "backend.module:BackendSession"},
             "dispatch_branches": [{"schemas": None, "executor": "m:execute_runtime_action"}]}
    record = routes._tool(
        {"name": "dynamic", "description": "Submit something. Use when: testing."}, {"dynamic": _dynamic_entry},
        _scanned_surface(), route, "motion_action_v3", ROOT,
    )
    assert record["summary"] == "Submit something."
    assert record["handler"]["callable"].endswith("_dynamic_entry")
    proposal = record["proposal"]
    assert proposal["resolution"] == "generic" and proposal["actions"] == []
    assert proposal["route"] == (
        "submitted as a motion_action_v3 proposal (action type not resolved statically) -> "
        "Supervisor dispatch_action -> EnvironmentSession.execute on BackendSession"
    )


def test_proposal_executors_are_the_supervisor_dispatch_branches(explanation):
    from ora.orchestrator.episode import actions as episode_actions

    source = inspect.getsource(episode_actions.dispatch_action)
    for instance in explanation["instances"].values():
        branches = instance["proposal_route"]["dispatch_branches"]
        assert branches[-1]["schemas"] is None  # the default branch
        for branch in branches:
            assert branch["executor"].split(":")[1].split(".")[-1] in source


def test_selection_reasons_name_the_default_and_maintained_baselines(explanation, text):
    selection = explanation["selection"]
    assert selection["default"]["id"] == BASELINE_ID
    assert "policy.baseline" in text and "policy.tool_surface" in text
    maintained = {item["id"]: item for item in explanation["maintained_baselines"]}
    assert set(maintained) == set(BASELINE_CATALOGS) - {BASELINE_ID}
    for baseline_id, item in maintained.items():
        assert item["path"] == BASELINE_CATALOGS[baseline_id]
        assert baseline_id in text
    assert "frozen_baselines" not in explanation


def test_runtime_choices_and_policy_defaults_follow_the_expansion(explanation):
    robotwin = {item["key"]: item for item in explanation["instances"]["robotwin"]["runtime_template"]}
    assert robotwin["task_config"]["runspec"] == "choice"
    assert "demo_randomized" in robotwin["task_config"]["choices"]
    assert robotwin["required_revision"]["runspec"] == "fixed"
    assert robotwin["request_timeout_s"]["runspec"] == "may_override"
    for instance in explanation["instances"].values():
        policy = {item["field"]: item for item in instance["policy_defaults"]}
        assert policy["action_codec"]["runspec"] == "fixed"
        assert policy["timeout_s"]["runspec"] == "may_override"
        slots = instance["prompt_bundle"]["policy_context"]
        assert list(slots) == ["harness_system", "project_instructions", "episode_task", "tool_manual"]


def test_environment_filter(capsys):
    assert main(["baselines", "--explain", "--environment", "robotwin"]) == 0
    out = capsys.readouterr().out
    assert "== robotwin " in out and "== libero_pro " not in out
    assert main(["baselines", "--explain", "--environment", "robotwin", "--json"]) == 0
    assert list(json.loads(capsys.readouterr().out)["instances"]) == ["robotwin"]
    assert main(["baselines", "--explain", "--environment", "synthetic"]) == 2
    assert "synthetic" in capsys.readouterr().err
    assert main(["baselines", "--environment", "robotwin"]) == 2
    assert "--explain" in capsys.readouterr().err


def test_explain_json_round_trips(capsys, explanation):
    assert main(["baselines", "--explain", "--json"]) == 0
    out = capsys.readouterr().out
    loaded = json.loads(out)
    assert loaded == json.loads(json.dumps(explanation))
    assert out == json.dumps(loaded, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def test_text_fits_a_hundred_columns_and_ends_with_the_caveat(text):
    assert max(len(line) for line in text.splitlines()) <= 100
    assert text.rstrip("\n").splitlines()[-1] == (
        "Provider, task, seed, budgets and live qualifications remain explicit."
    )


#: The checkout's ignored directories the Runtime writes into when no workspace is set.
RUNTIME_STATE = ("outputs", ".harness-runtime")


def _tree_state() -> list[str]:
    """Tracked and untracked files that differ from HEAD; ignored ones (caches, builds, node_modules) left out.

    Other tools write ignored files concurrently (vitest, npm, docs builds), and
    listing every ignored file of a checkout takes seconds, so the Runtime's
    own state directories are checked separately (``_runtime_state``).
    """
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return sorted(listed)


def _runtime_state() -> dict[str, list[tuple[str, int, int]] | None]:
    """Every entry under the checkout's ``outputs/`` and ``.harness-runtime/``: path, size, mtime."""
    state: dict[str, list[tuple[str, int, int]] | None] = {}
    for name in RUNTIME_STATE:
        root = ROOT / name
        if not root.exists():
            state[name] = None
            continue
        entries = []
        for directory, directories, files in os.walk(root):
            for entry in (*directories, *files):
                path = Path(directory, entry)
                status = path.lstat()
                entries.append((path.relative_to(root).as_posix(), status.st_size, status.st_mtime_ns))
        state[name] = sorted(entries)
    return state


def test_the_command_writes_no_files(tmp_path):
    workspace, scratch = tmp_path / "workspace", tmp_path / "tmp"
    workspace.mkdir()
    scratch.mkdir()
    environment = {
        **os.environ, "ORA_WORKSPACE_ROOT": str(workspace),
        "TMPDIR": str(scratch), "PYTHONDONTWRITEBYTECODE": "1",
    }
    before, runtime_before = _tree_state(), _runtime_state()
    for arguments in (["--explain"], ["--explain", "--json"]):
        completed = subprocess.run(
            [sys.executable, "-m", "ora.app", "baselines", *arguments],
            cwd=ROOT, env=environment, check=True, capture_output=True, text=True,
        )
        assert completed.stdout
    assert _tree_state() == before
    assert _runtime_state() == runtime_before
    assert list(workspace.iterdir()) == []
    assert list(scratch.iterdir()) == []
