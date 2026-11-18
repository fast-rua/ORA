"""``ora plan``: one RunSpec bound, shown, previewed or preflighted, never launched.

    ora plan validate SPEC     bind the RunSpec; print its plan id and sha256
    ora plan show SPEC         the resolved, content-addressed RunPlan
    ora plan context SPEC      the exact project-owned model context and tools
    ora plan preflight SPEC    the launch gates, without starting an episode

``display_plan`` is the summary ``ora plan show``, ``ora launch`` and
``ora retry`` print before the plan itself or the confirmation prompt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

ACTIONS = {
    "validate": "validate and bind one RunSpec without writing a run",
    "show": "print the resolved, content-addressed RunPlan",
    "context": "preview exact project-owned model context and tools",
    "preflight": "run launch gates without starting an episode",
}


def register(sub: Any) -> None:
    from ora.orchestrator.episode import agent_core

    plan = sub.add_parser("plan", help="validate, show, preview or preflight one RunSpec",
                          description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = plan.add_subparsers(dest="plan_command", required=True, metavar="<action>")
    for name, text in ACTIONS.items():
        command = actions.add_parser(name, help=text, description=text)
        command.add_argument("run_spec", type=Path)
        agent_core.add_cli_option(command)


def _runtime_revision(runtime: Mapping[str, Any]) -> str | None:
    """The backend's revision label, else ``runtime.required_revision``."""

    from ora.orchestrator.backends.registry import find

    selected = find(runtime["kind"])
    label = selected.revision_label if selected is not None else None
    return label(runtime) if label is not None else runtime.get("required_revision")


def display_plan(plan: Mapping[str, Any]) -> None:
    import json

    from ora.orchestrator import paths

    policy = plan["policy"]
    runtime = plan["runtime"]
    runtime_revision = _runtime_revision(runtime)
    rows = [
        ("plan", plan["plan_id"]), ("experiment/config", f"{plan['experiment']} / {plan['config']}"),
        ("milestone/task/seed", f"{plan['milestone']} / {plan['task']} / {plan['seed']}"),
        ("experiment identity", str(plan.get("experiment_identity") or "legacy/unmapped")),
        ("evidence", plan["evidence_tier"]),
        ("result tier", str(plan.get("result_tier", "unlabelled"))),
        ("partition", str((plan.get("partition") or {}).get("name", "unlabelled"))),
        ("model", f"{policy['model']} / {policy['reasoning_effort']} / {policy['provider']}"),
        ("policy/runtime", f"{policy['kind']} / {runtime['kind']}"),
        ("agent core", policy.get("agent_core", "native (not recorded; retired)") if policy["kind"] == "native_api" else "n/a"),
        ("baseline", str(policy.get("baseline", {}).get("id", "custom/legacy"))),
        ("tool surface", str(policy.get("tool_surface", {}).get("profile", "n/a"))),
        ("context retention", str(policy.get("context_retention", "n/a"))),
        ("tools", ", ".join(policy.get("tool_allowlist", ["robot.submit_action"]))),
        (
            "auto image",
            str(
                policy.get("supports_images", False)
                and policy.get("initial_observation") == "automatic"
            ),
        ),
        ("context projection", str(policy.get("context_projection", policy.get("context_profile", "n/a")))),
        ("agent session", str(policy.get("agent_session_backend", "n/a"))),
        ("rollover policy", str(policy.get("context_rollover_policy", "n/a"))),
        ("source", f"{plan['source']['commit']} dirty={plan['source']['dirty']}"),
        ("model visible", str(plan.get("policy_identity", {}).get("model_visible_sha256", "not recorded"))),
        ("evidence workspace", str(paths.workspace_from_plan(plan))),
        (
            "policy composition",
            plan.get("policy_composition", {}).get(
                "composition_sha256", plan["policy_context"]["composition_sha256"]
            ),
        ),
        ("runtime revision", runtime_revision or "synthetic"),
        (
            "execution resource",
            str(
                runtime.get("resource_binding", {}).get(
                    "profile_id", "implicit-exclusive-host"
                )
            ),
        ),
        ("budgets", json.dumps(plan["budgets"], sort_keys=True)),
        ("plan sha256", plan["plan_sha256"]),
        ("confirmation code", plan["confirmation_code"]),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")


def command_context(plan: Mapping[str, Any]) -> int:
    import json
    import tempfile

    from ora.agent.composition import materialize_policy
    from ora.orchestrator.evaluation.context_manifest import build_context_manifest, context_placeholder
    from ora.orchestrator.spec.run_spec import PROJECT_ROOT

    policy = plan["policy"]
    if policy["kind"] != "native_api":
        raise RuntimeError("context preview is meaningful only for a model policy")
    placeholder = context_placeholder(plan)
    with tempfile.TemporaryDirectory(prefix="ora-context-") as temporary:
        compiled = materialize_policy(
            plan["policy_context"], project_root=PROJECT_ROOT,
            destination=Path(temporary),
        )
        manifest = build_context_manifest(
            plan, compiled_policy=compiled, context=placeholder
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> int:
    import json

    from ora.orchestrator.spec.run_spec import load_run_spec, resolve_run_spec

    plan = resolve_run_spec(load_run_spec(args.run_spec))
    if args.plan_command == "validate":
        print(json.dumps({"valid": True, "plan_id": plan["plan_id"], "plan_sha256": plan["plan_sha256"]}, indent=2))
        return 0
    if args.plan_command == "show":
        display_plan(plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.plan_command == "context":
        return command_context(plan)
    from ora.orchestrator.launch import _preflight

    return 0 if _preflight(plan)["passed"] else 2


__all__ = ["ACTIONS", "command_context", "display_plan", "register", "run"]
