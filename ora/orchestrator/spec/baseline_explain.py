"""Explain how the Native robot default composition resolves, derived from code.

`ora baselines --explain` is for a reader who has not yet opened
`orchestrator/spec/defaults.py` or the tool runtime. Every statement it prints is
read from the code it describes, never restated by hand. The resolution chain
comes from running `expand_native_defaults` on a task-free probe RunSpec;
which fields a RunSpec may still choose comes from probing that function (and
its runtime contract check) with a value nothing can equal. The tool half --
toolkit registry entries, proposals, the host adapters that admit them and their
route to the backend -- is derived in `baseline_tool_routes`.

The command is read-only: it reads tracked sources and imports host-safe
modules; it writes no file, starts no service or simulator and calls no
provider. `baselines` and `baselines --json` keep their earlier output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from ora.agent.composition.baseline import (
    BASELINE_CATALOGS, BASELINE_ID, ENVIRONMENT_BASELINES, default_baseline_id, load_native_baseline,
)
from ora.agent.composition.context_scheme import CONTEXT_FIELDS, context_schemes, load_context_scheme
from ora.agent.composition.surface.tool_surface import ResolvedToolSurface, resolve_tool_surface
from ora.environment.benchmarks.robotwin import TASK_CONFIGS
from ora.environment.definitions import resolve_runtime_descriptor

from . import defaults
from .baseline_tool_routes import BaselineExplainError, explain_tools, location

__all__ = ["BaselineExplainError", "EXPLANATION_SCHEMA", "command_baselines",
           "explain_baselines", "render_explanation"]

EXPLANATION_SCHEMA = "ora.baseline-explanation.v1"
CAVEAT = "Provider, task, seed, budgets and live qualifications remain explicit."
WIDTH = 100
# A probe value no catalog default can equal: a field that refuses it is owned
# by the baseline, a field that keeps it is the RunSpec's to choose.
_PROBE = object()


# -- selection and the per-instance resolution chain ----------------------------

def _catalog_identity(root: Path, baseline_id: str) -> dict[str, str]:
    catalog = load_native_baseline(root, baseline_id)
    return {"id": baseline_id, "path": catalog["path"], "sha256": catalog["sha256"]}


def _selection(root: Path, catalog: Mapping[str, Any]) -> dict[str, Any]:
    kinds = " or ".join(catalog["instances"])
    native = location(defaults.expand_native_defaults, root)
    rules = [
        f"policy.kind native_api with runtime.kind {kinds} is eligible; any other RunSpec "
        "passes through unchanged.",
        "An explicit policy.baseline naming a registered id selects that catalog "
        "(selection=explicit); every field the catalog owns must then agree with the RunSpec.",
        "policy.tool_surface without policy.baseline bypasses the defaults: the explicit "
        "surface and every field it implies are kept as written.",
        f"Otherwise the environment's default applies (selection=omitted_tool_surface): {BASELINE_ID}; "
        + "; ".join(f"on {kind}, {baseline_id}" for kind, baseline_id in sorted(ENVIRONMENT_BASELINES.items()))
        + ".",
    ]
    return {
        "default": _catalog_identity(root, BASELINE_ID),
        "native_expansion": native,
        "rules": rules,
    }


def _maintained(root: Path) -> list[dict[str, Any]]:
    return [
        {"id": baseline_id, "path": path, "selected_by": "policy.baseline"}
        for baseline_id, path in BASELINE_CATALOGS.items() if baseline_id != BASELINE_ID
    ]


def _probe_spec(environment: str) -> dict[str, Any]:
    return {"policy": {"kind": "native_api", "supports_images": True},
            "runtime": {"kind": environment}}


def _policy_ownership(root: Path, environment: str, field: str) -> str:
    spec = _probe_spec(environment)
    spec["policy"][field] = _PROBE
    try:
        defaults.expand_native_defaults(spec, root=root)
    except ValueError as error:
        return "fixed" if f"policy.{field}" in str(error) else "undetermined"
    except Exception:  # noqa: BLE001 - an unrelated refusal decides nothing here
        return "undetermined"
    return "may_override"


def _scheme_ownership(root: Path, environment: str, own: str) -> str:
    """Whether a RunSpec may name another registered context scheme than the catalog's ``own``."""
    others = sorted(set(context_schemes(root)) - {own})
    if not others:
        return "undetermined"
    spec = _probe_spec(environment)
    spec["policy"]["context_scheme"] = others[0]
    try:
        defaults.expand_native_defaults(spec, root=root)
    except ValueError as error:
        return "fixed" if "policy.context_scheme" in str(error) else "undetermined"
    except Exception:  # noqa: BLE001 - an unrelated refusal decides nothing here
        return "undetermined"
    return "may_override"


def _policy_defaults(root: Path, environment: str, catalog: Mapping[str, Any]) -> list[dict]:
    """The catalog's policy fields; a catalog that names a context scheme lists the scheme's record and fields
    first, then its own fields."""
    rows, fields = [], list(catalog["policy"].items())
    named = catalog.get("context_scheme")
    if named is not None:
        scheme = load_context_scheme(root, named)
        rows.append({"field": "context_scheme", "value": scheme["record"],
                     "runspec": _scheme_ownership(root, environment, named)})
        fields = [*scheme["policy"].items(), *((field, value) for field, value in fields
                                               if field not in CONTEXT_FIELDS)]
    if catalog.get("code_mode") is not None:
        # The code mode a RunSpec gets when it names no policy.code_mode; null (tool calls) or another overrides it.
        rows.append({"field": "code_mode", "value": catalog["code_mode"], "runspec": "may_override"})
    code_mode = catalog.get("code_mode_context_scheme")
    if code_mode is not None:
        # Code mode's scheme replaces the context fields above when the RunSpec sets policy.code_mode.
        rows.append({"field": "code_mode_context_scheme", "value": load_context_scheme(root, code_mode)["record"],
                     "runspec": _scheme_ownership(root, environment, code_mode)})
    return rows + [
        {"field": field, "value": value,
         "runspec": _policy_ownership(root, environment, field)}
        for field, value in fields
    ]


def _runtime_template(catalog: Mapping[str, Any], environment: str) -> list[dict]:
    template = catalog["instances"][environment]["runtime"]
    rows = []
    for key, value in template.items():
        row: dict[str, Any] = {"key": key, "value": value}
        if (environment == "robotwin" and key == "task_config"
                and catalog["id"] in defaults.SCENE_CONFIGURATION_CHOICE_BASELINES):
            row.update(runspec="choice", choices=sorted(TASK_CONFIGS))
        else:
            # The contract check expand_native_defaults applies to a RunSpec's
            # runtime: a key it refuses to see changed is owned by the baseline.
            try:
                defaults._check_runtime_contract({key: _PROBE}, template)
            except ValueError:
                row["runspec"] = "fixed"
            else:
                row["runspec"] = "may_override"
        rows.append(row)
    return rows


def _surface_record(surface: ResolvedToolSurface) -> dict[str, Any]:
    return {
        "profile": surface.profile_id, "revision": surface.revision,
        "runtime_id": surface.runtime_id,
        "manifest_sha256": surface.manifest_sha256,
        "composition_sha256": surface.composition_sha256,
        "components": len(surface.components), "capabilities": len(surface.capabilities),
        "call_effect_profile": surface.call_effect_profile,
    }


def _instance(root: Path, catalog: Mapping[str, Any], environment: str) -> dict[str, Any]:
    instance = catalog["instances"][environment]
    expanded = defaults.expand_native_defaults(_probe_spec(environment), root=root)
    policy, runtime = expanded["policy"], expanded["runtime"]
    surface = resolve_tool_surface(instance["tool_surface"], toolkit_grants=expanded.get("toolkit_grants"))
    descriptor = resolve_runtime_descriptor(
        {key: value for key, value in runtime.items() if key != "public_environment_context"},
        action_codec=policy["action_codec"],
    )
    context = {key: value for key, value in expanded["policy_context"].items() if key != "schema"}
    return {
        "baseline": catalog["id"],
        "instance_id": instance["id"],
        "tool_surface": {**_surface_record(surface),
                         "component_overrides": policy["tool_surface"]["component_overrides"]},
        "policy_defaults": _policy_defaults(root, environment, catalog),
        "runtime_template": _runtime_template(catalog, environment),
        "public_environment_context": runtime["public_environment_context"],
        "environment_prompt_profile": runtime["environment_prompt_profile"],
        "prompt_bundle": {**policy["baseline"]["prompt_bundle"], "policy_context": context,
                          "task_contract": policy["task_contract"]},
        "required_service_capabilities": list(expanded["prerequisite"]["capability_scope"]),
        "action": {"codec": policy["action_codec"], "transport": descriptor.transport,
                   "backend_action_modes": list(descriptor.action_modes)},
        **explain_tools(surface, runtime, policy["action_codec"], root),
    }


def explain_baselines(root: Path, *, environment: str | None = None) -> dict[str, Any]:
    """Structured explanation of the default composition; reads, never writes."""
    catalog = load_native_baseline(root)
    environments = list(catalog["instances"])
    if environment is not None and environment not in environments:
        raise BaselineExplainError(
            f"unknown environment {environment!r}; choose one of {', '.join(environments)}"
        )
    return {
        "schema": EXPLANATION_SCHEMA,
        "selection": _selection(root, catalog),
        "maintained_baselines": _maintained(root),
        "instances": {
            name: _instance(root, load_native_baseline(root, default_baseline_id(name)), name)
            for name in environments
            if environment in {None, name}
        },
        "caveat": CAVEAT,
    }


# -- plain-text rendering -------------------------------------------------------

def _fill(prefix: str, text: str, indent: int | None = None) -> list[str]:
    """Wrap at spaces; break a path or dotted name that cannot fit after / . or :."""
    pad = " " * (len(prefix) if indent is None else indent)
    lines, current, fresh = [], prefix, True
    for word in text.split():
        pieces = re.split(r"(?<=[/.:])", word) if len(pad) + len(word) > WIDTH else [word]
        for index, piece in enumerate(pieces):
            joined = piece if fresh or index else " " + piece
            if current.strip() and len(current) + len(joined) > WIDTH:
                lines.append(current.rstrip())
                current, joined = pad, piece
            current, fresh = current + joined, False
    return [*lines, current]


def _compact(value: Any, limit: int = 46) -> str:
    if isinstance(value, Mapping) and "path" in value:
        text = str(value["path"])
    elif isinstance(value, Mapping) and "schema" in value:
        text = f"{value['schema']} (+{len(value) - 1} settings)"
        text = text if len(text) <= limit else str(value["schema"])
    elif isinstance(value, (Mapping, list)):
        text = json.dumps(value, separators=(",", ":"), sort_keys=True)
    else:
        text = str(value).lower() if isinstance(value, bool) else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


_RUNSPEC = {"fixed": "fixed", "may_override": "may override", "choice": "choice",
            "undetermined": "undetermined"}
_EFFECT = {"state_changing_proposal": "state-changing proposal", "lifecycle": "lifecycle",
           "read_only_observation": "read-only observation", "read_only": "read-only"}


def _row(name: str, value: Any, runspec: str) -> str:
    return f"      {name:<31} {_compact(value):<46} {_RUNSPEC[runspec]}"


def _hop(record: Mapping[str, str]) -> list[str]:
    name = record["callable"].split(":")[1]
    line = f"  -> {name:<42} {record['location']}"
    return [line] if len(line) <= WIDTH else [f"  -> {name}", f"       {record['location']}"]


def _render_selection(explanation: Mapping[str, Any]) -> list[str]:
    selection = explanation["selection"]
    default = selection["default"]
    lines = [f"Native robot default composition: {default['id']}",
             *_fill("catalog  ", default["path"]), f"sha256   {default['sha256']}", "",
             f"Selection ({selection['native_expansion']} expand_native_defaults)"]
    for index, rule in enumerate(selection["rules"], 1):
        lines += _fill(f"  {index}. ", rule)
    lines += ["", "Maintained baselines, selectable from this tree"]
    for item in explanation["maintained_baselines"]:
        lines += [f"  {item['id']}  [{item['selected_by']}]", *_fill("    ", item["path"])]
    return lines


def _render_chain(instance: Mapping[str, Any]) -> list[str]:
    surface, bundle, action = (instance["tool_surface"], instance["prompt_bundle"],
                               instance["action"])
    public, profile = instance["public_environment_context"], instance["environment_prompt_profile"]
    lines = ["  1 policy defaults  fixed: must equal the catalog; may override: a RunSpec "
             "value is kept"]
    lines += [_row(row["field"], row["value"], row["runspec"])
              for row in instance["policy_defaults"]]
    overrides = ", ".join(surface["component_overrides"]) or "none"
    lines += [f"  2 tool surface     {surface['profile']} (revision {surface['revision']})",
              f"       component overrides: {overrides}",
              *_fill("       ", f"runtime {surface['runtime_id']}; {surface['components']} "
                     f"components, {surface['capabilities']} capabilities; effects "
                     f"{surface['call_effect_profile']}"),
              f"       manifest sha256     {surface['manifest_sha256']}",
              f"       composition sha256  {surface['composition_sha256']}",
              "  3 runtime template  fixed | may override | choice: the RunSpec picks a choice"]
    for row in instance["runtime_template"]:
        lines.append(_row(row["key"], row["value"], row["runspec"]))
        lines += _fill("         choices: ", ", ".join(row["choices"])) if "choices" in row else []
    lines += _fill("  4 descriptor       ", (
        f"environment {public['environment_id']}, embodiment {public['embodiment_id']}, "
        f"controller {public['controller']}, cameras {', '.join(public['public_cameras'])}; "
        f"{action['codec']} over {action['transport']} accepts "
        f"{', '.join(action['backend_action_modes'])}"), 7)
    lines += [f"  5 env prompt       {profile['profile_id']}",
              *_fill("       ", profile["profile_path"]), f"       sha256 {profile['profile_sha256']}",
              *_fill("  6 prompt bundle    ", bundle["path"], 7), f"       sha256 {bundle['sha256']}",
              "       becomes policy_context, in delivery order"]
    for slot, sources in bundle["policy_context"].items():
        lines.append(f"         {slot}")
        for source in sources:
            lines += _fill("           ", source)
    contract = bundle["task_contract"]
    lines += _fill("         ", (
        "episode_task gains the task's reviewed completion contract once a RunSpec binds a "
        f"task (registry {contract['registry']})" if contract.get("registry") else
        f"no reviewed task-contract layer for this backend ({contract.get('reason')})"))
    return lines + _fill("  7 resident services ",
                         ", ".join(instance["required_service_capabilities"]), 7)


def _render_route(instance: Mapping[str, Any]) -> list[str]:
    route, runtime = instance["proposal_route"], instance["tool_runtime"]
    backend = route["backend_session"]
    lines = ["Tool runtime", *_fill("  class     ", runtime["class"]),
             f"            {runtime['location']}",
             f"  dispatch  {runtime['dispatch']} (Dispatch.run, toolkit registry entries)",
             "Proposal route of a state-changing tool",
             f"  its host adapter persists {route['persisted_to']}: one proposal per observation"]
    for hop in route["supervisor"]:
        lines += _hop(hop)
    for branch in route["dispatch_branches"]:
        schemas = ", ".join(branch["schemas"]) if branch["schemas"] else "any other schema"
        lines += _fill("       ", f"{schemas} -> {branch['executor'].split(':')[1]} "
                       f"{branch['location']}", 9)
    for hop in route["environment"]:
        lines += _hop(hop)
    lines += _fill("  -> ", f"isolated worker, backend {route['worker_backend']!r} over "
                   f"{route['transport']}: {backend['class']}", 7)
    return lines + _fill("       ", f"{backend['location']}; execute {backend['execute']}")


def _render_tools(instance: Mapping[str, Any]) -> list[str]:
    lines = [f"tools ({len(instance['tools'])}, surface order)"]
    for tool in instance["tools"]:
        handler, proposal = tool["handler"], tool["proposal"]
        lines += [f"  {tool['name']:<24} {_EFFECT[tool['effect']]}",
                  *_fill("     does     ", tool["summary"], 14),
                  *_fill("     handler  ", handler["callable"], 14),
                  f"              {handler['location']}"]
        if proposal is not None:
            for item in proposal["actions"]:
                lines += _fill("     submits  ", f"{item['action']} [{item['schema']}]", 14)
            if proposal["admitted_by"] is not None:
                lines += [*_fill("     admits   ", proposal["admitted_by"]["callable"], 14),
                          f"              {proposal['admitted_by']['location']}"]
            lines += _fill("     route    ", proposal["route"], 14)
    return lines


def render_explanation(explanation: Mapping[str, Any]) -> str:
    lines = _render_selection(explanation)
    for environment, instance in explanation["instances"].items():
        own = instance["baseline"] != explanation["selection"]["default"]["id"]
        lines += ["", f"== {environment} (instance {instance['instance_id']}"
                  + (f", baseline {instance['baseline']}" if own else "") + ") ==",
                  "Resolution, in the order expand_native_defaults applies it"]
        lines += _render_chain(instance) + _render_route(instance) + _render_tools(instance)
    return "\n".join([*lines, "", explanation["caveat"]]) + "\n"


# -- the `baselines` subcommand -------------------------------------------------

def command_baselines(args: Any, root: Path) -> int:
    """`baselines [--json]` as before; `--explain [--environment KIND] [--json]` added."""
    if getattr(args, "environment", None) is not None and not getattr(args, "explain", False):
        raise BaselineExplainError("--environment requires --explain")
    if getattr(args, "explain", False):
        explanation = explain_baselines(root, environment=args.environment)
        if args.json:
            print(json.dumps(explanation, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(render_explanation(explanation), end="")
        return 0
    baseline = load_native_baseline(root)
    if args.json:
        print(json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"{baseline['id']}\tsha256={baseline['sha256']}")
        for environment in baseline["instances"]:
            selected = load_native_baseline(root, default_baseline_id(environment))
            instance = selected["instances"][environment]
            print(f"{environment}\ttools={instance['tool_surface']}"
                  f"\tprompt={instance['prompt_bundle']}"
                  + (f"\tbaseline={selected['id']}" if selected["id"] != baseline["id"] else ""))
        print(CAVEAT)
    return 0
