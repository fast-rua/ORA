"""Which agent core runs a native_api policy, and whether the Pi core can start.

``policy.agent_core`` names the implementation of the robot-agnostic agent
core (provider loop, request assembly, context management, agent-side
evidence) behind a ``native_api`` policy. ORA has one: ``pi``, the TypeScript
core in ``ora-pi/`` on the pinned Pi agent packages, run by Node.js >= 22.19
(``ora.agent.pi``). Every fresh plan records it.

The Python Native core (``native``) was removed from ORA; it remains only in
the robot-harness repository. A RunSpec, ``ORA_AGENT_CORE`` or
``--agent-core`` that names it is refused, and so is a persisted plan resolved
for it: one that records ``agent_core: native``, or one resolved before the
field existed (``UNRECORDED_AGENT_CORE``: those ran on Native). Only starting a
policy session selects a core; evaluation, sealing, the viewer, ``ora watch``
and ``ora replay`` read such a run's sealed evidence as before.

The field names an implementation, not what the model sees: it is part of the
plan identity, not of ``policy_identity.model_visible_sha256``.

``ORA_AGENT_CORE`` (and ``--agent-core``, which sets it for one command) is
still read, so that a script naming the retired core fails instead of running
on Pi unannounced; ``pi`` is its only value.

Importing this module loads only the standard library: setup and doctor use it
before the locked dependencies exist.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from ora.version import ora_command

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PI_ROOT = PROJECT_ROOT / "ora-pi"
AGENT_CORES = ("pi",)
DEFAULT_AGENT_CORE = "pi"
#: The Python core ORA no longer includes (it remains in the robot-harness repository).
RETIRED_AGENT_CORE = "native"
#: The core of a plan resolved before ``policy.agent_core`` existed: the retired one.
UNRECORDED_AGENT_CORE = RETIRED_AGENT_CORE
OVERRIDE_ENV = "ORA_AGENT_CORE"
NODE_ENV = "ORA_NODE"
#: Node 22.19 runs the TypeScript sources directly (type stripping): no build step.
MIN_NODE = (22, 19, 0)
#: The Node release the installation guide unpacks into ``~/.local`` without sudo.
NODE_RELEASE = "v22.20.0"
NPM_CI = ("ci", "--prefer-offline", "--no-audit", "--no-fund")
_MIN_NODE_TEXT = "Node.js >= {}.{}".format(*MIN_NODE[:2])


class AgentCoreError(RuntimeError):
    """The agent core selection is invalid, or the selected core cannot start."""


def retired_message(named: str) -> str:
    """Why ``named`` (how the caller spelled the retired core) is refused, and what to use."""

    return (f"{named} names the Python Native agent core, which was removed from ORA and remains only in "
            "the robot-harness repository; ORA runs on the Pi agent core only: use pi, or leave it unset")


def retired_plan_message(reason: str) -> str:
    """Why a persisted plan resolved for the retired core does not start."""

    return (f"this plan was resolved for the Python Native agent core ({reason}), which was removed from ORA "
            "and remains only in the robot-harness repository, so ORA does not start it: resolve its RunSpec "
            "again to run it on the Pi agent core. Its sealed evidence still evaluates, opens in the viewer "
            "and replays with ora watch and ora replay")


def _choice(value: Any, label: str, separator: str) -> str:
    if value == RETIRED_AGENT_CORE:
        raise AgentCoreError(retired_message(f"{label}{separator}{value}"))
    if value not in AGENT_CORES:
        raise AgentCoreError(f"{label} must be pi, not {value!r}")
    return str(value)


def override(environ: Mapping[str, str] | None = None) -> str | None:
    """The ``ORA_AGENT_CORE`` override (``pi``), or ``None`` when it is unset or empty."""

    value = (os.environ if environ is None else environ).get(OVERRIDE_ENV)
    return None if not value else _choice(value, OVERRIDE_ENV, "=")


def resolve(policy: MutableMapping[str, Any], environ: Mapping[str, str] | None = None) -> str:
    """Set ``policy["agent_core"]`` of a native_api policy being resolved: override, field, default."""

    forced = override(environ)
    if forced is not None:
        policy["agent_core"] = forced
    policy["agent_core"] = _choice(policy.get("agent_core", DEFAULT_AGENT_CORE), "policy.agent_core", ": ")
    return policy["agent_core"]


def selected(policy: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> str:
    """The core a persisted native_api plan's session builds: ``pi``.

    A plan resolved for the retired Native core, recorded or from before the
    field existed, is refused whatever the override says: its plan identity
    names that core.
    """

    override(environ)  # ORA_AGENT_CORE=native is refused; pi changes nothing
    if "agent_core" not in policy:
        raise AgentCoreError(retired_plan_message(
            "it predates policy.agent_core, and every plan without the field ran on Native"))
    if policy["agent_core"] == RETIRED_AGENT_CORE:
        raise AgentCoreError(retired_plan_message("policy.agent_core: native"))
    return _choice(policy["agent_core"], "policy.agent_core", ": ")


def apply_cli_choice(value: str | None, environ: MutableMapping[str, str] | None = None) -> None:
    """``--agent-core``: set the override for this command and its children."""

    if value is None:
        return
    values = os.environ if environ is None else environ
    _choice(value, "--agent-core", " ")
    current = override(values)
    if current is not None and current != value:
        raise AgentCoreError(f"--agent-core {value} conflicts with {OVERRIDE_ENV}={current}")
    values[OVERRIDE_ENV] = value


class _AgentCoreOption(argparse.Action):
    """Apply ``--agent-core`` while parsing, so every later step of the command sees the override."""

    def __call__(self, parser: Any, namespace: Any, values: Any, option_string: str | None = None) -> None:
        try:
            apply_cli_choice(values)
        except AgentCoreError as exc:
            parser.error(str(exc))
        setattr(namespace, self.dest, values)


def add_cli_option(parser: Any) -> None:
    """``--agent-core``, hidden: ``pi`` is its only value; the retired ``native`` is refused with why."""

    parser.add_argument("--agent-core", action=_AgentCoreOption, metavar="pi", help=argparse.SUPPRESS)


# --- the Pi core's runtime: Node and the locked ora-pi packages -------------------------------

def _node_version(path: str) -> tuple[int, int, int] | None:
    try:
        text = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", text)
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def node_candidates(environ: Mapping[str, str] | None = None) -> list[str]:
    """``ORA_NODE`` alone when set; else ``node`` on PATH, ``~/.local/bin/node``, nvm newest first."""

    values = os.environ if environ is None else environ
    if values.get(NODE_ENV):
        return [values[NODE_ENV]]
    found = [shutil.which("node", path=values.get("PATH")), os.path.expanduser("~/.local/bin/node")]
    nvm = sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/v*/bin/node")),
                 key=lambda path: [int(part) if part.isdigit() else 0
                                   for part in Path(path).parts[-3].lstrip("v").split(".")])
    return list(dict.fromkeys(path for path in (*found, *reversed(nvm)) if path))


def find_node(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The first candidate that is Node.js >= 22.19, and every executable candidate seen."""

    seen = []
    for candidate in node_candidates(environ):
        if not (os.path.isfile(candidate) and os.access(candidate, os.X_OK)):
            continue
        version = _node_version(candidate)
        text = "v{}.{}.{}".format(*version) if version else None
        seen.append({"path": candidate, "version": text})
        if version is not None and version >= MIN_NODE:
            return {"passed": True, "path": candidate, "version": text, "candidates": seen}
    return {"passed": False, "path": None, "version": None, "candidates": seen}


def npm_for(node: str | None) -> str | None:
    """The npm installed beside ``node``, else the first one on PATH."""

    if node:
        beside = Path(node).parent / "npm"
        if beside.is_file() and os.access(beside, os.X_OK):
            return str(beside)
    return shutil.which("npm")


def node_install_command() -> str:
    """Unpack the pinned Node release into ``~/.local`` (no sudo) for this platform."""

    system = "darwin" if platform.system() == "Darwin" else "linux"
    machine = "arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "x64"
    name = f"node-{NODE_RELEASE}-{system}-{machine}"
    return (f"mkdir -p ~/.local && curl -fsSL https://nodejs.org/dist/{NODE_RELEASE}/{name}.tar.xz "
            "| tar -xJ -C ~/.local --strip-components=1")


def pi_dependencies(root: Path = PI_ROOT) -> dict[str, Any]:
    """Whether ``ora-pi/node_modules`` holds every runtime package at its locked version.

    Reads ``package-lock.json`` and each installed ``package.json``; runs no npm
    and touches no network.
    """

    try:
        lock = json.loads((root / "package-lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"passed": False, "detail": f"{root / 'package-lock.json'} is unreadable: {exc}"}
    expected = {
        path: entry.get("version") for path, entry in lock.get("packages", {}).items()
        if path.startswith("node_modules/") and not (entry.get("dev") or entry.get("optional") or entry.get("link"))
    }
    missing = []
    for path, version in sorted(expected.items()):
        try:
            installed = json.loads((root / path / "package.json").read_text(encoding="utf-8")).get("version")
        except (OSError, ValueError):
            installed = None
        if installed != version:
            missing.append(f"{path[len('node_modules/'):]}@{version} ({installed or 'absent'})")
    pi = lock.get("packages", {}).get("node_modules/@earendil-works/pi-coding-agent", {}).get("version")
    return {"passed": not missing, "pi_version": pi, "locked_packages": len(expected),
            "detail": (f"{len(expected)} locked runtime packages installed" if not missing
                       else f"{len(missing)} of {len(expected)} locked packages differ: " + ", ".join(missing[:3])
                       + (", ..." if len(missing) > 3 else ""))}


def pi_runtime(environ: Mapping[str, str] | None = None, *, root: Path = PI_ROOT) -> dict[str, Any]:
    """Readiness of the Pi core's runtime, with a remedy when it is not ready. No credential is read."""

    node, dependencies = find_node(environ), pi_dependencies(root)
    status = {"passed": node["passed"] and dependencies["passed"], "node": node, "dependencies": dependencies}
    if not status["passed"]:
        status["remedy"] = unavailable_message(status)
    return status


def _node_problem(node: Mapping[str, Any]) -> str:
    found = ", ".join(f"{item['version'] or 'unknown version'} at {item['path']}" for item in node["candidates"])
    return (f"{_MIN_NODE_TEXT} is required ("
            + (f"found {found}" if found else f"no node on PATH, {NODE_ENV}, ~/.local/bin or ~/.nvm")
            + f"); install it without sudo with `{node_install_command()}` and put ~/.local/bin first on PATH, "
            f"or set {NODE_ENV} to a {_MIN_NODE_TEXT} binary")


def unavailable_message(status: Mapping[str, Any]) -> str:
    problems = [] if status["node"]["passed"] else [_node_problem(status["node"])]
    if not status["dependencies"]["passed"]:
        problems.append(
            f"the ora-pi packages are not installed as locked ({status['dependencies']['detail']}); run "
            f"{ora_command()} setup, or npm {' '.join(NPM_CI)} --prefix ora-pi")
    return "the Pi agent core cannot start: " + "; ".join(problems)


def require_pi_runtime(environ: Mapping[str, str] | None = None) -> str:
    """The Node.js binary for the Pi core; an actionable ``AgentCoreError`` when it cannot start."""

    status = pi_runtime(environ)
    if not status["passed"]:
        raise AgentCoreError(status["remedy"])
    return str(status["node"]["path"])


def npm_ci_failure(detail: str) -> str:
    """The error for a failed ``npm ci`` in ora-pi; an offline host needs a populated npm cache."""

    return (f"npm ci in ora-pi failed: {detail.strip()[-600:]}. It downloads the locked Pi packages from the "
            "npm registry unless the npm cache already holds them (--prefer-offline); on an offline host, "
            "run it once where the registry is reachable with the same npm cache, or point "
            "npm_config_cache at a populated cache, then rerun ora setup")


def setup_requirement(environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    """``(node, npm)`` for ``ora setup``; an actionable ``AgentCoreError`` when either is missing."""

    node = find_node(environ)
    if not node["passed"]:
        raise AgentCoreError(f"ora setup: {_node_problem(node)}; then rerun ora setup")
    npm = npm_for(node["path"])
    if npm is None:
        raise AgentCoreError(f"ora setup: npm is neither beside {node['path']} nor on PATH; install Node.js "
                             f"with npm (`{node_install_command()}` includes it), then rerun ora setup")
    return str(node["path"]), npm


__all__ = [
    "AGENT_CORES", "AgentCoreError", "DEFAULT_AGENT_CORE", "MIN_NODE", "NODE_ENV", "NPM_CI", "OVERRIDE_ENV",
    "PI_ROOT", "RETIRED_AGENT_CORE", "UNRECORDED_AGENT_CORE", "add_cli_option", "apply_cli_choice",
    "find_node", "node_install_command", "npm_ci_failure", "npm_for", "override", "pi_dependencies", "pi_runtime",
    "require_pi_runtime", "resolve", "retired_message", "retired_plan_message", "selected", "setup_requirement",
]
