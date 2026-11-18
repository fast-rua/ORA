"""``ora task submit``: one run of a task as a scheduler request.

The request asks a scheduler (``--scheduler-url``, or ``ORA_SCHEDULER_URL``)
to run ``ora task run --resolved TASK`` in this checkout once it has bound a
card, a model replica and perception services. The request carries no code
version: the scheduler runs the command in this checkout (``run.cwd``) as it
is when the run starts, so pinning a commit means submitting from a worktree
of it and leaving that worktree alone until the run ends. ``--defaults FILE`` holds the
submitter's standing request fields (owner, window, environment, environment
files, scheduler URL); flags given on the command line win.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

URL_VARIABLE = "ORA_SCHEDULER_URL"
DEFAULT_KEYS = {"scheduler_url", "owner", "priority", "job_type", "window", "workspace", "cpu", "pin",
                "env", "env_files"}


class SubmitError(RuntimeError):
    """The request could not be built or was refused."""


def load_defaults(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = sorted(set(value) - DEFAULT_KEYS) if isinstance(value, Mapping) else ["<not a mapping>"]
    if unknown:
        raise SubmitError(f"{path}: unknown submit defaults {unknown}")
    return dict(value)


def checkout_root(start: Path) -> Path:
    out = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, check=False)
    if out.returncode:
        raise SubmitError(f"{start} is not inside a git checkout")
    return Path(out.stdout.strip())


def build_request(task: Mapping[str, Any], *, repo: Path, defaults: Mapping[str, Any],
                  name: str | None = None, env: Mapping[str, str] | None = None,
                  env_files: Mapping[str, str] | None = None, summary: Path | None = None,
                  **fields: Any) -> dict[str, Any]:
    """The scheduler request; ``fields`` override ``defaults`` (owner, priority, window, ...).

    ``summary`` is passed to ``ora task run --summary``: where the caller will find the task record.
    """

    chosen = {**defaults, **{key: value for key, value in fields.items() if value is not None}}
    if not chosen.get("owner"):
        raise SubmitError("a request needs an owner (--owner or the defaults file)")
    resources = {key: chosen[key] for key in ("window", "workspace", "cpu", "pin") if chosen.get(key) is not None}
    request: dict[str, Any] = {
        "type": "submit", "owner": chosen["owner"],
        "name": name or f"{task['experiment']}-{task['task']}-s{task['seed']}-l{task['layout']}",
        "ora": {key: task[key] for key in ("simulator", "model", "task", "seed", "layout", "experiment")},
        "resources": resources,
        "run": {
            "argv": [str(repo / "scripts" / "ora"), "task", "run", "--resolved",
                     json.dumps(dict(task), ensure_ascii=False, sort_keys=True),
                     *(["--summary", str(summary)] if summary is not None else [])],
            "cwd": str(repo),
            "env": {**(chosen.get("env") or {}), **dict(env or {})},
            "env_files": {**(chosen.get("env_files") or {}), **dict(env_files or {})},
        },
    }
    for key in ("priority", "job_type"):
        if chosen.get(key) is not None:
            request[key] = chosen[key]
    return request


def scheduler_url(defaults: Mapping[str, Any], given: str | None,
                  environ: Mapping[str, str] = os.environ) -> str:
    url = given or environ.get(URL_VARIABLE) or defaults.get("scheduler_url")
    if not url:
        raise SubmitError(f"no scheduler URL (--scheduler-url, {URL_VARIABLE} or the defaults file)")
    return str(url).rstrip("/")


def post(url: str, request: Mapping[str, Any], *, timeout_s: float = 60.0) -> dict[str, Any]:
    """POST the request to ``<url>/api/requests``; the scheduler's JSON reply."""

    data = json.dumps(request).encode("utf-8")
    message = urllib.request.Request(f"{url}/api/requests", data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(message, timeout=timeout_s) as response:  # noqa: S310 - operator-given URL
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as error:
        raise SubmitError(f"scheduler refused the request ({error.code}): "
                          f"{error.read().decode('utf-8', 'replace')}") from error


def pairs(values: Sequence[str], flag: str) -> dict[str, str]:
    result = {}
    for value in values:
        key, sep, text = value.partition("=")
        if not sep or not key:
            raise SubmitError(f"{flag} takes NAME=VALUE, not {value!r}")
        result[key] = text
    return result
