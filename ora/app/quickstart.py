"""``ora run`` on Synthetic: the bounded quickstart episode through the normal plan, preflight and launch path.

The RunSpec is ``ora.orchestrator.spec.quickstart.synthetic_spec``; this module
launches it, waits for the Supervisor to land and seal it, and prints the outcome.
Its model comes from the ORA_* variables (``spec.quickstart.environment_policy``),
or from a Provider profile with ``--provider-profile`` (``profile_policy``).
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any, MutableMapping

from ora.version import ora_command


def profile_policy(profile_id: str, environ: MutableMapping[str, str] | None = None) -> dict[str, Any]:
    """The quickstart's policy from a Provider profile: ``ora run --env synthetic --provider-profile ID``.

    As on a GPU task (``run.prepare_environment``), the profile's base-URL
    variable defaults to its endpoint and its key comes from the environment
    or the user's store (``ora providers key``); a missing key or another
    endpoint stops here, before preflight, with what to set.
    """
    from ora.orchestrator.spec.inputs import provider_policy

    from .local import provider_check, provider_environment

    values = os.environ if environ is None else environ
    policy = provider_policy(profile_id)
    values.update(provider_environment(values, profile_id))
    check = provider_check(values, profile_id)
    if not check["passed"]:
        expected = check["detail"].get("expected_endpoint")
        if expected:
            raise ValueError(f"{policy['base_url_env']} must be {expected} for {profile_id} (or unset it)")
        raise ValueError(f"{profile_id} needs {policy['api_key_env']}: export it, or store it once with "
                         f"{ora_command()} providers key {profile_id}")
    return policy


def command_run(args: argparse.Namespace) -> int:
    from ora.orchestrator import launch, leases
    from ora.orchestrator.spec.quickstart import synthetic_spec
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    profile = getattr(args, "provider_profile", None)
    plan = resolve_run_spec(synthetic_spec(profile_policy(profile) if profile else None))
    preflight = launch._preflight(plan)
    if not preflight["passed"]:
        return 2
    run_id = launch._spawn(plan, preflight, yes=True)
    root = launch._run_root(run_id)
    pid = leases._load(root / "process.json")["pid"]
    print(f"episode: {root}", flush=True)
    deadline = time.monotonic() + plan["budgets"]["wall_time_s"] + 15
    try:
        while time.monotonic() < deadline:
            try:
                finished = os.waitpid(pid, os.WNOHANG)[0] == pid
            except ChildProcessError:
                finished = not leases._pid_alive(pid)
            if (root / "result.json").is_file() and finished:
                result = leases._load(root / "result.json")
                print(f"state: {result['state']}; model calls: {result.get('model_calls', 0)}; "
                      "official_success: null (Synthetic has no benchmark verdict)")
                if (root / "seal.json").is_file():
                    print("evidence: evaluated and sealed")
                else:
                    print(f"evidence was not sealed; inspect {root / 'supervisor.log'}")
                    return 1
                return 0 if result["state"] == "completed" else 1
            if finished and not (root / "result.json").is_file():
                raise RuntimeError(f"Supervisor stopped before writing a result; inspect {root / 'supervisor.log'}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        launch.request_stop(run_id)
        return 130
    launch.request_stop(run_id)
    raise RuntimeError(f"Quickstart exceeded its bounded deadline; inspect {root / 'supervisor.log'}")
