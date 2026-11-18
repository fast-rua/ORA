"""Fail-closed host, provider, simulator and qualification admission."""

from __future__ import annotations

from ora.agent.composition.baseline import BASELINE_CATALOGS

import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping


from .provider_probe_diagnostics import ProviderProbeDiagnostics
from .provider_image_answer import image_answer_word as _image_answer_word, names_red as _names_red
from .provider_probe_stages import (
    _NATIVE_PROVIDER_PROBE_SCHEMA,
    ProbeRun,
    probe_failure_detail,
    probe_request_options,
    response_level_miss,
    provider_marker_verdict,
    run_probe_stages,
)
from ..episode import agent_core
from . import host_storage
from ..backends.registry import PreflightContext, backend, find, manifest_admitter
from .checks import check as _check
from ..spec.run_spec import PROJECT_ROOT
from ..spec.run_spec import file_sha256
from ..paths import resolve_workspace_root
from .resources import (
    ExecutionResourceError,
    RESOURCE_BINDING_SCHEMA,
    resolve_execution_resource,
)


Runner = Callable[..., subprocess.CompletedProcess[str]]


SERVICE_SET_BINDING_SCHEMA = "ora.model-service-set-binding.v2"

# Binding schema -> (model_services.clients class, endpoint override variable,
# default endpoint).
# A schema absent from this table is one this check cannot verify; it is
# reported as unverified rather than silently treated as satisfied.
_RESIDENT_SERVICE_CLIENTS: dict[str, tuple[str, str, str]] = {
    "ora.segmentation-service-binding.v1": (
        "Sam3ApiSegmenter", "ORA_SEGMENTATION_URL",
        "http://127.0.0.1:8114",
    ),
    "ora.grasp-proposal-service-binding.v1": (
        "ContactGraspNetServiceClient", "ORA_GRASP_URL",
        "http://127.0.0.1:8115",
    ),
}

_SERVICE_HEALTH_FIELDS = (
    "instance_id", "pid", "adapter_sha256", "model", "warmup_complete",
)


def _bound_tool_surface(plan: Mapping[str, Any]) -> Any:
    policy = plan.get("policy")
    surface = policy.get("tool_surface") if isinstance(policy, Mapping) else None
    return surface.get("profile") if isinstance(surface, Mapping) else None


def _verify_resident_service(
    binding: Mapping[str, Any], profile: Any,
) -> dict[str, Any]:
    """Compare one bound service against the live service, or raise."""

    schema = binding.get("schema")
    entry = _RESIDENT_SERVICE_CLIENTS.get(str(schema))
    if entry is None:
        raise ValueError(f"unknown resident service binding schema: {schema!r}")
    class_name, endpoint_variable, default_endpoint = entry
    if binding.get("tool_surface") != profile:
        raise ValueError(
            f"{schema} prerequisite is bound to a different tool surface"
        )
    from ora.agent.model_services import clients as service_clients
    from ora.agent.model_services.binding import service_client_arguments

    client = getattr(service_clients, class_name)(
        **service_client_arguments(binding, endpoint_variable, default_endpoint),
        timeout_s=5.0,
    )
    try:
        health = client.require_ready()
    finally:
        client.close()
    return {
        "schema": schema,
        "endpoint": binding.get("endpoint"),
        **{field: health.get(field) for field in _SERVICE_HEALTH_FIELDS},
    }


def _resident_service_prerequisite(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a source-bound resident service or service set before launch.

    Both shapes are checked: the single v1 segmentation binding, and the v2
    set binding an experiment emits when it holds several services at once.
    Only an absent prerequisite is "not required" -- a set binding used to fall
    through the single-schema comparison and pass vacuously while SAM3 and
    Contact-GraspNet went unverified.
    """

    prerequisite = plan.get("prerequisite")
    if not isinstance(prerequisite, Mapping):
        return _check("resident_service_prerequisite", True, "not required")
    schema = str(prerequisite.get("schema", ""))
    profile = _bound_tool_surface(plan)
    try:
        if schema == SERVICE_SET_BINDING_SCHEMA:
            services = prerequisite.get("services")
            if not isinstance(services, Mapping) or not services:
                raise ValueError("service-set binding declares no services")
            if plan.get("policy", {}).get("baseline", {}).get("id") in BASELINE_CATALOGS:
                schemas = {
                    child.get("schema") for child in services.values()
                    if isinstance(child, Mapping)
                }
                if schemas != set(_RESIDENT_SERVICE_CLIENTS):
                    raise ValueError(
                        "Native full baseline requires both segmentation and grasp services"
                    )
            scope = prerequisite.get("capability_scope")
            if not isinstance(scope, list) or not scope:
                raise ValueError("service-set binding declares no capability scope")
            observed = {
                name: _verify_resident_service(child, profile)
                for name, child in sorted(services.items())
                if isinstance(child, Mapping)
            }
            if len(observed) != len(services):
                raise ValueError("service-set binding holds a malformed service")
            return _check("resident_service_prerequisite", True, {
                "schema": schema,
                "capability_scope": list(scope),
                "services": observed,
            })
        if schema in _RESIDENT_SERVICE_CLIENTS:
            return _check(
                "resident_service_prerequisite", True,
                _verify_resident_service(prerequisite, profile),
            )
    except Exception as exc:
        return _check("resident_service_prerequisite", False, str(exc))
    # A prerequisite this check does not own. Say so instead of implying
    # a resident service was checked and found healthy.
    return _check("resident_service_prerequisite", True, {
        "verified": False,
        "reason": "no resident-service verifier for this binding schema",
        "schema": schema or "prerequisite carries no schema",
    })


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True,
        timeout=60, check=False,
    )


def _native_provider_candidate(
    plan: Mapping[str, Any], api_key: str, credential_pool: list[str],
) -> dict[str, Any]:
    """Run the complete qualification against exactly one host credential.

    The stages, their requests and the verdict live in
    ``provider_probe_stages.py``; a stage that raises refuses the candidate
    with that stage's name and the probes recorded so far.
    """

    policy = plan["policy"]
    request_options = probe_request_options(policy)
    marker = f"ROBOT-HARNESS-{secrets.token_hex(8)}"
    run = ProbeRun(ProviderProbeDiagnostics(), credential_pool)
    try:
        outcome = run_probe_stages(
            run, policy, api_key=api_key, marker=marker,
            request_options=request_options,
        )
    except Exception as exc:
        return _check("provider_marker", False, probe_failure_detail(run, exc))
    passed, detail = provider_marker_verdict(policy, run, marker, outcome)
    return _check("provider_marker", passed, detail)



def provider_marker_configuration(
    policy: Mapping[str, Any], environ: Mapping[str, str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """The offline half of ``provider_marker`` for a Native API policy.

    Both variables are set and the base URL equals the plan endpoint. ``ora
    doctor`` runs exactly this before any task is prepared; launch preflight
    runs it, then the live probe. The detail names variables, never values.
    """
    values = os.environ if environ is None else environ
    base_url = values.get(policy["base_url_env"])
    api_key = values.get(policy["api_key_env"])
    if not base_url or not api_key:
        return False, {
            "configured": False,
            "required_environment": [policy["base_url_env"], policy["api_key_env"]],
        }
    if base_url.rstrip("/") != policy["endpoint"]:
        return False, {
            "configured": True,
            "endpoint_matches_plan": False,
            "expected_endpoint": policy["endpoint"],
        }
    return True, {"configured": True, "endpoint_matches_plan": True}


def _provider_marker(plan: Mapping[str, Any], runner: Runner) -> dict[str, Any]:
    policy = plan["policy"]
    if policy["kind"] == "native_api":
        if not policy.get("require_live_marker", True):
            return _check("provider_marker", True, "not required for this policy")
        configured, detail = provider_marker_configuration(policy)
        if not configured:
            return _check("provider_marker", False, detail)
        api_key = os.environ[policy["api_key_env"]]
        from ora.agent.providers.credentials import (
            CREDENTIAL_HTTP_STATUSES, SELECTION_SCHEMA, bind_credential_handoff,
            credential_ring, redact_credentials,
        )

        keys, _ = credential_ring(
            api_key, os.environ.get(f"{policy['api_key_env']}S", "").split(","),
            selected_index=0,
        )
        _, start = credential_ring(
            keys[0], keys[1:],
            (plan.get("runtime") or {}).get("resource_binding", {}).get("slot_id"),
        )
        attempts = []
        selected = None
        for offset in range(len(keys)):
            index = (start + offset) % len(keys)
            for response_retry in (False, True):
                check = _native_provider_candidate(plan, keys[index], keys)
                detail = redact_credentials(check["detail"], keys)
                check["detail"] = detail
                status = detail.get("http_status")
                attempts.append({
                    "credential_index": index, "passed": check["passed"],
                    "stage": detail.get("stage", "qualification"),
                    "http_status": status,
                    "error_type": detail.get("error_type"),
                    "response_retry": response_retry,
                })
                if check["passed"] or not response_level_miss(detail):
                    break
            if check["passed"]:
                selected = index
                break
            if status not in CREDENTIAL_HTTP_STATUSES:
                break
        check["detail"]["credential_selection"] = {
            "schema": SELECTION_SCHEMA, "pool_size": len(keys),
            "start_index": start, "selected_index": selected, "attempts": attempts,
        }
        if selected is not None:
            return bind_credential_handoff(check, plan, keys, selected)
        return check
    return _check("provider_marker", True, "not required for this policy")


def _compactor_provider_marker(plan: Mapping[str, Any]) -> dict[str, Any]:
    """A bound compactor's endpoint (``policy.context_compactor``), probed as the policy's is.

    The marker and image stages run (compactor packets carry images); the tool
    continuity stage does not, since the binding names no qualification schema:
    the compactor is tool-free. One key, no pool; a response-level miss reruns once.
    """

    from ora.agent.providers.compactor_binding import compactor_configuration
    from ora.agent.providers.credentials import redact_credentials

    binding = plan["policy"]["context_compactor"]
    if not plan["policy"].get("require_live_marker", True):
        return _check("compactor_provider_marker", True, "not required for this policy")
    configured, detail = compactor_configuration(binding)
    if not configured:
        return _check("compactor_provider_marker", False, detail)
    api_key = os.environ[binding["api_key_env"]]
    for response_retry in (False, True):
        run = ProbeRun(ProviderProbeDiagnostics(), [api_key])
        marker = f"ROBOT-HARNESS-{secrets.token_hex(8)}"
        try:
            outcome = run_probe_stages(run, binding, api_key=api_key, marker=marker,
                                       request_options=probe_request_options(binding))
        except Exception as exc:
            passed, detail = False, probe_failure_detail(run, exc)
        else:
            passed, detail = provider_marker_verdict(binding, run, marker, outcome)
        detail = redact_credentials({**detail, "response_retry": response_retry}, [api_key])
        if passed or not response_level_miss(detail):
            break
    return _check("compactor_provider_marker", bool(passed), detail)


def _agent_core(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The plan's agent core can start: Node.js >= 22.19 and the locked ora-pi packages for Pi.

    A plan resolved for the retired Python Native core fails this check with why."""
    policy = plan.get("policy", {})
    if policy.get("kind") != "native_api":
        return _check("agent_core", True, "not required for this policy")
    try:
        agent_core.selected(policy)
    except agent_core.AgentCoreError as exc:
        return _check("agent_core", False, str(exc))
    status = agent_core.pi_runtime()
    if not status["passed"]:
        return _check("agent_core", False, status["remedy"])
    return _check("agent_core", True, {
        "agent_core": "pi", "node": status["node"]["path"], "node_version": status["node"]["version"],
        "pi_version": status["dependencies"]["pi_version"], "dependencies": status["dependencies"]["detail"],
    })


def _code_mode_sandbox(plan: Mapping[str, Any]) -> dict[str, Any]:
    """A plan with ``policy.code_mode`` runs a script here: bwrap, user namespaces and the interpreter, held to its
    lock; the detail records the interpreter."""
    from ora.agent.code_mode.sandbox import sandbox_status

    status = sandbox_status()
    return _check("code_mode_sandbox", bool(status["ok"]), status)


def _qualification(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The plan's qualification manifest admits its seed (and every play scene's), by the backend that owns it.

    A plan without a required manifest passes, unless its maintained baseline
    needs one or it states an exception the plan's backend must check.
    """

    qualification = plan["qualification"]
    kind = plan.get("runtime", {}).get("kind")
    selected = find(kind)
    if not qualification["required"] and qualification.get("exception") is not None:
        exception = qualification["exception"]
        hooks = selected.qualification_exception if selected is not None else None
        if hooks is None:
            return _check("seed_qualification", False, {
                "exception": exception.get("kind") if isinstance(exception, Mapping) else None,
                "error": f"runtime.kind {kind} has no qualification exception"})
        passed, detail = hooks.check_exception(plan, resolve_workspace_root(PROJECT_ROOT))
        return _check("seed_qualification", passed, detail)
    if not qualification["required"]:
        if (
            plan.get("policy", {}).get("baseline", {}).get("id") in BASELINE_CATALOGS
            and selected is not None and selected.capabilities.baseline_requires_seed_manifest
        ):
            return _check(
                "seed_qualification", False,
                "Native full RoboTwin baseline requires an explicit qualified seed manifest",
            )
        return _check("seed_qualification", True, "not required")
    relative = Path(qualification["manifest"]["path"])
    path = (
        resolve_workspace_root(PROJECT_ROOT) / relative
        if relative.parts[:2] == ("outputs", plan["experiment"])
        else PROJECT_ROOT / relative
    )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _check("seed_qualification", False, str(exc))
    admitter = manifest_admitter(manifest.get("schema"), kind=kind)
    memo: dict[str, Any] = {}  # what an admitter computes once per plan (RoboDojo's play kernel)

    def admit(seed: int) -> tuple[bool, dict[str, Any]]:
        if admitter is None:
            return False, {"schema": manifest.get("schema"), "error": "no backend admits this manifest schema"}
        return admitter(manifest, plan, seed=seed, root=PROJECT_ROOT, memo=memo)

    passed, detail = admit(plan["seed"])
    # A free-play session may reset to each of its scenes: the manifest admits every one.
    refused = sorted({scene["seed"] for scene in plan.get("play", {}).get("scenes", ()) if not admit(scene["seed"])[0]})
    if refused:
        passed, detail = False, {**detail, "play_scene_seeds_not_admitted": refused}
    return _check("seed_qualification", passed, detail)


def preflight_context(runner: Runner | None = None) -> PreflightContext:
    """What a backend's preflight hook uses, read from this module now: a test that
    replaces ``_git``, ``resolve_workspace_root`` or ``subprocess.run`` reaches it."""

    return PreflightContext(project_root=PROJECT_ROOT, check=_check, git=_git,
                            workspace_root=resolve_workspace_root,
                            runner=runner if runner is not None else subprocess.run)


def _runtime(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The execution resource, then the runtime's own checks from its backend."""

    resource_check = _execution_resource(plan)
    return [resource_check, *backend(plan["runtime"]["kind"]).preflight_runtime(plan, preflight_context())]


def _execution_resource(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a source-tracked resource profile and its current GPU capacity."""

    runtime = plan["runtime"]
    binding = runtime.get("resource_binding")
    if binding is None:
        return _check(
            "execution_resource", True,
            "implicit exclusive-host Runtime lease",
        )
    detail: dict[str, Any] = {"binding": binding}
    try:
        if not isinstance(binding, Mapping) or binding.get("schema") != (
            RESOURCE_BINDING_SCHEMA
        ):
            raise ExecutionResourceError("resource binding schema is invalid")
        expected = resolve_execution_resource(
            binding.get("profile_id"),
            runtime_kind=str(runtime["kind"]),
            required_host=(
                str(runtime["required_host"])
                if isinstance(runtime.get("required_host"), str)
                else None
            ),
        )
        if dict(binding) != expected:
            raise ExecutionResourceError("resource profile drifted after planning")
        worker_environment = binding.get("worker_environment", {})
        cuda = worker_environment.get("CUDA_VISIBLE_DEVICES")
        if cuda is None:
            return _check("execution_resource", True, detail)
        # The first listed GPU is the lane's compute device (its CUDA ordinal 0);
        # a Mesa LIBERO lane also lists 0 so robosuite accepts EGL device 0.
        devices = cuda.split(",") if isinstance(cuda, str) else []
        if not devices or not all(item.isdigit() for item in devices):
            raise ExecutionResourceError(
                "CUDA_VISIBLE_DEVICES must list numeric GPUs"
            )
        cuda = devices[0]
        nvidia = shutil.which("nvidia-smi")
        if nvidia is None:
            raise ExecutionResourceError("nvidia-smi is unavailable")
        query = subprocess.run(
            [
                nvidia, "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if query.returncode:
            raise ExecutionResourceError(
                query.stderr.strip() or "cannot query GPU resources"
            )
        inventory: dict[str, int] = {}
        for line in query.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
                inventory[fields[0]] = int(fields[1])
        if cuda not in inventory:
            raise ExecutionResourceError(f"GPU {cuda} is unavailable")
        # One binding's own floor against live free memory. Co-residency is
        # admitted by the lease claim (leases._resource_refusal), which adds the
        # declared-capacity sum over every active lease on this device; this
        # check stays a per-binding probe and deliberately has no lease view.
        minimum = int(binding["minimum_free_memory_mib"])
        detail.update({
            "gpu_index": int(cuda),
            "free_memory_mib": inventory[cuda],
            "minimum_free_memory_mib": minimum,
        })
        if inventory[cuda] < minimum:
            raise ExecutionResourceError(
                f"GPU {cuda} has {inventory[cuda]} MiB free; {minimum} required"
            )
        return _check("execution_resource", True, detail)
    except (ExecutionResourceError, KeyError, TypeError, ValueError) as exc:
        detail["error"] = str(exc)
        return _check("execution_resource", False, detail)


def run_preflight(
    plan: Mapping[str, Any], *, runner: Runner = subprocess.run,
) -> dict[str, Any]:
    from ..spec.source_overlay import OVERLAY_FILE, matches_plan, source_state

    if (PROJECT_ROOT / OVERLAY_FILE).is_file() or "source_overlay" in plan:
        state = source_state(PROJECT_ROOT)
        current_source = {
            "expected": plan["source"]["commit"], "observed": state.commit, "dirty": state.dirty,
            "overlay_expected": (plan.get("source_overlay") or {}).get("sha256"),
            "overlay_observed": (state.overlay or {}).get("sha256"),
            **({"error": state.error} if state.error is not None else {}),
        }
        source_passed = matches_plan(state, plan)
    else:
        revision = _git(PROJECT_ROOT, "rev-parse", "HEAD")
        dirty = _git(PROJECT_ROOT, "status", "--porcelain", "--untracked-files=normal")
        current_source = {
            "expected": plan["source"]["commit"],
            "observed": revision.stdout.strip(),
            "dirty": bool(dirty.stdout.strip()),
        }
        source_passed = (
            revision.returncode == 0
            and dirty.returncode == 0
            and current_source["observed"] == current_source["expected"]
            and current_source["dirty"] is False
        )
    checks = [
        _check("source_identity", source_passed, current_source),
        *host_storage.storage_checks(plan, check=_check),
        _qualification(plan),
        *_runtime(plan),
        _resident_service_prerequisite(plan),
        _provider_marker(plan, runner),
        # Only a plan whose compactor is bound to its own profile has this check.
        *([_compactor_provider_marker(plan)] if "context_compactor" in plan["policy"] else []),
        _agent_core(plan),
        # Only a plan with policy.code_mode runs scripts in the sandbox.
        *([_code_mode_sandbox(plan)] if plan["policy"].get("code_mode") else []),
    ]
    with tempfile.TemporaryDirectory(prefix="ora-preflight-") as directory:
        root = Path(directory)
        probe = root / "scratch/probe.txt"
        probe.parent.mkdir(parents=True)
        probe.write_text("ok", encoding="utf-8")
        read_only = root / "inputs/read-only.txt"
        read_only.parent.mkdir(parents=True)
        read_only.write_text("visible", encoding="utf-8")
        read_only.chmod(0o444)
        checks.append(_check(
            "runtime_filesystem_contract",
            probe.read_text() == "ok" and read_only.read_text() == "visible",
            {
                "scratch_writable": True,
                "inputs_readable": True,
                "os_read_namespace_isolation_claimed": False,
            },
        ))
    return {
        "schema": "ora.preflight.v1",
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "created_unix_s": time.time(),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
