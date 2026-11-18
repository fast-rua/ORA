"""Run admission: missing qualification and changed worker code are warnings.

Historical qualification producers/readers remain strict. This adapter admits
an unqualified run without claiming that its worker has passed qualification.
Corrupt evidence, wrong task/layout/device and live health failures still fail.
"""

from pathlib import Path
from typing import Any, Mapping

from ora.orchestrator import robodojo_qualification as qualification

WARNING_SCHEMA = "ora.robodojo-unqualified-admission.v1"


def admission(runtime: Mapping[str, Any], *, workspace_root: Path, harness_commit: str) -> dict[str, Any]:
    """Validate available evidence, recording an explicit absence when unavailable."""
    reference = runtime.get("infrastructure_qualification")
    missing = "no infrastructure qualification binding"
    if reference is not None:
        try:
            return qualification.load_qualification(runtime, workspace_root=workspace_root,
                                      harness_commit=harness_commit, allow_kernel_drift=True)
        except FileNotFoundError as exc:
            missing = str(exc)
    return {
        "schema": WARNING_SCHEMA, "qualification_available": False,
        "reference": reference, "plan_commit": harness_commit,
        "plan_worker_kernel_sha256": qualification.worker_kernel_sha256(harness_commit),
        "warnings": [{"code": "qualification_missing", "detail": missing}],
        "claim": "Run admitted without matching qualification; no qualification success claimed.",
    }


def warnings(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect warnings from combined, layered and revalidated record projections."""
    result = list(record.get("warnings", []))
    for name in ("kernel_admission", "card", "task"):
        child = record.get(name)
        if isinstance(child, Mapping):
            result.extend(warnings(child))
    return result
