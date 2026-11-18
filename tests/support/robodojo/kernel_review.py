"""Hold the RoboDojo worker kernel manifest to the worker's import closure.

``WORKER_KERNEL_PATHS`` is written by hand; the worker's import graph is not.
Every module the worker statically imports must be a listed kernel path, a
reviewed exclusion below, or the target of a reviewed dispatcher edge that
only another backend takes. Anything else is a gap, and a table entry the
source no longer contains is drift; ``tests/ora/orchestrator/backends/robodojo/test_robodojo_kernel_paths.py``
requires both to be empty.

A reviewed exclusion is not bound, but its own imports are still followed, so
a new import inside it is checked like any other. A pruned dispatcher edge is
keyed by the (importer, imported) pair only: the check does not see which
branch or function of the dispatcher holds the import, so if the RoboDojo path
later starts using a module through an already-pruned pair, that use is not
detected here and needs review of the dispatcher itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ora.orchestrator.import_closure import import_closure, tree_reader
from ora.orchestrator.robodojo_kernel import ROBODOJO_BACKEND_PREFIX, WORKER_KERNEL_PATHS

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Where the isolated RoboDojo worker's executed Harness code starts: the RPC
#: worker process, the factory that composes its config, and the backend's
#: bootstrap and session.
ROBODOJO_WORKER_ENTRY_POINTS = (
    "ora/environment/worker.py",
    "ora/environment/factory.py",
    "ora/environment/backends/robodojo/bootstrap.py",
    "ora/environment/backends/robodojo/session.py",
)

#: Closure members deliberately not listed, each with the review that
#: excludes it. An excluded module is not bound; its imports are still followed.
#: ``contracts/skill_sequence.py`` is not here: it decides which skill-sequence
#: children the session admits and in what order (``begin_action_sequence``
#: derives them from ``sequence``/``resume_from``; the ordered-subset rule is
#: in ``validate_skill_sequence_action``), and no qualification check covers
#: it, so it is a listed kernel path.
KERNEL_REVIEWED_EXCLUSIONS: Mapping[str, str] = {
    "ora/contracts/toolkit_protocol.py":
        "declarative version evidence and warning records; no control, physics or production admission logic",
    "ora/contracts/__init__.py":
        "re-export aggregator the worker imports validate_runtime_action through; "
        "the re-exported code is bound in contracts/action.py, motion.py and pose.py",
    "ora/contracts/alignment.py":
        "read-only align_plan request/response schema; alignment plans are returned, never commanded",
    "ora/contracts/refusal.py":
        "the EnvironmentRefusal envelope carried back over RPC; it decides no command or verdict",
    "ora/contracts/workspace.py":
        "workspace-bounds schema imported only by workspace_profiles; CONFIGURATION_FIELDS compares "
        "the resolved workspace_bounds_m field for field",
    "ora/environment/benchmarks/robodojo_physics.py":
        "host admission table read only inside validate_runtime_extension, which is projected out of "
        "every card/scene dependency map; the worker never calls it; the device it selects is bound by "
        "CONFIGURATION_FIELDS.physics_device and resolved_environment_config.sim.device",
    "ora/environment/workspace_profiles.py":
        "its only effect on a record is the literal workspace_bounds_m that CONFIGURATION_FIELDS "
        "already compares field for field",
    "ora/contracts/motion_speed.py":
        "the end-effector speed request contracts/action.py validates only for a move_pose that carries "
        "speed; no RoboDojo surface declares end_effector_speed_v1, so no RoboDojo action carries one, "
        "and the RoboDojo worker reads none",
}

_OTHER_BACKEND = "another backend's branch of worker._build_session"
_HOST_ONLY = "host-side factory code for another backend's builder; the RoboDojo builder never calls it"
_HOST_PASS_THROUGH = (
    "host-side pass-through layer: build_environment_session calls resolve_descriptor and wraps the "
    "RPC session in DescribedEnvironmentSession, while the qualification probes build the worker "
    "through build_robodojo_worker_session and bypass it"
)

#: Imports the three shared dispatchers make for another backend or for the
#: host side. ``worker._build_session`` selects one backend per process and
#: ``factory._robodojo`` builds only the RPC session; these edges are pruned,
#: and a new dispatcher import is a gap until reviewed here.
DISPATCH_PRUNED_EDGES: Mapping[tuple[str, str], str] = {
    ("ora/environment/worker.py", "ora/environment/backends/libero_pro/session.py"): _OTHER_BACKEND,
    # The LIBERO render guard, inside worker.py's libero_pro branch.
    ("ora/environment/worker.py", "ora/environment/backends/libero_pro/render_guard.py"): _OTHER_BACKEND,
    ("ora/environment/worker.py", "ora/environment/backends/robotwin/session.py"): _OTHER_BACKEND,
    ("ora/environment/worker.py", "ora/environment/backends/robocasa/session.py"): _OTHER_BACKEND,
    ("ora/environment/worker.py", "ora/environment/backends/robolab/session.py"): _OTHER_BACKEND,
    ("ora/environment/worker.py", "ora/environment/backends/yam/session.py"): _OTHER_BACKEND,
    ("ora/environment/worker.py", "ora/environment/synthetic.py"): _OTHER_BACKEND,
    # The free-play branch (backend robodojo_play): ORA's own play scenes run on the benchmark
    # kernel plus the play modules, whose identity is the play kernel (robodojo_play_kernel),
    # held to its own closure by play_kernel_gaps.
    ("ora/environment/worker.py", "ora/environment/backends/robodojo/play_scenes/session.py"): (
        "the free-play branch of worker._build_session (backend robodojo_play); bound by the play kernel"),
    ("ora/environment/factory.py", "ora/environment/controllers/embodiment_control.py"): _HOST_ONLY,
    ("ora/environment/factory.py", "ora/environment/controllers/hybrid_motion.py"): _HOST_ONLY,
    ("ora/environment/factory.py", "ora/environment/definitions.py"): _HOST_PASS_THROUGH,
    ("ora/environment/factory.py", "ora/environment/session.py"): _HOST_PASS_THROUGH,
    ("ora/environment/factory.py", "ora/environment/synthetic.py"): _HOST_ONLY,
    ("ora/environment/factory.py", "ora/environment/benchmarks/yam.py"): _HOST_ONLY,
}


def worker_import_closure(
    project_root: Path | None = None, *, manifest: Sequence[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Every module the worker reaches through listed or excluded paths, with its imports."""

    listed = set(WORKER_KERNEL_PATHS if manifest is None else manifest)
    root = _PROJECT_ROOT if project_root is None else project_root

    def follow(importer: str, imported: str) -> bool:
        return ((imported in listed or imported in KERNEL_REVIEWED_EXCLUSIONS)
                and (importer, imported) not in DISPATCH_PRUNED_EDGES)

    return import_closure(ROBODOJO_WORKER_ENTRY_POINTS, read=tree_reader(root), follow=follow)


def kernel_manifest_gaps(
    project_root: Path | None = None, *, manifest: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Imports that are neither listed, excluded, nor a reviewed pruned edge.

    ``manifest`` defaults to ``WORKER_KERNEL_PATHS``. An empty result is the
    condition under which the kernel digest covers everything the worker runs.
    """

    listed = set(WORKER_KERNEL_PATHS if manifest is None else manifest)
    closure = worker_import_closure(project_root, manifest=manifest)
    return tuple(sorted({
        imported for importer, targets in closure.items() for imported in targets
        if imported not in listed and imported not in KERNEL_REVIEWED_EXCLUSIONS
        and (importer, imported) not in DISPATCH_PRUNED_EDGES
    }))


#: The one pruned dispatcher edge the play worker takes.
PLAY_DISPATCH_EDGE = ("ora/environment/worker.py", "ora/environment/backends/robodojo/play_scenes/session.py")
#: Edges only the play worker's closure has, pruned for the reason the benchmark prunes factory.py's.
PLAY_PRUNED_EDGES: Mapping[tuple[str, str], str] = {
    ("ora/environment/robodojo_play_factory.py", "ora/environment/definitions.py"): _HOST_PASS_THROUGH,
    ("ora/environment/robodojo_play_factory.py", "ora/environment/session.py"): _HOST_PASS_THROUGH,
}


def _play_pruned(importer: str, imported: str) -> bool:
    edge = (importer, imported)
    return (edge in DISPATCH_PRUNED_EDGES and edge != PLAY_DISPATCH_EDGE) or edge in PLAY_PRUNED_EDGES


def play_worker_import_closure(
    project_root: Path | None = None, *, manifest: Sequence[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Every module the free-play worker reaches: the benchmark closure's rules plus the play edge.

    ``manifest`` defaults to ``PLAY_KERNEL_PATHS``; the benchmark manifest,
    the reviewed exclusions and every other pruned dispatcher edge apply as
    they do to the benchmark worker.
    """
    from ora.orchestrator.robodojo_play_kernel import PLAY_KERNEL_PATHS, PLAY_WORKER_ENTRY_POINTS

    listed = {*WORKER_KERNEL_PATHS, *(PLAY_KERNEL_PATHS if manifest is None else manifest)}
    root = _PROJECT_ROOT if project_root is None else project_root

    def follow(importer: str, imported: str) -> bool:
        return (imported in listed or imported in KERNEL_REVIEWED_EXCLUSIONS) and not _play_pruned(importer, imported)

    return import_closure(PLAY_WORKER_ENTRY_POINTS, read=tree_reader(root), follow=follow)


def play_kernel_gaps(
    project_root: Path | None = None, *, manifest: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Imports of the free-play worker that neither kernel lists, excludes or prunes."""
    from ora.orchestrator.robodojo_play_kernel import PLAY_KERNEL_PATHS

    listed = {*WORKER_KERNEL_PATHS, *(PLAY_KERNEL_PATHS if manifest is None else manifest)}
    closure = play_worker_import_closure(project_root, manifest=manifest)
    return tuple(sorted({
        imported for importer, targets in closure.items() for imported in targets
        if imported not in listed and imported not in KERNEL_REVIEWED_EXCLUSIONS
        and not _play_pruned(importer, imported)
    }))


def play_kernel_review_drift(project_root: Path | None = None) -> tuple[str, ...]:
    """Play review entries the current source no longer justifies."""
    from ora.orchestrator.robodojo_play_kernel import PLAY_KERNEL_PATHS

    closure = play_worker_import_closure(project_root)
    edges = {(importer, target) for importer, targets in closure.items() for target in targets}
    drift = [f"play pruned edge absent: {edge[0]} -> {edge[1]}" for edge in (*PLAY_PRUNED_EDGES, PLAY_DISPATCH_EDGE)
             if edge not in edges]
    drift += [f"play path is also a benchmark kernel path: {path}" for path in PLAY_KERNEL_PATHS
              if path in WORKER_KERNEL_PATHS]
    # The play kernel is code only: a scene's index entry and YAML are in its own source binding.
    drift += [f"scene data in the code-only play kernel: {path}" for path in PLAY_KERNEL_PATHS
              if not path.endswith(".py")]
    # A package __init__ runs as the parent of its modules; the closure lists only named imports.
    drift += [f"play module outside the play closure: {path}" for path in PLAY_KERNEL_PATHS
              if path.endswith(".py") and not path.endswith("/__init__.py") and path not in closure]
    return tuple(sorted(drift))


def kernel_review_drift(project_root: Path | None = None) -> tuple[str, ...]:
    """Review-table entries the current source no longer justifies."""

    closure = worker_import_closure(project_root)
    imported = {target for targets in closure.values() for target in targets}
    edges = {(importer, target) for importer, targets in closure.items() for target in targets}
    drift = [f"exclusion never imported: {path}" for path in KERNEL_REVIEWED_EXCLUSIONS
             if path not in imported]
    drift += [f"exclusion is also listed: {path}" for path in KERNEL_REVIEWED_EXCLUSIONS
              if path in WORKER_KERNEL_PATHS]
    drift += [f"pruned edge absent: {edge[0]} -> {edge[1]}" for edge in DISPATCH_PRUNED_EDGES
              if edge not in edges]
    return tuple(sorted(drift))


__all__ = [
    "DISPATCH_PRUNED_EDGES", "KERNEL_REVIEWED_EXCLUSIONS", "PLAY_DISPATCH_EDGE", "PLAY_PRUNED_EDGES",
    "ROBODOJO_BACKEND_PREFIX", "ROBODOJO_WORKER_ENTRY_POINTS", "kernel_manifest_gaps", "kernel_review_drift",
    "play_kernel_gaps", "play_kernel_review_drift", "play_worker_import_closure", "worker_import_closure",
]
