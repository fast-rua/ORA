"""Host-side orchestration owned by one backend, reached through ``registry.py``.

``registry.BACKENDS`` maps each ``runtime.kind`` to its backend: ``synthetic.py``
and one subpackage per simulator, whose ``backend.py`` declares the hooks the
orchestrator core calls (runtime resolution, preflight, qualification, episode
and evaluation rules, play scenes, ``ora run``). The core imports no module
here except the registry. A subpackage runs in the Harness process: it may
import that backend's host-safe modules under ``ora.environment`` (for
example ``ora.environment.benchmarks.robocasa`` constants), but never starts
the simulator in-process: simulator code runs only in the isolated worker.
"""
