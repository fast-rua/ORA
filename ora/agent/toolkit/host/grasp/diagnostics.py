# Copyright contributors to the Robot Harness Runtime project.
#
# The policy-visible diagnostics of a multi-family grasp backend.
#
# A single-family backend returns the service's own diagnostics and the shared
# materializer projects them. A backend that runs two families has to decide
# what one set of diagnostics means, and the first attempt decided it by
# building a fresh dictionary: the request identity and the rejection ledger
# ended up nested under ``learned`` and three keys appeared that the result
# contract forbade. Every RoboTwin ``grasp_plan`` call then failed its own
# schema with a non-retryable error, after the backend had already compiled
# candidates -- the geometry was found and thrown away at the format gate.
#
# This module holds that decision in one place so the next family added to a
# proposer has somewhere to be added, and so the assembly can be tested
# against the resolved result schema without standing a backend up.
"""Compose one grasp backend's diagnostics from its per-family parts."""

from __future__ import annotations

from ora.agent.toolkit.tools.grasp_plan._diagnostics import hybrid_grasp_diagnostics


__all__ = ["hybrid_grasp_diagnostics"]
