"""Test helpers for x5 topdown, shared by several test modules.

They were defined in ``tests/ora/agent/toolkit/host/grasp/test_x5_topdown.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations


class _NoLearned:
    backend_id = "contact_graspnet_test_double"
    backend_version = "v1"

    def configuration(self):
        return {"pregrasp_offset_m": 0.04}

    def propose(self, **kwargs):
        return [], {"request_id": "cgn-empty", "rejected": [], "model": {
            "family": "contact_graspnet", "source_commit": "x", "checkpoint_sha256": "y",
            "config_sha256": "0" * 64}}
