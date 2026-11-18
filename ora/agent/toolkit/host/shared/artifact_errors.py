"""Typed failures of target references and the artifact store.

The segmented-target loader, the grasp proposers and the SAM3 and
Contact-GraspNet clients used to raise a bare ``ValueError`` for every
failure, and each tool turned that into one code. The model-service failure
(``ModelServiceError``) lives with those clients, in
``agent/model_services/clients.py``. A malformed ``target_ref``,
a store whose index no longer matched its manifest and a model service that
had restarted under a new instance identity all reached the policy as the same
argument refusal, so a Harness fault blamed the
argument and a refusal the policy could fix looked like infrastructure.

Every class here still subclasses ``ValueError``: a caller that catches
``ValueError`` keeps working, and a caller that can tell them apart raises the
envelope that says who failed. ``agent/toolkit/host/shared/target_refusals.py`` and
``agent/toolkit/host/shared/service_refusals.py`` map them onto existing codes only; no new
code enters ``agent/error_classes.py``.
"""

from __future__ import annotations

from ora.agent.toolkit.tools.grasp_plan._topdown import TargetInsufficient


class TargetRefMalformed(ValueError):
    """The value is not a ``target-`` reference at all."""


class TargetRefUnknown(ValueError):
    """A well-formed reference that no segmented target in this run carries."""


class ArtifactIntegrityError(ValueError):
    """A Harness-written artifact is missing, malformed or outside its root."""

    code = "artifact_integrity_fault"


class ArtifactHashMismatch(ArtifactIntegrityError):
    """A Harness-written artifact no longer matches its recorded hash."""

    code = "artifact_hash_mismatch"


__all__ = [
    "ArtifactHashMismatch", "ArtifactIntegrityError", "TargetInsufficient",
    "TargetRefMalformed", "TargetRefUnknown",
]
