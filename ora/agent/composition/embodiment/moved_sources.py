"""Repository paths a recorded binding names that are answered at another path now.

A RunPlan binds a descriptor as ``{path, sha256}``, a descriptor names its
repository provenance sources by path only, and an observation records the
sha256 of the descriptor its loaded geometry was verified against. A run sealed
before a file moved, or before its descriptor got a source-only successor,
still names the old path. ``MOVED`` is the one table that maps such a path,
with the sha256 of its bytes, to the path that answers for it now:

- :func:`resolve` reads a recorded path. When the file is not there it takes
  the new path only while the bytes there hash to the entry's sha256 (and to
  the binding's, when it records one): the same bytes, moved.
- :func:`succeeds` accepts evidence verified against a recorded descriptor for
  the descriptor at the new path, only while the recorded one still resolves
  to the bytes it recorded and both bind the same robot geometry.

Anything else is refused exactly as before.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

#: (recorded repository path, sha256 of its bytes) -> the repository path that answers for it now.
MOVED: dict[tuple[str, str], str] = {
    # The gripper module the v5 descriptor names as its reviewed nominal gripper parameters; same bytes.
    ("ora/agent/tools/grasp/arx_x5_gripper.py",
     "574d9742caa734321e7c47cc6e40e46f09172d35d66e403bd8d0f8fbd0d87ce2"): "ora/agent/toolkit/host/grasp/arx_x5_gripper.py",
    # v6 is v5 with that gripper module's provenance path updated: the same robot geometry and card.
    ("ora/policy/embodiments/arx-x5/descriptor-public-geometry-v5.json",
     "a463bca69953466303192f1d787fc706ecc2753986cdf95e29f305896fbe583f"):
        "ora/policy/embodiments/arx-x5/descriptor-public-geometry-v6.json",
    # The flat-surface successor cites the written-out LIBERO surface where this one cited the retired
    # explicit_motion_v1 definitions, which left the tree: the same robot geometry and card.
    ("ora/policy/embodiments/franka-panda/"
     "descriptor-public-geometry-direct-v1-controller-feedback-v1-session-split-v1.json",
     "17453ef570167721269d3ea3c9d7b77484c4201cfe859d5e940b5beb516f25f9"):
        "ora/policy/embodiments/franka-panda/"
        "descriptor-public-geometry-direct-v1-controller-feedback-v1-session-split-v1-flat-surface-v1.json",
    # PandaOmron v2 adds card facts (the arm mount, its reach admission, the wrist camera's mount) to v1: the same
    # robot geometry, so an observation verified against v1 holds for v2.
    ("ora/policy/embodiments/panda-omron/descriptor-public-geometry-v1.json",
     "82892720cdfb31d113e4cd6781fe107ed36fbb4e7cad83c3a6bfd6f977e0695e"):
        "ora/policy/embodiments/panda-omron/descriptor-public-geometry-v2.json",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(root: Path, path: str, sha256: str | None = None) -> Path | None:
    """The file a recorded repository ``path`` names, or None.

    The file at ``path`` when it is there (the caller checks its bytes, as
    before). Otherwise its new place from ``MOVED``, only when the bytes there
    hash to the sha256 the entry records and, for a binding that records one,
    to ``sha256`` too.
    """

    root = Path(root).resolve()
    here = (root / path).resolve()
    if here.is_relative_to(root) and here.is_file():
        return here
    for (old, moved_sha256), new in MOVED.items():
        if old != path or (sha256 is not None and sha256 != moved_sha256):
            continue
        there = (root / new).resolve()
        if there.is_relative_to(root) and there.is_file() and _digest(there) == moved_sha256:
            return there
    return None


def succeeds(root: Path, recorded_sha256: Any, current: Mapping[str, Any]) -> bool:
    """Whether evidence verified against the descriptor ``recorded_sha256`` holds for the ``current`` binding.

    ``current`` is a resolved ``{path, sha256}`` descriptor binding. True only
    when ``MOVED`` maps a recorded descriptor with that sha256 to
    ``current["path"]``, the recorded descriptor still resolves to those bytes,
    and both descriptors bind the same public robot geometry.
    """

    from .descriptor import EmbodimentDescriptorError
    from .geometry import load_public_geometry_binding

    for (old, old_sha256), new in MOVED.items():
        if old_sha256 != recorded_sha256 or new != current.get("path"):
            continue
        try:  # each binding re-reads its descriptor and checks its bytes
            then = load_public_geometry_binding({"path": old, "sha256": old_sha256}, project_root=Path(root))
            now = load_public_geometry_binding(dict(current), project_root=Path(root))
        except EmbodimentDescriptorError:
            return False
        return then["geometry"] == now["geometry"]
    return False


__all__ = ["MOVED", "resolve", "succeeds"]
