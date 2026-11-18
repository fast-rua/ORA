"""Resolve immutable public observations, never the mutable sandbox input mount."""

import json
import re

from ora.agent.toolkit.interface.types import AccessDenied


def published_manifest(store, identity):
    """Verify the control/state identity against its already published manifest.

    Publication precedes the policy turn. Missing or mismatched evidence is an
    integrity failure; neither the sandbox copy nor another observation is a
    fallback. BlobStore also rejects symlinks and paths outside its allowlist.
    """
    observation_id = identity["observation_id"]
    if not isinstance(observation_id, str) or not re.fullmatch(
        r"[^/.][^/\\]*", observation_id
    ):
        raise AccessDenied("invalid published observation identity")
    relative = f"policy-observations/{observation_id}/observation.json"
    ref = store.reference(relative, media_type="application/json")
    if ref["sha256"] != identity["observation_sha256"]:
        raise AccessDenied("published manifest hash drift")
    manifest = json.loads(store.read(ref))
    if manifest.get("observation_id") != observation_id:
        raise AccessDenied("published manifest observation identity differs")
    return (store.root / relative).parent, manifest, ref
