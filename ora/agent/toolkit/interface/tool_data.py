"""Granted tool inputs and derived artifacts; no caller-selected filesystem paths.

Like read_blob/save_array these storage operations do not advance simulator
replay. Trusted assembly supplies a boundary-frozen input table and may install
a publisher for live compatibility artifacts. Replay installs no publisher.
"""
from __future__ import annotations

import re
from typing import Callable

from .blobs import BlobStore
from .types import AccessDenied, CorruptRecord, NotRecorded, PublicToolFailure, canonical, decode, digest

KINDS = {
    "segmentation_query": (r"segment-query-[0-9a-f]{24}", frozenset({
        "manifest.json", "raw-masks.npz", "overview.png",
        *(f"candidate-T{i}.png" for i in range(5))})),
    "observation_overlay": (r"overlay-[0-9a-f]{24}", frozenset({"manifest.json", "image.png"})),
    "skill_preparation": (r"skill-[0-9a-f]{24}", frozenset({"manifest.json", "ledger.json", "commitment.json"})),
    "grasp_query": (r"grasp-query-[0-9a-f]{24}", frozenset({"manifest.json", "overlay.png", "input-pointcloud.npz", "review-pointcloud.npz", "global-review.png", "backend-output.npz"})),
    "target": (r"target-[0-9a-f]{24}", frozenset({"manifest.json", "mask.npy", "overlay.png"})),
    "grounding_query": (r"grounding-query-[0-9a-f]{24}", frozenset({"manifest.json", "overlay.png", "candidate-T0.png", "candidate-T1.png", "candidate-T2.png"})),
}


def artifact_key(kind: str, key: str) -> str:
    if kind not in KINDS or not isinstance(key, str) or re.fullmatch(KINDS[kind][0], key) is None:
        raise AccessDenied("invalid public tool artifact identity")
    return f"{kind}:{key}"


def filename(kind: str, name: str) -> None:
    allowed = KINDS[kind][1]
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+\.(json|png|npy|npz)", name) is None:
        raise AccessDenied("artifact member must be a single public filename")
    if allowed is not None and name not in allowed:
        raise AccessDenied("unsupported public tool artifact member")


def verify_artifact(kind: str, key: str, files: dict[str, bytes], metadata: dict) -> dict:
    """Verify manifest identity and every declared image/mask cross-reference."""
    artifact_key(kind, key)
    for name in files:
        filename(kind, name)
    try:
        manifest = decode(files['manifest.json'])
        identity = manifest['identity']
        if kind == 'observation_overlay':
            import json
            expected = "overlay-" + digest(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode())[:24]
            if (manifest['schema'] != 'ora.toolkit.observation-overlay.v1' or manifest['key'] != key
                    or expected != key or set(files) != {'manifest.json', 'image.png'}
                    or manifest['image_sha256'] != digest(files['image.png'])
                    or any(manifest[k] != metadata[k] for k in ('observation_id', 'camera'))):
                raise CorruptRecord('observation overlay identity drift')
            return manifest
        if kind == 'skill_preparation':
            if (manifest['schema'] != 'ora.toolkit.skill-preparation.v1' or manifest['key'] != key
                    or 'skill-' + digest(canonical(identity))[:24] != key
                    or metadata.get('observation_id') != identity['observation_id']
                    or set(files) != {'manifest.json', 'ledger.json', 'commitment.json'}
                    or digest(files['ledger.json']) != identity['ledger_sha256']
                    or digest(files['commitment.json']) != identity['commitment_sha256']):
                raise CorruptRecord('skill preparation identity drift')
            return manifest
        prefix, schema, reference = (
            ('target-', 'ora.segmented-target.v2', 'target_ref') if kind == 'target' else
            ('grounding-query-', 'ora.grounding-query.v2', 'grounding_ref') if kind == 'grounding_query' else
            ('segment-query-', 'ora.point-segmentation-batch.v2', 'query_ref') if kind == 'segmentation_query' else
            ('grasp-query-', 'ora.grasp-query.v7', 'query_ref'))
        if (manifest['schema'] != schema or manifest[reference] != key
                or prefix + digest(canonical(identity))[:24] != key):
            raise CorruptRecord('public tool artifact identity drift')
        if any(metadata.get(k) != identity.get(k) for k in ('observation_id', 'camera')):
            raise CorruptRecord('public tool artifact boundary metadata drift')
        declared = [] if kind == 'segmentation_query' else [manifest['overlay']]
        if kind == 'segmentation_query':
            declared.extend([manifest['raw_masks'], *manifest['images']])
        elif kind == 'target':
            declared.append(manifest['private_mask'])
        elif kind == 'grasp_query':
            declared.append(manifest['input_pointcloud']['artifact'])
            for role in ('review_visualization', 'review_pointcloud', 'backend_output'):
                value = manifest.get(role)
                if isinstance(value, dict) and 'artifact' in value:
                    declared.append(value['artifact'])
        else:
            visualization = manifest['visualization']
            if visualization['renderer'] != 'grounding-candidate-inspection.v2':
                raise CorruptRecord('grounding visualization identity drift')
            declared.extend(row['artifact'] for row in visualization['candidate_previews'])
            if visualization['overview'] != manifest['overlay']:
                raise CorruptRecord('grounding overview identity drift')
        names = []
        for ref in declared:
            name = ref['path']
            filename(kind, name)
            content = files[name]
            if ref['sha256'] != digest(content) or ref['bytes'] != len(content):
                raise CorruptRecord('public tool artifact member hash drift')
            names.append(name)
        if len(set(names)) != len(names) or set(files) != {'manifest.json', *names}:
            raise CorruptRecord('public tool artifact member set drift')
        return manifest
    except (KeyError, TypeError, AttributeError) as error:
        raise CorruptRecord('malformed public tool artifact') from error


class ToolData:
    """Own immutable input bytes and output refs; trusted publisher never reaches tools.

    Input artifacts must already be granted CAS/public-observation references.
    A lookup reveals only the selected boundary's table and this experiment's
    own results. Derived output cannot grant another source file or future input.
    """
    def __init__(self, *, store: BlobStore, configuration: dict, inputs: dict | None = None,
                 publisher: Callable | None = None, replay: bool = False):
        if replay and publisher is not None:
            raise AccessDenied('replay cannot publish live compatibility artifacts')
        self.store = store
        self.configuration = canonical(configuration)
        self.inputs = canonical(inputs or {})
        self.outputs: dict[str, bytes] = {}
        self.publisher = publisher
        self.replay = replay
        for identity, record in decode(self.inputs).items():
            kind, key = identity.split(":", 1)
            if artifact_key(kind, key) != identity:
                raise CorruptRecord("tool input artifact identity drift")
            if set(record) == {'error'}:
                PublicToolFailure(record['error'])  # Validate, but refuse only if this artifact is requested.
                continue
            for name, ref in record["files"].items():
                filename(kind, name)
                self.store.read(ref)
            self._verify(kind, key, record)

    def _verify(self, kind, key, record):
        if set(record) == {'error'}:
            raise PublicToolFailure(record['error'])
        verify_artifact(kind, key, {name: self.store.read(ref) for name, ref in record['files'].items()},
                        record['metadata'])
        return record

    def context(self) -> dict:
        return decode(self.configuration)

    def lookup(self, kind: str, key: str, *, cache: bool = False) -> dict | None:
        identity = artifact_key(kind, key)
        # External services are recomputed during replay, not bypassed by a
        # cached query result. Historical targets remain inputs to later tools.
        if self.replay and cache and kind in {"grounding_query", "grasp_query", "segmentation_query"}:
            return None
        if identity in self.outputs:
            return self._verify(kind, key, decode(self.outputs[identity]))
        record = decode(self.inputs).get(identity)
        return None if record is None else self._verify(kind, key, record)

    def publish(self, kind: str, key: str, files: dict[str, bytes], metadata: dict) -> dict:
        identity = artifact_key(kind, key)
        canonical(metadata)
        for name, content in files.items():
            filename(kind, name)
            if not isinstance(content, bytes):
                raise ValueError("public artifact contents must be bytes")
        if not files or "manifest.json" not in files:
            raise ValueError("artifact must contain a public manifest")
        verify_artifact(kind, key, files, metadata)
        refs = {}
        for name, content in files.items():
            media = "image/png" if name.endswith(".png") else "application/x-npy" if name.endswith(".npy") else "application/x-npz" if name.endswith(".npz") else "application/json"
            refs[name] = self.store.put(content, media_type=media)
        record = dict(files=refs, metadata=decode(canonical(metadata)))
        if self.publisher is not None:
            self.publisher(kind, key, dict(files), record["metadata"])
        self.outputs[identity] = canonical(record)
        return decode(self.outputs[identity])


class ToolDataMethods:
    """Storage-only methods shared by runtime and replay implementations."""
    tool_data: ToolData | None = None
    numeric_evaluator: Callable | None = None

    def evaluate_numeric(self, *, source: str, bindings: dict, outputs: list | None = None) -> dict:
        """Recompute bounded numeric code; no source lookup or simulator cursor step."""
        if self.numeric_evaluator is None:
            raise NotRecorded("bounded numeric evaluator is unavailable")
        return self.numeric_evaluator(source, bindings, outputs=outputs)


    def _tool_data(self) -> ToolData:
        if self.tool_data is None:
            raise NotRecorded("未记录: no boundary-bound tool inputs/configuration")
        return self.tool_data

    def tool_context(self) -> dict:
        return self._tool_data().context()

    def tool_artifact(self, *, kind: str, key: str, cache: bool = False) -> dict | None:
        return self._tool_data().lookup(kind, key, cache=cache)

    def publish_tool_artifact(self, *, kind: str, key: str, files: dict[str, bytes], metadata: dict) -> dict:
        return self._tool_data().publish(kind, key, files, metadata)
