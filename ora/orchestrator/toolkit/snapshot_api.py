"""Compose verified completed snapshots without a simulator or legacy replay fallback."""

from ora.agent.toolkit.interface.assemble import assemble_snapshot
from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.journal import Journal
from ora.agent.toolkit.interface.runtime import Runtime, envelope, request_wire
from ora.agent.toolkit.interface.snapshot_services import SnapshotServices
from ora.agent.toolkit.interface.tool_data import ToolData
from ora.agent.toolkit.interface.types import AccessDenied, NotRecorded, PublicData, canonical
from ora.agent.toolkit.host.numeric import evaluate


class Publications:
    """Only verified same-round publications at or before the selected boundary."""

    def __init__(self, boundary, backend):
        self.backend = backend
        self.current = boundary.context.json()["state"]["observation"]
        self.grants = {}
        for loaded in boundary.publications:
            values = loaded.access(reader="rsi").values()
            observed = values["env.observation"]
            store = (backend.store if loaded.root == boundary.source
                     else backend.evaluation.store if loaded.root == backend.evaluation.store.root else None)
            if store is None:
                raise AccessDenied("snapshot publication belongs to an ungranted root")
            self.grants[observed["observation_id"]] = (values, store)

    def read(self, method, observation):
        identity = self.current if observation is None else observation
        grant = self.grants.get(identity.get("observation_id"))
        values, store = grant if grant is not None else (None, None)
        if values is None or any(values["env.observation"][key] != identity.get(key)
                                 for key in ("observation_id", "observation_sha256")):
            raise NotRecorded("未记录: observation is not published at this snapshot")
        observed = values["env.observation"]
        refs = [observed["manifest"]]
        if "proprioception_file" in observed:
            refs.append(observed["proprioception_file"])
        for key, camera in values.items():
            if key.startswith("env.camera."):
                refs.extend(camera["files"].values())
        registered = []
        for ref in refs:
            suffix = ref["path"].rsplit(".", 1)[-1]
            media = {"png": "image/png", "npy": "application/x-npy"}.get(suffix, "application/json")
            registered.append(store.register({**ref, "media_type": media}))
        manifest, files = registered[0], registered[1:]
        if method == "observation":
            data = dict(observation=identity, manifest=manifest, files=files)
        elif method == "proprioception":
            found = [ref for ref in files if ref["path"].endswith("/proprioception.json")]
            if len(found) != 1:
                raise NotRecorded("未记录: snapshot has no published proprioception")
            data = dict(proprioception=found[0])
        else:
            raise ValueError("unsupported snapshot publication method")
        request = request_wire(method, {"observation": identity}, {},
                               context=self.backend.context, bindings=self.backend.bindings)
        return envelope(method, request, public=PublicData.from_json(data, sources=[manifest]))


def build(boundary, output, *, services, bindings, configuration=None, inputs=None, history=()):
    """Return classified RSI capabilities and their private evidence transport.

    Caller-supplied inputs must be frozen public grants for this boundary. This
    function never scans eventual artifact stores or imports a simulator.
    """
    output = output.resolve()
    if output.is_relative_to(boundary.source) or boundary.source.is_relative_to(output):
        raise AccessDenied("snapshot output and source must be disjoint")
    evaluation = Runtime(context=boundary.context, store=BlobStore(output, writable=True),
                         journal=Journal(output), adapters=services, bindings=bindings)
    backend = SnapshotServices(context=boundary.context, store=BlobStore(boundary.source),
                               evaluation=evaluation, history=history)
    for item in (inputs or {}).values():
        for ref in item.get("files", {}).values():
            backend.store.register(ref)
    backend.tool_data = (None if configuration is None else
                         ToolData(store=evaluation.store, configuration=configuration, inputs=inputs, replay=True))
    backend.numeric_evaluator = evaluate
    backend.snapshot_publication = Publications(boundary, backend).read

    def record(method, parameters):
        request = request_wire(method, parameters, {}, context=boundary.context, bindings=bindings)
        value = dict(schema="ora.toolkit.snapshot-control.v1", method=method,
                     parameters=parameters, boundary=boundary.context.json()["boundary"],
                     executed=False)
        with (output / "episode-controls.jsonl").open("ab") as stream:
            stream.write(canonical(value) + b"\n")
        return envelope(method, request, public=PublicData.from_json(value))

    def declare(attachment):
        # Persist declarations independently; no live context owner is connected.
        data = canonical(attachment)
        with (output / "context-attachments.jsonl").open("ab") as stream:
            stream.write(data + b"\n")
        return dict(recorded=True, delivered=False)

    api = assemble_snapshot(backend=backend, state=boundary.state.access(reader="rsi", writer="tool"),
                            context=declare, control_recorder=record)
    return api, backend
