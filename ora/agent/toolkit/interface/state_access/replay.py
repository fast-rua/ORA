"""Snapshot state reads use only exported keys and verified original file bytes."""
from dataclasses import dataclass
from ..types import CallContext, NotRecorded, PublicData
from ..runtime import envelope, request_wire
from .live import State as LiveState


@dataclass(frozen=True, slots=True)
class State(LiveState):
    def _response(self, method, parameters, data, sources=()):
        observed = self.read('env.observation')
        identity = {key: observed[key] for key in ('observation_id', 'observation_sha256')}
        context = CallContext.create(**dict(self._backend.context.json(), state={'observation': identity}))
        request = request_wire(method, parameters, {}, context=context, bindings=self._backend.bindings)
        return envelope(method, request, public=PublicData.from_json(data, sources=list(sources)))

    def _ref(self, ref):
        # The caller's source store was granted exactly the snapshot's registry.
        store = getattr(self._backend, 'store')
        media = ('image/png' if ref['path'].endswith('.png') else
                 'application/x-npy' if ref['path'].endswith('.npy') else 'application/json')
        return store.register({**ref, 'media_type': media})

    def observation(self, *, observation: dict | None = None):
        reader = getattr(self._backend, "snapshot_publication", None)
        if reader is not None:
            return reader("observation", observation)
        value = self.read('env.observation')
        identity = {key: value[key] for key in ('observation_id', 'observation_sha256')}
        if observation is not None and observation != identity:
            raise NotRecorded('未记录: another publication is not this snapshot')
        manifest = self._ref(value['manifest'])
        files = []
        if 'proprioception_file' in value:
            files.append(self._ref(value['proprioception_file']))
        for key, camera in self._access.values().items():
            if key.startswith('env.camera.'):
                files.extend(self._ref(ref) for ref in camera['files'].values())
        return self._response('observation', {'observation': identity},
                              dict(observation=identity, manifest=manifest, files=files), [manifest])

    def proprioception(self, *, observation: dict | None = None):
        reader = getattr(self._backend, "snapshot_publication", None)
        if reader is not None:
            return reader("proprioception", observation)
        self.observation(observation=observation)
        value = self.read('env.observation')
        if 'proprioception_file' not in value:
            raise NotRecorded('未记录: snapshot has no published proprioception')
        ref = self._ref(value['proprioception_file'])
        return self._response('proprioception', {'observation': observation}, {'proprioception': ref}, [ref])

    def receipt(self, *, action_id: str):
        value = self.read('exec.last_receipt')
        if not isinstance(value, dict) or value.get('action_id') != action_id:
            raise NotRecorded('未记录: receipt was not published at this snapshot')
        return self._response('receipt', {'action_id': action_id}, value)

    def check_task(self):
        from ...state.keys import StateError, StateAccessDenied
        try:
            value = self._access.read("privileged.task_check")
        except StateAccessDenied:
            raise
        except StateError as error:
            raise NotRecorded("未记录: snapshot has no task check") from error
        if not isinstance(value, dict):
            raise NotRecorded('未记录: no official task check at this snapshot')
        verdict = {k: value[k] for k in ('terminal', 'task_success') if k in value}
        return self._response('check_task', {}, dict(success=True,
                              task_success=bool(value.get('task_success', False)), verdict=verdict))
