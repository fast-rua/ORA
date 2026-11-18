"""Default-off recording around existing calls, without changing tool dispatch.

Only trusted composition installs this on a ManipulationToolRuntime. The
original return objects, exception objects and client call counts are preserved.
No worker, action executor or new execution capability is installed.
"""
from __future__ import annotations

from bisect import bisect_left
from contextlib import ExitStack
from contextvars import ContextVar
from threading import Lock
from dataclasses import replace
import json

from ora.agent.model_services.clients import SAM3_MODEL_IDENTITY, CONTACT_GRASPNET_MODEL_IDENTITY
from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.journal import Journal
from ora.agent.toolkit.interface.gaps import write_gap
from ora.agent.toolkit.interface.projections import alignment, control_result, kinematics, motion_probe, reach
from ora.agent.toolkit.interface.runtime import Runtime
from ora.agent.toolkit.interface.services import project_grasp, project_sam3
from ora.agent.toolkit.interface.types import AccessDenied, CallContext, PublicData
from ora.contracts.kinematics import REQUEST_SCHEMA as KINEMATICS_REQUEST
from ora.contracts.reach_check import REQUEST_SCHEMA as REACH_REQUEST

from .event_index import EventIndex
from .publication import published_manifest


def journal_for(runtime):
    """One writer per run runtime, shared by migrated and legacy tool calls.

    Keep a failed writer too: reopening it could hide an uncertain append.
    Parent calls are serialized by the existing tool RPC; this is not a new
    concurrent dispatch or multi-process journal protocol.
    """
    if not hasattr(runtime, "_toolkit_io_journal"):
        runtime._toolkit_io_journal = Journal(runtime.control_root.parent)
    return runtime._toolkit_io_journal


def _binding(client, model):
    config = dict(client.configuration()) if hasattr(client, 'configuration') else {}
    config.pop('endpoint', None)
    return dict(backend=client.backend_id, service_identity=getattr(client, 'expected_binding', None),
                model_identity=dict(model), configuration_identity=config)


def _publication(value, store, *, copy=False):
    control, root, manifest = value
    observed = {key: control[key] for key in ('observation_id', 'observation_sha256')}
    root, manifest, _ = published_manifest(store, observed)
    def reference(path, mime):
        # Historical public publications are copied into the CAS; the current
        # publication remains a hash-checked reference for snapshot registration.
        registered = store.reference(path.relative_to(store.root).as_posix(), media_type=mime)
        if copy:
            return store.put(store.read(registered), media_type=mime)
        return registered
    ref = reference(root / 'observation.json', 'application/json')
    if ref['sha256'] != observed['observation_sha256']:
        raise AccessDenied('published manifest hash drift')
    paths = [(root / 'proprioception.json', None)]
    for camera in manifest.get('cameras', {}).values():
        paths.extend((root / item['path'], item) for item in camera.get('files', {}).values())
    refs = []
    for path, declared in paths:
        mime = 'image/png' if path.suffix == '.png' else 'application/x-npy' if path.suffix == '.npy' else 'application/json'
        item = reference(path, mime)
        if declared is not None and any(item[key] != declared[key] for key in ('sha256', 'bytes')):
            raise AccessDenied('published camera artifact hash drift')
        refs.append(item)
    return PublicData.from_json(dict(observation=observed, manifest=ref, files=refs), sources=[ref])


class _ServiceProxy:
    def __init__(self, client, recorder, methods):
        self.client, self.recorder, self.methods = client, recorder, methods

    def __getattr__(self, name):
        original = getattr(self.client, name)
        if name not in self.methods:
            return original

        def call(**kwargs):
            if not self.recorder.recording.get():
                return original(**kwargs)
            method = 'grasp_candidates' if name == 'propose' else name
            try:
                store = self.recorder.interface.store
                if name == 'propose':
                    import numpy as np
                    method = 'grasp_candidates'
                    # The existing service casts to contiguous float32 before hashing and sending.
                    inputs = {role: store.put_array(np.ascontiguousarray(kwargs[key], dtype=np.float32))
                              for role, key in (('scene', 'scene_points_camera_m'), ('target', 'target_points_camera_m'))}
                    params = dict(camera_to_world=np.asarray(kwargs['camera_to_world']).tolist(),
                                  workspace_bounds_m=kwargs['workspace_bounds_m'],
                                  max_candidates=self.client.raw_candidate_limit, forward_passes=self.client.forward_passes)
                    project = project_grasp
                else:
                    method = name
                    inputs = {'rgb': store.put_array(kwargs['rgb_u8'])}
                    if name != 'ground':
                        inputs['depth'] = store.put_array(kwargs['depth_m'])
                    params = {key: value for key, value in kwargs.items() if key not in ('rgb_u8','depth_m')}
                    if name == 'segment':
                        params['point_uv'] = list(params['point_uv'])
                    project = lambda value, target: project_sam3(method, value, target)
            except Exception as error:
                self.recorder.gap(method, 'store', error)
                return original(**kwargs)
            return self.recorder.capture(method, params, inputs, lambda: original(**kwargs), project)
        return call


class Recorder:
    """One session's recorder; interface state is reset at every actual parent call."""
    def __init__(self, runtime):
        self.host = runtime
        self.root = runtime.control_root.parent
        self.store = BlobStore(self.root, writable=True)
        self.journal = journal_for(runtime)
        self.completed = {e['identity']['tool_call_id'] for e in self.journal.events}
        self.interface = None
        self.depth = 0
        self.original_invoke = runtime.invoke
        self.index = EventIndex(self.root / 'agent-events.jsonl')
        self.lock = Lock()
        self.recording = ContextVar('toolkit_recording', default=False)
        self.parent_id = None

    def context(self, name, supplied):
        event = self.index.select(name, supplied, self.completed)
        self.parent_id = event['tool_call_id']
        index = event['call']
        self.publication_limit = bisect_left(self.index.publication_seqs, event['seq'])
        state = {}
        round_index = 0
        if self.publication_limit:
            publication = self.index.publications[self.publication_limit - 1]
            oid = publication['payload']['observation_id']
            manifest_path = self.root / 'policy-observations' / oid / 'observation.json'
            ref = self.store.reference(manifest_path.relative_to(self.root).as_posix(), media_type='application/json')
            manifest = json.loads(self.store.read(ref))
            round_index = manifest['round_index']
            state = dict(observation=dict(observation_id=oid, observation_sha256=ref['sha256']), published_seq=publication['seq'])
            for key in ('world_mutation_sequence','action_sequence','physics_clock'):
                if key in manifest:
                    state[key] = manifest[key]
        return CallContext.create(run_id=event['run_id'], round_index=round_index,
                                  tool_call_id=event['tool_call_id'],
                                  boundary=dict(kind='before_call',call=index,seq=event['seq'],
                                                tool_call_id=event['tool_call_id'],model_call_id=event.get('model_call_id')),
                                  state=state, check_task='check_task' in self.host.allowed_tools)

    def disable(self):
        self.recording.set(False)

    def gap(self, method, stage, error, *, tool_call_id=None, disable=True):
        if disable:
            self.disable()
        write_gap(self.root, tool_call_id=self.parent_id if tool_call_id is None else tool_call_id,
                  method=method, stage=stage, error=error)

    def capture(self, method, params, inputs, call, project):
        if not self.recording.get() or self.depth:
            return call()  # Nested public callbacks remain atomic, bypass calls do no recording work.
        self.depth += 1
        try:
            return self.interface.capture(method, params, inputs, call=call, project=project, on_gap=self.disable)
        finally:
            self.depth -= 1

    def bypass(self, name, arguments):
        token = self.recording.set(False)
        try:
            return self.original_invoke(name, arguments)
        finally:
            self.recording.reset(token)

    def invoke(self, name, arguments, *, call_context=None):
        if hasattr(self.host, '_toolkit_tools_enabled') and self.host._toolkit_tools_enabled:
            from ora.agent.composition.surface.tool_names import canonical as canonical_tool
            if canonical_tool(name) in self.host._toolkit_installed_tools:
                return self.bypass(name, arguments)  # New handlers record their own interface once.
        supplied = call_context or {}
        if not self.lock.acquire(blocking=False):
            # This diagnostic belongs to the bypassed call, not the active owner.
            write_gap(self.root, tool_call_id=supplied.get('tool_call_id'), method=name, stage='context',
                      error=AccessDenied('concurrent or recursive invocation bypassed recording'))
            return self.bypass(name, arguments)
        stack = ExitStack()
        token = self.recording.set(False)
        called, original_error, traceback = False, None, None
        self.parent_id = supplied.get('tool_call_id')
        try:
            try:
                context = self.context(name, supplied)
                self.prepare(context, stack)
                self.recording.set(True)
                called = True
                try:
                    result = self.original_invoke(name, arguments)
                except BaseException as error:
                    original_error, traceback = error, error.__traceback__
            except Exception as error:
                self.gap(name, 'context', error)
            finally:
                try:
                    stack.close()
                except Exception as error:
                    self.gap(name, 'context', error)
                if self.parent_id is not None:
                    self.completed.add(self.parent_id)
                self.interface = None
            if not called:
                return self.bypass(name, arguments)
            if original_error is not None:
                raise original_error.with_traceback(traceback)
            return result
        finally:
            self.recording.reset(token)
            self.lock.release()

    def prepare(self, context, stack):
        bindings = {}
        host = self.host
        grasp = host.grasp_backend
        learned = getattr(grasp, 'contact_backend', grasp)
        for client, methods, model in (
            (host.segmentation_backend, ('segment','segment_points'), SAM3_MODEL_IDENTITY),
            (host.grounding_backend, ('ground',), SAM3_MODEL_IDENTITY),
            (learned, ('grasp_candidates',), CONTACT_GRASPNET_MODEL_IDENTITY),
        ):
            if client is not None:
                for method in methods:
                    bindings[method] = _binding(client, model)
        self.interface = Runtime(context=context, store=self.store, journal=self.journal, adapters={}, bindings=bindings)
        def patch(name, value):
            owned, previous = name in vars(host), getattr(host, name)
            setattr(host, name, value)
            def restore():
                if owned:
                    setattr(host, name, previous)
                else:
                    delattr(host, name)
            stack.callback(restore)
            return previous

        current = host._current
        patch('_current', lambda: self.capture('observation', {'observation':None}, {}, current, _publication))
        select = host._select_observation
        def select_observation(step):
            if not self.recording.get():
                return select(step)
            try:
                observed = context.json()['state'].get('observation')
                if step is not None:
                    matching = [p for p in self.index.publications[:self.publication_limit] if p['payload'].get('state_ref',{}).get('step') == step]
                    if not matching:
                        raise AccessDenied('historical observation lacks a publication at this boundary')
                    oid = matching[-1]['payload']['observation_id']
                    ref = self.store.reference(f'policy-observations/{oid}/observation.json', media_type='application/json')
                    observed = dict(observation_id=oid, observation_sha256=ref['sha256'])
            except Exception as error:
                self.gap('observation', 'begin', error)
                return select(step)
            def project(value, store):
                return _publication((observed, value[2], value[3]), store, copy=step is not None)
            return self.capture('observation', {'observation':observed}, {}, lambda: select(step), project)
        patch('_select_observation', select_observation)
        for attr, method, param in (('_check_task','check_task',None),
                                    ('_report_issue','report_issue','issue'),('_finish','finish_episode','request')):
            original = getattr(host, attr)
            def control(args, original=original, method=method, param=param):
                return self.capture(method, {} if param is None else {param:args}, {},
                                    lambda: original(args), lambda value, store: control_result(value))
            patch(attr, control)
        persist = host._persist_action
        patch('_persist_action', lambda target, control, action: self.capture(
            'submit_action', {'action':action}, {}, lambda: persist(target, control, action),
            lambda value, store: control_result(value)))
        for attr in ('motion_probe_callback','align_plan_callback'):
            original = getattr(host, attr)
            if original is None:
                continue
            def callback(request, original=original, attr=attr):
                method, projector = ('probe_motion', motion_probe) if attr == 'motion_probe_callback' else (
                    ('reach_check', reach) if request.get('schema') == REACH_REQUEST
                    else ('kinematics', kinematics) if request.get('schema') == KINEMATICS_REQUEST
                    else ('align_plan', alignment))
                return self.capture(method, {'request':request}, {}, lambda: original(request),
                                    lambda value, store: projector(value))
            patch(attr, callback)
        for attr, methods in (('segmentation_backend',('segment','segment_points')),
                               ('grounding_backend',('ground',))):
            client = getattr(host, attr)
            if client is not None:
                patch(attr, _ServiceProxy(client, self, methods))
        if learned is not None and hasattr(learned, 'raw_candidate_limit'):
            proxy = _ServiceProxy(learned, self, ('propose',))
            patch('grasp_backend', replace(grasp, contact_backend=proxy) if learned is not grasp else proxy)


def install(runtime):
    """Called only for runtime.record_toolkit_io=true; false installs nothing."""
    try:
        recorder = Recorder(runtime)
    except Exception as error:
        write_gap(runtime.control_root.parent, tool_call_id=None, method=None, stage='context', error=error)
        return runtime
    runtime.invoke = recorder.invoke
    runtime.invoke_with_context = recorder.invoke
    return runtime
