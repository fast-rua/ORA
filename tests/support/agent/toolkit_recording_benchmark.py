"""Offline recorder timing fixture; no simulator or service requests."""
from pathlib import Path
from statistics import median
from time import perf_counter

from ora.agent.toolkit.interface.types import canonical
from ora.agent.toolkit.host.event_index import EventIndex
from ora.agent.toolkit.host.recording import install
from tests.support.agent.grounding_grasp_runtime import _runtime
from tests.support.agent.toolkit_recording import publication, proposal


def measure(root: Path, *, calls=70, history_bytes=16*1024*1024):
    """Compare identical calls with off, incremental, and forced full-log parsing.

    Use a fresh root. Includes original check_task plus public manifest reads,
    envelope/blob writes and journal fsync, excludes installation and setup.
    """
    output = {}
    for mode in ('off', 'incremental', 'full_reread'):
        host, _ = _runtime(root/mode)
        publication(host)
        path = host.control_root.parent/'agent-events.jsonl'
        public = path.read_bytes()
        padding = canonical(dict(type='fixture_history', seq=0, payload='x'*history_bytes)) + b'\n'
        path.write_bytes(padding + public)
        for i in range(calls):
            proposal(host,'check_task',{},index=i)
        host._check_task = lambda args: {'success':True,'task_success':False}
        totals = {'bytes_read':0,'lines_parsed':0}
        if mode != 'off':
            install(host)
            recorder = host.invoke.__self__
            if mode == 'full_reread':
                context = recorder.context
                def reread(name, supplied):
                    recorder.index = EventIndex(path)
                    result = context(name, supplied)
                    totals['bytes_read'] += recorder.index.bytes_read
                    totals['lines_parsed'] += recorder.index.lines_parsed
                    return result
                recorder.context = reread
        elapsed = []
        for i in range(calls):
            start = perf_counter()
            assert host.invoke('check_task',{}) == {'success':True,'task_success':False}
            elapsed.append((perf_counter()-start)*1000)
        if mode == 'incremental':
            totals = dict(bytes_read=recorder.index.bytes_read, lines_parsed=recorder.index.lines_parsed)
        output[mode] = dict(calls=calls, file_bytes=path.stat().st_size,
            first_ms=elapsed[0], median_ms=median(elapsed),
            steady_median_ms=median(elapsed[1:]), p95_ms=sorted(elapsed)[int(.95*(calls-1))],
            mean_ms=sum(elapsed)/calls, total_ms=sum(elapsed), **totals)
    return output
