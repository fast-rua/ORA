"""Append-only parent/publication index; parse each complete event line once."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path

from ora.agent.evidence.agent_events import library_op
from ora.agent.toolkit.interface.types import AccessDenied


class EventIndex:
    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.file_identity = None
        self.proposals = []
        self.by_id = {}
        self.by_name = {}
        self.publications = []
        self.publication_seqs = []
        self.done = set()
        self.bytes_read = 0
        self.lines_parsed = 0

    def refresh(self):
        with self.path.open('rb') as stream:
            import os
            stat = os.fstat(stream.fileno())
            identity = stat.st_dev, stat.st_ino
            if self.file_identity is not None and (identity != self.file_identity or stat.st_size < self.offset):
                raise AccessDenied('agent event file replaced or truncated')
            self.file_identity = identity
            stream.seek(self.offset)
            while raw := stream.readline():
                self.bytes_read += len(raw)
                if not raw.endswith(b'\n'):
                    break  # Keep the offset before an unfinished trailing line.
                event = json.loads(raw)
                kind = event['type']
                if kind == 'tool_call_proposed' and not library_op(event):  # a script's library request: no call
                    # Drop arguments and unrelated event payloads from the cache.
                    row = {key:event.get(key) for key in ('run_id','seq','tool_call_id','model_call_id')}
                    row['name'] = event['payload']['name']
                    row['call'] = len(self.proposals)
                    self.proposals.append(row)
                    self.by_id.setdefault(row['tool_call_id'], []).append(row)
                    self.by_name.setdefault(row['name'], deque()).append(row)
                elif kind == 'fresh_observation_published':
                    self.publications.append(event)
                    self.publication_seqs.append(event['seq'])
                elif kind == 'tool_result':
                    self.done.add(event.get('tool_call_id'))
                self.lines_parsed += 1
                self.offset = stream.tell()

    def select(self, name, supplied, completed):
        self.refresh()
        tool_id = supplied.get('tool_call_id')
        if tool_id:
            rows = self.by_id.get(tool_id, [])
            if len(rows) != 1:
                raise AccessDenied('recording requires an unambiguous recorded parent proposal')
        else:
            rows = self.by_name.get(name, deque())
            while rows and (rows[0]['tool_call_id'] in completed or rows[0]['tool_call_id'] in self.done):
                rows.popleft()
        for row in rows:
            if row['name'] == name and row['tool_call_id'] not in completed and row['tool_call_id'] not in self.done:
                return row
        raise AccessDenied('recording requires an unfinished recorded parent proposal')
