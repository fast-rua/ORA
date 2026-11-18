"""Best-effort diagnostics for recorder failures; never policy-visible errors."""
from __future__ import annotations

import os
from pathlib import Path
import time

from .blobs import IO_ROOT
from .types import NotRecorded, canonical, decode


def write_gap(root: Path, *, tool_call_id: str | None, method: str | None,
              stage: str, error: Exception, interface_ordinal: int | None = None) -> None:
    """A single append; even formatting/open/write/close failures are swallowed."""
    try:
        value = dict(schema='ora.toolkit.gap.v1', run_id=Path(root).name,
                     tool_call_id=tool_call_id, method=method, stage=stage,
                     interface_ordinal=interface_ordinal,
                     error_type=type(error).__name__, message=str(error)[:4096], unix_s=time.time())
        data = canonical(value) + b'\n'
        path = Path(root) / IO_ROOT / 'gaps.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, data)
        finally:
            os.close(descriptor)
    except Exception:
        pass


def require_no_gaps(root: Path, tool_call_id: str) -> None:
    """A recording gap must never become a claim of complete interface replay."""
    path = Path(root) / IO_ROOT / 'gaps.jsonl'
    try:
        if not path.exists():
            return
        for line in path.read_bytes().splitlines():
            gap = decode(line)
            if gap.get('schema') != 'ora.toolkit.gap.v1' or 'tool_call_id' not in gap:
                raise ValueError('invalid gap record')
            if gap['tool_call_id'] in (None, tool_call_id):
                raise NotRecorded('未记录: selected call has a recording gap')
    except NotRecorded:
        raise
    except Exception as error:
        raise NotRecorded('未记录: gap evidence is unreadable') from error
