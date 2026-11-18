"""Read-only file access for the run dashboard: caches keyed by (path, mtime, size) and incremental jsonl.

Ported from Shaoxiong Duan's code-vs-tools dashboard (snapshot 2026-10-31). Nothing here writes: every
parsed file is cached in memory until its stat signature changes, and an append-only jsonl file is parsed
only from the byte offset it was last read to, keeping small projections of its records.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable

_lock = threading.RLock()
_cache: dict[tuple[str, str], tuple[tuple[int, int], Any]] = {}
_CACHE_MAX = 20000


def stat_sig(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def cached_file(path: str, fn: Callable[[str], Any], kind: str) -> Any:
    """``fn(path)``, cached until the file's (mtime, size) changes; None when it is missing or unreadable."""
    sig = stat_sig(path)
    if sig is None:
        return None
    key = (path, kind)
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    try:
        val = fn(path)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    with _lock:
        if len(_cache) > _CACHE_MAX:
            for k in list(_cache)[: _CACHE_MAX // 2]:
                _cache.pop(k, None)
        _cache[key] = (sig, val)
    return val


def memo(key: tuple, sig: Any, build: Callable[[], Any]) -> Any:
    """``build()``, cached under ``key`` until ``sig`` (any comparable signature) changes."""
    with _lock:
        hit = _cache.get(key)  # type: ignore[arg-type]
        if hit and hit[0] == sig:
            return hit[1]
    val = build()
    with _lock:
        _cache[key] = (sig, val)  # type: ignore[index]
    return val


def load_json(path: str) -> Any:
    with open(path, "rb") as f:
        return json.loads(f.read())


def read_json(path: str) -> Any:
    return cached_file(path, load_json, "json")


def read_json_proj(path: str, proj: Callable[[Any], Any], kind: str | None = None) -> Any:
    return cached_file(path, lambda p: proj(load_json(p)), "proj:" + (kind or proj.__name__))


class IncJsonl:
    """Append-only jsonl reader: parses only bytes added since the last call and keeps projected records.
    Each projection gets (obj, byte_offset, byte_length), so a full record can be re-read on demand."""

    def __init__(self, path: str, proj: Callable[[dict, int, int], Any]):
        self.path, self.proj = path, proj
        self.lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self.offset, self.ino, self.records = 0, None, []

    def get(self) -> list:
        with self.lock:
            try:
                st = os.stat(self.path)
            except OSError:
                self._reset()
                return []
            if st.st_ino != self.ino or st.st_size < self.offset:
                self._reset()
                self.ino = st.st_ino
            if st.st_size > self.offset:
                try:
                    with open(self.path, "rb") as f:
                        f.seek(self.offset)
                        pos, buf = self.offset, b""
                        while True:
                            chunk = f.read(8 << 20)
                            if not chunk:
                                break
                            buf += chunk
                            start = 0
                            while True:
                                nl = buf.find(b"\n", start)
                                if nl < 0:
                                    break
                                line = buf[start:nl]
                                if line.strip():
                                    try:
                                        obj = json.loads(line)
                                    except ValueError:
                                        obj = None
                                    if isinstance(obj, dict):
                                        try:
                                            rec = self.proj(obj, pos + start, nl - start)
                                        except Exception:  # a malformed record must not break the view
                                            rec = None
                                        if rec is not None:
                                            self.records.append(rec)
                                start = nl + 1
                            pos += start
                            buf = buf[start:]
                        self.offset = pos
                except OSError:
                    pass
            return list(self.records)


_inc: dict[tuple[str, str], IncJsonl] = {}


def jsonl(path: str, proj: Callable[[dict, int, int], Any]) -> list:
    key = (path, proj.__name__)
    with _lock:
        r = _inc.get(key)
        if r is None:
            r = _inc[key] = IncJsonl(path, proj)
    return r.get()


def read_line_at(path: str, off: int, ln: int) -> Any:
    with open(path, "rb") as f:
        f.seek(off)
        return json.loads(f.read(ln))


def tail_text(path: str, nbytes: int = 16384) -> str | None:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > nbytes:
                f.seek(size - nbytes)
            data = f.read()
    except OSError:
        return None
    txt = data.decode("utf-8", "replace")
    if size > nbytes and "\n" in txt:
        txt = txt[txt.index("\n") + 1:]
    return txt


def first_line_json(path: str) -> Any:
    def load(p: str) -> Any:
        with open(p, "rb") as f:
            return json.loads(f.readline())
    return cached_file(path, load, "first")


def now() -> float:
    return time.time()


def utc(ts: Any) -> str | None:
    if ts is None:
        return None
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))
    except (TypeError, ValueError, OverflowError):
        return None


def trunc(s: Any, n: int) -> str | None:
    if s is None:
        return None
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False)
    if len(s) <= n:
        return s
    head = int(n * 0.65)
    return s[:head] + "\n... [%d chars omitted] ...\n" % (len(s) - n) + s[-(n - head):]
