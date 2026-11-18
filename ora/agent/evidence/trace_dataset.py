"""Many runs as one trace dataset: one record per episode, each image once, an index, and a tool that reads it.

``export_dataset`` takes a list of run directories (each with the public episode id and the labels the caller
gives it) and writes:

- ``episodes/episodes-NNNNN.jsonl[.gz]``: one ``ora.trace-episode.v1`` record per episode. Its ``segments`` are,
  in order, the conversations the model saw (a new one starts where ORA rewrote the context); a segment's
  ``messages`` are its last request and that call's reply, ``calls`` places every earlier request in them
  (``trace_dataset_tool``), ``tools`` the tool definitions (once per episode when every call had the same);
- ``images/images-NNNNN.tar`` and ``images.jsonl``: every image a model was sent, once across the whole
  dataset, in plain tar shards with each image's shard, offset and size;
- ``index.jsonl``: one line per episode (model, benchmark, task, mode, result, labels, sizes, where it is);
- ``ora_traces.py``: ``trace_dataset_tool``, which verifies and converts the dataset with the standard library;
- ``README.md`` and ``dataset.json`` (``ora.trace-dataset.v1``: counts, shards and the sha256 of every file).

Each run goes through ``wire_export.export_wire`` first, so every request is rebuilt and checked against the
run's sealed hashes before anything is derived from it, a privileged field refuses the run, and (unless
``redact_host_paths`` is off) host paths are replaced by placeholders. The run id, which names the internal
experiment, is replaced by the episode id wherever the model-visible text carries it, and an episode whose
text still holds one of the caller's ``forbid`` strings is refused. Nothing in the dataset names a run or a
host: the map from episode ids to runs and the list of refused and failed runs are written beside the dataset
(``<destination>.run-map.jsonl``, ``<destination>.report.json``), not into it.

The work is in two steps. Runs are exported in parallel, each into ``<work>/<episode id>/``, and a run already
exported there is not exported again, so an interrupted export resumes. Then the dataset is assembled in one
pass, in the order of the list, in a staging directory that is moved into place once it verifies.
"""

from __future__ import annotations

import collections
import concurrent.futures
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import openai_package_tool as package_tool
from . import trace_dataset_tool as tool
from .canonical import TrajectoryError
from .host_paths import HostPaths
from .wire_export import export_wire

ENTRY_KEYS = frozenset(("run", "episode_id", "model", "benchmark", "labels"))
BLOCK = 512

README = """# ORA traces

{count} episodes of a language model operating a robot through ORA: what the model received on every call,
what it replied, its reasoning, the tool definitions and every image.

| | |
| --- | --- |
| Episodes | {count} ({success} solved by the benchmark's own check) |
| Model calls | {calls} |
| Images | {images} ({image_bytes} bytes), each stored once |
| Models | {models} |
| Benchmarks | {benchmarks} |
| Modes | {modes} |
| Format | `{schema}`, one `{episode_schema}` record per episode |{redaction}

## Files

| Path | Contents |
| --- | --- |
| `episodes/*.jsonl{suffix}` | One record per episode |
| `index.jsonl` | One line per episode: model, benchmark, task, mode, result, labels, sizes, and its shard and line |
| `images/*.tar`, `images.jsonl` | Every image once, in plain tar shards; `images.jsonl` gives each image's shard, offset and size |
| `ora_traces.py` | Verify and convert this dataset; Python 3 standard library only |
| `dataset.json` | Counts, shards and the sha256 of every file |

## An episode

```json
{{"schema": "{episode_schema}", "episode_id": "...", "model": "...", "reasoning_effort": "...",
 "benchmark": "...", "task": "...", "seed": 0, "instruction": "...", "mode": "code" | "tool_calls",
 "result": {{"success": true, "termination": "...", "learning_eligible": true}},
 "labels": {{}}, "tools": [...],
 "segments": [{{"messages": [...], "calls": [{{"prefix": 5, "tail": null, "finish_reason": "...", "usage": {{}}}}]}}],
 "usage": {{"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}}}}
```

`segments` are, in order, the conversations the model saw. A new segment starts where ORA rewrote the context
(older images become `<recoverable_artifact>` pointers the model can reopen), so later segments repeat the
system prompt and what was kept. A segment's `messages` are its last request followed by that call's reply:
standard OpenAI Chat Completions messages, with `reasoning_content` on each assistant turn and each image as a
reference `images/<sha256>.<ext>`. The request of an earlier call of the segment is `messages[:prefix]`, plus
`tails[tail]` when the call carried a rolling camera message that later calls did not keep. `tools` are the
tool definitions; a segment has its own `tools` only when they differ from the episode's.

In code mode the model writes Python programs that call the robot library. The program and what it printed
are messages; the robot calls the program made are not.

## Use

```bash
python3 ora_traces.py verify --images             # every file, shard, reference and image
python3 ora_traces.py stats
python3 ora_traces.py openai EPISODE out.jsonl    # one OpenAI conversation per segment, images inline
python3 ora_traces.py calls EPISODE out.jsonl     # one OpenAI sample per model call, images inline
python3 ora_traces.py images DIR EPISODE          # the episode's images as files
```

```python
import gzip, json
for line in gzip.open("episodes/episodes-00000.jsonl.gz", "rt"):
    episode = json.loads(line)
    for segment in episode["segments"]:
        messages, tools = segment["messages"], segment.get("tools", episode["tools"])
```
"""


def read_entries(path: Path) -> list[dict[str, Any]]:
    """The run list: one run directory per line, or one JSON object per line
    (``run``, and optionally ``episode_id``, ``model``, ``benchmark``, ``labels``)."""

    entries = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entry = json.loads(line) if line.startswith("{") else {"run": line}
        if not isinstance(entry, dict) or not isinstance(entry.get("run"), str) or set(entry) - ENTRY_KEYS:
            raise TrajectoryError(f"{path}:{number}: an entry is a run directory or an object with 'run' and any of "
                                  f"{sorted(ENTRY_KEYS - {'run'})}")
        if "labels" in entry and not isinstance(entry["labels"], dict):
            raise TrajectoryError(f"{path}:{number}: labels is an object")
        entries.append(entry)
    return entries


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def default_episode_id(run_id: str) -> str:
    return "ep-" + hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)


def _replace(value: Any, old: str, new: str, tally: list[int]) -> Any:
    if isinstance(value, str):
        if old in value:
            tally[0] += value.count(old)
            return value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace(item, old, new, tally) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, old, new, tally) for key, item in value.items()}
    return value


def episode_record(wire: Path, plan: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    """One episode's record from the wire export of its run (see the module docstring)."""

    manifest = _json(wire / "manifest.json")
    requests = package_tool.read_jsonl(wire / "requests.jsonl")
    replies = {row["model_call_id"]: row for row in package_tool.read_jsonl(wire / "responses.jsonl")}
    policy = plan.get("policy") or {}
    by_epoch: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in requests:
        if row["answered"]:
            by_epoch[row["epoch"]].append(row)
    tool_sets = [json.dumps(row["request"].get("tools", []), sort_keys=True) for row in requests if row["answered"]]
    shared = len(set(tool_sets)) <= 1
    segments, totals = [], collections.Counter()
    for _, rows in sorted(by_epoch.items()):
        last = rows[-1]
        reply = package_tool.reply_message(replies[last["model_call_id"]]["provider_reply"])
        messages = package_tool.reasoning_content(list(last["request"].get("messages") or [])) + [reply]
        tails: list[Any] = []
        calls = []
        for row in rows:
            answer = replies[row["model_call_id"]].get("provider_reply") or {}
            usage = answer.get("usage")
            tail = None
            if row["epoch_tail"] is not None:
                tail = len(tails)
                tails.append(package_tool.reasoning_content([row["epoch_tail"]])[0])
            calls.append({"prefix": row["epoch_prefix"], "tail": tail, "finish_reason": answer.get("finish_reason"),
                          "usage": usage})
            totals["model_calls"] += 1
            if isinstance(usage, Mapping):
                totals["prompt_tokens"] += usage.get("prompt_tokens") or 0
                totals["completion_tokens"] += usage.get("completion_tokens") or 0
                totals["reasoning_tokens"] += ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        segment: dict[str, Any] = {"messages": messages, "calls": calls}
        if tails:
            segment["tails"] = tails
        if not shared:
            segment["tools"] = last["request"].get("tools", [])
        segments.append(segment)
    first = next((row["request"] for row in requests if row["answered"]), {})
    redaction = manifest.get("redaction")
    record = {
        "schema": tool.EPISODE_SCHEMA,
        "episode_id": entry["episode_id"],
        "model": entry.get("model") or first.get("model") or policy.get("model"),
        "reasoning_effort": first.get("reasoning_effort", policy.get("reasoning_effort")),
        "benchmark": entry.get("benchmark") or (plan.get("runtime") or {}).get("kind"),
        "task": plan.get("task"),
        "seed": plan.get("seed"),
        "instruction": plan.get("task_instruction"),
        "mode": "code" if policy.get("code_mode") else "tool_calls",
        "result": {"success": manifest.get("official_success"), "termination": manifest.get("termination"),
                   "learning_eligible": manifest.get("learning_eligible")},
        "labels": dict(entry.get("labels") or {}),
        "tools": first.get("tools", []) if shared else [],
        "segments": segments,
        "usage": {key: totals[key] for key in ("model_calls", "prompt_tokens", "completion_tokens", "reasoning_tokens")},
        "unanswered_calls": manifest.get("unanswered_calls", 0),
        "ora_commit": str(manifest.get("source_commit") or "")[:9] or None,
        "redaction": {"host_paths": None if redaction is None else redaction.get("replacements", 0), "run_id": 0},
    }
    return record


def _export_one(job: Mapping[str, Any]) -> dict[str, Any]:
    """Export one run into its work directory (a worker process); the episode's outcome, never an exception."""

    entry, work = job["entry"], Path(job["work"])
    run = Path(entry["run"])
    done = work / "done.json"
    if done.is_file():
        earlier = _json(done)
        if earlier.get("run") == entry["run"] and earlier.get("settings") == job["settings"]:
            return {**earlier, "resumed": True}
    outcome: dict[str, Any] = {"run": entry["run"], "episode_id": entry.get("episode_id"), "status": "failed"}
    try:
        if not run.is_dir():
            raise TrajectoryError("not a run directory")
        root = run.resolve()
        run_id = _json(root / "result.json").get("run_id") or root.name
        outcome["run_id"] = run_id
        entry = {**entry, "episode_id": entry.get("episode_id") or default_episode_id(run_id)}
        outcome["episode_id"] = entry["episode_id"]
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        host_paths = HostPaths.for_run(root, job["redact_paths"]) if job["redact_host_paths"] else None
        export_wire(root, work / "wire", host_paths=host_paths)
        record = episode_record(work / "wire", _json(root / "plan.json"), entry)
        tally = [0]
        for key in ("instruction", "tools", "segments"):  # what the model was sent; the run id names the experiment
            record[key] = _replace(record[key], run_id, entry["episode_id"], tally)
        record["redaction"]["run_id"] = tally[0]
        if not record["segments"]:
            raise TrajectoryError("no answered model call")
        text = "\n".join(_strings({key: value for key, value in record.items() if key != "labels"}))
        held = sorted(item for item in job["forbid"] if item and item in text)
        if held:
            outcome.update(status="refused", error="forbidden text: " + ", ".join(repr(item) for item in held))
            shutil.rmtree(work, ignore_errors=True)
            return outcome
        refs = sorted(set(tool.image_refs(record["segments"])))
        (work / "images").mkdir()
        for ref in refs:
            (work / "wire" / ref).rename(work / ref)
        shutil.rmtree(work / "wire")
        (work / "episode.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        outcome.update(status="exported", images=len(refs), model_calls=record["usage"]["model_calls"],
                       segments=len(record["segments"]), settings=job["settings"])
        done.write_text(json.dumps(outcome), encoding="utf-8")
    except (TrajectoryError, OSError, ValueError, KeyError) as error:
        outcome["error"] = f"{type(error).__name__}: {error}"[:500]
        shutil.rmtree(work, ignore_errors=True)
    return outcome


class _Shards:
    """Episode records appended to JSONL shards, a new one at ``limit`` uncompressed bytes."""

    def __init__(self, root: Path, compress: bool, limit: int) -> None:
        self.root, self.compress, self.limit = root, compress, limit
        self.shards: list[dict[str, Any]] = []
        self._handle: Any = None
        self._bytes = 0
        (root / "episodes").mkdir()

    def add(self, line: str) -> tuple[str, int]:
        size = len(line.encode("utf-8")) + 1
        if self._handle is None or (self._bytes and self._bytes + size > self.limit):
            self.close()
            name = f"episodes/episodes-{len(self.shards):05d}.jsonl" + (".gz" if self.compress else "")
            path = self.root / name
            self._handle = (gzip.open(path, "wt", encoding="utf-8", compresslevel=6) if self.compress
                            else path.open("w", encoding="utf-8"))
            self.shards.append({"path": name, "episodes": 0})
            self._bytes = 0
        self._handle.write(line + "\n")
        self._bytes += size
        shard = self.shards[-1]
        shard["episodes"] += 1
        return shard["path"], shard["episodes"] - 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class _Images:
    """Each image once, in plain (ustar) tar shards, with its place recorded."""

    def __init__(self, root: Path, limit: int) -> None:
        self.root, self.limit = root, limit
        self.rows: dict[str, dict[str, Any]] = {}
        self.shards: list[str] = []
        self._tar: tarfile.TarFile | None = None
        (root / "images").mkdir()

    def add(self, path: Path) -> None:
        sha, ext = path.name.split(".", 1)
        if sha in self.rows:
            return
        size = path.stat().st_size
        if self._tar is None or (self._tar.offset and self._tar.offset + size > self.limit):
            self.close()
            name = f"images/images-{len(self.shards):05d}.tar"
            self._tar = tarfile.open(self.root / name, "w", format=tarfile.USTAR_FORMAT)
            self.shards.append(name)
        info = tarfile.TarInfo(path.name)
        info.size, info.mode = size, 0o644
        offset = self._tar.offset + BLOCK
        with path.open("rb") as handle:
            self._tar.addfile(info, handle)
        self.rows[sha] = {"sha256": sha, "ext": ext, "media_type": tool.MIME.get(ext), "bytes": size,
                          "shard": self.shards[-1], "offset": offset}

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None


def _readme(index: list[Mapping[str, Any]], images: Mapping[str, Mapping[str, Any]], compress: bool, redacted: bool) -> str:
    def named(key: str) -> str:
        counts = collections.Counter(str(row.get(key)) for row in index)
        return ", ".join(f"{name} ({count})" for name, count in sorted(counts.items()))

    row = ("\n| Host paths | replaced by placeholders such as `<host-python>`; the rest of each path is kept |"
           if redacted else "")
    return README.format(
        count=len(index), success=sum(1 for item in index if item.get("success") is True),
        calls=sum(item.get("model_calls") or 0 for item in index), images=len(images),
        image_bytes=sum(item["bytes"] for item in images.values()), models=named("model"),
        benchmarks=named("benchmark"), modes=named("mode"), schema=tool.DATASET_SCHEMA,
        episode_schema=tool.EPISODE_SCHEMA, suffix=".gz" if compress else "", redaction=row)


def export_dataset(entries: Iterable[Mapping[str, Any]], destination: Path, *, work: Path | None = None,
                   workers: int = 4, redact_host_paths: bool = True, redact_paths: Iterable[str] = (),
                   forbid: Iterable[str] = (), compress: bool = True, shard_bytes: int = 512 << 20,
                   image_shard_bytes: int = 2 << 30) -> dict[str, Any]:
    """Write the dataset (see the module docstring); the report, which is also written beside it."""

    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise TrajectoryError(f"export destination exists: {destination}")
    work = Path(work).expanduser().resolve() if work is not None else destination.with_name(destination.name + ".work")
    entries = [dict(entry) for entry in entries]
    if not entries:
        raise TrajectoryError("no runs to export")
    named = [entry["episode_id"] for entry in entries if entry.get("episode_id")]
    twice = sorted(name for name, count in collections.Counter(named).items() if count > 1)
    if twice:
        raise TrajectoryError(f"episode ids given twice: {', '.join(twice[:5])}")
    work.mkdir(parents=True, exist_ok=True)
    settings = {"redact_host_paths": redact_host_paths, "redact_paths": sorted(redact_paths), "forbid": sorted(forbid)}
    jobs = [{"entry": entry, "work": str(work / f"{number:07d}"), "redact_host_paths": redact_host_paths,
             "redact_paths": list(redact_paths), "forbid": list(forbid), "settings": settings}
            for number, entry in enumerate(entries)]
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_export_one, jobs, chunksize=1))
    else:
        outcomes = [_export_one(job) for job in jobs]
    exported = [(job, outcome) for job, outcome in zip(jobs, outcomes) if outcome["status"] == "exported"]
    ids = collections.Counter(outcome["episode_id"] for _, outcome in exported)
    clash = sorted(name for name, count in ids.items() if count > 1)
    if clash:
        raise TrajectoryError(f"two runs have the episode id {clash[0]}; give them ids in the run list")

    staging = destination.with_name(f".{destination.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        shards, images = _Shards(staging, compress, shard_bytes), _Images(staging, image_shard_bytes)
        index: list[dict[str, Any]] = []
        for job, _ in exported:
            folder = Path(job["work"])
            line = (folder / "episode.json").read_text(encoding="utf-8")
            record = json.loads(line)
            refs = sorted(set(tool.image_refs(record["segments"])))
            for ref in refs:
                images.add(folder / ref)
            shard, place = shards.add(line)
            index.append({
                "episode_id": record["episode_id"], "model": record["model"], "benchmark": record["benchmark"],
                "task": record["task"], "seed": record["seed"], "mode": record["mode"],
                "success": record["result"]["success"], "labels": record["labels"],
                "model_calls": record["usage"]["model_calls"], "segments": len(record["segments"]),
                "images": len(refs), "prompt_tokens": record["usage"]["prompt_tokens"],
                "completion_tokens": record["usage"]["completion_tokens"], "shard": shard, "line": place,
            })
        shards.close()
        images.close()
        if not index:
            raise TrajectoryError("no run exported; see the report")
        package_tool.write_jsonl(staging / tool.INDEX_FILE, index)
        package_tool.write_jsonl(staging / tool.IMAGES_FILE, list(images.rows.values()))
        shutil.copyfile(Path(tool.__file__), staging / tool.TOOL_FILE)
        (staging / "README.md").write_text(_readme(index, images.rows, compress, redact_host_paths), encoding="utf-8")
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        (staging / tool.DATASET_FILE).write_text(json.dumps({
            "schema": tool.DATASET_SCHEMA, "episode_schema": tool.EPISODE_SCHEMA,
            "episodes": {"count": len(index), "shards": shards.shards},
            "images": {"count": len(images.rows), "bytes": sum(row["bytes"] for row in images.rows.values()),
                       "shards": images.shards},
            "host_paths_redacted": redact_host_paths,
            "files": [{"path": str(path.relative_to(staging)), "sha256": tool.file_sha256(path),
                       "bytes": path.stat().st_size} for path in files],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        check = tool.verify(staging, images=True)
        if not check["identical"]:
            raise TrajectoryError("the dataset does not verify: " + json.dumps(check["failures"][:3]))
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for job in jobs:  # only what this export put there
        shutil.rmtree(job["work"], ignore_errors=True)
    try:
        work.rmdir()
    except OSError:
        pass
    report = {
        "destination": str(destination), "runs": len(entries), "exported": len(index),
        "refused": [item for item in outcomes if item["status"] == "refused"],
        "failed": [item for item in outcomes if item["status"] == "failed"],
        "images": len(images.rows), "episode_shards": len(shards.shards), "image_shards": len(images.shards),
    }
    destination.with_name(destination.name + ".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    package_tool.write_jsonl(destination.with_name(destination.name + ".run-map.jsonl"), [
        {"episode_id": outcome["episode_id"], "run_id": outcome.get("run_id"), "run": outcome["run"]}
        for _, outcome in exported])
    return report


__all__ = ["default_episode_id", "episode_record", "export_dataset", "read_entries"]
