"""The tool a trace dataset ships: verify it and read it, with the standard library only.

``ora export-traces`` copies this file into the dataset as ``ora_traces.py``, so whoever receives the dataset
can check and convert it without ORA; ORA's own check runs the same code.

A dataset holds one record per episode (``episodes/*.jsonl`` or ``*.jsonl.gz``, schema ``ora.trace-episode.v1``).
An episode's ``segments`` are, in order, the conversations the model saw: a new one starts where ORA rewrote the
context. A segment's ``messages`` are its last request followed by that call's reply; ``calls`` says how every
earlier request of the segment reads from them (``messages[:prefix]`` plus, for a rolling camera tail,
``tails[tail]``). Images are references ``images/<sha256>.<ext>``; the bytes are in ``images/*.tar``, each image
once, found through ``images.jsonl`` (``shard``, ``offset``, ``bytes``).

    python3 ora_traces.py verify [--images]          every shard, index row and image reference (and image bytes)
    python3 ora_traces.py stats                      episodes by model, benchmark, mode and result
    python3 ora_traces.py openai EPISODE OUT.jsonl   one OpenAI conversation per segment, images inline
    python3 ora_traces.py calls EPISODE OUT.jsonl    one OpenAI sample per model call, images inline
    python3 ora_traces.py images DIR [EPISODE]       the images as DIR/<sha256>.<ext>
"""

from __future__ import annotations

import argparse
import base64
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

DATASET_SCHEMA = "ora.trace-dataset.v1"
EPISODE_SCHEMA = "ora.trace-episode.v1"
DATASET_FILE = "dataset.json"
INDEX_FILE = "index.jsonl"
IMAGES_FILE = "images.jsonl"
TOOL_FILE = "ora_traces.py"
MIME = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path, mode: str = "rt") -> Any:
    return gzip.open(path, mode, encoding="utf-8") if path.name.endswith(".gz") else path.open(mode, encoding="utf-8")


def read_jsonl(path: Path) -> list[Any]:
    with open_text(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("images/") and "://" not in value


def image_refs(value: Any) -> Iterator[str]:
    """Every image reference in a message tree (an OpenAI ``image_url`` or an Anthropic image ``source``)."""
    if isinstance(value, list):
        for item in value:
            yield from image_refs(item)
    elif isinstance(value, dict):
        image, source = value.get("image_url"), value.get("source")
        if value.get("type") == "image_url" and isinstance(image, dict) and is_ref(image.get("url")):
            yield image["url"]
        elif value.get("type") == "image" and isinstance(source, dict) and is_ref(source.get("data")):
            yield source["data"]
        else:
            for item in value.values():
                yield from image_refs(item)


class Dataset:
    """A trace dataset on disk: its manifest, index, episodes and images."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / DATASET_FILE).read_text(encoding="utf-8"))
        self.index = read_jsonl(self.root / INDEX_FILE)
        self._images: dict[str, dict[str, Any]] | None = None
        self._where = {row["episode_id"]: row for row in self.index}

    @property
    def images(self) -> dict[str, dict[str, Any]]:
        if self._images is None:
            self._images = {row["sha256"]: row for row in read_jsonl(self.root / IMAGES_FILE)}
        return self._images

    def episodes(self) -> Iterator[dict[str, Any]]:
        for shard in self.manifest["episodes"]["shards"]:
            with open_text(self.root / shard["path"]) as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)

    def episode(self, episode_id: str) -> dict[str, Any]:
        row = self._where.get(episode_id)
        if row is None:
            raise SystemExit(f"no such episode: {episode_id}")
        with open_text(self.root / row["shard"]) as handle:
            for number, line in enumerate(handle):
                if number == row["line"]:
                    return json.loads(line)
        raise SystemExit(f"{row['shard']} has no line {row['line']}")

    def image_bytes(self, ref: str) -> bytes:
        name = ref.rsplit("/", 1)[1]
        row = self.images[name.split(".", 1)[0]]
        with (self.root / row["shard"]).open("rb") as handle:
            handle.seek(row["offset"])
            return handle.read(row["bytes"])

    def inline(self, value: Any) -> Any:
        """Each image reference back to the data URL or base64 source the model was sent."""
        if isinstance(value, list):
            return [self.inline(item) for item in value]
        if not isinstance(value, dict):
            return value
        out = {key: self.inline(item) for key, item in value.items()}
        image, source = value.get("image_url"), value.get("source")
        if value.get("type") == "image_url" and isinstance(image, dict) and is_ref(image.get("url")):
            data = base64.b64encode(self.image_bytes(image["url"])).decode("ascii")
            out["image_url"] = {**image, "url": f"data:{MIME[image['url'].rsplit('.', 1)[1]]};base64,{data}"}
        if value.get("type") == "image" and isinstance(source, dict) and is_ref(source.get("data")):
            out["source"] = {**source, "data": base64.b64encode(self.image_bytes(source["data"])).decode("ascii")}
        return out


def segment_tools(episode: dict[str, Any], segment: dict[str, Any]) -> list[Any]:
    return segment["tools"] if "tools" in segment else episode.get("tools", [])


def call_samples(episode: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every model call of an episode as the request the model received and the reply it gave."""
    for number, segment in enumerate(episode["segments"]):
        messages, tails, calls = segment["messages"], segment.get("tails", []), segment["calls"]
        for position, call in enumerate(calls):
            request = messages[:call["prefix"]] + ([tails[call["tail"]]] if call.get("tail") is not None else [])
            last = position == len(calls) - 1
            yield {"episode_id": episode["episode_id"], "segment": number, "call": position,
                   "messages": request, "response": messages[-1] if last else messages[call["prefix"]],
                   "tools": segment_tools(episode, segment), "finish_reason": call.get("finish_reason"),
                   "usage": call.get("usage")}


def verify(root: Path, *, images: bool = False) -> dict[str, Any]:
    """Check a dataset against its own manifest: files, shards, the index, every image reference, and with
    ``images`` every image's bytes against its sha256."""
    failures: list[dict[str, str]] = []

    def fail(check: str, detail: str) -> None:
        if len(failures) < 50:
            failures.append({"check": check, "detail": detail})

    root = Path(root)
    try:
        dataset = Dataset(root)
    except (OSError, ValueError, KeyError) as error:
        return {"identical": False, "failures": [{"check": "manifest", "detail": str(error)}]}
    manifest = dataset.manifest
    if manifest.get("schema") != DATASET_SCHEMA:
        fail("schema", str(manifest.get("schema")))
    for item in manifest.get("files", []):
        path = root / item["path"]
        if not path.is_file():
            fail("file_missing", item["path"])
        elif path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            fail("file_changed", item["path"])
    listed = {item["path"] for item in manifest.get("files", [])} | {DATASET_FILE}
    extra = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
                   and str(path.relative_to(root)) not in listed)
    known = dataset.images
    seen, used, count = set(), set(), 0
    for shard in manifest["episodes"]["shards"]:
        for line, episode in enumerate(read_jsonl(root / shard["path"])):
            count += 1
            episode_id = episode.get("episode_id")
            if episode.get("schema") != EPISODE_SCHEMA:
                fail("episode_schema", str(episode_id))
            if episode_id in seen:
                fail("episode_twice", str(episode_id))
            seen.add(episode_id)
            row = dataset._where.get(episode_id)
            if row is None or row["shard"] != shard["path"] or row["line"] != line:
                fail("index_place", str(episode_id))
            refs = set(image_refs(episode.get("segments", [])))
            for ref in refs:
                name = ref.rsplit("/", 1)[1]
                if name.split(".", 1)[0] not in known:
                    fail("image_unknown", f"{episode_id}: {ref}")
                used.add(name.split(".", 1)[0])
            if row is not None and row.get("images") != len(refs):
                fail("index_images", str(episode_id))
            for number, segment in enumerate(episode.get("segments", [])):
                size = len(segment["messages"]) - 1
                for call in segment["calls"]:
                    tail = call.get("tail")
                    if not 0 <= call["prefix"] <= size or (tail is not None and not 0 <= tail < len(segment.get("tails", []))):
                        fail("call_place", f"{episode_id} segment {number}")
    if count != manifest["episodes"]["count"] or count != len(dataset.index):
        fail("episode_count", f"{count} in shards, {len(dataset.index)} in the index, {manifest['episodes']['count']} in the manifest")
    if set(known) - used:
        fail("image_unused", f"{len(set(known) - used)} images no episode names")
    if images:
        handles: dict[str, Any] = {}
        try:
            for sha, row in known.items():
                handle = handles.get(row["shard"])
                if handle is None:
                    handle = handles[row["shard"]] = (root / row["shard"]).open("rb")
                handle.seek(row["offset"])
                if hashlib.sha256(handle.read(row["bytes"])).hexdigest() != sha:
                    fail("image_bytes", sha)
        finally:
            for handle in handles.values():
                handle.close()
    return {"identical": not failures, "episodes": count, "images": len(known), "images_checked": images,
            "files_not_listed": extra, "failures": failures}


def stats(root: Path) -> dict[str, Any]:
    table: dict[str, collections.Counter] = {key: collections.Counter() for key in ("model", "benchmark", "mode", "success")}
    rows = read_jsonl(Path(root) / INDEX_FILE)
    for row in rows:
        for key in table:
            table[key][str(row.get(key))] += 1
    return {"episodes": len(rows), "model_calls": sum(row.get("model_calls") or 0 for row in rows),
            **{key: dict(sorted(value.items())) for key, value in table.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="the dataset (default: this file's folder)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify").add_argument("--images", action="store_true")
    commands.add_parser("stats")
    for name in ("openai", "calls"):
        command = commands.add_parser(name)
        command.add_argument("episode")
        command.add_argument("out", type=Path)
    command = commands.add_parser("images")
    command.add_argument("directory", type=Path)
    command.add_argument("episode", nargs="?")
    args = parser.parse_args(argv)
    if args.command == "verify":
        report = verify(args.root, images=args.images)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["identical"] else 2
    if args.command == "stats":
        print(json.dumps(stats(args.root), indent=2, sort_keys=True))
        return 0
    dataset = Dataset(args.root)
    if args.command in ("openai", "calls"):
        episode = dataset.episode(args.episode)
        if args.command == "openai":
            rows = [{"episode_id": episode["episode_id"], "segment": number, "messages": dataset.inline(segment["messages"]),
                     "tools": segment_tools(episode, segment)} for number, segment in enumerate(episode["segments"])]
        else:
            rows = [{**sample, "messages": dataset.inline(sample["messages"])} for sample in call_samples(episode)]
        with args.out.open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        print(f"{len(rows)} rows -> {args.out}")
        return 0
    wanted = None if args.episode is None else {ref.rsplit("/", 1)[1] for ref in image_refs(dataset.episode(args.episode)["segments"])}
    args.directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for sha, row in dataset.images.items():
        name = f"{sha}.{row['ext']}"
        if wanted is None or name in wanted:
            (args.directory / name).write_bytes(dataset.image_bytes("images/" + name))
            written += 1
    print(f"{written} images -> {args.directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
