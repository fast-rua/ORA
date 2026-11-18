"""The tool an OpenAI package ships: verify it, and turn it into other training formats, with the standard library only.

``ora export-sft RUN --view openai`` copies this file into the package as ``ora_package.py``, so whoever receives
the package can check it without ORA; ORA's ``--check`` runs the same code. It imports nothing outside the
standard library, and the package's own files are derived by ``derive`` here, so the exporter and the check
cannot disagree about them.

A package exported with ``--redact-host-paths`` lists in ``wire/manifest.json`` ``redaction.rows`` each row
whose host paths it replaced, with the sha256 of its redacted form; its sealed sha256 is not shipped (it would
confirm a guessed prefix). ``verify`` checks such a row against the redacted sha256 (``request_redacted``,
``stream_redacted``, ...) and says that the package alone cannot prove it is what the model was sent.

    python3 ora_package.py verify [--tokenize-url URL --key-env NAME]
    python3 ora_package.py calls OUT.jsonl          one OpenAI sample per model call, images inline
    python3 ora_package.py images DIR               every image as DIR/<sha256>.<ext>
    python3 ora_package.py speculator OUT.jsonl [--per-epoch]
                                                    {input_ids, loss_mask} rows from the sampled token ids
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import os
import shutil
import ssl
import struct
import sys
import urllib.request
from pathlib import Path
from typing import Any

PACKAGE_SCHEMA = "ora.openai-package.v1"
PACKAGE_FILE = "package.json"
TOOL_FILE = "ora_package.py"
MIME = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}


def canonical(value: Any) -> bytes:
    """The serialization the run archived each request and stream in (``ora.agent.evidence.canonical``)."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ids_sha256(ids: list[int]) -> str:
    return sha256(struct.pack(f"<{len(ids)}I", *ids))


def read_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def inline(value: Any, base: Path) -> Any:
    """Each image file reference (relative to ``base``) back to the data URL or base64 source that was sent."""
    if isinstance(value, list):
        return [inline(item, base) for item in value]
    if not isinstance(value, dict):
        return value
    out = {key: inline(item, base) for key, item in value.items()}
    image, source = value.get("image_url"), value.get("source")
    if value.get("type") == "image_url" and isinstance(image, dict) and _is_ref(image.get("url")):
        data = base64.b64encode((base / image["url"]).read_bytes()).decode("ascii")
        out["image_url"] = {**image, "url": f"data:{MIME[image['url'].rsplit('.', 1)[1]]};base64,{data}"}
    if value.get("type") == "image" and isinstance(source, dict) and _is_ref(source.get("data")):
        out["source"] = {**source, "data": base64.b64encode((base / source["data"]).read_bytes()).decode("ascii")}
    return out


def _is_ref(value: Any) -> bool:
    return isinstance(value, str) and "images/" in value and not value.startswith("data:") and "://" not in value


def rebase(value: Any, prefix: str) -> Any:
    """Image references of a wire row (``images/...``) made relative to the package root (``wire/images/...``)."""
    if isinstance(value, list):
        return [rebase(item, prefix) for item in value]
    if not isinstance(value, dict):
        return value
    out = {key: rebase(item, prefix) for key, item in value.items()}
    image, source = value.get("image_url"), value.get("source")
    if value.get("type") == "image_url" and isinstance(image, dict) and _is_ref(image.get("url")):
        out["image_url"] = {**image, "url": prefix + image["url"]}
    if value.get("type") == "image" and isinstance(source, dict) and _is_ref(source.get("data")):
        out["source"] = {**source, "data": prefix + source["data"]}
    return out


def reasoning_content(messages: list[Any]) -> list[Any]:
    """vLLM's history field ``reasoning`` under the usual name ``reasoning_content``."""
    return [{("reasoning_content" if key == "reasoning" else key): item for key, item in message.items()}
            if isinstance(message, dict) and "reasoning" in message else message for message in messages]


def reply_message(reply: dict[str, Any] | None) -> dict[str, Any] | None:
    if reply is None:
        return None
    message: dict[str, Any] = {"role": "assistant", "content": reply.get("content") or ""}
    if reply.get("reasoning_content") is not None:
        message["reasoning_content"] = reply["reasoning_content"]
    if reply.get("tool_calls"):
        message["tool_calls"] = reply["tool_calls"]
    return message


def derive(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The package's conversations and calls, from its ``wire/`` export.

    A conversation is an epoch (the calls whose requests extend each other): its last request with every image
    inline, followed by that call's reply. A call is the request as sent with images as ``wire/images/...``
    references, its reply, finish reason, usage and token counts.
    """
    wire = root / "wire"
    requests = read_jsonl(wire / "requests.jsonl")
    replies = {row["model_call_id"]: row for row in read_jsonl(wire / "responses.jsonl")}
    tokens = {row["model_call_id"]: row for row in read_jsonl(wire / "tokens.jsonl")}
    run_id = json.loads((wire / "manifest.json").read_text(encoding="utf-8"))["run_id"]
    calls = []
    for row in requests:
        call_id = row["model_call_id"]
        response, token_row = replies[call_id], tokens[call_id]
        reply = response.get("provider_reply")
        request = rebase(row["request"], "wire/")
        calls.append({
            "index": row["index"], "model_call_id": call_id, "epoch": row["epoch"], "protocol": row["protocol"],
            "answered": row["answered"], "accepted": response.get("accepted"),
            "request": {**request, "messages": reasoning_content(request.get("messages") or [])},
            "response": reply_message(reply), "finish_reason": None if reply is None else reply.get("finish_reason"),
            "usage": None if reply is None else reply.get("usage"),
            "token_counts": {
                "prompt": None if token_row["prompt"] is None else token_row["prompt"]["length"],
                "sampled": None if token_row["output"] is None else token_row["output"]["length"],
                "prompt_equal_to_usage": token_row["checks"]["prompt_length"],
                "sampled_equal_to_usage": token_row["checks"]["output_length"],
            },
        })
    last: dict[int, dict[str, Any]] = {}
    for row in requests:
        if row["answered"]:
            last[row["epoch"]] = row
    conversations = []
    for epoch, row in sorted(last.items()):
        body = inline(row["request"], wire)
        conversations.append({
            "run_id": run_id, "epoch": epoch,
            "model_call_ids": [item["model_call_id"] for item in requests if item["epoch"] == epoch and item["answered"]],
            "messages": reasoning_content(body.get("messages") or [])
            + [reply_message(replies[row["model_call_id"]]["provider_reply"])],
            "tools": body.get("tools", []),
        })
    return conversations, calls


def _stream_usage(response: Any) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for chunk in response if isinstance(response, list) else [response]:
        if isinstance(chunk, dict) and isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
    return usage


def _stream_ids(response: Any) -> tuple[list[int] | None, list[int] | None]:
    if isinstance(response, dict):
        choices = response.get("choices") or [{}]
        prompt, output = response.get("prompt_token_ids"), choices[0].get("token_ids")
        return (prompt if isinstance(prompt, list) else None), (output if isinstance(output, list) else None)
    prompt, output, sampled = None, [], False
    for chunk in response if isinstance(response, list) else []:
        if prompt is None and isinstance(chunk.get("prompt_token_ids"), list):
            prompt = chunk["prompt_token_ids"]
        for choice in chunk.get("choices") or []:
            if isinstance(choice.get("token_ids"), list):
                sampled, output = True, output + choice["token_ids"]
    return prompt, (output if sampled else None)


def prompt_ids(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Every row's whole prompt, rebuilt from the prefix it shares with the previous call and its suffix."""
    prompts: dict[str, list[int]] = {}
    for row in rows:
        prompt = row["prompt"]
        if prompt is None:
            continue
        prefix = prompt["prefix"]
        prompts[row["model_call_id"]] = (prompts[prefix["model_call_id"]][: prefix["length"]] if prefix else []) + prompt["suffix"]
    return prompts


def tokenize_payload(body: dict[str, Any]) -> dict[str, Any]:
    """The tokenize request for a chat request; vLLM's chat endpoint merges reasoning_effort into the template kwargs."""
    payload = {key: body[key] for key in ("model", "messages", "tools") if key in body}
    kwargs = dict(body.get("chat_template_kwargs") or {})
    if body.get("reasoning_effort") not in (None, "auto"):
        kwargs["reasoning_effort"] = body["reasoning_effort"]
        kwargs.setdefault("enable_thinking", body["reasoning_effort"] != "none")
    if kwargs:
        payload["chat_template_kwargs"] = kwargs
    return payload


def verify(root: Path, *, tokenize_url: str | None = None, key: str | None = None) -> dict[str, Any]:
    """Check the package against the hashes the run archived; see the module docstring."""
    root = Path(root)
    wire = root / "wire"
    counts: collections.Counter = collections.Counter()
    failures: list[dict[str, str]] = []

    def check(kind: str, ok: bool, detail: str = "") -> None:
        counts[kind] += ok
        if not ok:
            failures.append({"check": kind, "detail": detail})

    try:
        package = json.loads((root / PACKAGE_FILE).read_text(encoding="utf-8"))
        manifest = json.loads((wire / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"identical": False, "failures": [{"check": "missing_file", "detail": str(exc)}]}
    check("package_schema", package.get("schema") == PACKAGE_SCHEMA, str(package.get("schema")))
    check("wire_manifest", sha256((wire / "manifest.json").read_bytes()) == package.get("wire_manifest_sha256"))
    listed = {entry["path"] for entry in package.get("files", [])}
    # Files a user added (outputs of this tool, notes) are listed, not refused; wire/ admits none.
    unlisted = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
                      and path.relative_to(root).parts[0] != "wire" and path.name != PACKAGE_FILE
                      and str(path.relative_to(root)) not in listed and "__pycache__" not in path.parts)
    for entry in package.get("files", []):
        path = root / entry["path"]
        check("package_file", path.is_file() and sha256(path.read_bytes()) == entry["sha256"]
              and path.stat().st_size == entry["bytes"], entry["path"])
    for entry in manifest["files"]:
        path = wire / entry["path"]
        check("wire_file", path.is_file() and sha256(path.read_bytes()) == entry["sha256"]
              and path.stat().st_size == entry["bytes"], entry["path"])
    index = {item["path"]: item for item in read_jsonl(wire / "images.jsonl")}
    images = sorted((wire / "images").iterdir()) if (wire / "images").is_dir() else []
    check("image_index", {f"images/{path.name}" for path in images} == set(index))
    for path in images:
        data = path.read_bytes()
        check("image", path.stem == sha256(data) and index.get(f"images/{path.name}", {}).get("bytes") == len(data),
              path.name)
    redaction = manifest.get("redaction")
    if redaction is not None or "redaction" in package:
        check("redaction_record", redaction is not None and "redaction" in package,
              "package.json and wire/manifest.json disagree")
    changed = {(item["file"], item["model_call_id"], item["field"]): item for item in (redaction or {}).get("rows", [])}
    seen = set()

    def expected(file: str, call_id: str, field: str, sealed: str | None) -> tuple[str | None, str]:
        """The sha256 a row must have, and the check's suffix: a redacted row's is the one recorded at export."""
        item = changed.get((file, call_id, field))
        if item is None:
            return sealed, ""
        seen.add((file, call_id, field))
        check("redaction_record", sealed is None, f"{call_id}: a redacted row keeps its sealed sha256")
        return item["redacted_sha256"], "_redacted"

    requests = read_jsonl(wire / "requests.jsonl")
    epochs = {item["epoch"]: item["messages"] for item in read_jsonl(wire / "epochs.jsonl")}
    for row in requests:
        want, suffix = expected("requests.jsonl", row["model_call_id"], "request", row["request_sha256"])
        check("request" + suffix, sha256(canonical(inline(row["request"], wire))) == want, row["model_call_id"])
        messages = epochs.get(row["epoch"], [])[: row["epoch_prefix"]] + (
            [row["epoch_tail"]] if row.get("epoch_tail") is not None else [])
        check("request_from_epoch" + suffix, sha256(canonical(inline({**row["request"], "messages": messages}, wire)))
              == want, row["model_call_id"])
    streams = {row["model_call_id"]: row for row in read_jsonl(wire / "streams.jsonl")}
    for call_id, row in streams.items():
        if row["response"] is not None:
            want, suffix = expected("streams.jsonl", call_id, "response", row["response_sha256"])
            check("stream" + suffix, sha256(canonical(row["response"])) == want, call_id)
    token_rows = read_jsonl(wire / "tokens.jsonl")
    tokens_by_call = {row["model_call_id"]: row for row in token_rows}
    for (file, call_id, field), item in changed.items():
        if (file, field) in (("requests.jsonl", "request"), ("streams.jsonl", "response")):
            check("redaction_record", (file, call_id, field) in seen, f"{call_id}: no such row in {file}")
        # Token ids of the original text would contradict the redacted one: a redacted call ships none.
        response = (streams.get(call_id) or {}).get("response")
        if file == "requests.jsonl" and item["replacements"]:
            check("redaction_ids", (tokens_by_call.get(call_id) or {}).get("prompt") is None
                  and _stream_ids(response)[0] is None, call_id)
        if file == "streams.jsonl" and item["replacements"]:
            check("redaction_ids", (tokens_by_call.get(call_id) or {}).get("output") is None
                  and _stream_ids(response)[1] is None, call_id)
    try:
        prompts = prompt_ids(token_rows)
    except (KeyError, TypeError) as exc:
        check("prompt_ids", False, f"the prefix chain breaks: {exc}")
        prompts = {}
    for row in token_rows:
        call_id, prompt = row["model_call_id"], row["prompt"]
        response = (streams.get(call_id) or {}).get("response")
        streamed_prompt, streamed_output = _stream_ids(response)
        usage = _stream_usage(response)
        if prompt is not None and call_id in prompts:
            ids = prompts[call_id]
            check("prompt_ids", ids_sha256(ids) == prompt["sha256"] and len(ids) == prompt["length"], call_id)
            if usage.get("prompt_tokens") is not None and prompt["image_form"] is not None:
                image_tokens = (usage.get("prompt_tokens_details") or {}).get("image_tokens") or 0
                expected = len(ids) if prompt["image_form"] == "expanded" else len(ids) - prompt["images"] + image_tokens
                check("prompt_length_usage", expected == usage["prompt_tokens"], call_id)
            if streamed_prompt is not None:
                check("prompt_ids_stream", streamed_prompt == ids, call_id)
        if row["output"] is not None:
            check("sampled_ids_stream", streamed_output == row["output"]["ids"]
                  and ids_sha256(row["output"]["ids"]) == row["output"]["sha256"], call_id)
    conversations, calls = derive(root)
    for name, rows in (("conversations.jsonl", conversations), ("calls.jsonl", calls)):
        recorded = read_jsonl(root / name) if (root / name).is_file() else []
        check(name, len(recorded) == len(rows), f"{len(recorded)} rows, wire/ gives {len(rows)}")
        for position, (expected, actual) in enumerate(zip(rows, recorded)):
            check(name, canonical(expected) == canonical(actual), f"row {position}")
    if tokenize_url:
        context = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
        for row in requests:
            if row["protocol"] != "openai_chat_completions":
                continue
            data = json.dumps(tokenize_payload(inline(row["request"], wire))).encode()
            with urllib.request.urlopen(urllib.request.Request(tokenize_url, data, headers), context=context,
                                        timeout=300) as answer:
                ids = json.loads(answer.read())["tokens"]
            check("engine_tokenize", row["model_call_id"] in prompts
                  and ids_sha256(ids) == ids_sha256(prompts[row["model_call_id"]]), row["model_call_id"])
    report = {"identical": not failures, "checks": dict(sorted(counts.items())), "failures": failures,
              "unlisted": unlisted}
    if redaction is not None:
        report["redaction"] = redaction_summary(redaction, len(requests), package.get("redaction") or {})
    return report


def redaction_summary(redaction: dict[str, Any], calls: int, package: dict[str, Any]) -> dict[str, Any]:
    """How much of a redacted package was changed, and what ``verify`` can then prove."""
    rows = redaction.get("rows") or []
    requests = sum(1 for item in rows if item["file"] == "requests.jsonl" and item["field"] == "request")
    streams = sum(1 for item in rows if item["file"] == "streams.jsonl")
    files = {path: count for path, count in (package.get("files") or {}).items() if count}
    if not rows and not files:
        text = ("screened for host paths (--redact-host-paths): none found; every request and stream is checked "
                "against its sealed sha256")
    else:
        text = (f"{requests} of {calls} requests and {streams} streams are redacted ({redaction.get('replacements')} "
                "host-path replacements; wire/manifest.json redaction.rows"
                + (", and " + ", ".join(f"{count} in {path}" for path, count in files.items()) if files else "")
                + "): each is checked against the sha256 of its redacted form recorded at export; their sealed "
                "sha256 is not shipped, so this package alone cannot prove they are what the model was sent; with "
                "the sealed run, "
                "ora export-sft RUN --view openai --destination THIS_DIR --check redacts the run again and "
                "compares, and an export without --redact-host-paths gives the original bytes")
    return {"requests": requests, "streams": streams, "replacements": redaction.get("replacements"),
            "files": files, "statement": text}


def inline_calls(root: Path, out: Path) -> int:
    """One standard OpenAI sample per call: the request's messages with images inline, then the reply."""
    rows = []
    for call in read_jsonl(Path(root) / "calls.jsonl"):
        if call["response"] is None:
            continue
        request = inline(call["request"], Path(root))
        rows.append({"model_call_id": call["model_call_id"], "messages": request["messages"] + [call["response"]],
                     "tools": request.get("tools", []), "usage": call["usage"]})
    write_jsonl(out, rows)
    return len(rows)


def dump_images(root: Path, directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    images = sorted((Path(root) / "wire" / "images").iterdir())
    for path in images:
        shutil.copyfile(path, directory / path.name)
    return len(images)


def speculator(root: Path, out: Path, *, per_epoch: bool = False) -> list[dict[str, Any]]:
    """{input_ids, loss_mask} rows: prompt ids 0, sampled ids 1; per epoch, calls that extend each other in one row."""
    rows = read_jsonl(Path(root) / "wire" / "tokens.jsonl")
    if any(row["prompt"] is None or row["output"] is None for row in rows):
        raise SystemExit("speculator rows need prompt and sampled ids on every call; this run has "
                         f"{sum(row['output'] is not None for row in rows)} of {len(rows)} sampled")
    prompts = prompt_ids(rows)
    samples: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in rows:
        prompt, sampled = prompts[row["model_call_id"]], row["output"]["ids"]
        if not per_epoch or current is None or row["extends_previous"] is not True:
            current = {"model_call_ids": [], "input_ids": list(prompt), "loss_mask": [0] * len(prompt)}
            samples.append(current)
        else:
            added = prompt[len(current["input_ids"]):]
            current["input_ids"] += added
            current["loss_mask"] += [0] * len(added)
        current["model_call_ids"].append(row["model_call_id"])
        current["input_ids"] += sampled
        current["loss_mask"] += [1] * len(sampled)
    write_jsonl(out, samples)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="the package (default: here)")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify", help="check every file, image, request, stream and token id (a redacted row "
                                          "against the sha256 recorded at export)")
    check.add_argument("--tokenize-url", help="also re-tokenize every request: vLLM <root>/tokenize, sglang <root>/v1/tokenize")
    check.add_argument("--key-env", help="the environment variable holding that endpoint's bearer key")
    sub.add_parser("calls", help="one OpenAI sample per model call, images inline").add_argument("out", type=Path)
    sub.add_parser("images", help="write every image to a directory").add_argument("directory", type=Path)
    spec = sub.add_parser("speculator", help="{input_ids, loss_mask} rows from the sampled token ids")
    spec.add_argument("out", type=Path)
    spec.add_argument("--per-epoch", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "verify":
        report = verify(args.root, tokenize_url=args.tokenize_url,
                        key=os.environ.get(args.key_env) if args.key_env else None)
        for kind, number in report.get("checks", {}).items():
            print(f"ok   {kind}: {number}")
        for failure in report["failures"][:20]:
            print(f"FAIL {failure['check']}: {failure['detail']}")
        if report.get("unlisted"):
            print(f"note: {len(report['unlisted'])} files not in the package's list: " + ", ".join(report["unlisted"][:5]))
        redaction = report.get("redaction")
        if redaction:
            print(f"redaction: {redaction['statement']}")
        if not report["identical"]:
            print(f"{len(report['failures'])} FAILURES")
        elif redaction and (redaction["requests"] or redaction["streams"] or redaction["files"]):
            print(f"PACKAGE VERIFIED AS REDACTED: {redaction['requests']} requests and {redaction['streams']} "
                  "streams against their redacted sha256, not the sealed run's")
        else:
            print("PACKAGE VERIFIED")
        return 0 if report["identical"] else 1
    if args.command == "calls":
        print(f"wrote {inline_calls(args.root, args.out)} samples to {args.out}")
    elif args.command == "images":
        print(f"wrote {dump_images(args.root, args.directory)} images to {args.directory}")
    else:
        samples = speculator(args.root, args.out, per_epoch=args.per_epoch)
        print(f"wrote {len(samples)} rows to {args.out}: " + ", ".join(
            f"{len(item['input_ids'])} ids / {sum(item['loss_mask'])} trained" for item in samples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
