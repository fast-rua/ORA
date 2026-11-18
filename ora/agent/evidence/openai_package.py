"""A run as a self-contained OpenAI Chat Completions package: conversations, calls, token ids, proof and a tool.

``export_openai`` writes, for one run:

- ``conversations.jsonl``: one standard OpenAI conversation per epoch (the calls whose requests extend each
  other): the epoch's last request with every image inline as a data URL, then that call's reply;
- ``calls.jsonl``: every model call, the request as sent (images as ``wire/images/...``), the reply, usage and
  token counts;
- ``wire/``: the wire export it is derived from (``wire_export.export_wire``), with the token ids and the
  replay check against the run's archived hashes;
- ``replay-program/``: the run's actions as code, when the run has one (``ora replay RUN --check``);
- ``ora_package.py``: ``openai_package_tool``, which verifies the package and converts it with the standard
  library only;
- ``README.md`` and ``package.json`` (``ora.openai-package.v1``: the sha256 of every file outside ``wire/``
  and of ``wire/manifest.json``).

With ``host_paths`` (``--redact-host-paths``) the wire export is redacted (``wire_export``), and so are the
README and the replay program's text; ``package.json``'s ``redaction`` counts their replacements by file, and
the README says what was redacted and what ``verify`` can then prove.

The package is written to a staging directory and moved into place only when ``verify`` passes.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import openai_package_tool as tool
from .canonical import TrajectoryError, file_sha256
from .host_paths import HostPaths
from .token_ids import Tokenizer
from .wire_export import _open_permissions, check_wire_export, export_wire

README = """# {run_id}

One ORA run as standard OpenAI Chat Completions data: what the model received on every call, what it
replied, every image, and the token ids, with a proof that it is byte for byte what was sent{proof_scope}.

| | |
| --- | --- |
| Environment / task | {environment} `{task}`, seed {seed}: "{instruction}" |
| Result | official success {official_success}; learning eligible {learning_eligible} |
| Mode | {mode} |
| Model | `{model}`, reasoning effort `{reasoning_effort}` |
| Size | {calls} model calls, {epochs} epochs, {images} images |
| Token ids | prompt ids on {prompted} of {calls} calls{prompt_sources}; sampled ids on {sampled} of {calls} calls |{host_paths}
| Source | run at ORA `{source_commit}`; exported as `{schema}` |

## Files

| Path | Contents |
| --- | --- |
| `conversations.jsonl` | One conversation per epoch: `{{run_id, epoch, model_call_ids, messages, tools}}`, every image inline as a data URL; the last message is the epoch's final reply |
| `calls.jsonl` | One line per model call: `request` as sent (images as `wire/images/<sha256>.<ext>`), `response`, `finish_reason`, `usage`, `token_counts` |
| `wire/` | The wire export: `requests.jsonl`, `responses.jsonl`, `streams.jsonl` (the engine's own stream), `tokens.jsonl`, `epochs.jsonl`, `images/`, `manifest.json` with the replay check |
| `replay-program/` | The run's actions as code, for `ora replay RUN --check` (when the run has one) |
| `ora_package.py` | Verify and convert this package; Python 3 standard library only |
| `package.json` | The sha256 of every file above outside `wire/`, and of `wire/manifest.json` |

An epoch ends where ORA rewrote the context (older images become `<recoverable_artifact>` pointers the
model can reopen); each conversation is exactly what the model saw.

## Use

```bash
python3 ora_package.py verify                      # every file, image, request, stream and token id
python3 ora_package.py calls calls.openai.jsonl    # one OpenAI sample per call, images inline
python3 ora_package.py images images/              # every image as <sha256>.<ext>
python3 ora_package.py speculator spec.jsonl       # {{input_ids, loss_mask}} rows (needs sampled ids)
```

```python
import json
for line in open("conversations.jsonl"):
    conversation = json.loads(line)
    messages, tools = conversation["messages"], conversation["tools"]
```

Differences from textbook OpenAI data:

- `reasoning_content` carries each assistant turn's reasoning (a DeepSeek and vLLM convention, not an
  OpenAI field); the history the model saw kept it.
- Tool results are text; the images a tool produced arrive in the next `user` message.
- Each tool call's arguments are the JSON the engine's tool-call parser produced, sent back unchanged in the
  history. vLLM coerces values to the tool schema, so a number the model wrote as `0.0` can read `0`; the
  sampled tokens in `wire/tokens.jsonl` are what the model wrote.
- Request options are kept as sent (`max_tokens`, `stream`, `reasoning_effort`, ...).

## Replay

1. **Bytes.** `python3 ora_package.py verify` puts every image back into every request, serializes it as the
   run archived it (sorted keys, compact, UTF-8) and requires the archived sha256, from the request's own line
   and from its epoch; it checks every stream, every token id against its sha256, the engine's usage and the
   stream, and `calls.jsonl` and `conversations.jsonl` against `wire/`.
2. **Tokens.** `python3 ora_package.py verify --tokenize-url URL --key-env NAME` has a serving engine of the
   same model tokenize every request again (vLLM `<root>/tokenize`, sglang `<root>/v1/tokenize`) and compares
   the sequences. Replies are not sampled again; train on the recorded reply.
3. **Actions.** `ora replay RUN --check OUT` re-executes `replay-program/` on the run's simulator and compares
   every action's end pose bit for bit and the official verdict.

`package.json` proves the package is consistent with itself. To tie it to the sealed run, check it against
the run directory: `ora export-sft RUN --view openai --destination THIS_DIR --check`.
{redaction}"""

REDACTION = """
## Redaction

Host filesystem paths that name the machine the run was made on were replaced by placeholders that keep the
rest of the path (`<host-python>/lib/python3.11/json/encoder.py`): {replacements} replacements in {requests} of
{calls} requests and {streams} streams, by the rules in `wire/manifest.json` `redaction` ({rules}). Every other
byte is as sent, images included. A redacted request carries no prompt token ids, and a stream whose text
changed no sampled ids: they were the original text's. Nothing derived from the redacted bytes is shipped
(no prefix, no hash of one, and for a redacted row no sealed sha256, nor the trace's final event hash), so
nothing here confirms a guessed prefix.

`verify` checks each redacted row against the sha256 of its redacted form recorded at export
(`wire/manifest.json` `redaction.rows`) and every other row against its sealed sha256. For the redacted rows
this package alone cannot prove that they are what the model was sent.
Whoever has the sealed run can: `ora export-sft RUN --view openai --destination THIS_DIR --check` redacts the
run again and compares, and an export without `--redact-host-paths` gives the original bytes.
"""


def _redaction_text(manifest: Mapping[str, Any], files: Mapping[str, int] | None) -> tuple[str, str, str]:
    """The README's proof scope, table row and section for a redacted package (all empty without redaction)."""

    record = manifest.get("redaction")
    if record is None:
        return "", "", ""
    rows = record.get("rows") or []
    requests = sum(1 for item in rows if item["field"] == "request")
    streams = sum(1 for item in rows if item["field"] == "response")
    others = sum((files or {}).values())
    if not rows and not others:
        return "", "\n| Host paths | screened with `--redact-host-paths`: none found |", ""
    rules = ", ".join(f"`{item['placeholder']}` {item['replacements']}" for item in record["rules"]
                      if item["replacements"]) or "none in wire/"
    extra = f"; {others} in " + ", ".join(f"`{path}`" for path, count in (files or {}).items() if count) if others else ""
    row = (f"\n| Host paths | redacted: {record['replacements']} replacements in {requests} of {manifest['calls']} "
           f"requests and {streams} streams ({rules}){extra}; see Redaction |")
    section = REDACTION.format(replacements=record["replacements"], requests=requests, calls=manifest["calls"],
                               streams=streams, rules=rules)
    return ", except for the host paths it redacts (see Redaction)", row, section


def _readme(manifest: Mapping[str, Any], plan: Mapping[str, Any], files: Mapping[str, int] | None = None) -> str:
    policy = plan.get("policy") or {}
    tokens = manifest.get("tokens") or {}
    sources = {key: value for key, value in (tokens.get("prompt_ids") or {}).items() if key != "none"}
    prompted = sum(sources.values())
    described = (" (" + ", ".join(f"{value} from the {key}" for key, value in sources.items()) + ")" if prompted
                 else " (export with --tokenize-url to add them)")
    proof_scope, host_paths, redaction = _redaction_text(manifest, files)
    return README.format(
        proof_scope=proof_scope, host_paths=host_paths, redaction=redaction,
        run_id=manifest["run_id"], environment=(plan.get("runtime") or {}).get("kind", "?"), task=plan.get("task", "?"),
        seed=plan.get("seed", "?"), instruction=str(plan.get("task_instruction") or "").replace("\n", " "),
        official_success=manifest["official_success"], learning_eligible=manifest["learning_eligible"],
        mode="code mode (scripts call the robot library)" if policy.get("code_mode") else "tool calls",
        model=policy.get("model", "?"), reasoning_effort=policy.get("reasoning_effort", "?"),
        calls=manifest["calls"], epochs=manifest["epochs"], images=manifest["images"]["files"],
        prompted=prompted, prompt_sources=described, sampled=tokens.get("output_ids", 0),
        source_commit=str(manifest.get("source_commit") or "?")[:9], schema=tool.PACKAGE_SCHEMA,
    )


def _redact_file(host_paths: HostPaths, path: Path) -> int:
    """Redact one UTF-8 file of the package in place; its replacements."""

    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        raise TrajectoryError(f"{path.name} is not text; its host paths cannot be redacted") from None
    redacted, replacements = host_paths.text(text)
    if replacements:
        path.write_bytes(redacted.encode("utf-8"))
    return replacements


def export_openai(run_root: Path, destination: Path, *, tokenizer: Tokenizer | None = None,
                  host_paths: HostPaths | None = None) -> Path:
    """Write the OpenAI package of one run (see the module docstring) and verify it before moving it into place."""

    root = run_root.resolve()
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise TrajectoryError(f"export destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        export_wire(root, staging / "wire", tokenizer=tokenizer, host_paths=host_paths)
        conversations, calls = tool.derive(staging)
        tool.write_jsonl(staging / "conversations.jsonl", conversations)
        tool.write_jsonl(staging / "calls.jsonl", calls)
        redacted: dict[str, int] | None = None if host_paths is None else {}
        if (root / "replay-program").is_dir():
            shutil.copytree(root / "replay-program", staging / "replay-program")
            if host_paths is not None and redacted is not None:
                for path in sorted(item for item in (staging / "replay-program").rglob("*") if item.is_file()):
                    redacted[str(path.relative_to(staging))] = _redact_file(host_paths, path)
        shutil.copyfile(Path(tool.__file__), staging / tool.TOOL_FILE)
        manifest = json.loads((staging / "wire" / "manifest.json").read_text(encoding="utf-8"))
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
        (staging / "README.md").write_text(_readme(manifest, plan, redacted), encoding="utf-8")
        if host_paths is not None and redacted is not None:
            redacted["README.md"] = _redact_file(host_paths, staging / "README.md")
        files = sorted(path for path in staging.rglob("*") if path.is_file() and path.relative_to(staging).parts[0] != "wire")
        (staging / tool.PACKAGE_FILE).write_text(json.dumps({
            "schema": tool.PACKAGE_SCHEMA, "run_id": manifest["run_id"],
            "wire_manifest_sha256": file_sha256(staging / "wire" / "manifest.json"),
            "files": [{"path": str(path.relative_to(staging)), "sha256": file_sha256(path), "bytes": path.stat().st_size}
                      for path in files],
            **({} if redacted is None else {"redaction": {"files": dict(sorted(redacted.items()))}}),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = tool.verify(staging)
        if not report["identical"]:
            raise TrajectoryError("OpenAI package does not verify: " + json.dumps(report["failures"][:3]))
        _open_permissions(staging)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def check_openai_package(package_root: Path, run_root: Path | None = None, *,
                         redact_paths: Iterable[str] = ()) -> dict[str, Any]:
    """``openai_package_tool.verify``; with ``run_root`` also the wire export derived again from the run
    (``redact_paths``: the ``--redact-path`` prefixes a redacted export was given)."""

    report = tool.verify(Path(package_root))
    if run_root is not None:
        wire = check_wire_export(Path(package_root) / "wire", run_root, redact_paths=redact_paths)
        report["wire"] = {key: value for key, value in wire.items() if key not in ("failures", "redaction")}
        report["failures"] = report["failures"] + [{"check": f"wire_{item['check']}", "detail": str(item.get("detail") or
                                                     item.get("model_call_id") or "")} for item in wire["failures"]]
        report["identical"] = not report["failures"]
        if "redaction" in report and "redaction" in wire:  # what the run, unlike the package alone, proves
            report["redaction"]["statement"] = wire["redaction"]
    return report


__all__ = ["check_openai_package", "export_openai"]
