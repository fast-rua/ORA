"""Host paths in an export: the literal prefixes that name the machine a run was made on, and their placeholders.

``ora export-sft RUN --view wire|openai --redact-host-paths`` (``wire_export.export_wire``) replaces every
occurrence of these prefixes in the text it exports by its rule's placeholder, keeping the rest of the path, so a
traceback still reads ``File "<host-python>/lib/python3.11/json/encoder.py"``. The prefixes are a fixed list
(``RULES``), read from the run's own evidence and the account, each as written and as resolved; nothing that only
looks like a path is touched (the sandbox's ``/work/...`` stays), and no image (a data URL or a base64 image
source). A prefix matches where the next character does not continue a file name, a longer prefix first.

The export records each rule's placeholder and replacements, and each changed row's replacements and the sha256
of its redacted form (``Redaction.record``). It ships nothing derived from the redacted bytes, so that nothing
lets a reader confirm a guessed prefix: no prefix, no hash of one, no sealed sha256 of a changed row. A check
against the run redacts it again with the prefixes found there (``HostPaths.for_run``) and compares.
"""

from __future__ import annotations

import json
import os
import pwd
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .canonical import canonical_bytes, sha256_bytes
from .token_ids import without_token_ids

#: Each rule's placeholder, and where its prefixes come from.
RULES = {
    "run": "<host-run>",              # the run directory
    "outputs": "<host-outputs>",      # launch.json workspace.outputs_root
    "workspace": "<host-workspace>",  # launch.json workspace.workspace_root
    "checkout": "<host-checkout>",    # launch.json workspace.source_worktree, the checkout the run ran from
    "python": "<host-python>",        # the code-mode interpreter preflight.json recorded, and the base prefix its
                                      # pyvenv.cfg names: the prefixes the sandbox's tracebacks carry
    "home": "<host-home>",            # the home directory of the run directory's owner and of this account
    "path": "<host-path>",            # each --redact-path
}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _interpreter(executable: Any) -> list[str]:
    """A recorded interpreter's environment and base prefix (``pyvenv.cfg``'s ``home`` is the base's ``bin``)."""

    if not isinstance(executable, str) or not executable.startswith("/"):
        return []
    prefix = Path(executable).parent.parent
    found = [str(prefix), os.path.dirname(os.path.dirname(os.path.realpath(executable)))]
    try:
        config = (prefix / "pyvenv.cfg").read_text(encoding="utf-8")
    except OSError:
        config = ""
    for line in config.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "home" and value.strip().startswith("/"):
            found.append(os.path.dirname(value.strip()))
    return found


def candidates(run_root: Path, extra: Iterable[str] = ()) -> list[tuple[str, str]]:
    """Every ``(rule, prefix)`` of ``RULES`` for one run, each prefix as written and as resolved."""

    root = Path(run_root)
    workspace = (_json(root / "launch.json") or {}).get("workspace")
    workspace = workspace if isinstance(workspace, Mapping) else {}
    checks = (_json(root / "preflight.json") or {}).get("checks")
    sandbox = next((item.get("detail") for item in checks if isinstance(item, Mapping)
                    and item.get("check") == "code_mode_sandbox"), None) if isinstance(checks, list) else None
    interpreter = (sandbox or {}).get("interpreter") if isinstance(sandbox, Mapping) else None
    homes: list[str] = []
    try:
        homes.append(pwd.getpwuid(root.stat().st_uid).pw_dir)
    except (KeyError, OSError):
        pass
    try:
        homes.append(str(Path.home()))
    except RuntimeError:
        pass
    found = [("run", str(root)), ("outputs", workspace.get("outputs_root")),
             ("workspace", workspace.get("workspace_root")), ("checkout", workspace.get("source_worktree")),
             *(("python", item) for item in _interpreter((interpreter or {}).get("executable")
                                                         if isinstance(interpreter, Mapping) else None)),
             *(("home", item) for item in homes), *(("path", item) for item in extra)]
    out = []
    for name, path in found:
        if isinstance(path, str) and path.startswith("/"):
            written = os.path.normpath(path)
            out += [(name, spelling) for spelling in dict.fromkeys((written, os.path.realpath(written)))]
    return out


def _texts(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        if not value.startswith("data:"):
            yield value
    elif isinstance(value, list):
        for item in value:
            yield from _texts(item)
    elif isinstance(value, Mapping):
        image = value.get("type") == "base64"  # an Anthropic image source: its data is the image
        for key, item in value.items():
            yield from _texts(key)
            if not (image and key == "data"):
                yield from _texts(item)


def _walk(value: Any, sub: Any) -> Any:
    if isinstance(value, str):
        return value if value.startswith("data:") else sub(value)
    if isinstance(value, list):
        return [_walk(item, sub) for item in value]
    if isinstance(value, Mapping):
        image = value.get("type") == "base64"
        return {(_walk(key, sub) if isinstance(key, str) else key): (item if image and key == "data" else _walk(item, sub))
                for key, item in value.items()}
    return value


class HostPaths:
    """Literal host prefixes, each under a rule of ``RULES``, replaced longest first by the rule's placeholder."""

    def __init__(self, prefixes: Iterable[tuple[str, str]]) -> None:
        rules: dict[str, str] = {}
        for name, prefix in prefixes:
            if name not in RULES:
                raise ValueError(f"unknown host-path rule: {name}")
            if prefix.startswith("/") and prefix != "/":
                rules.setdefault(prefix, name)
        #: ``(prefix, rule)``, longest first.
        self.rules = sorted(rules.items(), key=lambda item: (-len(item[0]), item[0]))
        alternatives = "|".join(f"({re.escape(prefix)})" for prefix, _ in self.rules)
        self._pattern = re.compile(f"(?:{alternatives})(?![\\w.-])") if self.rules else None

    @classmethod
    def for_run(cls, run_root: Path, extra: Iterable[str] = ()) -> HostPaths:
        return cls(candidates(run_root, extra))

    def redact(self, value: Any, tally: list[int]) -> Any:
        """``value`` (text, or a JSON value) with every prefix replaced; ``tally[i]`` counts ``rules[i]``'s."""

        if self._pattern is None:
            return value
        pattern = self._pattern

        def replace(match: re.Match[str]) -> str:
            index = (match.lastindex or 1) - 1
            tally[index] += 1
            return RULES[self.rules[index][1]]

        return _walk(value, lambda text: pattern.sub(replace, text))

    def text(self, value: Any) -> tuple[Any, int]:
        """``value`` redacted, and the number of replacements."""

        tally = [0] * len(self.rules)
        return self.redact(value, tally), sum(tally)

    def found(self, value: Any) -> bool:
        """Whether a prefix is left anywhere in ``value``'s text."""

        return self._pattern is not None and any(self._pattern.search(text) for text in _texts(value))


class Redaction:
    """One export's redaction: the replacements by rule, and each row it changed with its redacted sha256."""

    def __init__(self, host_paths: HostPaths) -> None:
        self.host_paths = host_paths
        self.tally = [0] * len(host_paths.rules)
        self.rows: list[dict[str, Any]] = []

    def text(self, value: Any) -> tuple[Any, int]:
        before = sum(self.tally)
        out = self.host_paths.redact(value, self.tally)
        return out, sum(self.tally) - before

    def _changed(self, file: str, call_id: str, field: str, replacements: int, value: Any) -> None:
        self.rows.append({"file": file, "model_call_id": call_id, "field": field, "replacements": replacements,
                          "redacted_sha256": sha256_bytes(canonical_bytes(value))})

    def row(self, file: str, call_id: str, field: str, value: Any) -> tuple[Any, int]:
        """One exported value (a request) redacted, recorded when that changes it; and its replacements."""

        out, replacements = self.text(value)
        if replacements:
            self._changed(file, call_id, field, replacements, out)
        return out, replacements

    def stream(self, file: str, call_id: str, response: Any, *, prompt_redacted: bool) -> tuple[Any, bool]:
        """An archived stream redacted, without the token ids that would contradict the redacted text (the prompt's
        when its request changed, the sampled ones when its own text did); and whether it changed."""

        out, replacements = self.text(response)
        out, dropped = without_token_ids(out, prompt=prompt_redacted, sampled=replacements > 0)
        if replacements or dropped:
            self._changed(file, call_id, "response", replacements, out)
        return out, bool(replacements or dropped)

    def left(self, tables: Mapping[str, list[Mapping[str, Any]]]) -> str | None:
        """Where a prefix is still found after redaction (a reply assembled from chunks that each hold only part
        of it), or None."""

        for name, rows in tables.items():
            for row in rows:
                if self.host_paths.found(row):
                    return f"{name}.jsonl ({row.get('model_call_id')})"
        return None

    def record(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for (_, name), count in zip(self.host_paths.rules, self.tally):
            counts[name] = counts.get(name, 0) + count
        return {
            "rules": [{"rule": name, "placeholder": placeholder, "replacements": counts[name]}
                      for name, placeholder in RULES.items() if name in counts],
            "replacements": sum(self.tally),
            "rows": self.rows,
        }


def statement(record: Mapping[str, Any], calls: int, *, against_run: bool) -> str:
    """What a check of a redacted export can and cannot say."""

    rows = record.get("rows") or []
    if not rows:
        return ("screened for host paths (--redact-host-paths): none found; every request and stream is checked "
                "against its sealed sha256")
    requests = sum(1 for item in rows if item.get("field") == "request")
    streams = sum(1 for item in rows if item.get("field") == "response")
    head = (f"{requests} of {calls} requests and {streams} streams are redacted ({record.get('replacements')} "
            "host-path replacements; manifest redaction.rows): each is checked against the sha256 of its redacted "
            "form recorded at export; their sealed sha256 is not shipped")
    if against_run:
        return head + "; derived again from the sealed run with the recorded rules, every row equals the export"
    return (head + ", so the export alone cannot prove they are what the model was sent; with the sealed run, "
            "check it with --check (which redacts the run again and compares), or export again without "
            "--redact-host-paths for the original bytes")


__all__ = ["RULES", "HostPaths", "Redaction", "candidates", "statement"]
