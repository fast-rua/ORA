"""Sealed acceptance runs that the opt-in agent-core checks replay.

The runs live outside the repository. Each environment's run is named by one
environment variable holding either the sealed run directory
(``agent-events.jsonl`` plus ``native/provider-requests.tar.zst``) or a replay
fixture extracted from it (``fixture.json``, see ``replay/extract.py``):

==========================  ====================================================
``ORA_TEST_SEALED_LIBERO``    LIBERO-Pro default cell, 37 provider calls (2026-10-13)
``ORA_TEST_SEALED_ROBOTWIN``  RoboTwin ``adjust_bottle`` demo_clean s100001, 17 calls
``ORA_TEST_SEALED_ROBODOJO``  RoboDojo ``stack_bowls`` layout 0, 96 calls
==========================  ====================================================

A check whose run is unset or unusable is skipped with the reason; it never
passes without the evidence. Extraction (it needs ``zstd``) goes to a fresh
temporary directory unless ``ORA_TEST_SEALED_CACHE`` names one to keep.

    python -m tests.support.agent_core.sealed requests ENV CACHE  # canonical request files (JSON list)
    python -m tests.support.agent_core.sealed archive ENV CACHE   # sealed run and its extracted request archive
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sys
from pathlib import Path
from typing import Iterator

ENVIRONMENTS = {
    "libero": "ORA_TEST_SEALED_LIBERO",
    "robotwin": "ORA_TEST_SEALED_ROBOTWIN",
    "robodojo": "ORA_TEST_SEALED_ROBODOJO",
}


class Unavailable(Exception):
    """The sealed run is not configured or not usable here; the message is the skip reason."""


def _is_fixture(path: Path) -> bool:
    return (path / "fixture.json").is_file()


def _is_sealed_run(path: Path) -> bool:
    return (path / "agent-events.jsonl").is_file() and (path / "seal.json").is_file()


def source(env: str) -> Path:
    """The configured sealed run or replay fixture of ``env``; ``Unavailable`` otherwise."""

    variable = ENVIRONMENTS[env]
    value = os.environ.get(variable)
    if not value:
        raise Unavailable(f"{variable} is not set (the sealed {env} run or its replay fixture "
                          "lives outside the repository)")
    path = Path(value).resolve()
    if not (_is_fixture(path) or _is_sealed_run(path)):
        raise Unavailable(f"{variable}={value} is neither a sealed run nor a replay fixture")
    return path


def sealed_run(env: str) -> Path:
    """The sealed run directory itself (a replay fixture names its source run)."""

    path = source(env)
    if _is_fixture(path):
        recorded = Path(json.loads((path / "fixture.json").read_text(encoding="utf-8"))["source_run"])
        if not _is_sealed_run(recorded):
            raise Unavailable(f"{ENVIRONMENTS[env]} names a replay fixture whose source run "
                              f"{recorded} is not present; this check reads the sealed run")
        return recorded
    return path


@contextlib.contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """Serialize extraction into one cache directory (test files run in parallel)."""

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def replay_fixture(env: str, cache: Path) -> Path:
    """A replay fixture of ``env``: the configured one, or extracted once into ``cache``."""

    from tests.support.agent_core.replay.extract import extract_fixture

    path = source(env)
    if _is_fixture(path):
        return path
    destination = Path(cache) / env / "fixture"
    with _locked(Path(cache) / env):
        manifest = destination / "fixture.json"
        if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8"))["source_run"] == str(path):
            return destination
        extract_fixture(path, destination, archive_root=Path(cache) / env / "archive", overwrite=True)
    return destination


def request_archive(env: str, cache: Path) -> tuple[Path, Path]:
    """The sealed run and the directory its provider-request archive is extracted into."""

    from tests.support.agent_core.replay.extract import SealedRun, extract_request_archive

    run = sealed_run(env)
    archive = Path(cache) / env / "archive"
    with _locked(Path(cache) / env):
        extract_request_archive(SealedRun(run), archive)
    return run, archive


def canonical_requests(env: str, cache: Path) -> list[Path]:
    """Every archived canonical provider request of ``env``, in call order."""

    from tests.support.agent_core.replay.runtime import Fixture

    path = source(env)
    if _is_fixture(path):
        fixture = Fixture.load(path)
        return [fixture.path(call["canonical_request"]["path"]) for call in fixture.provider_calls]
    run, archive = request_archive(env, cache)
    events = [json.loads(line) for line in (run / "agent-events.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    refs = [event["payload"]["canonical_request_ref"]["path"] for event in events
            if event["type"] == "provider_request"]
    return [run / ref if (run / ref).is_file() else archive / ref for ref in refs]


def main(argv: list[str]) -> int:
    command, env, cache = argv[1], argv[2], Path(argv[3])
    try:
        if command == "requests":
            answer: object = {"paths": [str(path) for path in canonical_requests(env, cache)]}
        elif command == "archive":
            run, archive = request_archive(env, cache)
            answer = {"run": str(run), "archive": str(archive)}
        else:
            raise SystemExit(f"unknown command: {command}")
    except Unavailable as exc:
        answer = {"skip": str(exc)}
    print(json.dumps(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
