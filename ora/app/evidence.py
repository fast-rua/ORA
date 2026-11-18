"""A run's evidence: ``ora evaluate``, ``seal``, ``revalidate``, ``export``, ``export-sft``,
``metadata``, ``state``, ``metrics``, ``ownership`` and ``trace-validate``.

Each command but ``revalidate`` takes one run ID; ``revalidate`` reviews completed
runs into separate reports and never changes the originals. The evaluation and
sealing are ``orchestrator.evaluation.evaluate``'s; this module prints them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

RUN_COMMANDS = {
    "metadata": "print one run's context and environment manifest",
    "state": "verify and print the latest or one saved public policy state",
    "metrics": "rebuild and print deterministic Runtime metrics",
    "ownership": "rebuild and print the behavioral-ownership audit",
    "evaluate": "deterministically evaluate one completed trace",
    "seal": "make an evaluated run inventory immutable",
    "export": "archive one sealed run for transfer",
    "trace-validate": "validate one canonical agent trace and artifact graph",
    "export-sft": "export native model calls as OpenAI tool-call JSONL",
}
COMMANDS = ("revalidate", "export-traces", *RUN_COMMANDS)


def register(sub: Any) -> None:
    revalidate = sub.add_parser("revalidate", help="review completed runs into separate reports without changing originals")
    revalidate.add_argument("run_dirs", nargs="*", type=Path)
    revalidate.add_argument("--run-list", type=Path, help="JSON list of absolute run directories")
    revalidate.add_argument("--output-dir", required=True, type=Path)
    revalidate.add_argument("--workers", type=int, default=2)
    traces = sub.add_parser(
        "export-traces",
        help="export many runs as one trace dataset: one record per episode, each image once, an index and a reader")
    traces.add_argument("--runs", required=True, type=Path,
                        help="the run list: one run directory per line, or one JSON object per line with 'run' and "
                             "optionally 'episode_id', 'model', 'benchmark' and 'labels'")
    traces.add_argument("--destination", required=True, type=Path)
    traces.add_argument("--work", type=Path, help="where runs are exported before the dataset is assembled "
                                                  "(default: <destination>.work); an interrupted export resumes from it")
    traces.add_argument("--workers", type=int, default=4)
    traces.add_argument("--keep-host-paths", action="store_true",
                        help="do not replace host paths by placeholders (the default replaces them, as "
                             "export-sft --redact-host-paths does)")
    traces.add_argument("--redact-path", metavar="PREFIX", action="append", default=[],
                        help="one more literal path prefix to redact, as <host-path> (repeatable)")
    traces.add_argument("--forbid", metavar="TEXT", action="append", default=[],
                        help="refuse an episode whose text holds this string (repeatable): a host name, an "
                             "internal name")
    traces.add_argument("--no-compress", action="store_true", help="write episodes as .jsonl, not .jsonl.gz")
    traces.add_argument("--shard-mb", type=int, default=512, help="uncompressed size of one episode shard")
    for name, help_text in RUN_COMMANDS.items():
        command = sub.add_parser(name, help=help_text)
        command.add_argument("run_id")
    sub.choices["state"].add_argument("--step", type=int)
    sub.choices["seal"].add_argument(
        "--backfill", action="store_true",
        help="seal an old unsealed run with today's evaluator, pinning its "
             "recorded source identity and marking the seal as a backfill",
    )
    sub.choices["export"].add_argument("--destination", type=Path)
    sub.choices["export-sft"].add_argument("--destination", type=Path)
    sub.choices["export-sft"].add_argument(
        "--view", choices=("sft", "wire", "openai"), default="sft",
        help="sft: one training record per accepted call (default); wire: every call as sent, "
             "each image once, rebuilt and checked against the archived requests; openai: a self-contained "
             "package of standard OpenAI conversations with the wire export, a README and a verify tool",
    )
    sub.choices["export-sft"].add_argument(
        "--check", action="store_true",
        help="with --view wire or openai: check an existing export at --destination instead of writing one",
    )
    sub.choices["export-sft"].add_argument(
        "--tokenize-url", metavar="URL",
        help="with --view wire or openai: the serving engine's tokenize endpoint (vLLM <root>/tokenize, sglang "
             "<root>/v1/tokenize); each policy call's prompt token ids are taken from it and checked "
             "against the call's usage",
    )
    sub.choices["export-sft"].add_argument(
        "--tokenize-key-env", metavar="NAME",
        help="with --tokenize-url: the environment variable holding the endpoint's bearer key",
    )
    sub.choices["export-sft"].add_argument(
        "--redact-host-paths", action="store_true",
        help="with --view wire or openai: replace the host paths that name this machine (the run directory, the "
             "outputs, workspace and checkout roots the run records, the code-mode interpreter's prefixes, the home "
             "directory) by placeholders that keep the rest of the path, such as <host-python>/lib/..., and record "
             "the redaction in the manifest; a redacted request ships no prompt token ids and verifies against the "
             "sha256 recorded at export, not the sealed one",
    )
    sub.choices["export-sft"].add_argument(
        "--redact-path", metavar="PREFIX", action="append", default=[],
        help="with --redact-host-paths: one more literal path prefix to redact, as <host-path> (repeatable); "
             "with --check, the prefixes the export was given",
    )


def _print(value: Any) -> None:
    import json

    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def run(args: argparse.Namespace) -> int:
    from ora.orchestrator import paths
    from ora.orchestrator.leases import _load

    if args.command == "revalidate":
        import json

        from ora.orchestrator.evaluation.evidence_io import EvaluationError
        from ora.orchestrator.evaluation.revalidation import revalidate_runs

        roots = list(args.run_dirs)
        if args.run_list:
            values = json.loads(args.run_list.read_text(encoding="utf-8"))
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise EvaluationError("--run-list requires a JSON list of run directory paths")
            roots.extend(Path(value) for value in values)
        _print(revalidate_runs(roots, args.output_dir, workers=args.workers))
        return 0
    if args.command == "export-traces":
        from ora.agent.evidence.trace_dataset import export_dataset, read_entries

        if args.keep_host_paths and args.redact_path:
            raise ValueError("--redact-path redacts; it cannot go with --keep-host-paths")
        report = export_dataset(
            read_entries(args.runs), args.destination, work=args.work, workers=args.workers,
            redact_host_paths=not args.keep_host_paths, redact_paths=args.redact_path, forbid=args.forbid,
            compress=not args.no_compress, shard_bytes=args.shard_mb << 20)
        _print({**report, "refused": len(report["refused"]), "failed": len(report["failed"])})
        return 2 if report["refused"] or report["failed"] else 0
    if args.command == "export-sft" and args.view in ("wire", "openai") and Path(args.run_id).is_dir():
        # A run directory (a link whose name is not the run id) instead of a run id.
        root = Path(args.run_id).resolve()
        if args.destination is None:
            raise ValueError("a run directory needs --destination")
    else:
        root = paths.resolve_run_root(args.run_id)
    if args.command == "metadata":
        _print(_load(root / "context-manifest.json"))
    elif args.command == "state":
        from ora.agent.evidence.policy_state import load_policy_state, verify_policy_state

        verification = verify_policy_state(root)
        if verification["status"] != "valid":
            raise RuntimeError("invalid policy-state evidence: " + ", ".join(verification["errors"]))
        _print(load_policy_state(root, args.step))
    elif args.command == "metrics":
        from ora.orchestrator.evaluation.context_audit import write_context_audit
        from ora.orchestrator.evaluation.metrics import write_runtime_metrics

        write_context_audit(root)
        _print(write_runtime_metrics(root))
    elif args.command == "ownership":
        from ora.orchestrator.evaluation.ownership_audit import write_ownership_audit

        _print(write_ownership_audit(root))
    elif args.command == "evaluate":
        from ora.orchestrator.evaluation.evaluate import evaluate

        _print(evaluate(root))
    elif args.command == "seal":
        from ora.orchestrator.evaluation.evaluate import seal

        from ora.agent.evidence import sandbox_retention

        _print(seal(root, mode="backfill" if args.backfill else "operator"))
        omission = sandbox_retention.omit_after_seal(root)  # save_sandbox: false keeps only hashes
        if omission["status"] in {"omitted", "refused"}:
            print(f"sandbox files: {omission}", file=sys.stderr)
    elif args.command == "export":
        from ora.orchestrator.evaluation.evaluate import export

        destination = args.destination or (paths.workspace_for_run(root) / "exports" / f"{args.run_id}.tar.gz")
        print(export(root, destination))
    elif args.command == "trace-validate":
        from ora.agent.evidence.trace_verify import verify_agent_trace

        verification = verify_agent_trace(root)
        _print(verification)
        return 0 if verification["status"] == "valid" else 2
    elif args.command == "export-sft" and args.view in ("wire", "openai"):
        from ora.agent.evidence.openai_package import check_openai_package, export_openai
        from ora.agent.evidence.wire_export import check_wire_export, export_wire

        export, check = (export_wire, check_wire_export) if args.view == "wire" else (export_openai, check_openai_package)
        destination = args.destination or (paths.workspace_for_run(root) / "exports" / f"{args.run_id}-{args.view}")
        if args.check:
            if args.tokenize_url:
                raise ValueError("--check reads the recorded token ids; it takes no --tokenize-url")
            if args.redact_host_paths:
                raise ValueError("--check reads the redaction from the export's manifest; it takes no "
                                 "--redact-host-paths (give --redact-path again if the export had it)")
            report = check(destination, root, redact_paths=args.redact_path)
            _print(report)
            return 0 if report["identical"] else 2
        if args.redact_path and not args.redact_host_paths:
            raise ValueError("--redact-path needs --redact-host-paths")
        host_paths = None
        if args.redact_host_paths:
            from ora.agent.evidence.host_paths import HostPaths

            host_paths = HostPaths.for_run(root, args.redact_path)
        tokenizer = None
        if args.tokenize_url:
            import os

            from ora.agent.evidence.token_ids import Tokenizer

            key = os.environ.get(args.tokenize_key_env) if args.tokenize_key_env else None
            if args.tokenize_key_env and not key:
                raise ValueError(f"--tokenize-key-env names an unset variable: {args.tokenize_key_env}")
            tokenizer = Tokenizer(args.tokenize_url, key)
        try:
            print(export(root, destination, tokenizer=tokenizer, host_paths=host_paths))
        finally:
            if tokenizer is not None:
                tokenizer.close()
    elif args.command == "export-sft":
        from ora.agent.evidence.sft_export import export_sft

        if args.check or args.tokenize_url or args.redact_host_paths or args.redact_path:
            raise ValueError("--check, --tokenize-url and --redact-host-paths need --view wire or openai")
        destination = args.destination or (paths.workspace_for_run(root) / "exports" / f"{args.run_id}-sft")
        print(export_sft(root, destination))
    return 0


__all__ = ["COMMANDS", "RUN_COMMANDS", "register", "run"]
