"""Replay recorded runs through a context pack and compare each request with reference examples.

The reference examples are a pack owner's record of what a model trained on
the pack must receive at each step: ``decision-examples.jsonl``
(``view.messages`` per sample, ``source.run_id`` and ``source.call_index``
name the recorded request), the ``tools`` of the matching ``swift-*.jsonl``
row, ``rationales.jsonl`` (the rationale each history turn shows) and
``conversion-report.json`` (where each sealed run lives). No model is called:

1. each sample's recorded request is the history the Context pipeline had
   projected at that step; it becomes the pack user's own history -- calls to
   the tools the pack withholds and their results leave it (a turn left
   without calls goes), and each assistant turn says the rationale the
   examples record for it;
2. the static replacements come from the run's own compiled prompt layers
   (``compiled-policy/``) with the composition functions a plan uses
   (``ora.agent.composition.prompt.context_pack``); the tools are the recorded tools
   without the withheld ones, with the pack's overlay;
3. the Pi core's view (``ora-pi/src/context/context_pack.ts``) applies the
   pack, through ``view_driver.ts``; its images are downscaled by the same
   Python function the session's RPC uses;
4. messages and tools are compared with the example, images by the sha256 of
   their bytes.

Samples whose reference view depended on the sample's own target (an older
turn kept whole, a turn's original text kept, a tool result cut later than the
cap: ``target_dependencies``) are skipped: no view can know the next call in
advance.

    python -m tests.support.context_pack.replay --pack DIR --manifest-sha256 HEX \\
        --profile PROFILE.json --examples DIR [--report REPORT.json]
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ora.agent.composition.prompt import context_pack as cp

REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER = Path(__file__).resolve().with_name("view_driver.ts")
#: Reference-view annotations that are not part of the delivered message.
VIEW_ANNOTATIONS = ("view_inserted", "view_promoted")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_sha256(url: str) -> str:
    return hashlib.sha256(base64.b64decode(url.partition(",")[2])).hexdigest()


# ---- the reference examples -------------------------------------------------------------------


@dataclass
class Examples:
    root: Path
    samples: list[dict[str, Any]]
    tools: dict[str, list[dict[str, Any]]]
    rationales: dict[tuple[str, str], str]
    runs: dict[str, Path]

    @classmethod
    def load(cls, root: Path) -> "Examples":
        samples = read_jsonl(root / "decision-examples.jsonl")
        tools: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(root.glob("swift-*.jsonl")):
            for row in read_jsonl(path):
                tools[row["id"]] = json.loads(row["tools"])
        rationales = {(item["run_id"], item["model_call_id"]): item["chosen"]
                      for item in read_jsonl(root / "rationales.jsonl")}
        report = json.loads((root / "conversion-report.json").read_text(encoding="utf-8"))
        runs = {item["run_id"]: Path(item["run"]) for item in report}
        return cls(root=root, samples=samples, tools=tools, rationales=rationales, runs=runs)


def target_dependencies(sample: Mapping[str, Any], cap: int | None) -> list[str]:
    """The reference view's rules that depended on the sample's own target (none: comparable).

    ``promoted_turns``: an older turn kept whole; ``history_text_kept_for_dependency``:
    a turn's original text kept over its rationale; a widened cap: a recent tool
    result cut later than ``cap`` (or not at all) because the target used a value
    past the cut. A view built before the next call is known can do none of them.
    """

    view = sample["view"]
    rules = [name for name in ("promoted_turns", "history_text_kept_for_dependency") if view.get(name)]
    if cap is not None:
        for message in view["messages"]:
            text = message.get("content")
            if message.get("role") == "tool" and isinstance(text, str):
                shown = text.rsplit("\n…[", 1)[0] if text.endswith(" more characters omitted by the view]") else text
                if len(shown) > cap:
                    rules.append("widened_tool_text_cap")
                    break
    return rules


# ---- one recorded run ------------------------------------------------------------------------


def _ora_layer_text(reference: Mapping[str, Any]) -> str:
    """A plan's layer, found in this tree by its file name and sha256 (a run may predate a package move)."""

    for path in [REPO_ROOT / str(reference["path"]),
                 *sorted((REPO_ROOT / "ora/policy").rglob(Path(str(reference["path"])).name))]:
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]:
            return path.read_text(encoding="utf-8")
    raise LookupError(f"no file in this tree has the layer {reference['path']} ({reference['sha256'][:12]})")


def static_delivery(pack: cp.ContextPack, profile: Mapping[str, Any], *, harness_system: str,
                    project_instructions: str, episode_task: str, task_instruction: str,
                    task_template: str | None) -> dict[str, Any]:
    """The static replacements of ``context_pack.delivery`` for these compiled layers.

    ``ora.agent.pi.context_pack_rpc`` computes the same from the session's
    templates and the plan's binding: the system text from the core file and
    the compiled embodiment card (here found inside the project instructions,
    whose last layer it is), the task from ``task_template``.
    """

    messages: list[dict[str, Any]] = []
    if "system" in profile:
        system, _kept = cp.compact_system_text(
            pack.read_text(profile["system"]["core_file"]), project_instructions,
            profile["system"]["embodiment_sections"],
        )
        messages.append({"role": "system", "content_sha256": cp.sha256_text(harness_system.strip()),
                         "content": system})
        messages.append({"role": "user", "content_sha256": cp.sha256_text(project_instructions.strip()),
                         "content": None})
    if profile.get("episode_task") == "task_and_contract_v1":
        original = cp.render_episode_task(episode_task, task_instruction)
        messages.append({"role": "user", "content_sha256": cp.sha256_text(original),
                         "content": cp.render_episode_task(str(task_template), task_instruction)})
    return {"messages": messages}


class RecordedRun:
    """A sealed run's provider requests, compiled prompt layers and plan."""

    def __init__(self, root: Path) -> None:
        from ora.agent.evidence.sft_export import _iter_provider_exchanges
        from ora.agent.evidence.agent_events import read_agent_events

        self.root = root
        self.plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
        self.exchanges = list(_iter_provider_exchanges(read_agent_events(root)))
        self._requests: dict[int, dict[str, Any]] = {}

    def request(self, index: int) -> dict[str, Any]:
        from ora.agent.evidence import request_archive
        from ora.agent.evidence.sft_export import _canonical_provider_request_ref

        if index not in self._requests:
            data = request_archive.verified_bytes(self.root, _canonical_provider_request_ref(self.exchanges[index][0]))
            if data is None:
                raise LookupError(f"{self.root.name}: request {index} is not in its archive")
            self._requests[index] = json.loads(data)
        return self._requests[index]

    def compiled(self, name: str) -> str:
        return (self.root / "compiled-policy" / name).read_text(encoding="utf-8")

    def episode_task_template(self) -> str:
        """``task_and_contract_v1`` from the run's compiled episode task: its template layer becomes ``{{TASK}}``."""

        compiled = self.compiled("episode-task.md").strip()
        template = _ora_layer_text(self.plan["policy_context"]["episode_task"][0]).strip()
        if not compiled.startswith(template):
            raise LookupError(f"{self.root.name}: the compiled episode task does not start with its template layer")
        contract = compiled[len(template):].lstrip("\n")
        return cp.episode_task_template([contract] if contract else [])

    def delivery(self, pack: cp.ContextPack, profile: Mapping[str, Any]) -> dict[str, Any]:
        """``context_pack.delivery`` as the session's RPC computes it, from the run's compiled layers."""

        return static_delivery(
            pack, profile, harness_system=self.compiled("harness-system.md"),
            project_instructions=self.compiled("project-instructions.md"),
            episode_task=self.compiled("episode-task.md"), task_instruction=str(self.plan["task_instruction"]),
            task_template=self.episode_task_template() if profile.get("episode_task") == "task_and_contract_v1" else None,
        )

    def history(self, index: int, withheld: set[str], rationale_of: Mapping[str, str]) -> list[dict[str, Any]]:
        """The recorded request ``index`` as the pack user's own history (see the module docstring)."""

        kept, dropped_ids = [], set()
        for message in copy.deepcopy(self.request(index)["messages"]):
            if message["role"] == "assistant" and message.get("tool_calls"):
                calls = [call for call in message["tool_calls"]
                         if (call.get("function") or {}).get("name") not in withheld]
                dropped_ids |= {call.get("id") for call in message["tool_calls"] if call not in calls}
                if not calls:
                    continue
                message["tool_calls"] = calls
                said = rationale_of.get(calls[0].get("id"))
                if said is not None:
                    message["content"] = said
            if message["role"] == "tool" and message.get("tool_call_id") in dropped_ids:
                continue
            kept.append(message)
        return kept

    def rationales_by_call(self, rationales: Mapping[tuple[str, str], str], withheld: set[str]) -> dict[str, str]:
        """The recorded rationale of each turn, keyed by its first kept tool call id."""

        by_call: dict[str, str] = {}
        for request_event, response_event in self.exchanges:
            message = (response_event.get("payload") or {}).get("message") or {}
            calls = [call for call in message.get("tool_calls") or []
                     if (call.get("function") or {}).get("name") not in withheld]
            chosen = rationales.get((self.root.name, request_event["model_call_id"]))
            if calls and chosen is not None:
                by_call[calls[0].get("id")] = chosen
        return by_call


# ---- the view and the comparison -------------------------------------------------------------


class ImageRefs:
    """Data URLs swapped for short ``ref:`` strings, downscaled here as the session's RPC would.

    The view never reads an image's bytes, so the driver gets ``ref:<sha256 of
    the URL>`` in each URL's place, and the ref of the downscaled URL as the
    ``context_pack.images`` answer; the comparison maps refs back to the
    sha256 of the delivered image bytes.
    """

    def __init__(self, long_edge_px: int | None) -> None:
        self.long_edge_px = long_edge_px
        self.answers: dict[str, str] = {}   # sha256(source ref) -> delivered ref (the driver's lookup)
        self.images: dict[str, str] = {}    # ref -> sha256 of the image bytes

    @staticmethod
    def _ref(url: str) -> str:
        return "ref:" + hashlib.sha256(url.encode()).hexdigest()

    def swap(self, value: Any) -> Any:
        if isinstance(value, dict):
            url = (value.get("image_url") or {}).get("url") if value.get("type") == "image_url" else None
            if isinstance(url, str):
                ref = self._ref(url)
                key = hashlib.sha256(ref.encode()).hexdigest()
                if key not in self.answers:
                    delivered = url if self.long_edge_px is None else cp.downscale_data_url(url, self.long_edge_px)
                    self.answers[key] = self._ref(delivered)
                    self.images[self._ref(delivered)] = image_sha256(delivered)
                    self.images.setdefault(ref, image_sha256(url))
                return {**value, "image_url": {**value["image_url"], "url": ref}}
            return {key: self.swap(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.swap(item) for item in value]
        return value


def run_view(binding: Mapping[str, Any], delivery: Mapping[str, Any], steps: list[dict[str, Any]],
             images: ImageRefs, *, scratch: Path) -> list[dict[str, Any]]:
    """The Pi core's pack view of each step (``view_driver.ts``)."""

    payload = {"binding": binding, "delivery": delivery, "images": images.answers, "steps": steps}
    source, target = scratch / "view-input.json", scratch / "view-output.json"
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(["node", str(DRIVER), str(source), str(target)], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=1800, check=False)
    if result.returncode:
        raise RuntimeError(f"view_driver.ts failed: {result.stderr.strip()[-2000:]}")
    return json.loads(target.read_text(encoding="utf-8"))["steps"]


def normalized_ours(messages: list[dict[str, Any]], images: ImageRefs) -> list[dict[str, Any]]:
    def part(item: Any) -> Any:
        if isinstance(item, dict) and item.get("type") == "image_url":
            return {"type": "image", "sha256": images.images[item["image_url"]["url"]]}
        return item
    return [{**message, "content": [part(item) for item in message["content"]]}
            if isinstance(message.get("content"), list) else message for message in messages]


def normalized_expected(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def part(item: Any) -> Any:
        if isinstance(item, dict) and item.get("type") == "image":
            return {"type": "image", "sha256": item["sha256"]}
        return item
    output = []
    for message in messages:
        message = {key: value for key, value in message.items() if key not in VIEW_ANNOTATIONS}
        if isinstance(message.get("content"), list):
            message["content"] = [part(item) for item in message["content"]]
        output.append(message)
    return output


def first_difference(ours: list[Any], expected: list[Any]) -> str | None:
    if ours == expected:
        return None
    for index, (left, right) in enumerate(zip(ours, expected)):
        if left != right:
            if isinstance(left, dict) and isinstance(right, dict):
                keys = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
                detail = {key: [str(left.get(key))[:240], str(right.get(key))[:240]] for key in keys}
                return f"message {index} ({right.get('role')}): {json.dumps(detail, ensure_ascii=False)[:1200]}"
            return f"message {index}: {str(left)[:300]} != {str(right)[:300]}"
    return f"{len(ours)} messages, expected {len(expected)}"


@dataclass
class Report:
    total: int = 0
    compared: int = 0
    passed: int = 0
    failed: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    skipped_identical: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"total": self.total, "compared": self.compared, "passed": self.passed,
                "failed": len(self.failed), "skipped_dependency_rule": len(self.skipped),
                "skipped_but_identical": self.skipped_identical,
                "failures": self.failed, "skipped": self.skipped}


def replay(pack: cp.ContextPack, profile: Mapping[str, Any], examples: Examples, *,
           runs: Iterable[str] | None = None, scratch: Path | None = None) -> Report:
    """Replay every sample of ``examples`` (or of the named runs) and compare it."""

    profile = cp.validate_profile(profile, pack, label="profile")
    withheld = set(cp.dropped_tools(profile))
    overlay_file = (profile.get("tools") or {}).get("overlay_file")
    overlay = cp.load_tool_overlay(pack, overlay_file) if overlay_file else None
    binding = {"schema": cp.BINDING_SCHEMA, "name": pack.name, "profile": profile}
    view = profile.get("compact_view") or {}
    report = Report()
    wanted = set(runs) if runs is not None else None
    by_run: dict[str, list[dict[str, Any]]] = {}
    for sample in examples.samples:
        if wanted is None or sample["source"]["run_id"] in wanted:
            by_run.setdefault(sample["source"]["run_id"], []).append(sample)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        for run_id, samples in by_run.items():
            run = RecordedRun(examples.runs[run_id])
            rationale_of = run.rationales_by_call(examples.rationales, withheld)
            delivery = run.delivery(pack, profile)
            images = ImageRefs(view.get("image_long_edge_px"))
            steps, tools = [], None
            for sample in samples:
                index = sample["source"]["call_index"]
                recorded = cp.drop_tools(run.request(index).get("tools") or [], sorted(withheld))
                if tools is not None and recorded != tools:
                    raise AssertionError(f"{run_id}: the recorded tools change within the run")
                tools = recorded
                history = run.history(index, withheld, rationale_of)
                steps.append({"id": sample["id"], "messages": images.swap(history), "tools": recorded})
            delivery = {**delivery, "tools": cp.overlay_tools(tools, overlay) if overlay is not None else None}
            outputs = {item["id"]: item for item in run_view(binding, delivery, steps, images, scratch=Path(directory))}
            for sample in samples:
                report.total += 1
                output = outputs[sample["id"]]
                ours = normalized_ours(output["messages"], images)
                expected = normalized_expected(sample["view"]["messages"])
                problems = [problem for problem in (
                    first_difference(ours, expected),
                    None if output["tools"] == examples.tools[sample["id"]] else "tools differ",
                ) if problem]
                entry = {"id": sample["id"], "run_id": run_id, "call_index": sample["source"]["call_index"]}
                rules = target_dependencies(sample, view.get("tool_text_cap_chars"))
                if rules:
                    report.skipped.append({**entry, "rules": rules, "differences": problems})
                    report.skipped_identical += not problems
                    continue
                report.compared += 1
                if problems:
                    report.failed.append({**entry, "differences": problems})
                else:
                    report.passed += 1
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--profile", type=Path, help="the delivery profile, for a manifest without ora_profile")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--run", action="append", help="replay only these run ids")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    pack = cp.load_context_pack(args.pack.resolve(), args.manifest_sha256)
    profile, _source = cp.resolve_profile(
        pack, json.loads(args.profile.read_text(encoding="utf-8")) if args.profile else None)
    report = replay(pack, profile, Examples.load(args.examples), runs=args.run).as_dict()
    if args.report:
        args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("total", "compared", "passed", "failed",
                                                     "skipped_dependency_rule", "skipped_but_identical")}))
    for failure in report["failures"][:20]:
        print(json.dumps(failure, ensure_ascii=False)[:1500])
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
