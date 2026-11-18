"""Synthetic episodes on the Pi agent core, for the agent-core tests and their goldens.

Each run names its core explicitly (``ORA_AGENT_CORE=pi``), never through the
default, pins the policy-session id, and reports the core that actually ran
(``ran_core``). The Python Native core these episodes were once compared with
left ORA; ``tests/support/agent_core/golden.py`` holds what it sent.

    python -m tests.support.agent_core.episodes OUT_DIR [--actions N] [--composition native_slots_v4]
        [--retention action_step_lossless_v4|rewrite_compact_v1]

prints the run's outcome and wall-clock time for a manual or long-episode run.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import time
import types
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[3]
CORES = ("pi",)
# The remaining wall-time budget is a clock reading, not core behaviour.
WALL_TIME = re.compile(rb'\\"wall_time_s\\": [0-9.e+-]+')


def normalize_wall_time(data: bytes) -> bytes:
    return WALL_TIME.sub(rb'\\"wall_time_s\\": 0', data)


def tool_reply(*calls: tuple[str, str, Mapping[str, Any]], usage: Mapping[str, int] | None = None,
               reasoning: str | None = None) -> dict[str, Any]:
    """One scripted assistant reply calling ``(call_id, name, arguments)`` in order."""

    message: dict[str, Any] = {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments, sort_keys=True)}}
        for call_id, name, arguments in calls
    ]}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"message": message, "usage": dict(usage or {"input_tokens": 4000, "output_tokens": 40})}


def action_script(actions: int) -> list[dict[str, Any]]:
    """Per action: one read-only observe, then one atomic mutation; finish at the end."""

    responses: list[dict[str, Any]] = []
    for index in range(actions):
        responses.append(tool_reply((f"r{index}", "observe", {"views": ["agentview"]}),
                                    usage={"input_tokens": 4000 + 900 * index, "output_tokens": 40}))
        name = "close_gripper" if index % 2 == 0 else "open_gripper"
        responses.append(tool_reply((f"m{index}", name, {}),
                                    usage={"input_tokens": 4400 + 900 * index, "output_tokens": 30}))
    responses.append({"message": tool_reply(("f0", "finish_episode", {
        "outcome": "failure", "reason": "scripted end", "attempts": ["scripted"]}))["message"]})
    return responses


def management(stride: int = 4) -> dict[str, Any]:
    return {
        "schema": "ora.native-context-management.v6",
        "causal_exact_tail_action_steps": 3, "inline_media_tail_action_steps": 2,
        "eager_large_result_bytes": 50000, "eager_preview_bytes": 2048,
        "eager_exact_deduplication": True, "trigger_inline_images": 24,
        "trigger_estimated_tokens": 48000, "trigger_provider_input_ratio": 0.7,
        "episode_checkpoint_trigger_tokens": 150000, "emergency_fallback_ratio": 0.8,
        "emergency_target_ratio": 0.65, "checkpoint_target_estimated_tokens": 1200,
        "checkpoint_max_estimated_tokens": 1600, "tail_aging_stride_action_steps": stride,
    }


class _FixedUuid:
    """The Pi core mints ``native-policy-<uuid4 hex>``; the goldens pin it."""

    hex = "0" * 31 + "1"


@contextlib.contextmanager
def pinned_session_ids() -> Iterator[None]:
    """Pin the policy-session id the Pi core mints; restore the module afterwards."""

    import ora.agent.pi.policy_session as pi_session

    modules = [pi_session]
    originals = {module: module.uuid for module in modules}
    for module, original in originals.items():
        shim = types.SimpleNamespace(**{name: getattr(original, name) for name in dir(original)
                                        if not name.startswith("__")})
        shim.uuid4 = lambda: _FixedUuid()
        module.uuid = shim
    try:
        yield
    finally:
        for module, original in originals.items():
            module.uuid = original


@contextlib.contextmanager
def environment(**values: str | None) -> Iterator[None]:
    """Set (or, for ``None``, remove) environment variables; restore them afterwards."""

    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def pinned_wall_time(seconds: float = 3000.125) -> Iterator[None]:
    """Pin the remaining wall-time budget the policy sees (a clock reading), so two
    runs' observations, requests and every event agree field for field."""

    from ora.orchestrator.episode import observation as episode_observation

    original = episode_observation._remaining_budget

    def pinned(loop: Any) -> dict[str, Any]:
        remaining = original(loop)
        remaining["wall_time_s"] = seconds
        return remaining

    episode_observation._remaining_budget = pinned
    try:
        yield
    finally:
        episode_observation._remaining_budget = original


def ran_core(run_root: Path) -> str:
    """The core that actually ran: only the Pi core starts ``native/pi-core.log``."""

    observed = "pi" if (run_root / "native" / "pi-core.log").exists() else "native"
    session = run_root / "native" / "agent-session.json"
    recorded = json.loads(session.read_text(encoding="utf-8")).get("agent_core") if session.is_file() else None
    if recorded is not None and recorded != observed:
        raise AssertionError(f"{run_root}: agent-session.json names {recorded!r}, the run used {observed!r}")
    return observed


def synthetic_plan(*, actions: int, composition: str = "native_slots_v4",
                   retention: str = "action_step_lossless_v4",
                   policy_overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The Synthetic test cell with budgets for ``actions`` and the chosen Context.

    ``policy_overrides`` are policy fields set before resolution, so the plan
    resolves (and binds) them as a RunSpec's own.
    """

    from tests.support.orchestrator.native_synthetic_spec import synthetic_spec
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    spec = synthetic_spec(seed=0)
    spec["runtime"]["terminal_actions"] = actions + 5
    spec["budgets"] = {"max_robot_decisions": actions + 5, "max_model_calls": 2 * actions + 10,
                       "max_sim_steps": actions + 5, "wall_time_s": 3600}
    spec["policy"]["context_retention"] = retention
    if retention == "action_step_lossless_v4":
        spec["policy"]["summary_policy"] = "rollout_checkpoint_v1"
        spec["policy"]["context_management"] = management()
    spec["policy"].update(policy_overrides or {})
    plan = resolve_run_spec(spec, root=ROOT)
    plan["policy"]["context_composition"] = composition
    return plan


def canonical_requests(root: Path, events: list[dict[str, Any]]) -> list[bytes]:
    return [normalize_wall_time((root / event["payload"]["canonical_request_ref"]["path"]).read_bytes())
            for event in events if event["type"] == "provider_request"]


def run_core(core: str, out: Path, *, responses: list[dict[str, Any]] | None = None, actions: int = 6,
             composition: str = "native_slots_v4", retention: str = "action_step_lossless_v4",
             configure: Callable[[dict[str, Any]], Mapping[str, str | None]] | None = None,
             provider: Any = None, run_id: str = "synthetic-compare",
             policy_overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One Synthetic episode on ``core`` in ``out/core``.

    ``responses`` is the FakeProvider script (default ``action_script``), or
    ``provider`` a FakeProvider-like override built by the test (the Pi core
    takes its ``responses`` or ``pi_provider_spec``); pass ``configure``
    instead to point the plan at a real provider: it edits the plan and returns
    the environment the provider reads (``None`` removes a variable).
    ``policy_overrides`` go to ``synthetic_plan``.
    """

    if core not in CORES:
        raise ValueError(f"unknown agent core: {core}")
    from ora.agent.providers.fake import FakeProvider
    from ora.orchestrator.episode.coordinator import run_episode
    from ora.orchestrator.evaluation.evaluate import evaluate

    with environment(ORA_AGENT_CORE=core), pinned_session_ids():
        plan = synthetic_plan(actions=actions, composition=composition, retention=retention,
                              policy_overrides=policy_overrides)
        provider_environment = dict(configure(plan)) if configure is not None else {}
        if provider is None and configure is None:
            provider = FakeProvider(responses=action_script(actions) if responses is None else responses)
        root = out / core
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        (root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
        started = time.perf_counter()
        with environment(**provider_environment):
            result = run_episode(run_root=root, plan=plan, run_id=run_id, provider_override=provider)
        elapsed = time.perf_counter() - started
    evaluation = evaluate(root)
    events = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    return {"core": ran_core(root), "result": result, "evaluation": evaluation, "events": events,
            "requests": canonical_requests(root, events), "elapsed_s": elapsed, "root": root}


def first_difference(a: Any, b: Any, path: str = "") -> str | None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key}: present left={key in a} right={key in b}"
            found = first_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for index, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{path}[{index}]")
            if found:
                return found
        return None
    if a != b:
        return f"{path}: {str(a)[:160]!r} vs {str(b)[:160]!r}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--actions", type=int, default=6)
    parser.add_argument("--composition", default="native_slots_v4")
    parser.add_argument("--retention", default="action_step_lossless_v4",
                        choices=("action_step_lossless_v4", "rewrite_compact_v1"))
    args = parser.parse_args()
    run = run_core("pi", args.out, actions=args.actions, composition=args.composition, retention=args.retention)
    print(run["core"], run["result"]["state"], run["evaluation"]["status"], run["evaluation"].get("reasons"),
          len(run["requests"]), round(run["elapsed_s"], 3))


if __name__ == "__main__":
    main()
