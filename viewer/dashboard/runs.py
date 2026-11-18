"""Run parsing for the run dashboard: one run's summary and timeline, cohorts and conditions.

Ported from Shaoxiong Duan's code-vs-tools dashboard (snapshot 2026-10-31). The code-vs-tools round
manager and its arm rule are replaced by the plan's own fields: a run's cohort is its launch admission's
``cohort_id`` (else the plan's ``config`` without the task and seed, else the experiment) and its
condition is the value at a plan path (default: code mode on or off). Every file a run may lack is
optional, so runs of every backend open.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

from viewer.dashboard.config import Config
from viewer.dashboard.store import (
    cached_file, first_line_json, jsonl, now, read_json, read_json_proj, stat_sig, tail_text, trunc, utc,
)

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
BACKEND_NAMES = {"robodojo": "RoboDojo", "robotwin": "RoboTwin", "libero_pro": "LIBERO-Pro",
                 "robocasa": "RoboCasa365", "robolab": "RoboLab", "synthetic": "Synthetic"}
#: The default condition: code mode on or off.
CODE, TOOLS = "code", "tools"
#: The pseudo-cohort holding every run of an experiment.
ALL_COHORTS = "*"


# ----------------------------------------------------------------------------- where runs are

def run_dirs(cfg: Config) -> list[tuple[str, str]]:
    """(experiment, run directory) for every run directory under the configured experiments."""
    out = []
    for name, path in cfg.experiments().items():
        runs = os.path.join(path, "runs")
        try:
            ids = sorted(n for n in os.listdir(runs) if os.path.isdir(os.path.join(runs, n)))
        except OSError:
            continue
        out.extend((name, os.path.join(runs, n)) for n in ids)
    return out


def find_run_dir(cfg: Config, run_id: str | None) -> str | None:
    if not run_id or not RUN_ID_RE.match(run_id) or ".." in run_id:
        return None
    for path in cfg.experiments().values():
        cand = os.path.join(path, "runs", run_id)
        if os.path.isdir(cand):
            return cand
    return None


def safe_path(cfg: Config, p: Any) -> str | None:
    """An absolute path inside an allowed root (after resolving links), else None."""
    if not p or not isinstance(p, str) or "\x00" in p or ".." in p or not os.path.isabs(p):
        return None
    rp = os.path.realpath(p)
    for root in cfg.allowed:
        if rp == root or rp.startswith(root.rstrip(os.sep) + os.sep):
            return rp
    return None


# ----------------------------------------------------------------------------- plan, condition, cohort

def plan_value(plan: Any, dotted: str) -> Any:
    cur = plan
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def condition_label(plan: dict, condition: str | None) -> str:
    """The run's condition: ``code``/``tools`` by default, else the value at the dotted plan path."""
    if condition is None:
        return CODE if (plan.get("policy") or {}).get("code_mode") else TOOLS
    value = plan_value(plan, condition)
    if value is None:
        return "none"
    if isinstance(value, (str, int, float, bool)):
        return str(value).lower() if isinstance(value, bool) else str(value)
    return trunc(json.dumps(value, ensure_ascii=False, sort_keys=True), 60) or "none"


def config_prefix(config: str | None, task: str | None) -> str | None:
    """The spec builder's config without the task and what follows it (seed, layout), else its first token."""
    if not config:
        return None
    if task and f"-{task}" in config:
        return config[:config.index(f"-{task}")] or config
    return config.split("-")[0]


def _proj_plan(p: dict, condition: str | None) -> dict:
    pol = p.get("policy") or {}
    base = pol.get("baseline") or {}
    ts = pol.get("tool_surface") or {}
    rt = p.get("runtime") or {}
    task = p.get("task")
    if isinstance(task, dict):
        task = task.get("id") or task.get("name")
    seed = p.get("seed") if not isinstance(p.get("seed"), dict) else rt.get("eval_seed")
    admission = p.get("launch_admission") if isinstance(p.get("launch_admission"), dict) else {}
    config = p.get("config") if isinstance(p.get("config"), str) else None
    experiment = p.get("experiment")
    cohort = admission.get("cohort_id") or p.get("cohort_id") or config_prefix(config, task) or experiment
    layout = rt.get("layout_id")
    return {"baseline": (base.get("id") if isinstance(base, dict) else base) or None,
            "tool_surface": ts.get("profile") if isinstance(ts, dict) else None,
            "code_mode": bool(pol.get("code_mode")), "condition": condition_label(p, condition),
            "task": task, "layout": layout, "seed": seed, "backend": rt.get("kind"),
            "column": ("l%s" % layout) if layout is not None else (("s%s" % seed) if seed is not None else "-"),
            "column_sort": layout if isinstance(layout, int) else (seed if isinstance(seed, int) else 0),
            "model": pol.get("model"), "endpoint": pol.get("endpoint"), "experiment": experiment,
            "config": config, "cohort": str(cohort) if cohort else None,
            "task_instruction": p.get("task_instruction") if isinstance(p.get("task_instruction"), str) else None}


def plan_info(run_dir: str, cfg: Config) -> dict:
    return read_json_proj(os.path.join(run_dir, "plan.json"), lambda p: _proj_plan(p, cfg.condition),
                          kind="plan:" + (cfg.condition or "")) or {}


def _proj_launch(launch: dict) -> dict:
    er = launch.get("execution_resource") or {}
    card = None
    cvd = (er.get("worker_environment") or {}).get("CUDA_VISIBLE_DEVICES")
    if cvd not in (None, ""):
        card = str(cvd).split(",")[0]
    else:
        m = re.search(r"-g(\d+)s\d+$", str(er.get("slot_id") or ""))
        if m:
            card = m.group(1)
    return {"node": er.get("host_id"), "card": card, "created_unix": launch.get("created_unix_s"),
            "slot": er.get("slot_id"), "retry_of": launch.get("retry_of")}


# ----------------------------------------------------------------------------- receipts and summary

def _proj_receipt(r: dict) -> dict:
    args = r.get("arguments")
    return {"action": r.get("action"), "action_id": r.get("action_id"),
            "status": r.get("execution_status") or r.get("status"),
            "failure_code": r.get("failure_code"), "failure_stage": r.get("failure_stage"),
            "native_actions": r.get("native_actions"), "physics_steps": r.get("physics_steps"),
            "terminal_after_action": r.get("terminal_after_action"), "started": r.get("started_unix_s"),
            "finished": r.get("finished_unix_s"), "target_arms": r.get("target_arms"),
            "motion_strategy": r.get("motion_strategy"),
            "arguments": trunc(json.dumps(args, ensure_ascii=False), 1200) if args is not None else None}


def receipts(run_dir: str) -> list[dict]:
    d = os.path.join(run_dir, "native", "receipts")
    try:
        names = sorted(n for n in os.listdir(d) if n.endswith(".json"))
    except OSError:
        return []
    out = []
    for n in names:
        r = read_json_proj(os.path.join(d, n), _proj_receipt)
        if r is not None:
            r = dict(r)
            r["file"] = "native/receipts/" + n
            m = re.search(r"(\d+)", n)
            r["index"] = int(m.group(1)) if m else None
            out.append(r)
    return out


def short_id(run_id: str, experiment: str | None) -> str:
    return run_id[len(experiment) + 1:] if experiment and run_id.startswith(experiment + "-") else run_id


def run_summary(run_dir: str, cfg: Config) -> dict:
    """Headline facts for one run directory; tolerant of every file being missing."""
    run_id = os.path.basename(run_dir.rstrip("/"))
    plan = plan_info(run_dir, cfg)
    launch = read_json_proj(os.path.join(run_dir, "launch.json"), _proj_launch) or {}
    result = read_json(os.path.join(run_dir, "result.json"))
    status = read_json(os.path.join(run_dir, "status.json")) or {}
    experiment = plan.get("experiment") or os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
    s: dict[str, Any] = {"run_id": run_id, "run_dir": run_dir, **plan, "experiment": experiment,
                         "cohort": plan.get("cohort") or experiment, "short_id": short_id(run_id, experiment),
                         "backend_name": BACKEND_NAMES.get(plan.get("backend"), plan.get("backend")),
                         "node": launch.get("node"), "card": launch.get("card"), "slot": launch.get("slot"),
                         "started_unix": launch.get("created_unix"),
                         "sealed": os.path.isfile(os.path.join(run_dir, "seal.json"))}
    if s.get("started_unix") is None:
        ev = first_line_json(os.path.join(run_dir, "events.jsonl")) or {}
        s["started_unix"] = ev.get("time")
    if isinstance(result, dict):
        v = result.get("verdict") or {}
        wa = result.get("wait_accounting") or {}
        s.update({
            "state_raw": result.get("state"), "termination": result.get("termination"),
            "termination_authority": result.get("termination_authority"),
            "official_success": result.get("official_success"),
            "success_after_official_limit": result.get("success_after_official_limit"),
            "behavioral_sample_valid": result.get("behavioral_sample_valid"),
            "model_calls": result.get("model_calls"), "decisions": result.get("decisions"),
            "refused_proposals": result.get("refused_proposals"),
            "native_actions": v.get("native_actions", wa.get("native_actions")),
            "native_action_limit": v.get("native_action_limit"),
            "wall_s": result.get("wall_duration_s"), "finished_unix": result.get("finished_unix_s"),
            "fault": v.get("fault_reason") or v.get("abort_reason"),
        })
        if s["native_actions"] is None:
            counted = [r.get("native_actions") for r in receipts(run_dir) if isinstance(r.get("native_actions"), int)]
            s["native_actions"] = sum(counted) if counted else None
        if result.get("started_unix_s"):
            s["started_unix"] = result.get("started_unix_s")
        if s["state_raw"] == "completed" and s["behavioral_sample_valid"] is not False and not s["fault"]:
            s["state"] = "done"
        elif s["behavioral_sample_valid"] is False or s["fault"]:
            s["state"] = "fault"
        else:
            s["state"] = "failed"
    else:
        rc = receipts(run_dir)
        s.update({"model_calls": status.get("model_calls"), "decisions": status.get("decisions"),
                  "termination": status.get("termination"), "official_success": status.get("official_success"),
                  "native_actions": sum((r.get("native_actions") or 0) for r in rc) if rc else None})
        last = 0.0
        for f in ("events.jsonl", "agent-events.jsonl", "status.json", "supervisor.log",
                  "native/code-sandbox/calls.jsonl", "native/receipts"):
            sg = stat_sig(os.path.join(run_dir, f))
            if sg:
                last = max(last, sg[0] / 1e9)
        s["last_activity_unix"] = last or None
        if last and now() - last > cfg.stale_s:
            s["state"] = "fault"
            sup = (tail_text(os.path.join(run_dir, "supervisor.log"), 4096) or "").strip()
            s["fault"] = (sup.splitlines()[-1] if sup else
                          "no result.json and no activity for %d min" % ((now() - last) // 60))
        else:
            s["state"] = "running"
        if s.get("started_unix"):
            s["wall_s"] = (last or now()) - s["started_unix"]
    s["verdict"] = verdict_word(s)
    s["started_utc"] = utc(s.get("started_unix"))
    s["finished_utc"] = utc(s.get("finished_unix"))
    return s


def verdict_word(record: dict) -> str | None:
    """``success``, ``fail`` or none; ``late``, a success past the official limit, is not official and no failure."""
    official = record.get("official_success")
    if official is False and record.get("success_after_official_limit") is True:
        return "late"
    return "success" if official is True else ("fail" if official is False else None)


def round_verdicts(run_dir: str) -> list[str] | None:
    """A multi-round session keeps each round's verdict in rounds/round-NNNN/result.json."""
    rd = os.path.join(run_dir, "rounds")
    if not os.path.isdir(rd):
        return None
    out = []
    for n in sorted(os.listdir(rd)):
        r = read_json(os.path.join(rd, n, "result.json")) or {}
        out.append(verdict_word(r if isinstance(r, dict) else {}) or "-")
    return out or None


# ----------------------------------------------------------------------------- cohorts

ROW_KEYS = ("run_id", "short_id", "experiment", "cohort", "backend", "backend_name", "task", "column",
            "column_sort", "layout", "seed", "condition", "config", "baseline", "state", "verdict",
            "termination", "wall_s", "model_calls", "native_actions", "node", "card", "fault", "started_utc",
            "finished_utc", "sealed")


def run_row(run_dir: str, cfg: Config) -> dict:
    s = run_summary(run_dir, cfg)
    if s.get("state") == "running":
        seen = sum(1 for e in jsonl(os.path.join(run_dir, "agent-events.jsonl"), proj_agent_event)
                   if e["t"] == "provider_response")
        s["model_calls"] = max(s.get("model_calls") or 0, seen)
    row = {k: s.get(k) for k in ROW_KEYS}
    rv = round_verdicts(run_dir)
    if rv:
        row["round_verdicts"] = rv
        if row.get("verdict") is None and rv[0] in ("success", "late", "fail"):
            row["verdict"] = rv[0]
    return row


#: The scan of every run is reused this long, so a page's requests in a row read the runs once.
ROWS_TTL_S = 5.0
_rows_lock = threading.Lock()
_rows_memo: dict[tuple, tuple[float, list[dict]]] = {}


def all_rows(cfg: Config) -> list[dict]:
    key = (cfg.outputs, cfg.experiment, cfg.condition, cfg.stale_s)
    with _rows_lock:
        hit = _rows_memo.get(key)
        if hit and now() - hit[0] < ROWS_TTL_S:
            return hit[1]
    rows = [run_row(rd, cfg) for _, rd in run_dirs(cfg)]
    with _rows_lock:
        _rows_memo[key] = (now(), rows)
    return rows


def cohort_brief(name: str, experiment: str, rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for it in rows:
        counts[it.get("state") or "?"] = counts.get(it.get("state") or "?", 0) + 1
    conditions: dict[str, dict] = {}
    for label in sorted({r.get("condition") or "none" for r in rows}):
        eps = [r for r in rows if (r.get("condition") or "none") == label]
        # A multi-round session counts each played round, not the session once.
        judged = [v for r in eps for v in (r.get("round_verdicts") or [r.get("verdict")]) if v in ("success", "late", "fail")]
        conditions[label] = {"n": len(judged), "success": sum(1 for v in judged if v == "success"),
                             "episodes": len(eps)}
    started = [r["started_utc"] for r in rows if r.get("started_utc")]
    columns = sorted({(r.get("column_sort") or 0, r.get("column") or "-") for r in rows})
    return {"name": name, "experiment": experiment, "counts": counts, "total": len(rows),
            "finished": sum(counts.get(k, 0) for k in ("done", "failed", "fault")),
            "conditions": conditions, "tasks": sorted({str(r.get("task")) for r in rows if r.get("task") is not None}),
            "columns": [c for _, c in columns],
            "backends": sorted({r.get("backend_name") or "?" for r in rows}),
            "configs": sorted({r.get("config") for r in rows if r.get("config")})[:50],
            "created_utc": min(started) if started else None, "last_utc": max(started) if started else None}


def cohorts(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        out.setdefault((r["experiment"], r["cohort"]), []).append(r)
    return out


# ----------------------------------------------------------------------------- one run's timeline

def proj_agent_event(o: dict, off: int, ln: int) -> dict:
    t = o.get("type")
    p = o.get("payload") or {}
    r: dict[str, Any] = {"t": t, "step": o.get("observation_step"), "mc": o.get("model_call_id"),
                         "tc": o.get("tool_call_id"), "ts": o.get("unix_s")}
    origin = p.get("origin") if isinstance(p, dict) else None
    if isinstance(origin, dict) and origin.get("parent_tool_call_id"):
        r["parent"] = origin.get("parent_tool_call_id")   # a code-mode script's own call (`<call>#NNN`)
    ar = o.get("artifact_ref") or {}
    if isinstance(ar, dict) and ar.get("path"):
        r["art"] = ar.get("path")
        r["art_bytes"] = ar.get("bytes")
    if t == "provider_response":
        m = p.get("message") or {}
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(b.get("text", "") for b in c if isinstance(b, dict))
        r.update({"reasoning": m.get("reasoning_content") or "", "content": c or "", "usage": p.get("usage"),
                  "finish": p.get("finish_reason") or p.get("stop_reason"), "error": p.get("error"),
                  "attempts": p.get("attempts")})
    elif t == "tool_call_proposed":
        r.update({"name": p.get("name"), "args": p.get("arguments"), "effect": p.get("effect"),
                  "batch_valid": p.get("batch_valid")})
    elif t == "tool_result":
        ex = p.get("execution") or {}
        r.update({"name": p.get("name"), "ok": p.get("ok"),
                  "error": trunc(p.get("error"), 800) if p.get("error") else None,
                  "native_actions": p.get("native_actions", ex.get("native_actions_used")),
                  "actions": [a.get("action_id") for a in (ex.get("actions") or []) if isinstance(a, dict)]})
    elif t in ("action_proposed", "environment_receipt"):
        r.update({"action_id": p.get("action_id"), "obs": p.get("observation_id")})
    elif t in ("fresh_observation_published", "observation_received"):
        r["obs"] = p.get("observation_id")
    elif t in ("official_verdict", "run_terminated", "session_closed"):
        r["payload"] = {k: v for k, v in p.items() if not isinstance(v, (dict, list))}
    elif t in ("context_projected", "provider_request", "tool_image", "session_started"):
        pass
    else:
        r["payload"] = trunc(json.dumps(p, ensure_ascii=False), 400)
    return r


def proj_call(o: dict, off: int, ln: int) -> dict:
    """One call of ``native/code-sandbox/calls.jsonl`` (a coding surface's call log), outputs truncated."""
    ex = o.get("execution") or {}
    inp = o.get("input") or {}
    out = o.get("output")
    if not isinstance(out, str):
        out = json.dumps(out, ensure_ascii=False) if out is not None else ""
    shown = {k: trunc(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False), 6000) for k, v in inp.items()}
    acts = []
    for a in ex.get("actions") or []:
        if not isinstance(a, dict):
            continue
        act = a.get("action") or {}
        rc = a.get("receipt") or {}
        acts.append({"action_id": a.get("action_id"), "kind": act.get("action") if isinstance(act, dict) else None,
                     "status": rc.get("status"), "failure_code": rc.get("failure_code"),
                     "native_actions": a.get("native_actions_used", rc.get("native_actions")),
                     "refusal": trunc(a.get("refusal"), 600) if a.get("refusal") else None,
                     "before": (a.get("before_observation") or {}).get("step"),
                     "after": (a.get("after_observation") or {}).get("step")})
    ch = o.get("changes") or {}
    return {"index": o.get("index"), "tool": o.get("tool"), "tc": o.get("tool_call_id"), "mc": o.get("model_call_id"),
            "input": shown, "input_truncated": any(len(str(v)) > 6000 for v in inp.values()),
            "output": trunc(out, 4000), "output_chars": o.get("output_chars", len(out)),
            "output_truncated": len(out) > 4000, "exit_code": o.get("exit_code"), "is_error": o.get("is_error"),
            "native_actions": o.get("native_actions"), "interface_calls": o.get("interface_calls") or {},
            "changes": {"added": ch.get("added") or [], "modified": ch.get("modified") or [],
                        "removed": ch.get("removed") or []},
            "started": o.get("started_unix_s"), "finished": o.get("finished_unix_s"),
            "step": (ex.get("initial_observation") or {}).get("step"),
            "final_step": (ex.get("final_observation") or {}).get("step"),
            "finished_episode": o.get("finished_episode"), "actions": acts, "off": off, "len": ln}


def proj_pi(o: dict, off: int, ln: int) -> dict | None:
    t = o.get("type")
    if t == "message":
        m = o.get("message") or {}
        if m.get("role") != "assistant":
            return None
        th, tx, ids = [], [], []
        cont = m.get("content")
        if isinstance(cont, str):
            tx.append(cont)
        for b in cont if isinstance(cont, list) else []:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "thinking":
                th.append(b.get("thinking") or b.get("text") or "")
            elif bt == "text":
                tx.append(b.get("text") or "")
            elif bt == "toolCall":
                ids.append(b.get("id"))
        u = m.get("usage") or {}
        return {"k": "a", "thinking": "\n\n".join(x for x in th if x), "text": "\n".join(x for x in tx if x).strip(),
                "ids": ids, "usage": {"input": u.get("input"), "output": u.get("output"), "reasoning": u.get("reasoning")},
                "ts": o.get("timestamp"), "stop": m.get("stopReason"), "error": m.get("errorMessage")}
    if t == "custom_message":
        return {"k": "c", "type": o.get("customType"), "content": trunc(o.get("content"), 3000), "ts": o.get("timestamp")}
    return None


def proj_run_event(o: dict, off: int, ln: int) -> dict:
    small: dict[str, Any] = {}
    for k, v in o.items():
        if k in ("type", "time", "run_id"):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            small[k] = trunc(v, 200) if isinstance(v, str) else v
        else:
            small[k] = "{...}" if isinstance(v, dict) else "[...]"
    return {"type": o.get("type"), "time": o.get("time"), "utc": utc(o.get("time")), "fields": small}


def art_preview(path: str) -> str | None:
    """A short preview of a tool-result artifact: a code-mode result's output, else its JSON."""
    def load(p: str) -> str | None:
        if os.path.getsize(p) > 400000:
            return None
        with open(p, "rb") as f:
            d = json.loads(f.read())
        if isinstance(d, dict) and isinstance(d.get("output"), str):
            head = " ".join(f"{k} {d[k]}" for k in ("status", "exit_code") if d.get(k) is not None)
            return trunc((head + "\n" if head else "") + d["output"], 1500)
        return trunc(json.dumps(d, ensure_ascii=False), 700)
    return cached_file(path, load, "artprev")


def pi_session_file(run_dir: str) -> str | None:
    d = os.path.join(run_dir, "native", "pi-session")
    try:
        fs = sorted(f for f in os.listdir(d) if f.endswith(".jsonl"))
    except OSError:
        return None
    return os.path.join(d, fs[-1]) if fs else None


def observation_dirs(run_dir: str) -> dict[int, dict]:
    """step -> {obs_id, images: {camera: relative rgb path}}; cameras are whatever the backend recorded."""
    d = os.path.join(run_dir, "policy-observations")
    out: dict[int, dict] = {}
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for n in names:
        m = re.search(r"observation-(\d+)$", n)
        if not m:
            continue
        try:
            cams = sorted(c for c in os.listdir(os.path.join(d, n))
                          if os.path.isfile(os.path.join(d, n, c, "rgb.png")))
        except OSError:
            cams = []
        out[int(m.group(1))] = {"obs_id": n,
                                "images": {c: "policy-observations/%s/%s/rgb.png" % (n, c) for c in cams}}
    return out


def code_text(args: Any) -> str | None:
    """The source a code-mode call ran: a ``bash`` command or a ``run_python`` script."""
    if isinstance(args, dict):
        for key in ("command", "code"):
            if isinstance(args.get(key), str):
                return args[key]
    return None


RAW_FILES = ("result.json", "plan.json", "launch.json", "status.json", "evaluation.json", "seal.json",
             "runtime-metrics.json", "events.jsonl", "agent-events.jsonl", "supervisor.log",
             "native/code-sandbox/calls.jsonl", "native/pi-calls.jsonl", "native/official-verdict.json",
             "compiled-policy/episode-task.md", "compiled-policy/tool-manual.md",
             "compiled-policy/project-instructions.md", "native/environment-worker.log", "native/pi-core.log")


def run_detail(run_dir: str, cfg: Config) -> dict:
    s = run_summary(run_dir, cfg)
    events = jsonl(os.path.join(run_dir, "agent-events.jsonl"), proj_agent_event)
    turns: dict[str, dict] = {}
    turn_order: list[str] = []
    calls_by_tc: dict[Any, dict] = {}
    actions: dict[Any, dict] = {}
    action_order: list[Any] = []
    verdict_ev = terminated = None
    tok_in = tok_out = tok_reason = 0
    max_step = -1

    def turn(mc: Any, step: Any = None, ts: Any = None) -> dict:
        key = mc or "turn-%d" % len(turn_order)
        t = turns.get(key)
        if t is None:
            t = turns[key] = {"mc": key, "step": step, "ts": ts, "reasoning": "", "content": "", "usage": None,
                              "calls": []}
            turn_order.append(key)
        return t

    for e in events:
        t = e["t"]
        if isinstance(e.get("step"), int):
            max_step = max(max_step, e["step"])
        if t == "provider_response":
            tu = turn(e.get("mc"), e.get("step"), e.get("ts"))
            tu.update({"step": e.get("step"), "ts": e.get("ts"), "reasoning": e.get("reasoning") or "",
                       "content": e.get("content") or "", "usage": e.get("usage"), "finish": e.get("finish"),
                       "error": e.get("error")})
            u = e.get("usage") or {}
            tok_in += u.get("input_tokens") or 0
            tok_out += u.get("output_tokens") or 0
            tok_reason += u.get("reasoning_output_tokens") or 0
        elif t == "tool_call_proposed":
            c = {"id": e.get("tc"), "name": e.get("name"), "args": e.get("args"), "effect": e.get("effect"),
                 "step": e.get("step"), "ts": e.get("ts"), "ok": None, "actions": [], "images": [],
                 "code": code_text(e.get("args"))}
            calls_by_tc[e.get("tc")] = c
            parent = calls_by_tc.get(e.get("parent")) if e.get("parent") else None
            if parent is not None:
                c["script_seq"] = len(parent.setdefault("script_calls", [])) + 1
                parent["script_calls"].append(c)
            else:
                turn(e.get("mc"), e.get("step"), e.get("ts"))["calls"].append(c)
        elif t == "tool_result":
            c = calls_by_tc.get(e.get("tc"))
            if c is not None:
                c.update({"ok": e.get("ok"), "error": e.get("error"), "native_actions": e.get("native_actions"),
                          "actions": e.get("actions") or c["actions"], "result_art": e.get("art"),
                          "result_bytes": e.get("art_bytes"), "done_ts": e.get("ts")})
        elif t == "tool_image":
            c = calls_by_tc.get(e.get("tc"))
            if c is not None and e.get("art"):
                c["images"].append(e["art"])
        elif t == "action_proposed":
            aid = e.get("action_id")
            if aid not in actions:
                action_order.append(aid)
            actions.setdefault(aid, {"action_id": aid})
            actions[aid].update({"step": e.get("step"), "mc": e.get("mc"), "proposal_art": e.get("art"),
                                 "ts": e.get("ts")})
            c = calls_by_tc.get(e.get("tc"))     # a tool-call surface: the call that proposed this action
            if c is not None and aid and aid not in c["actions"]:
                c["actions"].append(aid)
        elif t == "environment_receipt":
            aid = e.get("action_id")
            if aid not in actions:
                action_order.append(aid)
            actions.setdefault(aid, {"action_id": aid})["receipt_art"] = e.get("art")
        elif t == "official_verdict":
            verdict_ev = e.get("payload")
        elif t == "run_terminated":
            terminated = e.get("payload")

    # receipts on disk carry status, failure code and native actions
    for r in receipts(run_dir):
        aid = r.get("action_id")
        if aid not in actions:
            action_order.append(aid)
            actions[aid] = {"action_id": aid, "step": r.get("index")}
        actions[aid]["receipt"] = r

    # a coding surface's own call log (native/code-sandbox/calls.jsonl), when the run has one
    calls_file = os.path.join(run_dir, "native", "code-sandbox", "calls.jsonl")
    bcalls = jsonl(calls_file, proj_call) if os.path.exists(calls_file) else []
    for bc in bcalls:
        c = calls_by_tc.get(bc.get("tc"))
        if c is None:  # agent-events missing or lagging: synthesize from calls.jsonl
            c = {"id": bc.get("tc"), "name": bc.get("tool"), "args": None, "effect": None, "step": bc.get("step"),
                 "ts": bc.get("started"), "ok": not bc.get("is_error"), "actions": [], "images": []}
            calls_by_tc[bc.get("tc")] = c
            turn(bc.get("mc"), bc.get("step"), bc.get("started"))["calls"].append(c)
        c["b"] = {k: v for k, v in bc.items() if k not in ("off", "len")}
        c["native_actions"] = bc.get("native_actions")
        c["actions"] = [a.get("action_id") for a in bc.get("actions") or []]
        for a in bc.get("actions") or []:
            aid = a.get("action_id")
            if aid not in actions:
                action_order.append(aid)
                actions[aid] = {"action_id": aid, "step": a.get("before")}
            actions[aid].setdefault("kind", a.get("kind"))
            if a.get("refusal"):
                actions[aid]["refusal"] = a.get("refusal")
            actions[aid]["call_index"] = bc.get("index")

    # the model's reasoning from the Pi session where agent-events lack it (by tool-call id, else by call index)
    pif = pi_session_file(run_dir)
    custom: list[dict] = []
    if pif:
        pis = jsonl(pif, proj_pi)
        assistants = [p for p in pis if p["k"] == "a"]
        custom = [p for p in pis if p["k"] == "c"]
        by_tc = {}
        for i, a in enumerate(assistants):
            a["i"] = i
            for tid in a["ids"]:
                by_tc[tid] = a
        used = set()
        for key in turn_order:
            tu = turns[key]
            a = None
            for c in tu["calls"]:
                a = by_tc.get(c.get("id"))
                if a:
                    break
            if a is None:
                m = re.search(r"coding-call-(\d+)$", key)
                if m and int(m.group(1)) < len(assistants):
                    a = assistants[int(m.group(1))]
            if a is not None and a["i"] not in used:
                used.add(a["i"])
                if not tu.get("reasoning"):
                    tu["reasoning"] = a.get("thinking") or ""
                if not tu.get("content"):
                    tu["content"] = a.get("text") or ""
                if not tu.get("usage") and a.get("usage"):
                    u = a["usage"]
                    tu["usage"] = {"input_tokens": u.get("input"), "output_tokens": u.get("output")}
                    tok_in += u.get("input") or 0
                    tok_out += u.get("output") or 0

    # previews of tool results (small json artifacts; a code-mode result's output)
    for c in calls_by_tc.values():
        if c.get("result_art") and "b" not in c:
            c["result_preview"] = art_preview(os.path.join(run_dir, c["result_art"]))
        c["moved"] = bool(c.get("native_actions")) or bool(c.get("actions"))

    obs = observation_dirs(run_dir)
    all_steps = set(obs)
    for key in turn_order:
        if isinstance(turns[key].get("step"), int):
            all_steps.add(turns[key]["step"])
    for a in actions.values():
        if isinstance(a.get("step"), int):
            all_steps.add(a["step"])
    steps = []
    for st in sorted(all_steps):
        o = obs.get(st) or {}
        steps.append({"step": st, "obs_id": o.get("obs_id"), "images": o.get("images") or {},
                      "turns": [turns[k] for k in turn_order if turns[k].get("step") == st],
                      "actions": [actions[a] for a in action_order if actions[a].get("step") == st]})
    orphan_turns = [turns[k] for k in turn_order if not isinstance(turns[k].get("step"), int)]
    if orphan_turns:
        steps.append({"step": None, "obs_id": None, "images": {}, "turns": orphan_turns,
                      "actions": [actions[a] for a in action_order if not isinstance(actions[a].get("step"), int)]})

    raw = []
    for rel in RAW_FILES:
        sg = stat_sig(os.path.join(run_dir, rel))
        if sg:
            raw.append({"rel": rel, "size": sg[1]})
    if pif:
        raw.append({"rel": os.path.relpath(pif, run_dir), "size": (stat_sig(pif) or (0, 0))[1]})

    s["tokens_in"] = tok_in or None
    s["tokens_out"] = tok_out or None
    s["tokens_reasoning"] = tok_reason or None
    s["model_calls_seen"] = len([k for k in turn_order if turns[k].get("usage") is not None]) or len(turn_order)
    if s.get("model_calls") is None or (s.get("state") == "running" and s["model_calls_seen"] > (s.get("model_calls") or 0)):
        s["model_calls"] = s["model_calls_seen"] or None
    s["tool_calls"] = sum(1 for c in calls_by_tc.values() if "script_seq" not in c)
    s["script_calls"] = sum(1 for c in calls_by_tc.values() if "script_seq" in c)
    s["actions"] = len(actions)
    s["steps"] = len(obs)
    s["last_step"] = max(list(obs) + [max_step]) if (obs or max_step >= 0) else None
    s["round_verdicts"] = round_verdicts(run_dir)
    run_events = jsonl(os.path.join(run_dir, "events.jsonl"), proj_run_event)
    return {
        "summary": s, "steps": steps, "raw": raw,
        "verdict_event": verdict_ev, "terminated": terminated,
        "observations_custom": custom[-60:],
        "supervisor_tail": tail_text(os.path.join(run_dir, "supervisor.log"), 8192),
        "run_events": run_events[-80:],
        "now": now(),
    }
