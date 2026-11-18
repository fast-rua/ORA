"""The agent's side of a run for the run dashboard: its trace on the replay's clock and its files.

Ported from Shaoxiong Duan's code-vs-tools dashboard (snapshot 2026-10-31). One entry per tool call: the
turn's thinking and text, the code or tool arguments, what came back (text and the exact images the model
was shown) and the robot commands the call caused. The files the agent wrote come from ORA's code-mode
evidence: each code call's workspace snapshot (``native/code-workspace-objects``, ``glue-workspace-objects``
for runs made before the rename) or a coding surface's call log (``native/code-sandbox/``).
"""

from __future__ import annotations

import ast
import base64
import json
import os
import re
from typing import Any
from urllib.parse import quote

from viewer.dashboard import replay
from viewer.dashboard.config import Config
from viewer.dashboard.runs import pi_session_file, proj_agent_event, run_detail
from viewer.dashboard.store import cached_file, jsonl, read_json, read_line_at, trunc

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp")
OBJECT_DIRS = ("native/code-workspace-objects", "native/glue-workspace-objects")


def file_url(path: str) -> str:
    return "api/file?path=" + quote(path)


# ----------------------------------------------------------------------------- images the model was shown

def proj_pi_result(o: dict, off: int, ln: int) -> dict | None:
    """Pi toolResult messages that carried images: which tool call, the image labels, where the line is."""
    if o.get("type") != "message":
        return None
    m = o.get("message") or {}
    if m.get("role") != "toolResult":
        return None
    labels, pending = [], None
    for b in m.get("content") or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and (b.get("text") or "").startswith("[image]"):
            pending = b["text"][7:].strip()
        elif b.get("type") == "image":
            labels.append(pending or "image %d" % len(labels))
            pending = None
    return {"tc": m.get("toolCallId"), "labels": labels, "off": off, "ln": ln} if labels else None


def pi_image(run_dir: str, off: int, ln: int, k: int) -> tuple[bytes | None, str | None]:
    """The bytes of image k of the Pi tool result at (off, ln): exactly what the model was shown."""
    pif = pi_session_file(run_dir)
    if not pif or not any(r["off"] == off and r["ln"] == ln for r in jsonl(pif, proj_pi_result)):
        return None, None
    m = read_line_at(pif, off, ln).get("message") or {}
    imgs = [b for b in m.get("content") or [] if isinstance(b, dict) and b.get("type") == "image"]
    if not 0 <= k < len(imgs):
        return None, None
    return base64.b64decode(imgs[k].get("data") or ""), imgs[k].get("mimeType") or "image/png"


# ----------------------------------------------------------------------------- workspace file versions

def blob_path(run_dir: str, sha: str) -> str | None:
    """One stored file version by sha256: a code-mode workspace object or a coding surface's blob."""
    if not re.match(r"^[0-9a-f]{64}$", sha or ""):
        return None
    for d in OBJECT_DIRS:
        p = os.path.join(run_dir, d, sha)
        if os.path.isfile(p):
            return p
    p = os.path.join(run_dir, "native", "code-sandbox", "blobs", sha[:2], sha)
    return p if os.path.isfile(p) else None


def _art_text(path: str, n: int = 6000) -> str | None:
    def load(p: str) -> str | None:
        if os.path.getsize(p) > 2000000:
            return None
        with open(p, "rb") as f:
            d = json.loads(f.read())
        if isinstance(d, dict) and isinstance(d.get("output"), str):   # a code-mode result: its output first
            rest = {k: v for k, v in d.items() if k not in ("output", "workspace")}
            return d["output"] + "\n\n" + json.dumps(rest, ensure_ascii=False, indent=1)
        return json.dumps(d, ensure_ascii=False, indent=1)
    t = cached_file(path, load, "arttext")
    return trunc(t, n) if t else None


def _snapshot(path: str) -> dict[str, str] | None:
    """A code-mode result's workspace snapshot, path -> sha256; None when the result has none."""
    def load(p: str) -> dict[str, str] | None:
        if os.path.getsize(p) > 4000000:
            return None
        with open(p, "rb") as f:
            d = json.loads(f.read())
        ws = d.get("workspace") if isinstance(d, dict) else None
        if not isinstance(ws, list):
            return None
        return {w["path"]: w.get("sha256") for w in ws if isinstance(w, dict) and isinstance(w.get("path"), str)}
    return cached_file(path, load, "wsnap")


def workspace_changes(run_dir: str) -> list[dict]:
    """Each top-level code call whose result recorded a workspace snapshot, in order: its tool-call id and
    what it added, modified or removed relative to the snapshot before it."""
    out, prev = [], {}
    for e in jsonl(os.path.join(run_dir, "agent-events.jsonl"), proj_agent_event):
        if e["t"] != "tool_result" or not e.get("art") or e.get("parent"):
            continue
        snap = _snapshot(os.path.join(run_dir, e["art"]))
        if snap is None:
            continue
        out.append({"tc": e.get("tc"), "snapshot": snap,
                    "added": sorted(p for p in snap if p not in prev),
                    "modified": sorted(p for p in snap if p in prev and snap[p] != prev[p]),
                    "removed": sorted(p for p in prev if p not in snap)})
        prev = snap
    return out


def _proj_call_files(o: dict, off: int, ln: int) -> dict:
    return {"index": o.get("index"), "changes": o.get("changes") or {}, "after": o.get("workspace_after")}


def _py_outline(src: str) -> dict | None:
    """Module docstring and top-level defs (signature, first docstring line) of an agent-written Python file."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    doc = (ast.get_docstring(tree) or "").strip()
    defs = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            try:
                sig = ast.unparse(node.args) if not isinstance(node, ast.ClassDef) else ""
            except Exception:
                sig = ""
            d = (ast.get_docstring(node) or "").strip().split("\n")[0]
            defs.append({"name": node.name, "kind": "class" if isinstance(node, ast.ClassDef) else "def",
                         "sig": trunc(sig, 300), "doc": trunc(d, 300), "line": node.lineno})
    return {"doc": trunc(doc, 1500), "defs": defs}


def _skip(path: str) -> bool:
    return path.startswith(("outputs/", ".pi-output/")) or path.lower().endswith(IMAGE_EXT + (".npy", ".npz"))


def agent_files(run_dir: str, entries: list[dict] | None = None) -> list[dict]:
    """Every file the agent wrote, each version with the call that wrote it (and its trace entry, if given)."""
    entries = entries or []
    files: dict[str, dict] = {}
    by_call = {e.get("call"): e for e in entries if e.get("call") is not None}
    by_tc = {e.get("tc"): e for e in entries if e.get("tc")}
    calls_file = os.path.join(run_dir, "native", "code-sandbox", "calls.jsonl")
    if os.path.exists(calls_file):
        for bc in jsonl(calls_file, _proj_call_files):
            ch = bc["changes"]
            touched = (ch.get("added") or []) + (ch.get("modified") or [])
            p = blob_path(run_dir, bc["after"]) if touched and bc.get("after") else None
            man = read_json(p) if p else None
            for path in touched:
                if _skip(path):
                    continue
                f = files.setdefault(path, {"path": path, "versions": [], "removed": None})
                ent = by_call.get(bc["index"])
                f["versions"].append({"call": bc["index"], "entry": ent["i"] if ent else None,
                                      "n": ent["n0"] if ent else None, "sha": (man or {}).get(path)})
            for path in ch.get("removed") or []:
                if path in files:
                    files[path]["removed"] = bc["index"]
    else:
        for k, wc in enumerate(workspace_changes(run_dir), 1):
            ent = by_tc.get(wc["tc"])
            for path in wc["added"] + wc["modified"]:
                if _skip(path):
                    continue
                f = files.setdefault(path, {"path": path, "versions": [], "removed": None})
                f["versions"].append({"call": k, "tc": wc["tc"], "entry": ent["i"] if ent else None,
                                      "n": ent["n0"] if ent else None, "sha": wc["snapshot"].get(path)})
            for path in wc["removed"]:
                if path in files:
                    files[path]["removed"] = k
    for f in files.values():
        last = next((v["sha"] for v in reversed(f["versions"]) if v["sha"]), None)
        p = blob_path(run_dir, last) if last else None
        f["size"] = os.path.getsize(p) if p else None
        if p and f["path"].endswith(".py") and (f["size"] or 0) < 400000:
            with open(p, "rb") as fh:
                f["outline"] = _py_outline(fh.read().decode("utf-8", "replace"))
        stem = os.path.splitext(os.path.basename(f["path"]))[0]
        made = {v.get("entry") for v in f["versions"]}
        if len(stem) >= 3:
            f["used_by"] = [e["i"] for e in entries if e["i"] not in made and e.get("code") and
                            re.search(r"\b%s\b" % re.escape(stem), e["code"])]
    return sorted(files.values(), key=lambda f: f["path"])


# ----------------------------------------------------------------------------- the trace on the replay's clock

def replay_agent(run_dir: str, cfg: Config, round_index: int = 0) -> dict:
    """One entry per tool call (the model's thinking and text on the first call of each turn), placed on the
    replay's clock (``replay.bundle``) at its robot commands' span, else where the robot stood when it ran. A call
    whose action is a skill sequence spans the receipts of its stages (``<action_id>:<stage>``)."""
    det = run_detail(run_dir, cfg)
    here = replay.bundle(run_dir, cfg, round_index)
    span: dict[str, list[dict]] = {}
    for a in here.get("actions") or []:
        span.setdefault(a["action_id"], []).append(a)
        if ":" in (a.get("action_id") or ""):     # a stage of a skill sequence (RoboTwin "<id>:01-pregrasp")
            span.setdefault(a["action_id"].split(":", 1)[0], []).append(a)
    # which round each robot command ran in; an entry without commands belongs to the round of the one before it
    owner: dict[str, int] = {}
    for r in here.get("rounds") or [0]:
        for a in replay.bundle(run_dir, cfg, r).get("actions") or []:
            owner[a["action_id"]] = r
            owner.setdefault((a.get("action_id") or "").split(":", 1)[0], r)
    current = (here.get("rounds") or [0])[0]
    pif = pi_session_file(run_dir)
    pimg = {r["tc"]: r for r in jsonl(pif, proj_pi_result)} if pif else {}
    changes = {wc["tc"]: wc for wc in workspace_changes(run_dir)}
    rid = os.path.basename(run_dir.rstrip("/"))
    turns = [t for st in det["steps"] for t in st["turns"]]
    turns.sort(key=lambda t: t.get("ts") or 0)
    entries: list[dict] = []
    cursor = 0
    for ti, tu in enumerate(turns):
        calls = tu.get("calls") or [None]
        for ci, c in enumerate(calls):
            e: dict[str, Any] = {"turn": ti, "step": tu.get("step")}
            if ci == 0:
                e["think"] = trunc(tu.get("reasoning") or "", 8000)
                e["say"] = trunc(tu.get("content") or "", 4000)
                u = tu.get("usage") or {}
                e["tokens"] = [u.get("input_tokens"), u.get("output_tokens")]
            if c is not None:
                b = c.get("b")
                e["tool"] = c.get("name") or (b or {}).get("tool")
                e["tc"] = c.get("id")
                if b:
                    inp = b.get("input") or {}
                    e["code"] = inp.get("command") if isinstance(inp.get("command"), str) else json.dumps(inp, ensure_ascii=False, indent=1)
                    e["out"] = b.get("output")
                    e["out_chars"] = b.get("output_chars")
                    e["exit"] = b.get("exit_code")
                    e["error"] = bool(b.get("is_error")) or (b.get("exit_code") not in (0, None))
                    e["dur"] = round(b["finished"] - b["started"], 1) if b.get("finished") and b.get("started") else None
                    e["call"] = b.get("index")
                    e["changes"] = b.get("changes") or {}
                    e["finished_episode"] = b.get("finished_episode")
                    e["cmds"] = [{"id": a.get("action_id"), "kind": a.get("kind"), "status": a.get("status"),
                                  "fc": a.get("failure_code"), "refusal": a.get("refusal")} for a in b.get("actions") or []]
                else:
                    args = c.get("args")
                    e["code"] = c.get("code") or (json.dumps(args, ensure_ascii=False, indent=1) if not isinstance(args, str) else args)
                    e["out"] = (_art_text(os.path.join(run_dir, c["result_art"])) if c.get("result_art") else None) or c.get("error")
                    e["error"] = c.get("ok") is False
                    e["dur"] = round(c["done_ts"] - c["ts"], 1) if c.get("done_ts") and c.get("ts") else None
                    e["cmds"] = [{"id": a} for a in c.get("actions") or []]
                    if c.get("script_calls"):
                        e["script"] = [{"name": sc.get("name"), "ok": sc.get("ok")} for sc in c["script_calls"]]
                    wc = changes.get(c.get("id"))
                    if wc and (wc["added"] or wc["modified"] or wc["removed"]):
                        e["changes"] = {k: wc[k] for k in ("added", "modified", "removed")}
                imgs = [{"label": os.path.basename(art), "url": file_url(os.path.join(run_dir, art))}
                        for art in c.get("images") or []]
                pr = pimg.get(c.get("id"))
                if pr and not imgs:   # the same images as the tool_image artifacts when both are recorded
                    for k, lab in enumerate(pr["labels"]):
                        imgs.append({"label": lab, "url": "api/piimg?id=%s&off=%d&ln=%d&k=%d" % (quote(rid), pr["off"], pr["ln"], k)})
                e["imgs"] = imgs
            current = next((owner[cm["id"]] for cm in e.get("cmds") or [] if cm.get("id") in owner), current)
            if current != round_index:
                continue
            n0 = n1 = None
            for cm in e.get("cmds") or []:
                acts = span.get(cm["id"]) or []
                if acts:
                    last = acts[-1]
                    starts = [a["n0"] for a in acts if a.get("n0") is not None]
                    ends = [a["n1"] for a in acts if a.get("n1") is not None]
                    cm.update({"n0": min(starts) if starts else None, "n1": max(ends) if ends else None,
                               "kind": cm.get("kind") or (last.get("action") if len(acts) == 1 else "sequence"),
                               "status": cm.get("status") or last.get("status"),
                               "fc": cm.get("fc") or next((a.get("failure_code") for a in acts if a.get("failure_code")), None),
                               "modes": sorted({x.get("interaction") for a in acts for x in (a.get("arms") or {}).values()
                                                if x.get("interaction")})})
                    if starts:
                        n0 = min(starts) if n0 is None else min(n0, *starts)
                        n1 = max(ends) if n1 is None else max(n1, *ends)
            e["n0"] = cursor if n0 is None else n0
            e["n1"] = e["n0"] if n1 is None else n1
            cursor = max(cursor, e["n1"])
            e["i"] = len(entries)
            entries.append(e)
    return {"run_id": rid, "condition": det["summary"].get("condition"), "task": det["summary"].get("task"),
            "code_mode": det["summary"].get("code_mode"), "entries": entries,
            "files": agent_files(run_dir, entries)}
