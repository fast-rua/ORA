"""``ora-viewer dashboard``: a read-only JSON API and single-page UI over ORA run directories.

Ported from Shaoxiong Duan's code-vs-tools dashboard (snapshot 2026-10-31). Python standard library only on
the server (numpy, Pillow and usd-core are imported lazily, for the 3D replay's point clouds and meshes); plain
JavaScript in the browser. The server never writes to the runs: parsed files are cached in memory by
(path, mtime, size), append-only jsonl files are parsed incrementally, and derived point clouds and meshes go
to ``--cache``. ``api/file`` and ``api/ls`` read only inside the outputs root (403 elsewhere).

    ora-viewer dashboard --outputs OUTPUTS [--experiment NAME] [--host 127.0.0.1] [--port 5013]
        [--robodojo-assets DIR] [--robotwin-root DIR] [--scene-traces DIR]
"""

from __future__ import annotations

import argparse
import gzip
import json
import mimetypes
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from viewer.dashboard import agent, replay, robots, runs, scene_trace
from viewer.dashboard.config import ASSET_ROOT_ENV, ROBOTWIN_ROOT_ENV, Config, default_cache_dir
from viewer.dashboard.store import jsonl, now, read_line_at, stat_sig, tail_text

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
#: The shared light template every Viewer page uses (served as ``static/base.css``).
SHARED_CSS = HERE.parent / "static" / "base.css"
TEXT_EXT = {".json", ".jsonl", ".log", ".md", ".txt", ".py", ".yaml", ".yml", ".toml", ".sh", ".csv", ".cfg", ".ini"}
DEFAULT_PORT = 5013


class ApiError(Exception):
    def __init__(self, code: int, msg: str, **extra: Any):
        super().__init__(msg)
        self.code, self.msg, self.extra = code, msg, extra


def q1(qs: dict, k: str, default: str | None = None) -> str | None:
    v = qs.get(k)
    return v[0] if v else default


# ----------------------------------------------------------------------------- API

def _shown(cfg: Config, qs: dict, rows: list[dict]) -> list[dict]:
    """Rows without the hidden cohorts, unless the request asks for them (``hidden=1``)."""
    return rows if q1(qs, "hidden") == "1" else [r for r in rows if not cfg.hides(r["cohort"])]


def api_overview(cfg: Config, qs: dict) -> dict:
    everything = runs.all_rows(cfg)
    hidden = [r for r in everything if cfg.hides(r["cohort"])]
    rows = _shown(cfg, qs, everything)
    briefs = [runs.cohort_brief(name, exp, rs) for (exp, name), rs in runs.cohorts(rows).items()]
    briefs.sort(key=lambda b: b.get("last_utc") or "", reverse=True)
    experiments: dict[str, int] = {}
    for r in rows:
        experiments[r["experiment"]] = experiments.get(r["experiment"], 0) + 1
    # Each experiment's runs across all its cohorts first: launches usually make one cohort per treatment,
    # so this is where conditions meet on the same task and seed.
    briefs[:0] = [runs.cohort_brief(runs.ALL_COHORTS, exp, [r for r in rows if r["experiment"] == exp])
                  for exp in sorted(experiments)]
    recent = sorted(rows, key=lambda r: r.get("started_utc") or "", reverse=True)[:40]
    return {"outputs": cfg.outputs, "experiment": cfg.experiment, "condition_path": cfg.condition,
            "experiments": [{"name": k, "runs": v} for k, v in sorted(experiments.items())],
            "cohorts": briefs, "recent": recent, "now": now(),
            "hidden": {"cohorts": len({(r["experiment"], r["cohort"]) for r in hidden}), "runs": len(hidden)}}


def api_cohort(cfg: Config, qs: dict) -> dict:
    exp, name = q1(qs, "experiment"), q1(qs, "name")
    if not exp or not name:
        raise ApiError(400, "experiment and name required")
    rows = [r for r in runs.all_rows(cfg) if r["experiment"] == exp]
    # A hidden cohort's own page opens as any other; only "all cohorts" leaves it out.
    rows = _shown(cfg, qs, rows) if name == runs.ALL_COHORTS else [r for r in rows if r["cohort"] == name]
    if not rows:
        raise ApiError(404, "no such cohort")
    rows.sort(key=lambda r: r.get("started_utc") or "")
    return {"brief": runs.cohort_brief(name, exp, rows), "items": rows, "condition_path": cfg.condition,
            "now": now()}


def _run_dir_from_qs(cfg: Config, qs: dict) -> str:
    rd = runs.find_run_dir(cfg, q1(qs, "id"))
    if rd is None:
        raise ApiError(404, "run not found")
    return rd


def api_run(cfg: Config, qs: dict) -> dict:
    return runs.run_detail(_run_dir_from_qs(cfg, qs), cfg)


def api_run_summary(cfg: Config, qs: dict) -> dict:
    return runs.run_summary(_run_dir_from_qs(cfg, qs), cfg)


def api_call(cfg: Config, qs: dict) -> dict:
    """One full call of a coding surface's call log (untruncated input and output)."""
    rd = _run_dir_from_qs(cfg, qs)
    try:
        idx = int(q1(qs, "index", "-1") or "-1")
    except ValueError:
        raise ApiError(400, "index must be an integer") from None
    path = os.path.join(rd, "native", "code-sandbox", "calls.jsonl")
    for c in jsonl(path, runs.proj_call):
        if c.get("index") == idx:
            o = read_line_at(path, c["off"], c["len"])
            return {"index": idx, "tool": o.get("tool"), "input": o.get("input"), "output": o.get("output"),
                    "exit_code": o.get("exit_code"), "changes": o.get("changes"),
                    "interface_calls": o.get("interface_calls")}
    raise ApiError(404, "no such call")


def api_run_files(cfg: Config, qs: dict) -> dict:
    rd = _run_dir_from_qs(cfg, qs)
    return {"run_id": os.path.basename(rd), "files": agent.agent_files(rd)}


def api_runs(cfg: Config, qs: dict) -> dict:
    exp = q1(qs, "experiment")
    keys = ("run_id", "short_id", "experiment", "cohort", "backend_name", "task", "column", "condition",
            "state", "verdict", "started_utc")
    rows = [r for r in runs.all_rows(cfg) if not exp or r["experiment"] == exp]
    rows.sort(key=lambda r: r.get("started_utc") or "")
    return {"runs": [{k: r.get(k) for k in keys} for r in rows]}


def api_ls(cfg: Config, qs: dict) -> dict:
    p = runs.safe_path(cfg, q1(qs, "path"))
    if not p:
        raise ApiError(403, "path not allowed")
    if not os.path.isdir(p):
        raise ApiError(404, "not a directory")
    out = []
    for base, dirs, fns in os.walk(p):
        dirs.sort()
        for fn in sorted(fns):
            fp = os.path.join(base, fn)
            sg = stat_sig(fp)
            out.append({"rel": os.path.relpath(fp, p), "size": sg[1] if sg else None})
            if len(out) >= 3000:
                return {"path": p, "files": out, "truncated": True}
        if q1(qs, "recursive") != "1":
            break
    return {"path": p, "files": out, "truncated": False}


def _replay_run_dir(cfg: Config, qs: dict) -> str:
    """A run of a backend the 3D replay supports; a clear refusal otherwise (a Synthetic run has no robot)."""
    rd = _run_dir_from_qs(cfg, qs)
    backend = runs.plan_info(rd, cfg).get("backend")
    if backend not in replay.SUPPORTED_BACKENDS:
        name = runs.BACKEND_NAMES.get(backend, backend or "an unknown backend")
        supported = ", ".join(runs.BACKEND_NAMES[b] for b in replay.SUPPORTED_BACKENDS)
        raise ApiError(422, f"The 3D replay supports {supported} runs; this run is {name}. "
                            "Its timeline, cameras and receipts are in the run browser.",
                       backend=backend, supported=list(replay.SUPPORTED_BACKENDS))
    return rd


def _round_from_qs(qs: dict) -> int:
    value = q1(qs, "round")
    return int(value) if value and value.isdigit() else 0


def api_replay(cfg: Config, qs: dict) -> dict:
    return replay.bundle(_replay_run_dir(cfg, qs), cfg, _round_from_qs(qs))


def api_replay_agent(cfg: Config, qs: dict) -> dict:
    return agent.replay_agent(_replay_run_dir(cfg, qs), cfg, _round_from_qs(qs))


ROUTES: dict[str, Callable[[Config, dict], dict]] = {
    "overview": api_overview, "cohort": api_cohort, "runs": api_runs, "ls": api_ls,
    "run": api_run, "run/summary": api_run_summary, "run/call": api_call, "run/files": api_run_files,
    "replay": api_replay, "replay/agent": api_replay_agent,
}


# ----------------------------------------------------------------------------- HTTP

class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], cfg: Config):
        super().__init__(address, Handler)
        self.cfg = cfg


class Handler(BaseHTTPRequestHandler):
    server_version = "ora-dashboard/1"
    protocol_version = "HTTP/1.1"
    server: DashboardServer

    @property
    def cfg(self) -> Config:
        return self.server.cfg

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.cfg.verbose:
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes | str, ctype: str, extra: dict | None = None) -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        if len(body) > 4096 and "gzip" in (self.headers.get("Accept-Encoding") or "") and \
                not ctype.startswith("image/"):
            body = gzip.compress(body, 5)
            extra = dict(extra or {}, **{"Content-Encoding": "gzip"})
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":")),
                   "application/json; charset=utf-8", {"Cache-Control": "no-store"})

    def do_HEAD(self) -> None:  # noqa: N802 - http.server's naming
        self.do_GET()

    def _read_only(self) -> None:
        self._json(405, {"error": "the dashboard is read-only"})

    do_POST = do_PUT = do_DELETE = do_PATCH = _read_only  # noqa: N815

    def do_GET(self) -> None:  # noqa: N802
        try:
            u = urlsplit(self.path)
            path = u.path
            qs = parse_qs(u.query)
            i = path.rfind("/api/")
            if i >= 0:
                route = path[i + 5:].strip("/")
                if route == "file":
                    return self._file(qs)
                if route in ("cloud", "mesh"):
                    return self._binary(route, qs)
                if route == "scene-trace":
                    return self._scene_trace(qs)
                if route in ("piimg", "blob"):
                    return self._agent_binary(route, qs)
                fn = ROUTES.get(route)
                if fn is None:
                    return self._json(404, {"error": "unknown api route", "route": route})
                return self._json(200, fn(self.cfg, qs))
            k = path.rfind("/robot/")
            if k >= 0:     # robot/<id>/<file>: a robot description's URDF, meshes and textures (robots.py)
                robot_id, _, rel = path[k + 7:].partition("/")
                p = robots.robot_file(robot_id, rel, self.cfg)
                if not p:
                    flag = robots.FLAGS.get(robot_id, f"--robodojo-assets or {ASSET_ROOT_ENV}")
                    return self._json(404, {"error": f"robot description file not found (set {flag})"})
                ctype = "text/xml" if p.endswith(".urdf") else (mimetypes.guess_type(p)[0] or "application/octet-stream")
                with open(p, "rb") as f:
                    return self._send(200, f.read(), ctype, {"Cache-Control": "private, max-age=86400"})
            j = path.rfind("/static/")
            if j >= 0:
                return self._static(path[j + 8:])
            return self._static("index.html")
        except ApiError as e:
            return self._json(e.code, {"error": e.msg, **e.extra})
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as e:  # never kill the thread on a bad request
            try:
                return self._json(500, {"error": "%s: %s" % (type(e).__name__, e)})
            except Exception:
                return None

    def _binary(self, route: str, qs: dict) -> None:
        """Point clouds and object meshes for the 3D replay (built by replay.py, cached under --cache)."""
        if route == "cloud":
            p = replay.cloud(_replay_run_dir(self.cfg, qs), q1(qs, "obs") or "", self.cfg)
        else:
            p = replay.mesh(q1(qs, "usd"), self.cfg)
        if not p:
            return self._json(404, {"error": "not found"})
        with open(p, "rb") as f:
            return self._send(200, f.read(), "application/octet-stream", {"Cache-Control": "private, max-age=3600"})

    def _scene_trace(self, qs: dict) -> None:
        """A file of the run's scene trace (poses, a body's mesh), only from inside its trace directory."""
        p = scene_trace.trace_file(_replay_run_dir(self.cfg, qs), q1(qs, "file") or "", self.cfg)
        if not p:
            return self._json(404, {"error": "not found"})
        ctype = "model/gltf-binary" if p.endswith(".glb") else "application/octet-stream"
        with open(p, "rb") as f:
            return self._send(200, f.read(), ctype, {"Cache-Control": "private, max-age=3600"})

    def _agent_binary(self, route: str, qs: dict) -> None:
        """An image the agent was shown (decoded from the Pi session) or one version of a file it wrote."""
        rd = _run_dir_from_qs(self.cfg, qs)
        data: bytes | None
        ctype: str | None
        if route == "piimg":
            try:
                data, ctype = agent.pi_image(rd, int(q1(qs, "off") or ""), int(q1(qs, "ln") or ""), int(q1(qs, "k") or ""))
            except (TypeError, ValueError):
                data, ctype = None, None
        else:
            p = agent.blob_path(rd, q1(qs, "sha") or "")
            data, ctype = (Path(p).read_bytes(), "text/plain; charset=utf-8") if p else (None, None)
        if data is None:
            return self._json(404, {"error": "not found"})
        return self._send(200, data, ctype or "application/octet-stream", {"Cache-Control": "private, max-age=86400"})

    def _static(self, rel: str) -> None:
        if ".." in rel or rel.startswith("/") or not re.match(r"^[A-Za-z0-9._/-]+$", rel):
            return self._json(404, {"error": "not found"})
        p = SHARED_CSS if rel == "base.css" else STATIC_DIR / rel
        if not p.is_file():
            return self._json(404, {"error": "not found"})
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, p.read_bytes(), ctype, {"Cache-Control": "no-cache"})

    def _file(self, qs: dict) -> None:
        p = runs.safe_path(self.cfg, q1(qs, "path"))
        if not p:
            return self._json(403, {"error": "path not allowed"})
        if not os.path.isfile(p):
            return self._json(404, {"error": "no such file"})
        st = os.stat(p)
        etag = '"%x-%x"' % (st.st_mtime_ns, st.st_size)
        ext = os.path.splitext(p)[1].lower()
        if ext in TEXT_EXT or q1(qs, "text") == "1":
            ctype = "text/plain; charset=utf-8"
        else:
            ctype = mimetypes.guess_type(p)[0] or "application/octet-stream"
        cache = "private, max-age=3600" if ctype.startswith("image/") else "no-cache"
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", cache)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        tail = q1(qs, "tail")
        if tail:
            try:
                nbytes = max(1024, min(int(tail), 8 << 20))
            except ValueError:
                return self._json(400, {"error": "tail must be an integer"})
            return self._send(200, tail_text(p, nbytes) or "", "text/plain; charset=utf-8", {"Cache-Control": "no-cache"})
        size = st.st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(p, "rb") as f:
            remaining = size
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


# ----------------------------------------------------------------------------- command line

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ora-viewer dashboard", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", required=True, type=Path,
                    help="outputs root holding <experiment>/runs/<run_id>/, or one experiment directory")
    ap.add_argument("--experiment", help="show only this experiment")
    ap.add_argument("--condition", metavar="PLAN_PATH",
                    help="dotted plan path whose value labels each run's condition (e.g. policy.tool_surface.profile); "
                         "default: code mode on (code) or off (tools)")
    ap.add_argument("--hide-cohorts", metavar="REGEX",
                    help="cohorts whose name matches stay out of the overview until shown there; nothing is deleted")
    ap.add_argument("--host", default="127.0.0.1", help="bind address; keep the default for local-only viewing")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"HTTP listen port (default {DEFAULT_PORT})")
    ap.add_argument("--robodojo-assets", metavar="DIR",
                    help=f"RoboDojo's Assets directory for the 3D replay (Robots/x5, Object/); default ${ASSET_ROOT_ENV}")
    ap.add_argument("--robotwin-root", metavar="DIR",
                    help="a RoboTwin checkout for the 3D replay's aloha-agilex robot "
                         f"(assets/embodiments/aloha-agilex); default ${ROBOTWIN_ROOT_ENV}")
    ap.add_argument("--scene-traces", metavar="DIR",
                    help="a directory of scene traces (<run_id>/, written by `ora replay RUN --scene`); "
                         "default: scene-trace/ beside each run's runs/")
    ap.add_argument("--mesh-python", default=sys.executable,
                    help="interpreter with usd-core for object meshes (default: this interpreter)")
    ap.add_argument("--cache", default=None, help=f"derived point clouds and meshes (default {default_cache_dir()})")
    ap.add_argument("--verbose", action="store_true", help="log every request to stderr")
    return ap


def config_from_args(args: argparse.Namespace) -> Config:
    if not args.outputs.is_dir():
        raise SystemExit(f"ora-viewer dashboard: --outputs {args.outputs} is not a directory")
    if args.hide_cohorts is not None:
        try:
            re.compile(args.hide_cohorts)
        except re.error as exc:
            raise SystemExit(f"ora-viewer dashboard: --hide-cohorts {args.hide_cohorts!r}: {exc}") from exc
    return Config.build(str(args.outputs), experiment=args.experiment, condition=args.condition,
                        hide_cohorts=args.hide_cohorts,
                        asset_root=args.robodojo_assets, robotwin_root=args.robotwin_root,
                        scene_traces=args.scene_traces, mesh_python=args.mesh_python,
                        cache=os.path.abspath(args.cache) if args.cache else default_cache_dir(),
                        verbose=args.verbose)


def make_server(cfg: Config, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> DashboardServer:
    return DashboardServer((host, port), cfg)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    srv = make_server(cfg, args.host, args.port)
    host, port = srv.server_address[:2]
    sys.stderr.write("run dashboard on http://%s:%d/ (outputs %s, pid %d)\n" % (host, port, cfg.outputs, os.getpid()))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0
