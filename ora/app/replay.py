"""`ora replay`: a finished run in the browser, replayed from its evidence, or made into a video.

    ora replay RUN [--speed 16] [--motion-speed 1|same] [--from S] [--port N] [--no-browser]
                   [--view compact|detailed] [--expand-thinking] [--lang en|zh] [--publish]
    ora replay RUN --program DIR | --check OUT [--resource-profile P]
    ora replay RUN --scene [--scene-traces DIR] [--resource-profile P]
    ora replay RUN --incident N --report OUT [--stack-run RUN] [--branch both|recorded|current] [--preview]
                   [--resource-profile P]
    ora replay RUN --export OUT.mp4 [--speed 16] [--motion-speed 1|same] [--fps 30] [--size 1920x1080] [--scale 2] [--zoom 1]
                   [--from S] [--to S] [--workers N] [--publish] [--ffmpeg PATH] [--chromium PATH]
                   [--export-python PY]

The page is `ora web`'s own (ora-pi/web), shown at a clock over the run's recorded
time: its web server rebuilds what the live page showed from the run's evidence
(Pi's session, the Supervisor's events, the execution capture and the operator
views: ora-pi/web/server/replay.ts) and attaches to nothing. The gateway in front
is `ora web`'s (launch token, cookie, loopback Host), passing the page and its one
route, GET /api/replay. Nothing in the run is written, and a live run is refused
(`ora web` shows it).

The replay has two rates: the waiting and thinking between the robot's actions play at
`--speed` (16x by default), and each action, from its start to its end with any settling
frames, plays its execution capture by the frames' simulated time at `--motion-speed` (1x by
default: real simulated time, never shorter than the action at `--speed`). Only captured frames
are shown, each held until the next; nothing is interpolated. `--motion-speed same` keeps one
speed throughout. The page has the same switches. `--export` draws the replay into a video at
those rates: a headless Chromium opens the page, the page's playback is set to each frame's
moment in turn (a virtual clock, so the video keeps the replay's rates whatever the machine's
speed and is the same every time), each frame is a screenshot drawn at `--scale` device pixels
per CSS pixel and scaled down to `--size`, and ffmpeg encodes H.264 (yuv420p, CRF 16). A corner
of the page says what plays and how fast ("robot motion · real time", "thinking · 16×"); the
report beside the video lists every stretch with its rate and times. `--publish` leaves the
run's ID and absolute paths out of the page.

`--program` and `--check` work on the run's replay program: its agent actions as code
that reproduces the run (`environment/replay_program.py`). Every run sealed with this
version carries it in `replay-program/`; `--program DIR` writes it for any run, outside
the run. `--check OUT` re-executes it on the run's simulator (the plan's interpreter,
the caller's CUDA_VISIBLE_DEVICES first), compares every action's end pose with its
receipt bit for bit and the official verdict with the run's, and writes
OUT/replay-program-check.json. A RoboCasa365 check also compares the base's end pose and a
hash of the whole scene at every action. A RoboDojo check runs the qualification control
replay on `--resource-profile` and needs the run's host marker. A RoboLab check runs in the
environment worker a run uses, on one card. A run that records no per-tick controls (a
RoboCasa365 or RoboLab run sealed before they were recorded) has no program, so `--check`
refuses it and `--program DIR` writes DIR/unavailable.json.

`--scene` runs the same check on a RoboTwin, LIBERO-Pro, RoboCasa365, RoboDojo or RoboLab run and, when it passes,
keeps the scene of that re-execution for the run dashboard's 3D replay: every body's world pose (about 50 Hz on
RoboTwin and LIBERO-Pro, every fifth SAPIEN step and every tenth MuJoCo step, every native action on
RoboDojo, 25 Hz, RoboCasa365, 20 Hz, and RoboLab, 15 Hz, plus each action's last step) and each body's visual mesh,
in `scene-trace/<run_id>/` beside the run's `runs/` directory or under `--scene-traces DIR` (`environment/scene_trace.py`). A
RoboDojo run needs `--resource-profile`, as its `--check` does. The run itself is not written. A run
whose re-execution diverges gets no trace; the report beside the traces says why. A re-execution that matches
the run bit for bit keeps its trace even if its verdict reads differently: the result and the report say
`verdict_differs`. Scene poses are
private simulator state: the trace is for a developer and reaches no policy.

`--incident N` replays the run up to action N and runs N again with no model, to see what
this checkout does with a move that once went wrong (`orchestrator/evaluation/incident_replay.py`).
On RoboTwin and LIBERO-Pro the `recorded` branch runs the recorded controls and the `current`
branch submits the recorded target to the motion stack of `--stack-run`, a run sealed on this
checkout such as a merge smoke; `--stack-run` must share the source run's simulator revisions.
On RoboDojo `--resource-profile` names a slot on the run's host and the run must be a toolkit
run: only `current` runs, through `ora toolkit replay --live --execute-action`, on the default
baseline's tool surface, planner and timing, and it needs a clean checkout.
OUT/incident-replay.json says whether the prefix matched bit for bit, each branch's receipt or
refusal, the stack it ran on, and which bodies moved. Scene poses are private simulator state:
the report is for a developer and reaches no policy.

The export needs Playwright for Python with its Chromium (`pip install
playwright==1.56.0 && playwright install chromium`, or `uv` on PATH, which runs it
as `uv run --no-project --with playwright==1.56.0`) and ffmpeg (on PATH,
ORA_FFMPEG or --ffmpeg). Neither is a dependency of ORA.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .web import WebError, WebServer, rpc_endpoint

#: The exporter (stdlib and Playwright only), run by the Python that has Playwright.
EXPORTER = Path(__file__).with_name("replay_export.py")
PLAYWRIGHT_VERSION = "1.56.0"
DEFAULT_SPEED = 16.0


class ReplayError(WebError):
    """`ora replay` cannot replay or export this run."""


def replay_run(value: str) -> Path:
    """The finished run named by an ID or a directory; a live run is refused."""

    from .interactive import resolve_run

    root = resolve_run(value)
    if not root.is_dir():
        raise ReplayError(f"no run {value!r}")
    if rpc_endpoint(root) is not None:
        raise ReplayError(f"{root.name} is live: `ora web {root.name}` shows it as it runs")
    sessions = list((root / "native" / "pi-session").glob("*.jsonl")) if (root / "native" / "pi-session").is_dir() else []
    if not sessions and not (root / "events.jsonl").is_file():
        raise ReplayError(f"{root.name} has no Pi session and no Supervisor events to replay")
    return root


def parse_size(value: str) -> tuple[int, int]:
    """`1920x1080` as (1920, 1080); even sides (H.264 in yuv420p needs them)."""

    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"size must be WIDTHxHEIGHT, not {value!r}") from None
    if not (320 <= width <= 7680 and 240 <= height <= 4320) or width % 2 or height % 2:
        raise argparse.ArgumentTypeError("size must be even sides between 320x240 and 7680x4320")
    return width, height


def parse_motion_speed(value: str) -> float | str:
    """`1`, `0.5`, ... (the motion against its simulated time) or `same` (one speed throughout)."""

    if value == "same":
        return value
    try:
        speed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"motion speed must be a positive number or `same`, not {value!r}") from None
    if not speed > 0:
        raise argparse.ArgumentTypeError("motion speed must be positive")
    return speed


def page_query(args: argparse.Namespace, *, exporting: bool) -> dict[str, str]:
    """The page's own options (lib/replay.ts replayOptions, lib/display.ts)."""

    motion = args.motion_speed
    query = {"replay": "1", "speed": f"{args.speed:g}", "motion": motion if motion == "same" else f"{motion:g}"}
    if exporting:
        query["export"] = "1"
    if args.publish:
        query["publish"] = "1"
    if args.view:
        query["view"] = args.view
    if args.expand_thinking:
        query["thinking"] = "expanded"
    if args.lang:
        query["lang"] = args.lang
    if not exporting and args.start:
        query["from"] = f"{args.start:g}"
    return query


def find_ffmpeg(explicit: str | None) -> str:
    for candidate in (explicit, os.environ.get("ORA_FFMPEG"), shutil.which("ffmpeg")):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise ReplayError("no ffmpeg: put it on PATH, set ORA_FFMPEG or pass --ffmpeg (a static build will do)")


def exporter_python(explicit: str | None) -> list[str]:
    """The command that runs the exporter with Playwright importable."""

    chosen = explicit or os.environ.get("ORA_EXPORT_PYTHON")
    if chosen:
        return [chosen]
    if importlib.util.find_spec("playwright") is not None:
        return [sys.executable]
    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "--no-project", "--with", f"playwright=={PLAYWRIGHT_VERSION}", "python"]
    raise ReplayError(f"the export needs Playwright for Python: pip install playwright=={PLAYWRIGHT_VERSION} "
                      "and `playwright install chromium`, or put uv on PATH, or pass --export-python")


def export(root: Path, args: argparse.Namespace) -> int:
    import subprocess

    out = Path(args.export).expanduser().resolve()
    if out.suffix.lower() != ".mp4":
        raise ReplayError("--export writes an .mp4 file")
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg(args.ffmpeg)
    python = exporter_python(args.export_python)
    width, height = args.size
    server = WebServer(root, port=0, replay=True, publish=args.publish).start()
    try:
        job = {
            # The link with its one-time launch token goes on stdin, never on a command line.
            "url": f"{server.url}&{urlencode(page_query(args, exporting=True))}",
            "out": str(out), "speed": args.speed, "fps": args.fps, "width": width, "height": height,
            "motion_speed": args.motion_speed, "scale": args.scale, "zoom": args.zoom, "from_s": args.start,
            "to_s": args.end, "workers": args.workers,
            "ffmpeg": ffmpeg, "chromium": args.chromium, "crf": args.crf,
            "run": None if args.publish else root.name,
        }
        motion = "one speed throughout" if args.motion_speed == "same" else f"motion at {args.motion_speed:g}x simulated time"
        print(f"ora replay: exporting {'the run' if args.publish else root.name} at {args.speed:g}x ({motion}), "
              f"{width}x{height} {args.fps} fps (drawn at {args.scale:g}x) to {out}", flush=True)
        started = time.monotonic()
        process = subprocess.run([*python, str(EXPORTER)], input=json.dumps(job), text=True)
        if process.returncode:
            return process.returncode
        report = json.loads(out.with_name(out.name + ".json").read_text(encoding="utf-8"))
        print(f"ora replay: wrote {out} ({report['width']}x{report['height']}, {report['fps']} fps, "
              f"{report['duration_s']:.1f} s of which {report['motion_s']:.1f} s robot motion, {report['frames']} frames, "
              f"{report['bytes'] / 1e6:.1f} MB) in {time.monotonic() - started:.0f} s", flush=True)
        return 0
    finally:
        server.close()


def _robodojo_control_replay(root: Path, output: Path, profile: str, *extra: str) -> None:
    """RoboDojo's `--check` and `--scene`: the qualification control replay on a slot, report under ``output``."""
    from ora.orchestrator.backends.robodojo.motion.replay import main as motion_replay

    motion_replay(["--run", str(root), "--output", str(output), "--resource-profile", profile,
                   "--mode", "metrics-after-replay", *extra])


def replay_program(root: Path, args: argparse.Namespace) -> int:
    """``--program`` writes the run's replay program; ``--check`` re-executes and checks it."""

    from ora.environment.replay_program import (
        ReplayProgramUnavailable, controller_replay_outcome, finish_check, prepare_check, robodojo_outcome,
        write_program, write_unavailable,
    )

    if args.program:
        try:
            print(json.dumps(write_program(root, args.program), indent=1))
        except ReplayProgramUnavailable as exc:
            print(json.dumps(write_unavailable(args.program, root.name, str(exc)), indent=1))
            return 1
        if not args.check:
            return 0
    root, output, program = prepare_check(root, args.check, program_dir=args.program)
    if program["backend"] == "robodojo":
        if not args.resource_profile:
            raise ReplayError("a RoboDojo --check needs --resource-profile (a qualified RoboDojo slot)")
        motion = output / "motion-replay"
        _robodojo_control_replay(root, motion, args.resource_profile)
        outcome = robodojo_outcome(program, motion)
    else:
        outcome = controller_replay_outcome(root, output, program)
    report = finish_check(root, output, program, outcome)
    print(json.dumps({"report": report["path"], "passed": report["passed"], "failures": report["failures"]}, indent=1))
    return 0 if report["passed"] else 1


def replay_scene(root: Path, args: argparse.Namespace) -> int:
    """``--scene``: the check, and the scene trace of a re-execution that passes it."""

    from ora.environment.replay_program import export_scene_trace

    profile = args.resource_profile
    result = export_scene_trace(root, args.scene_traces, robodojo_replay=(
        (lambda run, output, trace: _robodojo_control_replay(run, output, profile, "--scene-trace", str(trace)))
        if profile else None))
    print(json.dumps(result, indent=1))
    return 0 if result["passed"] else 1


def replay_incident(root: Path, args: argparse.Namespace) -> int:
    """``--incident``: the action again, from the replayed state before it."""

    from ora.environment.incident_replay_worker import BRANCHES
    from ora.orchestrator.evaluation.incident_replay import replay_incident as replay

    if args.report is None:
        raise ReplayError("--incident needs --report OUT")
    report = replay(root, args.report, args.incident, stack_run=args.stack_run,
                    branches=BRANCHES if args.branch == "both" else (args.branch,),
                    resource_profile=args.resource_profile, preview=args.preview)
    print(json.dumps({"report": report["path"], "branches": sorted(report["branches"])}, indent=1))
    return 0


def register(sub: Any) -> None:
    parser = sub.add_parser("replay", help="replay a finished run in the browser from its evidence (ORA's web "
                            "page, 16x by default), --export a video, or --check its replay program",
                            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="the finished run's ID or directory")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                        help="the speed of the waiting and thinking between the robot's actions (default 16)")
    parser.add_argument("--motion-speed", type=parse_motion_speed, default=1.0, metavar="X|same",
                        help="the speed of each action's motion against its simulated time (default 1: real time); "
                             "`same`: --speed throughout")
    parser.add_argument("--view", choices=("compact", "detailed"), help="how the conversation is drawn (default compact)")
    parser.add_argument("--expand-thinking", action="store_true", help="every thought open")
    parser.add_argument("--lang", choices=("en", "zh"), help="the page's language (default: the browser's)")
    parser.add_argument("--publish", action="store_true", help="leave the run's ID and absolute paths out of the page")
    parser.add_argument("--from", dest="start", type=float, default=0.0, metavar="S",
                        help="start S seconds into the run's recorded time")
    page = parser.add_argument_group("the page")
    page.add_argument("--port", type=int, default=0, help="loopback port (default: a free one)")
    page.add_argument("--no-browser", action="store_true", help="print the link instead of opening it")
    video = parser.add_argument_group("a video (--export)")
    video.add_argument("--export", metavar="OUT.mp4", help="draw the replay into this H.264 video and exit")
    video.add_argument("--fps", type=int, default=30, help="frames per second (default 30)")
    video.add_argument("--size", type=parse_size, default=(1920, 1080), help="the video's WIDTHxHEIGHT (default 1920x1080)")
    video.add_argument("--scale", type=float, default=2.0,
                       help="device pixels per CSS pixel the page is drawn at before scaling to --size (default 2)")
    video.add_argument("--zoom", type=float, default=1.0,
                       help="the page's zoom: its layout is --size / zoom CSS pixels (default 1)")
    video.add_argument("--to", dest="end", type=float, default=None, metavar="S",
                       help="end S seconds into the run's recorded time (default: its end)")
    video.add_argument("--workers", type=int, default=4, help="pages drawing frames side by side (default 4)")
    video.add_argument("--crf", type=int, default=16, help="H.264 quality, lower is better (default 16)")
    video.add_argument("--ffmpeg", help="ffmpeg to encode with (default: ORA_FFMPEG, then PATH)")
    video.add_argument("--chromium", help="Chromium to draw with (default: Playwright's)")
    video.add_argument("--export-python", help="a Python with Playwright (default: this one, else uv)")
    program = parser.add_argument_group("the replay program (--program, --check)")
    program.add_argument("--program", type=Path, metavar="DIR",
                         help="write the run's replay program (program.json, program.py) to DIR outside the run "
                              "and exit; a sealed run already carries it in replay-program/")
    program.add_argument("--check", type=Path, metavar="OUT",
                         help="re-execute the replay program on the run's simulator, check every action's end pose "
                              "against its receipt bit for bit and the verdict, write OUT/replay-program-check.json")
    program.add_argument("--resource-profile", help="RoboDojo --check, --scene or --incident: the qualified slot to "
                         "replay on")
    scene = parser.add_argument_group("the scene trace (--scene)")
    scene.add_argument("--scene", action="store_true",
                       help="re-execute the run as --check does on RoboTwin, LIBERO-Pro, RoboCasa365, RoboDojo or RoboLab and, if it passes, "
                            "write its scene trace for the run dashboard: every body's pose per step and visual mesh")
    scene.add_argument("--scene-traces", type=Path, metavar="DIR",
                       help="--scene: write DIR/<run_id>/ (default: scene-trace/<run_id>/ beside the run's runs/)")
    incident = parser.add_argument_group("one bad action (--incident)")
    incident.add_argument("--incident", type=int, metavar="N",
                          help="replay the run up to action N, then run N again on this checkout's motion stack; "
                               "needs --report, and --stack-run or --resource-profile")
    incident.add_argument("--report", type=Path, metavar="OUT", help="--incident: an empty directory for the report")
    incident.add_argument("--stack-run", type=Path, metavar="RUN",
                          help="--incident on RoboTwin or LIBERO-Pro: a run sealed on this checkout, whose motion "
                               "stack the action runs on")
    incident.add_argument("--branch", choices=("both", "recorded", "current"), default="both",
                          help="--incident: which branches to run (default both)")
    incident.add_argument("--preview", action="store_true",
                          help="--incident on RoboDojo: what a dry run of the action shows at that state, written to "
                               "OUT/preview.png, instead of running it")


def run(args: argparse.Namespace) -> int:
    try:
        if args.speed <= 0 or args.fps <= 0 or args.scale <= 0 or args.zoom <= 0 or args.workers <= 0:
            raise ReplayError("--speed, --fps, --scale, --zoom and --workers must be positive")
        root = replay_run(args.run)
        if args.scene:
            return replay_scene(root, args)
        if args.incident is not None:
            return replay_incident(root, args)
        if args.program or args.check:
            return replay_program(root, args)
        if args.export:
            return export(root, args)
        server = WebServer(root, port=args.port, replay=True, publish=args.publish).start()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ora replay: {exc}", file=sys.stderr)
        return 2
    try:
        name = "a recorded run" if args.publish else root.name
        link = f"{server.url}&{urlencode(page_query(args, exporting=False))}"
        print(f"ORA replay of {name}: {link} (Ctrl+C stops it)", flush=True)
        remote = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))
        if not args.no_browser and not remote:
            webbrowser.open_new_tab(link)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()


__all__ = ["ReplayError", "export", "exporter_python", "find_ffmpeg", "page_query", "parse_motion_speed",
           "parse_size", "register", "replay_program", "replay_run", "replay_scene", "run"]

