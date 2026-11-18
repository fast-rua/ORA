"""``ora cameras``: a run's cameras in the terminal, beside its conversation.

    ora cameras [RUN] [--last] [--protocol auto|kitty|iterm2] [--interval S] [--once]

The page of ``ora web`` shows a run's cameras next to its conversation; a terminal
shows them in a pane of its own: run Pi's terminal (``ora play``, ``ora attach``) in
one pane and ``ora cameras`` in the pane beside it. It draws one picture, the run's
cameras two by two with their names, the capture's phase and action and how old it
is, and redraws it whenever the run captures a new frame
(``native/execution-capture/timeline.jsonl``: at every observation and every action's
end). The line under it says which observation step the agent last read
(``policy-observations``).

The picture goes to the terminal through its own image protocol: kitty's (kitty,
Ghostty) or iTerm2's (iTerm2, WezTerm). A terminal with neither (macOS Terminal) is
refused with that reason. Inside tmux the picture passes through to the terminal
outside, which needs ``set -g allow-passthrough on``; the command says so when it is
off. ``--protocol`` names the protocol when the terminal cannot be recognised (over
SSH, say). Nothing is written to the run; ``q`` or Ctrl-C ends the command.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

TIMELINE = Path("native") / "execution-capture" / "timeline.jsonl"
CAPTURE = TIMELINE.parent
OBSERVATIONS = "policy-observations"
GAP_PX = 6
LABEL_PX = 22
CHUNK = 4096


class CamerasError(RuntimeError):
    """The cameras cannot be shown here: no run, no capture, or a terminal without images."""


def register(sub: Any) -> None:
    parser = sub.add_parser("cameras", help="a run's cameras in the terminal, beside its conversation",
                            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", help="run ID or directory (default: the live run)")
    parser.add_argument("--last", action="store_true", help="the most recent live run when several are")
    parser.add_argument("--protocol", choices=("auto", "kitty", "iterm2"), default="auto",
                        help="the terminal's image protocol (default: recognised from the terminal)")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between looks for a new frame")
    parser.add_argument("--once", action="store_true", help="draw the latest frame once and end")


# ---------------------------------------------------------------- the terminal
def protocol(environ: dict[str, str] | None = None) -> str | None:
    """The image protocol of the terminal outside any tmux: ``kitty``, ``iterm2`` or None."""

    env = os.environ if environ is None else environ
    term, program = env.get("TERM", "").lower(), env.get("TERM_PROGRAM", "").lower()
    if env.get("KITTY_WINDOW_ID") or "kitty" in term or "ghostty" in term or program == "ghostty":
        return "kitty"
    if env.get("ITERM_SESSION_ID") or program in {"iterm.app", "wezterm"} or env.get("LC_TERMINAL") == "iTerm2" \
            or env.get("WEZTERM_PANE"):
        return "iterm2"
    return None


def _outer_environment() -> dict[str, str]:
    """Inside tmux, the client terminal's variables as tmux recorded them, over this process's own."""

    env = dict(os.environ)
    if not env.get("TMUX"):
        return env
    try:
        listed = subprocess.run(["tmux", "show-environment", "-g"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return env
    for line in listed.splitlines():
        if "=" in line and not line.startswith("-"):
            key, _, value = line.partition("=")
            env.setdefault(key, value)
            if key in {"TERM_PROGRAM", "LC_TERMINAL", "KITTY_WINDOW_ID", "ITERM_SESSION_ID", "WEZTERM_PANE"}:
                env[key] = value
    return env


def tmux_passthrough() -> bool | None:
    """Inside tmux: whether it passes images through (``allow-passthrough``); None outside tmux."""

    if not os.environ.get("TMUX"):
        return None
    try:
        value = subprocess.run(["tmux", "show", "-gv", "allow-passthrough"], capture_output=True, text=True,
                               timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return value in {"on", "all"}


def _through_tmux(sequence: str) -> str:
    return "\x1bPtmux;" + sequence.replace("\x1b", "\x1b\x1b") + "\x1b\\"


def kitty_image(png: bytes, *, columns: int, rows: int, image_id: int = 1) -> list[str]:
    """kitty's graphics protocol: the PNG transmitted and placed at the cursor in ``columns`` x ``rows`` cells."""

    data = base64.standard_b64encode(png).decode("ascii")
    chunks = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)] or [""]
    out = []
    for index, chunk in enumerate(chunks):
        more = 1 if index < len(chunks) - 1 else 0
        keys = f"a=T,f=100,i={image_id},q=2,c={columns},r={rows},C=1,m={more}" if index == 0 else f"m={more}"
        out.append(f"\x1b_G{keys};{chunk}\x1b\\")
    return out


def iterm2_image(png: bytes, *, columns: int, rows: int) -> list[str]:
    data = base64.standard_b64encode(png).decode("ascii")
    return [f"\x1b]1337;File=inline=1;size={len(png)};width={columns};height={rows};preserveAspectRatio=1:{data}\x07"]


# ---------------------------------------------------------------- the run
def capture_root(run: Path) -> Path:
    return run / CAPTURE


def latest_frame(run: Path) -> dict[str, Any] | None:
    """The run's last captured frame record, or None before the first."""

    try:
        with open(run / TIMELINE, "rb") as timeline:
            timeline.seek(0, os.SEEK_END)
            size = timeline.tell()
            timeline.seek(max(0, size - 65536))
            lines = timeline.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and isinstance(record.get("images"), dict):
            return record
    return None


def agent_step(run: Path) -> int | None:
    """The newest observation step the agent was given."""

    folder = run / OBSERVATIONS
    steps = []
    for path in folder.glob("*observation-*") if folder.is_dir() else ():
        tail = path.name.rsplit("observation-", 1)[-1]
        if tail.isdigit():
            steps.append(int(tail))
    return max(steps) if steps else None


def ordered_roles(run: Path, frame: dict[str, Any]) -> list[str]:
    """The frame's cameras in the capture's panel order (``manifest.json``), then any others."""

    try:
        panels = json.loads((capture_root(run) / "manifest.json").read_text()).get("camera_panels") or []
    except (OSError, ValueError):
        panels = []
    return [r for r in panels if r in frame["images"]] + [r for r in frame["images"] if r not in panels]


def compose(run: Path, frame: dict[str, Any], *, width_px: int, height_px: int) -> bytes:
    """The frame's cameras two by two, each named, as one PNG of about ``width_px`` x ``height_px``."""

    from PIL import Image, ImageDraw, ImageFont, ImageOps

    roles = ordered_roles(run, frame)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:                                   # Pillow before 10.1: the fixed bitmap font
        font = ImageFont.load_default()
    cells = 2 if len(roles) > 1 else 1
    grid_rows = (len(roles) + cells - 1) // cells
    cell_w = max(80, (width_px - GAP_PX * (cells - 1)) // cells)
    cell_h = max(60, (height_px - GAP_PX * (grid_rows - 1)) // grid_rows)
    canvas = Image.new("RGB", (cell_w * cells + GAP_PX * (cells - 1), cell_h * grid_rows + GAP_PX * (grid_rows - 1)),
                       (16, 16, 18))
    draw = ImageDraw.Draw(canvas)
    for index, role in enumerate(roles):
        x, y = (index % cells) * (cell_w + GAP_PX), (index // cells) * (cell_h + GAP_PX)
        try:
            picture = Image.open(capture_root(run) / frame["images"][role]["path"]).convert("RGB")
        except (OSError, KeyError, TypeError):
            draw.text((x + 8, y + 8), f"{role}: no picture", fill=(200, 200, 200), font=font)
            continue
        picture = ImageOps.contain(picture, (cell_w, cell_h - LABEL_PX))
        canvas.paste(picture, (x + (cell_w - picture.width) // 2, y + LABEL_PX))
        draw.text((x + 6, y + 2), role, fill=(235, 235, 235), font=font)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def caption(run: Path, frame: dict[str, Any], now: float) -> str:
    age = now - float(frame.get("captured_unix_s") or now)
    action = frame.get("action") or "-"
    step = agent_step(run)
    return (f"{run.name} · frame {frame.get('frame_index')} · {frame.get('phase')} · action {action} · "
            f"{age:.0f} s ago · agent read step {'-' if step is None else step} · q quits")


# ---------------------------------------------------------------- drawing
def _terminal_px(columns: int, rows: int) -> tuple[int, int]:
    """The terminal's size in pixels when it says (TIOCGWINSZ), else an estimate from its cells."""

    try:
        import fcntl
        import struct
        import termios

        packed = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
        _, _, width, height = struct.unpack("HHHH", packed)
        if width and height:
            return width, height
    except (OSError, ImportError, ValueError):
        pass
    return columns * 9, rows * 18


def draw(run: Path, frame: dict[str, Any], kind: str, *, tmux: bool, out: Any = None) -> None:
    out = sys.stdout if out is None else out
    columns, rows = shutil.get_terminal_size((100, 32))
    image_rows = max(4, rows - 2)
    width_px, height_px = _terminal_px(columns, rows)
    png = compose(run, frame, width_px=width_px, height_px=int(height_px * image_rows / rows))
    sequences = (kitty_image(png, columns=columns, rows=image_rows) if kind == "kitty"
                 else iterm2_image(png, columns=columns, rows=image_rows))
    if kind == "kitty":
        sequences = ["\x1b_Ga=d,d=i,i=1,q=2\x1b\\", *sequences]
    if tmux:
        sequences = [_through_tmux(s) for s in sequences]
    out.write("\x1b[H\x1b[2J" + "".join(sequences) + f"\x1b[{image_rows + 1};1H\x1b[2K"
              + caption(run, frame, time.time())[:columns])
    out.flush()


# ---------------------------------------------------------------- the command
def _run_root(value: str | None, last: bool) -> Path:
    from ora.orchestrator import paths

    from .interactive import resolve_run
    from .web import WebError, live_run

    if value:
        return resolve_run(value)
    try:
        return live_run(None, last=True if last else False)
    except WebError as exc:
        captured = [root for root in paths.iter_run_roots() if (root / TIMELINE).exists()]
        if not captured:
            raise CamerasError(f"{exc}; and no run here has captured frames") from None
        return max(captured, key=lambda root: (root / TIMELINE).stat().st_mtime)


def _keys(timeout: float) -> Iterable[str]:
    """Characters typed within ``timeout`` (none when stdin is no terminal)."""

    import select

    if not sys.stdin.isatty():
        time.sleep(timeout)
        return ()
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if ready else ()


def run(args: argparse.Namespace) -> int:
    try:
        root = _run_root(args.run, args.last)
        kind = protocol(_outer_environment()) if args.protocol == "auto" else args.protocol
        if kind is None:
            raise CamerasError("this terminal shows no images: use kitty, Ghostty, iTerm2 or WezTerm "
                               "(or name the protocol with --protocol when it is one of them over SSH)")
        passthrough = tmux_passthrough()
        if passthrough is False:
            raise CamerasError("tmux does not pass images through: `tmux set -g allow-passthrough on`, then again")
    except CamerasError as exc:
        print(f"ora cameras: {exc}", file=sys.stderr)
        return 1
    frame = latest_frame(root)
    if args.once:
        if frame is None:
            print(f"ora cameras: {root.name} has captured no frame yet", file=sys.stderr)
            return 1
        draw(root, frame, kind, tmux=bool(passthrough))
        print()
        return 0
    import termios
    import tty

    saved = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
    shown: tuple[Any, ...] | None = None
    try:
        if saved is not None:
            tty.setcbreak(sys.stdin)
        sys.stdout.write("\x1b[?25l")
        while True:
            frame = latest_frame(root)
            size = shutil.get_terminal_size((100, 32))
            key = None if frame is None else (frame.get("capture_index"), size, agent_step(root), int(time.time()) // 5)
            if frame is None and shown is None:
                sys.stdout.write("\x1b[H\x1b[2J" + f"{root.name}: waiting for the first captured frame…")
                sys.stdout.flush()
                shown = ()
            elif frame is not None and key != shown:
                draw(root, frame, kind, tmux=bool(passthrough))
                shown = key
            if "q" in _keys(args.interval):
                return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if kind == "kitty":
            clear = "\x1b_Ga=d,d=i,i=1,q=2\x1b\\"
            sys.stdout.write(_through_tmux(clear) if passthrough else clear)
        sys.stdout.write("\x1b[?25h\n")
        sys.stdout.flush()
        if saved is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)
