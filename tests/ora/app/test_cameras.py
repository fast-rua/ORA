"""``ora cameras``: the terminal's image protocol is recognised (kitty for kitty and Ghostty, iTerm2's for
iTerm2 and WezTerm, none for macOS Terminal), the picture is the last captured frame's cameras two by two
in the capture's panel order, and a terminal without images is refused with what to use instead."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from ora.app import cameras
from ora.app.main import main


def _run(tmp_path: Path, frames: int = 2) -> Path:
    run = tmp_path / "ora-play-play-test"
    capture = run / cameras.CAPTURE
    (capture / "frames").mkdir(parents=True)
    (capture / "manifest.json").write_text(json.dumps({"camera_panels": ["overhead", "front"]}))
    lines = []
    for index in range(frames):
        images = {}
        for role, colour in (("front", (200, 0, 0)), ("overhead", (0, 0, 200))):
            path = f"frames/{index:08d}-{role}.png"
            Image.new("RGB", (64, 48), colour).save(capture / path)
            images[role] = {"path": path}
        lines.append(json.dumps({"capture_index": index, "frame_index": index, "phase": "action_end",
                                 "action": "move_pose", "captured_unix_s": 100.0 + index, "images": images}))
    (capture / "timeline.jsonl").write_text("\n".join(lines) + "\n{broken")
    (run / cameras.OBSERVATIONS / "yam:t:on:round-0:observation-0007").mkdir(parents=True)
    return run


@pytest.mark.parametrize("environ,expected", [
    ({"TERM": "xterm-kitty"}, "kitty"), ({"TERM": "xterm-ghostty"}, "kitty"), ({"KITTY_WINDOW_ID": "1"}, "kitty"),
    ({"TERM_PROGRAM": "iTerm.app"}, "iterm2"), ({"TERM_PROGRAM": "WezTerm"}, "iterm2"), ({"LC_TERMINAL": "iTerm2"}, "iterm2"),
    ({"TERM": "xterm-256color", "TERM_PROGRAM": "Apple_Terminal"}, None), ({}, None)])
def test_the_terminals_image_protocol_is_recognised(environ, expected) -> None:
    assert cameras.protocol(environ) == expected


def test_the_picture_is_the_last_frame_in_panel_order(tmp_path) -> None:
    run = _run(tmp_path)
    frame = cameras.latest_frame(run)
    assert frame["frame_index"] == 1 and cameras.agent_step(run) == 7
    assert cameras.ordered_roles(run, frame) == ["overhead", "front"]
    picture = Image.open(io.BytesIO(cameras.compose(run, frame, width_px=400, height_px=200)))
    assert picture.getpixel((100, 120))[2] > 150 and picture.getpixel((300, 120))[0] > 150    # blue, then red
    assert "frame 1 · action_end · action move_pose · 3 s ago · agent read step 7" in cameras.caption(run, frame, 104.0)


def test_kitty_and_iterm2_sequences_carry_the_picture_whole(tmp_path) -> None:
    png = b"\x89PNG" + bytes(range(256)) * 40
    chunks = cameras.kitty_image(png, columns=80, rows=20)
    assert len(chunks) > 1 and chunks[0].startswith("\x1b_Ga=T,f=100,i=1,q=2,c=80,r=20,C=1,m=1;")
    assert chunks[-1].startswith("\x1b_Gm=0;")
    payload = "".join(chunk.split(";", 1)[1][:-2] for chunk in chunks)
    assert base64.standard_b64decode(payload) == png
    [inline] = cameras.iterm2_image(png, columns=80, rows=20)
    assert inline.startswith(f"\x1b]1337;File=inline=1;size={len(png)};width=80;height=20") and inline.endswith("\x07")
    assert cameras._through_tmux("\x1b_Gx\x1b\\") == "\x1bPtmux;\x1b\x1b_Gx\x1b\x1b\\\x1b\\"


def test_a_terminal_without_images_is_refused_with_what_to_use(tmp_path, monkeypatch, capsys) -> None:
    run = _run(tmp_path)
    monkeypatch.delenv("TMUX", raising=False)
    for key in ("KITTY_WINDOW_ID", "ITERM_SESSION_ID", "LC_TERMINAL", "WEZTERM_PANE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    assert main(["cameras", str(run), "--once"]) == 1
    assert "use kitty, Ghostty, iTerm2 or WezTerm" in capsys.readouterr().err


def test_once_draws_the_frame_through_the_named_protocol(tmp_path, monkeypatch, capsys) -> None:
    run = _run(tmp_path)
    monkeypatch.delenv("TMUX", raising=False)
    assert main(["cameras", str(run), "--once", "--protocol", "kitty"]) == 0
    out = capsys.readouterr().out
    assert "\x1b_Ga=T,f=100" in out and "agent read step 7" in out
