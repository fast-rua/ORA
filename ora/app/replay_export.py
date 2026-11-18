"""Draw a recorded run's replay page into a video (run by `ora replay --export`).

Python's standard library and Playwright only (it runs under whichever Python has
Playwright: `ora replay` chooses it), never ORA's own modules. One JSON job on stdin:

    url       the replay page through `ora web`'s gateway, with its one-time launch token (its link
              names the speed and the motion speed: the page builds the replay's stretches)
    out       the .mp4 to write (a report beside it: <out>.json)
    speed     the waiting's speed                       fps      frames per second
    motion_speed  the robot's motion against its simulated time, or "same" (one speed throughout)
    width     height: the video's size                  scale    device pixels per CSS pixel
    zoom      the page's zoom (its layout: size / zoom)  crf      H.264 quality
    from_s    to_s: the stretch of recorded time, in seconds from the replay's start (to_s null: its end)
    workers   pages drawing frames side by side          ffmpeg   chromium (null: Playwright's)

The page's clock is set to each frame's playback moment in turn (`window.__oraReplay
.seekPlayback`, which returns once the page has drawn it and its images are decoded), so the
video keeps the replay's rates whatever this machine's speed, and is the same every time:
frame i shows the playback moment first + i / fps, where the waiting plays at `speed` and
each action's motion by its frames' simulated time at `motion_speed` (lib/replay.ts
buildTimeline; only captured frames, each held until the next). The report lists every
stretch of the video with its rate and its recorded and simulated times. A frame the page
draws exactly as the one before (its digest says so) reuses that screenshot. Each worker draws one
stretch of frames into its own segment (lossless PNG screenshots at `scale`, taken through
the DevTools protocol with its fast PNG encoder, scaled to the video's size with Lanczos,
H.264 yuv420p); the segments are joined without re-encoding.
"""

from __future__ import annotations

import base64
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

READY_TIMEOUT_MS = 180_000
CHROMIUM_ARGS = ["--force-color-profile=srgb", "--hide-scrollbars", "--disable-lcd-text", "--font-render-hinting=none"]


def frames_between(first_ms: float, last_ms: float, fps: int) -> tuple[float, float, int]:
    """(first playback moment, playback ms per frame, frame count) from `first_ms` to `last_ms`."""

    step = 1000 / fps
    count = max(1, math.ceil((last_ms - first_ms) / step - 1e-9)) if last_ms > first_ms else 1
    return first_ms, step, count


def stretch_report(segments: list[dict[str, Any]], first_ms: float, last_ms: float, speed: float,
                   motion: Any) -> list[dict[str, Any]]:
    """The video's stretches between two playback moments: their rate, video times, recorded and simulated times."""

    report = []
    for segment in segments:
        begin, end = max(segment["playFrom"], first_ms), min(segment["playTo"], last_ms)
        if end <= begin:
            continue
        length = segment["playTo"] - segment["playFrom"]

        def recorded(position: float, segment: dict[str, Any] = segment, length: float = length) -> float:
            share = (position - segment["playFrom"]) / length if length > 0 else 1.0
            return segment["from"] + share * (segment["to"] - segment["from"])

        item = {"kind": segment["kind"], "video_from_s": round((begin - first_ms) / 1000, 3),
                "video_to_s": round((end - first_ms) / 1000, 3),
                "recorded_from_ms": round(recorded(begin)), "recorded_to_ms": round(recorded(end))}
        if segment["kind"] == "motion" and motion != "same":
            def simulated(position: float, segment: dict[str, Any] = segment) -> float:
                return min(segment["simTo"], segment["simFrom"] + (position - segment["playFrom"]) * motion / 1000)

            item.update(rate=f"{motion:g}x simulated time" + (" (real time)" if motion == 1 else ""),
                        sim_from_s=round(simulated(begin), 3), sim_to_s=round(simulated(end), 3),
                        action=segment.get("action"), step=segment.get("step"))
            if segment.get("settleFrom") is not None and segment.get("settleTo") is not None:
                # The settling window after the action (post-action-settle.jsonl), in simulated time.
                item.update(settle_from_s=round(segment["settleFrom"], 3), settle_to_s=round(segment["settleTo"], 3),
                            settle_ended=segment.get("settleEnded"))
        else:
            item["rate"] = f"{speed:g}x recorded time"
        report.append(item)
    return report


def stretches(count: int, workers: int) -> list[tuple[int, int]]:
    """Frame ranges [first, last) for each worker, contiguous and in order."""

    workers = max(1, min(workers, count))
    size = math.ceil(count / workers)
    return [(index, min(count, index + size)) for index in range(0, count, size)]


def layout(job: dict[str, Any]) -> tuple[int, int, float]:
    """The page's CSS size and its device pixel ratio: (width / zoom, height / zoom, scale * zoom)."""

    zoom = float(job.get("zoom") or 1.0)
    return round(job["width"] / zoom), round(job["height"] / zoom), float(job["scale"]) * zoom


def png_size(data: bytes) -> tuple[int, int] | None:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


class Camera:
    """Screenshots of the whole page at its device pixels: the DevTools protocol's fast PNG encoder
    (lossless, several times faster than the default one), else Playwright's own screenshot."""

    def __init__(self, page: Any, job: dict[str, Any]) -> None:
        self.page = page
        width, height, ratio = layout(job)
        self.expected = (round(width * ratio), round(height * ratio))
        self.params = {"format": "png", "optimizeForSpeed": True, "captureBeyondViewport": False,
                       "clip": {"x": 0, "y": 0, "width": width, "height": height, "scale": ratio}}
        self.session: Any = page.context.new_cdp_session(page)

    def shoot(self) -> bytes:
        if self.session is not None:
            data = base64.b64decode(self.session.send("Page.captureScreenshot", self.params)["data"])
            if png_size(data) == self.expected:
                return data
            self.session = None  # this Chromium draws the clip otherwise: Playwright's screenshot from here
        return self.page.screenshot(type="png", animations="disabled", caret="hide")


def open_page(playwright: Any, job: dict[str, Any], url: str, cookies: list | None = None) -> tuple[Any, Any]:
    width, height, ratio = layout(job)
    browser = playwright.chromium.launch(executable_path=job.get("chromium") or None, args=CHROMIUM_ARGS)
    context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=ratio,
                                  locale="en-US", color_scheme="light")
    if cookies:
        context.add_cookies(cookies)
    page = context.new_page()
    page.goto(url, wait_until="load")
    try:
        page.wait_for_function("() => Boolean(window.__oraReplay)", timeout=READY_TIMEOUT_MS)
    except Exception as exc:  # the page says why (its loading line)
        said = page.evaluate("() => document.body ? document.body.innerText.slice(0, 400) : ''")
        browser.close()
        raise RuntimeError(f"the replay page did not open: {said or exc}") from exc
    return browser, page


def encoder(job: dict[str, Any], segment: Path) -> subprocess.Popen:
    return subprocess.Popen([
        job["ffmpeg"], "-hide_banner", "-loglevel", "error", "-y",
        "-f", "image2pipe", "-framerate", str(job["fps"]), "-c:v", "png", "-i", "-",
        "-vf", f"scale={job['width']}:{job['height']}:flags=lanczos,format=yuv420p",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(job.get("crf", 16)), "-profile:v", "high",
        "-r", str(job["fps"]), str(segment),
    ], stdin=subprocess.PIPE)


def draw(job: dict[str, Any]) -> dict[str, Any]:
    """One worker: frames [first, last) into job['segment']."""

    from playwright.sync_api import sync_playwright

    segment = Path(job["segment"])
    first, last = job["first"], job["last"]
    drawn = 0
    seeking = shooting = 0.0
    started = time.monotonic()
    with sync_playwright() as playwright:
        browser, page = open_page(playwright, job, job["page_url"], job["cookies"])
        camera = Camera(page, job)
        ffmpeg = encoder(job, segment)
        try:
            previous: tuple[str, bytes] | None = None
            for index in range(first, last):
                moment = job["t0"] + index * job["step"]
                begun = time.monotonic()
                digest = page.evaluate("(t) => window.__oraReplay.seekPlayback(t)", moment)
                seeking += time.monotonic() - begun
                if previous is None or previous[0] != digest:
                    begun = time.monotonic()
                    previous = (digest, camera.shoot())
                    shooting += time.monotonic() - begun
                    drawn += 1
                ffmpeg.stdin.write(previous[1])
                done = index - first + 1
                if done % max(1, (last - first) // 10) == 0:
                    print(f"[worker {job['worker']}] {done}/{last - first} frames, {drawn} drawn, "
                          f"{time.monotonic() - started:.0f} s", file=sys.stderr, flush=True)
        finally:
            ffmpeg.stdin.close()
            code = ffmpeg.wait()
            browser.close()
    if code:
        raise RuntimeError(f"ffmpeg failed on segment {segment.name} ({code})")
    return {"frames": last - first, "drawn": drawn, "seconds": round(time.monotonic() - started, 1),
            "seek_s": round(seeking, 1), "screenshot_s": round(shooting, 1)}


def probe(ffmpeg: str, path: Path) -> dict[str, Any]:
    """The video's size, frame rate and duration as ffprobe (beside ffmpeg) or ffmpeg reads them."""

    ffprobe = Path(ffmpeg).with_name("ffprobe")
    if ffprobe.is_file():
        output = subprocess.run([str(ffprobe), "-v", "error", "-select_streams", "v:0", "-count_frames",
                                 "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames:format=duration",
                                 "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
        data = json.loads(output)
        stream = data["streams"][0]
        num, den = (int(part) for part in stream["r_frame_rate"].split("/"))
        return {"width": stream["width"], "height": stream["height"], "fps": num / den,
                "frames": int(stream.get("nb_read_frames") or 0), "duration_s": float(data["format"]["duration"])}
    return {}


def page_moment(info: dict[str, Any], seconds: float | None, default: float) -> float:
    """The playback moment `seconds` into the recorded time (as the page computed it), or `default`."""

    return float(info.get("playback_of", {}).get(str(seconds), default)) if seconds else default


def main() -> int:
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(draw(json.load(sys.stdin))), flush=True)
        return 0
    job = json.load(sys.stdin)
    from playwright.sync_api import sync_playwright

    out = Path(job["out"])
    with sync_playwright() as playwright:
        browser, page = open_page(playwright, job, job["url"])
        info = page.evaluate("() => window.__oraReplay.info()")
        # Seconds into the recorded time, as playback moments of this replay's stretches.
        info["playback_of"] = {str(value): page.evaluate("(t) => window.__oraReplay.playbackOf(t)", info["start"] + value * 1000)
                               for value in (job.get("from_s"), job.get("to_s")) if value}
        cookies = page.context.cookies()
        page_url = page.url  # the launch token is spent: the page's own address and its cookie from here
        browser.close()
    to_s = job.get("to_s")
    first_ms = page_moment(info, job.get("from_s") or 0.0, 0.0)
    last_ms = page_moment(info, to_s, info["duration"]) if to_s is not None else info["duration"]
    t0, step, count = frames_between(first_ms, last_ms, job["fps"])
    work = Path(tempfile.mkdtemp(prefix="ora-replay-export-"))
    try:
        jobs = []
        for worker, (first, last) in enumerate(stretches(count, int(job.get("workers") or 1))):
            jobs.append({**job, "worker": worker, "first": first, "last": last, "t0": t0, "step": step,
                         "page_url": page_url, "cookies": cookies, "segment": str(work / f"segment-{worker:03d}.mp4")})
        print(f"[export] {count} frames ({count / job['fps']:.1f} s at {job['fps']} fps) in {len(jobs)} workers",
              file=sys.stderr, flush=True)
        processes = [subprocess.Popen([sys.executable, __file__, "--worker"], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True) for _ in jobs]
        for process, item in zip(processes, jobs):
            process.stdin.write(json.dumps(item))
            process.stdin.close()
        results = []
        for process in processes:
            output = process.stdout.read()
            if process.wait():
                raise RuntimeError("a worker failed; see its lines above")
            results.append(json.loads(output.strip().splitlines()[-1]))
        listing = work / "segments.txt"
        listing.write_text("".join(f"file '{item['segment']}'\n" for item in jobs), encoding="utf-8")
        subprocess.run([job["ffmpeg"], "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    measured = probe(job["ffmpeg"], out)
    stretches_ = stretch_report(info["segments"], t0, t0 + count * step, info["speed"], info["motion"])
    report = {
        "schema": "ora.replay-video.v1", "run": job.get("run"), "speed": info["speed"], "motion_speed": info["motion"],
        "fps": job["fps"], "width": job["width"], "height": job["height"], "scale": job["scale"],
        "camera_aspect": round(float(info.get("aspect") or 4 / 3), 4),
        "zoom": job.get("zoom") or 1.0, "crf": job.get("crf", 16), "frames": count,
        "drawn": sum(item["drawn"] for item in results), "duration_s": round(count / job["fps"], 3),
        "recorded_from_ms": stretches_[0]["recorded_from_ms"] if stretches_ else None,
        "recorded_to_ms": stretches_[-1]["recorded_to_ms"] if stretches_ else None,
        "motion_s": round(sum(item["video_to_s"] - item["video_from_s"] for item in stretches_ if item["kind"] == "motion"), 3),
        "segments": stretches_, "notes": info.get("notes", []), "bytes": out.stat().st_size,
        "measured": measured, "workers": results,
    }
    out.with_name(out.name + ".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ora replay export: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
