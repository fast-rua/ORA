"""A depth completion service: RGB + raw metric depth in, completed metric depth out.

    ora cell depth-service --model lingbot [--port 8770] [--host 0.0.0.0]
    python -m ora.cell.depth.service --model lingbot ...      # under the model's own interpreter

It runs in whatever interpreter holds the model (it needs only NumPy and, for ``inpaint``,
OpenCV), on the cell's computer or any GPU machine the cell can reach. The cell board
calls it (``client.py``); nothing in an ORA episode does.

    GET  /v1/info       {"model", "checkpoint", "device", "ready"}
    POST /v1/complete   body: an .npz with rgb (H,W,3 uint8), depth (H,W float32 metres,
                        0 where the sensor has none) and intrinsic (3,3, pixels);
                        reply: an .npz with depth (H,W float32 metres) and ms (inference time)

Models (``MODELS``), lightest first, for the GPU the cell happens to have:

- ``raw`` returns the input (to test the plumbing); ``inpaint`` fills holes with OpenCV on the CPU.
- ``promptda``: Prompt Depth Anything (arXiv 2412.14015, via transformers), the sensor depth as
  a metric prompt. ViT-S by default, about 25M parameters, fp16 on a GPU; ``--checkpoint``
  takes the b and l variants.
- ``lingbot``: LingBot-Depth (Robbyant, arXiv 2601.17895; ViT-L, about 300M parameters;
  installed from github.com/robbyant/lingbot-depth).
- ``auto``: the largest of these the free GPU memory allows when the service starts.

``--max-side`` runs any of them on shrunk frames. Check each checkpoint's licence on its
model card. One request runs at a time: a GPU holds one copy of the model.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

MAX_BODY_BYTES = 64 * 1024 * 1024


class Raw:
    name = "raw"

    def __init__(self, checkpoint: str | None, device: str) -> None:
        self.checkpoint, self.device = checkpoint, "cpu"

    def complete(self, rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        return depth


class Inpaint:
    """OpenCV Telea inpainting of the holes, at a quarter of the resolution: a CPU baseline."""

    name = "inpaint"

    def __init__(self, checkpoint: str | None, device: str) -> None:
        self.checkpoint, self.device = checkpoint, "cpu"

    def complete(self, rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        import cv2

        height, width = depth.shape
        small = cv2.resize(depth, (width // 4, height // 4), interpolation=cv2.INTER_NEAREST)
        holes = (small <= 0).astype(np.uint8)
        scale = float(small.max()) or 1.0
        filled = cv2.inpaint((small / scale * 65535).astype(np.uint16), holes, 3, cv2.INPAINT_TELEA)
        out = cv2.resize(filled.astype(np.float32) / 65535 * scale, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.where(depth > 0, depth, out).astype(np.float32)


class LingBot:
    """LingBot-Depth's masked depth model: sensor dropouts are masks filled from the image."""

    name = "lingbot"
    DEFAULT_CHECKPOINT = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"

    def __init__(self, checkpoint: str | None, device: str) -> None:
        import torch
        from mdm.model.v2 import MDMModel

        self.torch = torch
        self.checkpoint = checkpoint or self.DEFAULT_CHECKPOINT
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MDMModel.from_pretrained(self.checkpoint).to(self.device).eval()

    def complete(self, rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        torch = self.torch
        height, width = depth.shape
        normalised = np.asarray(intrinsic, np.float32).copy()
        normalised[0] /= width          # the model takes fx, cx over width and fy, cy over height
        normalised[1] /= height
        with torch.inference_mode():
            image = torch.tensor(rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)[None]
            raw = torch.tensor(depth, dtype=torch.float32, device=self.device)[None]
            k = torch.tensor(normalised, dtype=torch.float32, device=self.device)[None]
            out = self.model.infer(image, depth_in=raw, use_fp16=self.device == "cuda", intrinsics=k)
            return out["depth"][0].float().cpu().numpy().astype(np.float32)


class PromptDA:
    """Prompt Depth Anything: Depth Anything with the sensor's depth, downsampled and its holes
    filled, as a metric prompt. ViT-S (about 25M parameters) runs on small GPUs; b and l
    checkpoints trade memory for accuracy."""

    name = "promptda"
    DEFAULT_CHECKPOINT = "depth-anything/prompt-depth-anything-vits-hf"
    PROMPT_SIZE = (256, 144)            # width, height of the depth prompt

    def __init__(self, checkpoint: str | None, device: str) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.checkpoint = checkpoint or self.DEFAULT_CHECKPOINT
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint, dtype=dtype).to(self.device).eval()
        self.dtype = dtype

    def complete(self, rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        import cv2

        torch = self.torch
        height, width = depth.shape
        prompt = Inpaint(None, "cpu").complete(rgb, depth, intrinsic)
        prompt = cv2.resize(prompt, self.PROMPT_SIZE, interpolation=cv2.INTER_AREA)
        inputs = self.processor(images=rgb, prompt_depth=prompt, prompt_scale_to_meter=1.0, return_tensors="pt")
        inputs = {key: value.to(self.device, self.dtype if value.is_floating_point() else value.dtype)
                  for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        predicted = outputs.predicted_depth.float()[:, None]
        resized = torch.nn.functional.interpolate(predicted, size=(height, width), mode="bilinear", align_corners=False)
        return resized[0, 0].cpu().numpy().astype(np.float32)


class Auto:
    """The largest model the free GPU memory allows: lingbot, then promptda ViT-S, then inpaint."""

    LINGBOT_FREE_GB = 6.0
    PROMPTDA_FREE_GB = 0.4      # ViT-S fp16 measured at 218 MB on an RTX 3070 Ti

    def __new__(cls, checkpoint: str | None, device: str):
        try:
            import torch

            free_gb = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0.0
        except ImportError:
            free_gb = 0.0
        for model, need in ((LingBot, cls.LINGBOT_FREE_GB), (PromptDA, cls.PROMPTDA_FREE_GB)):
            if free_gb >= need:
                try:
                    return model(None, device)
                except ImportError:
                    continue
        return Inpaint(None, "cpu")


class Downscaled:
    """Runs ``inner`` on the frame shrunk to ``max_side`` pixels and scales the depth back up."""

    def __init__(self, inner: Any, max_side: int) -> None:
        self.inner, self.max_side = inner, max_side
        self.name, self.checkpoint, self.device = inner.name, inner.checkpoint, inner.device

    def complete(self, rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        import cv2

        height, width = depth.shape
        scale = self.max_side / max(height, width)
        if scale >= 1.0:
            return self.inner.complete(rgb, depth, intrinsic)
        size = (int(round(width * scale)), int(round(height * scale)))
        k = np.asarray(intrinsic, float).copy()
        k[:2] *= scale
        small = self.inner.complete(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA),
                                    cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST), k)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


MODELS = {model.name: model for model in (Raw, Inpaint, PromptDA, LingBot)}
MODELS["auto"] = Auto


def encode(**arrays: Any) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def decode(body: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(body), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def make_handler(model: Any, gate: threading.Lock):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code: int, message: str) -> None:
            self._reply(code, json.dumps({"error": message}).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/info":
                info = {"model": model.name, "checkpoint": model.checkpoint, "device": model.device, "ready": True}
                self._reply(200, json.dumps(info).encode(), "application/json")
            else:
                self._error(404, "not found")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/complete":
                return self._error(404, "not found")
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY_BYTES:
                return self._error(413, f"body must be 1..{MAX_BODY_BYTES} bytes")
            try:
                request = decode(self.rfile.read(length))
                rgb, depth, intrinsic = request["rgb"], request["depth"].astype(np.float32), request["intrinsic"]
                if rgb.shape[:2] != depth.shape or rgb.ndim != 3 or intrinsic.shape != (3, 3):
                    return self._error(400, "rgb (H,W,3), depth (H,W) and intrinsic (3,3) must agree")
                depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
                with gate:
                    started = time.perf_counter()
                    completed = model.complete(rgb.astype(np.uint8), depth, intrinsic.astype(np.float64))
                    elapsed = (time.perf_counter() - started) * 1000
                if completed.shape != depth.shape:
                    return self._error(500, f"model returned {completed.shape}, expected {depth.shape}")
                self._reply(200, encode(depth=completed.astype(np.float32), ms=np.float32(elapsed)),
                            "application/octet-stream")
            except (KeyError, ValueError) as exc:
                self._error(400, f"bad request: {exc}")
            except Exception as exc:  # noqa: BLE001 - the caller falls back to raw depth and shows why
                self._error(500, f"{type(exc).__name__}: {exc}")

        def log_message(self, *_: Any) -> None:
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="inpaint")
    parser.add_argument("--checkpoint", help="model checkpoint (lingbot: a Hugging Face id or local path)")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:1, cpu")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve another machine")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--max-side", type=int, default=0, help="run the model on frames shrunk to this many pixels")
    args = parser.parse_args(argv)
    model = MODELS[args.model](args.checkpoint, args.device)
    if args.max_side:
        model = Downscaled(model, args.max_side)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, threading.Lock()))
    print(f"depth service: {model.name} ({model.checkpoint or '-'}) on {model.device} at "
          f"http://{args.host}:{args.port}/v1/complete", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
