"""Unnormalized provider-request differences, including every image and its label.

Pixel metrics supplement byte equality; they never turn a different request
into an equal one. 48--52 dB is the measured M7.4b rendering-noise reference,
not a proof that any particular difference came from the renderer.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from typing import Any

import numpy as np
from PIL import Image


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def summary(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 240:
        return {"bytes": len(value.encode()), "sha256": digest(value.encode()), "prefix": value[:120]}
    return value


def differences(a: Any, b: Any, path: str = "$") -> list[dict]:
    """Keep every changed field; compact large strings without erasing their hashes."""
    if type(a) is not type(b):
        return [{"path": path, "reason": "type", "before": type(a).__name__, "after": type(b).__name__}]
    if isinstance(a, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            child = f"{path}/{key}"
            if key not in a or key not in b:
                result.append({"path": child, "reason": "added" if key in b else "removed"})
            else:
                result.extend(differences(a[key], b[key], child))
        return result
    if isinstance(a, list):
        result = ([] if len(a) == len(b) else [{"path": path, "reason": "length", "before": len(a), "after": len(b)}])
        for index, (x, y) in enumerate(zip(a, b)):
            result.extend(differences(x, y, f"{path}/{index}"))
        return result
    if a == b:
        return []
    return [{"path": path, "reason": "value", "before": summary(a), "after": summary(b)}]


def images(request: dict) -> list[dict]:
    result = []
    for i, message in enumerate(request.get("messages", [])):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        label = None
        for j, block in enumerate(content):
            if block.get("type") == "text":
                label = block["text"]
            elif block.get("type") == "image_url":
                result.append({"path": f"$/messages/{i}/content/{j}", "label": label,
                               "reference": block["image_url"], "type": block["type"]})
    return result


def image_difference(a: dict, b: dict) -> dict:
    value = {"before_path": a["path"], "after_path": b["path"],
             "position_equal": a["path"] == b["path"], "label_equal": a["label"] == b["label"],
             "before_label": summary(a["label"]), "after_label": summary(b["label"]),
             "reference_equal": a["reference"] == b["reference"]}
    urls = [item["reference"]["url"] for item in (a, b)]
    try:
        payload = [base64.b64decode(url.split(",", 1)[1], validate=True) for url in urls
                   if url.startswith("data:image/") and ";base64," in url]
        if len(payload) != 2:
            raise ValueError("non-inline image reference")
        value["byte_equal"] = payload[0] == payload[1]
        value["sha256"] = [digest(raw) for raw in payload]
        arrays = [np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float64) for raw in payload]
        if arrays[0].shape != arrays[1].shape:
            value.update(classification="dimension_difference", shapes=[list(x.shape) for x in arrays])
            return value
        error = np.abs(arrays[0] - arrays[1])
        mse = float(np.mean(error ** 2))
        psnr = 10 * math.log10(255 ** 2 / mse) if mse else None
        value.update(psnr_db=psnr, pixels_equal=mse == 0,
                     pixels_above_16_fraction=float(np.mean(np.max(error, axis=2) > 16)),
                     classification=("pixel_equal" if mse == 0 else "rendering_noise_reference_48_52_db"
                                     if 48 <= psnr <= 52 else "below_rendering_noise_reference" if psnr < 48
                                     else "above_rendering_noise_reference"))
    except (ValueError, OSError, KeyError) as error:
        value["metric_error"] = str(error)
    return value


def compare_requests(before: bytes, after: bytes) -> dict:
    a, b = json.loads(before), json.loads(after)
    ai, bi = images(a), images(b)
    return {"byte_equal": before == after, "sha256": [digest(before), digest(after)],
            "json_equal": a == b, "differences": differences(a, b),
            "image_counts": [len(ai), len(bi)], "image_count_equal": len(ai) == len(bi),
            "images": [image_difference(x, y) for x, y in zip(ai, bi)],
            "unmatched_images": [{"side": side, "path": item["path"], "label": summary(item["label"])}
                                 for side, items in (("before", ai[len(bi):]), ("after", bi[len(ai):])) for item in items]}
