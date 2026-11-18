"""A deterministic point-prompt RGB-D segmenter: a qualification baseline and test
double behind the same artifact boundary as the SAM3 service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DepthPromptSegmenter:
    """Conservative point-prompt RGB-D segmentation baseline.

    Depth selects a bounded probable foreground region; GrabCut uses RGB only
    to refine its boundary.  The connected component containing the exact
    prompt is retained.  This is a qualification baseline, not a claim of
    open-vocabulary semantic segmentation.
    """

    backend_id: str = "depth_prompt_grabcut"
    backend_version: str = "v1"
    depth_band_min_m: float = 0.035
    depth_band_fraction: float = 0.08
    depth_band_max_m: float = 0.15
    grabcut_iterations: int = 3

    def configuration(self) -> Mapping[str, Any]:
        return {
            "depth_band_min_m": self.depth_band_min_m,
            "depth_band_fraction": self.depth_band_fraction,
            "depth_band_max_m": self.depth_band_max_m,
            "grabcut_iterations": self.grabcut_iterations,
        }

    def segment(
        self, *, rgb_u8: Any, depth_m: Any, point_uv: tuple[int, int],
    ) -> tuple[Any, Mapping[str, Any]]:
        import cv2
        import numpy as np

        rgb = np.asarray(rgb_u8, dtype=np.uint8)
        depth = np.asarray(depth_m, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError("segmentation requires aligned HxWx3 RGB and HxW depth")
        u, v = point_uv
        height, width = depth.shape
        if not (0 <= u < width and 0 <= v < height):
            raise ValueError("segmentation point is outside the source image")
        radius = 2
        seed_patch = depth[
            max(0, v - radius):min(height, v + radius + 1),
            max(0, u - radius):min(width, u + radius + 1),
        ]
        valid_seed = seed_patch[np.isfinite(seed_patch) & (seed_patch > 1e-6)]
        if not valid_seed.size:
            raise ValueError("segmentation point has no valid metric depth")
        seed_depth = float(np.median(valid_seed))
        depth_band = max(
            self.depth_band_min_m,
            min(self.depth_band_max_m, self.depth_band_fraction * seed_depth),
        )
        valid = np.isfinite(depth) & (depth > 1e-6)
        probable = valid & (np.abs(depth - seed_depth) <= depth_band)
        if int(np.count_nonzero(probable)) < 4:
            raise ValueError("segmentation prompt has insufficient nearby geometry")

        if min(height, width) >= 16:
            state = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
            state[probable] = cv2.GC_PR_FGD
            seed_radius = max(1, min(4, min(height, width) // 48))
            cv2.circle(state, (u, v), seed_radius, cv2.GC_FGD, thickness=-1)
            state[0, :] = cv2.GC_BGD
            state[-1, :] = cv2.GC_BGD
            state[:, 0] = cv2.GC_BGD
            state[:, -1] = cv2.GC_BGD
            background = np.zeros((1, 65), dtype=np.float64)
            foreground = np.zeros((1, 65), dtype=np.float64)
            try:
                cv2.grabCut(
                    rgb, state, None, background, foreground,
                    self.grabcut_iterations, cv2.GC_INIT_WITH_MASK,
                )
                foreground_mask = (
                    (state == cv2.GC_FGD) | (state == cv2.GC_PR_FGD)
                ) & probable
            except cv2.error:
                foreground_mask = probable
        else:
            foreground_mask = probable

        components, labels = cv2.connectedComponents(
            foreground_mask.astype(np.uint8), connectivity=8,
        )
        seed_label = int(labels[v, u])
        if components <= 1 or seed_label == 0:
            components, labels = cv2.connectedComponents(
                probable.astype(np.uint8), connectivity=8,
            )
            seed_label = int(labels[v, u])
        if components <= 1 or seed_label == 0:
            raise ValueError("segmentation could not isolate the prompted component")
        mask = labels == seed_label
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1,
        ).astype(bool)
        components, labels = cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=8,
        )
        seed_label = int(labels[v, u])
        mask = labels == seed_label if components > 1 and seed_label else mask
        pixel_count = int(np.count_nonzero(mask))
        area_fraction = pixel_count / float(height * width)
        if pixel_count < 4:
            raise ValueError("segmentation result is too small")
        if area_fraction > 0.80:
            raise ValueError("segmentation result is implausibly broad")
        return mask, {
            "seed_depth_m": seed_depth,
            "depth_band_m": depth_band,
            "probable_pixel_count": int(np.count_nonzero(probable)),
        }
