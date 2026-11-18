"""Test helpers for batch segmentation, shared by several test modules.

They were defined in ``tests/ora/agent/toolkit/host/perception/test_batch_segmentation.py``;
test modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

import numpy as np


def rectangle(x0, y0, x1, y1, size=96):
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def raw_candidate(q, c, point, mask, score):
    return {"query_index": q, "source_index": c, "point_uv": point,
            "mask": mask, "backend_score": score}


class BatchBackend:
    backend_id, backend_version = "batch-fixture", "1"

    def __init__(self, fail=(), empty=(), excluded=()):
        self.calls = 0
        self.fail, self.empty, self.excluded = set(fail), set(empty), set(excluded)

    def configuration(self):
        return {"fixture": "independent-point-proposals"}

    def segment_points(self, *, rgb_u8, depth_m, points_uv):
        self.calls += 1
        raw, queries = [], []
        for q, point in enumerate(points_uv):
            failed, produced = q in self.fail, 0
            if not failed and q not in self.empty:
                x, y = point
                mask = (rectangle(2, 2, 6, 6) if q in self.excluded
                        else rectangle(x - 5, y - 5, x + 6, y + 6))
                raw.extend([raw_candidate(q, i, point, mask, .8 - i * .1) for i in range(2)])
                produced = 2
            queries.append({"query_index": q, "point_uv": point,
                "raw_candidate_count": produced,
                "error_code": "point_inference_failed" if failed else None})
        return raw, {"query_results": queries, "image_encodings": 1}
