"""Tool policy: per-point reserves, rank-major allocation and cross-point IoU dedup."""

import math
from typing import Any

from ...interface.stages import active, stage

SCHEMA = "ora.point-segmentation-batch.v2"
RENDERER = "point-batch-per-point-inspection.v2"
SELECTION_METHOD = "per_point_reserved_then_rank_major_v1"
MAX_CANDIDATES = 3
# One returned mask per queried point is reserved before any surplus is shared,
# so this ceiling must never fall below the point bound (clients.MAX_POINTS). The remaining slots bound
# the delivered result against the eager large-result threshold.
MAX_RETURNED_CANDIDATES = 12
# Enlarged crops are the expensive half of the host cost and the only part that
# adds inline images, so they are capped separately from the mask ledger.
MAX_PREVIEW_IMAGES = 5
MASK_IOU_THRESHOLD = .9
POINT_COLOURS = ((48, 220, 96), (64, 180, 255), (255, 190, 48), (255, 96, 160),
                 (170, 120, 255), (64, 232, 232), (255, 132, 64), (200, 200, 80))
UNANSWERED_COLOUR = (168, 168, 168)


def _normalize(raw: list[dict[str, Any]], *, points_uv: list[list[int]],
               shape: tuple[int, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the prompt-connected component of every mask that contains its point."""
    from scipy.ndimage import label
    import numpy as np

    normalized, rejected, seen = [], [], set()
    per_query = [0] * len(points_uv)
    for item in raw:
        q, c, score = item.get("query_index"), item.get("source_index"), item.get("backend_score")
        if (type(q) is not int or not 0 <= q < len(points_uv) or type(c) is not int or c < 0
                or (q, c) in seen or item.get("point_uv") != points_uv[q]
                or type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError("batch candidate lost its query identity or finite score")
        seen.add((q, c))
        per_query[q] += 1
        if per_query[q] > MAX_CANDIDATES:
            raise ValueError("batch backend exceeded its per-query mask bound")
        mask = np.asarray(item["mask"])
        if mask.shape != shape or not np.isfinite(mask).all():
            raise ValueError("batch candidate mask is not aligned with the source image")
        mask = mask.astype(bool)
        u, v = points_uv[q]
        if not mask[v, u]:
            rejected.append({"query_index": q, "source_index": c, "reason": "mask_excludes_positive_point"})
            continue
        labels, components = label(mask, structure=np.ones((3, 3), dtype=bool))
        count = components + 1
        component = labels == int(labels[v, u])
        raw_count, kept_count = int(mask.sum()), int(component.sum())
        normalized.append({**item, "mask": component, "backend_score": float(score),
            "normalization": {"method": "positive_prompt_8_connected_component_v1",
                "raw_component_count": count - 1, "raw_pixel_count": raw_count,
                "prompt_component_pixel_count": kept_count, "retained_fraction": kept_count / raw_count},
            "sources": [{"query_index": q, "source_index": c}]})
    return normalized, rejected


def _deduplicate(normalized: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge near-identical masks so one object costs one artifact, not one per point."""
    import numpy as np

    normalized.sort(key=lambda item: (-item["backend_score"], item["query_index"], item["source_index"]))
    distinct: list[dict[str, Any]] = []
    for item in normalized:
        for representative in distinct:
            intersection = int(np.count_nonzero(item["mask"] & representative["mask"]))
            union = int(np.count_nonzero(item["mask"] | representative["mask"]))
            if intersection / union >= MASK_IOU_THRESHOLD:
                representative["sources"].extend(item["sources"])
                break
        else:
            distinct.append(item)
    return distinct


def select_masks(raw: list[dict[str, Any]], *, points_uv: list[list[int]],
                 shape: tuple[int, int], max_candidates: int,
                 max_returned: int = MAX_RETURNED_CANDIDATES,
                 max_previews: int = MAX_PREVIEW_IMAGES,
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Give every point its own bounded mask list under one shared delivery cap.

    Rank 0 of every point is admitted before any point's rank 1, so a point that
    produced a usable mask always keeps one. Only the surplus competes, and each
    point reports exactly how many of its own masks it lost and to which limit.
    """
    if type(max_candidates) is not int or not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError("max_candidates must be an integer in 1..3")
    if (type(max_returned) is not int
            or not len(points_uv) <= max_returned <= MAX_RETURNED_CANDIDATES):
        raise ValueError("max_returned must reserve one mask for every queried point")
    if type(max_previews) is not int or not 1 <= max_previews <= MAX_PREVIEW_IMAGES:
        raise ValueError("max_previews must be an integer in 1..5")
    if not isinstance(raw, list) or len(raw) > MAX_CANDIDATES * len(points_uv):
        raise ValueError("batch backend exceeded its per-query mask bound")

    normalized, rejected = _normalize(raw, points_uv=points_uv, shape=shape)
    distinct = _deduplicate(normalized)
    owned = [[item for item in distinct
              if any(source["query_index"] == q for source in item["sources"])]
             for q in range(len(points_uv))]
    beyond = [max(0, len(masks) - max_candidates) for masks in owned]
    if active():
        for q, masks in enumerate(owned):
            for item in masks[max_candidates:]:
                stage("mask_omitted", query_index=q, reason="point_limit", backend_score=item["backend_score"])
    owned = [masks[:max_candidates] for masks in owned]

    admitted: list[dict[str, Any]] = []
    keys: set[int] = set()
    for rank in range(max_candidates):
        tier = sorted(((masks[rank], q) for q, masks in enumerate(owned) if len(masks) > rank),
                      key=lambda pair: (-pair[0]["backend_score"], pair[1]))
        for item, _query in tier:
            if id(item) in keys or len(admitted) >= max_returned:
                continue
            keys.add(id(item))
            admitted.append(item)

    kept = [[item for item in masks if id(item) in keys] for masks in owned]
    if active():
        for q, masks in enumerate(owned):
            for item in masks:
                if id(item) not in keys:
                    stage("mask_omitted", query_index=q, reason="call_cap", backend_score=item["backend_score"])
    for masks, survivors in zip(owned, kept):
        if masks and not survivors:
            raise ValueError("per-point mask reserve failed under the delivery cap")
    previews: list[dict[str, Any]] = []
    preview_keys: set[int] = set()
    for survivors in kept:
        if not survivors or id(survivors[0]) in preview_keys or len(previews) >= max_previews:
            continue
        preview_keys.add(id(survivors[0]))
        previews.append(survivors[0])
    ordered = previews + [item for item in admitted if id(item) not in preview_keys]
    position = {id(item): index for index, item in enumerate(ordered)}
    preview_index = {id(item): index + 1 for index, item in enumerate(previews)}

    queries = []
    for query, (masks, survivors) in enumerate(zip(owned, kept)):
        queries.append({"query_index": query, "point_uv": points_uv[query],
            "candidate_ids": [f"T{position[id(item)]}" for item in survivors],
            "omitted_over_point_limit": beyond[query],
            "omitted_over_call_cap": len(masks) - len(survivors),
            "preview_image_index": preview_index.get(id(survivors[0])) if survivors else None,
            "previews_omitted": int(bool(survivors) and id(survivors[0]) not in preview_index)})
    return ordered, {
        "method": SELECTION_METHOD, "mask_iou_threshold": MASK_IOU_THRESHOLD,
        "max_candidates_per_point": max_candidates, "max_returned_candidates": max_returned,
        "raw_candidate_count": len(raw), "usable_candidate_count": len(normalized),
        "distinct_candidate_count": len(distinct), "duplicate_count": len(normalized) - len(distinct),
        "returned_count": len(ordered),
        "omitted_over_point_limit_total": sum(beyond),
        "omitted_over_call_cap_total": sum(item["omitted_over_call_cap"] for item in queries),
        "rejected_candidates": rejected, "semantic_selection": "agent",
        "queries": queries, "preview_count": len(previews),
    }

