"""All currently exposed SAM3 prompt routes. Replay requests services afresh.

Box/negative-point/joint-multipoint/mask-input prompts have no current service
route; no fake implementation is provided. Independent points are not a joint prompt.
"""
from dataclasses import dataclass

from ..interface.api import DataInterface
from .blobs import read_array
from .values import Array, JsonValue, PublicResult


@dataclass(frozen=True)
class MaskCandidate:
    """Full client mask HxW in pixels, score/rank/query identity unmodified in fields."""
    mask: Array
    fields: JsonValue


@dataclass(frozen=True)
class Segmentation:
    """All returned alternatives, in source order; no tool ranking, dedup or truncation."""
    candidates: tuple[MaskCandidate, ...]
    metadata: JsonValue
    result: PublicResult

    def __post_init__(self):
        object.__setattr__(self,'candidates',tuple(self.candidates))


def _decode(api, response):
    result=PublicResult.of(response)
    data=result.data.json()
    rows=data.get('candidates',[{'mask':data['mask']}] if 'mask' in data else [])
    candidates=tuple(MaskCandidate(read_array(api,JsonValue.of(row['mask'])),
                                   JsonValue.of({k:v for k,v in row.items() if k!='mask'})) for row in rows)
    return Segmentation(candidates,JsonValue.of(data['metadata']),result)


def segment(api: DataInterface, *, rgb: JsonValue, depth_m: JsonValue, point_uv: tuple[int,int]) -> Segmentation:
    """Replay: re-request SAM3 at positive (u,v) pixels; historical result only comparison.

    Example: segment(api, rgb=refs.rgb, depth_m=refs.depth_m, point_uv=(12,34)).
    Returns the existing client's prompt-connected mask, not unnormalized HTTP bytes.
    Invalid inputs/service failure propagate; no simulator fallback.
    """
    return _decode(api,api.perception.segment(rgb=rgb.json(),depth=depth_m.json(),point_uv=point_uv))


def segment_points(api: DataInterface, *, rgb: JsonValue, depth_m: JsonValue,
                   points_uv: tuple[tuple[int,int], ...]) -> Segmentation:
    """Replay: re-request 1–8 independent positive pixel prompts, all multimask alternatives.

    Example: points_uv=((12,34),(50,60)). Preserve per-query failures in metadata;
    do not silently merge prompts, select one mask or hide partial errors.
    """
    return _decode(api,api.perception.segment_points(rgb=rgb.json(),depth=depth_m.json(),points_uv=points_uv))


def ground(api: DataInterface, *, rgb: JsonValue, text: str, max_candidates: int = 3) -> Segmentation:
    """Replay: re-request text grounding; new text works, unmatched history gives unavailable comparison.

    Example: ground(api, rgb=refs.rgb, text='red cup'). All 1–3 requested candidates,
    pixel masks/boxes, scores and source order survive; failures propagate.
    """
    return _decode(api,api.perception.ground(rgb=rgb.json(),text=text,max_candidates=max_candidates))
