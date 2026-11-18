"""Public pixels give byte-identical legacy measurements and inspection images."""
from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ora.agent.toolkit.lib import vision
from ora.agent.toolkit.lib.values import Array, JsonValue
from ora.agent.toolkit.lib.perception import MaskCandidate
from ora.agent.toolkit.host.perception.mask_localization import _mask_depth_localization, _mask_summary
from ora.agent.toolkit.host.perception.grounding_preview import render_candidate_preview, profile_input

ROOT = Path(__file__).resolve().parents[5]
FIXTURE = ROOT / 'tests/data/toolkit-general-pickup'
# RGB uint8 pixels the legacy grounding overview,
# ora.agent.tools.perception.grounding._render_grounding_overlay, drew for inputs().
GROUNDING_OVERVIEW = dict(dtype='|u1', shape=[480, 640, 3],
                          sha256='6ee6151b8d758ff1941538b3d5f65236c17076527a95d518ccd8465b54acbedf')


def inputs():
    snapshot = json.loads((FIXTURE/'snapshot.json').read_bytes())
    path = FIXTURE/Path(snapshot['available_at_boundary']['observation']['manifest']['path']).parent/'head'
    rgb = np.asarray(Image.open(path/'rgb.png').convert('RGB'))
    depth = np.load(path/'depth-m.npy', allow_pickle=False)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[160:240, 240:340] = True
    fields = dict(rank=0, source_index=0, backend_score=.9, bbox_xyxy=[240,160,340,240], target_ref='target-test')
    return path, rgb, depth, mask, fields


def pixels(array):
    return dict(dtype=array.dtype.str, shape=list(array.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest())


def png(image):
    stream = BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


@pytest.mark.parametrize('extended', [False, True])
def test_localization_and_summary_match_original(extended):
    path, _, depth, mask, _ = inputs()
    calibration = JsonValue((path/'calibration.json').read_bytes())
    actual = vision.localize_mask(Array.of(mask), Array.of(depth), calibration, extended_frame=extended)
    expected = _mask_depth_localization(mask=mask, depth_m=depth, calibration_path=path/'calibration.json', extended_frame=extended)
    assert actual.json() == expected
    assert vision.mask_summary(Array.of(mask), Array.of(depth)).json() == _mask_summary(mask, depth)


@pytest.mark.parametrize('profile', [False, True])
def test_rendered_png_and_metadata_match_original(profile):
    path, rgb, depth, mask, fields = inputs()
    candidate = MaskCandidate(Array.of(mask), JsonValue.of(fields))
    calibration = JsonValue((path/'calibration.json').read_bytes())
    localized = vision.localize_mask(candidate.mask, Array.of(depth), calibration)
    actual = vision.candidate_inspection(Array.of(rgb), candidate, text='mint green scissors',
        **dict(depth_m=Array.of(depth), calibration=calibration, localization=localized, support_line=True) if profile else {})
    settings = dict(renderer='grounding-candidate-depth-profile.v1', support_line=True) if profile else None
    expected, metadata = render_candidate_preview(rgb_u8=rgb, candidate=dict(fields, mask=mask), text='mint green scissors',
        depth_profile=profile_input(settings, depth_m=depth, calibration_path=path/'calibration.json', localization=localized.json()))
    assert vision.png_bytes(actual.rgb) == png(expected)
    assert actual.metadata.json() == metadata
    overlay = vision.mask_overview(Array.of(rgb), (candidate,))
    assert pixels(overlay.numpy()) == GROUNDING_OVERVIEW


def test_npy_and_partial_profile_refusal():
    _, rgb, depth, mask, fields = inputs()
    array = Array.of(mask.astype(np.uint8))
    stream = BytesIO(); np.save(stream, array.numpy(), allow_pickle=False)
    assert vision.npy_bytes(array) == stream.getvalue()
    with pytest.raises(ValueError, match='requires'):
        vision.candidate_inspection(Array.of(rgb), MaskCandidate(Array.of(mask), JsonValue.of(fields)),
                                    text='cup', depth_m=Array.of(depth))
