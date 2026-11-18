"""Known-pixel diagnostics, finite JSON output and calibration field accounting."""

from io import BytesIO
import numpy as np
import pytest
from PIL import Image
from ora.agent.toolkit.interface.types import canonical
from ora.orchestrator.toolkit.sensor_metrics import rgb, depth, calibration


def png(value):
    output = BytesIO()
    Image.fromarray(value).save(output, format="PNG")
    return output.getvalue()


def npy(value):
    output = BytesIO()
    np.save(output, value)
    return output.getvalue()


def test_rgb_pixel_fraction_uses_any_channel_and_psnr_all_channels():
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = a.copy()
    b[0, 0, 0] = 32
    report = rgb(png(a), png(b))
    assert report["fraction_pixels_gt_16"] == 0.25
    assert report["mse"] == pytest.approx(32**2 / 12)
    assert report["psnr_db"] == pytest.approx(10 * np.log10(255**2 / (32**2 / 12)))
    same = rgb(png(a), png(a))
    assert same["psnr_db"] is None and same["psnr_infinite"] is True
    canonical(same)


def test_depth_quantiles_are_absolute_metres_and_account_for_nonfinite():
    report = depth(
        npy(np.array([0.0, 1.0, np.nan, 4.0])),
        npy(np.array([0.001, 1.003, 2.0, np.inf])),
    )
    assert report["finite_pair_count"] == 2
    assert report["finite_mask_difference_count"] == 2
    assert report["absolute_difference_m"]["p50"] == pytest.approx(0.002)
    assert report["absolute_difference_m"]["max"] == pytest.approx(0.003)
    canonical(report)


def test_calibration_retains_render_counters_and_per_field_maximum():
    a = dict(
        extrinsic_cv=[[1.0, 0.0]],
        renderer_calibration=dict(
            reference_time=dict(referenceTimeNumerator=153, referenceTimeDenominator=30)
        ),
    )
    b = dict(
        extrinsic_cv=[[1.0, 1e-8]],
        renderer_calibration=dict(
            reference_time=dict(referenceTimeNumerator=119, referenceTimeDenominator=30)
        ),
    )
    report = calibration(canonical(a), canonical(b))
    assert report["max_absolute_difference_by_field"]["/extrinsic_cv"] == 1e-8
    assert (
        report["max_absolute_difference_by_field"][
            "/renderer_calibration/reference_time/referenceTimeNumerator"
        ]
        == 34
    )
    assert (
        report["rendering"]["original"]["reference_time"]["referenceTimeNumerator"]
        == 153
    )
    canonical(report)
