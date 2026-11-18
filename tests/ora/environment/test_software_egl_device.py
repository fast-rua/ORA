"""Mesa's software EGL device is chosen by what EGL reports of it, not by index (``graphics_probe``).

The device lists are fakes in the shape Mesa 23.2 and NVIDIA's EGL reported on
an eight-GPU H200 host with NVIDIA render nodes on 2026-11-06; nothing loads libEGL.
"""

from __future__ import annotations

from typing import Any

import pytest

from ora.environment import graphics_probe

DRM = ["EGL_EXT_device_drm", "EGL_EXT_device_drm_render_node"]
SOFTWARE = ["EGL_MESA_device_software", "EGL_EXT_device_drm_render_node"]
NVIDIA = ["EGL_NV_device_cuda", "EGL_EXT_device_drm", "EGL_EXT_device_drm_render_node",
          "EGL_EXT_device_query_name", "EGL_EXT_device_persistent_id"]
MESA = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
LANE = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "__EGL_VENDOR_LIBRARY_FILENAMES": MESA,
        "LIBGL_ALWAYS_SOFTWARE": "1", "CUDA_VISIBLE_DEVICES": "3,0", "MUJOCO_EGL_DEVICE_ID": "0"}
IDENTITY = {"vendor": "Mesa Project", "driver": "llvmpipe (LLVM 15.0.7, 256 bits)", "egl_version": "1.5",
            "gl_version": "4.5 (Compatibility Profile) Mesa 23.2.1"}


def _device(index: int, extensions: list[str], drm: str | None = None, **names: str) -> dict[str, Any]:
    return {"index": index, "extensions": extensions, "drm_device_file": drm,
            "vendor": names.get("vendor"), "renderer": names.get("renderer"), "driver": names.get("driver")}


def _render_nodes(count: int) -> list[dict[str, Any]]:
    """Mesa on a host with ``/dev/dri``: every render node first, its software device last."""

    nodes = [_device(index, DRM, f"/dev/dri/card{count - 1 - index}") for index in range(count)]
    return nodes + [_device(count, SOFTWARE)]


def _bind(environment: dict[str, str], devices: list[dict[str, Any]]) -> tuple[Any, list[tuple[str, int]]]:
    opened: list[tuple[str, int]] = []

    def identity(_library: Any, handle: str, index: int) -> dict[str, str]:
        opened.append((handle, index))
        return IDENTITY

    record = graphics_probe.bind_software_egl_device(
        environment, enumerate_devices=lambda: (None, [f"handle-{d['index']}" for d in devices], devices),
        render_identity=identity)
    return record, opened


def test_a_host_with_render_nodes_binds_mesa_s_software_device_not_device_0() -> None:
    environment = dict(LANE)
    record, opened = _bind(environment, _render_nodes(8))

    assert environment["MUJOCO_EGL_DEVICE_ID"] == "8"
    assert opened == [("handle-8", 8)]  # only the chosen device is opened
    assert record["schema"] == graphics_probe.SOFTWARE_EGL_DEVICE_SCHEMA
    assert (record["index"], record["vendor"], record["driver"]) == (8, "Mesa Project", IDENTITY["driver"])
    assert record["configured_device_id"] == "0"
    assert record["devices"][0] == "0:/dev/dri/card7" and len(record["devices"]) == 9


def test_a_host_without_render_nodes_keeps_device_0() -> None:
    environment = dict(LANE)
    record, _ = _bind(environment, _render_nodes(0))
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "0" and record["index"] == 0


def test_nvidia_devices_without_a_drm_file_are_not_taken_for_the_software_device() -> None:
    nvidia = [_device(index, NVIDIA, vendor="NVIDIA", renderer="NVIDIA H200", driver="nvidia") for index in range(2)]
    mesa = [_device(2 + index, DRM, f"/dev/dri/card{1 - index}") for index in range(2)] + [_device(4, SOFTWARE)]
    assert graphics_probe.select_software_egl_device(nvidia + mesa)["index"] == 4


@pytest.mark.parametrize(("device", "software"), [
    (_device(0, SOFTWARE), True),
    (_device(0, DRM, "/dev/dri/card0"), False),
    (_device(0, NVIDIA, vendor="NVIDIA", renderer="NVIDIA H200", driver="nvidia"), False),
    (_device(0, ["EGL_EXT_device_drm", "EGL_EXT_device_query_name"], vendor="Mesa", renderer="softpipe"), True),
    (_device(0, []), True),  # no DRM device file and no hardware extension
])
def test_what_counts_as_the_software_device(device: dict[str, Any], software: bool) -> None:
    assert graphics_probe.is_software_egl_device(device) is software


def test_no_software_device_is_refused_naming_every_device_seen() -> None:
    environment = dict(LANE)
    with pytest.raises(graphics_probe.SoftwareEglDeviceError,
                       match="none of the 2 EGL devices .* devices seen: 0:/dev/dri/card2, 1:/dev/dri/card1"):
        _bind(environment, _render_nodes(3)[:2])
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "0"


def test_gpu_rendering_is_left_alone() -> None:
    environment = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "CUDA_VISIBLE_DEVICES": "3",
                   "MUJOCO_EGL_DEVICE_ID": "3"}

    def unexpected() -> Any:
        raise AssertionError("devices enumerated without a software request")

    assert graphics_probe.bind_software_egl_device(environment, enumerate_devices=unexpected) is None
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "3"


@pytest.mark.parametrize(("environment", "requested"), [
    ({"MUJOCO_GL": "egl", "LIBGL_ALWAYS_SOFTWARE": "1"}, True),
    ({"MUJOCO_GL": "egl", "__EGL_VENDOR_LIBRARY_FILENAMES": MESA}, True),
    ({"MUJOCO_GL": "egl", "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"}, False),
    ({"MUJOCO_GL": "egl", "LIBGL_ALWAYS_SOFTWARE": "0"}, False),
    ({"MUJOCO_GL": "osmesa", "LIBGL_ALWAYS_SOFTWARE": "1"}, False),
])
def test_software_rendering_is_requested_by_mesa_s_vendor_or_libgl_always_software(
        environment: dict[str, str], requested: bool) -> None:
    assert graphics_probe.software_egl_requested(environment) is requested
