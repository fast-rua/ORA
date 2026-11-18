"""Host-safe offscreen graphics probe for isolated simulator interpreters.

Generic preflight executes this module in the configured simulator Python
before the LIBERO import probe.  MuJoCo and PyOpenGL resolve their offscreen
backend from process environment that the Harness only forwards, so a missing
EGL vendor ICD or driver library surfaces downstream as an opaque
``Could not import OffScreenRenderEnv``.  Enumerating devices here names the
actual cause while the graphics stack is still the only thing under test.

``bind_software_egl_device`` serves a worker that renders on Mesa's CPU
device: it picks that device by what EGL reports of it rather than by index.

The module imports nothing from a simulator backend and is deliberately
dependency-free.  Preflight runs it by file path rather than with ``-m`` so the
check never depends on ``ora.environment`` importing cleanly inside an
external interpreter whose package set the Harness does not own.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

GRAPHICS_PROBE_SCHEMA = "ora.offscreen-graphics-probe.v1"
SOFTWARE_EGL_DEVICE_SCHEMA = "ora.software-egl-device.v1"

DEFAULT_EGL_VENDOR_DIR = "/usr/share/glvnd/egl_vendor.d"

_REPORTED_ENVIRONMENT = (
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "__EGL_VENDOR_LIBRARY_DIRS",
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "LIBGL_ALWAYS_SOFTWARE",
    "LP_NUM_THREADS",
    "MUJOCO_EGL_DEVICE_ID",
)

# eglQueryDevicesEXT writes into a caller-owned array; 32 exceeds any host the
# Runtime qualifies and keeps the call a single pass.
_MAX_EGL_DEVICES = 32

# EGL_EXT_device_query, EGL_EXT_device_drm, EGL_EXT_device_query_name and
# EGL_EXT_device_persistent_id names, and the few EGL/GL tokens a context needs.
_EGL_VENDOR, _EGL_VERSION, _EGL_EXTENSIONS = 0x3053, 0x3054, 0x3055
_EGL_DRM_DEVICE_FILE_EXT, _EGL_DRIVER_NAME_EXT, _EGL_RENDERER_EXT = 0x3233, 0x335E, 0x335F
_EGL_PLATFORM_DEVICE_EXT, _EGL_OPENGL_API = 0x313F, 0x30A2
_EGL_SURFACE_TYPE, _EGL_PBUFFER_BIT, _EGL_RENDERABLE_TYPE, _EGL_OPENGL_BIT, _EGL_NONE = (
    0x3033, 0x0001, 0x3040, 0x0008, 0x3038)
_GL_VERSION, _GL_RENDERER = 0x1F02, 0x1F01

# Mesa marks its CPU device with this extension; a driver may instead name the rasterizer.
_MESA_SOFTWARE_DEVICE = "EGL_MESA_device_software"
_SOFTWARE_RASTERIZERS = ("llvmpipe", "softpipe", "swrast")
_HARDWARE_DEVICE_EXTENSIONS = frozenset({"EGL_EXT_device_drm", "EGL_NV_device_cuda"})


class SoftwareEglDeviceError(RuntimeError):
    """Software EGL rendering was requested and no usable software device exists."""


def default_libero_backend() -> str:
    """Match MuJoCo's native macOS context while retaining Linux EGL defaults."""
    return "cgl" if platform.system() == "Darwin" else "egl"


def _opengl_platform(backend: str) -> str | None:
    return {"egl": "egl", "osmesa": "osmesa", "cgl": "darwin",
            "glfw": "darwin" if platform.system() == "Darwin" else "glx",
            "glx": "glx", "wgl": "win32"}.get(backend)


def configure_libero_graphics() -> None:
    """Select graphics before importing LIBERO/robosuite; preserve explicit settings.

    MuJoCo's macOS CGL context and robosuite's hidden GLFW window use the
    native Darwin OpenGL loader. They must never inherit the Linux EGL default.
    An explicit conflicting loader is an error rather than an implicit switch.
    """
    backend = os.environ.setdefault("MUJOCO_GL", default_libero_backend())
    expected = _opengl_platform(backend)
    selected = os.environ.get("PYOPENGL_PLATFORM")
    if expected is not None and selected is not None and selected != expected:
        raise ValueError(f"PYOPENGL_PLATFORM={selected!r} contradicts MUJOCO_GL={backend!r}; expected {expected!r}")
    if expected is not None:
        os.environ.setdefault("PYOPENGL_PLATFORM", expected)


def _vendor_icd_files() -> list[str]:
    """List the EGL vendor ICD manifests visible to this process."""

    # libglvnd uses an explicit file list instead of any directory scan.
    explicit = os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES")
    if explicit:
        return [entry for entry in explicit.split(os.pathsep) if entry]
    raw = os.environ.get("__EGL_VENDOR_LIBRARY_DIRS")
    directories = (
        [Path(entry) for entry in raw.split(os.pathsep) if entry]
        if raw else [Path(DEFAULT_EGL_VENDOR_DIR)]
    )
    files: list[str] = []
    for directory in directories:
        try:
            files.extend(sorted(str(path) for path in directory.glob("*.json")))
        except OSError:
            continue
    return files


def query_egl_devices() -> int:
    """Return how many EGL devices the current loader can enumerate."""

    library = ctypes.CDLL("libEGL.so.1")
    library.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    library.eglGetProcAddress.restype = ctypes.c_void_p
    address = library.eglGetProcAddress(b"eglQueryDevicesEXT")
    if not address:
        raise RuntimeError(
            "libEGL.so.1 does not export eglQueryDevicesEXT; the loaded EGL is "
            "not a device-enumerating GLVND/NVIDIA implementation"
        )
    query = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
    )(address)
    devices = (ctypes.c_void_p * _MAX_EGL_DEVICES)()
    count = ctypes.c_int(0)
    if not query(_MAX_EGL_DEVICES, devices, ctypes.byref(count)):
        raise RuntimeError("eglQueryDevicesEXT reported failure")
    return int(count.value)


def software_egl_requested(environment: Mapping[str, str] = os.environ) -> bool:
    """Whether MuJoCo's EGL is asked for Mesa's CPU rasterizer rather than a GPU.

    Either ``LIBGL_ALWAYS_SOFTWARE`` is set or Mesa's vendor ICD is the only one
    libglvnd may load.
    """

    if environment.get("MUJOCO_GL") != "egl":
        return False
    if environment.get("LIBGL_ALWAYS_SOFTWARE", "").strip().lower() in {"1", "true", "yes", "y"}:
        return True
    vendors = [entry for entry in environment.get("__EGL_VENDOR_LIBRARY_FILENAMES", "").split(os.pathsep) if entry]
    return bool(vendors) and all("mesa" in Path(entry).name.lower() for entry in vendors)


def is_software_egl_device(device: Mapping[str, Any]) -> bool:
    """Mesa's CPU device: it carries Mesa's software marker, names a software
    rasterizer, or has no DRM device file and no hardware device extension."""

    extensions = set(device.get("extensions") or ())
    names = " ".join(str(device.get(key) or "") for key in ("vendor", "renderer", "driver")).lower()
    return (_MESA_SOFTWARE_DEVICE in extensions
            or any(rasterizer in names for rasterizer in _SOFTWARE_RASTERIZERS)
            or (not device.get("drm_device_file") and not extensions & _HARDWARE_DEVICE_EXTENSIONS))


def _device_summary(device: Mapping[str, Any]) -> str:
    label = (device.get("drm_device_file")
             or " ".join(str(device[key]) for key in ("vendor", "renderer") if device.get(key))
             or " ".join(device.get("extensions") or ()) or "no DRM file, no extensions")
    return f"{device.get('index')}:{label}"


def select_software_egl_device(devices: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The first software device of ``devices`` (``is_software_egl_device``).

    Mesa lists every DRM render node first and its software device last, so the
    software device is device 0 only on a host without ``/dev/dri`` render nodes.
    Refuses, naming every device seen, when there is none.
    """

    for device in devices:
        if is_software_egl_device(device):
            return device
    seen = ", ".join(_device_summary(device) for device in devices) or "none"
    raise SoftwareEglDeviceError(
        f"software EGL rendering is requested but none of the {len(devices)} EGL devices is Mesa's "
        f"software device ({_MESA_SOFTWARE_DEVICE}, llvmpipe, softpipe or swrast); devices seen: {seen}")


def _egl_function(library: Any, name: str, restype: Any, *argtypes: Any) -> Any:
    address = library.eglGetProcAddress(name.encode())
    if not address:
        raise SoftwareEglDeviceError(f"libEGL.so.1 does not export {name}")
    return ctypes.CFUNCTYPE(restype, *argtypes)(address)


def _text(value: bytes | None) -> str | None:
    return value.decode(errors="replace") if value else None


def _enumerate_egl_devices() -> tuple[Any, list[Any], list[dict[str, Any]]]:
    """The loader's EGL devices in order, each with what EGL_EXT_device_query says of it."""

    library = ctypes.CDLL("libEGL.so.1")
    library.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    library.eglGetProcAddress.restype = ctypes.c_void_p
    query = _egl_function(library, "eglQueryDevicesEXT", ctypes.c_uint, ctypes.c_int,
                          ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))
    describe = _egl_function(library, "eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
    array = (ctypes.c_void_p * _MAX_EGL_DEVICES)()
    count = ctypes.c_int(0)
    if not query(_MAX_EGL_DEVICES, array, ctypes.byref(count)):
        raise SoftwareEglDeviceError("eglQueryDevicesEXT reported failure")
    handles = [array[index] for index in range(count.value)]
    devices = []
    for index, handle in enumerate(handles):
        devices.append({
            "index": index,
            "extensions": (_text(describe(handle, _EGL_EXTENSIONS)) or "").split(),
            "drm_device_file": _text(describe(handle, _EGL_DRM_DEVICE_FILE_EXT)),
            "vendor": _text(describe(handle, _EGL_VENDOR)),
            "renderer": _text(describe(handle, _EGL_RENDERER_EXT)),
            "driver": _text(describe(handle, _EGL_DRIVER_NAME_EXT)),
        })
    library.eglGetError.restype = ctypes.c_int
    library.eglGetError()  # a name a device lacks leaves EGL_BAD_PARAMETER behind; robosuite reads the error next
    return library, handles, devices


def _render_identity(library: Any, handle: Any, index: int) -> dict[str, str | None]:
    """Make one OpenGL context current on the device and read who renders.

    The display is terminated again afterwards, so the simulator's own context
    initializes it afresh.
    """

    for name, restype, argtypes in (
        ("eglInitialize", ctypes.c_uint, [ctypes.c_void_p] * 3),
        ("eglQueryString", ctypes.c_char_p, [ctypes.c_void_p, ctypes.c_int]),
        ("eglChooseConfig", ctypes.c_uint, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]),
        ("eglBindAPI", ctypes.c_uint, [ctypes.c_uint]),
        ("eglCreateContext", ctypes.c_void_p, [ctypes.c_void_p] * 4),
        ("eglMakeCurrent", ctypes.c_uint, [ctypes.c_void_p] * 4),
        ("eglDestroyContext", ctypes.c_uint, [ctypes.c_void_p] * 2),
        ("eglTerminate", ctypes.c_uint, [ctypes.c_void_p]),
        ("eglReleaseThread", ctypes.c_uint, []),
    ):
        function = getattr(library, name)
        function.restype, function.argtypes = restype, argtypes
    display = _egl_function(library, "eglGetPlatformDisplayEXT", ctypes.c_void_p, ctypes.c_int,
                            ctypes.c_void_p, ctypes.c_void_p)(_EGL_PLATFORM_DEVICE_EXT, handle, None)
    if not display or not library.eglInitialize(display, None, None):
        raise SoftwareEglDeviceError(f"EGL device {index} does not initialize (EGL error {library.eglGetError():#x})")
    context = None
    try:
        attributes = (ctypes.c_int * 5)(_EGL_SURFACE_TYPE, _EGL_PBUFFER_BIT, _EGL_RENDERABLE_TYPE, _EGL_OPENGL_BIT,
                                        _EGL_NONE)
        config, matched = ctypes.c_void_p(), ctypes.c_int(0)
        if not (library.eglChooseConfig(display, attributes, ctypes.byref(config), 1, ctypes.byref(matched))
                and matched.value and library.eglBindAPI(_EGL_OPENGL_API)):
            raise SoftwareEglDeviceError(f"EGL device {index} offers no OpenGL pbuffer configuration")
        context = library.eglCreateContext(display, config, None, None)
        if not context or not library.eglMakeCurrent(display, None, None, context):
            raise SoftwareEglDeviceError(
                f"EGL device {index} cannot make an OpenGL context current (EGL error {library.eglGetError():#x})")
        try:
            gl = ctypes.CDLL("libOpenGL.so.0")
        except OSError:
            gl = ctypes.CDLL("libGL.so.1")
        gl.glGetString.restype, gl.glGetString.argtypes = ctypes.c_char_p, [ctypes.c_uint]
        return {
            "vendor": _text(library.eglQueryString(display, _EGL_VENDOR)),
            "driver": _text(gl.glGetString(_GL_RENDERER)),
            "egl_version": _text(library.eglQueryString(display, _EGL_VERSION)),
            "gl_version": _text(gl.glGetString(_GL_VERSION)),
        }
    finally:
        library.eglMakeCurrent(display, None, None, None)
        if context:
            library.eglDestroyContext(display, context)
        library.eglTerminate(display)
        library.eglReleaseThread()
        library.eglGetError()


def bind_software_egl_device(
    environment: MutableMapping[str, str] = os.environ,
    *,
    enumerate_devices: Callable[[], tuple[Any, list[Any], list[dict[str, Any]]]] | None = None,
    render_identity: Callable[[Any, Any, int], Mapping[str, str | None]] | None = None,
) -> dict[str, Any] | None:
    """Point ``MUJOCO_EGL_DEVICE_ID`` at Mesa's software EGL device when software rendering is requested.

    Mesa enumerates every DRM render node first (NVIDIA's too, which it cannot
    drive) and its software device last, so a fixed device 0 is right only on
    a host without ``/dev/dri``. MuJoCo and robosuite read the variable when
    they create their first context; call this before that. The chosen device
    must make an OpenGL context current, and the record names it: its index,
    vendor (EGL) and driver (the OpenGL renderer, e.g. llvmpipe). Returns
    ``None`` and changes nothing when software rendering is not requested.
    """

    if not software_egl_requested(environment):
        return None
    library, handles, devices = (enumerate_devices or _enumerate_egl_devices)()
    device = select_software_egl_device(devices)
    identity = (render_identity or _render_identity)(library, handles[device["index"]], device["index"])
    configured = environment.get("MUJOCO_EGL_DEVICE_ID")
    environment["MUJOCO_EGL_DEVICE_ID"] = str(device["index"])
    return {
        "schema": SOFTWARE_EGL_DEVICE_SCHEMA,
        "index": device["index"],
        **dict(identity),
        "extensions": list(device.get("extensions") or ()),
        "configured_device_id": configured,
        "devices": [_device_summary(entry) for entry in devices],
    }


def probe() -> dict[str, Any]:
    """Report the offscreen graphics binding this interpreter would resolve."""

    # Unset bindings follow the same platform selection as the isolated worker.
    backend = os.environ.get("MUJOCO_GL") or default_libero_backend()
    pyopengl = os.environ.get("PYOPENGL_PLATFORM")
    result: dict[str, Any] = {
        "schema": GRAPHICS_PROBE_SCHEMA,
        "backend": backend,
        "environment": {
            name: os.environ.get(name) for name in _REPORTED_ENVIRONMENT
        },
        "ld_library_path_configured": bool(os.environ.get("LD_LIBRARY_PATH")),
        "vendor_icd_files": [],
        "egl_devices": None,
        "usable": False,
        "error": None,
    }
    expected_platform = _opengl_platform(backend)
    if expected_platform is None:
        result["error"] = f"Unsupported offscreen graphics backend: {backend!r}"
        return result
    if pyopengl is not None and pyopengl != expected_platform:
        result["error"] = (
            f"PYOPENGL_PLATFORM={pyopengl!r} contradicts MUJOCO_GL={backend!r}; "
            "MuJoCo and PyOpenGL would bind different offscreen backends"
        )
        return result
    if backend != "egl":
        # osmesa and glfw are legitimate offscreen backends that do not
        # enumerate EGL devices; report the binding without inventing a verdict.
        result["usable"] = True
        return result
    result["vendor_icd_files"] = _vendor_icd_files()
    try:
        devices = query_egl_devices()
    except (OSError, RuntimeError) as exc:
        result["error"] = str(exc)
        return result
    result["egl_devices"] = devices
    result["usable"] = devices > 0
    if devices == 0:
        result["error"] = (
            "eglQueryDevicesEXT enumerated 0 devices; the EGL vendor ICD or the "
            "driver libraries are absent from this process environment"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print(json.dumps(probe(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
