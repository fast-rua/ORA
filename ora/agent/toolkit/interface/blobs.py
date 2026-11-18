"""Hash-verified public artifact references, not an arbitrary file API."""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .types import AccessDenied, CorruptRecord, InterfaceError, canonical, digest

IO_ROOT = "native/toolkit-io"


class BlobStore:
    """Only explicitly registered public artifacts and this store's blobs are readable.

    Registration belongs to trusted assembly. Tools receive references, not a
    registration capability. This is a same-process boundary, not a sandbox.
    """

    def __init__(self, root: Path, *, writable: bool = False) -> None:
        self.root = Path(root).resolve()
        self.writable = writable
        self.input_store: BlobStore | None = None
        self._registered: set[bytes] = set()

    def _path(self, name: str) -> Path:
        path = PurePosixPath(name)
        if not name or str(path) != name or path.is_absolute() or ".." in path.parts or "\\" in name:
            raise AccessDenied("public reference must be run-relative")
        resolved = (self.root / name).resolve()
        if not resolved.is_relative_to(self.root) or resolved != self.root / name:
            raise AccessDenied("public reference escapes run or aliases another data category")
        return resolved

    def register(self, ref: dict) -> dict:
        """Trusted assembly registers a verified reference; no discovery by tools."""
        allowed = {"path", "sha256", "bytes", "media_type", "array", "json_pointer", "byte_scope"}
        if (set(ref) - allowed or not {"path", "sha256", "bytes", "media_type"} <= ref.keys()
                or type(ref["bytes"]) is not int or ref["bytes"] < 0
                or not isinstance(ref["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])
                or not isinstance(ref["media_type"], str)):
            raise CorruptRecord("invalid public blob reference fields")
        if "json_pointer" in ref and (not isinstance(ref["json_pointer"], str) or ref.get("byte_scope") != "whole_file"):
            raise CorruptRecord("JSON pointer must declare the whole-file byte scope")
        name = ref["path"]
        blob = re.fullmatch(r"native/toolkit-io/blobs/sha256/([0-9a-f]{64})", name)
        observation = re.fullmatch(
            r"policy-observations/[^/.][^/]*/(?:observation\.json|proprioception\.json|"
            r"[^/.][^/]*/(?:rgb\.png|depth-m\.npy|calibration\.json))", name)
        if not blob and not observation:
            raise AccessDenied("path is outside the public artifact allowlist")
        if blob and blob.group(1) != ref.get("sha256"):
            raise CorruptRecord("blob filename differs from its content hash")
        self._path(name)
        self._verify(ref)
        self._registered.add(canonical(ref))
        return json.loads(canonical(ref))

    def reference(self, path: str, *, media_type: str = "application/octet-stream") -> dict:
        data = self._path(path).read_bytes()
        ref = dict(path=path, sha256=digest(data), bytes=len(data), media_type=media_type)
        return self.register(ref)

    def _verify(self, ref: dict) -> bytes:
        try:
            data = self._path(ref["path"]).read_bytes()
        except FileNotFoundError as error:
            raise CorruptRecord("missing public blob") from error
        if len(data) != ref.get("bytes") or digest(data) != ref.get("sha256"):
            raise CorruptRecord("public blob length/hash mismatch")
        if "array" in ref:
            import numpy as np
            meta = ref["array"]
            if ref["media_type"] != "application/x-numpy-raw" or set(meta) != {"dtype", "shape", "order", "byteorder"}:
                raise CorruptRecord("array metadata must describe a raw array")
            dtype = np.dtype(meta["dtype"])
            shape = meta["shape"]
            if (dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufc" or meta["order"] not in ("C", "F")
                    or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)):
                raise CorruptRecord("invalid raw array description")
            import math
            if math.prod(shape) * dtype.itemsize != len(data):
                raise CorruptRecord("array shape does not describe blob bytes")
            expected = "not_applicable" if dtype.byteorder == "|" else (
                "little" if dtype.str[0] == "<" else "big")
            if meta["byteorder"] != expected:
                raise CorruptRecord("array byte order mismatch")
        return data

    def read(self, ref: dict) -> bytes:
        if canonical(ref) not in self._registered:
            if self.input_store is not None:
                return self.input_store.read(ref)
            raise AccessDenied("unregistered public artifact")
        return self._verify(ref)

    def put(self, data: bytes, *, media_type: str = "application/octet-stream",
            array: dict | None = None) -> dict:
        if not self.writable:
            raise AccessDenied("read-only evidence store")
        name = f"{IO_ROOT}/blobs/sha256/{digest(data)}"
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(data)
        except FileExistsError:
            if path.read_bytes() != data:
                raise CorruptRecord("content address already contains different bytes")
        ref = dict(path=name, sha256=digest(data), bytes=len(data), media_type=media_type)
        if array is not None:
            ref["array"] = array
        return self.register(ref)

    def put_array(self, value: Any) -> dict:
        import numpy as np
        value = np.asarray(value)
        if value.dtype.hasobject or value.dtype.fields is not None or value.dtype.kind not in "biufc":
            raise InterfaceError("object/structured arrays are not public wire arrays")
        order = "F" if value.flags.f_contiguous and not value.flags.c_contiguous else "C"
        dtype = value.dtype.str
        byteorder = "not_applicable" if dtype[0] == "|" else "little" if dtype[0] == "<" else "big"
        return self.put(value.tobytes(order=order), media_type="application/x-numpy-raw",
                        array=dict(dtype=dtype, shape=list(value.shape), order=order, byteorder=byteorder))

    def array(self, ref: dict) -> Any:
        import numpy as np
        meta = ref["array"]
        return np.frombuffer(self.read(ref), dtype=meta["dtype"]).reshape(meta["shape"], order=meta["order"])


def input_identity(ref: dict) -> dict:
    """Storage relocation does not change the semantic request key."""
    keys = ("sha256", "bytes", "media_type", "array", "json_pointer", "byte_scope")
    return {key: ref[key] for key in keys if key in ref}
