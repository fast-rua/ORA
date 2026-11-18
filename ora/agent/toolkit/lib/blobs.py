"""Typed blob decoding through DataInterface only; no files, clients or registration."""
from io import BytesIO
import numpy as np

from ..interface.api import DataInterface
from .values import Array, JsonValue


def read_json(api: DataInterface, ref: JsonValue) -> JsonValue:
    """Replay: read granted recorded bytes; absent/hash-invalid evidence fails. No cursor step.

    Example: read_json(api, observation.manifest_ref).json(). No arbitrary paths.
    """
    return JsonValue(api.artifacts.read_blob(ref.json()))


def read_array(api: DataInterface, ref: JsonValue) -> Array:
    """Replay: read granted source/recomputed bytes; missing fails. Decode raw array/NPY/RGB PNG.

    Units/frames come from the caller's typed camera/cloud; no implicit depth scaling.
    Example: read_array(api, camera.depth_ref).numpy(). PNG is RGB uint8.
    """
    spec = ref.json()
    data = api.artifacts.read_blob(spec)
    if 'array' in spec:
        meta = spec['array']
        return Array(data,meta['dtype'],tuple(meta['shape']),meta['order'])
    if spec['media_type']=='application/x-npy':
        return Array.of(np.load(BytesIO(data),allow_pickle=False))
    if spec['media_type']=='image/png':
        from PIL import Image
        with Image.open(BytesIO(data)) as image:
            return Array.of(np.asarray(image.convert('RGB')))
    raise ValueError('unsupported public array encoding')


def save_array(api: DataInterface, value: Array) -> JsonValue:
    """Replay: save derived input in evaluation CAS, not source run; no cursor step.

    No simulator/service call. Example: save_array(api, cloud.xyz_m). Hash and layout
    are included in subsequent service requests; arbitrary path writes are unavailable.
    """
    return JsonValue.of(api.artifacts.save_array(value.numpy()))
