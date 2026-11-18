"""Small file helpers of the real-cell kit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, value: Any) -> None:
    """Atomic JSON write; NumPy values become lists, NaN is refused."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                                    default=lambda item: np.asarray(item).tolist()) + "\n")
    temporary.replace(path)
