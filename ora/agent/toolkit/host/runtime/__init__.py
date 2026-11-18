"""The default manipulation tool runtime.

``core`` holds ``ManipulationToolRuntime``: its fields, construction, the
``invoke`` dispatch table and per-round reset. The toolkit serves every tool
(``dispatch.install``); the mixin modules in this package keep the host
state it reads. ToolSurfaces and RunPlans still select it by the
``native_atomic_causal`` runtime id. The package imports ``core`` only when
``ManipulationToolRuntime`` is first used, so importing ``protocol`` or
``base`` alone does not load the mixins.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ManipulationToolRuntime": (".core", "ManipulationToolRuntime"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
