"""Embodiment descriptors and the cards compiled from them.

The descriptor API (validation, the source audit, activation records and card
compilation) is importable from this package and lives in ``descriptor.py``;
``control_card``, ``geometry`` and ``validation`` are its parts, and
``environment_context`` binds a compiled card into a run's environment
context. ``ora dev embodiment`` (``descriptor.main``) previews, compiles and
checks one descriptor. The package imports ``descriptor`` only
when one of its names is first read.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ACTIVATION_RECORD_SCHEMA": (".descriptor", "ACTIVATION_RECORD_SCHEMA"),
    "COMPILED_CARD_SCHEMA": (".descriptor", "COMPILED_CARD_SCHEMA"),
    "DESCRIPTOR_SCHEMA": (".descriptor", "DESCRIPTOR_SCHEMA"),
    "DESCRIPTOR_SCHEMA_V2": (".descriptor", "DESCRIPTOR_SCHEMA_V2"),
    "EmbodimentDescriptorError": (".descriptor", "EmbodimentDescriptorError"),
    "REVIEW_TOKEN_LIMIT": (".descriptor", "REVIEW_TOKEN_LIMIT"),
    "RUNTIME_VERIFICATION_SCHEMA": (".descriptor", "RUNTIME_VERIFICATION_SCHEMA"),
    "audit_embodiment_sources": (".descriptor", "audit_embodiment_sources"),
    "canonical_sha256": (".descriptor", "canonical_sha256"),
    "compile_embodiment_card": (".descriptor", "compile_embodiment_card"),
    "compiled_card_files": (".descriptor", "compiled_card_files"),
    "descriptor_activation_ready": (".descriptor", "descriptor_activation_ready"),
    "descriptor_template": (".descriptor", "descriptor_template"),
    "load_embodiment_descriptor": (".descriptor", "load_embodiment_descriptor"),
    "render_embodiment_card": (".descriptor", "render_embodiment_card"),
    "validate_activation_record": (".descriptor", "validate_activation_record"),
    "validate_embodiment_descriptor": (".descriptor", "validate_embodiment_descriptor"),
    "write_compiled_card": (".descriptor", "write_compiled_card"),
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
