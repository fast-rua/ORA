"""Categorized data capabilities and evidence primitives, loaded lazily."""
from importlib import import_module

__all__ = [
    'AccessDenied', 'ActionNotExecuted', 'AmbiguousRecord', 'CallContext', 'CorruptRecord', 'Divergence',
    'InterfaceError', 'NotRecorded', 'PublicData', 'Response', 'ServiceUnavailable', 'SimulatorRequired',
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module('.types', __name__), name)
