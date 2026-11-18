"""A history message as a provider receives it, for tests that assemble a provider request.

The agent core strips Harness metadata (every key that starts with ``_`` except
the opaque provider-continuation key) and an earlier response's
``reasoning_content`` before a message reaches a provider wire; the TypeScript
core does it in ``providerMessage`` (``ora-pi/src/context/projection.ts``). The
Python copy left with the test-only part of ``ora/agent/context/projection.py``;
tests that build a wire request from a message they assembled use this one.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from ora.agent.providers.continuation import PROVIDER_CONTINUATION_MESSAGE_KEY


def provider_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Strip Harness metadata while retaining opaque provider continuity."""

    return {
        key: copy.deepcopy(value) for key, value in message.items()
        if (
            not str(key).startswith("_")
            or key == PROVIDER_CONTINUATION_MESSAGE_KEY
        ) and key != "reasoning_content"
    }
