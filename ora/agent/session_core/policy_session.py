"""The PolicySession of a ``native_api`` policy and its capability declaration.

The Pi agent core returns it around ``PiPlanner`` (``ora.agent.pi.factory``).
The class keeps its name; ``runtime_kind`` "native" names the ``native_api``
runtime (harness-owned context, direct tools), not an agent core.
"""

from __future__ import annotations

from typing import Any

from ora.agent.contracts import PolicySessionCapabilities
from ora.agent.sessions.base import DelegatingPolicySession


class NativePolicySession(DelegatingPolicySession):
    def __init__(self, delegate: Any, *, supports_images: bool) -> None:
        super().__init__(
            delegate=delegate,
            capabilities=PolicySessionCapabilities(
                runtime_kind="native",
                context_owner="harness",
                tool_transport="direct",
                request_visibility="wire_exact",
                supports_images=supports_images,
                supports_interrupt=True,
            ),
        )
