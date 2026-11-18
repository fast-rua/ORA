"""Build the provider a profile's protocol names: ``openai_compatible`` or
``anthropic_messages``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ora.agent.contracts import PolicySessionError
from ora.agent.providers.recovery import RECOVERY_POLICIES
from .anthropic_messages import AnthropicMessagesProvider, _ANTHROPIC_CONTINUATION_REPLAY_MODES
from .openai_compatible import OpenAICompatibleProvider


def build_provider(
    provider: str, *, base_url: str, api_key: str, timeout_seconds: float = 900.0,
    auth_scheme: str | None = None, continuation_replay: str = "exact_v1",
    session_affinity_id: str | None = None,
    standby_api_keys: Sequence[str] = (),
    credential_start_index: int = 0,
    recovery_policy: str = "legacy_v1",
    recovery_window_s: float | None = None,
    output_token_parameter: str | None = None,
    tool_strictness: str | None = None,
    reasoning_field: str | None = None,
    request_options: Mapping[str, Any] | None = None,
) -> OpenAICompatibleProvider | AnthropicMessagesProvider:
    """Construct the explicit RunSpec-selected native provider."""

    if recovery_policy != "legacy_v1" and recovery_policy not in RECOVERY_POLICIES:
        raise PolicySessionError("unsupported native provider recovery policy")
    if recovery_window_s is not None and recovery_policy == "legacy_v1":
        raise PolicySessionError("recovery_window_s requires the same-credential deadline recovery policy")
    if provider == "openai_compatible":
        if auth_scheme not in {None, "bearer"}:
            raise PolicySessionError("OpenAI-compatible providers require bearer auth")
        arguments: dict[str, Any] = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout_seconds": timeout_seconds,
        }
        if recovery_policy != "legacy_v1":
            arguments["recovery_policy"] = recovery_policy
        if recovery_window_s is not None:
            arguments["recovery_window_s"] = float(recovery_window_s)
        if output_token_parameter is not None:
            arguments["output_token_parameter"] = output_token_parameter
        if tool_strictness is not None:
            arguments["tool_strictness"] = tool_strictness
        if reasoning_field is not None:
            arguments["reasoning_field"] = reasoning_field
        if request_options is not None:
            arguments["request_options"] = request_options
        if session_affinity_id is not None:
            arguments["session_affinity_id"] = session_affinity_id
        return OpenAICompatibleProvider(**arguments)
    if provider == "anthropic_messages":
        if (output_token_parameter is not None or tool_strictness is not None or reasoning_field is not None
                or request_options is not None):
            raise PolicySessionError(
                "output_token_parameter, tool_strictness, reasoning_field and request_options "
                "apply only to OpenAI-compatible providers"
            )
        arguments: dict[str, Any] = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout_seconds": timeout_seconds,
        }
        if recovery_policy != "legacy_v1":
            arguments["recovery_policy"] = recovery_policy
        if recovery_window_s is not None:
            arguments["recovery_window_s"] = float(recovery_window_s)
        if auth_scheme is not None:
            arguments["auth_scheme"] = auth_scheme
        if continuation_replay not in _ANTHROPIC_CONTINUATION_REPLAY_MODES:
            raise PolicySessionError("unsupported Anthropic continuation replay mode")
        if session_affinity_id is not None:
            arguments["session_affinity_id"] = session_affinity_id
        if standby_api_keys:
            arguments["standby_api_keys"] = tuple(standby_api_keys)
        if credential_start_index:
            arguments["credential_start_index"] = int(credential_start_index)
        return AnthropicMessagesProvider(**arguments)
    raise PolicySessionError(f"unsupported native provider protocol: {provider}")
