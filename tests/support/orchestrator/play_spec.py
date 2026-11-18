"""Test helpers for play spec, shared by several test modules.

They were defined in ``tests/ora/orchestrator/spec/test_play_spec.py``; test modules
import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations


POLICY = {
    "kind": "native_api", "provider": "openai_compatible", "model": "fake", "endpoint": "http://127.0.0.1:9/v1",
    "endpoint_transport": "loopback_http_v1", "api_key_env": "ORA_API_KEY", "base_url_env": "ORA_BASE_URL",
    "context_window_tokens": 32768, "max_output_tokens": 2048, "supports_images": True, "stream": False,
    "require_live_marker": False, "reasoning_effort": "provider_default",
}
