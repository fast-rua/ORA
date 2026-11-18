"""Python FakeProvider outcomes for scripted responses (TS FakeProvider parity).

Usage: python tests/support/agent_core/provider/fake_provider.py IN.json OUT.json
IN.json: {"responses": [...], "calls": N, "close_after": K|null}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import canonical_text, describe_error, normalize_timing  # noqa: E402
from ora.agent.providers.fake import FakeProvider  # noqa: E402


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    provider = FakeProvider(spec["responses"])
    outcomes = []
    for index in range(spec["calls"]):
        if spec.get("close_after") == index:
            provider.close()
        request = {"model": "m", "messages": [{"role": "user", "content": f"call {index}"}]}
        try:
            outcomes.append({"response": canonical_text(normalize_timing(
                provider.complete(request=request, model_call_id=f"model-call-{index:06d}")))})
        except Exception as exc:  # noqa: BLE001
            outcomes.append({"error": describe_error(exc)})
    outcomes.append({"requests": canonical_text(provider.requests)})
    Path(sys.argv[2]).write_text(json.dumps({"outcomes": outcomes}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
