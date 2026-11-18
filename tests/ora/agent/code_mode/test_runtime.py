"""The code tools' result schemas (``ora/agent/code_mode/runtime.py``, ``extra_result_schemas``).

The causal projection keeps only the fields a result schema declares, so an extended contract's ``bash``
declares ``allowance_left`` and, in its ``calls`` entries, the ``explain`` of a failed call; a base contract's
schemas, which the plan's composition hash covers, are unchanged.
"""

from __future__ import annotations

from ora.agent.code_mode import coding
from ora.agent.code_mode.contract import resolve_contract
from ora.agent.code_mode.runtime import ALLOWANCE_FIELD, ALLOWANCE_SCHEMA, CALLS_SCHEMA, extra_result_schemas


def test_only_an_extended_bash_declares_its_allowance() -> None:
    base = resolve_contract(True)
    script = coding.BASH_TOOL
    expected = coding.result_schemas()
    assert extra_result_schemas(base) == expected
    extended = extra_result_schemas({**base, "library": "extended"})
    assert extended[script]["properties"][ALLOWANCE_FIELD] == ALLOWANCE_SCHEMA
    assert {**extended[script]["properties"]} == {**expected[script]["properties"], ALLOWANCE_FIELD: ALLOWANCE_SCHEMA,
                                                  "calls": CALLS_SCHEMA}
    assert CALLS_SCHEMA["items"]["properties"]["explain"] == {"type": "string"}, "a failed call's explanation"
    assert expected[script]["properties"]["calls"] == {"type": "array"}, "a base contract's entries are as before"
    assert ALLOWANCE_FIELD not in coding.BASH_RESULT_SCHEMA["properties"]
