"""RFC 8785 JSON Canonicalization Scheme, matching ``ora-pi/src/jcs.ts``.

ORA's ``canonical_bytes`` is Python ``json.dumps`` with sorted keys and compact
separators. JCS differs in number layout (ECMAScript ``Number::toString``),
key order (UTF-16 code units) and integer range (IEEE-754 doubles only). Agent
event schema v2 hashes with JCS so a TypeScript writer and the Python verifier
agree without emulating each other's float formatting.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

MAX_SAFE_INTEGER = 2**53 - 1


class JcsError(ValueError):
    """A value that JCS cannot represent exactly."""


def _digits(value: float) -> tuple[str, int]:
    """Shortest round-trip digits and the decimal-point position of ``value``."""

    text = repr(abs(value))
    if "e" in text:
        mantissa, exponent = text.split("e")
        point = mantissa.find(".")
        integer_digits = len(mantissa) if point < 0 else point
        digits = mantissa.replace(".", "")
        decpt = integer_digits + int(exponent)
    else:
        integer, fraction = text.split(".")
        digits = integer + fraction
        decpt = len(integer)
    stripped = digits.lstrip("0")
    decpt -= len(digits) - len(stripped)
    digits = stripped.rstrip("0") or "0"
    return digits, decpt


def es_number(value: float) -> str:
    """ECMAScript ``Number::toString`` for a finite double."""

    if not math.isfinite(value):
        raise JcsError("JCS forbids NaN and Infinity")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    digits, n = _digits(value)
    k = len(digits)
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    exponent = n - 1
    head = digits if k == 1 else digits[0] + "." + digits[1:]
    return sign + head + "e" + ("+" if exponent >= 0 else "-") + str(abs(exponent))


def _string(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise JcsError("JCS forbids lone surrogates") from exc
    return json.dumps(value, ensure_ascii=False)


def jcs_string(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise JcsError(f"JCS cannot represent integer {value} exactly; send it as a string")
        return str(value)
    if isinstance(value, float):
        return es_number(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(jcs_string(item) for item in value) + "]"
    if isinstance(value, Mapping):
        keys = sorted(value, key=lambda key: str(key).encode("utf-16-be"))
        return "{" + ",".join(
            _string(str(key)) + ":" + jcs_string(value[key]) for key in keys
        ) + "}"
    raise JcsError(f"cannot canonicalize {type(value).__name__}")


def jcs_bytes(value: Any) -> bytes:
    return jcs_string(value).encode("utf-8")
