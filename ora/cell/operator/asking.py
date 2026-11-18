"""Questions a guided setup step asks the person at the cell, on the terminal or through the broker.

A setup step (``handeye.calibrate``, ``tools.orient_grippers``) asks through an ``Asker``:
``confirm`` for a yes or no (``task``, ``motion``, ``write``) and ``value`` for an answer
in words or numbers. ``BrokerAsker`` posts each question to the approval broker
(``approval.py``), where the cell board, ``ora cell approve`` and the play terminal answer
it; ``TerminalAsker`` asks on this process's terminal.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from . import approval

ARM_WORDS = {"left": "左臂", "right": "右臂"}


class Asker(Protocol):
    def confirm(self, kind: str, title: str, text: str, details: Mapping[str, Any], timeout_s: float) -> tuple[bool, str | None]: ...
    def value(self, title: str, text: str, details: Mapping[str, Any], timeout_s: float) -> tuple[str | None, str | None]: ...


class BrokerAsker:
    """Questions through the approval broker: whoever answers it (board, ``ora cell approve``, play terminal)."""

    def __init__(self, label: str) -> None:
        self.label = label

    def confirm(self, kind, title, text, details, timeout_s):
        return approval.ask(kind, title, text, details, timeout_s=timeout_s, asker=self.label)

    def value(self, title, text, details, timeout_s):
        return approval.ask_value(title, text, details, timeout_s=timeout_s, asker=self.label)


class TerminalAsker:
    """Questions on this process's terminal (``/dev/tty``)."""

    def _line(self, text: str, prompt: str) -> str | None:
        try:
            with open("/dev/tty", "r+", buffering=1) as tty:
                tty.write("\n" + text.rstrip() + "\n" + prompt)
                return tty.readline().strip()
        except OSError:
            return None

    def confirm(self, kind, title, text, details, timeout_s):
        prompt = "[回车：完成 / no：停止] > " if kind == "task" else "[yes/no] > "
        reply = self._line(f"{title}\n{text}", prompt)
        if reply is None:
            return False, "no terminal to ask on"
        ok = reply.lower() in ({"", "y", "yes", "ok"} if kind == "task" else {"y", "yes"})
        return (True, None) if ok else (False, reply if reply.lower() not in {"n", "no"} else "declined at the terminal")

    def value(self, title, text, details, timeout_s):
        reply = self._line(f"{title}\n{text}", "> ")
        return (reply, None) if reply else (None, "no value given")


def arm_words(arms: Sequence[str]) -> str:
    return "和".join(ARM_WORDS.get(a, f"{a} 臂") for a in arms)
