"""Durable parent-call execution evidence, never an instruction to re-execute.

An unclosed intent means physics may have changed without a published state.
Both closed and open interrupted parents terminate the run. Reading a ledger
only recovers evidence; it never authorizes continuing or repeating the tool.
"""

from copy import deepcopy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

SCHEMA = "ora.tool-execution.v1"


class ToolExecutionInterrupted(RuntimeError):
    """An interrupted parent has a known state but may not resume."""


class WorldStateUnknown(RuntimeError):
    """An execution intent has no matching receipt and publication."""


class ToolExecutionLog:
    def __init__(self, root: Path, tool_call_id: str, model_call_id: str):
        if not tool_call_id or not model_call_id:
            raise ValueError("execution ledger requires model and tool invocation identities")
        identity = json.dumps([model_call_id, tool_call_id], ensure_ascii=False, separators=(",", ":"))
        self.path = root / "native" / "tool-executions" / (hashlib.sha256(identity.encode()).hexdigest() + ".json")
        self.model_call_id = model_call_id
        self.tool_call_id = tool_call_id
        self.value: dict[str, Any] | None = None

    def create(self, *, requested: int, admitted: int, observation: dict) -> None:
        if self.path.exists():
            raise RuntimeError("parent tool invocation already has execution evidence")
        self.value = {
            "schema": SCHEMA, "tool_call_id": self.tool_call_id, "model_call_id": self.model_call_id,
            "native_action_budget_requested": requested,
            "native_action_budget_admitted": admitted,
            "native_actions_used": 0, "native_actions_remaining": admitted,
            "initial_observation": deepcopy(observation), "final_observation": deepcopy(observation),
            "actions": [], "pending": None, "finished": False,
        }
        self._save()

    def begin(self, *, action: dict, admitted: int, observation: dict) -> None:
        if self.value["pending"] is not None:
            raise WorldStateUnknown("an earlier execution intent is still open")
        if observation != self.value["final_observation"]:
            raise ValueError("action does not start at the last published observation")
        # A refused action ran nothing and published nothing, so the next action proposed on the same
        # observation carries its identity (a code-mode script's next call); an executed one never repeats.
        if any(item["action_id"] == action["action_id"] and item["refusal"] is None
               for item in self.value["actions"]):
            raise ValueError("action identity is already present in this tool invocation")
        self.value["pending"] = {
            "action_id": action["action_id"], "action": deepcopy(action),
            "native_action_budget_admitted": admitted,
            "before_observation": deepcopy(observation),
        }
        self._save()

    def complete(self, *, used: int, observation: dict, receipt: dict,
                 receipt_event_id: str, receipt_ref: dict, refusal: str | None = None) -> None:
        pending = self.value["pending"]
        if pending is None:
            raise ValueError("receipt has no prior execution intent")
        if type(used) is not int or not 0 <= used <= min(
            pending["native_action_budget_admitted"], self.value["native_actions_remaining"]
        ):
            raise ValueError("execution receipt exceeds admitted budget")
        if receipt.get("action_id") != pending["action_id"]:
            raise WorldStateUnknown("receipt belongs to another action")
        if not receipt_event_id or not isinstance(receipt_ref.get("sha256"), str):
            raise ValueError("execution needs an immutable receipt reference")
        if any(item["receipt_event_id"] == receipt_event_id for item in self.value["actions"]):
            raise ValueError("duplicate receipt event identity")
        item = {**pending, "after_observation": deepcopy(observation),
                "native_actions_used": used, "receipt_event_id": receipt_event_id,
                "receipt_ref": deepcopy(receipt_ref), "receipt": deepcopy(receipt),
                "refusal": refusal}
        self.value["actions"].append(item)
        self.value["pending"] = None
        self.value["native_actions_used"] += used
        self.value["native_actions_remaining"] -= used
        self.value["final_observation"] = deepcopy(observation)
        self._save()

    def finish(self) -> dict:
        if self.value["pending"] is not None:
            raise WorldStateUnknown("execution intent lacks a receipt or published state")
        self.value["finished"] = True
        self._save()
        return self.summary()

    def summary(self) -> dict:
        return deepcopy({key: value for key, value in self.value.items()
                         if key not in {"pending", "finished"}})

    def recover(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            self.value = json.loads(self.path.read_text(encoding="utf-8"))
            self._validate()
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise WorldStateUnknown("execution evidence cannot establish the last world state") from error
        if self.value["pending"] is not None:
            raise WorldStateUnknown("unclosed execution intent; refuse further physical actions")
        return self.summary()

    def _validate(self) -> None:
        value = self.value
        if (value["schema"] != SCHEMA or value["tool_call_id"] != self.tool_call_id
                or value["model_call_id"] != self.model_call_id):
            raise ValueError("execution ledger identity mismatch")
        def count(number):
            if type(number) is not int or number < 0:
                raise ValueError("invalid execution count")
            return number
        requested = count(value["native_action_budget_requested"])
        admitted = count(value["native_action_budget_admitted"])
        used = count(value["native_actions_used"])
        remaining = count(value["native_actions_remaining"])
        if admitted > requested or used + remaining != admitted:
            raise ValueError("execution budget does not balance")
        previous = value["initial_observation"]
        executed_actions, seen_receipts, total = set(), set(), 0
        for action in value["actions"]:
            consumed = count(action["native_actions_used"])
            refused = action["refusal"] is not None
            if (action["action_id"] in executed_actions or action["receipt_event_id"] in seen_receipts
                    or (refused and (consumed != 0 or action["after_observation"] != previous))
                    or action["receipt"].get("action_id") != action["action_id"]
                    or not action["receipt_event_id"]
                    or action["before_observation"] != previous
                    or consumed > count(action["native_action_budget_admitted"])
                    or not re.fullmatch(r"[0-9a-f]{64}", action["receipt_ref"]["sha256"])):
                raise ValueError("invalid action/receipt chain")
            previous = action["after_observation"]
            if not refused:
                executed_actions.add(action["action_id"])
            seen_receipts.add(action["receipt_event_id"])
            total += consumed
        if total != used or previous != value["final_observation"]:
            raise ValueError("execution final publication or consumption differs")
        if "pending" not in value or type(value["finished"]) is not bool:
            raise ValueError("execution completion status missing")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
