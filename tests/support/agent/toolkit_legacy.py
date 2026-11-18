"""Real runtime with an offline execution owner for model-visible parity tests."""
from contextlib import contextmanager
from types import SimpleNamespace

from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from tests.support.agent.toolkit_tools import install
from tests.support.agent.toolkit_tools import make_host
from tests.support.agent.toolkit_atomic import complete_cameras

GRANTS = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}


class Executor:
    def __init__(self, mode):
        self.loop = SimpleNamespace(plan={"toolkit_receipt_delivery": mode})
        self.requests, self.budgets = [], []
        self.spent = 3
        self.actions = []
        self.log = SimpleNamespace(summary=lambda: {"native_actions_used": sum(a["receipt"]["native_actions"] for a in self.actions), "actions": self.actions})

    @contextmanager
    def invocation(self, name, context, requested):
        self.budgets.append(requested)
        self.actions = []
        yield self

    def execute_action(self, action, budget):
        self.requests.append(action)
        assert budget == 20
        receipt = {"status": "reached", "action_id": action["action_id"], "native_actions": self.spent}
        self.actions.append({"refusal": None, "receipt": receipt})
        return {"receipt": receipt, "remaining_budget": self.budgets[-1] - self.spent}


def host(root, monkeypatch, *, migrated=True, mode="tool_output"):
    value, client, _ = make_host(root, monkeypatch)
    complete_cameras(value)
    if migrated:
        value.toolkit_grants = GRANTS
        # The legacy-visible surface left: v5, tool_output receipts only (v3's observation delivery was retired on
        # 2026-10-31).
        value.surface = resolve_tool_surface("robodojo_placement_toolkit_v5", toolkit_grants=GRANTS,
                                            toolkit_receipt_delivery=mode)
        value.allowed_tools = tuple(d["name"] for d in value.surface.definitions)
        value.toolkit_executor = Executor(mode)
        install(value)
    return value, client
