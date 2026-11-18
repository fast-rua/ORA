"""A constructed tool runtime served the way a run serves it, for tests that drove the legacy tool handlers.

``served(host)`` does to a runtime what ``build_tool_runtime`` and the episode loop do once it is constructed on a
registered surface:

* the surface keeps every capability it declares, ``toolkit_synchronous_execution_v1`` and
  ``toolkit_legacy_projection_v1`` included (``surface_catalog.handlers_only`` dropped them);
* the grants are the run's: the host was constructed with them (``surface_catalog.catalog_grants``);
* where the surface executes synchronously, an offline execution owner stands in for the episode loop's
  ``ToolActionExecutor`` (``bind_script_actions``): each action it is given reaches its target and spends
  ``NATIVE_ACTIONS``;
* the toolkit is installed (``dispatch.install``), which binds the state store. With no Supervisor to own one,
  it is the standalone store, which imports the host's current publication on each call, as every run's store does
  but RoboDojo's;
* each ``invoke`` is a call of the policy session: it is proposed first in ``agent-events.jsonl`` and carries that
  parent identity, as ``ToolGateway`` passes it;
* the current observation is where a run publishes it, ``policy-observations/<id>/`` under the run root, each
  camera file's size beside its digest, and its manifest digest and generation in the control pointer
  (``toolkit_recording.publication`` moves a fixture's there).
"""

from __future__ import annotations

import itertools
import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Mapping

from ora.agent.toolkit.interface.types import canonical, digest
from ora.agent.toolkit.host.runtime.dispatch import install
from tests.support.agent.toolkit_recording import proposal, publication

#: The native actions each action the offline owner executes spends.
NATIVE_ACTIONS = 3


class OfflineExecutor:
    """The episode loop's execution owner with no backend behind it: every action reaches its target."""

    def __init__(self, *, receipts: str = "tool_output") -> None:
        self.loop = SimpleNamespace(plan={"toolkit_receipt_delivery": receipts})
        self.requests: list[Mapping[str, Any]] = []
        self.budgets: list[int] = []
        self.actions: list[dict[str, Any]] = []
        self.log = SimpleNamespace(summary=lambda: {
            "native_actions_used": sum(item["receipt"]["native_actions"] for item in self.actions),
            "actions": self.actions})

    @contextmanager
    def invocation(self, name: str, context: Mapping[str, Any], requested: int):
        self.budgets.append(requested)
        self.actions = []
        yield self

    def execute_action(self, action: Mapping[str, Any], budget: int) -> dict[str, Any]:
        self.requests.append(action)
        receipt = {"status": "reached", "action_id": action["action_id"], "native_actions": NATIVE_ACTIONS}
        self.actions.append({"refusal": None, "receipt": receipt})
        return {"receipt": receipt, "remaining_budget": self.budgets[-1] - NATIVE_ACTIONS}


def run_surface(profile: str, **options: Any) -> Any:
    """``profile`` as a run resolves it, with the grants a plan gives it: every capability it declares."""
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from tests.support.agent.surface_catalog import catalog_grants

    return resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile), **options)


def on_surface(host: Any, profile: str) -> Any:
    """A constructed ``host`` put on ``profile`` as a run binds it: the resolved surface, its tools, its grants."""
    from tests.support.agent.surface_catalog import catalog_grants

    host.toolkit_grants = catalog_grants(profile)
    host.surface = run_surface(profile)
    host.allowed_tools = tuple(item["name"] for item in host.surface.definitions)
    return host


def _published(host: Any) -> None:
    """Move the current observation, if there is one, to the run's own layout."""
    try:
        control, source, _ = host._current()
    except Exception:  # noqa: BLE001  no observation is open: the tools that need one refuse for themselves
        return
    root = host.control_root.parent / "policy-observations" / control["observation_id"]
    if not (root / "observation.json").is_file():  # the toolkit reads it there, never from the sandbox mount
        control = publication(host)
    manifest = json.loads((root / "observation.json").read_bytes())
    # A run's manifest names each camera file's size beside its digest.
    files = [record for camera in manifest.get("cameras", {}).values() for record in camera.get("files", {}).values()]
    if any("bytes" not in record for record in files):
        for record in files:
            record.setdefault("bytes", (root / record["path"]).stat().st_size)
        (root / "observation.json").write_bytes(canonical(manifest))
        control = {**control, "observation_sha256": digest((root / "observation.json").read_bytes())}
    # A run's pointer counts its publications.
    (host.control_root / "current-observation.json").write_bytes(canonical({"generation": 0, **control}))


def served(host: Any) -> Any:
    """``host`` with the toolkit installed as a run installs it, each ``invoke`` a proposed policy call."""
    _published(host)
    if host.surface.supports("toolkit_synchronous_execution_v1") and getattr(host, "toolkit_executor", None) is None:
        host.toolkit_executor = OfflineExecutor()
    install(host)
    parent = host.invoke
    calls = itertools.count()

    def invoke(name: str, arguments: Mapping[str, Any], *, call_context: Mapping[str, Any] | None = None) -> Any:
        if call_context is None:
            call_context = proposal(host, name, dict(arguments), next(calls))
        return parent(name, arguments, call_context=call_context)

    host.invoke = host.invoke_with_context = invoke
    return host


def republished(host: Any, observation_id: str) -> None:
    """The Supervisor's next publication: the current observation again under ``observation_id``, one generation on."""
    import shutil

    control = json.loads((host.control_root / "current-observation.json").read_bytes())
    root = host.control_root.parent / "policy-observations"
    target = root / observation_id
    shutil.copytree(root / control["observation_id"], target)
    manifest = json.loads((target / "observation.json").read_bytes())
    (target / "observation.json").chmod(0o644)
    (target / "observation.json").write_bytes(canonical({**manifest, "observation_id": observation_id}))
    (host.control_root / "current-observation.json").write_bytes(canonical({
        **control, "observation_id": observation_id, "source_root": str(target),
        "observation_sha256": digest((target / "observation.json").read_bytes()),
        "generation": _next_generation(control.get("generation", 0), observation_id)}))


def _next_generation(generation: Any, observation_id: str) -> Any:
    """A counted generation counts on; the projector's is a digest, which a new publication changes."""
    if isinstance(generation, int):
        return generation + 1
    return digest(f"{generation}:{observation_id}".encode())


def recameraed(host: Any, cameras: Mapping[str, str]) -> None:
    """The current publication with its cameras renamed (``{published: new}``), its files where they are."""
    control = json.loads((host.control_root / "current-observation.json").read_bytes())
    path = host.control_root.parent / "policy-observations" / control["observation_id"] / "observation.json"
    manifest = json.loads(path.read_bytes())
    manifest["cameras"] = {cameras.get(name, name): record for name, record in manifest["cameras"].items()}
    path.chmod(0o644)
    path.write_bytes(canonical(manifest))
    (host.control_root / "current-observation.json").write_bytes(canonical({
        **control, "observation_sha256": digest(path.read_bytes())}))
