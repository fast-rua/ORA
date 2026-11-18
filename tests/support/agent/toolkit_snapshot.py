"""Two completed publications and intervening tool writes, with real public pixels."""
import json

from ora.agent.toolkit.interface.types import canonical
from ora.agent.toolkit.state.deltas import bind_journal
from ora.agent.toolkit.state.snapshot import export_snapshot, file_ref
from ora.agent.toolkit.state.store import StateStore
from tests.support.agent.toolkit_atomic import timeline
from tests.support.agent.toolkit_tools import make_host


def make_source(root, monkeypatch):
    host, client, _ = make_host(root, monkeypatch)
    timeline(host)
    client.close()
    state = StateStore()
    journal = bind_journal(state, root)
    tool = state.access(writer="tool")
    environment = state.access(reader="supervisor", writer="environment")
    execution = state.access(reader="supervisor", writer="execution")
    with journal.boundary({"kind": "owner", "operation": "initialize"}):
        tool.write("tool.marks", {})
        tool.write("tool.held.left", None)
    events = []
    for step in range(2):
        name = f"policy-state/steps/step-{step:04d}.json"
        policy = json.loads((root / name).read_bytes())
        observed = policy["observation"]
        environment.update({
            "env.observation": dict(step=step, round_index=0, observation_id=observed["id"],
                                    observation_sha256=observed["manifest"]["sha256"],
                                    manifest=observed["manifest"],
                                    proprioception_file=file_ref(root, str(
                                        (root / observed["manifest"]["path"]).parent.relative_to(root)
                                        / "proprioception.json"))),
            "privileged.task_check": dict(terminal=False, task_success=False),
            **{f"env.camera.{camera}": value for camera, value in policy["cameras"].items()},
        })
        execution.write("exec.last_receipt", {"action_id": f"a-{step}", "native_actions": step})
        state.publication_completed()
        ref = {**file_ref(root, name, prefix="policy-state/steps/"), "step": step,
               "state_sha256": policy["state_sha256"]}
        export_snapshot(state, root, ref)
        events.append(dict(seq=step * 2, type="fresh_observation_published", run_id="run",
                           payload=dict(state_ref=ref, observation_id=observed["id"])))
        events.append(dict(seq=step * 2 + 1, type="tool_call_proposed", run_id="run",
                           model_call_id=f"m-{step}", tool_call_id=f"t-{step}",
                           payload=dict(name="mark_point", arguments={"fixture": step})))
        with journal.boundary(dict(kind="parent", run_id="run", round_index=0,
                                   model_call_id=f"m-{step}", tool_call_id=f"t-{step}")):
            tool.write("tool.marks", {"rim": [step, 2, 3]})
    (root / "agent-events.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in events))
    return state
