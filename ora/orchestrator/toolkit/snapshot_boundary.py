"""Select completed state evidence, without legacy fallback or simulator queries."""

from dataclasses import dataclass, field
from pathlib import Path

from ora.agent.evidence.agent_events import library_op
from ora.agent.toolkit.state.deltas import before_call, unsettled_publication
from ora.agent.toolkit.state.keys import StateError
from ora.agent.toolkit.state.snapshot import load_publications, load_snapshot
from ora.agent.toolkit.state.store import StateStore
from ora.agent.toolkit.interface.types import CallContext, CorruptRecord, NotRecorded, decode
from .selection import rows, read_local


@dataclass
class SnapshotBoundary:
    source: Path
    step: int
    state: StateStore
    context: CallContext
    selected: dict | None
    publications: list
    snapshot_sha256: str
    missing_state_fields: list = field(default_factory=list)
    # The parent publication's policy-state reference: the runtime's current state at a --at boundary.
    policy_state: dict | None = None


def load(source, step, *, publications=False):
    try:
        return load_publications(source, step) if publications else load_snapshot(source, step)
    except FileNotFoundError as error:
        raise NotRecorded(f"未记录: complete snapshot chain at step {step}") from error
    except (StateError, ValueError, KeyError, TypeError) as error:
        raise CorruptRecord(str(error)) from error


def select_snapshot(source, *, step=None, at=None, raw_tool_state=False):
    """--step is the complete publication, --at includes preceding tool deltas."""
    source = Path(source).resolve(strict=True)
    if (step is None) == (at is None):
        raise ValueError("choose exactly one snapshot step or parent boundary")
    selected = None
    ref = None
    missing = []
    if at is not None:
        events = rows(source, "agent-events.jsonl")
        seqs = [row["seq"] for row in events]
        if any(type(seq) is not int or seq < 0 for seq in seqs) or any(a >= b for a, b in zip(seqs, seqs[1:])):
            raise CorruptRecord("agent event sequence must be strictly increasing")
        proposals = [row for row in events if row["type"] == "tool_call_proposed" and not library_op(row)]
        matches = [(index, row) for index, row in enumerate(proposals)
                   if str(index) == at or row["tool_call_id"] == at]
        if len(matches) != 1:
            raise NotRecorded("未记录: absent or ambiguous snapshot parent; use its ordinal")
        index, proposal = matches[0]
        published = [row for row in events if row["type"] == "fresh_observation_published"
                     and row["seq"] < proposal["seq"]]
        if not published:
            raise NotRecorded("未记录: no publication before snapshot parent")
        publication = max(published, key=lambda row: row["seq"])
        if any(row["type"] in {"action_proposed", "environment_receipt", "round_started"}
               for row in events if publication["seq"] < row["seq"] < proposal["seq"]):
            raise NotRecorded("selected call is inside an unpublished mutation/reset boundary")
        ref = publication["payload"].get("state_ref")
        if ref is None:
            raise NotRecorded("未记录: parent publication has no policy-state reference")
        if not ref.get("path", "").startswith("policy-state/steps/"):
            raise CorruptRecord("parent publication has an unexpected policy-state path")
        from .selection import reference
        actual = reference(source, ref["path"])
        if actual["sha256"] != ref["sha256"]:
            raise CorruptRecord("parent publication policy-state reference drift")
        policy = decode(read_local(source, ref["path"]))
        step = policy["step"]
        selected = dict(call=index, tool_call_id=proposal["tool_call_id"],
                        tool=proposal["payload"]["name"], proposal=proposal)
    if type(step) is not int or step < 0:
        raise ValueError("snapshot step must be a nonnegative integer")
    chain = load(source, step, publications=True)  # one validation of the chain for every step up to here
    loaded = chain[-1]
    state = loaded.store
    if selected is not None:
        try:
            state = before_call(state, source, model_call_id=proposal["model_call_id"],
                                tool_call_id=proposal["tool_call_id"])
        except StateError as error:
            if str(error).startswith("未记录"):
                raise NotRecorded(str(error)) from error
            raise CorruptRecord(str(error)) from error
    else:
        overlay = StateStore()
        overlay._values = dict(state._values)
        overlay.complete_boundary = True
        overlay.tool_watermark = state.tool_watermark
        state = overlay
        if not raw_tool_state:
            try:
                reason = unsettled_publication(state, source)
            except StateError as error:
                raise CorruptRecord(str(error)) from error
            if reason:
                state._values = {key: value for key, value in state._values.items()
                                 if not key.startswith("tool.")}
                missing.append(dict(field="tool.*", reason=reason))
    document = decode(read_local(source, f"state/steps/step-{step:04d}/snapshot.json"))
    observed = state.access(reader="rsi").read("env.observation")
    identity = {key: observed[key] for key in ("observation_id", "observation_sha256")}
    if selected is not None:
        if (document["run_id"] != proposal["run_id"]
                or identity["observation_id"] != publication["payload"]["observation_id"]):
            raise CorruptRecord("snapshot differs from parent publication")
        boundary = dict(kind="before_call", call=selected["call"], **{
            key: proposal[key] for key in ("seq", "tool_call_id", "model_call_id")})
        tool_call_id = proposal["tool_call_id"]
    else:
        boundary = dict(kind="snapshot_step", step=step)
        tool_call_id = f"snapshot-step-{step}"
    context = CallContext.create(run_id=document["run_id"], round_index=observed["round_index"],
                                 tool_call_id=tool_call_id, boundary=boundary,
                                 state={"observation": identity}, check_task=True)
    # Earlier snapshots grant retained observations only, never their future tool values.
    publications = chain
    publications = [item for item in publications if
                    item.access(reader="rsi").read("env.observation")["round_index"] == observed["round_index"]]
    return SnapshotBoundary(source, step, state, context, selected, publications, document["snapshot_sha256"], missing,
                            policy_state=None if ref is None else dict(ref))
