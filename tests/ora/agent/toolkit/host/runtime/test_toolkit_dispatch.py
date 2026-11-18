"""The opt-in handlers preserve coordinator results with either recording switch."""
import json
from tests.support.agent.toolkit_tools import install
from ora.agent.toolkit.host.recording import install as record
from tests.support.agent.toolkit_tools import make_host
from tests.support.agent.toolkit_recording import proposal

def test_recreated_dispatch_does_not_refuse_an_already_recorded_parent(tmp_path,monkeypatch):
    from ora.agent.toolkit.host.runtime.dispatch import Dispatch
    host,client,_=make_host(tmp_path/'run',monkeypatch)
    try:
        args=dict(camera='head',text='object',max_candidates=1)
        proposal(host,'vision_ground',args)
        first=Dispatch(host).run('vision_ground',args)
        second=Dispatch(host).run('vision_ground',args)
        assert first==second
        gaps=[json.loads(line) for line in (tmp_path/'run/native/toolkit-io/gaps.jsonl').read_bytes().splitlines()]
        assert any(row['stage']=='context' and row['tool_call_id']=='call-0' for row in gaps)
    finally:client.close()


def test_uncertain_append_freezes_both_dispatch_paths(tmp_path, monkeypatch):
    from ora.agent.toolkit.interface.journal import Journal

    host, client, _ = make_host(tmp_path / "run", monkeypatch)
    try:
        record(host)
        install(host)
        append = Journal.append

        def uncertain_append(writer, event):
            value = append(writer, event)
            if event["event"] == "end":
                # The bytes reached disk, but the caller cannot know that.
                raise OSError("injected uncertain completion")
            return value

        monkeypatch.setattr(Journal, "append", uncertain_append)
        args = dict(camera="head", text="object", max_candidates=1)
        first = host.invoke_with_context(
            "vision_ground", args, call_context=proposal(host, "vision_ground", args)
        )
        assert first["candidates"]
        monkeypatch.setattr(Journal, "append", append)
        root = tmp_path / "run/native/toolkit-io"
        recorded = (root / "calls.jsonl").read_bytes()
        for index, name, arguments in (
            (1, "mark_point", dict(point_ids=["P1"], view="head", points_uv=[[300, 195]])),
            (2, "vision_ground", dict(camera="head", text="another object", max_candidates=1)),
        ):
            result = host.invoke_with_context(
                name, arguments, call_context=proposal(host, name, arguments, index=index)
            )
            assert result
        assert (root / "calls.jsonl").read_bytes() == recorded
        gaps = [json.loads(line) for line in (root / "gaps.jsonl").read_bytes().splitlines()]
        assert {row["tool_call_id"] for row in gaps} == {"call-0", "call-1", "call-2"}
        assert any(row["stage"] == "end" for row in gaps)
        assert all(row["stage"] == "begin" for row in gaps if row["tool_call_id"] != "call-0")
    finally:
        client.close()
