"""Append-only event indexing preserves exact identities without rereading history."""
import pytest

from ora.agent.toolkit.interface.types import AccessDenied, canonical
from ora.agent.toolkit.host.event_index import EventIndex


def event(index):
    return dict(type='tool_call_proposed', seq=index, run_id='run',
                tool_call_id=f'call-{index}', model_call_id='model',
                payload=dict(name='check_task', arguments={}))


def test_incremental_append_reads_each_complete_line_once(tmp_path):
    path = tmp_path/'events.jsonl'
    path.touch()
    index = EventIndex(path)
    completed = set()
    for i in range(70):
        with path.open('ab') as stream:
            stream.write(canonical(event(i)) + b'\n')
        selected = index.select('check_task', {}, completed)
        assert selected['tool_call_id'] == f'call-{i}' and selected['call'] == i
        completed.add(selected['tool_call_id'])
        before = index.bytes_read
        index.refresh()
        assert index.bytes_read == before
    assert index.bytes_read == path.stat().st_size
    assert index.lines_parsed == 70
    assert 'arguments' not in index.proposals[0]


def test_partial_tail_and_publication_order(tmp_path):
    path = tmp_path/'events.jsonl'
    publication = dict(type='fresh_observation_published', seq=0, payload={'observation_id':'obs'})
    head = canonical(publication) + b'\n'
    tail = canonical(event(1))
    path.write_bytes(head + tail[:20])
    index = EventIndex(path)
    index.refresh()
    assert index.offset == len(head) and index.lines_parsed == 1
    with path.open('ab') as stream:
        stream.write(tail[20:] + b'\n')
    assert index.select('check_task', {}, set())['tool_call_id'] == 'call-1'
    assert index.publication_seqs == [0]
    assert index.lines_parsed == 2


@pytest.mark.parametrize('change', ['truncate','replace','duplicate','finished'])
def test_changed_or_ambiguous_parent_is_rejected(tmp_path, change):
    path = tmp_path/'events.jsonl'
    row = canonical(event(1)) + b'\n'
    path.write_bytes(row)
    index = EventIndex(path)
    index.refresh()
    if change == 'truncate':
        path.write_bytes(b'')
    elif change == 'replace':
        replacement = tmp_path/'replacement'
        replacement.write_bytes(row)
        replacement.replace(path)
    else:
        with path.open('ab') as stream:
            stream.write(row if change == 'duplicate' else canonical(
                dict(type='tool_result', tool_call_id='call-1')) + b'\n')
    with pytest.raises(AccessDenied):
        index.select('check_task', {'tool_call_id':'call-1'}, set())
