"""Replay assembly of boundary-frozen public tool inputs; never discover later artifacts."""
from ora.agent.toolkit.interface.tool_data import ToolData
from ora.agent.toolkit.interface.types import CorruptRecord, NotRecorded, canonical,decode,digest
from .selection import read_local,rows


def attach(api,source,output,selected,bindings):
    """Load the one exact input-index occurrence; a run without one recorded no frozen inputs."""
    path='native/toolkit-io/tool-inputs.jsonl'
    records=[row for row in rows(source,path) if row.get('tool_call_id')==selected['tool_call_id']]
    if len(records)>1:raise CorruptRecord('ambiguous parent tool input index')
    if records:
        record=records[0]
        if record.get('schema')!='ora.toolkit.tool-inputs.v1' or record.get('context')!=api.context.json():
            raise CorruptRecord('tool input index boundary differs from replay context')
        for item in record['inputs'].values():
            for ref in item.get('files',{}).values():api.store.register(ref)
        data=canonical(record)+b'\n'
        (output/'source-tool-inputs.json').write_bytes(data)
        api.tool_data=ToolData(store=api.evaluation.store,configuration=record['configuration'],inputs=record['inputs'],replay=True)
        return dict(mode='recorded',path='source-tool-inputs.json',sha256=digest(data),bytes=len(data))
    raise NotRecorded('未记录: this run recorded no boundary-frozen tool inputs')
