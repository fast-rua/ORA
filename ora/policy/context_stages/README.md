# Context stage parameter schemas

One closed schema per context stage (`<stage>.schema.json`, the stages of
`ora-pi/src/context/stages/`) and `scheme.schema.json` for the settings of an
`ora.context-scheme.v2` scheme (`segments`, `limits`, `estimator`,
`references`). They use the JSON-Schema subset both languages enforce
(`ora/agent/composition/surface/schema_validation.py`,
`ora-pi/src/context/memory/schema_validation.ts`); `default` marks what a scheme
may leave out.

Python validates v2 schemes with them (`ora/agent/composition/schemas/context_stages.py`)
and sends them to the agent core at session start (`domain.context_stage_schemas`);
the core validates every stage it builds with them and keeps no copy of the
rules. A new parameter, or a new value of one, is a change here and in its
stage; see [context schemes](../../../docs/reference/context-schemes.md).
