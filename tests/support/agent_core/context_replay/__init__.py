"""Context-level replay of sealed Pi-core runs: what the Context manager received at every model call.

``extract.py`` turns one sealed run into a replay directory; the TypeScript
runner ``ora-pi/test/support/context_replay.ts`` rebuilds the context pipeline
and request assembler the way ``ora-pi/src/core/context_wiring.ts`` builds them
and steps through every recorded model call, reporting per call the canonical
request sha256 and size, the ``context_projected`` manifest and its projection
transitions, compared with what the run recorded.

Replay directory (schema ``ora.context-replay.v1``)::

    replay.json                    settings + one record per model call (below)
    blobs/<sha256><suffix>         every image the history or the tool-image
                                   registrations reference, stored once
    recorded-manifests.jsonl.gz    {"model_call_id", "manifest"}: the recorded
                                   ``context_projected`` payload, per call
    recorded-states.jsonl.gz       {"model_call_id", "context"}: the ``ora.state``
                                   context snapshot flushed just before the call's
                                   projection (pipeline, compactor, assembler)
    recorded-requests.jsonl.gz     {"model_call_id", "skeleton"}: the recorded
                                   canonical request with data URLs replaced by
                                   ``<data sha256 length>`` (optional, for diffs)

A call record holds what the context manager received for that call:
``history_append`` (the ``ora.history`` messages flushed since the previous
call, images as ``ora_image_ref`` blocks), ``history_length``,
``observation_step``, ``observation_index`` (the turn's observation message),
``tool_image_registrations`` (``registerToolImages`` calls since the previous
call), ``compactor`` (the compactor calls its projection made, with the
recorded responses), ``observed_usage`` (the Provider usage of its response,
fed to ``observeProviderUsage`` after the call) and ``recorded`` (the hashes
and transitions the run recorded).
"""

SCHEMA = "ora.context-replay.v1"
