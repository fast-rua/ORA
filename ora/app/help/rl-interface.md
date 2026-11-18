# Connect a training loop

ORA needs no reinforcement-learning code of its own. A trainer uses three
things ORA already has: a replaceable model endpoint, model calls that the
evidence identifies, and exports of what the policy saw and did.

## 1. The policy is an endpoint

Serve the policy being trained as an OpenAI-compatible (or Anthropic
Messages) endpoint and give it a provider profile. HTTPS endpoints, and plain
HTTP on this machine's loopback, are admitted.

    ora providers add my-policy --protocol openai_compatible \
        --endpoint http://127.0.0.1:8000/v1 --model my-checkpoint --images
    ora providers test my-policy
    ora run --env robotwin --task TASK --provider-profile my-policy
    ora spec robotwin compose TASK ... --provider-profile my-policy

A new checkpoint is a new profile, or the same profile at a new endpoint. A
plan records the profile id and endpoint, never the key.

## 2. Every model call is identified

The agent trace (`agent-events.jsonl`) records each call as a
provider_request and provider_response pair carrying the run_id,
policy_session_id, observation_step and model_call_id. The exact request
bytes are a content-addressed artifact the request event names by sha256. A
catalog profile that declares session_affinity x_session_id_run_v1 also
sends the run on the wire, as the header `X-Session-ID: ora:<run_id>:native`.

## 3. Rewards and exports

    result.json            official_success: the environment's own verdict
    evaluation.json        learning_eligibility: whether the run may train
    ora export-sft RUN_ID  manifest.json, conversations.openai-tools.jsonl,
                           attribution.jsonl, loss-mask.jsonl and tokens.jsonl
    ora trace-validate RUN_ID   check the trace before you rely on it
    ora report RUN ...     stop reasons, tokens and cost across runs

The reward is the official verdict; a model's own claim never counts.
`export-sft` writes one record per accepted model call (the request's
messages and tools, and the reply). It refuses a run without a valid
evaluation, and a run that is not learning-eligible: incomplete,
human-assisted, free play or a runtime fault. tokens.jsonl holds each
call's usage as the provider reported it; the record itself is shorter when
the call's history carried earlier reasoning.

    ora export-sft RUN_ID --view wire          every call exactly as sent, each image
                                               once, rebuilt and checked against the archive
    ora export-sft RUN_ID --view wire --check  check an existing wire export again
    ora export-sft RUN_ID --view openai        a package for sharing: OpenAI conversations,
                                               calls, the wire export and ora_package.py
    ... --view wire --tokenize-url URL --tokenize-key-env NAME
                                               prompt token ids from the serving engine
                                               (sampled ids come from streams only)

## Many runs at once

Each launch holds its plan's execution-resource lease; a host profile's
qualified simulator slots run side by side. `ora suite run` queues one
attempt per cell across fixed lanes (`ora suite --help`).

More: docs/usage/rl-interface.md.
