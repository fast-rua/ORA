# Connect a training loop

ORA runs a language model as a robot policy and keeps a full record of every
episode. A reinforcement-learning or fine-tuning loop needs nothing else from
ORA's side: it serves the policy it trains as a model endpoint, launches
episodes against it, and reads the verdict and the model calls back from the
evidence. ORA has no training code of its own.

`ora help rl-interface` prints a short version of this page.

## Serve the policy as an endpoint

ORA talks to a policy through a provider profile: an OpenAI-compatible (chat
completions) or Anthropic Messages endpoint, a model id and a key variable.
A checkpoint you are training is one more endpoint. HTTPS endpoints are
admitted, and so is plain HTTP on this machine's loopback address, for a
server such as vLLM or SGLang running next to ORA.

Add the endpoint as a profile of your own. `ora providers add` asks for the
key without echoing it and probes the endpoint with it:

```bash
./scripts/ora providers add my-policy --protocol openai_compatible \
    --endpoint http://127.0.0.1:8000/v1 --model my-checkpoint --images
```

It prints what it recorded, including the context window (131072 tokens
unless `--context-window` says otherwise). `--images` declares that the model
reads images, which the robot environments' observations carry;
`./scripts/ora providers test my-policy` probes the endpoint again later.
[Model APIs](../getting-started/model-apis.md#self-hosted-endpoints) lists
every option.

A new checkpoint is a new profile, or the same profile once the server serves
the new weights. The plan of every run records the profile id and endpoint,
never the key, so runs of different checkpoints stay apart.

## Run episodes against it

`ora run` prepares one task and opens it in the terminal UI; the profile
replaces the default model:

```bash
./scripts/ora run --env robotwin --task adjust_bottle --provider-profile my-policy
```

For unattended batches, compose RunSpecs with the environment's composer and
launch the path it prints, one RunSpec per task and seed (`ora spec robotwin
compose --help` lists the composer's inputs):

```bash
./scripts/ora spec robotwin compose adjust_bottle --seed SEED --qualification MANIFEST \
    --experiment my-training --resource-profile SLOT --service-report REPORT \
    --cohort round-1 --provider-profile my-policy
./scripts/ora launch outputs/my-training/specs/CONFIG.json --yes
```

Each launch holds its plan's execution-resource lease, so episodes on the
qualified simulator slots of one host profile run side by side and a busy
slot refuses a second launch. `ora suite run` queues one attempt per cell
across fixed lanes ([evaluation protocol](../evaluation/protocol.md)).
Launching still passes the normal preflight; see [run and inspect](runs.md).

## Identify each model call

Each episode's agent trace, `agent-events.jsonl` (`ora.agent-event.v1`),
records every model call as a `provider_request` and `provider_response`
pair. Every event carries `run_id`, `policy_session_id`, `observation_step`
and `model_call_id`, and is hash-chained to the previous one. The exact
request bytes are a content-addressed artifact that the request event names
by its sha256; after sealing they are packed losslessly into
`native/provider-requests.tar.zst`, and ORA's readers return the original
bytes either way.

The wire request carries no ORA identity by default. A catalog profile that
declares `session_affinity: x_session_id_run_v1` sends the header
`X-Session-ID: ora:<run_id>:native` with every policy request, so a server or
router can group requests by run.

## Read the reward and export the calls

The reward is the environment's official verdict, `official_success` in the
run's `result.json` (RoboTwin `check_success`, LIBERO termination, RoboDojo's
task verdict). A model's own claim of success never counts, and Synthetic and
free play record `null`. `evaluation.json` says whether the run may be used
for training (`learning_eligibility`).

Check the trace, then export the model calls of an evaluated run (a sealed
run's files are first checked against its seal):

```bash
./scripts/ora trace-validate RUN_ID
./scripts/ora export-sft RUN_ID --destination sft/RUN_ID
```

The destination directory holds:

| File | Contents |
| --- | --- |
| `conversations.openai-tools.jsonl` | One record per accepted model call: the request's messages and tools, then the model's reply |
| `attribution.jsonl` | What each reply decided |
| `loss-mask.jsonl` | Which message is the target (only the final reply) |
| `tokens.jsonl` | Each record's model call and its token usage as the provider reported it |
| `manifest.json` | The run id, the source commit, `official_success`, the termination, the sha256 of every exact request and the usage totals |

The export refuses a run whose trace does not verify, whose evaluation is
missing or invalid, or that is not learning-eligible: an incomplete episode, a
run with operator messages, free play, or a recorded runtime fault.

The usage in `tokens.jsonl` is exact for what the serving model processed in
each call. It is not the length of the exported record: the record leaves out
the earlier reasoning the call's history carried. Count a record's tokens with
the tokenizer and chat template you train with.

## Export every call as it was sent

The training records above are a projection. To keep a run's model calls
exactly as the provider received them, with every image sent to a model and
a proof that the export rebuilds them, use the wire view. It records learning
eligibility in its manifest instead of requiring it: `learning_eligible` is true
only for a sealed run whose evaluation is valid and found it eligible. Pass a
run id, or a run directory together with `--destination`:

```bash
./scripts/ora export-sft RUN_ID --view wire --destination wire/RUN_ID
```

| File | Contents |
| --- | --- |
| `requests.jsonl` | One line per model call, rejected replies included: the request body the provider received (OpenAI Chat Completions: messages with earlier reasoning, tools, sampling and template options), images replaced by `images/<sha256>.<ext>` references |
| `responses.jsonl` | ORA's record of each reply (accepted or rejected, its output rejection, usage) and `provider_reply`, the reply rebuilt from the archived stream: content, reasoning, tool calls including a truncated fragment, the provider's own finish reason |
| `streams.jsonl` | The provider's response exactly as archived (the streamed chunks) |
| `compactor.jsonl` | Each context compactor call: its request, ORA's record of the reply, the stream and the provider reply |
| `images/`, `images.jsonl` | Each image once, with its media type, size and the calls that sent it |
| `epochs.jsonl` | The messages once for each run of calls whose requests extend one another; each request line names its epoch, its prefix length and its last message when that differs (a rolling camera tail) |
| `tokens.jsonl` | Each policy call's prompt and sampled token ids where the run or the serving engine has them, checked against the call's usage ([token ids](#token-ids)) |
| `manifest.json` | Run identity, verdict, learning eligibility, counts and the replay check |

Before it moves the export into place, ORA rebuilds every request twice, from
its own line and from its epoch, puts the images back, serializes it the way
the run archived it, and requires the archived sha256. Compactor requests,
every stream, every provider reply and the image index (each file's sha256 and
the calls that sent it) are checked as well. A mismatch writes nothing; the
export directory gets the group permissions your umask allows. To check an
export again later: this needs the manifest and every file it lists, checks
their hashes, and derives every row and the manifest's description again from
the run, so an edit anywhere in the export is found:

```bash
./scripts/ora export-sft RUN_ID --view wire --destination wire/RUN_ID --check
```

It prints the counts and exits 2 when anything differs. Tool results, script
frames and receipts need no separate file: they are messages and images inside
the requests. Images the run recorded but no request carried (for example
script frames a model never saw) are counted in the manifest, not exported.

The export refuses a run whose seal no longer matches, or whose trace does not
verify. Two exceptions are labelled in `manifest.trace.status`: `incomplete`, a
run whose last request got no reply, exports every call and marks that request
`answered: false`; and
`sealed_valid_unverified_here`, a sealed run that its own evaluation found
valid but whose trace this checkout's verifier reads differently (an older or
newer trace shape), exports when every request still rebuilds to its sealed
sha256. For a provider that speaks another protocol (Anthropic
Messages), `request` is the canonical OpenAI-format request and
`wire_request` the body actually sent; both are checked.

### Token ids

A vLLM-served profile can ask for them on every call with
`request_options: {return_token_ids: true}`: each archived stream then carries
the prompt's ids and every sampled id, and `tokens.jsonl` reads both from it.
For any other run, give the export the serving engine's tokenize endpoint and
the variable that holds its key:

```bash
./scripts/ora export-sft RUN_ID --view wire --destination wire/RUN_ID \
  --tokenize-url https://HOST:PORT/v1/tokenize --tokenize-key-env SERVE_API_KEY
```

vLLM serves it at `<root>/tokenize`, sglang at `<root>/v1/tokenize`. ORA sends
each call's `messages`, `tools` and `chat_template_kwargs`, with its
`reasoning_effort` merged into the template kwargs as vLLM's chat endpoint
does: a tokenize request has no effort field, and the template would otherwise
render its default effort, a prompt of the same length with one token changed.
Use the engine that served the run (same model, tokenizer and template): every
call is checked against the usage it reported, so another one shows up as
length mismatches, but a same-length difference only shows where the run also
streamed its ids (`tokenize_equals_stream`). sglang
tokenizes before it expands images, so its ids hold one placeholder token per
image (`image_form: placeholder`, which the check counts as the usage's
`image_tokens`); vLLM's are expanded. Sampled ids exist only where the stream
carried them: sglang refuses token ids while streaming, so its replies stay a
count. Each prompt is stored as the prefix it shares with the previous call
plus its own suffix, with the sha256 of the whole sequence; `extends_previous`
says whether it starts with the previous prompt followed by that call's
sampled ids, which is where one epoch can train as one token sequence.
`--check` cannot ask the engine again: it rebuilds every row from the recorded
ids and the run's streams and usage.

## Share a run as an OpenAI package

To hand a run to someone who has neither ORA nor the run directory, export it
as a package of standard OpenAI Chat Completions data:

```bash
./scripts/ora export-sft RUN_ID --view openai --destination pkg/RUN_ID
./scripts/ora export-sft RUN_ID --view openai --destination pkg/RUN_ID --check
```

| Path | Contents |
| --- | --- |
| `conversations.jsonl` | One conversation per epoch, every image inline as a data URL, the epoch's final reply last: `{run_id, epoch, model_call_ids, messages, tools}` |
| `calls.jsonl` | One line per model call: the request as sent (images as `wire/images/...`), the reply, usage and token counts |
| `wire/` | The wire export above, with its token ids and replay check |
| `replay-program/` | The run's actions as code, when the run has one |
| `ora_package.py` | A standard-library tool: `verify`, `calls OUT` (one OpenAI sample per call, images inline), `images DIR`, `speculator OUT [--per-epoch]` (`{input_ids, loss_mask}` rows) |
| `README.md`, `package.json` | What the run was and how to use it; the sha256 of every file outside `wire/` |

`--tokenize-url` and `--tokenize-key-env` work as for the wire view. Whoever
receives the package checks it with `python3 ora_package.py verify`: every
request rebuilt to its archived sha256, every stream, image and token id, and
the conversations and calls derived again from `wire/`; with
`--tokenize-url URL --key-env NAME` a serving engine of the same model also
tokenizes every request again. `--check` here does the same and checks `wire/`
against the run itself. The messages keep two conventions of the serving stack:
each assistant turn's reasoning is in `reasoning_content`, and each tool call's
arguments are the JSON the engine's tool-call parser produced, which ORA sends
back in the history unchanged. vLLM's parser coerces values to the tool schema
(a number the model wrote as `0.0` or `-0.0` comes back as `0`), so where it
did, the sampled token ids, not the messages, are what the model wrote.

### A dataset of many runs

A package holds one run and repeats its history in several files. To publish or
train on many runs, export them together:

```bash
./scripts/ora export-traces --runs runs.jsonl --destination traces/v1 --workers 8 \
    --forbid my-cluster-name
```

`runs.jsonl` has one run directory per line, or one JSON object per line when
you want to name the episode or label it:

```json
{"run": "/data/outputs/ns/runs/RUN_ID", "episode_id": "robodojo-0001", "labels": {"within_step_limit": false}}
```

| Path | Contents |
| --- | --- |
| `episodes/episodes-NNNNN.jsonl.gz` | One record per episode: `model`, `benchmark`, `task`, `seed`, `instruction`, `mode`, `result`, `labels`, `tools`, `segments`, `usage` |
| `index.jsonl` | One line per episode with the same fields without the messages, its sizes, and its shard and line |
| `images/images-NNNNN.tar`, `images.jsonl` | Every image once across the dataset, in plain tar shards; each image's shard, offset and size |
| `ora_traces.py` | A standard-library tool: `verify [--images]`, `stats`, `openai EPISODE OUT` (one OpenAI conversation per segment, images inline), `calls EPISODE OUT` (one sample per model call), `images DIR [EPISODE]` |
| `README.md`, `dataset.json` | The data card; counts, shards and the sha256 of every file |

An episode's `segments` are, in order, the conversations the model saw: a new
one starts where ORA rewrote the context. A segment's `messages` are its last
request followed by that call's reply, with each image as a reference
`images/<sha256>.<ext>`; `calls` gives every earlier request of the segment as
`messages[:prefix]` (plus `tails[tail]` for a rolling camera message).

Every run goes through the wire export first, so each request is checked
against the run's sealed hashes and a privileged field refuses the run. Host
paths are replaced by placeholders (`--keep-host-paths` turns that off), the
run id is replaced by the episode id wherever the text carries it, and an
episode whose text holds a `--forbid` string is left out. Nothing in the
dataset names a run: `traces/v1.run-map.jsonl` maps episode ids to runs and
`traces/v1.report.json` lists the runs that were refused or failed, both beside
the dataset. Runs are exported into `--work` (default `traces/v1.work`) before
the dataset is assembled, and an interrupted export resumes from there. The
command exits 2 when a run was refused or failed.

`ora report RUN ...` summarizes stop reasons, validity, tokens and cost over
many runs. [Configuration and evidence](../reference/configuration-and-evidence.md#episode-evidence)
describes the rest of the run directory.
