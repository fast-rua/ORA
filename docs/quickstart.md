# First episode

One episode on a Synthetic robot: a real model, the real agent loop, no simulator, no model download and no GPU. You need Linux or macOS with Git, curl and tar, and a model API key.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/fast-rua/ORA/main/scripts/install.sh | sh
```

The installer needs no sudo. It installs uv and Node.js 22 when they are missing, checks ORA out into `~/.local/share/ora/ORA`, puts an `ora` command in `~/.local/bin` and ends with `ora doctor` ([what it does, and the same installation by hand](getting-started/installation.md)). If your shell then says `ora: command not found`, add that directory to your `PATH`, as the installer says when it is not there:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Working from a source checkout instead? Write `./scripts/ora` (or `uv run --locked ora`) wherever this page says `ora`; the [installation page](getting-started/installation.md#by-hand) has the two commands that prepare a checkout.

## Model API

Store a key once. It is asked without echo and kept in your user-level ORA config (`~/.config/ora/pi-auth.json`, mode 0600), never in a run:

```bash
ora providers key DEEPSEEK_LOW_API_KEY
```

That is the key of `deepseek_flash_openai_low_16k_v1`, the built-in default profile: DeepSeek's public API. `ora providers` lists every profile and where each key comes from; a profile brings its endpoint, model and key variable ([model APIs](getting-started/model-apis.md#keys-configured-once-in-the-terminal)).

To use an endpoint without a profile, set it in your shell and leave `--provider-profile` out of the run below. For the default OpenAI endpoint, set `ORA_API_KEY` or `OPENAI_API_KEY`; the default model is [gpt-4.1-mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), which accepts image inputs and tool calls. For another OpenAI-compatible API, set these non-secret values and supply its key as `ORA_API_KEY`:

```bash
export ORA_BASE_URL=https://your-model-server.example/v1
export ORA_MODEL=your-served-model-id
export ORA_PROVIDER=openai_compatible
```

A local HTTP server is accepted only on loopback. Remote endpoints require HTTPS. For a private certificate authority, set `SSL_CERT_FILE` to a trust bundle containing that CA. Do not disable certificate verification.

For Anthropic Messages, set `ORA_API_KEY` or `ANTHROPIC_API_KEY`, plus:

```bash
export ORA_PROVIDER=anthropic_messages
export ORA_BASE_URL=https://api.anthropic.com
export ORA_MODEL=your-available-model-id
```

`ORA_REASONING_EFFORT` optionally selects the provider's reasoning setting. A text-only endpoint can use `ORA_SUPPORTS_IMAGES=0` for Synthetic; the robot benchmarks need the observation capabilities their selected surface declares. Keys are read from the environment or the key store and never stored in RunSpecs.

## Run

```bash
ora run --env synthetic --provider-profile deepseek_flash_openai_low_16k_v1
```

The model reads a robot manual and moves a Synthetic end effector once. The normal resolver, preflight, resource lease, supervisor, evaluator and seal are used, on the Pi agent core. Preflight probes the provider before the episode. Running this command explicitly starts the bounded demonstration; it does not require a separate confirmation prompt.

The episode has a 120-second budget and at most six model calls; with a profile, the profile's own output limit applies, otherwise 2,048 output tokens per call. Actual latency and input-token use depend on your provider and the tool calls it chooses. The command ends with the run's ID, its directory and its outcome, and exits successfully only after a completed episode has been sealed:

```text
watch: ora watch quickstart-m0-…
episode: …/outputs/quickstart/runs/quickstart-m0-…
state: completed; model calls: 2; official_success: null (Synthetic has no benchmark verdict)
evidence: evaluated and sealed
```

The episode directory, under the ORA checkout's `outputs/quickstart/runs/`, holds `plan.json`, `result.json`, `seal.json`, canonical events and archived provider evidence. `official_success: null` is expected: Synthetic tests infrastructure and has no benchmark success predicate.

## Look at it, then go on

`ora watch RUN_ID` replays the episode in the terminal, with the model's words and each tool call ([terminal UI](getting-started/installation.md#terminal-ui-optional)); `ora web RUN_ID` shows it in the browser once `ora web setup` has built the page. From a source checkout, `uv run --locked ora-viewer --open-browser` opens the viewer, which discovers the saved run.

To steer the agent yourself, `ora interactive` runs a longer Synthetic episode in Pi's terminal UI ([interactive sessions](usage/interactive.md)), and `ora play --env synthetic` starts [free play](usage/play.md): no task, your instructions one at a time. Both offer the provider profiles from `ora providers`, or `--provider env` for the variables above.

For a robot task, [choose a robot environment](support.md): on a Linux machine with one NVIDIA GPU of 32 GB or more, `ora setup --env robodojo` installs a simulator and `ora run --env robodojo --task general_pickup --confirm-launch` runs a task ([one-GPU guide](getting-started/single-gpu.md)).

Synthetic reserves 1 GiB of free workspace and 0.25 GiB of scratch by default; explicit host storage settings take precedence. No simulator disk reserve is required.

## Troubleshooting

| Symptom | Check |
|---|---|
| `ora: command not found` | Add `~/.local/bin` to your `PATH` (above), or open a new shell after adding it to your shell's startup file. |
| Missing key / 401 | Store or export the key expected by the chosen provider. `ora doctor` reports presence only; it does not authenticate. |
| 429 or provider unavailable | Check the provider's capacity and billing. The run remains bounded; inspect its `supervisor.log`. |
| Source is dirty | In a source checkout, commit intentional source changes: launches bind the exact source commit. Keep credentials and generated files ignored. |
| Context overflow | Use a model with the declared window or reduce the task/context in a separate RunSpec. Without a profile this quickstart declares 32,768 tokens; a profile declares its own window. |
| Episode appears stalled | Inspect the printed run directory and `ora status`; `ora stop <run-id>` stops that exact run. |
| No GPU, EGL or simulator | None is needed here. For robot episodes, run `ora doctor --env <environment>` and follow that environment's installation page. |

Without a key, a source checkout can run the same CLI against a deterministic local HTTP model, as CI does; it spends no API credits:

```bash
uv run --locked python -m examples.quickstart_smoke --keep outputs/smoke
```

`--keep DIR` keeps the run in the workspace `DIR` and prints its run ID and the `./scripts/ora watch` command that replays it; without `--keep` the run is removed afterwards (CI). Run it with `uv run` from the checkout: a Python without ORA's environment stops at once and says so. This is separate from acceptance against a real model API.
