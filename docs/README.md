# ORA documentation

<p class="ora-intro">Run a language model as a robot policy. ORA supplies the robot manual, observation tools, supervised actions, and a record of every episode.</p>

<p class="ora-meta">Open Robot-Use Agents · Python 3.11 · OpenAI-compatible and Anthropic APIs</p>

<div class="ora-next" markdown>
[Run your first episode](quickstart.md){ .md-button .md-button--primary }
[Run robot tasks on one GPU](getting-started/single-gpu.md){ .md-button }
[Choose a robot environment](environments/README.md){ .md-button }
</div>

## Start here

<div class="grid cards" markdown>

-   **Getting started**

    Install the locked Python environment and connect your model API. The first
    episode uses Synthetic: no simulator, model download, or GPU.

    [Installation →](getting-started/installation.md)

-   **One GPU, one command**

    On a Linux machine with one NVIDIA GPU and a DeepSeek key, `ora setup --env`
    installs LIBERO, RoboTwin or RoboDojo and `ora run --env … --task …` runs
    one task in the terminal UI.

    [LIBERO →](getting-started/libero.md) · [RoboTwin →](getting-started/robotwin.md) · [RoboDojo →](getting-started/robodojo.md) · [One-GPU guide →](getting-started/single-gpu.md)

-   **Robot environments**

    Set up LIBERO / LIBERO-Pro, RoboTwin, or RoboDojo in an isolated simulator
    environment, then compose and run one task.

    [Environment guides →](environments/README.md)

-   **Runs and evidence**

    Inspect the plan, model calls, executed actions, official result, and seal.
    Browse trajectories in the local read-only viewer.

    [Run and inspect →](usage/runs.md)

-   **Free play, remote and web**

    `ora play` gives the robot your instructions one at a time in a scene with
    no task, and the agent can script it through the supervisor. `ora-remote`
    and `ora web` bring a run on a GPU server to your terminal and browser.

    [Free play →](usage/play.md) · [Remote GPU server →](usage/remote.md) · [Web view →](usage/remote.md#web-view)

-   **Build on ORA**

    Read the architecture, the code map and the Python interfaces. Extend a
    backend, tool surface, or prompt bundle using the authoring guides.

    [Architecture →](architecture.md) · [Code map →](code-map.md) · [Python API →](api/README.md)

</div>

## First episode

Install ORA, store a model key and run one episode on a Synthetic robot:

```bash
curl -fsSL https://raw.githubusercontent.com/fast-rua/ORA/main/scripts/install.sh | sh
ora providers key DEEPSEEK_LOW_API_KEY
ora run --env synthetic --provider-profile deepseek_flash_openai_low_16k_v1
```

This calls a real model and saves a sealed episode under
`outputs/quickstart/runs/<run-id>/`. Synthetic verifies the agent/runtime
connection; its `official_success` is `null`.

[Complete quickstart and API configuration](quickstart.md)

## Terminal UI, free play and the web page

![ORA replaying a LIBERO episode in the terminal](assets/images/ora-watch.png)

`ora watch RUN_ID` follows a live run or replays a finished one as a terminal
transcript; `ora interactive --spec` and `ora local` add a task picker and let
you steer the running agent. Their pi-tui front end ships in `ora-pi/` with
the agent core, so the `npm ci --prefix ora-pi` above installs it. `ora interactive` without `--spec`
shows a run in the Pi agent core's own terminal UI, which you can detach and
re-attach (`ora attach`); it needs nothing beyond the agent core above.
See [run and inspect](usage/runs.md),
[interactive sessions](usage/interactive.md) and
[run ORA on your computer](getting-started/local.md).

```bash
./scripts/ora play --env synthetic        # free play: a scene, no task, your instructions in turn
./scripts/ora web setup                   # once: build ORA's web page
./scripts/ora web RUN_ID                  # a live run's conversation and cameras in the browser
./scripts/ora replay RUN_ID               # a finished run on the same page at 16x (--export out.mp4: a video)
```

In [free play](usage/play.md) the agent is Pi as a coding agent with the robot
as its device: besides the robot tools it can script the robot in Pi's
sandboxed codemode or with the `ora-robot` command, and the supervisor still
executes every action. On a GPU server without a display, `ora-remote` runs
the command over SSH and shows the terminal UI on your computer and the
cameras, or ORA's [web page](usage/remote.md#web-view), in your browser
([remote GPU server](usage/remote.md)).

## Choose an environment

| Environment | Robot / simulator | Guide |
|---|---|---|
| LIBERO / LIBERO-Pro | Panda manipulation in MuJoCo | [Install and run](environments/libero.md) |
| RoboTwin | Dual-arm manipulation in SAPIEN | [Install and run](environments/robotwin.md) |
| RoboDojo | Isaac Sim task environments | [Install and run](environments/robodojo.md) |

The simulators, assets, perception weights, and GPU requirements are installed
separately from ORA. [Supported setups](support.md) records the tested scope.

## Find what you need

| I want to… | Read |
|---|---|
| Use a self-hosted model or Anthropic | [Model APIs](getting-started/model-apis.md) |
| Store a key or add my own endpoint (`ora providers`) | [Keys in the terminal](getting-started/model-apis.md#keys-configured-once-in-the-terminal) |
| Give the robot instructions myself, or script it | [Free play](usage/play.md) · [Scripting the robot](usage/play.md#scripting-the-robot) |
| Bring up, calibrate and recover a real robot cell (`ora cell`) | [Run a real robot cell](usage/real-cells.md) |
| Run on a GPU server from my laptop, or watch in the browser | [Remote GPU server](usage/remote.md) · [Web view](usage/remote.md#web-view) |
| Fix an installation or run problem | [Troubleshooting](getting-started/troubleshooting.md) |
| Inspect a run or export evidence | [User guide](usage/runs.md) |
| Understand the action and trust boundaries | [Runtime lifecycle](reference/runtime-lifecycle.md) |
| Configure a benchmark task | [Configuration](reference/configuration-and-evidence.md) |
| Run the official suites on a cluster | [Official suites](getting-started/official-suite.md) · [Evaluation protocol](evaluation/protocol.md) |
| Train a model on ORA's episodes | [Connect a training loop](usage/rl-interface.md) |
| Find where something lives in the source | [Architecture](architecture.md) · [Code map](code-map.md) · [Repository structure](developer/repository-structure.md) |
| Change tools or robot instructions | [Tool surfaces](developer/authoring/tool-surface.md) · [Prompt bundles](developer/authoring/prompt-bundle.md) |
| Add a simulator | [Add an environment](developer/add-backend.md) · [Environment compatibility](developer/authoring/environment-compatibility.md) |
| Contribute or maintain these docs | [Start contributing](developer/onboarding.md) · [Contributing](../CONTRIBUTING.md) · [Documentation guide](developer/documentation.md) |
