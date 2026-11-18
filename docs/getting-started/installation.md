# Install ORA

## One-line install

On Linux or macOS, with Git and curl:

```bash
curl -fsSL https://raw.githubusercontent.com/fast-rua/ORA/main/scripts/install.sh | sh
```

`scripts/install.sh` needs no sudo. It installs [uv](https://docs.astral.sh/uv/) into
`~/.local/bin` and Node.js 22.20 into `~/.local/share/ora/node` when they are missing
(the Node.js download is checked against its published SHA-256), checks ORA out into
`~/.local/share/ora/ORA`, runs the two install commands of the next section there, writes
an `ora` command into `~/.local/bin` and ends with `ora doctor`. That command
exports `ORA_COMMAND=ora`, so the hints ORA prints name `ora`; from a source
checkout they name its wrapper, `./scripts/ora`. ORA stays a Git checkout,
so every run still binds the exact commit it ran. Rerun the command to update; it refuses
a checkout with local changes. Options: `sh -s -- --ref TAG` installs a release tag, branch
or commit, `--dir`, `--bin-dir` and `--repo` change where it installs and from where, and
`--no-doctor` skips the check. Then install a simulator with `ora setup --env <env>`
([one GPU](single-gpu.md)). The rest of this page is the same installation by hand.

## By hand

ORA's host environment needs Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
Python 3.11 and Node.js 22.19 or newer with npm. uv can install the pinned
Python version. Start with a source checkout so launches can record an exact
Git commit.

```bash
git clone https://github.com/fast-rua/ORA.git
cd ORA
uv sync --all-extras --locked
npm ci --prefix ora-pi --prefer-offline --no-audit --no-fund
uv run --locked ora doctor
```

`--all-extras` also installs the media extra (`imageio`, used by RoboTwin
qualification captures) and the test runner; the same command is used everywhere
in these docs.

`ora doctor` reports Python and ORA versions, the agent core's Node version
and `ora-pi` packages, credential presence, and optional graphics/media
dependencies. It does not print keys, authenticate a provider, or start a
simulator. Missing simulator dependencies are expected at this step.

## Node.js and the agent core

Model API policies run on the Pi agent core by default: TypeScript in
`ora-pi/` on the exactly pinned Pi packages, which Node.js 22.19 or newer runs
directly, with no build step. Without sudo, unpack Node into `~/.local`
(macOS: `darwin-arm64`):

```bash
mkdir -p ~/.local && curl -fsSL https://nodejs.org/dist/v22.20.0/node-v22.20.0-linux-x64.tar.xz \
  | tar -xJ -C ~/.local --strip-components=1
export PATH=~/.local/bin:$PATH
npm ci --prefix ora-pi --prefer-offline --no-audit --no-fund
```

ORA uses `ORA_NODE` when it is set, else the first Node.js 22.19 or newer among
`node` on `PATH`, `~/.local/bin/node` and nvm installs (newest first), so a
distribution Node 18 earlier on `PATH` does not block it. `ora setup` (both the
CPU and `--env` forms) runs the same `npm ci` and leaves `ora-pi/node_modules/`
alone when every package already matches `ora-pi/package-lock.json`. On a host
without registry access, `npm ci --prefer-offline` succeeds only from a
populated npm cache: run it once where the registry is reachable with the same
cache, or point `npm_config_cache` at one.

Preflight refuses a plan whose Node or packages are missing, naming the fix.
Every `native_api` policy runs on the Pi core, so a model run needs them; see
[agent core](../reference/configuration-and-evidence.md#agent-core).

## Requirements by step

| Step | Model API | GPU | Simulator / assets |
|---|---|---|---|
| Install and run `ora doctor` | Not required | Not required | Not required |
| Synthetic first episode | Required | Not required | Not required |
| Robot episode | Required, with the selected observation capabilities | See environment guide | Required |
| Build these docs | Not required | Not required | Not required |

Synthetic uses a real model API. Cost and latency depend on the endpoint; a
self-hosted service can use the same interface.

## Terminal UI (optional)

`ora watch`, `ora interactive` and `ora local` draw with a small Node front end
(pi-tui) that ships with the agent core in `ora-pi/src/watch/`. The agent
core's `npm ci --prefix ora-pi` above installs everything it needs; there is
no separate step.

Without those packages, `ora watch` uses a built-in text view; the interactive
commands need the front end (`ora setup` installs it for `ora local`).

`ora interactive` without `--spec`, `ora play`, `ora attach` and
`ora sessions` use the Pi agent core's own terminal UI instead, and need only
the agent core above; see [interactive sessions](../usage/interactive.md).

ORA's web page for a live run (`ora web`) is built once with
`./scripts/ora web setup` (`npm ci` and a build in `ora-pi/web/`); see
[web view](../usage/remote.md#web-view).

## Pick your machine

| Your machine | What runs | Start with | Verified on |
|---|---|---|---|
| Linux with one NVIDIA GPU, 32 GB or more (Turing to Blackwell) | LIBERO / LIBERO-Pro, RoboTwin and RoboDojo, full tool set | `ora setup --env <env>`, then `ora run --env <env> --task <task> --confirm-launch` ([one GPU](single-gpu.md)) | RTX PRO 6000 Blackwell (all three, end to end); H200 (through launch checks) |
| Multi-GPU cluster | Official suites at benchmark scale, many runs in parallel | [Benchmark setup and official suites](official-suite.md) | 8× H200 nodes |
| Any computer, no GPU | Synthetic, and LIBERO on the CPU | `ora run --env synthetic`, or `ora setup` then `ora local --confirm-launch` ([on your computer](local.md)) | |

## Install with an AI coding agent

If you use Claude Code, Codex, Cursor or another coding agent, give it this prompt:

```text
Install ORA (https://github.com/fast-rua/ORA) and run one robot task.

1. Read docs/getting-started/installation.md and the guide for the
   environment you will run (docs/getting-started/robotwin.md here).
2. Detect the machine: run `uname -m`, `nvidia-smi` and `nvcc --version`.
   With an NVIDIA GPU of 32 GB or more, follow docs/getting-started/single-gpu.md;
   otherwise follow docs/getting-started/local.md.
3. Follow the guide exactly. On a GPU: `./scripts/ora setup --env robotwin`, then
   `./scripts/ora run --env robotwin --task click_bell --confirm-launch`.
   I will provide DEEPSEEK_LOW_API_KEY in the shell; never print, store or commit it.
4. If a step fails, find the matching row in the guide's troubleshooting table,
   fix it, and tell me what you changed.
5. Report the run ID and its official_success.
```

## Next step

[Run the first episode](../quickstart.md), then run robot tasks
[on one GPU](single-gpu.md) or choose a [robot environment](../environments/README.md).
To give a robot instructions yourself, try [free play](../usage/play.md)
(`ora play --env synthetic` needs no simulator); to drive a GPU server from
your own computer, see [remote use](../usage/remote.md).

## Developer installation

```bash
uv sync --all-extras --locked
npm ci --prefix ora-pi --prefer-offline --no-audit --no-fund
uv run --locked python -m pytest tests
```

The full suite needs system FFmpeg for video tests. See
[contributing](../../CONTRIBUTING.md). Preview these docs with
`./scripts/docs serve` (Node.js 22.19 or later); documentation dependencies are
isolated from the runtime.
