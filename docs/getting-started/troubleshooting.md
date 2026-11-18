# Troubleshooting

Start with discovery and the exact run ID:

```bash
uv run --locked ora doctor
uv run --locked ora status
```

For robot dependencies, use `ora doctor --env robotwin`, `--env libero_pro`, or
`--env robodojo`. Discovery does not start a simulator or authenticate a provider.

## Installation

| Symptom | Check |
|---|---|
| `uv` is missing | Install uv and ensure it is on `PATH`. |
| Wrong Python or missing package | Run `uv sync --all-extras --locked` from the source checkout, then use `uv run --locked`. |
| Dirty-source refusal | Review and commit intentional changes; keep keys and outputs ignored. |
| Missing FFmpeg | Install system FFmpeg, including `ffprobe`, for video export and tests. Synthetic itself does not need it. |
| `the Pi agent core cannot start` (preflight `agent_core`, `ora doctor`) | Install Node.js 22.19 or newer and run `npm ci --prefix ora-pi --prefer-offline --no-audit --no-fund` ([agent core install](installation.md#nodejs-and-the-agent-core)). |
| `... names the Python Native agent core, which was removed from ORA` or `this plan was resolved for the Python Native agent core` | A RunSpec, `ORA_AGENT_CORE`, `--agent-core` or a persisted plan names the retired Native core: drop the setting (Pi is the only agent core) or resolve the RunSpec again; the sealed runs of that core still evaluate, open in the viewer and replay. |

## Model API

| Symptom | Check |
|---|---|
| Missing key / 401 | Export the key for the selected protocol and check the endpoint's credential configuration. |
| 429 / unavailable | Check capacity and quota; inspect the bounded run's `supervisor.log`. |
| TLS failure | Check the hostname and trust chain; set `SSL_CERT_FILE` for a private CA. |
| Unknown model | Use the model identifier actually served by the endpoint. |
| Images or tools refused | Compare the selected surface's capabilities with the endpoint and its qualification report. |
| Context overflow | Check the model's actual window and declared limit; inspect saved request/context evidence. |

## Simulator

| Symptom | Check |
|---|---|
| Import fails | Use the simulator's isolated interpreter, not ORA's locked host environment. |
| Missing assets or weights | Follow the environment and perception installation guides, including upstream access requirements. |
| EGL / Vulkan / GPU failure | Check the driver, device assignment, and headless graphics requirements. |
| Revision or seed refused | Use the pinned source and source-bound qualification expected by the plan. |

Use the [environment guides](../environments/README.md) for the exact setup.

## A run appears stalled

Inspect the printed run directory with `ora status RUN_ID` or `ora watch RUN_ID`.
Use `ora stop RUN_ID` to stop that exact active run. Preserve the original files
and seal; later revalidation writes a separate report.

For a bug report, include doctor output, source commit, run ID, refusal code,
and a redacted log excerpt. Never include keys. The
[command guide](../usage/commands.md) lists inspection/export commands.
