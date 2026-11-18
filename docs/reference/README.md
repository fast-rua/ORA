# Current system reference

ORA supports Native model API policies and scripted infrastructure checks. Its robot integrations are RoboTwin, LIBERO/LIBERO-Pro, RoboDojo, RoboCasa365 and RoboLab; Synthetic provides simulator-free runtime checks.

`native_api` policies run on the Pi agent core (`policy.agent_core: pi`, TypeScript in `ora-pi/`, Node.js >= 22.19), ORA's only agent core; the Python Native core was removed (it remains in the robot-harness repository). See [agent core](configuration-and-evidence.md#agent-core).

- [Runtime lifecycle and authority](runtime-lifecycle.md)
- [Public interfaces](public-interfaces.md)
- [Toolkit I/O evidence](toolkit-io.md)
- [Toolkit state and admission](toolkit-state.md)
- [File-backed toolkit surfaces](toolkit-surface.md)
- [Context schemes and policy selection](context-schemes.md)
- [Append-only context (management v8)](append-only-context.md)
- [Exchange aging (management v9)](exchange-aging-context.md)
- [Keep what the model wrote (management v10)](keep-written-context.md)
- [Task files](tasks.md)
- [Source overlays](source-overlay.md)
- [Configuration and evidence](configuration-and-evidence.md)
- [Context packs](context-packs.md)
- [Experiment identity](experiment-identity.md)
- [RoboTwin contact recovery](robotwin-contact-recovery.md)
- [RoboTwin seed and scene qualification](robotwin-qualification.md)
- [Motion speed and wrist checks](motion-speed.md)
- [Latency profile](latency-profile.md): `sim_benchmark` or `realtime`, and the switches each resolves to
- [RoboDojo motion qualification](robodojo-motion-qualification.md)
- [RoboCasa365 and RoboLab backends](robocasa-robolab-backends.md)
- [RoboLab tasks](robolab-tasks.md): the RoboLab-120 roster, its generated contracts and their witness

The selected default robot baseline is `native_full_explicit_motion_direct_v4`. It binds the current direct Panda and ALOHA closed-skill compositions, and the direct PandaOmron (RoboCasa365, with `move_base`) and Robotiq-TCP (RoboLab) compositions, with Native context. RoboDojo defaults to a baseline of its own, `robodojo_toolkit_v7`: the file-backed ARX toolkit surface `robodojo_placement_toolkit_v6` with the keep-written context and no official check (`baseline.ENVIRONMENT_BASELINES`, 2026-11-02), in code mode since 2026-11-05 (extended library, prompt v3, ORA's source; `"code_mode": null` keeps tool calls; [code mode by default](context-schemes.md#code-mode-by-default)), so a RoboDojo host needs bubblewrap, unprivileged user namespaces and the sandbox interpreter `ora setup --env robodojo` builds; `native_full_explicit_motion_direct_v4`'s ARX closed-skill instance was retired on 2026-11-01. `ora baselines --explain` shows the actual resolved tools and bindings. New benchmark specifications select the baseline by task/provider/resource inputs; they do not copy an older tool surface.

The built-in provider default is `deepseek_flash_openai_low_16k_v1`. The simulator-free `ora run --env synthetic` demonstration has its own small, explicit provider configuration described in the [quickstart](../quickstart.md), or runs on a registered profile with `--provider-profile ID`.

Public namespaces are `ora.*`, including schemas and `ORA_*` runtime variables. External simulator variables keep their benchmark names. Runs from earlier internal schema namespaces must be read at their recorded source revision; this initial distribution makes no implicit evidence migration.

When changing an API, schema, default, context policy, runtime lifecycle, trust boundary or evidence format, update its page here in the same change. The [code map](../code-map.md) identifies implementation owners.

- [Toolkit categorized data API](toolkit-data.md): state ownership, robot/RSI
  grants, nine categories and snapshot-only capability rules.

- [Synchronous toolkit execution](toolkit-execution.md): budgets, grants, receipts and recovery.
