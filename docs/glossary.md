# Glossary

| Term | Meaning |
|---|---|
| ORA | Open Robot-Use Agents; the Python import and CLI are `ora`, the distribution is `ora-agent`. |
| `native_api` policy | The policy kind (`policy.kind: native_api`) in which ORA itself calls the model API (OpenAI-compatible or Anthropic) and supplies the prompts, tools and context. The other kind, `scripted`, runs fixed actions for infrastructure checks. "Native" in names such as `native_full_explicit_motion_direct_v4` refers to this policy kind, not to an agent core: `native_api` is only the name of the policy kind. The run's `native/` evidence directory, the `native_slots_*` compositions and the `ora.native-*` schemas keep the word for the same reason. |
| Agent core | The implementation that runs a `native_api` policy's agent loop: provider calls, request assembly, context management and agent-side evidence. ORA has one, Pi, and every plan records it in `policy.agent_core` ([agent core](reference/configuration-and-evidence.md#agent-core)). |
| Pi | ORA's agent core: TypeScript in `ora-pi/` on the pinned Pi agent packages, run by Node.js. Its terminal UI is the one `ora interactive`, `ora attach` and `ora play` open; in free play the agent is Pi itself, working as a coding agent. |
| Native core | The Python agent core ORA started with (`ora.agent.native`). Removed from ORA; it remains only in the robot-harness repository. A plan or setting that names it (`policy.agent_core: native`, `ORA_AGENT_CORE=native`, `--agent-core native`) is refused; its sealed runs still evaluate, open in the viewer and replay. |
| RunSpec | Portable task, provider, runtime and budget inputs. |
| RunPlan | The resolved, content-addressed execution contract, including source and asset identities. |
| Baseline | A maintained composition selected by the default resolver. |
| ToolSurface | The advertised tools, schemas, capabilities and state-effect contract. |
| Prompt bundle | Ordered project-owned policy layers used with an embodiment and environment. |
| Embodiment | Public robot geometry, frames, units and control semantics. |
| Supervisor | Trusted authority that checks and dispatches robot mutations. |
| Receipt | Measured execution feedback returned by the environment/controller. |
| Qualification | Evidence that a specific infrastructure dependency satisfies a stated contract. |
| Official verdict | The benchmark's success boundary; never the model's self-assessment. |
| Seal | Hash-bound immutable inventory of evaluator-valid run evidence. |
| Context artifact | Recoverable episode-local tool/media evidence, not cross-episode Task Memory. |
| Free play | `ora play`: a qualified scene with no task. The operator gives instructions one at a time; the session is sealed but never scored ([free play](usage/play.md)). |
| Practice | The plan partition `partition: practice` (the default is `evaluation`): development episodes that may not use any benchmark's evaluation episodes. Practice runs are never evaluation results ([practice and evaluation partitions](evaluation/protocol.md#practice-and-evaluation-partitions)). |
| Operator view | A camera image for the human watching a run, such as a play scene's third-person camera. It is recorded as evidence and shown in `ora web`, never given to the policy ([operator view](reference/configuration-and-evidence.md#operator-view)). |
| Replay | Watching a finished run again from its sealed evidence: `ora watch` in the terminal, or `ora replay` on `ora web`'s page or as an exported video ([replay a finished run](usage/remote.md#replay-a-finished-run)). A replay reads the evidence and never runs the model or the simulator. |
| Experiment family, generation | Optional lineage labels for a study: a family such as `EXP90A` names the question, and a generation `FAMILY/branch/gNNN` names one treatment ([experiment identity](reference/experiment-identity.md)). Example cohort names such as `libero-example-g001` borrow the `gNNN` suffix. |
