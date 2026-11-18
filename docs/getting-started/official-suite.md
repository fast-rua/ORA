# Benchmark setup and official suites

Start with a [single Synthetic episode](../quickstart.md), then install one robot environment:

- [LIBERO / LIBERO-Pro](../environments/libero.md)
- [RoboTwin](../environments/robotwin.md)
- [RoboDojo](../environments/robodojo.md)
- [Shared host, planner and perception setup](../environments/common.md)

Each page describes the isolated interpreter, pinned sources/assets, discovery checks and a one-task run. Run one task end to end before you run a suite. RoboDojo's default policy is code mode, so a RoboDojo host also needs bubblewrap, unprivileged user namespaces and the sandbox interpreter ([code-mode sandbox](../environments/robodojo.md#code-mode-sandbox)); the LIBERO-Pro and RoboTwin defaults use tool calls and need none of them.

[examples/official_suite](../../examples/official_suite/README.md) expands the LIBERO-Pro and RoboTwin suites from the current default composition into cells, one episode on one seed or initial state each. Official seed selection, task/source versions, episode budget and environment verdict stay explicit in every cell. `./scripts/ora suite --help` runs composed cells on the simulator slots of your host profile that you name, one cell per slot at a time, and does not launch on a slot that another run holds.

An agent that acts through robot tools is not comparable one-to-one with a trained VLA policy. When you report results, state its observation and action interface, the task and seed protocol, the attempts and the budget. A passing installation or qualification check shows that the setup works; it does not measure the model or establish benchmark success.
