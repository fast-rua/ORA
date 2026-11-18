# Start contributing

Your first hour with the ORA source: install it, run the checks, find your way
around, run one episode without a simulator and one with, and learn the three
rules every change keeps. [Contributing](../../CONTRIBUTING.md) is the full path
to a pull request. The same page in Chinese follows the English one.

## English

### 1. Clone, install, check

```bash
git clone https://github.com/fast-rua/ORA.git && cd ORA
uv sync --all-extras --locked        # Python 3.11 from uv.lock
scripts/dev/check --quick            # the Python suite and the repository checks
```

`--quick` installs the agent core's Node packages (Node.js 22.19 or newer) and
needs no model key, simulator or GPU. `scripts/dev/check` without it runs
everything CI runs; [Testing](testing.md) says which check a change needs and
how the test tree mirrors the code.

### 2. Find your way around

The [code map](../code-map.md) has one row per package and a link to its
README; each README says what belongs in the package and what may import it.
The [repository structure](repository-structure.md) states the layers and the
import rules, and [how ORA works](../how-ora-works.md) follows one run through
them. In short: `ora/app/` is the `ora` command, `ora/orchestrator/` turns a
RunSpec into a plan and runs and seals the episode, `ora/agent/` is the policy
side (tools, composition, context, providers), `ora/environment/` the simulator
side, `ora/contracts/` what both share, `ora/policy/` the prompts and tool
surfaces, `ora-pi/` the agent core in TypeScript, and `tests/ora/` mirrors
`ora/`.

### 3. One episode without a simulator

```bash
export ORA_API_KEY=...                       # an OpenAI key; other APIs: Model APIs
uv run --locked ora run --env synthetic      # prints the run directory, named by the run ID
./scripts/ora watch RUN_ID                   # replay it in the terminal
```

Without a key, `uv run --locked python -m examples.quickstart_smoke --keep
outputs/smoke` runs the same command against a local fake model, as CI does,
and prints the `./scripts/ora watch` command for its run. With a registered
provider profile's key instead, add `--provider-profile ID` to `ora run`. See
the [first episode](../quickstart.md) and [model APIs](../getting-started/model-apis.md).

### 4. One episode on a simulator

With an NVIDIA GPU of 32 GB or more:

```bash
./scripts/ora setup --env libero
./scripts/ora run --env libero --task libero_goal/0 --confirm-launch
```

Without one, `./scripts/ora setup` and `./scripts/ora local --confirm-launch`
run a LIBERO task on the CPU. See [LIBERO](../getting-started/libero.md), the
[one-GPU guide](../getting-started/single-gpu.md) and
[run ORA on your computer](../getting-started/local.md).

### 5. Three rules you must not break

1. **The policy sees only public information, and what it sees changes only on
   purpose.** No ground truth, segmentation, contacts or success predicate
   reaches it. After a refactor,
   `scripts/dev/compare-model-visible <base> HEAD` reports no difference; any
   other change explains each difference it reports.
2. **Evidence and pinned files stay valid.** Evidence formats change only on
   purpose and runs sealed earlier still evaluate. The files that
   [may not move](repository-structure.md#12-files-that-may-not-move) keep
   their path and bytes, and a change under `ora/policy/` is re-pinned with
   `./scripts/ora lock` in the same commit.
3. **The structure holds.** Layers import downward only, tests mirror the code
   and never import one another, every package keeps its README, and a moved
   module leaves no compatibility shim. `tests/repository/test_structure.py`
   checks it.

### 6. Where training research goes

This repository holds the harness, not training code. Reinforcement-learning
and fine-tuning work lives in its own repository and uses ORA through
[the training-loop interface](../usage/rl-interface.md): a provider endpoint
for the model being trained, the agent trace that identifies each call, and
`ora export-sft` for the accepted calls.

## 中文

### 1. 克隆、安装、检查

```bash
git clone https://github.com/fast-rua/ORA.git && cd ORA
uv sync --all-extras --locked        # 按 uv.lock 装 Python 3.11 环境
scripts/dev/check --quick            # Python 测试和仓库检查
```

`--quick` 会装好 agent core 的 Node 依赖（Node.js 22.19 及以上），不需要模型
key、模拟器或 GPU。不加 `--quick` 就跑 CI 的全部检查；改动该跑哪些检查、测试树怎样
与代码对应，见[测试](testing.md)。

### 2. 代码在哪

先看[代码地图](../code-map.md)：每个包一行，链到它的 README，README 写明这个包
放什么、不放什么、谁可以 import 它。分层和 import 规则见
[仓库结构](repository-structure.md)，一次运行怎么穿过各层见
[ORA 如何工作](../how-ora-works.md)。简单说：`ora/app/` 是 `ora` 命令，
`ora/orchestrator/` 把 RunSpec 变成计划、跑完并封存一个 episode，`ora/agent/`
是策略一侧（工具、组装、上下文、模型接口），`ora/environment/` 是模拟器一侧，
`ora/contracts/` 是两边共用的契约，`ora/policy/` 是提示词和工具面，`ora-pi/`
是 TypeScript 写的 agent core，`tests/ora/` 与 `ora/` 一一对应。

### 3. 不用模拟器跑一个 episode

```bash
export ORA_API_KEY=...                       # OpenAI 的 key；其他接口见「模型 API」
uv run --locked ora run --env synthetic      # 打印运行目录，目录名就是 run ID
./scripts/ora watch RUN_ID                   # 在终端里回放
```

没有 key 时，`uv run --locked python -m examples.quickstart_smoke --keep outputs/smoke`
用本地假模型跑同一条命令（CI 就这么跑），并打印回放这次运行的 `./scripts/ora watch`
命令。有已登记的 provider profile 的 key 时，给 `ora run` 加上 `--provider-profile ID`。
详见[第一个 episode](../quickstart.md) 和[模型 API](../getting-started/model-apis.md)。

### 4. 在模拟器上跑一个 episode

有 32 GB 以上的 NVIDIA GPU：

```bash
./scripts/ora setup --env libero
./scripts/ora run --env libero --task libero_goal/0 --confirm-launch
```

没有 GPU 时，`./scripts/ora setup` 加 `./scripts/ora local --confirm-launch` 在
CPU 上跑一个 LIBERO 任务。详见 [LIBERO](../getting-started/libero.md)、
[单卡指南](../getting-started/single-gpu.md)和[在本机运行](../getting-started/local.md)。

### 5. 不能破的三条规则

1. **策略只看到公开信息，它看到的内容只能有意改变。** 真值、分割、接触、成功判据
   都不能给它。纯重构之后 `scripts/dev/compare-model-visible <base> HEAD` 必须
   没有差异；其他改动要逐条解释差异。
2. **证据和钉住的文件保持有效。** 证据格式只能有意改变，之前封存的运行仍要能评估。
   [不能移动的文件](repository-structure.md#12-files-that-may-not-move)路径和字节
   都不变；改了 `ora/policy/` 下的文件，同一个提交里用 `./scripts/ora lock` 重新钉住。
3. **结构不能破。** 只能自上而下 import，测试与代码一一对应、测试之间不互相
   import，每个包都有 README，搬走的模块不留兼容垫片。
   `tests/repository/test_structure.py` 检查这些。

### 6. 训练相关的研究放哪

这个仓库只放 harness，不放训练代码。强化学习和微调在各自的仓库里做，通过
[训练循环接口](../usage/rl-interface.md)使用 ORA：被训练的模型作为一个 provider
端点，agent trace 标识每次调用，`ora export-sft` 导出被接受的调用。
