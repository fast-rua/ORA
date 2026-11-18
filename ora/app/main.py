"""``ora``: one root parser, one command table, the grouped help and the dispatch.

Every command belongs to one ``ora.app`` module named in ``COMMANDS``. The module
exposes ``register(subparsers)``, which adds its commands to the root parser, and
``run(args)``, which carries one out and imports what it needs inside. ``GROUPS``
orders the commands for ``ora --help``. The app modules import nothing heavy at
module level, so ``ora --help`` and a read-only command such as ``ora status``
never load the evaluation or the episode coordinator
(``tests/ora/app/test_import_budget.py``).

    ora <command> --help      one command's options
    ora help [COMMAND ...]    the same, or this overview
    ora help topics           the topic pages (``ora/app/help/<topic>.md``, shipped with ORA)
    ora help TOPIC            one topic page (``ora help play`` is the page; ``ora play --help`` the options)
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .passthrough import Parser

#: Every ``ora`` command and the ``"module:register"`` that adds it. A module that
#: adds several commands is named once per command and registered once.
COMMANDS: dict[str, str] = {
    "setup": "ora.app.setup:register",
    "doctor": "ora.app.doctor:register",
    "run": "ora.app.run:register",
    "play": "ora.app.play:register",
    "replay": "ora.app.replay:register",
    "web": "ora.app.web:register",
    "cameras": "ora.app.cameras:register",
    "interactive": "ora.app.interactive:register",
    "attach": "ora.app.interactive:register",
    "sessions": "ora.app.interactive:register",
    "local": "ora.app.interactive:register",
    "plan": "ora.app.plan:register",
    "launch": "ora.app.runs:register",
    "task": "ora.app.runs:register",
    "retry": "ora.app.runs:register",
    "status": "ora.app.runs:register",
    "stop": "ora.app.runs:register",
    "watch": "ora.app.runs:register",
    "workspace": "ora.app.runs:register",
    "evaluate": "ora.app.evidence:register",
    "seal": "ora.app.evidence:register",
    "revalidate": "ora.app.evidence:register",
    "export": "ora.app.evidence:register",
    "export-sft": "ora.app.evidence:register",
    "export-traces": "ora.app.evidence:register",
    "report": "ora.app.report:register",
    "metadata": "ora.app.evidence:register",
    "state": "ora.app.evidence:register",
    "metrics": "ora.app.evidence:register",
    "ownership": "ora.app.evidence:register",
    "trace-validate": "ora.app.evidence:register",
    "spec": "ora.app.spec:register",
    "qualify": "ora.app.qualify:register",
    "suite": "ora.app.suite:register",
    "providers": "ora.app.providers:register",
    "services": "ora.app.services:register",
    "cell": "ora.app.cell:register",
    "baselines": "ora.app.catalog:register",
    "lock": "ora.app.catalog:register",
    "dev": "ora.app.dev:register",
    "toolkit": "ora.app.toolkit:register",
    "help": "ora.app.main:register",
    "experiments": "ora.app.catalog:register",
}

#: ``ora --help``, group by group: (command, English line, Chinese line). ``experiments``
#: (a development view of the experiment registry) runs but is not listed.
GROUPS: tuple[tuple[tuple[str, str], tuple[tuple[str, str, str], ...]], ...] = (
    (("Run an episode", "跑一局"), (
        ("setup", "install the local environment or one simulator", "安装本地环境或某个模拟器"),
        ("doctor", "check the host, credentials, simulators and services", "检查主机、凭证、模拟器、服务"),
        ("run", "compose, preflight, launch and seal one episode", "组合计划、预检、启动并封存一局"),
        ("play", "free play in one scene", "自由试玩一个场景"),
        ("replay", "replay a finished run in the browser, export a video, or check its replay program",
         "回放、导出一局，或校验它的重放程序"),
        ("web", "a live run in the browser", "在浏览器里看一局进行中的运行"),
        ("cameras", "a run's cameras in the terminal, beside its conversation", "在终端里看一局的相机画面"),
        ("interactive", "run ORA in Pi's terminal UI and steer it", "在 Pi 的终端界面里运行并引导 ORA"),
        ("attach", "show a running run's terminal UI here", "在这个终端显示一局运行的界面"),
        ("sessions", "list runs with Pi sessions, or export one", "列出带 Pi 会话的运行，或导出一个"),
        ("local", "the local CPU MuJoCo task picker", "本地 CPU MuJoCo 任务选择"),
    )),
    (("Formal runs and evidence", "正式运行与证据"), (
        ("plan", "validate, show, preview or preflight one RunSpec", "校验、显示、预检一个 RunSpec"),
        ("launch", "preflight and launch one confirmed run", "预检并启动一局确认过的运行"),
        ("retry", "retry an unsealed, action-free incomplete run", "重试一局未封存、未发出动作的运行"),
        ("status", "one run, or every run on this workspace", "一局或全部运行的状态"),
        ("stop", "stop one active run", "停止一局正在进行的运行"),
        ("watch", "follow a live run, or replay a finished one, as a transcript", "跟随或回放一局的记录"),
        ("workspace", "the shared evidence and state roots", "证据与状态的根目录"),
        ("task", "resolve, run or submit a task file (ora.task.v1)", "解析、运行或提交一个任务文件"),
        ("evaluate", "official verdict and evidence checks of one run", "官方判定与证据检查"),
        ("seal", "make an evaluated run immutable", "封存一局评估过的运行"),
        ("revalidate", "review completed runs into separate reports", "复核已完成的运行，另写报告"),
        ("export", "archive one sealed run for transfer", "归档一局封存的运行"),
        ("export-sft", "export a run's model calls as tool-call JSONL", "把一局的模型调用导出为 JSONL"),
        ("export-traces", "export many runs as one trace dataset", "把多局运行导出为一个轨迹数据集"),
        ("report", "stop-reason and cost report of runs", "停止原因与成本报告"),
        ("metadata", "one run's context and environment manifest", "一局的上下文与环境清单"),
        ("state", "one run's public policy state", "一局的公开 policy 状态"),
        ("metrics", "one run's Runtime metrics", "一局的运行指标"),
        ("ownership", "one run's behavioral-ownership audit", "一局的行为归属审计"),
        ("trace-validate", "one run's agent trace and artifact graph", "校验一局的 agent 轨迹与产物图"),
    )),
    (("Environment qualification and suites", "环境资格与套件"), (
        ("spec", "compose tasks and RunSpecs per environment", "按环境组合任务与计划"),
        ("qualify", "environment, lane and seed qualification", "环境/车道/种子资格"),
        ("suite", "the official-suite queue", "官方套件队列"),
        ("play audit", "audit free-play scenes", "试玩场景审计"),
    )),
    (("Configuration", "配置"), (
        ("providers", "model API keys and profiles", "模型 API 的 key 与 profile"),
        ("services", "the resident perception services", "常驻感知服务"),
        ("cell", "set up, watch and recover a real robot cell", "真机 cell 的设置、观察与恢复"),
        ("baselines", "the default composition and how it resolves", "默认组合及其解析"),
        ("lock", "policy.lock, the hash of every policy asset", "policy.lock：每个 policy 资产的哈希"),
    )),
    (("Developer", "开发者"), (
        ("toolkit", "try toolkit code at a recorded tool call", "在历史工具调用边界试运行工具代码"),
        ("dev", "authoring tools (embodiment, prompt bundle, task contract, CLI reference)",
         "作者工具（embodiment、prompt bundle、task contract、CLI 参考）"),
        ("help", "ora help COMMAND, or a topic: runs, evidence, play, remote, safety, rl-interface",
         "ora help <topic>：runs、evidence、play、remote、safety、rl-interface"),
    )),
)

#: The commands ``ora --help`` lists, in its order; ``INTERNAL_COMMANDS`` run but are not listed.
PUBLIC_COMMANDS = tuple(name for _, commands in GROUPS for name, *_ in commands if name in COMMANDS)
INTERNAL_COMMANDS = ("experiments",)

TITLE = "ora — Open Robot-Use Agents"
FOOTER = {
    "en": "ora <command> --help shows one command; ora help topics lists the topic pages.",
    "zh": "ora <command> --help 看每个命令；ora help topics 列出主题页。",
}
#: Commands that use no provider key: they run before the user's stored keys are applied, and ``ora setup``
#: runs before the dependencies the credential store needs are installed.
NO_CREDENTIALS = frozenset({"help", "setup", "toolkit"})
#: ``ora help TOPIC``: short pages shipped with ORA (package data), each distilled from
#: the documentation page it names at its end. ``ora help topics`` lists them in this order.
TOPICS = ("runs", "evidence", "play", "remote", "safety", "rl-interface")
TOPIC_DIR = Path(__file__).resolve().parent / "help"


def language(environ: Mapping[str, str] | None = None) -> str:
    """``zh`` or ``en``: ``ORA_UI_LANG``, else a ``zh*`` locale reads Chinese (as ora-pi/src/tui/i18n.ts)."""

    values = os.environ if environ is None else environ
    chosen = str(values.get("ORA_UI_LANG", "")).strip().lower()
    if chosen.startswith(("zh", "en")):
        return chosen[:2]
    locale = values.get("LC_ALL") or values.get("LC_MESSAGES") or values.get("LANG") or ""
    return "zh" if locale.lower().startswith("zh") else "en"


def overview(lang: str | None = None) -> str:
    """The grouped ``ora --help``: one line per command."""

    lang = lang or language()
    index = 1 if lang == "zh" else 0
    width = max(len(name) for _, commands in GROUPS for name, *_ in commands)
    lines = [TITLE]
    for titles, commands in GROUPS:
        lines += ["", titles[index]]
        lines += [f"  {name:<{width}}  {texts[index]}" for name, *texts in commands]
    return "\n".join([*lines, "", FOOTER[lang]]) + "\n"


class _Root(Parser):
    """The root parser: its help is the grouped overview."""

    def format_help(self) -> str:
        return overview()


class _Version(argparse.Action):
    """``--version``: print ``ora <version>`` and exit; the version is read only when asked."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest=dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        from ora.version import ora_version

        print(f"{parser.prog} {ora_version()}")
        parser.exit()


def _target(command: str) -> tuple[str, str]:
    module, _, attribute = COMMANDS[command].partition(":")
    return module, attribute


def build_parser() -> argparse.ArgumentParser:
    """The root parser with every command's module registered."""

    parser = _Root(prog="ora", description=TITLE)
    parser.add_argument("--version", action=_Version, help="print the ORA version and exit")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>", parser_class=Parser)
    parser.usage = "ora <command> [options]"  # after add_subparsers, which names each command's prog "ora NAME"
    for target in dict.fromkeys(COMMANDS.values()):
        module, _, attribute = target.partition(":")
        getattr(importlib.import_module(module), attribute)(sub)
    missing = sorted(set(COMMANDS) - set(sub.choices))
    unlisted = sorted(set(sub.choices) - set(COMMANDS))
    ungrouped = sorted(set(COMMANDS) - set(PUBLIC_COMMANDS) - set(INTERNAL_COMMANDS))
    if missing or unlisted or ungrouped:
        raise RuntimeError(f"ora's command table, its modules and its help disagree: missing {missing}, "
                           f"unlisted {unlisted}, not in GROUPS {ungrouped}")
    return parser


def register(sub: Any) -> None:
    command = sub.add_parser("help", help="ora help COMMAND, or ora help TOPIC",
                             description="Print the overview, one command's help, or a topic page "
                                         "(ora help topics lists them). A topic comes before the command "
                                         "of the same name: ora help play is the page, ora play --help "
                                         "the command's options.")
    command.add_argument("subject", nargs="*", metavar="COMMAND|TOPIC|topics",
                         help="a command (and its subcommands), a topic page, or `topics` to list them")


def topic_page(name: str) -> str:
    """The text of one topic page."""

    return (TOPIC_DIR / f"{name}.md").read_text(encoding="utf-8")


def topic_titles() -> dict[str, str]:
    """Each topic and its page's title (the first line, without its ``#``)."""

    return {name: topic_page(name).split("\n", 1)[0].lstrip("# ").strip() for name in TOPICS}


def run(args: argparse.Namespace) -> int:
    subject = list(args.subject)
    if not subject:
        print(overview(), end="")
        return 0
    if subject == ["topics"]:
        titles = topic_titles()
        width = max(map(len, titles))
        print("Topic pages (ora help TOPIC):")
        print("\n".join(f"  {name:<{width}}  {title}" for name, title in titles.items()))
        return 0
    if len(subject) == 1 and subject[0] in TOPICS:
        # Before a command of the same name: `ora help play` is the page, `ora play --help` the options.
        print(topic_page(subject[0]), end="")
        return 0
    if subject[0] in COMMANDS:
        # Prints that command's help and exits (`ora help help` too); a tool's leaf hands --help to the tool.
        parsed = build_parser().parse_args([*subject, "--help"])
        module, _ = _target(parsed.command)
        return int(importlib.import_module(module).run(parsed) or 0)
    print(f"ora help: {' '.join(subject)} is neither a command nor a topic page (ora help topics)", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        # The same ora-remote command again attaches to the run it left going (remote_session).
        from .remote_session import resume_arguments

        values = resume_arguments(values) or values
        args = build_parser().parse_args(values)
        if args.command not in NO_CREDENTIALS:
            # The user's stored provider keys, where the environment sets none (ora providers, /login).
            from ora.agent.providers.user_store import apply_stored_credentials

            apply_stored_credentials()
        module, _ = _target(args.command)
        return int(importlib.import_module(module).run(args) or 0)
    except ModuleNotFoundError as exc:
        if (exc.name or "ora").split(".")[0] == "ora":
            raise
        from ora.version import ora_command  # here, not at import: the entry point loads almost nothing

        print(f"ora: dependency {exc.name!r} is unavailable; run {ora_command()} setup first", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError, OSError) as exc:  # RunSpecError, EvaluationError, PolicyStateError, ...
        print(f"ora: {exc}", file=sys.stderr)
        return 2


__all__ = ["COMMANDS", "GROUPS", "INTERNAL_COMMANDS", "PUBLIC_COMMANDS", "TOPICS", "build_parser", "language",
           "main", "overview", "register", "run", "topic_page", "topic_titles"]
