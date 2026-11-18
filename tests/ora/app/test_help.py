"""``ora``'s help, at the parser: the grouped overview, every command's one-line help,
``ora help COMMAND|TOPIC`` and the topic pages shipped in ``ora/app/help/``.

Every command's ``--help`` runs in process here, and so does every subcommand's
and every backend tool's (``ora spec|qualify KIND TOOL --help`` and ``ora play
audit --env KIND --help``); one subprocess checks that the repository wrapper
prints the overview.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import tomllib

import pytest

from ora.app import main as app_main
from ora.app.main import COMMANDS, GROUPS, INTERNAL_COMMANDS, PUBLIC_COMMANDS, TOPICS, build_parser
from ora.orchestrator.backends import registry
from tests.support.orchestrator.control_plane import ROOT


def _commands(parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()):
    """Every command and subcommand under ``parser``: (path, its one-line help, its parser)."""

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            helps = {choice.dest: choice.help for choice in action._choices_actions}
            for name, child in action.choices.items():
                yield (*prefix, name), helps.get(name), child
                yield from _commands(child, (*prefix, name))


def test_every_command_and_subcommand_has_a_one_line_help() -> None:
    found = {path: text for path, text, _ in _commands(build_parser())}
    assert {path[0] for path in found} == set(COMMANDS)
    for path, text in found.items():
        if path[0] in INTERNAL_COMMANDS:
            assert text is None, path  # it runs, but neither ora --help nor its parser lists it
            continue
        assert isinstance(text, str) and text.strip() == text and text and "\n" not in text, path
    # The nested ones are walked too: plan's, providers', dev's tools and play audit.
    assert {("plan", "show"), ("providers", "add"), ("dev", "cli-reference"), ("play", "audit")} <= set(found)
    # Each name in ora --help is a command or subcommand of the parser.
    for _, commands in GROUPS:
        for name, *_ in commands:
            assert tuple(name.split()) in found, name


def test_every_group_is_named_and_lists_commands_in_both_languages() -> None:
    assert len(GROUPS) == 5
    for titles, commands in GROUPS:
        assert len(titles) == 2 and all(title.strip() for title in titles), titles
        assert commands, titles
        for name, english, chinese in commands:
            assert english.strip() and chinese.strip() and "\n" not in english + chinese, name


def test_grouped_help_lists_every_public_command_once(capsys) -> None:
    listed = [name for _, commands in GROUPS for name, *_ in commands]
    assert [name for name in listed if name in COMMANDS] == list(PUBLIC_COMMANDS)
    assert len(set(listed)) == len(listed) and "play audit" in listed
    assert "experiments" not in listed
    for lang, headings in (("en", ("Run an episode", "Formal runs and evidence", "Environment qualification "
                                    "and suites", "Configuration", "Developer")),
                           ("zh", ("跑一局", "正式运行与证据", "环境资格与套件", "配置", "开发者"))):
        text = app_main.overview(lang)
        lines = text.splitlines()
        assert lines[0] == "ora — Open Robot-Use Agents"
        assert [line for line in lines if line and not line.startswith(" ")][1:-1] == list(headings)
        for name in listed:  # one line per command
            assert sum(1 for line in lines if line.startswith(f"  {name}  ")) == 1, (lang, name)
        assert lines[-1] == app_main.FOOTER[lang] and "ora help topics" in lines[-1]
        help_line = next(line for line in lines if line.startswith("  help  "))
        assert all(topic in help_line for topic in TOPICS), (lang, help_line)
    with pytest.raises(SystemExit) as raised:
        app_main.main(["--help"])
    assert raised.value.code == 0
    assert capsys.readouterr().out == app_main.overview()


def test_help_language_follows_ora_ui_lang_then_the_locale() -> None:
    assert app_main.language({}) == "en"
    assert app_main.language({"LANG": "zh_CN.UTF-8"}) == "zh"
    assert app_main.language({"LANG": "zh_CN.UTF-8", "ORA_UI_LANG": "en"}) == "en"
    assert app_main.language({"LC_ALL": "en_US.UTF-8", "LANG": "zh_CN.UTF-8"}) == "en"
    assert app_main.language({"ORA_UI_LANG": "zh"}) == "zh"


def test_every_public_command_answers_help_in_process(capsys) -> None:
    """Each command's ``--help``; a tool leaf's comes from the tool's own parser, named as the ora command."""

    for command in PUBLIC_COMMANDS:
        try:
            code = app_main.main([command, "--help"])
        except SystemExit as raised:
            code = raised.code
        assert code == 0, command
        assert f"usage: ora {command}" in capsys.readouterr().out, command


def _help_status(argv: list[str]) -> int:
    try:
        return app_main.main(argv)
    except SystemExit as raised:
        return int(raised.code or 0)


@pytest.mark.parametrize("command", list(COMMANDS))
def test_every_subcommand_and_tool_leaf_renders_its_help(command: str, capsys) -> None:
    """A parsed subcommand's help renders from its parser; a tool leaf's from the tool it fronts."""

    for path, _, child in _commands(build_parser()):
        if path[0] != command:
            continue
        usage = "usage: ora " + " ".join(path)
        if getattr(child, "tool", None) is None:
            assert child.format_help().startswith(usage), path
        else:
            assert _help_status([*path, "--help"]) == 0, path
            assert capsys.readouterr().out.startswith(usage), path


@pytest.mark.parametrize("kind", list(registry.BACKENDS))
def test_every_backend_tool_answers_help_as_an_ora_command(kind: str, capsys) -> None:
    item = registry.backend(kind)
    for command, tools in (("spec", item.spec_tools), ("qualify", item.qualify_tools)):
        for name in tools:
            assert _help_status([command, kind, name, "--help"]) == 0, (command, name)
            text = capsys.readouterr().out
            assert text.startswith(f"usage: ora {command} {kind}") and name in text, (command, name)


@pytest.mark.parametrize("kind", [
    "robotwin",
    pytest.param("robodojo", marks=pytest.mark.xfail(strict=True, reason=(
        "ora/orchestrator/robodojo_play_audit.py:390 fixes its parser's prog to 'python -m "
        "ora.orchestrator.robodojo_play_audit', so its usage does not name the ora command; "
        "the file stays as it is until its owner next changes it"))),
])
def test_each_play_audit_answers_help_as_an_ora_command(kind: str, capsys) -> None:
    assert registry.backend(kind).play_audit is not None
    assert _help_status(["play", "audit", "--env", kind, "--help"]) == 0
    assert capsys.readouterr().out.startswith(f"usage: ora play audit --env {kind}")


def test_ora_help_prints_the_overview_a_command_or_a_topic(capsys) -> None:
    assert app_main.main(["help"]) == 0
    assert capsys.readouterr().out == app_main.overview()
    with pytest.raises(SystemExit) as raised:
        app_main.main(["help", "plan", "show"])
    assert raised.value.code == 0
    assert capsys.readouterr().out.startswith("usage: ora plan show")
    assert app_main.main(["help", "topics"]) == 0
    listing = capsys.readouterr().out.splitlines()
    assert listing[0] == "Topic pages (ora help TOPIC):"
    assert [line.split()[0] for line in listing[1:]] == list(TOPICS)
    for line, (name, title) in zip(listing[1:], app_main.topic_titles().items()):
        assert line.rstrip().endswith(title) and title, name
    for name in TOPICS:  # a topic comes before the command of the same name (play)
        assert app_main.main(["help", name]) == 0
        assert capsys.readouterr().out == (ROOT / "ora/app/help" / f"{name}.md").read_text(encoding="utf-8")
    assert app_main.main(["help", "play", "audit"]) == 0  # the tool's listing, on stdout for --help
    assert "ora play audit" in capsys.readouterr().out
    with pytest.raises(SystemExit) as raised:
        app_main.main(["help", "help"])
    assert raised.value.code == 0
    assert capsys.readouterr().out.startswith("usage: ora help")
    assert app_main.main(["help", "no-such-thing"]) == 2
    assert "neither a command nor a topic page" in capsys.readouterr().err


def test_every_topic_page_exists_is_short_and_names_real_commands_and_pages() -> None:
    assert TOPICS == ("runs", "evidence", "play", "remote", "safety", "rl-interface")
    assert sorted(path.stem for path in app_main.TOPIC_DIR.glob("*.md")) == sorted(TOPICS)
    for name in TOPICS:
        text = app_main.topic_page(name)
        lines = text.splitlines()
        assert lines[0].startswith("# ") and text.endswith("\n"), name
        assert 30 <= len(lines) <= 80, (name, len(lines))
        # Distilled, not copied: each page ends by naming the documentation it comes from.
        assert "More: docs/" in "\n".join(lines[-3:]), name
        for page in re.findall(r"docs/[\w/.-]+\.md", text):
            assert (ROOT / page).is_file(), (name, page)
        for command in re.findall(r"(?<![\w./-])ora ([a-z][a-z-]*)", text):
            assert command in COMMANDS, (name, command)


def test_topic_pages_ship_with_the_package() -> None:
    patterns = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"]["setuptools"]["package-data"]["ora"]
    for name in TOPICS:
        assert any(fnmatch.fnmatchcase(f"app/help/{name}.md", pattern) for pattern in patterns), name


def test_the_repository_wrapper_prints_the_overview() -> None:
    completed = subprocess.run([str(ROOT / "scripts/ora"), "--help"], cwd=ROOT, check=True, text=True,
                               capture_output=True, env={**os.environ, "ORA_UI_LANG": "en"})
    assert completed.stdout == app_main.overview("en")
