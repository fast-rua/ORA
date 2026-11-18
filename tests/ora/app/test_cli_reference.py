"""docs/usage/cli-reference.md is the parser's: ``ora dev cli-reference`` generates it, and a stale page fails here."""

from __future__ import annotations

import os

from ora.app import cli_reference
from ora.app.main import PUBLIC_COMMANDS, main


def test_the_page_is_what_the_parser_generates() -> None:
    page = cli_reference.PAGE.read_text(encoding="utf-8")
    assert page == cli_reference.render(), "run ./scripts/ora dev cli-reference --write"
    assert main(["dev", "cli-reference", "--check"]) == 0


def test_the_page_covers_every_public_command_in_order() -> None:
    page = cli_reference.render()
    headings = [line.split("`")[1] for line in page.splitlines() if line.startswith("### `ora ")]
    assert headings == [f"ora {name}" for name in PUBLIC_COMMANDS]
    # Subcommands and tool leaves have their own sections.
    for section in ("#### `ora plan show`", "#### `ora providers add`", "#### `ora play audit`",
                    "#### `ora dev cli-reference`"):
        assert section in page, section
    assert "Runs `ora.app.cli_reference:main` on the rest of the line." in page


def test_the_page_does_not_depend_on_the_terminal_or_the_host(monkeypatch) -> None:
    first = cli_reference.render()
    monkeypatch.setenv("COLUMNS", "40")
    assert cli_reference.render() == first
    assert os.environ["COLUMNS"] == "40"
    assert str(cli_reference.PAGE.parents[2]) not in first
    assert os.path.expanduser("~") not in first


def test_the_command_guide_lists_every_public_command_in_order() -> None:
    """The hand-written guide's table (what each command writes) keeps the parser's commands and order."""

    guide = cli_reference.PAGE.parent / "commands.md"
    documented = tuple(line.split("`", 2)[1] for line in guide.read_text(encoding="utf-8").splitlines()
                       if line.startswith("| `"))
    assert documented == PUBLIC_COMMANDS
