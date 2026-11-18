"""``ora task``: resolve and submit --print from the command line."""
import json
from pathlib import Path

from ora.app.main import main

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = str(ROOT / "examples/tasks/robodojo-toolkit-v6.yaml")


def test_resolve_prints_the_task_after_overrides(capsys):
    assert main(["task", "resolve", EXAMPLE, "--task", "align_blocks", "--layout", "3",
                 "--set", "policy.context_scheme=context_keep_written"]) == 0
    task = json.loads(capsys.readouterr().out)
    assert (task["task"], task["layout"], task["policy"]["context_scheme"]) == ("align_blocks", 3, "context_keep_written")
    assert main(["task", "resolve", EXAMPLE, "--set", "runtime.gpu=3"]) == 2


def test_submit_print_shows_the_request_without_sending_it(capsys):
    assert main(["task", "submit", EXAMPLE, "--owner", "me", "--layout", "2", "--print"]) == 0
    request = json.loads(capsys.readouterr().out)
    assert request["run"]["argv"][1:4] == ["task", "run", "--resolved"]
    assert json.loads(request["run"]["argv"][4])["layout"] == 2 and request["ora"]["layout"] == 2
