"""Local SSH browser coordination without a simulator, provider or SSH server."""

import io
import json
import re
import shlex
import subprocess
import threading

import pytest

from ora.app.remote import open_when_ready, ssh_command


def test_ssh_forward_is_loopback_and_preserves_literal_remote_arguments():
    repo = "/space path/$(never-run)`also-never`"
    run = "/runs/special ' quote; $(never-run)"
    command, url = ssh_command("person@gpu-host", repo, ["watch", run], local_port=40201, remote_port=40202)
    assert command[command.index("-L") + 1] == "127.0.0.1:40201:127.0.0.1:40202"
    assert "ExitOnForwardFailure=yes" in command and "-tt" in command
    remote = command[-1]
    assert shlex.split(remote.split(" && exec ", 1)[0]) == ["cd", "--", repo]
    args = shlex.split(remote.split(" && exec ", 1)[1])
    assert args[:3] == ["./scripts/ora", "watch", run]
    assert args[-2:] == ["--world-public-url", url]
    assert "--confirm-launch" not in args
    assert args[args.index("--world-browser") + 1] == "off"


@pytest.mark.parametrize("host,repo,args", [
    ("-oProxyCommand=bad", "/repo", ["watch", "/run"]),
    ("gpu", "relative", ["watch", "/run"]),
    ("gpu", "/repo", ["launch", "/run"]),
    ("gpu", "/repo", ["watch", "/run", "--world-view=off"]),
])
def test_invalid_launcher_bindings_are_refused(host, repo, args):
    with pytest.raises(ValueError):
        ssh_command(host, repo, args, local_port=40201, remote_port=40202)


def test_only_healthy_world_service_opens_local_browser_once():
    calls, opened, warnings = [], [], []
    responses = [b'{"service":"wrong","status":"ready"}',
                 json.dumps({"service": "ora-live-world", "status": "ready"}).encode()]

    def fetch(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(responses.pop(0))

    assert open_when_ready("http://127.0.0.1:40201/live", threading.Event(), timeout=2,
                           fetch=fetch, opener=lambda url: opened.append(url) or True, report=warnings.append)
    assert len(calls) == 2 and calls[0][0].endswith("/health")
    assert opened == ["http://127.0.0.1:40201/live"] and warnings == []


def test_failed_or_closed_transport_does_not_open_browser_or_relaunch():
    opened, warnings = [], []
    stopped = threading.Event()

    def fetch(*args, **kwargs):
        stopped.set()
        raise OSError("SSH detached")

    assert not open_when_ready("http://127.0.0.1:40201/live", stopped, timeout=1,
                               fetch=fetch, opener=opened.append, report=warnings.append)
    assert opened == [] and warnings == []


def test_remote_launcher_help_needs_no_ssh_connection(capsys):
    from ora.app.remote import main
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "SSH hostname" in capsys.readouterr().out


def test_browser_os_failure_is_reported_without_retrying_tabs():
    opened, warnings = [], []

    def opener(url):
        opened.append(url)
        raise OSError("browser launcher failed")

    assert not open_when_ready("http://127.0.0.1:40201/live", threading.Event(), timeout=1,
                               fetch=lambda *args, **kwargs: io.BytesIO(b'{"service":"ora-live-world","status":"ready"}'),
                               opener=opener, report=warnings.append)
    assert len(opened) == len(warnings) == 1


def test_detaching_terminates_only_owned_ssh_process(monkeypatch):
    from ora.app import remote as ora_remote
    calls = []

    class Child:
        stderr = io.BytesIO(b"")

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout is None:
                raise KeyboardInterrupt
            return 0

        def terminate(self):
            calls.append(("terminate",))

    monkeypatch.setattr(ora_remote, "available_port", lambda requested: 40201)
    monkeypatch.setattr(ora_remote.subprocess, "Popen", lambda command, **kwargs: calls.append(command) or Child())
    monkeypatch.setattr(ora_remote, "open_when_ready", lambda *args, **kwargs: None)
    assert ora_remote.main(["gpu-host", "--repo", "/repo", "--", "interactive", "--spec", "/task.json"]) == 130
    assert sum(isinstance(call, list) for call in calls) == 1
    assert ("terminate",) in calls
    assert "--confirm-launch" not in calls[0][-1]


def remote_words(command):
    """The host-side words of an ssh invocation, after the shell's `cd` (or login-shell) prefix."""
    remote = command[-1]
    if " && exec " in remote:
        return shlex.split(remote.split(" && exec ", 1)[1])
    return shlex.split(shlex.split(remote)[-1])[1:]


def captured_ssh(monkeypatch, argv, environ=None, opened=None):
    from ora.app import remote as ora_remote

    commands = []

    class Child:
        stderr = io.BytesIO(b"")

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(ora_remote, "available_port", lambda requested: 40201)
    monkeypatch.setattr(ora_remote.subprocess, "Popen", lambda command, **kwargs: commands.append(command) or Child())
    monkeypatch.setattr(ora_remote, "open_when_ready", lambda *args, **kwargs: None if opened is None else opened.append(kwargs.get("browse")))
    # Everything terminal_overrides reads, so the terminal the suite runs in (tmux sets TERM=screen) changes nothing.
    for name in ("TERM", "TERM_PROGRAM", "KITTY_WINDOW_ID", "ITERM_SESSION_ID", "TMUX", "COLORTERM", "WT_SESSION",
                 "GHOSTTY_RESOURCES_DIR", "WEZTERM_PANE", "WARP_SESSION_ID", "WARP_TERMINAL_SESSION_UUID",
                 "PI_IMAGE_PROTOCOL", "PI_TRUE_COLOR", "PI_HYPERLINKS", "ORA_UI_LANG", "LC_ALL", "LC_MESSAGES",
                 "LANG", "ORA_REMOTE_REPO"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (environ or {}).items():
        monkeypatch.setenv(name, value)
    assert ora_remote.main(argv) == 0
    assert len(commands) == 1
    return commands[0]


def test_play_runs_as_a_named_remote_session_with_the_world_view_forwarded(monkeypatch):
    command = captured_ssh(monkeypatch, ["gpu-host", "--repo", "/repo", "play", "--env", "synthetic"])
    words = remote_words(command)
    assert words[0] == "env" and words[words.index("./scripts/ora"):][:4] == ["./scripts/ora", "play", "--env", "synthetic"]
    session = next(word for word in words if word.startswith("ORA_REMOTE_SESSION="))
    assert re.fullmatch(r"ORA_REMOTE_SESSION=play-[0-9a-f]{16}", session)
    assert "ORA_REMOTE_FRESH=1" not in words
    assert words[words.index("--world-port") + 1] == "40201" and words[-1] == "http://127.0.0.1:40201/live"
    # The same command names the same session (so it re-attaches); --new asks for a fresh run.
    again = remote_words(captured_ssh(monkeypatch, ["gpu-host", "--repo", "/repo", "--new", "play", "--env", "synthetic"]))
    assert session in again and "ORA_REMOTE_FRESH=1" in again
    other = remote_words(captured_ssh(monkeypatch, ["gpu-host", "--repo", "/repo", "play", "--env", "libero"]))
    assert session not in other


def test_attach_names_its_run_and_without_repo_the_login_shell_finds_ora(monkeypatch):
    command = captured_ssh(monkeypatch, ["gpu-host", "attach", "run-7"])
    remote = command[-1]
    assert remote.startswith('exec "${SHELL:-/bin/sh}" -lc ')
    words = remote_words(command)
    assert words[:3] == ["ora", "attach", "run-7"] and not any(w.startswith("ORA_REMOTE_SESSION=") for w in words)
    legacy = remote_words(captured_ssh(monkeypatch, ["gpu-host", "--repo", "/repo", "--", "interactive", "--ui", "pi"]))
    assert legacy[legacy.index("./scripts/ora"):][:4] == ["./scripts/ora", "interactive", "--ui", "pi"]


@pytest.mark.parametrize("environ,expected", [
    ({"TERM_PROGRAM": "kitty"}, {"PI_IMAGE_PROTOCOL": "kitty", "PI_TRUE_COLOR": "1", "PI_HYPERLINKS": "1"}),
    ({"ITERM_SESSION_ID": "w0t0p0"}, {"PI_IMAGE_PROTOCOL": "iterm2", "PI_TRUE_COLOR": "1", "PI_HYPERLINKS": "1"}),
    ({"WEZTERM_PANE": "3"}, {"PI_IMAGE_PROTOCOL": "kitty", "PI_TRUE_COLOR": "1", "PI_HYPERLINKS": "1"}),
    ({"TERM_PROGRAM": "kitty", "TMUX": "/tmp/tmux-1/default,1,0"}, {}),
    ({"COLORTERM": "truecolor"}, {"PI_TRUE_COLOR": "1"}),
    ({"TERM_PROGRAM": "kitty", "PI_IMAGE_PROTOCOL": "none"},
     {"PI_IMAGE_PROTOCOL": "none", "PI_TRUE_COLOR": "1", "PI_HYPERLINKS": "1"}),
    ({"PI_IMAGE_PROTOCOL": "kitty; rm -rf ~"}, {}),
    # ORA's terminal words follow this terminal's language.
    ({"LANG": "zh_CN.UTF-8"}, {"ORA_UI_LANG": "zh"}),
    ({"LANG": "zh_CN.UTF-8", "ORA_UI_LANG": "en"}, {"ORA_UI_LANG": "en"}),
    ({"ORA_UI_LANG": "zh; rm -rf ~"}, {"ORA_UI_LANG": "zh"}),
    ({"LANG": "en_US.UTF-8"}, {}),
])
def test_this_terminals_image_protocol_reaches_the_remote_tui(environ, expected):
    from ora.app.remote import terminal_overrides

    assert terminal_overrides(environ) == expected


def test_kitty_terminal_settings_are_passed_to_the_host_command(monkeypatch):
    words = remote_words(captured_ssh(monkeypatch, ["gpu-host", "--repo", "/repo", "attach"], {"KITTY_WINDOW_ID": "1"}))
    assert {"PI_IMAGE_PROTOCOL=kitty", "PI_TRUE_COLOR=1", "PI_HYPERLINKS=1"} <= set(words)


def test_a_lost_connection_says_the_run_continues_and_how_to_return(monkeypatch, capsys):
    from ora.app import remote as ora_remote

    class Lost:
        stderr = io.BytesIO(b"")

        def wait(self, timeout=None):
            return 255

    monkeypatch.setattr(ora_remote, "available_port", lambda requested: 40201)
    monkeypatch.setattr(ora_remote.subprocess, "Popen", lambda command, **kwargs: Lost())
    monkeypatch.setattr(ora_remote, "open_when_ready", lambda *args, **kwargs: None)
    assert ora_remote.main(["gpu-host", "--repo", "/repo", "play", "--env", "synthetic"]) == 255
    error = capsys.readouterr().err
    assert "keeps going" in error and "ora-remote gpu-host --repo /repo play --env synthetic" in error


def test_a_command_is_required():
    from ora.app.remote import main

    with pytest.raises(SystemExit) as result:
        main(["gpu-host", "--repo", "/repo"])
    assert result.value.code == 2


def test_the_forwards_startup_failures_stay_off_the_terminal_and_a_refused_forward_is_named():
    from ora.app.remote import FORWARD_REFUSED, relay_ssh_diagnostics

    written, refused = [], threading.Event()
    relay_ssh_diagnostics(io.BytesIO(b"channel 2: open failed: connect failed: Connection refused\r\n"
                                     b"Warning: Permanently added 'gpu' to the list of known hosts.\r\n"), refused,
                          write=written.append)
    assert written == ["\rWarning: Permanently added 'gpu' to the list of known hosts.\r\n"] and not refused.is_set()
    relay_ssh_diagnostics(io.BytesIO(b"channel 3: open failed: administratively prohibited: open failed\n"), refused,
                          write=written.append)
    assert refused.is_set() and len(written) == 1
    # The browser waiter says so at once (before the remote TUI draws) and stops probing.
    warnings, calls = [], []
    assert not open_when_ready("http://127.0.0.1:40201/live", threading.Event(), refused=refused,
                               fetch=lambda *args, **kwargs: calls.append(args) or io.BytesIO(b"{}"),
                               report=warnings.append)
    assert warnings == [FORWARD_REFUSED] and calls == []


def test_after_a_drop_the_terminal_modes_of_the_remote_tui_are_put_back(monkeypatch, capsys):
    from ora.app import remote as ora_remote

    restored = []
    monkeypatch.setattr(ora_remote, "saved_terminal", lambda: "modes")
    monkeypatch.setattr(ora_remote, "restore_terminal", lambda saved, abnormal: restored.append((saved, abnormal)))

    class Killed:
        stderr = io.BytesIO(b"")

        def wait(self, timeout=None):
            return -9

    monkeypatch.setattr(ora_remote, "available_port", lambda requested: 40201)
    monkeypatch.setattr(ora_remote.subprocess, "Popen", lambda command, **kwargs: Killed())
    monkeypatch.setattr(ora_remote, "open_when_ready", lambda *args, **kwargs: None)
    assert ora_remote.main(["gpu-host", "--repo", "/repo", "play", "--env", "synthetic"]) == -9
    assert restored == [("modes", True)] and "keeps going" in capsys.readouterr().err
    restored.clear()
    captured_ssh(monkeypatch, ["gpu-host", "--repo", "/repo", "attach"])  # a normal end: ssh put the modes back
    assert restored == [("modes", False)]


def test_the_reset_turns_off_what_pi_tui_turns_on():
    from ora.app.remote import TERMINAL_RESET

    for sequence in ("\x1b[?2004l", "\x1b[<u", "\x1b[>4;0m", "\x1b[?25h"):
        assert sequence in TERMINAL_RESET


WEB = {"socket": "/run/user/7/ora-remote/play-abc-12345678.sock", "token_file": "/run/user/7/ora-remote/play-abc-12345678.token"}


def test_the_web_command_forwards_the_gateway_socket_and_hands_over_the_token_file():
    from ora.app.remote import ssh_command

    command, url = ssh_command("gpu-host", "/repo", ["web", "--last"], local_port=40201, remote_port=40201,
                               web=WEB, options=["-o", "ControlPath=/tmp/x/cm"])
    assert url == "http://127.0.0.1:40201/" and command[command.index("-L") + 1] == f"127.0.0.1:40201:{WEB['socket']}"
    assert "ControlPath=/tmp/x/cm" in command
    words = remote_words(command)
    assert words[words.index("./scripts/ora"):] == ["./scripts/ora", "web", "--last", "--socket", WEB["socket"],
                                                    "--token-file", WEB["token_file"], "--no-browser"]
    play, _ = ssh_command("gpu-host", "/repo", ["play", "--env", "synthetic"], local_port=40201, remote_port=40201,
                          web=WEB)
    words = remote_words(play)
    assert f"ORA_WEB_SOCKET={WEB['socket']}" in words and f"ORA_WEB_TOKEN_FILE={WEB['token_file']}" in words
    assert words[-2:] == ["--world-view", "off"], "the cameras are in the web view's dock"
    with pytest.raises(ValueError):
        ssh_command("gpu-host", "/repo", ["web"], local_port=40201, remote_port=40201)


def test_the_token_reaches_the_host_on_stdin_never_on_a_command_line():
    from ora.app.remote import prepare_web

    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="/run/user/7/ora-remote\n", stderr="")

    paths = prepare_web("gpu-host", "play-abc-12345678", "secret-token-value", ["-o", "ControlMaster=auto"],
                        timeout=5, run=run)
    assert paths == WEB
    argv, kwargs = calls[0]
    assert kwargs["input"] == "secret-token-value\n" and not any("secret-token-value" in word for word in argv)
    assert "umask 077" in argv[-1] and "play-abc-12345678" in argv[-1]
    with pytest.raises(ValueError):
        prepare_web("gpu-host", "../x", "t", [], timeout=5, run=run)
    failing = lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, stdout="", stderr="denied")
    with pytest.raises(OSError, match="denied"):
        prepare_web("gpu-host", "play-x", "t", [], timeout=5, run=failing)


def test_the_web_view_opens_with_its_token_once_the_gateway_answers():
    calls, opened = [], []

    def fetch(url, timeout):
        calls.append(url)
        return io.BytesIO(json.dumps({"service": "ora-web", "status": "ready"}).encode())

    assert open_when_ready("http://127.0.0.1:40201/", threading.Event(), timeout=2, fetch=fetch,
                           opener=lambda url: opened.append(url) or True, browse="http://127.0.0.1:40201/?ora_token=t")
    assert calls == ["http://127.0.0.1:40201/ora/health"] and opened == ["http://127.0.0.1:40201/?ora_token=t"]


def test_web_mode_prepares_the_host_first_and_opens_the_tokened_link(monkeypatch):
    from ora.app import remote as ora_remote

    prepared, browsed = [], []
    monkeypatch.setattr(ora_remote, "prepare_web", lambda host, name, token, options, timeout:
                        prepared.append((host, name, token, options)) or WEB)
    command = captured_ssh(monkeypatch, ["gpu-host", "--repo", "/repo", "--web", "play", "--env", "synthetic"],
                           opened=browsed)
    (host, name, token, options), = prepared
    assert host == "gpu-host" and re.fullmatch(r"play-[0-9a-f]{16}-[0-9a-f]{8}", name)
    assert any(option.startswith("ControlPath=") for option in options) and set(options) <= set(command)
    assert browsed == [f"http://127.0.0.1:40201/?ora_token={token}"]
    assert not any(token in word for word in command), "the token is not on the ssh command line"
