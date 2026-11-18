"""Host-path redaction: a fixed list of literal prefixes, longest first, each replaced by a placeholder that keeps the tail."""

from __future__ import annotations

import json
from pathlib import Path

from ora.agent.evidence.host_paths import HostPaths, candidates


def test_prefixes_are_replaced_longest_first_and_keep_the_rest_of_the_path() -> None:
    paths = HostPaths([("home", "/h/me"), ("python", "/h/me/.local/py"), ("outputs", "/h/me/out")])
    text = ('File "/h/me/.local/py/lib/python3.11/json/encoder.py", line 406; /h/me/out/run-1; /h/me; '
            "/h/me2/other; /h/me.bak; /work/push.py")
    redacted, replacements = paths.text(text)
    assert redacted == ('File "<host-python>/lib/python3.11/json/encoder.py", line 406; <host-outputs>/run-1; '
                        "<host-home>; /h/me2/other; /h/me.bak; /work/push.py")
    assert replacements == 3
    # JSON values are redacted string by string, keys included; an image is never touched.
    value = {"content": [{"type": "text", "text": "/h/me/x"},
                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,/h/me/x"}},
                         {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "/h/me/x"}}]}
    redacted, replacements = paths.text(value)
    assert replacements == 1 and redacted["content"][0]["text"] == "<host-home>/x"
    assert redacted["content"][1:] == value["content"][1:]
    assert paths.found(value) and not paths.found(redacted)


def test_the_prefixes_come_from_the_run_and_the_account(tmp_path: Path) -> None:
    run = tmp_path / "outputs" / "runs" / "run-1"
    run.mkdir(parents=True)
    base, environment = tmp_path / "uv" / "cpython-3.11.12", tmp_path / "envs" / "code-mode"
    (environment / "bin").mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text(f"home = {base}/bin\n")
    (run / "launch.json").write_text(json.dumps({"workspace": {
        "outputs_root": str(tmp_path / "outputs"), "workspace_root": str(tmp_path / "ws"),
        "source_worktree": str(tmp_path / "checkout")}}))
    (run / "preflight.json").write_text(json.dumps({"checks": [
        {"check": "code_mode_sandbox", "detail": {"interpreter": {"executable": str(environment / "bin/python")}}}]}))
    found = candidates(run, ["/data/lab/"])
    assert {name for name, _ in found} == {"run", "outputs", "workspace", "checkout", "python", "home", "path"}
    assert ("python", str(base)) in found and ("python", str(environment)) in found
    assert ("home", str(Path.home())) in found and ("path", "/data/lab") in found
    # Without the run's records, the run directory and the home directories remain.
    assert {name for name, _ in candidates(tmp_path)} == {"run", "home"}
