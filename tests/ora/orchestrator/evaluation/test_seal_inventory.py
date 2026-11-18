"""The seal refuses a file named like a credential, except in the code-mode workspace, whose names the model chose.

Only the policy's ``write``/``edit`` calls and its sandboxed commands write ``native/code-workspace``; the sandbox
binds no credential, so a name there says nothing of what the file holds. Run
ora-glue-code-robodojo-15099b03d5ea--20261104T011143Z was refused for ``spots_auth.json``, hole positions it measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.orchestrator.evaluation.evaluate import EvaluationError, inventory

#: Names the model may give its own files.
MODEL_NAMES = ("spots_auth.json", "credentials.py", "api-key-notes.txt", "auth/auth.json")
#: Paths outside the workspace, which the harness names: a credential file there is refused.
HARNESS_PATHS = ("native/pi-agent/auth.json", "native/credentials.json", "sandbox/workspace/api-key",
                 "native/code-workspace-objects/auth.json")


def _run(root: Path, relative: str, *, policy: dict, play: bool = False) -> Path:
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({"policy": policy, **({"episode_mode": "play"} if play else {})}),
                                    encoding="utf-8")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"top_z": 0.784, "spots": [{"u": 227.0, "v": 296.9}]}), encoding="utf-8")
    return root


@pytest.mark.parametrize("plan_key", ["code_mode", "glue_code"])
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_a_file_the_model_named_in_its_workspace_is_sealed(tmp_path: Path, name: str, plan_key: str) -> None:
    root = _run(tmp_path / "run", f"native/code-workspace/{name}", policy={plan_key: {"contract": "c"}})
    assert f"native/code-workspace/{name}" in [record["path"] for record in inventory(root)]


@pytest.mark.parametrize("relative", HARNESS_PATHS)
def test_a_credential_named_file_the_harness_named_is_refused(tmp_path: Path, relative: str) -> None:
    root = _run(tmp_path / "run", relative, policy={"code_mode": {"contract": "c"}})
    with pytest.raises(EvaluationError, match=f"credential-like artifact cannot be sealed: {relative}"):
        inventory(root)


@pytest.mark.parametrize(("policy", "play"), [({}, False), ({"code_mode": {"contract": "c"}}, True)])
def test_the_workspace_keeps_the_rule_outside_sandboxed_code_mode(tmp_path: Path, policy: dict, play: bool) -> None:
    """Without code mode the model writes nothing there; in free play Pi's own shell runs on the host."""

    root = _run(tmp_path / "run", "native/code-workspace/spots_auth.json", policy=policy, play=play)
    with pytest.raises(EvaluationError, match="credential-like artifact cannot be sealed"):
        inventory(root)
