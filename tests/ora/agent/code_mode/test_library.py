"""The extended library's sandbox mounts (``ora/agent/code_mode/library.py``), on the host.

A contract without ``library: extended`` mounts nothing, and its bubblewrap command is the one it always was.
An extended contract gets ``ora_lib``'s modules and a copy of the current published observation: the
artifacts ``geometry_query`` reads, resolved as the toolkit resolves them and checked against their hashes.
The copy is brought up to date after each call, so an observation a call published is the one read next. A
closed or missing observation, or a camera file that drifted, is reported, never guessed. With ``source`` it also gets
ORA's code at its repository paths, found from the checkout and its imports, and never a judge, a simulator adapter or
a credential.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from ora.agent.code_mode import library
from ora.agent.code_mode.library import (LIBRARY_MOUNT, OBSERVATION_MOUNT, SCHEMA, ObservationMount, library_binds,
                                         observation_mount)
from ora.agent.code_mode.ora_lib import ObservationUnavailable
from ora.agent.code_mode.ora_lib import observation as library_observation
from ora.agent.code_mode.sandbox import bwrap_command
from ora.agent.code_mode.shell import command_bwrap
from ora.agent.toolkit.tools.vision_ground.tool import run as ground
from tests.support.agent.toolkit_tools import api_for


def index(root: Path) -> dict:
    return json.loads((root / "current.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("contract", [{}, {"library": "base"}, {"library": None}, {"interface": "coding"}])
def test_a_contract_without_library_extended_mounts_nothing(tmp_path: Path, contract: dict) -> None:
    with observation_mount(contract, tmp_path / "control") as mount:
        assert mount is None
    workspace = tmp_path / "work"
    workspace.mkdir()
    base = bwrap_command(sys.executable, "bwrap", 3, workspace)
    assert base == bwrap_command(sys.executable, "bwrap", 3, workspace, extended=False, observation=None)
    command = command_bwrap(sys.executable, "bwrap", 3, workspace, "true")
    for argv in (base, command):
        assert not any(LIBRARY_MOUNT in item or OBSERVATION_MOUNT in item for item in argv)
    assert command[command.index("PYTHONPATH") + 1] == "/ora" and "PYTHONSAFEPATH" not in command


def test_an_extended_command_binds_ora_lib_and_the_observation(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    argv = bwrap_command(sys.executable, "bwrap", 3, workspace, extended=True, observation=tmp_path)
    binds = library_binds()
    assert {binds[index + 2] for index in range(0, len(binds), 3)} >= {
        f"{LIBRARY_MOUNT}/ora_lib/{name}"
        for name in ("__init__.py", "frames.py", "geometry.py", "observation.py", "vision.py")}
    assert all(argv[argv.index(target) - 2] == "--ro-bind" for target in binds[2::3])
    assert argv[argv.index(OBSERVATION_MOUNT) - 2:argv.index(OBSERVATION_MOUNT) + 1] == [
        "--ro-bind", str(tmp_path), OBSERVATION_MOUNT]
    command = command_bwrap(sys.executable, "bwrap", 3, workspace, "true", extended=True, observation=tmp_path)
    assert command[command.index("PYTHONPATH") + 1] == LIBRARY_MOUNT and command[command.index("PYTHONSAFEPATH") + 1] == "1"
    assert {f"{LIBRARY_MOUNT}/robot.py", f"{LIBRARY_MOUNT}/sitecustomize.py"} <= set(command)


def test_a_contract_with_source_binds_the_source_read_only_and_leaves_the_path_alone(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    binds = library.source_binds()
    assert {f"{library.SOURCE_MOUNT}/{name}" for name in (
        "ora/agent/toolkit/tools/move_pose/tool.py", "ora/agent/toolkit/lib/README.md", "ora/agent/toolkit/interface/api.py",
        "ora/environment/planners/curobo_v2/worker.py", "ora/environment/controllers/motion_planner.py",
        "ora/policy/embodiments/arx-x5/manual-public-geometry-v5.md")} <= set(binds[2::3])
    assert all(source == str(library.SOURCE_ROOT / target.removeprefix(f"{library.SOURCE_MOUNT}/"))
               for source, target in zip(binds[1::3], binds[2::3])), "each file at its path in the repository"
    argv = command_bwrap(sys.executable, "bwrap", 3, workspace, "true", extended=True, source=True)
    assert all(argv[argv.index(target) - 2] == "--ro-bind" for target in binds[2::3])
    assert argv[argv.index("PYTHONPATH") + 1] == LIBRARY_MOUNT
    assert library.SOURCE_MOUNT not in " ".join(command_bwrap(sys.executable, "bwrap", 3, workspace, "true", extended=True))


def test_the_source_is_read_from_the_checkout_so_a_new_file_needs_no_list(tmp_path: Path, monkeypatch) -> None:
    files = {
        "ora/agent/toolkit/lib/motion_plan/new.py": "from ora.contracts.limits import LIMIT\n",
        "ora/agent/toolkit/tools/new_tool/tool.py": "def run(api, args):\n    from ...lib import shared\n",
        "ora/agent/toolkit/tools/new_tool/data.bin": "x",
        "ora/agent/toolkit/tools/check_task/tool.py": "from ora.contracts.extra import VALUE\n",
        "ora/agent/toolkit/lib/shared.py": "x = 1\n",
        "ora/agent/toolkit/interface/unused.py": "x = 1\n",
        "ora/contracts/limits.py": "from . import units\nfrom ora.contracts.units import Point\n",
        "ora/contracts/units.py": "import numpy\n",
        "ora/contracts/extra.py": "x = 1\n",
        "ora/environment/planners/new_planner/worker.py": "import ora.environment.controllers.new_motion\n",
        "ora/environment/controllers/new_motion.py": "import ora.missing_module\n",
        "ora/policy/embodiments/new-robot/manual-public-geometry-v1.md": "x",
        "ora/policy/embodiments/new-robot/descriptor-public-geometry-v1.json": "{}",
        "ora/policy/embodiments/new-robot/manual-v1.md": "x",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(library, "SOURCE_ROOT", tmp_path)

    def mounted() -> list[str]:
        return [path.relative_to(tmp_path).as_posix() for path in library.source_files()]

    assert mounted() == sorted([
        "ora/agent/toolkit/lib/motion_plan/new.py", "ora/agent/toolkit/lib/shared.py",
        "ora/agent/toolkit/tools/new_tool/tool.py", "ora/contracts/limits.py", "ora/contracts/units.py",
        "ora/environment/controllers/new_motion.py", "ora/environment/planners/new_planner/worker.py",
        "ora/policy/embodiments/new-robot/descriptor-public-geometry-v1.json",
        "ora/policy/embodiments/new-robot/manual-public-geometry-v1.md"])
    (tmp_path / "ora/environment/planners/new_planner/later.py").write_text("from ora.contracts import extra\n")
    assert {"ora/environment/planners/new_planner/later.py", "ora/contracts/extra.py"} <= set(mounted()), (
        "a file added later, and what it imports")


#: Where ORA judges an episode, adapts a simulator, keeps the Supervisor's state or reads credentials, by repository path.
NOT_SOURCE_PATH = re.compile(
    r"^ora/(orchestrator|app|environment/backends|environment/benchmarks|agent/toolkit/host/finish)/|^ora/environment/[^/]+$"
    r"|^ora/contracts/official_limit\.py$|(^|/)(check_task|finish_episode|report_issue)/"
    r"|verdict|judge|evaluat|reward|official|terminat")
#: The simulators' own packages, which only a backend imports.
SIMULATORS = frozenset({"carb", "envs", "gym", "gymnasium", "i2rt", "isaaclab", "isaacsim", "libero", "mujoco", "omni",
                        "pxr", "robocasa", "robolab", "robosuite", "sapien"})
NOT_IMPORTED = ("ora.orchestrator", "ora.app", "ora.environment.backends", "ora.environment.benchmarks",
                "ora.agent.toolkit.host.finish", "ora.contracts.official_limit")
#: What judging an episode reads or writes.
JUDGE = re.compile(r"RewardManager|reward_manager|is_episode_end|official_verdict|official_success|official_terminal"
                   r"|round_verdict|check_success|classify_round_ending|ENDING_CONDITIONS")
CREDENTIAL = re.compile(r"_API_KEYS?\b|\bapi_key\b|\bBearer\b|\bAuthorization\b|SECRET|_TOKEN_FILE\b|\bHF_TOKEN\b|\.netrc"
                        r"|getpass")


def not_source(relative: str, text: str) -> str | None:
    """Why a file may not be mounted, by its path and by what it holds; None when it may."""

    if NOT_SOURCE_PATH.search(relative):
        return "a judge's, a simulator adapter's or the Supervisor's path"
    if relative.endswith(".py"):
        imported = set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                imported.add(node.module)
            elif (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) in (
                    "import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant)):
                imported.add(str(node.args[0].value))
        found = sorted(name for name in imported if name.split(".")[0] in SIMULATORS or name.startswith(NOT_IMPORTED))
        if found:
            return f"imports {found[0]}"
    for kind, pattern in (("judges", JUDGE), ("names a credential", CREDENTIAL)):
        if pattern.search(text):
            return kind
    return None


def test_the_source_holds_no_judge_no_simulator_adapter_and_no_credential() -> None:
    files = library.source_files()
    assert len(files) > 200
    secrets = [value for name, value in os.environ.items()
               if re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", name, re.I) and len(value) >= 16]
    for path in files:
        relative = path.relative_to(library.SOURCE_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        assert not library.SOURCE_EXCLUDED_TOOLS & set(path.relative_to(library.SOURCE_ROOT).parts), relative
        assert not_source(relative, text) is None, (relative, not_source(relative, text))
        assert not any(value in text for value in secrets), f"{relative} holds a credential's value"


@pytest.mark.parametrize("relative", [
    "ora/environment/backends/robodojo/round_verdict.py", "ora/orchestrator/evaluation/verdict_guard.py",
    "ora/agent/toolkit/tools/finish_episode/_verdict.py", "ora/agent/toolkit/host/finish/task_check.py",
    "ora/contracts/official_limit.py", "ora/environment/backends/robodojo/isaac_scene.py", "ora/app/env_file.py"])
def test_a_judge_a_simulator_adapter_or_a_credential_reader_would_be_refused_by_path_and_by_content(relative: str) -> None:
    text = (library.SOURCE_ROOT / relative).read_text(encoding="utf-8")
    assert not_source(relative, text) is not None
    assert not_source("ora/agent/toolkit/lib/moved.py", text) is not None, "even under a mounted path"


def test_a_planner_refusal_is_found_where_it_is_worded_and_an_import_where_it_names() -> None:
    files = library.source_files()
    worded = [path for path in files if "cm inside " in path.read_text(encoding="utf-8")]
    assert [path.relative_to(library.SOURCE_ROOT).as_posix() for path in worded] == [
        "ora/environment/planners/curobo_v2/worker.py"]
    tree = ast.parse(worded[0].read_text(encoding="utf-8"))
    named = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and not node.level
             and (node.module or "").startswith("ora.")}
    assert "ora.environment.controllers.motion_planner" in named
    for module in named:
        assert library.SOURCE_ROOT / f"{module.replace('.', '/')}.py" in files, module
    geometry = library.SOURCE_ROOT / "ora/policy/embodiments/arx-x5/manual-public-geometry-v5.md"
    assert geometry in files and "finger_1" in geometry.read_text(encoding="utf-8")


def test_the_mount_is_a_verified_copy_of_the_current_publication(published) -> None:
    from PIL import Image

    pointer = json.loads((published.control_root / "current-observation.json").read_text())
    source = published.control_root.parent / "policy-observations" / pointer["observation_id"]
    manifest = json.loads((source / "observation.json").read_text())
    with observation_mount({"library": "extended"}, published.control_root) as mount:
        root = mount.root
        found = index(root)
        assert found["schema"] == SCHEMA and found["step"] == 1 and found["camera_errors"] == {}
        assert (found["observation_id"], found["observation_sha256"]) == (
            pointer["observation_id"], pointer["observation_sha256"])
        assert sorted(found["cameras"]) == sorted(manifest["cameras"]) == ["head", "left_wrist", "right_wrist"]
        for name, camera in found["cameras"].items():
            files = manifest["cameras"][name]["files"]
            assert (camera["width"], camera["height"]) == (640, 480)
            assert (root / camera["files"]["depth_m"]).read_bytes() == (source / files["depth_numpy_m_f32"]["path"]).read_bytes()
            assert (root / camera["files"]["calibration"]).read_bytes() == (source / files["calibration"]["path"]).read_bytes()
            with Image.open(io.BytesIO((source / files["rgb_lossless_png"]["path"]).read_bytes())) as image:
                np.testing.assert_array_equal(np.load(root / camera["files"]["rgb"]), np.asarray(image.convert("RGB")))
        assert json.loads((root / found["proprioception"]).read_text()) == manifest["proprioception"]
        loaded = library_observation.load(str(root))
        assert loaded.observation_id == pointer["observation_id"] and loaded.arms() == ["left", "right"]
        assert loaded.camera("camera:head").depth_m.dtype == np.float32
    assert not root.exists()


def test_the_mount_follows_a_publication_a_call_made(published) -> None:
    run = published.control_root.parent
    pointer_path = published.control_root / "current-observation.json"
    first = "robodojo:general_pickup:layout-0:observation-0000"
    answers = []

    def publish(name: str, arguments: dict) -> dict:  # an action: the Supervisor publishes the next observation
        pointer = json.loads(pointer_path.read_text())
        pointer.update(observation_id=first, policy_state={"step": 0}, observation_sha256=hashlib.sha256(
            (run / "policy-observations" / first / "observation.json").read_bytes()).hexdigest())
        pointer_path.write_text(json.dumps(pointer))
        answers.append(name)
        return {"ok": True}

    with observation_mount({"library": "extended"}, published.control_root) as mount:
        before = index(mount.root)
        mount.served(lambda name, arguments: {"ok": True})("geometry_query", {})
        assert index(mount.root) == before  # nothing new published: the same copy
        served = mount.served(publish)
        assert served("move_pose", {}) == {"ok": True} and answers == ["move_pose"]
        after = index(mount.root)
        assert after["observation_id"] == first and after["step"] == 0
        assert after["proprioception"].startswith("0002/") and (mount.root / before["proprioception"]).is_file()
        assert library_observation.load(str(mount.root)).observation_id == first


GRIPPER = {"measured_opening_normalized": 0.5, "commanded_opening_normalized": 1.0, "command_target": "open",
           "measured_aperture_mm": None}


def context_with_facts(run: Path, observation_id: str | None, **facts) -> None:
    """The policy's current context (``context/observation/projector.py``), its public robot facts for one observation."""

    path = run / "sandbox/workspace/inputs/current-context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    delivered = {"schema": "ora.current-public-robot-facts.v1", "observation_id": observation_id, "cameras": {},
                 "end_effectors": {"left": {"gripper": GRIPPER}, "right": {"gripper": dict(GRIPPER, command_target="close")}},
                 **facts}
    path.write_text(json.dumps({"schema": "x", "public_robot_facts": delivered}), encoding="utf-8")


def test_the_public_robot_facts_of_the_mounted_observation_are_mounted(published) -> None:
    """What the observation message shows: the gripper and the mobile base, each arm's gripper on ``end_effectors``."""
    run = published.control_root.parent
    context_with_facts(run, "observation-0001", mobile_base={"x_m": 0.8, "y_m": -1.8, "yaw_deg": -179.5, "frame": "world"})
    measured = json.loads((run / "policy-observations/observation-0001/observation.json").read_text())["proprioception"]
    with observation_mount({"library": "extended"}, published.control_root) as mount:
        found = index(mount.root)
        assert found["robot_facts"] == "0001/robot-facts.json"
        loaded = library_observation.load(str(mount.root))
        assert loaded.robot_facts["mobile_base"]["yaw_deg"] == -179.5 and loaded.robot_facts["end_effectors"]["left"] == {
            "gripper": GRIPPER}
        assert loaded.end_effectors["left"]["gripper"] == GRIPPER
        assert loaded.end_effectors["right"]["gripper"]["command_target"] == "close"
        for arm, record in measured["end_effectors"].items():
            kept = {key: value for key, value in loaded.end_effectors[arm].items() if key != "gripper"}
            assert kept == {key: value for key, value in record.items() if key != "gripper"}, "the rest as published"
        assert json.loads((mount.root / found["proprioception"]).read_text()) == measured, "the mounted file is unchanged"
        assert loaded.pose("left").xyz.tolist() == measured["end_effectors"]["left"]["pose_xyz_xyzw"][:3]


def test_an_arm_the_facts_state_no_gripper_for_keeps_what_was_published(published) -> None:
    run = published.control_root.parent
    context_with_facts(run, "observation-0001", end_effectors={"left": {"pose_xyz_xyzw": [0] * 7}})
    measured = json.loads((run / "policy-observations/observation-0001/observation.json").read_text())["proprioception"]
    with observation_mount({"library": "extended"}, published.control_root) as mount:
        loaded = library_observation.load(str(mount.root))
        assert loaded.end_effectors == measured["end_effectors"] and "gripper" in measured["end_effectors"]["left"]


def test_a_surface_with_no_facts_or_a_context_of_another_observation_mounts_none(published) -> None:
    run = published.control_root.parent
    measured = json.loads((run / "policy-observations/observation-0001/observation.json").read_text())["proprioception"]
    with observation_mount({"library": "extended"}, published.control_root) as mount:  # no context at all
        loaded = library_observation.load(str(mount.root))
        assert "robot_facts" not in index(mount.root) and loaded.robot_facts == {}
        assert loaded.end_effectors == measured["end_effectors"]
    context_with_facts(run, "observation-0000")  # the context is another observation's: nothing is mixed
    with observation_mount({"library": "extended"}, published.control_root) as mount:
        assert library_observation.load(str(mount.root)).robot_facts == {}
    (run / "sandbox/workspace/inputs/current-context.json").write_text("{", encoding="utf-8")
    with observation_mount({"library": "extended"}, published.control_root) as mount:
        assert library_observation.load(str(mount.root)).robot_facts == {}


def test_the_facts_follow_a_publication_a_call_made(published) -> None:
    run = published.control_root.parent
    pointer_path = published.control_root / "current-observation.json"
    first = "robodojo:general_pickup:layout-0:observation-0000"
    context_with_facts(run, "observation-0001")

    def publish(name: str, arguments: dict) -> dict:  # the Supervisor writes the context, then opens the observation
        context_with_facts(run, first, mobile_base={"x_m": 1.0})
        pointer = json.loads(pointer_path.read_text())
        pointer.update(observation_id=first, policy_state={"step": 0}, observation_sha256=hashlib.sha256(
            (run / "policy-observations" / first / "observation.json").read_bytes()).hexdigest())
        pointer_path.write_text(json.dumps(pointer))
        return {"ok": True}

    with observation_mount({"library": "extended"}, published.control_root) as mount:
        assert "mobile_base" not in library_observation.load(str(mount.root)).robot_facts
        mount.served(publish)("move_pose", {})
        after = library_observation.load(str(mount.root))
        assert after.observation_id == first and after.robot_facts["mobile_base"] == {"x_m": 1.0}


def test_a_closed_or_missing_observation_and_a_drifted_camera_are_reported(published, tmp_path: Path) -> None:
    pointer_path = published.control_root / "current-observation.json"
    pointer = json.loads(pointer_path.read_text())
    source = published.control_root.parent / "policy-observations" / pointer["observation_id"]
    depth = source / "right_wrist" / "depth-m.npy"
    depth.chmod(0o644)
    depth.write_bytes(depth.read_bytes()[:-4] + b"\0\0\0\0")
    mount = ObservationMount(published.control_root, tmp_path / "mount")
    mount.root.mkdir()
    mount.refresh()
    found = index(mount.root)
    assert sorted(found["cameras"]) == ["head", "left_wrist"]
    assert found["camera_errors"]["right_wrist"]["code"] == "artifact_hash_mismatch"
    with pytest.raises(KeyError, match="hash drift"):
        library_observation.load(str(mount.root)).camera("right_wrist")

    pointer_path.write_text(json.dumps({**pointer, "open": False}))
    mount.refresh()
    assert index(mount.root)["error"]["code"] == "action_pending_fresh_observation"
    with pytest.raises(ObservationUnavailable, match="closed"):
        library_observation.load(str(mount.root))

    pointer_path.write_text(json.dumps({**pointer, "observation_sha256": "0" * 64}))
    mount.refresh()
    assert "hash drift" in index(mount.root)["error"]["message"]

    pointer_path.unlink()
    mount.refresh()
    assert index(mount.root)["error"]["code"] == "environment_unavailable"
    with pytest.raises(ObservationUnavailable, match="no published observation is mounted"):
        library_observation.load(str(tmp_path / "nowhere"))


def segment(published) -> dict:
    """A ``vision_ground`` call on the current observation: the candidate it returns and its mask file."""

    candidate = ground(api_for(published), {"camera": "head", "text": "object", "max_candidates": 1})["candidates"][0]
    pointer = json.loads((published.control_root / "current-observation.json").read_text())
    root = (published.artifact_root / "policy-observations" / pointer["observation_id"] / "head"
            / "segmentation-queries" / candidate["target_ref"])
    return {**candidate, "mask_file": root / "mask.npy"}


def test_the_mount_adds_the_masks_of_the_targets_recorded_on_the_observation(published) -> None:
    """A target recorded before the mount and one a call records are both there when the call returns, each with the
    tool's own mask, without copying the observation again; a later observation starts with none."""
    run = published.control_root.parent
    pointer_path = published.control_root / "current-observation.json"
    first = "robodojo:general_pickup:layout-0:observation-0000"
    earlier = segment(published)
    with observation_mount({"library": "extended"}, published.control_root) as mount:
        before = index(mount.root)
        assert list(before["targets"]) == [earlier["target_ref"]] and before["target_errors"] == {}
        record = before["targets"][earlier["target_ref"]]
        assert record["camera"] == "head" and record["mask"].startswith(before["proprioception"][:4])
        np.testing.assert_array_equal(np.load(mount.root / record["mask"]), np.load(earlier["mask_file"]))
        assert library_observation.load(str(mount.root)).target(earlier["target_ref"]).mask.dtype == bool

        found = {}

        def look(name: str, arguments: dict) -> dict:  # a call that records another target (another prompt)
            api = api_for(published, tool_call_id="call-1")
            found.update(ground(api, {"camera": "head", "text": "another object", "max_candidates": 1})["candidates"][0])
            return {"ok": True}

        taken_before = library_observation.load(str(mount.root))
        mount.served(look)("vision_ground", {})
        after = index(mount.root)
        assert after["cameras"] == before["cameras"] and after["proprioception"] == before["proprioception"]
        assert earlier["target_ref"] in after["targets"]
        scene = library_observation.load(str(mount.root))
        assert scene.target(found["target_ref"]).camera == "head"
        assert taken_before.target(found["target_ref"]).camera == "head", "taken before the call, same observation"
        before_mask = np.load(mount.root / after["targets"][earlier["target_ref"]]["mask"])
        mount.refresh()
        assert index(mount.root) == after, "nothing new: nothing rewritten"

        def publish(name: str, arguments: dict) -> dict:  # an action: the next observation has no target yet
            pointer = json.loads(pointer_path.read_text())
            pointer.update(observation_id=first, policy_state={"step": 0}, observation_sha256=hashlib.sha256(
                (run / "policy-observations" / first / "observation.json").read_bytes()).hexdigest())
            pointer_path.write_text(json.dumps(pointer))
            return {"ok": True}

        mount.served(publish)("move_pose", {})
        moved = library_observation.load(str(mount.root))
        assert moved.observation_id == first and moved.targets == {}
        with pytest.raises(KeyError, match="segment again"):
            moved.target(earlier["target_ref"])
        assert before_mask.shape == (480, 640)


def test_a_target_whose_mask_drifted_is_reported_not_mounted(published) -> None:
    target = segment(published)
    target["mask_file"].chmod(0o644)
    target["mask_file"].write_bytes(target["mask_file"].read_bytes()[:-1] + b"\x01")
    with observation_mount({"library": "extended"}, published.control_root) as mount:
        found = index(mount.root)
        assert found["targets"] == {} and found["target_errors"][target["target_ref"]]["code"] == "artifact_hash_mismatch"
        with pytest.raises(KeyError, match="did not verify"):
            library_observation.load(str(mount.root)).target(target["target_ref"])
