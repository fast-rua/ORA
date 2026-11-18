"""The frames card (``ora/agent/code_mode/ora_lib/card.py``): written from the current published observation.

The card states how this scene's world axes show in each camera's image, which cameras move, and where the arms
and their local axes are. Each statement is checked against the frames layer's own arithmetic on the same
observation (``direction``, ``Point.to``, ``Pose.axis``), and the card follows the observation: a camera that moved
or an arm that moved changes its lines, a fixed camera's lines stay.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from ora.agent.code_mode.library import ObservationMount, observation_mount
from ora.agent.code_mode.ora_lib import Point, direction, frames_card
from ora.agent.code_mode.ora_lib import observation as library_observation
from ora.agent.code_mode.sandbox import sandbox_status
from ora.agent.code_mode.shell import run_command
from ora.agent.toolkit.interface.types import canonical, digest

STATUS = sandbox_status()


@pytest.fixture
def observation(published, tmp_path):
    mount = ObservationMount(published.control_root, tmp_path / "mount")
    mount.root.mkdir()
    mount.refresh()
    return library_observation.load(str(mount.root))


def numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.\d+", text)]


def test_the_card_states_the_conventions_then_each_camera_and_each_arm(observation) -> None:
    card = frames_card(observation)
    lines = card.splitlines()
    assert lines[0] == "Frames card, observation observation-0001 (step 1)."
    assert "Metres and radians" in lines[1] and "[x, y, z, w]" in lines[1] and "world +Z is up" in lines[1]
    assert "camera:<name> is OpenCV (x right, y down, z forward)" in lines[1]
    assert "pixel:<name> has u right and v down from the top-left" in lines[1]
    cameras = [line.split(" (")[0] for line in lines if re.match(r"\w+ \((fixed|mount )", line)]
    assert cameras == ["head", "left_wrist", "right_wrist"], "the main camera first, fixed before moving"
    assert [line.split(": ")[0] for line in lines if line.startswith("tcp:")] == ["tcp:left", "tcp:right"]
    assert lines[-1].startswith("Which local axis is the approach axis")
    assert "(fixed, 640x480)" in card and "mount left_wrist, moves with the robot" in card
    assert card.count("This view moves") == 2, "only the cameras that move say so"
    assert len(card) < 2500 and frames_card(observation) == card, "short, and the same text for the same observation"


def test_a_camera_line_agrees_with_the_frames_layer(observation) -> None:
    """Position, optical axis, image right and image up along the table are the camera's own transform; the words say
    where each world axis lands in the image."""
    head = next(line for line in frames_card(observation).splitlines() if line.startswith("head ("))
    position = Point([0, 0, 0], "camera:head").to("world", observation=observation).xyz
    forward = Point([0, 0, 1], "camera:head").to("world", observation=observation).xyz - position
    stated = numbers(head)
    assert stated[:3] == pytest.approx(position.round(2).tolist(), abs=1e-9)
    assert stated[3:6] == pytest.approx(forward.round(2).tolist(), abs=1e-9)
    assert f"{np.degrees(np.arcsin(-forward[2])):.0f} deg below horizontal" in head
    shows = next(line for line in frames_card(observation).splitlines() if line.startswith("  Image right"))
    right = direction("right", view="head", observation=observation)
    up = direction("up", view="head", horizontal=True, observation=observation)
    assert numbers(shows) == pytest.approx([*right.v.round(2), *up.v.round(2)], abs=1e-9)
    # The head camera looks down toward +Y from the robot's side: +X is image-right, +Y runs up the image and away,
    # +Z runs up the image and toward the camera.
    assert "World +X shows right; +Y shows up and away from the camera; +Z shows toward the camera and up." \
        in frames_card(observation)


def test_an_arm_line_is_the_measured_pose_and_its_local_axes(observation) -> None:
    line = next(line for line in frames_card(observation).splitlines() if line.startswith("tcp:left:"))
    pose = observation.pose("left")
    expected = [*pose.xyz, *(pose.axis(name).v for name in "xyz")]
    assert numbers(line) == pytest.approx(np.round(np.concatenate([np.ravel(v) for v in expected]), 2), abs=1e-9)
    assert "-0.00" not in frames_card(observation), "a rounded zero has no sign"


def test_the_card_follows_the_observation(published, observation, tmp_path) -> None:
    """After the left arm moved, its line changes and the fixed camera's lines do not; a moved camera would too."""
    before = frames_card(observation)
    pointer_path = published.control_root / "current-observation.json"
    pointer = json.loads(pointer_path.read_bytes())
    source = Path(pointer["source_root"])
    root = source.parent / "observation-0002"
    shutil.copytree(source, root)
    manifest = json.loads((root / "observation.json").read_bytes())
    manifest.update(observation_id="observation-0002", sequence=2)
    manifest["proprioception"]["end_effectors"]["left"]["pose_xyz_xyzw"][:3] = [0.1, -0.2, 0.9]
    (root / "observation.json").write_bytes(canonical(manifest))
    (root / "proprioception.json").write_bytes(canonical(manifest["proprioception"]))
    pointer.update(source_root=str(root), observation_id="observation-0002", policy_state={"step": 2},
                   observation_sha256=digest((root / "observation.json").read_bytes()))
    pointer_path.write_bytes(canonical(pointer))
    mount = ObservationMount(published.control_root, tmp_path / "after")
    mount.root.mkdir()
    mount.refresh()
    after = frames_card(library_observation.load(str(mount.root)))
    assert after != before and after.splitlines()[0].startswith("Frames card, observation observation-0002")
    changed = [(a, b) for a, b in zip(before.splitlines()[1:], after.splitlines()[1:]) if a != b]
    assert [a.split(":")[0] for a, _ in changed] == ["tcp"] and "at world (0.10, -0.20, 0.90)" in changed[0][1]
    assert [line for line in before.splitlines() if line.startswith("head")] == [
        line for line in after.splitlines() if line.startswith("head")]


def test_a_camera_that_did_not_verify_is_listed_not_hidden(observation) -> None:
    observation.camera_errors["right_wrist"] = {"code": "artifact_hash_mismatch", "message": "depth drift"}
    del observation.cameras["right_wrist"]
    card = frames_card(observation)
    assert "right_wrist: not available in this observation (depth drift)." in card
    assert not [line for line in card.splitlines() if line.startswith("right_wrist (")]


@pytest.mark.skipif(not STATUS["ok"], reason=f"code-mode sandbox unavailable: {STATUS.get('reason')}")
def test_a_program_prints_the_card_the_host_computes(published, observation, tmp_path) -> None:
    """``ora_lib`` is bound read-only with its card module: a command's program gets the same text."""
    limits = {"wall_s": 60, "cpu_s": 30, "memory_mib": 2048}

    def answer(name, arguments):
        raise AssertionError("the card is no tool call")

    with observation_mount({"library": "extended"}, published.control_root) as mount:
        (tmp_path / "work").mkdir()
        (tmp_path / "work" / "card.py").write_text("from ora_lib import frames_card\nprint(frames_card())\n")
        command = run_command("python card.py", handle_call=answer, workspace=tmp_path / "work", extended=True,
                              observation=mount, **limits)
    assert command.status == "completed", command.output
    assert command.output.rstrip("\n") == frames_card(observation)
