"""Test helpers for controlled point projection, shared by several test modules.

They were defined in
``tests/ora/agent/context/observation/test_controlled_point_projection.py``; test modules
import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from tests.support.agent.public_robot_facts import fixture


DOJO_CALIBRATION = {
    "head": {
        "intrinsic_cv": [[288.1325423543974, 0.0, 320.0], [0.0, 288.1325423543974, 240.0], [0.0, 0.0, 1.0]],
        "extrinsic_cv": [[1.0, 0.0, 0.0, 0.0],
                         [-0.0, -0.8660253426938743, -0.5000001047713197, 0.29892973389791616],
                         [-0.0, 0.5000001047713196, -0.8660253426938744, 1.3377611621567869],
                         [0.0, 0.0, 0.0, 1.0]],
    },
    "left_wrist": {
        "intrinsic_cv": [[397.0412803766092, 0.0, 320.0], [0.0, 397.04128037660917, 240.0], [0.0, 0.0, 1.0]],
        "extrinsic_cv": [[0.9999999565494192, 0.00029479001915288296, -3.195105067183472e-09, 0.29963336078214503],
                         [0.00014740083371387824, -0.5000291473690891, -0.8660085324481392, 0.7084484986363131],
                         [-0.00025529228273259543, 0.8660084948190924, -0.5000291690948302, 0.717792426861671],
                         [0.0, 0.0, 0.0, 1.0]],
    },
    "right_wrist": {
        "intrinsic_cv": [[397.0412803766092, 0.0, 320.0], [0.0, 397.04128037660917, 240.0], [0.0, 0.0, 1.0]],
        "extrinsic_cv": [[0.9999999565494192, 0.00029479001915288296, -3.195105067183472e-09, -0.30036663698936344],
                         [0.00014740083371387824, -0.5000291473690891, -0.8660085324481392, 0.7083600581325704],
                         [-0.00025529228273259543, 0.8660084948190924, -0.5000291690948302, 0.7179456022373973],
                         [0.0, 0.0, 0.0, 1.0]],
    },
}
DOJO_POSES = {
    "left": [-0.2995290756225586, -0.35229986906051636, 0.921500027179718,
             -4.327419134142474e-08, -6.000607299938565e-08, 0.7072135210037231, 0.7069999575614929],
    "right": [0.3004709482192993, -0.35229986906051636, 0.921500027179718,
              -4.327419134142474e-08, -6.000607299938565e-08, 0.7072135210037231, 0.7069999575614929],
}


def _dojo_observation(tmp_path):
    import hashlib
    import json

    surface, observation, source = fixture(tmp_path, "robodojo")
    for name, calibration in DOJO_CALIBRATION.items():
        path = source / name / "calibration.json"
        path.write_text(json.dumps({**calibration, "source_camera": f"cam_{name}"}))
        observation["cameras"][name].update(width=640, height=480)
        observation["cameras"][name]["files"]["calibration"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for arm, pose in DOJO_POSES.items():
        observation["proprioception"]["end_effectors"][arm]["pose_xyz_xyzw"] = list(pose)
    (source / "observation.json").write_text(json.dumps(observation))
    return surface, observation, source
