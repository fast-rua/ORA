"""A RoboTwin public run on the default surface, shared by several test modules.

``RoboTwinPublicRun`` publishes a real public timeline with source-bound masks
and distinct measured arm poses; ``history`` and ``pose_target`` build the
recorded actions and receipts beside it. Test modules import them from here,
never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import copy
import json

import numpy as np
from PIL import Image

from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.agent.context.observation.contact_points import project_policy_public_facts
from ora.agent.context import ContextProjector
from ora.agent.evidence.policy_state import PolicyStateStore, REQUIRED_ABSENT, verify_policy_state

from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA

from tests.support.agent.segmented_targets import (
    SelectedMask, _hash, _json, MANUAL, PROMPT, PROJECT, materialize_segmented_target,
)
from tests.support.agent.surface_catalog import handlers_only
from tests.support.agent.toolkit_run import served


#: The RoboTwin default surface.
PROFILE = "robotwin_toolchain_b2_closed_noservo_v2"
#: The fixture publishes the observer view its callers name and the views the
#: kept RoboTwin surfaces admit.
FIXED_CAMERAS = ("observer_camera", "head_camera", "front_camera")

ABSOLUTE = {"position_mode": "absolute", "orientation_mode": "absolute_quaternion"}
POSES = {"left": [-0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
         "right": [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}


def action(targets, *, name="set_gripper", step=0, interactions=None):
    return {"schema": "ora-motion-action.v3", "action": name,
            "action_id": f"action-{step}", "observation_id": f"obs-{step}",
            "bound_observation_sha256": f"{step:064x}",
            "arguments": {"targets": targets, **({"interactions": interactions} if interactions else {})}}


def pose_target(arm, z):
    return {"position_m": [*POSES[arm][:2], z], "orientation_xyzw": POSES[arm][3:]}


def history(targets, *, name="set_gripper", step=0, status="executed", changed=True):
    request = action(targets, name=name, step=step)
    receipt = {"schema": "ora.robotwin-sparse-motion-receipt.v7",
        "action_id": request["action_id"], "action": name, "arguments": request["arguments"],
        "target_arms": list(targets), "consumed": True, "state_changed": changed,
        "execution_status": status,
        "arms": {arm: {"targeted": arm in targets, "command_preservation": {
            "gripper_command_target_after": targets.get(arm) if name == "set_gripper" else "open"}}
            for arm in POSES}}
    return {"previous_public_action": request, "previous_execution_receipt": receipt}


class RoboTwinPublicRun:
    """Real public timeline, source-bound masks and distinct measured arm poses."""

    def __init__(self, root, *, profile=PROFILE, bind_submitted=False, toolkit=False):
        """``toolkit=True`` publishes where a run publishes (``policy-observations/``) and serves the runtime as a run
        serves it (``toolkit_run.served``); otherwise the legacy tool handlers serve it."""
        self.root, self.frames = root, []
        self.bind_submitted = bind_submitted
        self.observations = root / ("policy-observations" if toolkit else "observations")
        self.grippers = {arm: "open" for arm in POSES}
        self.store = PolicyStateStore(root, "robotwin-placement-fixture", "a" * 64)
        self.projector = ContextProjector(root / "workspace", root / "control", PROMPT, MANUAL,
            PROJECT, expose_manual=False, main_camera="observer_camera")
        fixture = SelectedMask()
        runtime = ManipulationToolRuntime(control_root=root / "control", manual_path=MANUAL,
            event_path=root / "events.jsonl", state_root=self.store.root,
            workspace_bounds_m={"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
            tool_surface_profile=profile, controlled_end_effector="right",
            segmentation_backend=fixture, grounding_backend=fixture, grasp_backend=fixture,
            public_facts_projector=project_policy_public_facts)
        self.runtime = runtime if toolkit else handlers_only(runtime)
        self.publish()
        if toolkit:
            served(self.runtime)

    def publish(self, targets=None, *, name="set_gripper", reset=False):
        step = len(self.frames)
        source = self.observations / f"obs-{step}"
        cameras = {}
        yy, xx = np.mgrid[:96, :96]
        obj = (xx - 48) ** 2 + (yy - 48) ** 2 <= 9 ** 2
        support = (xx >= 12) & (xx < 84) & (yy >= 12) & (yy < 84) & ~obj
        for camera in (*FIXED_CAMERAS, "left_camera", "right_camera"):
            directory = source / camera
            directory.mkdir(parents=True)
            rgb = np.full((96, 96, 3), [120, 126, 130], dtype=np.uint8)
            rgb[obj] = [190, 54, 31]
            Image.fromarray(rgb).save(directory / "rgb.png")
            depth = np.full((96, 96), .8, dtype=np.float32)
            depth[obj] = .795
            np.save(directory / "depth-m.npy", depth, allow_pickle=False)
            _json(directory / "calibration.json", {
                "intrinsic_cv": [[200., 0., 48.], [0., 200., 48.], [0., 0., 1.]],
                "extrinsic_cv": [[1., 0., 0., 0.], [0., -1., 0., 0.],
                                 [0., 0., -1., 1.2], [0., 0., 0., 1.]], "depth_unit": "metres"})
            files = {}
            for role, filename in (("rgb_lossless_png", "rgb.png"),
                ("depth_numpy_m_f32", "depth-m.npy"), ("calibration", "calibration.json")):
                path = directory / filename
                files[role] = {"path": path.relative_to(source).as_posix(),
                    "sha256": _hash(path), "bytes": path.stat().st_size}
            cameras[camera] = {"width": 96, "height": 96, "mount": "fixed" if camera in FIXED_CAMERAS else "wrist",
                "modalities": ["rgb", "depth"], "files": files}
        poses = copy.deepcopy(POSES)
        if step and not reset:
            poses = copy.deepcopy(self.frames[-1]["poses"])
        if name == "move_pose":
            for arm, target in (targets or {}).items():
                poses[arm] = [*target["position_m"], *target["orientation_xyzw"]]
        if self.bind_submitted:
            if reset:
                self.grippers = {arm: "open" for arm in POSES}
            elif name == "set_gripper":
                self.grippers.update(targets or {})
        observation = {"schema": "ora-observation.v1", "observation_id": source.name,
            "sequence": step, "main_camera": "observer_camera", "forbidden_modalities_absent": sorted(REQUIRED_ABSENT),
            "cameras": cameras, "proprioception": {"end_effectors": {
                arm: {"pose_xyz_xyzw": pose, "gripper_command_target": self.grippers[arm]} for arm, pose in poses.items()}}}
        descriptor = (self.runtime.surface.view_config or {}).get("public_geometry_descriptor")
        if descriptor:
            # The fixture robot is the descriptor's grip-centre robot: public robot facts project.
            observation["robot_geometry_verification"] = {
                "schema": GEOMETRY_VERIFICATION_SCHEMA, "descriptor_sha256": descriptor["sha256"],
                "scope": "loaded_robot_geometry_only", "passed": True}
            for effector in observation["proprioception"]["end_effectors"].values():
                effector["controlled_point_semantics"] = "parallel_jaw_grip_center_tcp"
        _json(source / "observation.json", observation)
        request = receipt = None
        if targets:
            joined = history(targets, name=name, step=step-1, status="executed" if name == "set_gripper" else "reached")
            request, raw = joined["previous_public_action"], joined["previous_execution_receipt"]
            if self.bind_submitted:
                request = json.loads((self.root / "control/submitted-action.json").read_text())
                raw.update(action_id=request["action_id"], action=request["action"],
                           arguments=copy.deepcopy(request["arguments"]), controller_tracking={
                               "schema": "ora.motion-execution.v1",
                               "backend": "robotwin", "sampled_steps": 4,
                               "physics_step_attempts": 4, "execution_started": True})
            request["bound_observation_sha256"] = _hash(self.frames[-1]["root"] / "observation.json")
            receipt_path = self.root / "native" / "receipts" / f"action-{step-1}.json"
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            _json(receipt_path, raw)
            receipt = {"path": receipt_path.relative_to(self.root / "native").as_posix(),
                "sha256": _hash(receipt_path), "receipt": raw}
        snapshot = self.store.publish(observation=observation, observation_root=source,
            cause="reset" if reset or not step else "after_action", task_instruction="place the selected item",
            previous_public_action=request, previous_execution_receipt=receipt,
            remaining_budget={"robot_decisions": 100-step, "model_calls": 300-step})
        self.projector.publish(observation=observation, observation_root=source,
            task_instruction="place the selected item", remaining_budget=snapshot["remaining_budget"],
            policy_state=self.store.steps[-1], previous_execution_receipt=receipt)
        regions = {}
        for camera in cameras:
            regions[camera] = {}
            directory = source / camera
            for label, mask in (("object", obj), ("support", support)):
                y, x = np.argwhere(mask)[len(np.argwhere(mask)) // 2]
                manifest, _ = materialize_segmented_target(artifact_root=self.runtime.artifact_root,
                    observation_id=source.name, camera=camera, rgb_path=directory / "rgb.png",
                    depth_path=directory / "depth-m.npy", calibration_path=directory / "calibration.json",
                    point_uv=(int(x), int(y)), label=label, backend=SelectedMask(mask))
                regions[camera][label] = manifest["target_ref"]
        self.frames.append({"root": source, "regions": regions, "poses": poses})
        assert verify_policy_state(self.root, verify_artifacts=True)["status"] == "valid"
