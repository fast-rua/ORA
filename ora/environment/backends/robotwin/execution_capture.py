"""RoboTwin original-execution media hooks, separate from motion and verdicts.

A scene of one of ORA's play scene sets (``play_scene_set``, whose task object
is a ``play_scenes.base.PlaySceneTask``) records its own capture cameras: the
head and front cameras and ORA's operator camera, rendered alone rather than
through the camera manager's ``update_picture`` (which renders the wrist
cameras too), at the plan's rate (30 fps for a scene-set plan). After each of
its actions the scene settles (``play_settle.settle_after_action``) inside the
action's capture window. A benchmark task, and play on an official task, keep
the head and front cameras, ``update_picture`` and no settle.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from ora.environment.execution_capture import ExecutionCapture, PhysicsStepTap

#: A scene-set play capture renders its next frame only once the writer holds at most this many frames
#: (``_await_capture_writer``): rendering holds the interpreter lock, so without the wait the writer thread
#: starved and dropped frames (measured 2026-10-21: 104 of 253 dropped, 0.37 s per written frame against
#: 0.095 s of encoding and writing on its own). The wait slows the motion, never the frames' simulated time.
PLAY_CAPTURE_WRITER_BACKLOG = 1
#: The longest one frame waits for the writer before rendering anyway (a stalled writer drops frames).
PLAY_CAPTURE_WRITER_WAIT_S = 5.0


def _await_capture_writer(recorder: ExecutionCapture) -> None:
    """Hold the simulation (sleeping, the interpreter lock released) until the writer has caught up."""
    backlog = getattr(recorder, "_queue", None)
    deadline = time.monotonic() + PLAY_CAPTURE_WRITER_WAIT_S
    while (backlog is not None and backlog.qsize() > PLAY_CAPTURE_WRITER_BACKLOG and recorder.error is None
           and time.monotonic() < deadline):
        time.sleep(0.002)


class RoboTwinExecutionCaptureMixin:
    """Observe the session's existing steps without advancing its world."""

    def _capture_operator_reset_frames(self, *, label: str) -> None:
        """Synchronize the settled reset scene before its first camera sample."""

        if not (self.operator_world_video or self.operator_execution_video):
            return
        # RoboTwin settles newly loaded actors without updating the renderer.
        # take_picture alone still sees pre-settle render transforms, whereas
        # the later policy get_obs refreshes them. Synchronize camera and scene
        # transforms directly: _update_render also randomizes lights, so an
        # extra call there would change the task's RNG and illumination state.
        try:
            environment = self._require_environment()
            environment.cameras.update_wrist_camera(
                environment.robot.left_camera.get_pose(),
                environment.robot.right_camera.get_pose(),
            )
            environment.scene.update_render()
        except Exception as exc:
            self._record_operator_capture_failure(exc)
            return
        self._capture_operator_frame(label=label)
        if self.operator_execution_video:
            self._capture_operator_execution_frame(
                action=None, phase="reset", control_step_index=None,
            )

    def _start_execution_capture(self) -> None:
        if not self.execution_capture["enabled"]:
            return
        try:
            if self._execution_recorder is None:
                self._execution_recorder = ExecutionCapture(
                    self.output_root, backend="robotwin",
                    identity={"run_id": self.output_root.parent.name, "task": self.task,
                              "source_manifest": self.source_manifest},
                    cameras=self._capture_cameras(), config=self.execution_capture,
                )
            self._execution_physics_step = 0
            self._execution_action = None
            scene = self._require_environment().scene
            self._execution_step_tap = PhysicsStepTap(scene, self._after_capture_physics_step)
            self._capture_execution_boundary("reset")
        except Exception as exc:
            self._execution_capture_error = f"{type(exc).__name__}: {exc}"
            if self._execution_recorder is not None:
                self._execution_recorder.error = self._execution_capture_error

    def _play_capture_scene(self) -> Any:
        """The scene-set play scene's task object when it renders its own capture cameras, else None."""
        if getattr(self, "play_scene_set", None) is None:
            return None
        environment = self._require_environment()
        return environment if callable(getattr(environment, "capture_images", None)) else None

    def _capture_cameras(self) -> list[str]:
        play = self._play_capture_scene()
        return list(play.capture_cameras) if play is not None else ["head_camera", "front_camera"]

    def _after_capture_physics_step(self) -> None:
        self._execution_physics_step += 1
        self._capture_execution_boundary("control", force=False)

    def _capture_execution_boundary(self, phase: str, *, force: bool = True) -> None:
        recorder = self._execution_recorder
        if recorder is None:
            return
        try:
            environment = self._require_environment()
            scene = environment.scene
            sim_time = self._execution_physics_step * float(scene.get_timestep())

            def pixels() -> dict[str, Any]:
                if callable(getattr(environment, "capture_images", None)) and self._play_capture_scene() is not None:
                    # A scene-set play scene renders only the cameras the capture records, and every frame
                    # it renders reaches the writer.
                    _await_capture_writer(recorder)
                    return environment.capture_images(recorder.cameras)
                # Do not call Base_Task._update_render: randomized lighting there
                # consumes RNG. Synchronize actual poses without changing lights.
                manager = environment.cameras
                manager.update_wrist_camera(environment.robot.left_camera.get_pose(),
                                            environment.robot.right_camera.get_pose())
                scene.update_render()
                manager.update_picture()
                raw = manager.get_rgb()
                return {name: raw[name]["rgb"] for name in recorder.cameras}

            context = self._execution_action or {}
            recorder.capture(
                pixels, round_index=self.round_index,
                action_id=context.get("action_id"), action=context.get("action"),
                action_sequence=context.get("action_sequence"),
                sim_step=self._execution_physics_step, sim_time_s=sim_time,
                step_unit="sapien_step_since_round_reset", phase=phase, force=force,
            )
        except Exception as exc:
            self._execution_capture_error = f"{type(exc).__name__}: {exc}"
            recorder.error = self._execution_capture_error

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._execution_action = {"action_id": payload.get("action_id"),
                                  "action": payload.get("action"),
                                  "action_sequence": self.action_sequence}
        self._capture_execution_boundary("action_start")
        try:
            from .wait import execute_wait, is_wait
            if is_wait(payload):  # session.py is descriptor-pinned; wait.py drives its helpers
                result = execute_wait(self, payload)
            else:
                result = self._execute_without_capture(payload)
            if getattr(self, "play_scene_set", None) is not None:
                # A scene-set play scene settles after the receipt, inside the action's capture window.
                from .play_settle import settle_after_action
                settle_after_action(self, payload)
            return result
        finally:
            self._capture_execution_boundary("action_end")
            self._execution_action = None

    def _close_execution_capture(self) -> None:
        if self._execution_step_tap is not None:
            self._execution_step_tap.close()
            self._execution_step_tap = None
        if self._execution_recorder is not None:
            self._capture_execution_boundary("close")
            self._execution_recorder.close()
