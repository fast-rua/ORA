# preview_move

Render one candidate grip-site pose on the current agentview. It compiles the pose as `move_pose` does
(`move_pose/_atomic.py`), keeps the target under its `preview_id` in `tool.previews` for this observation, and moves nothing.
A later `move_pose` with that `preview_id` alone submits exactly that target. Only an atomic surface has it. The candidate
image is a drawing on the current picture, not a future world.
