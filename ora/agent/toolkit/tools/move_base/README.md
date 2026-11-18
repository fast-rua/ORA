# move_base

Drive a mobile base (RoboCasa's PandaOmron) to a world pose: absolute
(`x_m`, `y_m`, `yaw_deg`) or relative to the measured base pose
(`forward_m`, `left_m`, `turn_deg`). The tool hands one motion-v3 `move_base`
action (`{"target": {"x_m", "y_m", "yaw_rad"}}`) to the environment, whose base
servo drives it and owns the tolerances, speed and contact stop; the receipt
comes with the next observation. Only where the tool context says
`motion.environment_plans` and the surface declares `mobile_base_navigation_v1`;
elsewhere `tool_unavailable`. A mixed or empty form, a non-finite number or a
target more than 6.0 m from the base is refused before anything is handed over.
`dry_run` stops at the proposal; the environment answers with the commanded
line, unplanned, and the footprint check its real call makes (RoboCasa refuses a
move whose footprint would meet what the depth shows, `base_path_in_collision`). The definition the model reads is RoboCasa's
`robocasa_mobile_v1/move_base.json`: this `description.json`'s description and
result, with every optional input also taking null
(`optional_argument_null_as_omitted_v1`).
