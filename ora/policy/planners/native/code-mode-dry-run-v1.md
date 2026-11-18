# Asking Before a Move (code mode)

Any call that moves the robot can be asked for its plan without moving.

- **You have the call and want to see it before it runs:** pass `dry_run=True`
  to it. It plans that call as the real one does and does not act: a picture of
  the path and `r.summary` reach you whether or not you print them; read
  `r.feasible`, `r.predicted_pose` and `r.failure_code`. The same call on the
  same observation then runs exactly that path. It predicts the arm, not the
  world: no contact, nothing after the first contact. It earns its call near
  objects, while holding something, or for a long move.
