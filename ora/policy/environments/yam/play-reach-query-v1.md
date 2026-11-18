## Where an arm reaches: one dry run

A `move_pose` target that inverse kinematics cannot solve is refused with
`target_ik_unreachable`, and the refusal says where the arm does reach:

- `highest_reachable_z_m`: the highest height that solves at the same x, y
  and approach (null when none down to the table clearance);
- `nearest_reachable_position_m`: the closest position at the same height and
  approach that solves, within 20 cm, and `nearest_reachable_offset_m`, how
  far it is from the target (both null when none).

To find where an arm reaches near a spot, dry-run that spot once and read
these fields. Do not search for reach with a grid of dry runs: each one costs
a call, and the refusal already names the nearest spot.
