"""A receipt line shows the arm the action moved."""
from ora.agent.tui.condense import receipt_lines

AFTER = {"left": {"pose_xyz_xyzw": [-0.18, -0.09, 0.80, 0, 0, 0, 1]},
         "right": {"pose_xyz_xyzw": [0.30, -0.352, 0.922, 0, 0, 0, 1]}}


def test_the_receipts_target_arm_wins_over_the_right_arm_default():
    receipt = {"receipt": {"execution_status": "reached", "target_arms": ["left"], "after": AFTER}}
    assert receipt_lines(receipt, {"arm": "left", "block_index": 0}) == ["reached  end effector (-0.180, -0.090, 0.800) m"]


def test_a_toolsets_arm_argument_names_the_arm_when_the_receipt_does_not():
    receipt = {"receipt": {"execution_status": "reached", "after": AFTER}}
    assert "(-0.180, -0.090, 0.800)" in receipt_lines(receipt, {"arm": "left"})[0]
    assert "(0.300, -0.352, 0.922)" in receipt_lines(receipt, {})[0]


def test_a_cut_move_shows_the_distance_left_not_the_cut_points_residual():
    arms = {"left": {"pose_error": {"translation_m": 0.0001},
                     "motion_progress": {"outcome": "cut_short", "remaining_to_goal_m": 0.1157}}}
    receipt = {"receipt": {"execution_status": "not_reached", "target_arms": ["left"], "after": AFTER, "arms": arms}}
    assert receipt_lines(receipt, {})[0].endswith("error 115.7 mm")
