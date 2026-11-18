"""Versioned synchronous-tool contract, independent of released proposal tools."""


CAPABILITY = "toolkit_synchronous_execution_v1"
EXECUTION_TOOLS = frozenset({"move_to", "move_pose", "grasp_execute", "set_gripper", "recover_pose", "rotate_wrist", "rotate_pitch", "wait_ticks"})
