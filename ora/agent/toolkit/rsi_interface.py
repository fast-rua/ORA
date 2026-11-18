"""The interface an RSI agent's workdir is written against, and how it changed.

An RSI workdir (ora-rsi ``rsi-workdir``) holds a whole toolset and pins the ORA
commit it builds on. What it may rely on is this interface: the tool directory
layout, the description file schema, the data interface and lib a tool may call,
the surface file it may edit and the task file. ``RSI_INTERFACE_VERSION`` goes up
whenever any of that changes, and ``MIGRATION_NOTES[v]`` tells a workdir written
against ``v - 1`` what to change. ora-rsi reads both from a commit's source
(never by importing ORA) and copies the notes into a workdir it updates.
"""

RSI_INTERFACE_VERSION = 3

MIGRATION_NOTES = {
    1: (
        "Interface 1 (ORA 2026-10-26). A tool is ora/agent/toolkit/tools/<name>/: tool.py defines "
        "run(api, args); description.json (ora.toolkit-tool.v1: schema, name, description, inputSchema, "
        "resultSchema) is what the model sees. Tools import only the allowed standard modules, numpy, scipy "
        "and, relatively, ora.agent.toolkit.tools / lib / interface; lib and interface are fixed. Action "
        "tools return their completed receipt and remaining_budget (surface v5, tool_output). The surface "
        "file's tools list sets which tools the model sees and in what order; its prompt.md is the surface "
        "prompt. The task file (policy.task_file, ora/policy/task_files/<task>.md) replaces the reviewed "
        "completion contract in the first user message."
    ),
    2: (
        "Interface 2 (ORA 2026-10-26, D4). The surface prompt now reaches the model: under baseline "
        "robodojo_toolkit_v6 (the default for surface robodojo_placement_toolkit_v5) the system message is "
        "ORA's planner, then the surface's prompt.md, then the context scheme's prompt. Before this, "
        "prompt.md was recorded but never delivered. The planner and core layers no longer carry tool or "
        "context rules: the turn protocol, how tool images arrive, the compute rule and what each call "
        "spends are now in the v5 surface's prompt.md, which must have text. Start from ORA's v5 "
        "prompt.md and keep it true for your toolset (name your tools; drop rules for tools you removed)."
    ),
    3: (
        "Interface 3 (ORA 2026-11-02, one tool stack). Paths and RoboDojo's behaviour do not change: tools/, lib/ and "
        "interface/ stay where they are, and a toolset that runs only on RoboDojo needs no edit. The tools now also run "
        "where the environment plans each move, which a tool context says with motion.environment_plans; no surface "
        "selects it yet. A moving tool of yours that should run there asks move_pose._context.environment_plans(config) "
        "and hands over one action with move_pose.tool.propose(api, action, name, **fields), then returns without "
        "execute_action or any exec.* read. Arguments may be flat: move_pose.tool.surface_targets(arguments, config, "
        "scalar_key=...) reads both spellings, replacing a direct sparse_targets call, and measured_states(manifest, "
        "move_pose._context.arms(config)) reads a single arm, as observe does through its section's arms. compile_action "
        "and compile_and_submit gain keyword arguments (speed, scalar_key) with defaults that keep their old meaning. "
        "lib 0.21.0: observation.robot_state leaves ArmState.pose None where a backend publishes no pose_frame, so read "
        "the pose from the arm's public record if your tool runs there. Read every new key as optional: a context "
        "without one is RoboDojo's, and a sealed run's context never has one."
    ),
}
