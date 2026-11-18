"""One episode under the Supervisor: bind the sessions, run the loop, record the result.

``coordinator.run_episode`` binds a RunPlan to an EnvironmentSession, a
ToolRuntime and a PolicySession and hands them to ``loop.EpisodeLoop``, which
owns the observe -> policy turn -> execute loop; ``action_executor`` executes
one admitted action or closed skill sequence, ``rounds`` and ``terminal`` the
round and episode boundaries, ``result`` the session result, ``play`` free
play. The trusted stops (``termination``, ``stagnation``, ``operator_stop``,
``wait_accounting``, ``native_allowance``) and the live interaction state
(``interaction``, ``live``, ``operator_control``) live beside them.
"""
