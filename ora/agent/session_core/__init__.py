"""The Python half of a ``native_api`` policy session.

The Pi core (``ora.agent.pi`` with ``ora-pi/``) builds its sessions from these
modules: the tool gateway and its provider tool definitions (``gateway``,
``tool_arguments``), the static prefix (``composition``, ``constants``,
``stage_guard``), the observation and tool-result projections (``domain``,
``actual_snapshot``, ``receipt_projection`` and the details it composes:
``effect_delivery``, ``action_summary``, ``contact_detail``,
``refusal_detail``, ``stop_detail``), the free-play stop reasons (``operator_messages``) and the
``PolicySession`` wrapper (``policy_session``).

``native`` in these names means the ``native_api`` policy kind and its
evidence formats (``NATIVE_SLOTS_*``, the run's ``native/`` directory), not the
Python Native agent core, which ORA removed (it remains in the robot-harness
repository). Modules load on demand: this package imports none of them.
"""
