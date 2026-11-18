"""The ``ora`` command: one parser, one command table (``main.COMMANDS``), one module per command.

``main`` builds the root parser and dispatches; each command module exposes
``register(subparsers)`` and ``run(args)`` and imports its implementation inside
``run``. The package imports nothing itself.
"""
