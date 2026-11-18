"""ORA's default agent core on Pi (TypeScript), with tools and authority in Python.

The TypeScript process in ``ora-pi/`` owns the provider loop, request assembly,
context management and agent-side evidence. This package is the Python half:
the policy-session adapter the Supervisor drives, the JSON-lines bridge that
serves tool calls and observations, and the shared hashing helpers.
"""
