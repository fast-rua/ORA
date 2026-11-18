"""Offline replay of one sealed ORA run on the Pi agent core.

``extract`` turns a sealed run into a self-contained JSON fixture, ``runtime``
holds the replay ToolRuntime, recorded provider responses and session builder,
and ``run_native`` (named for the native_api policy kind) drives the episode
and checks byte identity of every wire request against the archived one.
"""
