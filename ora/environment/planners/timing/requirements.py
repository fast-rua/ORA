"""The exact package set of the timing interpreter (``ORA_TIMING_PYTHON``): CPython 3.11, CPU only.

The worker needs numpy 1.26, toppra and ruckig; this is what ``uv pip install "numpy==1.26.*"
"toppra==0.6.10" "ruckig==0.19.4"`` resolved to on 2026-10-31. toppra needs scipy and imports matplotlib at
module level; ruckig has no dependencies. Install it into a fresh venv of its own, never into a simulator's
or planner's interpreter; ``ora setup`` does (the ``motion-timing`` component). Standard library only.

A module rather than a ``.txt``: package data is declared in the pinned ``pyproject.toml``.
"""

PINS = (
    "contourpy==1.3.3",
    "cycler==0.12.1",
    "fonttools==4.68.0",
    "kiwisolver==1.5.1",
    "matplotlib==3.11.2",
    "numpy==1.26.4",
    "packaging==26.3",
    "pillow==13.0.0",
    "pyparsing==3.3.3",
    "python-dateutil==3.0.0",
    "ruckig==0.19.4",
    "scipy==1.17.1",
    "six==1.17.0",
    "toppra==0.6.10",
    "tzdata==2026.6",
)

__all__ = ["PINS"]
