"""Inside the sandbox, with ``library: extended``: where a ``bash`` command's Python finds the model's own modules.

``shell.py`` binds this file on ``PYTHONPATH`` and sets ``PYTHONSAFEPATH``, so Python leaves the script's directory and
the working directory off ``sys.path``. In their usual place, first, a workspace file stands for any module of its
name wherever that module is imported: ``/work/inspect.py`` broke ``import numpy`` and ``ora_lib``, and itself. ``site``
imports this file when ``sys.path`` holds the library, the standard library and site-packages; it puts those two
directories and ``/work`` after them. A workspace file is then found unless something installed has its name, and
``python /work/inspect.py`` still runs that file. ``sys.argv[0]`` is set by this point: the script as given, ``-c``,
``-m``, ``-`` or empty (a program on standard input). Standard library only.
"""

from __future__ import annotations

import os
import sys

WORKSPACE = "/work"


def _program_directory() -> str:
    """The directory Python would have put first: the script's, else the working directory."""

    program = sys.argv[0]
    return os.getcwd() if program in ("", "-", "-c", "-m") else os.path.dirname(os.path.realpath(program))


for directory in (_program_directory(), WORKSPACE):
    if directory not in sys.path:
        sys.path.append(directory)
