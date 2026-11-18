"""The ORA version and command name: one answer for ``ora --version``, the terminal headers and the hints ORA prints.

Run from a source checkout (the repository's ``pyproject.toml`` beside the
``ora`` package, naming the ``ora-agent`` project), ORA is that checkout's
version, whatever an install recorded: an editable install made before a
version change, or a shared environment installed from another checkout,
keeps reporting the version it was installed at. Otherwise (a wheel or
sdist install) it is the installed ``ora-agent`` distribution's metadata.
Standard library only, so the agent and the orchestrator can both import it.
"""

from __future__ import annotations

import importlib.metadata
import os
import tomllib
from pathlib import Path

DISTRIBUTION = "ora-agent"
#: Where a source checkout keeps the project file: the directory that holds the ``ora`` package.
CHECKOUT_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _checkout_version(pyproject: Path) -> str | None:
    """The version ``pyproject`` declares when it is ORA's own project file, else None."""

    try:
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError):
        return None
    if not isinstance(project, dict) or project.get("name") != DISTRIBUTION:
        return None
    version = project.get("version")
    return version if isinstance(version, str) and version else None


def ora_version(pyproject: Path = CHECKOUT_PYPROJECT) -> str:
    """The source checkout's version, else the installed ``ora-agent`` metadata's, else ``"unknown"``."""

    declared = _checkout_version(pyproject)
    if declared is not None:
        return declared
    try:
        return importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def version_label(pyproject: Path = CHECKOUT_PYPROJECT) -> str:
    """``ora_version`` as the terminal headers show it: ``v0.1.0``, or ``unknown``."""

    version = ora_version(pyproject)
    return version if version == "unknown" else f"v{version}"


#: Set by the command the installer writes (``scripts/install.sh``): the name that user types.
COMMAND_ENV = "ORA_COMMAND"
#: The wrapper of a source checkout, run from its root.
CHECKOUT_COMMAND = "./scripts/ora"


def ora_command() -> str:
    """The command a hint should name: what this user types to run ORA.

    The installer's ``ora`` command exports ``ORA_COMMAND=ora``; anyone else runs a source checkout,
    whose wrapper is ``./scripts/ora``. A hint that named the wrapper to an installed user would
    point at a path that does not exist where they stand."""

    return os.environ.get(COMMAND_ENV, "").strip() or CHECKOUT_COMMAND


__all__ = ["CHECKOUT_COMMAND", "CHECKOUT_PYPROJECT", "COMMAND_ENV", "DISTRIBUTION", "ora_command", "ora_version",
           "version_label"]
