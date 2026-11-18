"""Where the RoboDojo tests look for the pinned RoboDojo runtime (source checkout and Assets).

``ORA_TEST_ROBODOJO_RUNTIME`` names the runtime root, the directory
``ora setup --env robodojo`` installs (``source/RoboDojo``, ``assets/Assets``);
unset, it is ``~/robodojo-runtime``, setup's default. The cases that read it
skip with the reason when it is absent. Modules that already honoured
``ROBODOJO_ROOT`` or ``ROBODOJO_ASSET_ROOT`` still let those override the
source or Assets root. Every task reads the one pinned Assets checkout, as
the runtime does. A layout that is still a Git LFS pointer skips its case, the
reason naming each pointer file. When the variable is set,
``bind_named_robodojo_runtime``, which the ``conftest.py`` of
``tests/ora/environment/backends/robodojo/`` and ``tests/ora/orchestrator/backends/robodojo/``
run for every test there (their autouse ``named_robodojo_runtime`` fixture), also points product code at the same runtime: it sets
``ROBODOJO_ROOT``, ``ROBODOJO_ASSET_ROOT`` and ``ROBODOJO_PYTHON`` where they
are unset, which product code otherwise defaults to ``~/robodojo-runtime``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

RUNTIME_ENV = "ORA_TEST_ROBODOJO_RUNTIME"
RUNTIME_ROOT = Path(os.environ.get(RUNTIME_ENV) or Path.home() / "robodojo-runtime").expanduser()
SOURCE_ROOT = RUNTIME_ROOT / "source/RoboDojo"
ASSET_ROOT = RUNTIME_ROOT / "assets/Assets"
X5_ROOT = SOURCE_ROOT / "Assets/Robots/x5"


def product_environment() -> dict[str, str]:
    """The variables product code resolves this runtime by (``backends.robodojo.runtime_config.configure_environment``)."""

    return {"ROBODOJO_ROOT": str(SOURCE_ROOT), "ROBODOJO_ASSET_ROOT": str(ASSET_ROOT),
            "ROBODOJO_PYTHON": str(RUNTIME_ROOT / "env/bin/python")}


def bind_named_robodojo_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Product code defaults ``ROBODOJO_*`` to ``~/robodojo-runtime``; set them to the named runtime where unset."""

    if not os.environ.get(RUNTIME_ENV):
        return
    for name, value in product_environment().items():
        if name not in os.environ:
            monkeypatch.setenv(name, value)


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


def is_lfs_pointer(path: Path) -> bool:
    """Whether ``path`` holds a Git LFS pointer instead of the file (an Assets checkout without ``git lfs pull``)."""

    with path.open("rb") as handle:
        return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def skip_lfs_pointers(pointers: list[Path], checked: str) -> None:
    """Skip the case on Git LFS pointers, naming each file, whether or not ``ORA_TEST_ROBODOJO_RUNTIME`` is set.

    A pointer is an object that Assets checkout lacks. Pulling it is for the
    runtime's owner: a shared runtime that live runs read is never changed from
    a test, and upstream may no longer serve the object (swap_T's layouts at the earlier
    asset pin a14409d7, for example).
    """

    if not pointers:
        return
    pytest.skip(f"{', '.join(str(path) for path in pointers)} are Git LFS pointers, not their files (that Assets "
                f"checkout lacks those LFS objects; git lfs pull there fetches what upstream still serves); {checked}")
