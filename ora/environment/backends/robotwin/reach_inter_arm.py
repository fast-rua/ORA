"""What is left of RoboTwin's reach-check other-arm screen: the flag ``session.py`` still validates.

The reach check now answers in the shared planner's world, where the other arm stands at
its measured joints (``shared_motion``), so no surface asks for the screen. ``session.py``
feeds the seed scans' hash, so its session field and this check stay until that file changes.
"""
from __future__ import annotations


def screen_mode(*, screen: bool) -> str | None:
    """No mode: a session's flag is only checked for being Boolean, and only ``False`` is served."""
    if not isinstance(screen, bool):
        raise ValueError("reach-check inter-arm screen flag must be Boolean")
    if screen:
        raise ValueError("the reach-check inter-arm screen was retired: the other arm stands in the planner's world")
    return None


__all__ = ["screen_mode"]
