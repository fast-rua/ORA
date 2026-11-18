"""Worker-only modules of ORA's own RoboTwin free-play scenes (``ora play --scene-set``).

``base.py`` and each scene set's module (``tabletop_v1.py``) import RoboTwin's
``envs`` package and run only in the isolated RoboTwin worker. The host-safe
registry of the sets, their scenes and source bindings is
``ora/environment/benchmarks/robotwin_play_scenes.py``.
"""
