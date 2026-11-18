"""Configuration of one run-dashboard process: where the runs are and how they are grouped.

Every path comes from a flag or the environment, never from a host's file system layout:
``ROBODOJO_ASSET_ROOT`` names RoboDojo's ``Assets`` directory (robot URDF under ``Robots/x5``, object
assets under ``Object/``), ``ROBOTWIN_ROOT`` a RoboTwin checkout (the aloha-agilex embodiment under
``assets/embodiments/aloha-agilex``), the mesh-conversion interpreter defaults to the running one.
Scene traces sit beside each experiment's ``runs/`` unless ``--scene-traces`` names one directory.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

ASSET_ROOT_ENV = "ROBODOJO_ASSET_ROOT"
ROBOTWIN_ROOT_ENV = "ROBOTWIN_ROOT"


def default_cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "ora", "dashboard")


@dataclass(frozen=True)
class Config:
    #: The outputs root (``<experiment>/runs/<run_id>/``), or one experiment directory (``runs/<run_id>/``).
    outputs: str
    #: Show only this experiment.
    experiment: str | None = None
    #: Dotted plan path whose value labels a run's condition; None: code mode on ("code") or off ("tools").
    condition: str | None = None
    #: Regular expression over cohort names: matching cohorts stay out of the overview until asked for.
    hide_cohorts: str | None = None
    #: RoboDojo's ``Assets`` directory, for the 3D replay's robot and object meshes.
    asset_root: str | None = None
    #: A RoboTwin checkout, for the 3D replay's aloha-agilex robot.
    robotwin_root: str | None = None
    #: One directory of scene traces (``<run_id>/``); None: ``scene-trace/`` beside each run's ``runs/``.
    scene_traces: str | None = None
    #: Derived point clouds and meshes; never inside a run.
    cache: str = field(default_factory=default_cache_dir)
    #: Interpreter with usd-core for the object-mesh conversion.
    mesh_python: str = sys.executable
    #: A run without result.json and without file activity for this long counts as a fault.
    stale_s: float = 1800.0
    verbose: bool = False

    @classmethod
    def build(cls, outputs: str, **kwargs) -> "Config":
        asset_root = kwargs.pop("asset_root", None) or os.environ.get(ASSET_ROOT_ENV) or None
        robotwin_root = kwargs.pop("robotwin_root", None) or os.environ.get(ROBOTWIN_ROOT_ENV) or None
        scene_traces = kwargs.pop("scene_traces", None)
        return cls(outputs=os.path.realpath(outputs),
                   asset_root=os.path.realpath(asset_root) if asset_root else None,
                   robotwin_root=os.path.realpath(robotwin_root) if robotwin_root else None,
                   scene_traces=os.path.realpath(scene_traces) if scene_traces else None, **kwargs)

    @property
    def allowed(self) -> tuple[str, ...]:
        """The roots ``api/file`` and ``api/ls`` may read (after resolving links)."""
        return (self.outputs,)

    def hides(self, cohort: str | None) -> bool:
        return bool(self.hide_cohorts and cohort and re.search(self.hide_cohorts, cohort))

    @property
    def robot_dir(self) -> str | None:
        return os.path.join(self.asset_root, "Robots", "x5") if self.asset_root else None

    @property
    def aloha_dir(self) -> str | None:
        """RoboTwin's aloha-agilex embodiment (URDF under ``urdf/``, its ``config.yml``)."""
        return os.path.join(self.robotwin_root, "assets", "embodiments", "aloha-agilex") if self.robotwin_root else None

    @property
    def object_root(self) -> str | None:
        return os.path.join(self.asset_root, "Object") if self.asset_root else None

    def experiments(self) -> dict[str, str]:
        """Experiment name -> experiment directory (the one holding ``runs/``)."""
        root = self.outputs
        if os.path.isdir(os.path.join(root, "runs")):
            name = os.path.basename(root.rstrip(os.sep))
            return {name: root} if self.experiment in (None, name) else {}
        if self.experiment is not None:
            path = os.path.join(root, self.experiment)
            return {self.experiment: path} if os.path.isdir(os.path.join(path, "runs")) else {}
        try:
            names = sorted(os.listdir(root))
        except OSError:
            return {}
        return {n: os.path.join(root, n) for n in names
                if not n.startswith(".") and os.path.isdir(os.path.join(root, n, "runs"))}
