"""Test helpers for robodojo play scenes, shared by several test modules.

They were defined in ``tests/ora/orchestrator/backends/robodojo/test_robodojo_play_scenes.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import json
from pathlib import Path

from ora.environment.benchmarks import robodojo_play_layouts as layouts


BOX = [[x, y, z] for x in (-0.03, 0.03) for y in (-0.02, 0.02) for z in (0.0, 0.05)]


def _asset_root(root: Path) -> Path:
    """A RoboDojo asset tree holding a toaster, a plate, a cup and a block, and the fixture layout."""
    for section, category in (("Articulation", "toaster"), ("Clutter", "plate"), ("Rigid", "cup"),
                              ("Rigid", "block")):
        for index in range(2):
            path = root / f"Object/RoboDojo/{section}/{category}/{index:05d}/metadata.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"geometry": {"aligned_bbox": {"vertices": BOX}},
                                        "active": {"place": {"up": {"orientation": [1.0, 0.0, 0.0, 0.0]}}},
                                        "physics": {"mass": 0.1}}))
    fixture = root / layouts.FIXTURE_LAYOUT
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps({"Room": {"r": 1}, "Table": {"default": "wood"}, "Ground": {"g": 1},
                                   "Background": {"b": 1}, "Geometry": {"camera_stand": [{"label": "stand"}]}}))
    return root


SCENE = {
    "gripper_keep_out": {"clearance_m": 0.05, "zones": [{"xlim": [-0.42, -0.24], "ylim": [-0.40, -0.21]},
                                                         {"xlim": [0.24, 0.42], "ylim": [-0.40, -0.21]}]},
    "Articulation": [{"common": {"xlim": [-0.30, -0.25], "ylim": [0.0, 0.05], "relative_plane": "Table"},
                      "category": [{"name": "toaster", "index": [0, 1]}],
                      "select_mode": {"label": ["toaster"]}}],
    "Clutter": [{"common": {"xlim": [-0.05, 0.05], "ylim": [-0.25, -0.20], "rotate_rand": True,
                            "rotate_deg": [-180, 180], "relative_plane": "Table"},
                 "category": [{"name": "plate", "index": [0]}], "physics": {"mass": 0.3},
                 "select_mode": {"label": ["plate"]}}],
    "Rigid": [{"common": {"xlim": [-0.005, 0.005], "ylim": [-0.005, 0.005], "relative_plane": "plate"},
               "category": [{"name": "cup", "index": [1]}], "select_mode": {"label": ["cup"]}},
              {"common": {"xlim": [0.10, 0.30], "ylim": [-0.30, 0.0], "relative_plane": "Table"},
               "category": [{"name": "block", "index": [0, 1]}], "select_mode": {"label": ["block"]}}],
}
