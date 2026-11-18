"""The registry of profile ids, and the root their manifests live under.

Data, not logic: every registered profile is one line here, and a generation
that adds a sibling adds another. Splitting it keeps the resolver
about resolution; :mod:`tool_surface` reads both names from this module at
call time, so a test rebinds them here.
"""

from __future__ import annotations

from pathlib import Path


PROFILE_ROOT = Path(__file__).resolve().parents[3] / "policy" / "tool_surfaces"
PROFILE_CATALOG = {
    # RoboDojo's toolkit surfaces, file-backed (toolkit_surface.py): one description file
    # per tool and the tools' native tool_output receipts. v6 is the default (baseline
    # robodojo_toolkit_v7); v5 comes with robodojo_toolkit_v6 when that baseline is named.
    "robodojo_placement_toolkit_v5": PROFILE_ROOT / "robodojo_placement_toolkit_v5" / "surface.json",
    "robodojo_placement_toolkit_v6": PROFILE_ROOT / "robodojo_placement_toolkit_v6" / "surface.json",
    # The default generation of each backend (direct_v4). Each is its v1 base
    # manifest plus the later capability layers, with the published-limits
    # layers last.
    "robotwin_toolchain_b2_closed_noservo_v2": PROFILE_ROOT / "toolchain_b2_noservo_v1" / "robotwin_toolchain_b2_closed_noservo_v2-manifest.json",
    # The LIBERO default: direct_v2 plus the overhead footprint localization.
    "libero_placement_direct_overhead_v1": PROFILE_ROOT / "explicit_motion_v3" / "libero_placement_direct_overhead_v1-manifest.json",
    # RoboCasa365 PandaOmron: the LIBERO direct overhead single-arm surface
    # on the same grip-site convention plus the move_base mobile-base tool.
    "robocasa_mobile_direct_v1": PROFILE_ROOT / "robocasa_mobile_v1" / "robocasa_mobile_direct_v1-manifest.json",
    # RoboLab Droid Franka + Robotiq 2F-85: the RoboCasa surface without the
    # base, on the Robotiq TCP convention.
    "robolab_direct_v1": PROFILE_ROOT / "robolab_direct_v1" / "robolab_direct_v1-manifest.json",
    # The bimanual I2RT YAM station: the dual-arm direct tools without the
    # perception services or the RoboDojo-only worker callbacks.
    "yam_direct_v1": PROFILE_ROOT / "yam_direct_v1" / "yam_direct_v1-manifest.json",
    # The YAM station's default: yam_direct_v1 plus SAM3 segmentation and grounding
    # and Contact-GraspNet proposals on the yam_linear_4310_tcp controlled frame.
    "yam_perception_v1": PROFILE_ROOT / "yam_perception_v1" / "yam_perception_v1-manifest.json",
    # Synthetic's surface, which orchestrator/spec/quickstart.py names and which is
    # also the RunSpec default when policy.tool_surface is omitted (robot_runtime_v2,
    # orchestrator/spec/policy.py), and the Synthetic Native test surface
    # (native_atomic_causal_v1).
    "robot_runtime_v2": PROFILE_ROOT / "robot_runtime_v2" / "manifest.json",
    "native_atomic_causal_v1": (
        PROFILE_ROOT / "native_atomic_causal_v1" / "manifest.json"
    ),
    "local_atomic_cpu_v1": PROFILE_ROOT / "local_atomic_cpu_v1" / "manifest.json",
}

