"""ORA's own RoboDojo play scenes (``ora play --env robodojo``): registry, layouts, audit, admission and plan.

Offline: the RoboDojo assets are a small fixture tree in the pinned layout
(``Object/RoboDojo/<section>/<category>/<index>/metadata.json`` and one
official fixture layout), the play-scene audits are fixtures in the shape
``robodojo_play_audit.build_record`` writes, and no simulator, Provider or
resident service runs. The scenes' own build, settle and render evidence comes
from the audit worker on a GPU host (``backends/robodojo/play_scenes/audit.py``).
"""
from __future__ import annotations

import copy
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ora.environment.backends.robodojo.play_scenes import audit as audit_worker
from ora.environment.benchmarks import robodojo_play_layouts as layouts
from ora.environment.benchmarks import robodojo_play_scenes as registry
from ora.environment.benchmarks.robodojo import NATIVE_ACTION_LIMITS, REVISIONS, RoboDojoBenchmarkError
from ora.orchestrator import robodojo_play_audit, robodojo_play_kernel
from ora.orchestrator.play import scenes as play_scenes
from ora.orchestrator.play.binding import bind_play_runtime
from ora.orchestrator.play.scenes import PlaySceneError, build_play_spec, list_play_scenes
from ora.orchestrator.robodojo_play_audit import build_record, merge_seed, play_scene_admission
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from tests.support.orchestrator.play_spec import POLICY
from tests.support.robodojo.play_scenes import BOX, _asset_root, SCENE

ROOT = Path(__file__).resolve().parents[5]
SET = "isaac-tabletop-v1"
SCENES = ("breakfast", "desk", "toys", "kitchen_tidy", "blocks")
EXPECTATIONS = json.loads((ROOT / "tests/data/default-expectations.json").read_text())


# --- the registry --------------------------------------------------------------------------------

def test_the_isaac_tabletop_set_names_four_task_free_scenes_and_is_robodojos_default():
    assert registry.scene_sets() == (SET,) and registry.DEFAULT_SCENE_SET == SET
    assert registry.DEFAULT_SCENE_SET == EXPECTATIONS["play_scene_bindings"]["robodojo"]["play_scene_set"]
    assert EXPECTATIONS["play_scene_bindings"]["robodojo"]["worker_backend"] == registry.WORKER_BACKEND
    definitions = registry.scene_definitions(SET)
    assert tuple(item.name for item in definitions) == SCENES
    assert not set(SCENES) & set(NATIVE_ACTION_LIMITS)  # no scene is an official task
    assert registry.find_scene("toys").scene_set == SET and registry.find_scene("make_toast") is None
    with pytest.raises(registry.PlaySceneSetError, match="unknown RoboDojo play scene set"):
        registry.scene_definitions("tabletop-v1")
    with pytest.raises(registry.PlaySceneSetError, match="has no scene"):
        registry.scene_definition(SET, "make_toast")
    assert registry.scene_language(definitions[0]) == "ORA free-play scene isaac-tabletop-v1/breakfast: no task."
    # A scene's sources are its own YAML and the generator; the set's index lists the scenes.
    assert registry.source_files(SET, "desk") == tuple(sorted([
        registry.GENERATOR_MODULE, f"{registry.SCENE_ROOT}/isaac_tabletop_v1/desk.yml"]))
    assert registry.index_file(SET) == f"{registry.SCENE_ROOT}/isaac_tabletop_v1/index.yml"
    index = (ROOT / registry.index_file(SET)).read_text()
    for private in ("/mnt/", "/home/", "scratch", "preview"):
        assert private not in index


def _scene_tree(root: Path) -> Path:
    """A copy of the scene set's index, its YAMLs and the generator under ``root`` (a second ORA tree)."""
    for relative in (registry.index_file(SET), registry.GENERATOR_MODULE,
                     *(definition.scene_file for definition in registry.scene_definitions(SET))):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes((ROOT / relative).read_bytes())
    return root


def _edit_entry(root: Path, scene: str, field: str, value) -> None:
    path = root / registry.index_file(SET)
    data = yaml.safe_load(path.read_text())
    next(entry for entry in data["scenes"] if entry["name"] == scene)[field] = value
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def test_index_entries_are_validated_and_bound_per_scene_without_the_description(tmp_path):
    tree = _scene_tree(tmp_path)
    bindings = {name: registry.source_binding(SET, name) for name in SCENES}
    assert {name: registry.source_binding(SET, name, root=tree) for name in SCENES} == bindings
    desk = bindings["desk"]
    assert desk["schema"] == registry.SOURCE_SCHEMA and desk["scene"] == "desk"
    assert desk["entry"] == registry.canonical_entry(registry.scene_definition(SET, "desk"))
    assert "description" not in desk["entry"] and registry.UNBOUND_ENTRY_FIELDS == ("description",)
    assert [item["path"] for item in desk["files"]] == list(registry.source_files(SET, "desk"))
    assert len({binding["sha256"] for binding in bindings.values()}) == len(SCENES)
    # The operator's description binds nothing.
    _edit_entry(tree, "desk", "description", "A desk.")
    assert {name: registry.source_binding(SET, name, root=tree) for name in SCENES} == bindings
    for field, value, message in (("physics_device", "cuda:1", "physics_device"),
                                  ("settle_steps", 0, "settle_steps"), ("assets", ["phone"], "assets"),
                                  ("name", "Desk!", "scene name"), ("label", "", "label")):
        bad = _scene_tree(tmp_path / f"bad-{field}")
        _edit_entry(bad, "desk", field, value)
        with pytest.raises(registry.PlaySceneSetError, match=message):
            registry.scene_definitions(SET, root=bad)
    extra = _scene_tree(tmp_path / "extra")
    _edit_entry(extra, "toys", "seed", 3)
    with pytest.raises(registry.PlaySceneSetError, match="exactly the fields"):
        registry.scene_definition(SET, "toys", root=extra)


def test_one_scenes_edit_changes_only_that_scenes_binding(tmp_path):
    bindings = {name: registry.source_binding(SET, name)["sha256"] for name in SCENES}

    def changed(tree):
        return {name for name in SCENES if registry.source_binding(SET, name, root=tree)["sha256"] != bindings[name]}

    yaml_edit = _scene_tree(tmp_path / "yaml")
    (yaml_edit / registry.scene_definition(SET, "toys").scene_file).write_text("Rigid: []\n")
    assert changed(yaml_edit) == {"toys"}
    settle_edit = _scene_tree(tmp_path / "settle")
    _edit_entry(settle_edit, "breakfast", "settle_steps", 2500)
    assert changed(settle_edit) == {"breakfast"}
    assert registry.scene_definition(SET, "breakfast", root=settle_edit).settle_steps == 2500
    generator_edit = _scene_tree(tmp_path / "generator")
    (generator_edit / registry.GENERATOR_MODULE).write_bytes(
        (ROOT / registry.GENERATOR_MODULE).read_bytes() + b"# edited\n")
    assert changed(generator_edit) == set(SCENES)


@pytest.mark.parametrize("name", SCENES)
def test_each_scene_yaml_matches_its_registry_entry(name):
    definition = registry.scene_definition(SET, name)
    scene = layouts.load_scene(definition.path)
    sections = {f"{section}/{block['category'][0]['name']}" for section, block in layouts.blocks(scene)}
    assert sections == set(definition.assets)
    # An articulation needs CPU physics (RoboDojo reads its joint properties through numpy) and the
    # toaster's timer knob a 12 s settle; a rigid-only scene runs on CUDA and settles 6 s.
    articulated = bool(scene.get("Articulation"))
    assert definition.physics_device == ("cpu" if articulated else "cuda:0")
    assert definition.settle_steps == (3000 if articulated else 1500)
    keep_out = layouts.gripper_keep_out(scene)
    assert keep_out["clearance_m"] == 0.08 and len(keep_out["zones"]) == 2
    text = definition.path.read_text()
    for private in ("/mnt/", "/home/", "scratch", "preview"):
        assert private not in text


def test_the_registry_is_host_safe():
    probe = ("import sys; import ora.orchestrator.play.scenes, ora.orchestrator.robodojo_play_audit, "
             "ora.orchestrator.robodojo_play_kernel, ora.environment.benchmarks.robodojo_play_scenes, "
             "ora.environment.benchmarks.robodojo_play_layouts, ora.environment.robodojo_play_factory; "
             "assert not {'isaacsim', 'isaaclab', 'omni', 'pxr', 'carb', 'env', 'torch'} & set(sys.modules), "
             "sorted(sys.modules)")
    subprocess.run([sys.executable, "-c", probe], cwd=ROOT, check=True)


# --- the layout generator ------------------------------------------------------------------------


def test_a_seed_generates_the_same_saved_layout_every_time_and_keeps_its_rules(tmp_path):
    assets = _asset_root(tmp_path / "Assets")
    first = layouts.generate(SCENE, scene="fixture", seed=3, asset_root=assets)
    again = layouts.generate(copy.deepcopy(SCENE), scene="fixture", seed=3, asset_root=assets)
    assert registry.layout_sha256(first["layout"]) == registry.layout_sha256(again["layout"])
    other = layouts.generate(SCENE, scene="fixture", seed=4, asset_root=assets)
    assert registry.layout_sha256(other["layout"]) != registry.layout_sha256(first["layout"])
    layout = first["layout"]
    # RoboDojo's Eval_Layout record format: Clutter becomes cluttered Rigid records, fixtures are copied.
    assert set(layout) == {"Rigid", "Articulation", "Geometry", "Room", "Table", "Ground", "Background"}
    assert layout["Geometry"]["camera_stand"] == [{"label": "stand"}] and layout["Table"] == {"default": "wood"}
    plate = layout["Rigid"]["plate"][0]
    assert plate["type"] == "cluttered" and plate["physics"] == {"mass": 0.3, "type": "rigid"}
    assert layout["Articulation"]["toaster"][0]["physics"]["type"] == "articulation"
    placements = {item["label"]: item for item in first["placements"]}
    # The child sits on its parent: same x/y within its offset, bottom 3 mm above the parent's top.
    cup, base = placements["cup"], placements["plate"]
    assert abs(cup["spawn_pos"][0] - base["spawn_pos"][0]) <= 0.0071
    assert cup["spawn_pos"][2] == pytest.approx(base["spawn_pos"][2] + 0.05 + 0.003, abs=1e-6)
    assert base["spawn_pos"][2] == pytest.approx(registry.TABLE_TOP_Z_M + 0.002, abs=1e-6)
    # Table-level footprints keep the gripper keep-out clearance; children report none.
    assert all(item["gripper_zone_clearance_m"] >= 0.05 for item in first["placements"]
               if item["relative_plane"] == "Table")
    assert cup["gripper_zone_clearance_m"] is None
    assert layouts.object_kinds(first["placements"]) == {"toaster": "fixed", "plate": "movable", "cup": "movable",
                                                         "block": "movable"}
    assert set(first["metadata"]) == {item["metadata"] for item in first["placements"]}


def test_the_generator_refuses_what_it_cannot_place(tmp_path):
    assets = _asset_root(tmp_path / "Assets")
    orphan = {"Rigid": [{"common": {"xlim": [0, 0], "ylim": [0, 0], "relative_plane": "nowhere"},
                         "category": [{"name": "cup"}], "select_mode": {"label": ["cup"]}}]}
    with pytest.raises(layouts.LayoutGenerationError, match="unresolved relative_plane"):
        layouts.generate(orphan, scene="fixture", seed=0, asset_root=assets)
    missing = {"Rigid": [{"common": {"xlim": [0, 0], "ylim": [0, 0]}, "category": [{"name": "kettle"}],
                          "select_mode": {"label": ["kettle"]}}]}
    with pytest.raises(layouts.LayoutGenerationError, match="kettle/00000/metadata.json is missing"):
        layouts.generate(missing, scene="fixture", seed=0, asset_root=assets)
    crowded = copy.deepcopy(SCENE)
    crowded["gripper_keep_out"]["zones"] = [{"xlim": [-1.0, 1.0], "ylim": [-1.0, 1.0]}]
    with pytest.raises(layouts.LayoutGenerationError, match="could not place"):
        layouts.generate(crowded, scene="fixture", seed=0, asset_root=assets)
    with pytest.raises(layouts.LayoutGenerationError, match="non-negative"):
        layouts.generate(SCENE, scene="fixture", seed=-1, asset_root=assets)


def test_geometry_helpers_measure_overlap_and_clearance():
    square = layouts.zone_polygon({"xlim": [0.0, 1.0], "ylim": [0.0, 1.0]})
    assert layouts.separation(square, layouts.zone_polygon({"xlim": [1.5, 2.0], "ylim": [0.0, 1.0]})) == pytest.approx(0.5)
    assert layouts.separation(square, layouts.zone_polygon({"xlim": [0.5, 2.0], "ylim": [0.0, 1.0]})) < 0
    turned = layouts.footprint(BOX, [0.0, 0.0, 0.8], layouts.yaw_quaternion(90.0))
    xs = sorted({round(x, 6) for x, _ in turned})
    assert xs == [-0.02, 0.02]


# --- the audit worker's judgement ----------------------------------------------------------------

def _object(**changes):
    item = {"settled": {"position_m": [0.1, -0.1, 0.80]}, "final": {"position_m": [0.1, -0.1, 0.80]},
            "settle": {"drift_m": 0.002, "tilt_change_deg": 1.0},
            "window": {"displacement_m": 0.0, "peak_displacement_m": 0.0, "rotation_deg": 0.0, "joint_drift": {}},
            "contacts": {"<table>": -0.0005}, "ik": {"reachable_arms": ["right"]}, "gripper_clearance_m": 0.09}
    for key, value in changes.items():
        item[key] = value
    return item


def test_the_audit_judges_settle_window_contacts_reach_ik_and_gripper_clearance():
    assert audit_worker.judge_object(_object()) == []
    cases = {
        "settle": {"settle": {"drift_m": 0.05, "tilt_change_deg": 0.0}},
        "on_table": {"final": {"position_m": [0.1, -0.1, 0.70]}},
        "stability_window": {"window": {"displacement_m": 0.0, "peak_displacement_m": 0.0, "rotation_deg": 0.0,
                                        "joint_drift": {"0": 0.3}}},
        "penetration": {"contacts": {"plate": -0.004}},
        "robot_contact": {"contacts": {"<robot>": 0.0}},
        "reach_envelope": {"settled": {"position_m": [0.6, -0.1, 0.80]}},
        "ik_reachability": {"ik": {"reachable_arms": []}},
        "gripper_clearance": {"gripper_clearance_m": 0.01},
    }
    for check, change in cases.items():
        assert audit_worker.judge_object(_object(**change)) == [check], check
    # A child that came to rest within its support's footprint is settled, however far it slid.
    rolled = _object(settle={"drift_m": 0.05, "tilt_change_deg": 18.0, "on_support": True})
    assert audit_worker.judge_object(rolled) == []
    assert audit_worker.judge_object({**rolled, "settle": {**rolled["settle"], "on_support": False}}) == ["settle"]
    # An object resting on another one is checked through its support's position.
    leaning = _object(settled={"position_m": [0.1, 0.105, 0.85]})
    assert audit_worker.judge_object(leaning) == ["reach_envelope"]
    assert audit_worker.judge_object({**leaning, "envelope_position_m": [0.1, 0.09, 0.77]}) == []
    placements = {"holder": {"relative_plane": "Table"}, "pen": {"relative_plane": "holder"},
                  "cap": {"relative_plane": "pen"}}
    assert [audit_worker.table_support(label, placements) for label in ("cap", "pen", "holder")] == ["holder"] * 3
    # No contact report (a CUDA pass), no IK (a fixed articulation) and no clearance (a child) are not failures.
    assert audit_worker.judge_object(_object(contacts=None, ik=None, gripper_clearance_m=None)) == []
    failures = audit_worker.judge_seed({"mug": _object(), "egg": _object(ik={"reachable_arms": []})},
                                       missing=["bowl"], cameras={"head": {"std": 30.0}, "left_wrist": {"std": 0.5}})
    assert failures == ["bowl:missing", "egg:ik_reachability", "blank_camera:left_wrist"]


def test_the_ik_check_asks_align_plan_its_own_poses_for_every_declared_side():
    from ora.agent.toolkit.host.reach import alignment as tool
    from ora.contracts.alignment import APPROACH_SIDES, MAXIMUM_POSES
    from ora.environment.controllers.depth_world import PUBLIC_DEPTH_VOXEL_X5_V1

    criteria = audit_worker.CRITERIA
    # The same sides, directions and controlled frame as the policy's align_plan, at a standoff it permits.
    assert set(audit_worker.APPROACH_DIRECTIONS) == set(APPROACH_SIDES)
    assert audit_worker.APPROACH_DIRECTIONS == tool._APPROACH_DIRECTIONS
    standoff = audit_worker.approach_standoff_m()
    assert audit_worker.FINGERTIP_M == PUBLIC_DEPTH_VOXEL_X5_V1.fingertip_m == 0.1576
    assert standoff == pytest.approx(0.1576 + criteria["ik_fingertip_clearance_m"])
    assert tool.MINIMUM_STANDOFF_M <= standoff <= tool.MAXIMUM_STANDOFF_M
    axes = tool.frame_local_axes(audit_worker.CONTROLLED_FRAME)
    assert {name: list(value) for name, value in audit_worker.LINK6_LOCAL_AXES.items()} == {
        name: list(axes[name]) for name in ("approach", "closing")}
    rolls = audit_worker.roll_angles_deg()
    assert rolls == [30.0 * index for index in range(12)] and len(rolls) <= MAXIMUM_POSES
    face = [0.12, -0.04, 0.83]
    for side in APPROACH_SIDES:
        poses = audit_worker.approach_poses(face, side)
        # align_plan's six candidates are the first half; the second half are their jaw-symmetric flips.
        expected = tool.synthesise_approach_poses(face, approach_side=side, standoff_m=standoff,
                                                  candidate_count=6, local_axes=axes)
        assert poses[:6] == [pytest.approx(pose, abs=1e-9) for pose in expected], side
        direction = tool._APPROACH_DIRECTIONS[side]
        for pose in poses:
            assert pose[:3] == pytest.approx([face[i] - direction[i] * standoff for i in range(3)])
            x, y, z, w = pose[3:]
            # link6 +x (its approach axis) rotated by the quaternion points along the side's direction.
            approach = [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)]
            assert approach == pytest.approx(list(direction), abs=1e-9)
            # So the fingertips stop the clearance short of the face.
            tips = [pose[i] + approach[i] * audit_worker.FINGERTIP_M for i in range(3)]
            assert tips == pytest.approx([face[i] - direction[i] * criteria["ik_fingertip_clearance_m"]
                                          for i in range(3)])
    record = {"centre_m": [0.1, 0.05, 0.85], "top_z_m": 0.9,
              "footprint_xy": [[0.06, 0.03], [0.14, 0.02], [0.15, 0.08], [0.07, 0.09]]}
    assert {side: audit_worker.near_face(record, side) for side in APPROACH_SIDES} == {
        "top": [0.1, 0.05, 0.9], "front": [0.06, 0.05, 0.85], "back": [0.15, 0.05, 0.85],
        "left": [0.1, 0.09, 0.85], "right": [0.1, 0.02, 0.85]}
    assert audit_worker.box_centre([[-0.1, -0.05, 0.0], [0.1, 0.05, 0.2]], [1.0, 2.0, 0.8], [0.7071068, 0, 0, 0.7071068]
                                   ) == pytest.approx([1.0, 2.0, 0.9], abs=1e-6)
    assert audit_worker.rotation_deg([1, 0, 0, 0], [0.7071068, 0, 0, 0.7071068]) == pytest.approx(90.0, abs=1e-3)


#: Desk seed 0 as a failing audit recorded it (main 2086a93bc): each object's settled box centre and top, and how many
#: of its 24 top poses (12 rolls, 2 arms) the shared planner refused as ``goal_below_support``.
DESK_SEED_0 = {
    "book": ([-0.002277, -0.240771, 0.785073], 0.805123, 24), "keyboard": ([-0.00171, -0.030064, 0.776074], 0.787174, 24),
    "mouse": ([0.237522, -0.055727, 0.783211], 0.801717, 24), "mug": ([-0.307133, -0.038421, 0.802349], 0.839657, 24),
    "pen": ([-0.381452, 0.084944, 0.840155], 0.912754, 0), "pen_holder": ([-0.37316, 0.093527, 0.825014], 0.885034, 0),
    "phone": ([0.301493, 0.04701, 0.776139], 0.787538, 24), "stapler": ([-0.062236, 0.099641, 0.817372], 0.869772, 0),
    "tape_dispenser": ([0.17543, 0.099338, 0.791476], 0.817976, 24),
}


def test_the_recorded_desk_top_poses_were_below_the_table_and_now_stop_short_of_each_object():
    """The planner refuses a goal whose fingertips lie more than 1 cm below the support: the table, 0.765 m."""
    tolerance = 0.01
    for label, (centre, top, refused) in DESK_SEED_0.items():
        old = centre[2] + 0.10 - audit_worker.FINGERTIP_M
        assert (old < registry.TABLE_TOP_Z_M - tolerance) == (refused == 24), label
        pose = audit_worker.approach_poses(audit_worker.near_face({"centre_m": centre, "top_z_m": top}, "top"), "top")[0]
        tips = pose[2] - audit_worker.FINGERTIP_M
        assert tips == pytest.approx(top + audit_worker.CRITERIA["ik_fingertip_clearance_m"]), label
        assert tips > registry.TABLE_TOP_Z_M


def test_an_object_passes_ik_when_any_side_reaches_it_and_every_side_is_recorded():
    asked = []

    def plan(reachable):
        def answer(request):
            asked.append(request["approach_side"])
            assert request["arm"] == "both" and len(request["target_poses_xyzw"]) == 12
            return {"candidates": [{"arm": arm, "source_index": index,
                                    "reachable": (request["approach_side"], arm, index) in reachable,
                                    "rejection_reason": None if (request["approach_side"], arm, index) in reachable
                                    else "ik_failed"} for arm in ("left", "right") for index in range(12)]}
        return answer

    mug = {"centre_m": [0.1, 0.05, 0.85], "top_z_m": 0.9,
           "footprint_xy": [[0.06, 0.01], [0.14, 0.01], [0.14, 0.09], [0.06, 0.09]]}
    record = audit_worker.approach_reach(plan({("right", "right", 0), ("right", "right", 6)}), "mug", mug)
    assert asked == ["top", "left", "right", "front", "back"]
    assert record["reachable_sides"] == ["right"] and record["reachable_arms"] == ["right"]
    assert record["sides"]["right"]["reachable_rolls_deg"] == {"right": [0.0, 180.0]}
    standoff = audit_worker.approach_standoff_m()
    assert record["sides"]["top"] == {"face_m": [0.1, 0.05, 0.9], "link6_position_m": [0.1, 0.05, round(0.9 + standoff, 6)],
                                      "reachable_arms": [], "reachable_rolls_deg": {}, "rejections": {"ik_failed": 24}}
    assert record["sides"]["right"]["link6_position_m"] == [0.1, round(0.01 - standoff, 6), 0.85]
    assert record["standoff_m"] == round(standoff, 6) and record["controlled_frame"] == "arx_x5_link6"
    assert record["fingertip_m"] == 0.1576 and record["fingertip_clearance_m"] == 0.02
    assert audit_worker.judge_object(_object(ik=record)) == []
    unreachable = audit_worker.approach_reach(plan(set()), "mug", mug)
    assert unreachable["reachable_sides"] == [] and unreachable["reachable_arms"] == []
    assert audit_worker.judge_object(_object(ik=unreachable)) == ["ik_reachability"]
    # A side that reaches leaves the tilted approaches unplanned.
    planned = []
    reached = audit_worker.approach_reach(plan({("top", "left", 3)}), "mug", mug,
                                          plan_path=lambda arm, pose: planned.append(arm) or {"status": "planned"})
    assert reached["tilted"] is None and planned == []


def test_an_object_no_side_reaches_passes_when_a_tilted_approach_plans():
    def no_side(request):
        return {"candidates": [{"arm": arm, "source_index": index, "reachable": False, "rejection_reason": "ik_failed"}
                               for arm in ("left", "right") for index in range(12)]}

    # Beside the right arm's base, so the right arm is asked first.
    tape = {"centre_m": [0.2, 0.1, 0.79], "top_z_m": 0.82,
            "footprint_xy": [[0.15, 0.05], [0.25, 0.05], [0.25, 0.15], [0.15, 0.15]]}
    candidates = audit_worker.tilted_approaches(tape)
    asked = []

    def plan_path(winner):
        def answer(arm, pose):
            asked.append((arm, pose))
            return {"status": "planned"} if len(asked) == winner else {"status": "refused", "reason": "goal_unreachable"}
        return answer

    record = audit_worker.approach_reach(no_side, "tape", tape, plan_path=plan_path(5))
    assert record["rule"] == "any_declared_side_else_tilted_plan" and record["reachable_arms"] == []
    assert [arm for arm, _ in asked] == ["right", "left", "right", "left", "right"]
    assert [pose for _, pose in asked[::2]] == [candidate["pose_xyzw"] for candidate in candidates[:3]]
    assert [(candidate["side"], candidate["tilt_deg"]) for candidate in candidates] == [
        ("top", 10.0), ("right", 10.0), ("top", 20.0), ("right", 20.0), ("top", 30.0), ("right", 30.0)]
    tilted = record["tilted"]
    assert tilted["plans_tried"] == 5 and tilted["refusals"] == {"goal_unreachable": 4}
    assert tilted["planned"] == {"arm": "right", "side": "top", "tilt_deg": 20.0, "azimuth_deg": 90.0, "roll_deg": 0.0,
                                 "approach": candidates[2]["approach"],
                                 "link6_pose_xyzw": [round(value, 6) for value in candidates[2]["pose_xyzw"]]}
    assert audit_worker.judge_object(_object(ik=record)) == []
    asked.clear()
    nothing = audit_worker.approach_reach(no_side, "tape", tape, plan_path=plan_path(0))
    assert nothing["tilted"]["planned"] is None and nothing["tilted"]["plans_tried"] == 2 * len(candidates)
    assert audit_worker.judge_object(_object(ik=nothing)) == ["ik_reachability"]


def test_a_tilted_approach_turns_about_the_fingertips_of_its_side():
    criteria = audit_worker.CRITERIA
    tape = {"centre_m": [0.2, 0.1, 0.79], "top_z_m": 0.82,
            "footprint_xy": [[0.15, 0.05], [0.25, 0.05], [0.25, 0.15], [0.15, 0.15]]}
    wide = {**criteria, "tilt_sides": list(audit_worker.APPROACH_DIRECTIONS), "tilt_rolls_deg": [0.0, 90.0],
            "tilt_top_azimuths_deg": [0.0, 90.0, 180.0, 270.0]}
    candidates = audit_worker.tilted_approaches(tape, wide)
    assert len(candidates) == 3 * (4 + 4) * 2
    assert len(audit_worker.tilted_approaches(tape)) == 6
    assert [candidate["tilt_deg"] for candidate in candidates] == sorted(candidate["tilt_deg"] for candidate in candidates)
    for candidate in candidates:
        side = audit_worker.APPROACH_DIRECTIONS[candidate["side"]]
        tips = [value - side[axis] * criteria["ik_fingertip_clearance_m"]
                for axis, value in enumerate(audit_worker.near_face(tape, candidate["side"]))]
        x, y, z, w = candidate["pose_xyzw"][3:]
        approach = [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)]
        assert approach == pytest.approx(candidate["approach"], abs=1e-9)
        # The fingertips stay where the side's own pose puts them; the approach turns by the tilt.
        reached = [candidate["pose_xyzw"][axis] + approach[axis] * audit_worker.FINGERTIP_M for axis in range(3)]
        assert reached == pytest.approx(tips, abs=1e-9)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(approach, side))))))
        assert angle == pytest.approx(candidate["tilt_deg"], abs=1e-6)
        if candidate["side"] == "top":
            lean, azimuth = math.radians(candidate["tilt_deg"]), math.radians(candidate["azimuth_deg"])
            assert approach[:2] == pytest.approx([math.sin(lean) * math.cos(azimuth), math.sin(lean) * math.sin(azimuth)],
                                                 abs=1e-9)
        else:
            assert approach[2] < 0  # a side approach tilts down


#: What the recorded plans give each object the requal audits rejected: the side, tilt and arm that planned first.
TILTED_WINNERS = {
    ("breakfast", 0, "egg_0"): ("top", 10.0, "right"), ("breakfast", 0, "egg_1"): ("top", 10.0, "right"),
    ("breakfast", 1, "bread_0"): ("top", 10.0, "left"), ("breakfast", 1, "bread_1"): ("top", 10.0, "left"),
    ("breakfast", 1, "egg_0"): ("top", 10.0, "right"), ("breakfast", 1, "egg_1"): ("top", 10.0, "right"),
    ("desk", 0, "keyboard"): ("top", 10.0, "left"), ("desk", 0, "pen"): ("right", 20.0, "left"),
    ("desk", 0, "pen_holder"): ("top", 30.0, "left"), ("desk", 0, "stapler"): ("right", 20.0, "left"),
    ("desk", 0, "tape_dispenser"): ("top", 20.0, "right"), ("desk", 1, "keyboard"): ("top", 10.0, "left"),
    ("desk", 1, "pen"): ("right", 20.0, "left"), ("desk", 1, "pen_holder"): ("top", 30.0, "left"),
    ("desk", 1, "phone"): ("top", 10.0, "right"), ("desk", 1, "stapler"): ("right", 20.0, "left"),
    ("desk", 1, "tape_dispenser"): ("top", 20.0, "right"), ("kitchen_tidy", 0, "pot"): ("right", 10.0, "right"),
    ("kitchen_tidy", 1, "pot"): ("right", 10.0, "right"), ("toys", 0, "chick"): ("top", 10.0, "left"),
    ("toys", 0, "matryoshka_doll"): ("top", 20.0, "right"), ("toys", 0, "wooden_toy"): ("top", 20.0, "right"),
    ("toys", 1, "chick"): ("top", 20.0, "left"), ("toys", 1, "wooden_toy"): ("top", 20.0, "right"),
}


def test_the_recorded_requal_objects_replay_their_tilted_plans():
    """The 24 objects four scene audits rejected with no side reaching them, and the shared planner's answers to
    the audit's tilted approaches on each audit's recorded depth world, from the home joints, with each arm: every
    approach up to 20 degrees, and the 30 degree ones for the pen holder, the one object none of those reached."""
    fixture = json.loads((ROOT / "tests/data/robodojo/play-tilt-requal.json").read_text())
    assert {(item["scene"], item["seed"], item["object"]) for item in fixture["objects"]} == set(TILTED_WINNERS)

    def no_side(request):
        return {"candidates": [{"arm": arm, "source_index": index, "reachable": False, "rejection_reason": "ik_failed"}
                               for arm in ("left", "right") for index in range(12)]}

    for item in fixture["objects"]:
        key = (item["scene"], item["seed"], item["object"])
        candidates = audit_worker.tilted_approaches(item["record"])
        assert len(candidates) == 6 and len(item["plans"]) // 2 == (6 if item["object"] == "pen_holder" else 4)
        recorded = {}
        for plan in item["plans"]:
            # The geometry the planner was asked is the geometry the audit asks.
            index = next(i for i, candidate in enumerate(candidates) if (candidate["side"], candidate["tilt_deg"],
                         candidate["azimuth_deg"]) == (plan["side"], plan["tilt_deg"], plan["azimuth_deg"]))
            assert candidates[index]["pose_xyzw"] == pytest.approx(plan["link6_pose_xyzw"], abs=1e-8), key
            recorded[(plan["arm"], index)] = plan

        def replay(arm, pose):
            index = next(i for i, candidate in enumerate(candidates) if candidate["pose_xyzw"] == list(pose))
            assert (arm, index) in recorded, f"{key} asked an approach no recorded plan answered"
            plan = recorded[(arm, index)]
            return {"status": plan["status"], "reason": plan["reason"]}

        reach = audit_worker.approach_reach(no_side, item["object"], item["record"], plan_path=replay)
        planned = reach["tilted"]["planned"]
        winner = TILTED_WINNERS[key]
        assert (None if planned is None else (planned["side"], planned["tilt_deg"], planned["arm"])) == winner, key
        assert reach["tilted"]["plans_tried"] <= 9
        assert audit_worker.judge_object(_object(ik=reach)) == [], key


def test_the_audit_observes_a_seed_before_it_asks_for_ik(tmp_path, monkeypatch):
    """An IK in the observed world (the shared planner's) answers only after the seed's own observation."""
    from types import SimpleNamespace

    order = []
    (tmp_path / "mug.json").write_text(json.dumps({"geometry": {"aligned_bbox": {"vertices": BOX}}}))
    placement = {"label": "mug", "section": "Rigid", "category": "cup", "category_idx": 0, "relative_plane": "Table",
                 "spawn_pos": [0.1, -0.1, 0.77], "spawn_ori_wxyz": [1.0, 0.0, 0.0, 0.0], "metadata": "mug.json"}
    session = SimpleNamespace(
        environment=SimpleNamespace(sim_step=lambda render: None), config={"asset_root": str(tmp_path)},
        source={"play_layouts": {"0": {"placements": [placement]}}, "physics_device": "cuda:0"},
        reset_physics_steps=0, align_plan=None, _proprioception=lambda: {"end_effectors": {}},
        _loaded_layout=lambda: {"layout_sha256": "a" * 64, "planned_layout_sha256": "a" * 64})
    monkeypatch.setattr(audit_worker, "object_inventory", lambda session: {"mug": {"kind": "rigid",
                                                                                   "prim_path": "/World/mug"}})
    monkeypatch.setattr(audit_worker, "poses", lambda session, inventory: {"mug": ([0.1, -0.1, 0.77],
                                                                                    [1.0, 0.0, 0.0, 0.0])})
    monkeypatch.setattr(audit_worker, "gripper_hulls", lambda session: {"left": [[-0.4, -0.4], [-0.3, -0.4],
                                                                                 [-0.3, -0.3]]})
    monkeypatch.setattr(audit_worker, "camera_images", lambda session, destination: order.append("observe") or {
        "observation_id": "o", "cameras": {}})
    monkeypatch.setattr(audit_worker, "approach_reach", lambda plan, label, record, criteria, plan_path: order.append(
        f"ik:{label}") or {"reachable_arms": ["left"]})
    record = audit_worker.audit_seed(session, seed=0, pass_name="session", output=tmp_path / "seed-0", contacts=False)
    assert order == ["observe", "ik:mug"] and record["objects"]["mug"]["ik"] == {"reachable_arms": ["left"]}


# --- the audit record and its admission ---------------------------------------------------------

def _kernel(sha="k" * 64):
    return {"schema": robodojo_play_kernel.PLAY_KERNEL_SCHEMA, "commit": "c" * 40, "sha256": sha,
            "benchmark_paths": 57, "play_paths": list(robodojo_play_kernel.PLAY_KERNEL_PATHS)}


def _seed(seed, passed=True, layout=None):
    return {"passed": passed, "failures": [] if passed else ["session:mug:ik_reachability"], "object_count": 12,
            "layout_sha256": layout or f"{seed:064d}", "passes": {"session": {"passed": passed}}}


def _record(scene="breakfast", seeds=(0, 1, 2), failed=(), kernel="k" * 64, **changes):
    definition = registry.scene_definition(SET, scene)
    record = build_record(definition=definition, seeds={seed: _seed(seed, seed not in failed) for seed in seeds},
                          reports={"session": {"graphics_compatibility": {"g": 1}, "device_binding": {"d": 1}}},
                          play_kernel=_kernel(kernel), source_commit="c" * 40,
                          execution={"required_host": "test-host"}, criteria=audit_worker.CRITERIA,
                          created_unix_s=1.0)
    record.update(changes)
    return record


def test_an_audit_admits_its_accepted_seeds_for_the_audited_play_kernel_only(tmp_path):
    record = _record(failed=(2,))
    assert record["accepted_seeds"] == [0, 1] and record["verdict"] is None and record["model_calls"] == 0
    assert record["physics_device"] == "cpu" and record["passes"] == ["session"]
    admit = dict(scene_set=SET, scene="breakfast", play_kernel_sha256="k" * 64)
    assert play_scene_admission(record, seed=0, **admit)[0]
    assert play_scene_admission(record, seed=1, layout_sha256=f"{1:064d}", **admit)[0]
    assert not play_scene_admission(record, seed=1, layout_sha256="f" * 64, **admit)[0]
    passed, detail = play_scene_admission(record, seed=2, **admit)
    assert not passed and detail["seed"]["failures"] == ["session:mug:ik_reachability"]
    for change in ({"scene": "desk"}, {"scene_set": "tabletop-v1"}, {"play_kernel_sha256": "e" * 64},
                   {"play_kernel_sha256": None}):
        assert not play_scene_admission(record, seed=0, **{**admit, **change})[0]
    tampered = {**record, "accepted_seeds": [0, 1, 2]}
    assert play_scene_admission(tampered, seed=2, **admit)[1]["audit_sha256_valid"] is False
    other_revision = _record(robodojo_revisions={**REVISIONS, "assets": "0" * 40})
    other_revision["audit_sha256"] = robodojo_play_audit.digest(
        {key: value for key, value in other_revision.items() if key != "audit_sha256"})
    assert not play_scene_admission(other_revision, seed=0, **admit)[0]
    # A changed scene YAML needs a new audit of that scene (its source binding changed).
    tree = _scene_tree(tmp_path / "tree")
    assert play_scene_admission(record, seed=0, root=tree, **admit)[0]
    (tree / registry.scene_definition(SET, "breakfast").scene_file).write_bytes(
        (ROOT / registry.scene_definition(SET, "breakfast").scene_file).read_bytes() + b"# edited\n")
    passed, detail = play_scene_admission(record, seed=0, root=tree, **admit)
    assert not passed and detail["scene_source_sha256"] != detail["current_scene_source_sha256"]
    # The RoboTwin rule admits nothing a RoboDojo audit carries.
    from ora.orchestrator.backends.robotwin.play_audit import play_scene_admission as robotwin_admission
    assert not robotwin_admission(record, scene_set=SET, scene="breakfast", seed=0, task_config=None,
                                  revision=None)[0]


def test_one_scenes_edit_retires_only_its_audits_and_a_code_edit_retires_every_scene(tmp_path):
    from ora.orchestrator.robodojo_kernel import WORKER_KERNEL_PATHS

    # A second ORA tree holding the play kernel's code and the scene set's data.
    tree = _scene_tree(tmp_path / "tree")
    for relative in (*WORKER_KERNEL_PATHS, *robodojo_play_kernel.PLAY_KERNEL_PATHS):
        (tree / relative).parent.mkdir(parents=True, exist_ok=True)
        (tree / relative).write_bytes((ROOT / relative).read_bytes())
    kernel = robodojo_play_kernel.play_kernel_sha256(project_root=tree)
    assert kernel == robodojo_play_kernel.play_kernel_sha256()
    records = {name: _record(name, seeds=(0,), kernel=kernel) for name in SCENES}

    def admitted():
        current = robodojo_play_kernel.play_kernel_sha256(project_root=tree)
        return {name for name, record in records.items()
                if play_scene_admission(record, scene_set=SET, scene=name, seed=0, play_kernel_sha256=current,
                                        root=tree)[0]}

    assert admitted() == set(SCENES)
    # Scene data is outside the play kernel: one scene's YAML or entry edit retires that scene only.
    desk_yaml = tree / registry.scene_definition(SET, "desk").scene_file
    original = desk_yaml.read_bytes()
    desk_yaml.write_bytes(original.replace(b"ylim: [0.050, 0.055]", b"ylim: [0.045, 0.050]"))
    assert robodojo_play_kernel.play_kernel_sha256(project_root=tree) == kernel
    assert admitted() == set(SCENES) - {"desk"}
    desk_yaml.write_bytes(original)
    _edit_entry(tree, "breakfast", "settle_steps", 2500)
    assert admitted() == set(SCENES) - {"breakfast"}
    _edit_entry(tree, "breakfast", "settle_steps", 3000)
    _edit_entry(tree, "toys", "description", "Toys.")
    assert admitted() == set(SCENES)
    # A code edit to a play module changes the play kernel: every scene's audit is retired.
    module = tree / "ora/environment/backends/robodojo/play_scenes/lifecycle.py"
    module.write_bytes(module.read_bytes() + b"# edited\n")
    assert robodojo_play_kernel.play_kernel_sha256(project_root=tree) != kernel
    assert admitted() == set()


def test_a_seed_passes_only_when_every_pass_audited_it_on_one_layout(tmp_path):
    scene_root = tmp_path / SET / "desk"
    session = {"seeds": {"0": {"passed": True, "failures": [], "layout_sha256": "a" * 64, "object_count": 9,
                               "cameras": {"observation_id": "o", "cameras": {"head": {"path": "o/head/rgb.png",
                                                                                       "std": 40.0}}}},
                         "1": {"passed": False, "failures": ["mug:settle"], "layout_sha256": "b" * 64}}}
    contacts = {"seeds": {"0": {"passed": True, "failures": [], "layout_sha256": "a" * 64}}, "error": "crashed"}
    passes = ("session", "contacts")
    merged = merge_seed(0, {"session": session, "contacts": contacts}, "a" * 64, passes, tmp_path / SET, scene_root)
    assert merged["passed"] and merged["object_count"] == 9
    assert merged["cameras"]["head"]["path"] == "desk/pass-session/seed-0/observations/o/head/rgb.png"
    failed = merge_seed(1, {"session": session, "contacts": contacts}, "b" * 64, passes, tmp_path / SET, scene_root)
    assert failed["failures"] == ["session:mug:settle", "contacts:not_audited (crashed)"]
    drifted = merge_seed(0, {"session": session, "contacts": contacts}, "c" * 64, passes, tmp_path / SET, scene_root)
    assert not drifted["passed"] and "layout_differs_between_passes_or_from_the_generator" in drifted["failures"]


def test_the_audit_driver_runs_each_pass_once_per_scene_and_writes_its_records(tmp_path, monkeypatch):
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    monkeypatch.setenv("ROBODOJO_ASSET_ROOT", str(tmp_path / "Assets"))
    layouts_by_seed = {"0": "a" * 64, "1": "b" * 64}
    monkeypatch.setattr(robodojo_play_audit, "audit_runtime", lambda definition, **kwargs: {
        "play_layouts": dict(layouts_by_seed), "resource_binding": {"slot_id": "gpu3",
                                                                    "worker_environment": {"CUDA_VISIBLE_DEVICES": "3"}}})
    calls = []

    def runner(definition, pass_name, seeds, runtime, *, output, commit):
        calls.append((definition.name, pass_name, tuple(seeds)))
        output.mkdir(parents=True)
        report = {"graphics_compatibility": {"g": 1}, "device_binding": {"d": 1}, "seeds": {
            str(seed): {"passed": not (definition.name == "desk" and seed == 1 and pass_name == "contacts"),
                        "failures": ["mug:penetration"] if (definition.name == "desk" and seed == 1
                                                            and pass_name == "contacts") else [],
                        "layout_sha256": layouts_by_seed[str(seed)], "object_count": 9} for seed in seeds}}
        (output / audit_worker.WORKER_REPORT_NAME).write_text(json.dumps(report))
        return 0

    report = robodojo_play_audit.audit(SET, ["breakfast", "desk"], [0, 1], resource_profile="robodojo-gpu3",
                                       destination=tmp_path / "audit", runner=runner,
                                       sources={"commit": "c" * 40, "play_kernel": _kernel()})
    assert calls == [("breakfast", "session", (0, 1)), ("desk", "session", (0, 1)), ("desk", "contacts", (0, 1))]
    written = json.loads((tmp_path / "audit" / SET / "desk.json").read_text())
    assert written["accepted_seeds"] == [0] and written["passes"] == ["session", "contacts"]
    assert report["records"]["desk"]["rejected"] == {"1": ["contacts:mug:penetration"]}
    assert written["execution"]["worker_reports"]["contacts"]["path"] == f"desk/pass-contacts/{audit_worker.WORKER_REPORT_NAME}"
    assert play_scene_admission(written, scene_set=SET, scene="desk", seed=0, play_kernel_sha256="k" * 64,
                                layout_sha256="a" * 64)[0]


# --- what a host offers --------------------------------------------------------------------------

def _write(root: Path, stamp: str, record: dict) -> Path:
    path = root / stamp / record["scene_set"] / f"{record['scene']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


@pytest.fixture
def audits(tmp_path):
    """Audits of the current play kernel: the newer one binds seed 1; a failed or stale record admits nothing."""
    kernel = robodojo_play_kernel.play_kernel_sha256()
    root = tmp_path / "audits"
    _write(root, "20261019T100000Z-1", _record(seeds=(0, 1, 2), failed=(2,), kernel=kernel))
    _write(root, "20261019T110000Z-2", _record(seeds=(1, 3), kernel=kernel))
    _write(root, "20261019T110000Z-2", _record("toys", seeds=(4,), kernel=kernel))
    _write(root, "20261019T120000Z-3", _record("desk", seeds=(5,), kernel="e" * 64))
    return root


def test_robodojo_offers_only_the_audited_seeds_of_its_scene_sets(audits, tmp_path):
    scenes = list_play_scenes("robodojo", play_audit_root_dir=audits)
    assert [(scene.task, scene.seed) for scene in scenes] == [("breakfast", 0), ("breakfast", 1), ("breakfast", 3),
                                                              ("toys", 4)]
    newer = next(scene for scene in scenes if scene.seed == 1)
    assert newer.qualification["path"].startswith("20261019T110000Z-2/") and newer.scene_set == SET
    assert newer.qualification["kind"] == "robodojo_play_scene_audit"
    assert newer.qualification["layout_sha256"] == f"{1:064d}"
    record = newer.record()
    assert record["scene_id"] == "robodojo:isaac-tabletop-v1:breakfast:s1"
    assert record["label"] == "RoboDojo isaac-tabletop-v1 · breakfast table · seed 1"
    assert "description" not in record  # the operator's text never enters the plan
    assert list_play_scenes("robodojo", scene_set=SET, task="toys", play_audit_root_dir=audits)[0].seed == 4
    assert list_play_scenes("robodojo", play_audit_root_dir=tmp_path / "none") == []
    with pytest.raises(PlaySceneError, match="ora play audit --env robodojo --scene-set isaac-tabletop-v1"):
        play_scenes.draw_play_scene("robodojo", random.Random(0), play_audit_root_dir=tmp_path / "none")
    with pytest.raises(PlaySceneError, match="unknown RoboDojo play scene set"):
        list_play_scenes("robodojo", scene_set="tabletop-v1", play_audit_root_dir=audits)
    draws = {play_scenes.draw_play_scene("robodojo", random.Random(seed), play_audit_root_dir=audits).task
             for seed in range(20)}
    assert draws == {"breakfast", "toys"}


def test_build_play_spec_binds_the_scene_runtime_and_names_the_audit_record(audits):
    spec = build_play_spec("robodojo", host="test-host", rng=random.Random(2), play_audit_root_dir=audits,
                           resource_profile="robodojo-gpu3")
    scene = spec["play"]["scene"]
    runtime = spec["runtime"]
    assert scene["scene_set"] == SET and spec["task"] == scene["task"] == runtime["task_variant"]
    definition = registry.scene_definition(SET, scene["task"])
    assert runtime["play_scene_set"] == SET and runtime["physics_device"] == definition.physics_device
    assert (runtime["cuda_visible_devices"], runtime["renderer_gpu"]) == ("3", 3)
    assert runtime["native_action_limit"] == registry.PLAY_NATIVE_ACTION_LIMIT and runtime["eval_seed"] == 0
    assert runtime["expected_task_language"] == registry.scene_language(definition)
    assert not {"layout_id", "layout_file", "layout_sha256", "infrastructure_qualification"} & set(runtime)
    # The operator's capture at its highest rate (benchmark plans keep the default 10 fps).
    assert runtime["execution_capture"] == registry.PLAY_EXECUTION_CAPTURE == {"enabled": True, "fps": 30}
    assert spec["qualification"]["manifest"].endswith(f"-{scene['qualification']['sha256'][:16]}.json")
    registry.validate_play_runtime({key: value for key, value in runtime.items() if key != "resource_profile"})


def _play(audits, workspace, monkeypatch, **kwargs):
    from ora.app.play import play_spec
    from ora.orchestrator.spec.inputs import provider_policy

    monkeypatch.setattr(play_scenes, "play_audit_root", lambda: audits)
    spec, chosen = play_spec("robodojo", policy=provider_policy(None), host="test-host", count=4,
                             rng=random.Random(3), draw={"draw_seed": 3}, **kwargs)
    play_scenes.write_play_spec(spec, chosen, output=workspace / "play.json")
    return spec, chosen


def test_ora_play_resolves_a_robodojo_play_plan_and_preflight_readmits_its_seeds(audits, workspace, monkeypatch):
    from ora.orchestrator.preflight import gates as preflight

    monkeypatch.setattr(preflight, "resolve_workspace_root", lambda project_root: workspace)
    spec, chosen = _play(audits, workspace, monkeypatch, scene="breakfast")  # the scene implies its set
    assert chosen.scene_set == SET and chosen.task == "breakfast"
    assert {scene["seed"] for scene in spec["play"]["scenes"]} <= {0, 1, 3}
    plan = resolve_run_spec(spec)
    runtime = plan["runtime"]
    assert runtime["play_scene_set"] == SET and runtime["physics_device"] == "cpu"
    assert set(runtime["play_layouts"]) == {str(scene["seed"]) for scene in plan["play"]["scenes"]}
    assert runtime["play_kernel_sha256"] == robodojo_play_kernel.play_kernel_sha256()
    assert runtime["play_scene_source_sha256"] == registry.source_binding(SET, "breakfast")["sha256"]
    assert "infrastructure_qualification" not in runtime and runtime["eval_seed"] == 0
    assert (runtime["execution_capture"]["enabled"], runtime["execution_capture"]["fps"]) == (True, 30)
    assert plan["task"] == "breakfast" and plan["seed"] == plan["play"]["scene"]["seed"]
    # The session names 400 robot decisions; free play on the synchronous surface keeps at most 100.
    assert spec["budgets"]["max_robot_decisions"] == 400 and plan["budgets"]["max_robot_decisions"] == 100
    # The default RoboDojo surface, without the task checker and the lifecycle tool; no task contract.
    tools = plan["policy"]["tool_allowlist"]
    assert not {"check_task", "finish_episode"} & set(tools) and {"grasp_plan", "move_to", "observe"} <= set(tools)
    assert plan["policy"]["tool_surface"]["profile"] == "robodojo_placement_toolkit_v6"
    assert plan["policy"]["task_contract"]["delivered"] is False
    assert plan["task_instruction"] != runtime["expected_task_language"]
    # The default instance's shared motion base: the worker starts its planner and binds its timing (shared_motion).
    from ora.agent.composition.baseline import default_baseline_id, load_native_baseline

    instance = load_native_baseline(ROOT, default_baseline_id("robodojo"))["instances"]["robodojo"]["runtime"]
    assert {runtime[key]["implementation"] for key in ("motion_planner", "motion_timing")} == {"curobo_v2",
                                                                                               "toppra_ruckig"}
    assert (runtime["motion_planner"], runtime["motion_timing"]) == (instance["motion_planner"],
                                                                     instance["motion_timing"])
    # Preflight re-admits every scene of the session against the play kernel at the plan's commit.
    monkeypatch.setattr(robodojo_play_kernel, "play_kernel_sha256",
                        lambda commit=None, project_root=None: runtime["play_kernel_sha256"])
    check = preflight._qualification(plan)
    assert check["passed"] is True and check["detail"]["accepted"] is True
    # ... and against the scene's own source binding, which the plan carries.
    stale = copy.deepcopy(plan)
    stale["runtime"]["play_scene_source_sha256"] = "e" * 64
    refused = preflight._qualification(stale)
    assert refused["passed"] is False and refused["detail"]["plan_play_scene_source_sha256"] == "e" * 64
    monkeypatch.setattr(robodojo_play_kernel, "play_kernel_sha256", lambda commit=None, project_root=None: "e" * 64)
    assert preflight._qualification(plan)["passed"] is False


def test_bind_play_runtime_binds_the_robodojo_scene_and_no_benchmark_may_carry_it(audits):
    scenes = [scene.record() for scene in list_play_scenes("robodojo", task="breakfast", play_audit_root_dir=audits)]
    runtime = {"kind": "robodojo", "task_variant": "breakfast", "physics_device": "cpu", "play_scene_set": SET}
    bound = bind_play_runtime(runtime, scene=scenes[0], scenes=scenes)
    assert bound["play_layouts"] == {str(item["seed"]): item["qualification"]["layout_sha256"] for item in scenes}
    assert bound["play_kernel_sha256"] == scenes[0]["qualification"]["play_kernel_sha256"]
    assert bound["play_scene_source_sha256"] == scenes[0]["qualification"]["scene_source_sha256"] == \
        registry.source_binding(SET, "breakfast")["sha256"]
    assert bind_play_runtime(bound, scene=scenes[0], scenes=scenes) == bound
    with pytest.raises(RunSpecError, match="need episode_mode play"):
        bind_play_runtime({**runtime, "play_scene_set": None, "play_layouts": bound["play_layouts"]}, scene=None)
    with pytest.raises(RunSpecError, match="play_scene_set loads a play scene"):
        bind_play_runtime(runtime, scene=None)
    with pytest.raises(RunSpecError, match="runs on cpu physics"):
        bind_play_runtime({**runtime, "physics_device": "cuda:0"}, scene=scenes[0], scenes=scenes)
    with pytest.raises(RunSpecError, match="audited seed of one scene"):
        bind_play_runtime(runtime, scene=scenes[0], scenes=[scenes[0], {**scenes[1], "task": "toys"}])
    stale = copy.deepcopy(scenes[1])
    stale["qualification"]["play_kernel_sha256"] = "e" * 64
    with pytest.raises(RunSpecError, match="one audit of one play kernel"):
        bind_play_runtime(runtime, scene=scenes[0], scenes=[scenes[0], stale])
    other_source = copy.deepcopy(scenes[1])
    other_source["qualification"]["scene_source_sha256"] = "e" * 64
    with pytest.raises(RunSpecError, match="one source binding of its scene"):
        bind_play_runtime(runtime, scene=scenes[0], scenes=[scenes[0], other_source])
    with pytest.raises(RunSpecError, match="audited play binding"):
        bind_play_runtime({**runtime, "play_scene_source_sha256": "e" * 64}, scene=scenes[0], scenes=scenes)
    with pytest.raises(RunSpecError, match="one of ORA's scene sets"):
        bind_play_runtime(runtime, scene={**scenes[0], "scene_set": None})


def test_the_play_runtime_rule_pins_what_the_benchmark_pins_and_admits_only_the_scene(audits):
    from ora.environment.definitions import EnvironmentDefinitionError, validate_runtime_extension

    spec = build_play_spec("robodojo", host="test-host", rng=random.Random(2), play_audit_root_dir=audits,
                           resource_profile="robodojo-gpu3")
    runtime = {key: value for key, value in spec["runtime"].items() if key != "resource_profile"}
    validate_runtime_extension(runtime)  # routed to robodojo_play_scenes.validate_play_runtime
    for change, message in (({"native_action_limit": 1050}, "native_action_limit"),
                            ({"task_variant": "make_toast"}, "has no scene"),
                            ({"expected_task_language": "stack the bowls"}, "expected_task_language"),
                            ({"layout_id": 0}, "unsupported RoboDojo play runtime fields"),
                            ({"required_revision": "0" * 40}, "reviewed code/fork/asset revisions"),
                            ({"renderer_gpu": 4}, "renderer_gpu")):
        with pytest.raises(EnvironmentDefinitionError, match=message):
            validate_runtime_extension({**runtime, **change})
    with pytest.raises(RoboDojoBenchmarkError):
        registry.validate_play_runtime({**runtime, "play_layouts": {"x": "0"}})
    # Without the set the official rule applies: a play scene is no RoboDojo task.
    with pytest.raises(EnvironmentDefinitionError, match="reviewed fixed-layout tasks"):
        validate_runtime_extension({key: value for key, value in runtime.items() if key != "play_scene_set"})


def test_the_scenes_of_one_robodojo_session_share_its_scene_set():
    from ora.orchestrator.spec.play import _require_backend_scenes

    first = {"seed": 0, "scene_set": SET}
    _require_backend_scenes({"kind": "robodojo", "play_scene_set": SET}, [first, {**first, "seed": 3}])
    with pytest.raises(RunSpecError, match="one of ORA's scene sets"):
        _require_backend_scenes({"kind": "robodojo"}, [first])
    with pytest.raises(RunSpecError, match="one of ORA's scene sets"):
        _require_backend_scenes({"kind": "robodojo", "play_scene_set": SET}, [first, {**first, "scene_set": None}])


def test_the_operator_sees_the_scene_description_and_the_plan_never_carries_it(audits, workspace, monkeypatch):
    from ora.app.main import build_parser
    from ora.app.play import scene_description

    spec, chosen = _play(audits, workspace, monkeypatch)
    description = scene_description(spec["play"]["scene"])
    assert description == registry.scene_definition(SET, chosen.task).description
    assert description not in json.dumps(spec)
    args = build_parser().parse_args(["play", "--env", "robodojo", "--scene", "toys"])
    assert (args.environment, args.scene, args.scene_set) == ("robodojo", "toys", None)
