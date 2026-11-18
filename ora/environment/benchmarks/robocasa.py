"""Host-safe RoboCasa365 identity: pins, the admitted task roster and public context.

This module never imports robosuite or robocasa. The worker binds the same
constants under the RoboCasa interpreter, and the orchestrator validates a
RunSpec's runtime block against them before any simulator starts.
"""

from __future__ import annotations

from typing import Any, Mapping


class RoboCasaBenchmarkError(ValueError):
    """A RoboCasa runtime block does not match the admitted identity."""


REVISIONS = {
    "robocasa": "a9e152e2eea16d342dac40e4577fb4c68b932854",
    "robosuite": "5ce6643f3092639d08f7b0f90ed1c6a84f50552c",
}
PACKAGE_VERSIONS = {"robocasa": "1.0.1", "robosuite": "1.5.2", "mujoco": "3.3.1"}

# RoboCasa365 evaluates foundation-model target tasks in the ``target`` split:
# layout/style pairs 1..10 and the target object instances.
SPLIT = "target"
ROBOT = "PandaOmron"
EMBODIMENT_ID = "panda_omron"
CONTROL_FREQ_HZ = 20
PHYSICS_TIMESTEP_S = 0.002
IMAGE_SIZE = 256

# Public camera name -> RoboCasa camera. ``agentview`` is the left over-shoulder
# camera on the torso support; ``wrist`` is the eye-in-hand camera. Both move
# with the base, so calibration is published per observation.
PUBLIC_CAMERAS = {
    "agentview": "robot0_agentview_left",
    "wrist": "robot0_eye_in_hand",
}
MAIN_CAMERA = "agentview"

# The admitted roster: RoboCasa365 ``target50`` (``TARGET_TASKS`` of robocasa a9e152e2
# ``dataset_registry.py``: 18 atomic_seen, 16 composite_seen, 16 composite_unseen).
# The first three (user decision 2026-10-18) state the upstream success predicate
# in words and carry a reviewed public completion contract. The other 47 were
# admitted on 2026-11-05 (user: run all of RoboCasa365) without a contract: the
# policy receives the episode's own language, as RoboCasa's evaluation does, and
# ``official_success`` only names the task's ``_check_success``; their
# ``mobile_base_required`` is a coarse label (composite tasks), not an audit. The
# predicate itself stays inside the worker.
# ``horizon_control_steps`` is the task's evaluation horizon from
# robocasa a9e152e2 ``robocasa/utils/dataset_registry.py`` (the value RoboCasa
# attaches to each task's dataset metadata), not the Kitchen constructor's
# 1000-step default.
TASKS: dict[str, dict[str, Any]] = {
    "PickPlaceCounterToCabinet": {
        "horizon_control_steps": 750,
        "task_set": "atomic_seen",
        "skill": "pick_and_place",
        "official_success": (
            "the object is inside the cabinet and the gripper is far from it"
        ),
        "mobile_base_required": False,
    },
    "OpenDrawer": {
        "horizon_control_steps": 750,
        "task_set": "atomic_seen",
        "skill": "articulated_open",
        "official_success": "every joint of the target drawer is at least 0.95 open",
        "mobile_base_required": False,
    },
    "NavigateKitchen": {
        "horizon_control_steps": 450,
        "task_set": "atomic_seen",
        "skill": "navigation",
        "official_success": (
            "the base centre is within 0.20 m of the target fixture's approach "
            "pose and faces it (cosine of the yaw error at least 0.98)"
        ),
        "mobile_base_required": True,
    },
    "CloseBlenderLid": {
        "horizon_control_steps": 900,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_blender.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "CloseFridge": {
        "horizon_control_steps": 900,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_doors.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "CloseToasterOvenDoor": {
        "horizon_control_steps": 450,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_doors.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "CoffeeSetupMug": {
        "horizon_control_steps": 600,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_coffee.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "OpenCabinet": {
        "horizon_control_steps": 1050,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_doors.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "OpenStandMixerHead": {
        "horizon_control_steps": 450,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_stand_mixer.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "PickPlaceCounterToStove": {
        "horizon_control_steps": 600,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_pick_place.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "PickPlaceDrawerToCounter": {
        "horizon_control_steps": 750,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_pick_place.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "PickPlaceSinkToCounter": {
        "horizon_control_steps": 900,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_pick_place.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "PickPlaceToasterToCounter": {
        "horizon_control_steps": 600,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_pick_place.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "SlideDishwasherRack": {
        "horizon_control_steps": 450,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_drawer.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "TurnOffStove": {
        "horizon_control_steps": 750,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_stove.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "TurnOnElectricKettle": {
        "horizon_control_steps": 450,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_electric_kettle.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "TurnOnMicrowave": {
        "horizon_control_steps": 450,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_microwave.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "TurnOnSinkFaucet": {
        "horizon_control_steps": 600,
        "task_set": "atomic_seen",
        "skill": "atomic",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/atomic/kitchen_sink.py); no reviewed contract",
        "mobile_base_required": False,
    },
    "DeliverStraw": {
        "horizon_control_steps": 2550,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/serving_beverages/deliver_straw.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "GetToastedBread": {
        "horizon_control_steps": 3000,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/toasting_bread/get_toasted_bread.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "KettleBoiling": {
        "horizon_control_steps": 1500,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/brewing/kettle_boiling.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "LoadDishwasher": {
        "horizon_control_steps": 1800,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/loading_dishwasher/load_dishwasher.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "PackIdenticalLunches": {
        "horizon_control_steps": 3900,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/packing_lunches/pack_identical_lunches.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "PreSoakPan": {
        "horizon_control_steps": 2400,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/washing_dishes/pre_soak_pan.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "PrepareCoffee": {
        "horizon_control_steps": 1800,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/brewing/prepare_coffee.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "RinseSinkBasin": {
        "horizon_control_steps": 1350,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/cleaning_sink/rinse_sink_basin.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "ScrubCuttingBoard": {
        "horizon_control_steps": 1200,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/sanitizing_cutting_board/scrub_cutting_board.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "SearingMeat": {
        "horizon_control_steps": 4350,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/frying/searing_meat.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "SetUpCuttingStation": {
        "horizon_control_steps": 2400,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/slicing_meat/set_up_cutting_station.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "StackBowlsCabinet": {
        "horizon_control_steps": 2100,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/organizing_dishes_and_containers/stack_bowls_cabinet.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "SteamInMicrowave": {
        "horizon_control_steps": 2100,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/steaming_food/steam_in_microwave.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "StirVegetables": {
        "horizon_control_steps": 2400,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/sauteing_vegetables/stir_vegetables.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "StoreLeftoversInBowl": {
        "horizon_control_steps": 2550,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/storing_leftovers/store_leftovers_in_bowl.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "WashLettuce": {
        "horizon_control_steps": 1650,
        "task_set": "composite_seen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/making_salads/wash_lettuce.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "ArrangeBreadBasket": {
        "horizon_control_steps": 4350,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/setting_the_table/arrange_bread_basket.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "ArrangeTea": {
        "horizon_control_steps": 2250,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/brewing/arrange_tea.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "BreadSelection": {
        "horizon_control_steps": 1950,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/making_toast/bread_selection.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "CategorizeCondiments": {
        "horizon_control_steps": 1650,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/arranging_condiments/categorize_condiments.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "CuttingToolSelection": {
        "horizon_control_steps": 1200,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/chopping_vegetables/cutting_tool_selection.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "GarnishPancake": {
        "horizon_control_steps": 2700,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/garnishing_dishes/garnish_pancake.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "GatherTableware": {
        "horizon_control_steps": 2250,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/arranging_cabinets/gather_tableware.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "HeatKebabSandwich": {
        "horizon_control_steps": 2700,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/preparing_sandwiches/heat_kebab_sandwich.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "MakeIceLemonade": {
        "horizon_control_steps": 3000,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/adding_ice_to_beverages/make_ice_lemonade.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "PanTransfer": {
        "horizon_control_steps": 1800,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/serving_food/pan_transfer.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "PortionHotDogs": {
        "horizon_control_steps": 2250,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/portioning_meals/portion_hot_dogs.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "RecycleBottlesByType": {
        "horizon_control_steps": 2850,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/organizing_recycling/recycle_bottles_by_type.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "SeparateFreezerRack": {
        "horizon_control_steps": 2400,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/managing_freezer_space/separate_freezer_rack.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "WaffleReheat": {
        "horizon_control_steps": 4050,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/reheating_food/waffle_reheat.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "WashFruitColander": {
        "horizon_control_steps": 3150,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/washing_fruits_and_vegetables/wash_fruit_colander.py); no reviewed contract",
        "mobile_base_required": True,
    },
    "WeighIngredients": {
        "horizon_control_steps": 3000,
        "task_set": "composite_unseen",
        "skill": "composite",
        "official_success": "RoboCasa's own _check_success (robocasa/environments/kitchen/composite/measuring_ingredients/weigh_ingredients.py); no reviewed contract",
        "mobile_base_required": True,
    },
}
ACTION_CODECS = ("runtime_action_v3", "motion_action_v3")


def public_context() -> dict[str, Any]:
    """The public environment context the runtime descriptor is resolved from."""

    return {
        "schema": "ora.public-environment-context.v1",
        "environment_id": "robocasa",
        "embodiment_id": EMBODIMENT_ID,
        "robot_models": [ROBOT],
        "controller": "JOINT_POSITION_IK_PATH+BASE_VELOCITY_SERVO",
        "public_cameras": list(PUBLIC_CAMERAS),
    }


def validate_runtime_extension(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the RoboCasa-specific fields of a runtime block."""

    if not isinstance(runtime, Mapping):
        raise RoboCasaBenchmarkError("RoboCasa runtime must be an object")
    task = runtime.get("task")
    if task not in TASKS:
        raise RoboCasaBenchmarkError(
            f"RoboCasa task is not admitted: {task!r} (admitted: {sorted(TASKS)})"
        )
    if runtime.get("split", SPLIT) != SPLIT:
        raise RoboCasaBenchmarkError("RoboCasa runs use the target split")
    seed = runtime.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RoboCasaBenchmarkError("RoboCasa seed must be a non-negative integer")
    revisions = runtime.get("required_revisions", REVISIONS)
    if dict(revisions) != REVISIONS:
        raise RoboCasaBenchmarkError("RoboCasa source revisions differ from the pins")
    codec = runtime.get("action_codec", "motion_action_v3")
    if codec not in ACTION_CODECS:
        raise RoboCasaBenchmarkError(f"unsupported RoboCasa action codec: {codec!r}")
    return {
        "task": task,
        "split": SPLIT,
        "seed": int(seed),
        "required_revisions": dict(REVISIONS),
        "action_codec": codec,
    }
