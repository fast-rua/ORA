"""Host-safe RoboLab identity: pin, the admitted task roster and public context.

No simulator imports. The worker binds the same constants under the RoboLab
interpreter (Isaac Sim 5.1 / IsaacLab 2.3.2.post1); the orchestrator checks a
runtime block against them before any simulator starts.
"""

from __future__ import annotations

from typing import Any, Mapping


class RoboLabBenchmarkError(ValueError):
    """A RoboLab runtime block does not match the admitted identity."""


REVISION = "ad45d4f974725d020f82c2b0d77d78533aeba2b3"
PACKAGES = {"isaacsim": "5.1.0.0", "isaaclab": "2.3.2.post1", "warp-lang": "1.11.0"}
EMBODIMENT_ID = "droid_franka_robotiq"
ROBOT_MODEL = "Franka_Panda_Robotiq_2F85"
CONTROLLER = "ROBOLAB_ABS_IK_CARTESIAN_SERVO"
# 120 Hz physics, one native action every 8 physics steps (15 Hz).
PHYSICS_DT = 1 / 120
DECIMATION = 8
CONTROL_FREQ_HZ = 15
# Public camera name -> RoboLab scene camera. ``agentview`` is the fixed left
# over-shoulder camera, ``wrist`` the Robotiq-mounted one.
PUBLIC_CAMERAS = {"agentview": "over_shoulder_left_camera", "wrist": "wrist_cam"}
MAIN_CAMERA = "agentview"
# The admitted roster: all 120 RoboLab-120 tasks (user decision 2026-11-05;
# three were admitted on 2026-10-18 and keep their reviewed text). The block is
# generated from the pinned source by ``ora dev robolab-tasks generate``
# (orchestrator/backends/robolab/task_catalog.py): the upstream default
# instruction, episode_length_s and, as native_action_limit, that length at
# 15 Hz, and a plain-language paraphrase of the success termination (written
# into the official verdict, never shown to the policy). Admission is not a
# reviewed contract: a task whose review is still pending its seed-0 witness
# composes without one and launch preflight refuses it.
TASKS: dict[str, dict[str, Any]] = {
    # --- generated roster: ora dev robolab-tasks generate ---
    "AnimalsInBinTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "multi_object_pick_and_place",
        "instruction": "Put the lizards in the bin",
        "official_success": "both lizard figurines rest in the grey bin, with the gripper detached",
    },
    "AppleAndYogurtInBowlTask": {
        "episode_length_s": 120, "native_action_limit": 1800, "skill": "multi_object_pick_and_place",
        "instruction": "Put the apple and yogurt in the bowl",
        "official_success": "both the apple and the yogurt cup rest in the wooden bowl, with the gripper detached",
    },
    "BBQSauceInBinTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "multi_object_pick_and_place",
        "instruction": "Put the red BBQ sauce bottles in the grey bin",
        "official_success": "both BBQ sauce bottles rest in the grey bin, with the gripper detached",
    },
    "BagelsOnPlateTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "multi_object_place_on_surface",
        "instruction": "Put the bagels on the plate",
        "official_success": "both bagels rest on the plate, supported by it, with the gripper detached",
    },
    "BananaInBowlTask": {
        "episode_length_s": 50, "native_action_limit": 750, "skill": "pick_and_place",
        "instruction": "Pick up the banana and place it in the bowl",
        "official_success": "the banana rests in the bowl, touching it, with the gripper detached",
    },
    "BananaOnPlateTask": {
        "episode_length_s": 40, "native_action_limit": 600, "skill": "place_on_surface",
        "instruction": "Pick up the banana and put it on the plate",
        "official_success": "the banana rests on the plate, supported by it, with the gripper detached",
    },
    "BananaThenRubiksCubeTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "ordered_pick_and_place",
        "instruction": "put the banana then the cube in the bowl",
        "official_success": ("the banana and then the Rubik's cube came to rest in the bowl, touching it, in that "
                             "order, with the gripper detached"),
    },
    "BananasInBinOneMoreTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "counted_pick_and_place",
        "instruction": "Put one (1) more bananas in the grey bin.",
        "official_success": ("exactly one of the three bananas that start on the table rests in the grey bin, with the "
                             "gripper detached"),
    },
    "BananasInBinThreeTotalTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "counted_pick_and_place",
        "instruction": "Make sure there are 3 (three) bananas in the grey bin.",
        "official_success": ("exactly one of the three bananas that start on the table rests in the grey bin, with the "
                             "gripper detached"),
    },
    "BananasInCrateTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "counted_pick_and_place",
        "instruction": "Put 2 bananas in the crate",
        "official_success": "exactly two of the five bananas rest in the purple crate, with the gripper detached",
    },
    "BananasOutOfBinTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "take_out",
        "instruction": "Take the bananas out",
        "official_success": "both bananas are out of the grey bin, with the gripper detached",
    },
    "BigPumpkinInBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put the bigger pumpkin in the bin",
        "official_success": "the larger pumpkin rests in the bin, with the gripper detached",
    },
    "BlackItemsInBinTask": {
        "episode_length_s": 120, "native_action_limit": 1800, "skill": "multi_object_pick_and_place",
        "instruction": "Put the black items in the grey bin",
        "official_success": ("the keyboard, the remote control, the computer mouse, the smartphone and the eyeglasses "
                             "rest in the grey bin, with the gripper detached"),
    },
    "BlockStackingOrderAgnosticTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "stacking",
        "instruction": "Stack the blocks into a tower",
        "official_success": ("the red block, the blue block, the green block and the yellow block form one stack in "
                             "any order, each touching the one below"),
    },
    "BlockStackingSpecifiedOrderTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "stacking",
        "instruction": "Stack the blocks in the order from bottom to top: red, blue, green, yellow",
        "official_success": ("the red block, the blue block, the green block and the yellow block are stacked bottom "
                             "to top in that order, each touching the one below"),
    },
    "BlocksInBinTask": {
        "episode_length_s": 150, "native_action_limit": 2250, "skill": "multi_object_pick_and_place",
        "instruction": "Sort all colored blocks into the bin",
        "official_success": ("the yellow block, the red block, the green block and the blue block rest in the grey "
                             "bin, with the gripper detached"),
    },
    "BowlInBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "put the bowl in the grey bin",
        "official_success": "the bowl rests in the grey bin, with the gripper detached",
    },
    "BowlStackingLeftOnRightTask": {
        "episode_length_s": 20, "native_action_limit": 300, "skill": "pick_and_place",
        "instruction": "Stack the left bowl on the right bowl",
        "official_success": "the left bowl rests in the right bowl, touching it, with the gripper detached",
    },
    "BowlStackingRightOnLeftTask": {
        "episode_length_s": 20, "native_action_limit": 300, "skill": "pick_and_place",
        "instruction": "Stack the right bowl on the left bowl",
        "official_success": "the right bowl rests in the left bowl, touching it, with the gripper detached",
    },
    "ButterAboveRaisinTask": {
        "episode_length_s": 40, "native_action_limit": 600, "skill": "place_on_surface",
        "instruction": "Pick up the butter box and place it on top of the raisin box",
        "official_success": "the butter box rests on the raisin box, supported by it, with the gripper detached",
    },
    "CannedFoodInBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put the canned food in the grey bin",
        "official_success": "the can of tuna rests in the grey bin, with the gripper detached",
    },
    "ClampInRightBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put the spring clamp in the right bin",
        "official_success": "the spring clamp rests in the right bin, with the gripper detached",
    },
    "CleanUpToysTask": {
        "episode_length_s": 300, "native_action_limit": 4500, "skill": "multi_object_pick_and_place",
        "instruction": "Clean up all the smaller toys and leave the birdhouse on the table",
        "official_success": ("all three Rubik's cubes, both lizard figurines, the yellow block, the red block, the "
                             "green block and the blue block rest in the grey bin, with the gripper detached"),
    },
    "ClearOrganicObjectsTask": {
        "episode_length_s": 240, "native_action_limit": 3600, "skill": "multi_object_pick_and_place",
        "instruction": "Clear away the organic objects",
        "official_success": ("both lemons, both limes, both oranges, the pomegranate, the large pumpkin, the small "
                             "pumpkin, the red onion and the avocado rest in the bin, with the gripper detached"),
    },
    "ClutterPlasticTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_pick_and_place",
        "instruction": "Put all plastic bottles away in the bin",
        "official_success": ("the white plastic bottle, the milk jug and the plastic jug rest in the bin, with the "
                             "gripper detached"),
    },
    "ClutterPumpkinTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "multi_object_pick_and_place",
        "instruction": "Put all the pumpkins away in the bin",
        "official_success": "both the large pumpkin and the small pumpkin rest in the bin, with the gripper detached",
    },
    "CoffeePotInBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put the coffee pot in the grey bin",
        "official_success": "the coffee pot rests in the grey bin, with the gripper detached",
    },
    "CondimentsInBinTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_pick_and_place",
        "instruction": "Sort the sauce condiments into the grey bin",
        "official_success": ("the mustard bottle, the ranch dressing bottle and both BBQ sauce bottles rest in the "
                             "grey bin, with the gripper detached"),
    },
    "CookingClearPlateTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "take_out",
        "instruction": "Put the two measuring cups outside of the plate",
        "official_success": ("both the teal measuring spoons and the orange measuring cups are off the clay plate, "
                             "with the gripper detached"),
    },
    "CookingPickPastaToolTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Move the pink tool from this utensil container to the other utensil holder",
        "official_success": ("the pink utensil rests in the other utensil holder (not the one it starts in), touching "
                             "it, with the gripper detached"),
    },
    "CubesAndBlocksInBinTask": {
        "episode_length_s": 240, "native_action_limit": 3600, "skill": "multi_object_pick_and_place",
        "instruction": "Put all the cubes and blocks in the bin",
        "official_success": ("all three Rubik's cubes, the yellow block, the red block, the green block and the blue "
                             "block rest in the grey bin, with the gripper detached"),
    },
    "DishesInBinTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_pick_and_place",
        "instruction": "Put the dishware in the grey bin",
        "official_success": "both white mugs and the bowl rest in the grey bin, with the gripper detached",
    },
    "ElectronicsInBinTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_pick_and_place",
        "instruction": "Put the electronic devices in the grey bin",
        "official_success": ("the smartphone, the remote control, the computer mouse and the keyboard rest in the grey "
                             "bin, with the gripper detached"),
    },
    "FoodPacking1BoxesTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Pack boxed foods into the bin",
        "official_success": "the Cheez-It box rests in the bin, with the gripper detached",
    },
    "FoodPacking1CansTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Pack canned foods into the bin",
        "official_success": "the tomato soup can rests in the bin, with the gripper detached",
    },
    "FoodPacking2BoxesTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_pick_and_place",
        "instruction": "Pack boxed foods into the bin",
        "official_success": "both the Cheez-It box and the sugar box rest in the bin, with the gripper detached",
    },
    "FoodPacking2CansTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_pick_and_place",
        "instruction": "Pack canned foods into the bin",
        "official_success": "both the tomato soup can and the tuna can rest in the bin, with the gripper detached",
    },
    "FoodPacking3BoxesTask": {
        "episode_length_s": 240, "native_action_limit": 3600, "skill": "multi_object_pick_and_place",
        "instruction": "Pack boxed foods into the bin",
        "official_success": "the Cheez-It box, the pudding box and the sugar box rest in the bin, with the gripper detached",
    },
    "FoodPacking3CansTask": {
        "episode_length_s": 240, "native_action_limit": 3600, "skill": "multi_object_pick_and_place",
        "instruction": "Pack canned foods into the bin",
        "official_success": "the Spam can, the tomato soup can and the tuna can rest in the bin, with the gripper detached",
    },
    "FoodPackingByColorTask": {
        "episode_length_s": 120, "native_action_limit": 1800, "skill": "sorting",
        "instruction": "Pack yellow objects in right container and blue object in the left container",
        "official_success": ("the blue coffee can in the bin on the left; the yellow mustard bottle and the yellow "
                             "sugar box in the bin on the right, with the gripper detached"),
    },
    "FruitsGreenLimesOnPlateTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "multi_object_place_on_surface",
        "instruction": "Put all the green fruit on the plate",
        "official_success": "both limes rest on the clay plate, supported by it, with the gripper detached",
    },
    "FruitsMovingOrangeOrLimeTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Move an orange or a lime to the wood bowl",
        "official_success": "one of the oranges or limes rests in the wooden bowl, touching it, with the gripper detached",
    },
    "FruitsMovingTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Move an orange to the white bowl",
        "official_success": "one of the oranges rests in the white bowl, touching it, with the gripper detached",
    },
    "FruitsOnPlate3Task": {
        "episode_length_s": 200, "native_action_limit": 3000, "skill": "place_on_surface",
        "instruction": "Put three (3) fruits on the plate",
        "official_success": ("exactly three of the lemons, the limes, the oranges and the pomegranate rest on the clay "
                             "plate, supported by it, with the gripper detached"),
    },
    "FruitsOnPlateTask": {
        "episode_length_s": 300, "native_action_limit": 4500, "skill": "multi_object_place_on_surface",
        "instruction": "Put all the fruits on the plate",
        "official_success": ("both lemons, both limes, both oranges and the pomegranate rest on the clay plate, "
                             "supported by it, with the gripper detached"),
    },
    "FruitsOnionTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put the onion in the wood bowl",
        "official_success": "the red onion rests in the wooden bowl, touching it, with the gripper detached",
    },
    "FruitsOnionToPlateTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "place_on_surface",
        "instruction": "Put the onion on the plate",
        "official_success": "the red onion rests on the clay plate, supported by it, with the gripper detached",
    },
    "FruitsOrangesOnPlateTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "multi_object_place_on_surface",
        "instruction": "Put all the oranges on the plate",
        "official_success": "both oranges rest on the clay plate, supported by it, with the gripper detached",
    },
    "GrabABagelTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "pick_up",
        "instruction": "Grab a bagel",
        "official_success": "one of the three bagels is in contact with the gripper and lifted clear above the table",
    },
    "GrabAFruitTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "pick_up",
        "instruction": "Pick up a fruit",
        "official_success": ("one of the bananas, the apple or the orange is in contact with the gripper and lifted "
                             "clear above the table"),
    },
    "GreenSpoonsInPotTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_pick_and_place",
        "instruction": "Put the green spoons in the pot",
        "official_success": "both green ladles and the green slotted serving spoon rest in the pot, with the gripper detached",
    },
    "HammersInLeftBinTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_pick_and_place",
        "instruction": "Put the red hammer and black hammer in the left bin",
        "official_success": "both the red hammer and the black hammer rest in the left bin, with the gripper detached",
    },
    "JugsOnShelfTask": {
        "episode_length_s": 120, "native_action_limit": 1800, "skill": "multi_object_pick_and_place",
        "instruction": "Put all the jugs on the shelf",
        "official_success": "both plastic jugs rest on the storage shelf, with the gripper detached",
    },
    "KeyboardOutOfBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "take_out",
        "instruction": "Take the keyboard out of the bin and put it on the table",
        "official_success": "the keyboard is out of the grey bin and supported by the table, with the gripper detached",
    },
    "LargerObjectRaisinBoxInBinTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "pick_and_place",
        "instruction": "Place the larger object in the grey bin.",
        "official_success": "the raisin box (the larger of the two boxes) rests in the grey bin, with the gripper detached",
    },
    "MarkerInMugTask": {
        "episode_length_s": 40, "native_action_limit": 600, "skill": "pick_and_place",
        "instruction": "Put the whiteboard marker in the mug",
        "official_success": "the whiteboard marker rests in the mug, with the gripper detached",
    },
    "MouseOnKeyboardTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "place_on_surface",
        "instruction": "Put the computer mouse on the keyboard",
        "official_success": "the computer mouse rests on the keyboard, supported by it, with the gripper detached",
    },
    "MoveBananaToBagelPlateTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "multi_object_place_on_surface",
        "instruction": "Move the bananas to the bagel plate",
        "official_success": "both bananas rest on the bagel plate, supported by it, with the gripper detached",
    },
    "MustardAboveRaisinTask": {
        "episode_length_s": 40, "native_action_limit": 600, "skill": "place_on_surface",
        "instruction": "Place the mustard on the raisin box. ",
        "official_success": "the mustard bottle rests on the raisin box, supported by it, with the gripper detached",
    },
    "MustardInLeftBinTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "pick_and_place",
        "instruction": "Put the mustard in the left bin",
        "official_success": "the mustard bottle rests in the left grey bin, with the gripper detached",
    },
    "MustardInRightBinTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "pick_and_place",
        "instruction": "Put the mustard in the right bin",
        "official_success": "the mustard bottle rests in the right grey bin, with the gripper detached",
    },
    "NonHammerToolsInRightBinTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_pick_and_place",
        "instruction": "Put the non-hammer tools in the right bin",
        "official_success": "both the cordless drill and the spring clamp rest in the right bin, with the gripper detached",
    },
    "OneBottleInSquarePailTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put any white plastic bottle in the square pail",
        "official_success": "one of the white plastic bottles rests in the square pail, with the gripper detached",
    },
    "OneBottleOnShelfTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put any white plastic bottle on the shelf",
        "official_success": "one of the white plastic bottles rests on the storage shelf, with the gripper detached",
    },
    "PhoneOrRemoteInBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put the phone or the remote in the grey bin",
        "official_success": "either the phone or the remote control rests in the grey bin, with the gripper detached",
    },
    "PickDrillTask": {
        "episode_length_s": 40, "native_action_limit": 600, "skill": "pick_up",
        "instruction": "Pick up the cordless drill.",
        "official_success": "the cordless drill is in contact with the gripper and lifted clear above the table",
    },
    "PickGlassesTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "pick_up",
        "instruction": "Pick up the eye glasses",
        "official_success": "the eyeglasses are in contact with the gripper and lifted clear above the table",
    },
    "PickOrangeObjectTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_up",
        "instruction": "Pick up the orange measuring cup",
        "official_success": "the orange measuring cups are in contact with the gripper and lifted clear above the table",
    },
    "PickUpBluePitcherTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "pick_up",
        "instruction": "Pick up the large blue pitcher",
        "official_success": "the blue pitcher is in contact with the gripper and lifted clear above the table",
    },
    "PickUpGreenObjectTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "pick_up",
        "instruction": "Pick up the green vegetable block",
        "official_success": "the green frozen vegetable box is in contact with the gripper and lifted clear above the table",
    },
    "PinkSpoonInPotTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put the pink spaghetti spoon in the pot",
        "official_success": "the pink spaghetti spoon rests in the pot, with the gripper detached",
    },
    "PlasticBottlesInSquarePailTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_pick_and_place",
        "instruction": "Put all the small plastic bottles in the square pail",
        "official_success": "all three white plastic bottles rest in the square pail, with the gripper detached",
    },
    "PutBowlOnShelfTopTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "place_on_surface",
        "instruction": "Put the serving bowl anywhere on the shelf in front of you",
        "official_success": "the serving bowl rests on the shelf, supported by it, with the gripper detached",
    },
    "PutMugsOnShelfTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_place_on_surface",
        "instruction": "Put the two mugs on the shelf",
        "official_success": ("both the blue-patterned mug and the white mug rest on the shelf, supported by it, with "
                             "the gripper detached"),
    },
    "PutTwoMugsOnShelfTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "counted_pick_and_place",
        "instruction": "Put two (2) mugs on the wire shelf",
        "official_success": ("exactly two of the blue-patterned mug, the white mug and the red mug rest on the wire "
                             "shelf, touching it, with the gripper detached"),
    },
    "RecycleCartonTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "multi_object_pick_and_place",
        "instruction": "Put the recyclable cartons in the grey bin",
        "official_success": "both the milk carton and the orange juice carton rest in the bin, with the gripper detached",
    },
    "RecycleCartonsOnBoxTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "multi_object_pick_and_place",
        "instruction": "Put the cartons that can be recycled on the box",
        "official_success": ("both the milk carton and the orange juice carton rest on the cardboard box, with the "
                             "gripper detached"),
    },
    "RecycleCartonsVerticalCrateTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "multi_object_pick_and_place",
        "instruction": "Put the cartons that can be recycled in the vertical crate",
        "official_success": ("both the milk carton and the orange juice carton rest in the vertical crate, with the "
                             "gripper detached"),
    },
    "RedDishesInBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "multi_object_pick_and_place",
        "instruction": "Put the red dishware in the grey bin",
        "official_success": "both the red mug and the red bowl rest in the grey bin, with the gripper detached",
    },
    "RedItemsInBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "multi_object_pick_and_place",
        "instruction": "Put all the red things in the grey bin",
        "official_success": "both the red mug and the red bowl rest in the grey bin, with the gripper detached",
    },
    "ReorientAllMugsTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "reorient",
        "instruction": "Reorient all the mugs upright so that the opening is facing upwards.",
        "official_success": "the red mug, the blue-patterned mug and both white mugs stand upright, with the gripper detached",
    },
    "ReorientJugTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "reorient",
        "instruction": "Stand the jug upright",
        "official_success": "the plastic jug stands upright, with the gripper detached",
    },
    "ReorientRedMugTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "reorient",
        "instruction": "Put the red mug upright so that the opening is facing upwards.",
        "official_success": "the red mug stands upright, with the gripper detached",
    },
    "ReorientWhiteMugsTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "reorient",
        "instruction": "Make sure all the white mugs are upright so that the opening is facing upwards.",
        "official_success": "both white mugs stand upright, with the gripper detached",
    },
    "RubiksCubeAndBananaTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "multi_object_pick_and_place",
        "instruction": "Put the cube and the banana in the bowl",
        "official_success": "both the Rubik's cube and the banana rest in the bowl, touching it, with the gripper detached",
    },
    "RubiksCubeBehindBowlTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "relative_placement",
        "instruction": "Put the rubiks cube behind the bowl",
        "official_success": ("the Rubik's cube is behind the bowl, on its far side from the robot, touching the table, "
                             "with the gripper detached"),
    },
    "RubiksCubeInFrontOfBowlTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "relative_placement",
        "instruction": "Put the rubiks cube in front of the bowl",
        "official_success": "the Rubik's cube is in front of the bowl, on the robot's side of it, with the gripper detached",
    },
    "RubiksCubeLeftOfBowlTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "relative_placement",
        "instruction": "Put the rubiks cube to the left of the bowl",
        "official_success": "the Rubik's cube is to the left of the bowl as seen from the robot, with the gripper detached",
    },
    "RubiksCubeOrBananaTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "pick_and_place",
        "instruction": "Put the cube or the banana in the bowl",
        "official_success": "either the cube or the banana rests in the bowl, touching it, with the gripper detached",
    },
    "RubiksCubeRightOfBowlTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "relative_placement",
        "instruction": "Put the rubiks cube to the right of the bowl",
        "official_success": "the Rubik's cube is to the right of the bowl as seen from the robot, with the gripper detached",
    },
    "RubiksCubeTask": {
        "episode_length_s": 40, "native_action_limit": 600, "skill": "pick_and_place",
        "instruction": "Put the cube in the bowl",
        "official_success": "the Rubik's cube rests in the bowl, touching it, with the gripper detached",
    },
    "RubiksCubeThenBananaTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "ordered_pick_and_place",
        "instruction": "Put the cube then the banana in the bowl",
        "official_success": ("the Rubik's cube and then the banana came to rest in the bowl, touching it, in that "
                             "order, with the gripper detached"),
    },
    "RubiksCubesInBinTask": {
        "episode_length_s": 120, "native_action_limit": 1800, "skill": "multi_object_pick_and_place",
        "instruction": "Sort all rubiks cubes into the bin",
        "official_success": "all three Rubik's cubes rest in the grey bin, with the gripper detached",
    },
    "SauceBottlesCrateTask": {
        "episode_length_s": 40, "native_action_limit": 600, "skill": "pick_and_place",
        "instruction": "Put the red bbq sauce bottle in the crate",
        "official_success": "the BBQ sauce bottle rests in the purple crate, with the gripper detached",
    },
    "SmallPumpkinInBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put the small pumpkin in the bin",
        "official_success": "the small pumpkin rests in the bin, with the gripper detached",
    },
    "SmallerObjectButterInBinTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "pick_and_place",
        "instruction": "Place the smaller object in the grey bin.",
        "official_success": "the butter box (the smaller of the two boxes) rests in the grey bin, with the gripper detached",
    },
    "SmartphoneInBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put the smartphone in the grey bin",
        "official_success": "the smartphone rests in the grey bin, with the gripper detached",
    },
    "SpoonInMugTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put the metal spoon that's in the wooden bowl in the mug",
        "official_success": "the metal spoon rests in the mug, with the gripper detached",
    },
    "SpoonsInPotTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_pick_and_place",
        "instruction": "Put all of the serving spoons with no holes in the pot",
        "official_success": "both green ladles and the red serving spoon rest in the pot, with the gripper detached",
    },
    "Stack3RubiksCubeTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "stacking",
        "instruction": "Stack the rubiks cubes in a tower",
        "official_success": "all three Rubik's cubes form one stack in any order, each touching the one below",
    },
    "StackWhiteMugsTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "stacking",
        "instruction": "Stack the white mugs on top of each other.",
        "official_success": "the two white mugs are stacked one on the other, touching",
    },
    "StackYellowOnRedTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "stacking",
        "instruction": "Stack the yellow block on the red block",
        "official_success": "the yellow block is stacked on top of the red block",
    },
    "TakeMeasuringSpoonOutTask": {
        "episode_length_s": 40, "native_action_limit": 600, "skill": "take_out",
        "instruction": "Take the white colored measuring spoon out of the red bowl and put it on the table.",
        "official_success": "the measuring spoon is out of the red bowl, with the gripper detached",
    },
    "TakeMugsOffOfShelfTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "take_out",
        "instruction": "Take the mugs off the shelf",
        "official_success": ("both the blue-patterned mug and the white mug are off the shelf and supported by the "
                             "table, with the gripper detached"),
    },
    "TakeSpatulaOffShelfTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "take_out",
        "instruction": "Take the spatula off the shelf and put it on the table",
        "official_success": "the wooden spatula is off the wire shelf and supported by the table, with the gripper detached",
    },
    "ThrowAwayAppleTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Throw away the apple",
        "official_success": "the apple rests in the green pail, with the gripper detached",
    },
    "ThrowAwaySnacksTask": {
        "episode_length_s": 120, "native_action_limit": 1800, "skill": "multi_object_pick_and_place",
        "instruction": "Put away the snacks in the bin",
        "official_success": "both the apple and the yogurt cup rest in the green pail, with the gripper detached",
    },
    "ToolOrganizationBothTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_pick_and_place",
        "instruction": "Put hammers in the right bin and do not touch anything else",
        "official_success": "both the red hammer and the black hammer rest in the right bin, with the gripper detached",
    },
    "ToolOrganizationTask": {
        "episode_length_s": 180, "native_action_limit": 2700, "skill": "multi_object_pick_and_place",
        "instruction": "Put the red hammer and black hammer in the left bin",
        "official_success": "both the red hammer and the black hammer rest in the left bin, with the gripper detached",
    },
    "ToolsPickingAllHammersTask": {
        "episode_length_s": 240, "native_action_limit": 3600, "skill": "take_out",
        "instruction": "Take out all the hammers and put them on the table",
        "official_success": ("the black hammer, the blue hammer, the red hammer and the wooden hammer are out of the "
                             "left bin and supported by the table, with the gripper detached"),
    },
    "ToolsPickingDrillTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "take_out",
        "instruction": "Select the cordless drill and put it on the table",
        "official_success": "the cordless drill is out of the left bin and supported by the table, with the gripper detached",
    },
    "ToolsPickingHammerTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "take_out",
        "instruction": "Select the blue hammer and put it on the table",
        "official_success": "the blue hammer is out of the left bin and supported by the table, with the gripper detached",
    },
    "ToyInBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put the lizard away in the bin",
        "official_success": "the lizard figurine rests in the grey bin, with the gripper detached",
    },
    "UnstackRubiksCubeTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "take_out",
        "instruction": "Unstack the rubiks cube tower",
        "official_success": ("all three Rubik's cubes are out of the grey bin and supported by the table, with the "
                             "gripper detached"),
    },
    "UtensilsInMugTask": {
        "episode_length_s": 90, "native_action_limit": 1350, "skill": "multi_object_pick_and_place",
        "instruction": "Put the fork and spoon in the ceramicmug",
        "official_success": "both the fork and the metal spoon rest in the mug, with the gripper detached",
    },
    "WhiteMugInCenterOfTableTask": {
        "episode_length_s": 30, "native_action_limit": 450, "skill": "relative_placement",
        "instruction": "Put the white mug in the center of the table.",
        "official_success": "the white mug is at the center of the table, touching it, with the gripper detached",
    },
    "WhiteMugsInBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "multi_object_pick_and_place",
        "instruction": "Clean up the white mugs",
        "official_success": "both white mugs rest in the grey bin, with the gripper detached",
    },
    "WoodSpatulaToBowlTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "pick_and_place",
        "instruction": "Put the wooden spatula in the bowl",
        "official_success": "the wooden spatula rests in the wooden bowl, touching it, with the gripper detached",
    },
    "YellowAndWhiteObjectsInBinTask": {
        "episode_length_s": 60, "native_action_limit": 900, "skill": "multi_object_pick_and_place",
        "instruction": "Put all white objects and yellow objects in the grey bin",
        "official_success": "both the white mug and the banana rest in the grey bin, with the gripper detached",
    },
    "YogurtInBowlTask": {
        "episode_length_s": 40, "native_action_limit": 600, "skill": "pick_and_place",
        "instruction": "Put the small red yogurt in the red bowl",
        "official_success": "the yogurt cup rests in the red bowl, with the gripper detached",
    },
    # --- end of generated roster ---
}
ACTION_CODECS = ("motion_action_v3",)


def public_context() -> dict[str, Any]:
    return {
        "schema": "ora.public-environment-context.v1",
        "environment_id": "robolab",
        "embodiment_id": EMBODIMENT_ID,
        "robot_models": [ROBOT_MODEL],
        "controller": CONTROLLER,
        "public_cameras": list(PUBLIC_CAMERAS),
    }


def validate_runtime_extension(runtime: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(runtime, Mapping):
        raise RoboLabBenchmarkError("RoboLab runtime must be an object")
    task = runtime.get("task")
    if task not in TASKS:
        raise RoboLabBenchmarkError(f"RoboLab task is not admitted: {task!r} (admitted: {sorted(TASKS)})")
    seed = runtime.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RoboLabBenchmarkError("RoboLab seed must be a non-negative integer")
    if runtime.get("required_revision", REVISION) != REVISION:
        raise RoboLabBenchmarkError("RoboLab source revision differs from the pin")
    codec = runtime.get("action_codec", "motion_action_v3")
    if codec not in ACTION_CODECS:
        raise RoboLabBenchmarkError(f"unsupported RoboLab action codec: {codec!r}")
    return {"task": task, "seed": int(seed), "required_revision": REVISION, "action_codec": codec,
            "native_action_limit": TASKS[task]["native_action_limit"]}
