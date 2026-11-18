"""Public completion clauses for RoboLab-120 tasks, one template per predicate family.

Clauses name objects in natural language only (by asset, scene or the task's own wording; never an
identifier), state end conditions, never strategy, and carry no number. ``TASK_REVIEW`` holds what
a family template cannot phrase: reset relations, the instruction's stronger intent and the few
clauses written by hand. The private review keeps the predicate, thresholds and scoring gaps.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from .task_source import Facts, Term, _as_list, _asset_key, family, logical, targets

NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve")

# ----------------------------------------------------------------------------------------------- phrases
#: Public noun phrases by asset (the scene's payload), then by scene object name. Natural language only:
#: no identifiers reach a clause. A task's own wording (``TASK_PHRASES``) wins over both.
ASSET_PHRASES = {
    "objects/ycb/mug.usd": "red mug", "objects/hot3d/mug.usd": "white mug",
    "objects/hot3d/ceramic_mug.usd": "blue-patterned mug", "objects/ycb/bowl.usd": "bowl",
    "objects/ycb/banana.usd": "banana", "objects/hot3d/rubiks_cube.usd": "Rubik's cube",
    "objects/hot3d/lizard_figurine.usd": "lizard figurine",
}
#: Scene-specific names: the clutter scene's only bin is called "the bin" by its instructions.
SCENE_PHRASES = {("clutter_fruit_bottle_bluebin.usda", "right_bin"): "bin"}
OBJECT_PHRASES = {
    "anza_medium": "pot", "apple_01": "apple", "banana": "banana", "bowl": "bowl", "rubiks_cube": "Rubik's cube", "avocado01": "avocado", "bagel_00": "bagel", "bagel_01": "bagel",
    "bagel_02": "bagel", "bagel_06": "bagel", "bagel_07": "bagel", "bbq_sauce_bottle": "BBQ sauce bottle",
    "bbq_sauce_bottle_01": "BBQ sauce bottle", "bin_a06": "bin", "blue_block": "blue block",
    "blue_hammer": "blue hammer", "butter": "butter box", "canned_tuna": "can of tuna", "cheez_it": "Cheez-It box",
    "chocolate_pudding": "pudding box", "clay_plates": "clay plate", "coffee_can": "blue coffee can",
    "coffee_pot": "coffee pot", "computer_mouse": "computer mouse", "container_a01": "bin",
    "cordless_drill": "cordless drill", "cubebox_a02": "cardboard box", "fork_big": "fork",
    "frozen_vegetable_block": "green frozen vegetable box", "glasses": "eyeglasses",
    "green_block": "green block", "green_serving_spoon": "green slotted serving spoon", "grey_bin": "grey bin",
    "grey_bin_left": "left grey bin", "grey_bin_right": "right grey bin", "husky_hammer": "black hammer",
    "keyboard": "keyboard", "ladle": "green ladle", "ladle_01": "green ladle", "large_storage_rack": "storage shelf",
    "left_bin": "left bin", "lemon_01": "lemon", "lemon_02": "lemon", "lime01": "lime", "lime01_01": "lime",
    "marker": "whiteboard marker", "measuring_cup": "measuring cup", "measuring_cups_1": "orange measuring cups",
    "milk_carton": "milk carton", "milkjug_a01": "milk jug", "mustard": "mustard bottle",
    "mustard_bottle": "mustard bottle", "orange2": "orange", "orange_01": "orange", "orange_02": "orange",
    "orange_juice_carton": "orange juice carton", "pink_spaghetti_spoon": "pink spaghetti spoon",
    "pitcher": "blue pitcher", "plasticpail_a02": "green pail", "plate_large": "plate", "plate_small": "plate",
    "pomegranate01": "pomegranate", "pumpkinlarge": "large pumpkin", "pumpkinsmall": "small pumpkin",
    "purple_crate": "purple crate", "rack_l04": "shelf", "raisin_box": "raisin box", "ranch_dressing":
    "ranch dressing bottle", "red_block": "red block", "red_hammer": "red hammer", "red_onion": "red onion",
    "redonion": "red onion", "red_serving_spoon": "red serving spoon", "remote_control": "remote control",
    "right_bin": "right bin", "serving_bowl": "white serving bowl", "smartphone": "smartphone",
    "spam_can": "Spam can", "spatula_01": "wooden spatula", "spoon_1": "teal measuring spoons",
    "spoon_big": "metal spoon", "spring_clamp": "spring clamp", "squarepail_a01": "square pail",
    "storage_box_01": "utensil holder", "sugar_box": "sugar box", "table": "table",
    "tomato_soup_can": "tomato soup can", "tuna_can": "tuna can", "utilityjug_a01": "plastic jug",
    "utilityjug_a02": "plastic jug", "utilityjug_a03": "plastic jug",
    "whitepackerbottle_a01": "white plastic bottle", "whitepackerbottle_a02": "white plastic bottle",
    "whitepackerbottle_a03": "white plastic bottle", "wireshelving_a01": "wire shelf",
    "wood_hammer": "wooden hammer", "wooden_bowl": "wooden bowl", "wooden_spoons": "wooden spatula",
    "yellow_block": "yellow block", "yogurt_cup": "yogurt cup",
}
#: Phrases a task's own instruction fixes (the reviewed contracts call the Rubik's cube "the cube").
TASK_PHRASES: dict[str, dict[str, str]] = {
    "RubiksCubeAndBananaTask": {"rubiks_cube": "cube"}, "RubiksCubeOrBananaTask": {"rubiks_cube": "cube"},
    "RubiksCubeTask": {"rubiks_cube": "cube"}, "BananaThenRubiksCubeTask": {"rubiks_cube": "cube"},
    "RubiksCubeThenBananaTask": {"rubiks_cube": "cube"},
    "BowlStackingLeftOnRightTask": {"bowl_1": "right bowl", "bowl_2": "left bowl"},
    "BowlStackingRightOnLeftTask": {"bowl_1": "right bowl", "bowl_2": "left bowl"},
    "BigPumpkinInBinTask": {"pumpkinlarge": "larger pumpkin"},
    "CookingPickPastaToolTask": {"pink_spaghetti_spoon": "pink utensil",
                                 "storage_box_01": "other utensil holder (not the one it starts in)"},
    "FoodPackingByColorTask": {"bin_a06": "bin on the left", "bin_b03": "bin on the right",
                               "sugar_box": "yellow sugar box", "mustard": "yellow mustard bottle"},
    "FruitsMovingTask": {"serving_bowl": "white bowl"},
    "LargerObjectRaisinBoxInBinTask": {"raisin_box": "raisin box (the larger of the two boxes)"},
    "SmallerObjectButterInBinTask": {"butter": "butter box (the smaller of the two boxes)"},
    "MarkerInMugTask": {"ceramic_mug": "mug"}, "SpoonInMugTask": {"ceramic_mug": "mug"},
    "UtensilsInMugTask": {"ceramic_mug": "mug"},
    "MoveBananaToBagelPlateTask": {"plate_small": "bagel plate"},
    "PhoneOrRemoteInBinTask": {"smartphone": "phone"},
    "PickOrangeObjectTask": {"measuring_cups_1": "orange measuring cups"},
    "RecycleCartonsVerticalCrateTask": {"container_a01": "vertical crate"},
    "RedDishesInBinTask": {"bowl": "red bowl"}, "RedItemsInBinTask": {"bowl": "red bowl"},
    "TakeMeasuringSpoonOutTask": {"measuring_cup": "measuring spoon", "bowl": "red bowl"},
    "YogurtInBowlTask": {"bowl": "red bowl"},
    "ReorientAllMugsTask": {"ceramic_mug": "blue-patterned mug"},
    "PutBowlOnShelfTopTask": {"serving_bowl": "serving bowl"},
}
#: Containers whose open-top hull is a frame or a closed box, not a vessel: "on", not "inside".
FRAME_CONTAINERS = {"large_storage_rack": "rack", "rack_l04": "rack", "wireshelving_a01": "rack",
                    "cubebox_a02": "box"}
FLAT_REFERENCES = {"clay_plates", "plate_large", "plate_small"}
#: Phrases that name one object with a plural noun.
PLURAL_NOUNS = {"eyeglasses", "orange measuring cups", "teal measuring spoons"}
IRREGULAR_PLURALS = {"box": "boxes", "eyeglasses": "eyeglasses", "measuring cups": "measuring cups",
                     "teal measuring spoons": "teal measuring spoons", "orange measuring cups": "orange measuring cups"}

# ------------------------------------------------------------- objects the instruction leaves to infer
#: An instruction that picks its objects by category, attribute, comparison or property ("the black items",
#: "the larger object", "the green spoons", "the serving spoons with no holes") leaves which objects are meant
#: to the policy. Naming them in the public contract would answer the task, so such a contract keeps the
#: instruction's own descriptor and the private review keeps the names. An instruction that names its objects
#: concretely ("the banana", "the yellow block", "the red mug") keeps the names.
_CATEGORY = re.compile(r"\b(items?|objects?|things?|toys?|foods?|fruits?|devices?|dishware|condiments?|tools?"
                       r"|snacks?|recyclable|recycled)\b", re.I)
_COMPARISON = re.compile(r"\b(larger|smaller|bigger|heavier|lighter|taller|shorter|longer)\b", re.I)
#: An attribute in front of a plural noun selects a subset ("the white mugs", "all plastic bottles").
_ATTRIBUTE_SET = re.compile(r"\b(red|black|white|yellow|blue|green|pink|orange|grey|gray|purple|brown|colou?red|small"
                            r"|large|plastic|metal|wooden)\s+(?:\w+\s+){0,2}\w+s\b", re.I)
_PROPERTY = re.compile(r"\bwith(?:out)? no\b|\bwithout\b", re.I)


def refers_by_descriptor(instruction: str) -> bool:
    """Whether the instruction leaves the policy to infer which objects it means."""
    return any(pattern.search(instruction) for pattern in (_CATEGORY, _COMPARISON, _ATTRIBUTE_SET, _PROPERTY))


#: Each such task's descriptor in the instruction's own words: (text, "one" | "set"), or, for a sorting
#: task, one per container. ``descriptor`` checks that every word but an article is the instruction's.
INSTRUCTION_DESCRIPTORS: dict[str, Any] = {
    "BBQSauceInBinTask": ("the red BBQ sauce bottles", "set"),
    "BigPumpkinInBinTask": ("the bigger pumpkin", "one"),
    "BlackItemsInBinTask": ("the black items", "set"),
    "BlocksInBinTask": ("all the colored blocks", "set"),
    "CannedFoodInBinTask": ("the canned food", "one"),
    "CleanUpToysTask": ("all the smaller toys", "set"),
    "ClearOrganicObjectsTask": ("the organic objects", "set"),
    "ClutterPlasticTask": ("all the plastic bottles", "set"),
    "CondimentsInBinTask": ("the sauce condiments", "set"),
    "CookingPickPastaToolTask": ("the pink tool", "one"),
    "DishesInBinTask": ("the dishware", "set"),
    "ElectronicsInBinTask": ("the electronic devices", "set"),
    "FoodPacking1BoxesTask": ("the boxed foods", "set"), "FoodPacking2BoxesTask": ("the boxed foods", "set"),
    "FoodPacking3BoxesTask": ("the boxed foods", "set"), "FoodPacking1CansTask": ("the canned foods", "set"),
    "FoodPacking2CansTask": ("the canned foods", "set"), "FoodPacking3CansTask": ("the canned foods", "set"),
    "FoodPackingByColorTask": {"bin_a06": ("the blue object", "one"), "bin_b03": ("the yellow objects", "set")},
    "FruitsGreenLimesOnPlateTask": ("all the green fruit", "set"),
    "FruitsOnPlate3Task": ("three fruits", "set"),
    "FruitsOnPlateTask": ("all the fruits", "set"),
    "GrabAFruitTask": ("a fruit", "one"),
    "GreenSpoonsInPotTask": ("the green spoons", "set"),
    "LargerObjectRaisinBoxInBinTask": ("the larger object", "one"),
    "NonHammerToolsInRightBinTask": ("the non-hammer tools", "set"),
    "PlasticBottlesInSquarePailTask": ("all the small plastic bottles", "set"),
    "RecycleCartonTask": ("the recyclable cartons", "set"),
    "RecycleCartonsOnBoxTask": ("the cartons that can be recycled", "set"),
    "RecycleCartonsVerticalCrateTask": ("the cartons that can be recycled", "set"),
    "RedDishesInBinTask": ("the red dishware", "set"),
    "RedItemsInBinTask": ("all the red things", "set"),
    "ReorientWhiteMugsTask": ("all the white mugs", "set"),
    "SmallerObjectButterInBinTask": ("the smaller object", "one"),
    "SpoonsInPotTask": ("all of the serving spoons with no holes", "set"),
    "StackWhiteMugsTask": ("the white mugs", "set"),
    "ThrowAwaySnacksTask": ("the snacks", "set"),
    "WhiteMugsInBinTask": ("the white mugs", "set"),
    "YellowAndWhiteObjectsInBinTask": ("all white objects and yellow objects", "set"),
}
_ARTICLES = {"the", "a", "an", "all", "of"}


def descriptor(facts: Facts) -> Any:
    """The instruction's descriptor of a task that leaves its objects to infer, else None; fails closed."""
    entry = INSTRUCTION_DESCRIPTORS.get(facts.task)
    if refers_by_descriptor(facts.instruction) != (entry is not None):
        raise ValueError(f"{facts.task}: the descriptor rule and INSTRUCTION_DESCRIPTORS disagree on "
                         f"{facts.instruction!r}")
    if entry is None:
        return None
    words = set(re.findall(r"[a-z]+", facts.instruction.lower()))
    for text, _size in (entry.values() if isinstance(entry, dict) else [entry]):
        foreign = [word for word in re.findall(r"[a-z]+", text.lower()) if word not in _ARTICLES | words]
        if foreign:
            raise ValueError(f"{facts.task}: {text!r} is not the instruction's own descriptor ({foreign})")
    return entry


# ----------------------------------------------------------------------------- per-task review knowledge
#: What the termination alone cannot say: a task's reset relations, the instruction's stronger intent and
#: the clauses a family template cannot phrase. Every entry is reviewable source reading, not strategy.
TASK_REVIEW: dict[str, dict[str, Any]] = {
    "BananasInBinOneMoreTask": {
        "clauses": ["Of the bananas that start on the table, exactly one must end up inside the grey bin; a "
                    "second one inside the bin as well does not complete the task.",
                    "Let go of it there: a banana the gripper still touches does not count.",
                    "Leave the bananas that are already in the bin there."],
        "intent": [2], "reset": "two bananas start inside the grey bin and are not named by the termination",
        "counted": "exactly one of the three bananas that start on the table",
        "gap": "The termination counts only the three bananas that start on the table (exactly one of them in "
               "the bin, released); the two that start in the bin are not checked, so removing them is unscored."},
    "BananasInBinThreeTotalTask": {
        "counted": "exactly one of the three bananas that start on the table",
        "clauses": ["Of the bananas that start on the table, exactly one must end up inside the grey bin, so "
                    "that three are in it; a second one from the table does not complete the task.",
                    "Let go of it there: a banana the gripper still touches does not count.",
                    "Leave the bananas that are already in the bin there."],
        "intent": [2], "reset": "two bananas start inside the grey bin and are not named by the termination",
        "gap": "The termination counts only the three bananas that start on the table (exactly one of them in "
               "the bin, released); the two that start in the bin are not checked, so removing them is unscored."},
    "BananasInCrateTask": {
        "clauses": ["Exactly two bananas must be inside the purple crate at the end, counting the one that "
                    "starts there; a third banana inside does not complete the task.",
                    "Let go of them there: a banana the gripper still touches does not count."],
        "reset": "one banana starts inside the purple crate and counts toward the two"},
    "FruitsOnPlate3Task": {
        "clauses": ["Exactly three fruits must end up resting on the clay plate, each supported by the plate "
                    "itself with its center over it, counting any fruit that starts there; a fourth fruit on the "
                    "plate does not complete the task.",
                    "Let go of them there: a fruit the gripper still touches does not count."],
        "reset": "one orange starts on the clay plate and counts toward the three"},
    "CleanUpToysTask": {"intent_clause": "Leave the birdhouse on the table.",
                        "gap": "The termination does not check the birdhouse."},
    "ToolOrganizationBothTask": {"intent_clause": "Do not move any other object.",
                                 "gap": "The termination does not check that other objects stay untouched."},
    "TakeMeasuringSpoonOutTask": {
        "clauses": ["The measuring spoon that starts in the red bowl must end up out of the bowl, on the table.",
                    "Let go of it: the task is not complete while any part of the gripper still touches it."],
        "intent": [0], "relation": [0], "reset": "the measuring spoon starts inside the red bowl",
        "gap": "The termination checks only that the spoon is out of the bowl and released, not that it rests "
               "on the table."},
    "BananasOutOfBinTask": {
        "subject": "Both bananas that start in the grey bin", "container_ref": "it",
        "reset": "two bananas start inside the grey bin; the termination names exactly those two"},
    "ToolsPickingDrillTask": {
        "clauses": ["The cordless drill must end up resting directly on the table, out of the bins.",
                    "Let go of it: the task is not complete while any part of the gripper still touches it."],
        "reset": "the drill starts inside the right bin; the termination names the left bin",
        "gap": "The container term names the left bin while the drill starts in the right bin; resting "
               "directly on the table (the support term) is what excludes every bin."},
    "UnstackRubiksCubeTask": {
        "clauses": ["All three Rubik's cubes of the tower must end up resting directly on the table, none of them "
                    "on another cube and none in the grey bin.",
                    "Let go of each of them: the task is not complete while the gripper still touches any of "
                    "them."],
        "reset": "the three cubes start stacked into a tower beside the grey bin, the bottom one on the table"},
    "KeyboardOutOfBinTask": {"reset": "the keyboard starts inside the grey bin"},
    "BlackItemsInBinTask": {"reset": "the keyboard starts inside the grey bin and already satisfies its term"},
    "ElectronicsInBinTask": {"reset": "the keyboard starts inside the grey bin and already satisfies its term"},
    "TakeMugsOffOfShelfTask": {"reset": "both mugs start on the shelf"},
    "TakeSpatulaOffShelfTask": {"reset": "the spatula starts on the wire shelf"},
    "ToolsPickingAllHammersTask": {"reset": "the four hammers start inside the left bin"},
    "ToolsPickingHammerTask": {"reset": "the blue hammer starts inside the left bin with the other hammers"},
    "CookingClearPlateTask": {"reset": "the measuring spoons and the measuring cups start on the clay plate"},
    "FruitsOrangesOnPlateTask": {"reset": "one orange starts on the clay plate"},
    "FruitsOnPlateTask": {"reset": "one orange starts on the clay plate"},
    "ReorientWhiteMugsTask": {"reset": "one white mug starts upright and already satisfies its term"},
    "ReorientAllMugsTask": {"reset": "one white mug starts upright and already satisfies its term"},
    "CookingPickPastaToolTask": {"reset": "the pink utensil starts in the other utensil holder"},
    "SpoonInMugTask": {"reset": "the metal spoon starts in the wooden bowl"},
    "PutTwoMugsOnShelfTask": {"intent_clause": None},
}


# ----------------------------------------------------------------------------------------------- wording
#: Instruction wording a contract keeps but the roster's official paraphrase does not ("the cube").
WORDING_ONLY = {"rubiks_cube"}


def phrase(facts: Facts, name: str, *, canonical: bool = False) -> str:
    override = TASK_PHRASES.get(facts.task, {}).get(name)
    if override and not (canonical and name in WORDING_ONLY):
        return override
    if (facts.scene, name) in SCENE_PHRASES:
        return SCENE_PHRASES[(facts.scene, name)]
    asset = _asset_key((facts.prims.get(name) or {}).get("asset"))
    if asset in ASSET_PHRASES:
        return ASSET_PHRASES[asset]
    if name in OBJECT_PHRASES:
        return OBJECT_PHRASES[name]
    base = re.sub(r"_?\d+$", "", name)  # banana_01, rubiks_cube_2: a numbered copy of a named object
    if base in OBJECT_PHRASES:
        return OBJECT_PHRASES[base]
    raise ValueError(f"{facts.task}: no public phrase for {name}")


def plural(text: str) -> str:
    if text in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[text]
    head, _, word = text.rpartition(" ")
    if word in IRREGULAR_PLURALS:
        word = IRREGULAR_PLURALS[word]
    elif word.endswith(("s", "x", "ch", "sh")):
        word += "es"
    else:
        word += "s"
    return f"{head} {word}" if head else word


def number(count: int) -> str:
    return NUMBER_WORDS[count] if count < len(NUMBER_WORDS) else str(count)


def _groups(facts: Facts, names: Sequence[str], canonical: bool = False) -> list[tuple[str, int]]:
    counts = Counter(phrase(facts, name, canonical=canonical) for name in names)
    return list(counts.items())


def listing(facts: Facts, names: Sequence[str], canonical: bool = False) -> str:
    """'the a, both bs and the c' from object names, identical phrases grouped."""
    parts = []
    for text, count in _groups(facts, names, canonical):
        if count == 1:
            parts.append(f"the {text}")
        elif count == 2:
            parts.append(f"both {plural(text)}")
        else:
            parts.append(f"all {number(count)} {plural(text)}")
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def subject_all(facts: Facts, names: Sequence[str], canonical: bool = False) -> tuple[str, bool]:
    """The subject of an every-object clause, and whether it needs a trailing list ('Every one of these')."""
    groups = _groups(facts, names, canonical)
    if len(names) == 1:
        return f"The {phrase(facts, names[0], canonical=canonical)}", False
    if len(groups) == 1:
        count = groups[0][1]
        return (f"Both {plural(groups[0][0])}" if count == 2 else f"All {number(count)} {plural(groups[0][0])}"), False
    if len(names) == 2:
        return (f"Both the {phrase(facts, names[0], canonical=canonical)} and the "
                f"{phrase(facts, names[1], canonical=canonical)}"), False
    return "Every one of these", True


def uncap(text: str) -> str:
    return text[0].lower() + text[1:]


def counted(facts: Facts, names: Sequence[str], count: int) -> str:
    """'exactly two of the five bananas' or 'exactly three of the lemons, limes, oranges and the pomegranate'."""
    override = TASK_REVIEW.get(facts.task, {}).get("counted")
    if override:
        return override
    groups = _groups(facts, names, True)
    if len(groups) == 1:
        return f"exactly {number(count)} of the {number(groups[0][1])} {plural(groups[0][0])}"
    texts = [f"the {plural(text)}" if n > 1 else f"the {text}" for text, n in groups]
    return f"exactly {number(count)} of " + ", ".join(texts[:-1]) + " and " + texts[-1]


def statement_subject(facts: Facts, names: Sequence[str]) -> str:
    """The lowercase subject of the roster's official paraphrase."""
    subject, trailing = subject_all(facts, names, canonical=True)
    return listing(facts, names, canonical=True) if trailing else subject[0].lower() + subject[1:]


def subject_any(facts: Facts, names: Sequence[str]) -> str:
    groups = _groups(facts, names)
    if len(groups) == 1:
        return f"One of the {plural(groups[0][0])}"
    if len(names) == 2:
        return f"Either the {phrase(facts, names[0])} or the {phrase(facts, names[1])}"
    texts = [plural(text) if count > 1 else text for text, count in groups]
    return "One of the " + (", ".join(texts[:-1]) + " or " + texts[-1])


def release_clause(count: int, placed: bool, mode: str = "all", name: str | None = None) -> str:
    """The release clause; the reviewed contracts' wording for one and for two objects."""
    if mode == "any":
        return "Let go of it there: an object the gripper still touches does not count."
    if mode == "choose":
        return "Let go of them there: an object the gripper still touches does not count."
    if count == 1:
        where = " there" if placed else ""
        return f"Let go of the {name}{where}: the task is not complete while any part of the gripper still touches it."
    tail = "either one" if count == 2 else "any of them"
    return f"Let go of each of them: the task is not complete while the gripper still touches {tail}."


def _single_release(facts: Facts, name: str, placed: bool) -> str:
    return release_clause(1, placed, name=phrase(facts, name))


def relation_words(facts: Facts, container: str) -> tuple[str, str]:
    """('inside the bin', 'in the bin') for a vessel; ('on the shelf, within its frame', ...) for a frame."""
    text = phrase(facts, container)
    kind = FRAME_CONTAINERS.get(container)
    if kind == "rack":
        return f"on the {text}, inside its frame", f"on the {text}"
    if kind == "box":
        return f"on top of the {text}, over it", f"on the {text}"
    return f"inside the {text}", f"in the {text}"


# ----------------------------------------------------------------------------------------------- clauses
@dataclass
class Projection:
    clauses: list[str]
    conditions: list[dict[str, Any]]
    official_success: str
    counterexamples: list[dict[str, str]]
    forbidden: list[str] = field(default_factory=list)


def _condition(identifier: str, basis: str, interpretation: str, indices: list[int],
               disposition: str = "included", omission: str | None = None) -> dict[str, Any]:
    item = {"id": identifier, "source_basis": basis, "interpretation": interpretation,
            "disposition": disposition, "public_clause_indices": indices}
    if omission:
        item["omission_reason"] = omission
    return item


def _call(term: Term) -> str:
    inner = ", ".join(f"{key}={value!r}" for key, value in term.params.items())
    return f"{term.func}({inner})"


MECHANISM = {
    "in_container": "the object's convex-hull centroid inside the container's hull with its top faces removed, in "
                    "the container's local frame",
    "groups_in_containers": "per group, object_in_container's hull-centroid test against that group's container",
    "ordered_in_container": "object_in_container's hull-centroid test, contact and release per object, with each "
                            "object's first true step latched",
    "on_surface": "is_supported_on_surface (contact force from the surface above 0.1 N, upward, within 45 degrees "
                  "of vertical) and centroid_in_footprint (box centroid xy inside the surface's world AABB)",
    "stacked": "check_stacked: contact with the lower object and the upper one's box above the lower one's top (or "
               "bottom) face",
    "out_of_container": "not in_opentop_container (hull centroid outside the open-top hull)",
    "out_onto_surface": "not in_opentop_container and is_supported_on_surface(table)",
    "picked_up": "object_grabbed (gripper contact) and object_above(table, z_margin=distance): every box corner above "
                 "the table's top face by the margin, centroid over its footprint",
    "upright": "the object's local up axis within the tolerance angle of world +z",
    "relative_position": "the xy vector from the reference's root to the object's root, in the robot root frame, "
                         "inside the 45-degree cone of the direction",
    "centered": "center_of: box-centroid xy within the tolerance of the table's box-centroid xy",
}
RELEASE_BASIS = ("require_gripper_detached=True: gripper_detached checks that no gripper body touches the object "
                 "above 0.1 N")

PRIVATE_LIMITS = {
    "in_container": "Hull slack, contact force bar and the open-top hull construction",
    "groups_in_containers": "Hull slack, contact force bar and the open-top hull construction",
    "ordered_in_container": "Hull slack, contact force bar and the placement latch",
    "on_surface": "Support force bar, support cone and footprint slack",
    "stacked": "Contact force bar and bounding-box slack",
    "out_of_container": "Hull slack and contact force bar",
    "out_onto_surface": "Hull slack, support force bar and support cone",
    "picked_up": "Contact force bar and lift margin",
    "upright": "Tilt angle limit and contact force bar",
    "relative_position": "Direction cone and contact force bar",
    "centered": "Centering distance and contact force bar",
}


#: Defaults of the success predicates whose release or contact is on unless a task turns it off.
PREDICATE_DEFAULTS = {"objects_placed_in_container_in_order": {"require_contact_with": True,
                                                               "require_gripper_detached": True}}


def _flag(term: Term, key: str) -> Any:
    return term.params.get(key, PREDICATE_DEFAULTS.get(term.func, {}).get(key, False))


def _released(facts: Facts) -> bool:
    term = facts.terms[0]
    if term.func == "object_groups_in_containers":
        return all(group.get("require_gripper_detached", True) for group in term.params["groups"])
    return all(_flag(item, "require_gripper_detached") for item in facts.terms)


def _main_clauses(facts: Facts) -> tuple[list[str], str, str, bool]:
    """The family's relation clauses, their private interpretation, the roster paraphrase and 'placed'."""
    kind = family(facts)
    objects = targets(facts)
    mode, count = logical(facts)
    term = facts.terms[0]
    params = term.params
    contact = _flag(term, "require_contact_with")
    detached = ", with the gripper detached" if _released(facts) else ""
    if kind in ("in_container", "ordered_in_container"):
        container = params["container"]
        inside, short = relation_words(facts, container)
        canonical_short = relation_words(facts, container)[1]
        touching = " and touching it" if contact is True and len(objects) == 1 else (
            ", each touching it" if contact is True else "")
        official_touch = ", touching it" if contact is True else ""
        if kind == "ordered_in_container":
            first, second = (phrase(facts, name) for name in objects)
            clauses = [f"Both the {first} and the {second} must end up {inside}, each touching it.",
                       f"The {first} must come to rest {short} before the {second} does: the order is fixed the first "
                       f"time each of them rests there, released, and taking one out again does not undo it."]
            first_c, second_c = (phrase(facts, name, canonical=True) for name in objects)
            return (clauses, f"Both in the {phrase(facts, container)}, touching it, the {first} placed first.",
                    f"the {first_c} and then the {second_c} came to rest {canonical_short}{official_touch}, in that "
                    f"order{detached}", True)
        if mode == "all":
            subject, trailing = subject_all(facts, objects)
            clause = f"{subject} must end up {inside}{touching}"
            clauses = [clause + (f": {listing(facts, objects)}." if trailing else ".")]
            verb = "rests" if len(objects) == 1 else "rest"
            return (clauses, f"{listing(facts, objects)[0].upper()}{listing(facts, objects)[1:]} {short}"
                    f"{official_touch}.", f"{statement_subject(facts, objects)} {verb} {canonical_short}"
                    f"{official_touch}{detached}", True)
        if mode == "any":
            clauses = [f"{subject_any(facts, objects)} must end up {inside}{official_touch}; one of them is enough."]
            return (clauses, f"Any one of {listing(facts, objects)} {short}{official_touch}.",
                    f"{uncap(subject_any(facts, objects))} rests {canonical_short}{official_touch}{detached}", True)
        clauses = [f"Exactly {number(count)} of these must be {short} at the end"
                   + (", each touching it" if contact is True else "") + f": {listing(facts, objects)}; "
                   f"{number(count + 1)} or more there does not complete the task."]
        return (clauses, f"Exactly {number(count)} of {listing(facts, objects)} {short}{official_touch}.",
                f"{counted(facts, objects, count)} {'rests' if count == 1 else 'rest'} {canonical_short}"
                f"{official_touch}{detached}", True)
    if kind == "groups_in_containers":
        parts, canonical_parts = [], []
        for group in params["groups"]:
            names = _as_list(group["object"])
            inside, _short = relation_words(facts, group["container"])
            parts.append((listing(facts, names), inside))
            canonical_parts.append(f"{listing(facts, names, True)} {_short}")
        first, *rest = parts
        clause = f"{first[0][0].upper()}{first[0][1:]} must end up {first[1]}, and " + " and ".join(
            f"{names} {where}" for names, where in rest) + "."
        return ([clause], "Each group in its own container.", "; ".join(canonical_parts) + detached, True)
    if kind == "on_surface":
        reference = params["reference_object"]
        ref = phrase(facts, reference)
        support = f"supported by the {ref} itself with its center over it"
        if mode == "all":
            subject, trailing = subject_all(facts, objects)
            each = "" if len(objects) == 1 else "each "
            clause = f"{subject} must end up resting on the {ref}, {each}{support}"
            clauses = [clause + (f": {listing(facts, objects)}." if trailing else ".")]
            verb = "rests" if len(objects) == 1 else "rest"
            official = f"{statement_subject(facts, objects)} {verb} on the {ref}, supported by it{detached}"
        elif mode == "any":
            clauses = [f"{subject_any(facts, objects)} must end up resting on the {ref}, {support}; one of them is "
                       f"enough."]
            official = f"{uncap(subject_any(facts, objects))} rests on the {ref}, supported by it{detached}"
        else:
            clauses = [f"Exactly {number(count)} of these must end up resting on the {ref}, each {support}: "
                       f"{listing(facts, objects)}; {number(count + 1)} or more there does not complete the task."]
            official = (f"{counted(facts, objects, count)} {'rests' if count == 1 else 'rest'} on the {ref}, "
                        f"supported by it{detached}")
        return clauses, f"Supported by the {ref}, centered over it.", official, True
    if kind == "stacked":
        order = None if params.get("order") in (None, "None") else params["order"]
        names = list(params["objects"])
        if order == "top_to_bottom":
            names.reverse()
        if order and len(names) == 2:
            lower, upper = phrase(facts, names[0]), phrase(facts, names[1])
            return ([f"The {upper} must end up on top of the {lower}, touching it."],
                    f"The {upper} on top of the {lower}, touching it.",
                    f"the {upper} is stacked on top of the {lower}", False)
        if order:
            sequence = ", ".join(f"the {phrase(facts, name)}" for name in names[:-1]) + \
                f", then the {phrase(facts, names[-1])}"
            return ([f"The blocks must form one tower in this order from bottom to top: {sequence}; each one must sit "
                     f"on top of the one below it, touching it."], "One tower in the stated order.",
                    f"{listing(facts, names, True)} are stacked bottom to top in that order, each touching the one "
                    f"below", False)
        if len(names) == 2:
            same = phrase(facts, names[0]) == phrase(facts, names[1])
            pair = (f"the two {plural(phrase(facts, names[0]))}" if same
                    else f"the {phrase(facts, names[0])} and the {phrase(facts, names[1])}")
            return ([f"One of {pair} must end up on top of the other, touching it; either one may be on top."],
                    "One on the other, touching, either order.", f"{pair} are stacked one on the other, touching",
                    False)
        text = listing(facts, names)
        return ([f"{text[0].upper()}{text[1:]} must form a single tower, each one on top of another and touching it, "
                 f"in any order."], "One tower, any order.",
                f"{listing(facts, names, True)} form one stack in any order, each touching the one below", False)
    if kind in ("out_of_container", "out_onto_surface"):
        container = params["container"]
        relation = "off" if container in FLAT_REFERENCES or FRAME_CONTAINERS.get(container) else "out of"
        review = TASK_REVIEW.get(facts.task, {})
        subject, trailing = (review["subject"], False) if "subject" in review else subject_all(facts, objects)
        tail = ", resting directly on the table" if kind == "out_onto_surface" else ""
        where = review.get("container_ref") or f"the {phrase(facts, container)}"
        clause = f"{subject} must end up {relation} {where}{tail}"
        verb = "is" if len(objects) == 1 else "are"
        return ([clause + (f": {listing(facts, objects)}." if trailing else ".")],
                f"Out of the {phrase(facts, container)}" + (" and resting on the table." if tail else "."),
                f"{statement_subject(facts, objects)} {verb} {relation} the {phrase(facts, container)}"
                + (" and supported by the table" if tail else "") + detached, False)
    if kind == "picked_up":
        if len(objects) == 1:
            name = phrase(facts, objects[0])
            many = name in PLURAL_NOUNS
            it, be = ("them", "are") if many else ("it", "is")
            return ([f"The {name} must be held in the gripper and lifted so that all of {it} {be} clear above the "
                     f"table top, still over the table.",
                     f"The task completes while {'they are' if many else 'it is'} held: once set down or dropped, "
                     f"{'they do' if many else 'it does'} not count."],
                    "Held and lifted clear of the table.",
                    f"the {phrase(facts, objects[0], canonical=True)} {'are' if many else 'is'} in contact with the "
                    f"gripper and lifted clear above the table", False)
        groups = _groups(facts, objects)
        kinds = [f"the {plural(text)}" if n > 1 else f"the {text}" for text, n in groups]
        which = (f"the {number(groups[0][1])} {plural(groups[0][0])}" if len(groups) == 1
                 else ", ".join(kinds[:-1]) + " or " + kinds[-1])
        item = "bagel" if all(text == "bagel" for text, _n in groups) else "fruit"
        return ([f"One {item} must be held in the gripper and lifted so that all of it is clear above the table top, "
                 f"still over the table; any of {which} counts.",
                 f"The task completes while it is held: once set down or dropped, it does not count."],
                "Any one named object held and lifted clear of the table.",
                f"one of {which} is in contact with the gripper and lifted clear above the table", False)
    if kind == "upright":
        subject, trailing = subject_all(facts, objects)
        pose = ("upright on its base" if any("jug" in phrase(facts, name) for name in objects)
                else "upright, with its opening facing up" if len(objects) == 1
                else "upright, each with its opening facing up")
        clause = f"{subject} must end up standing {pose}"
        verb = "stands" if len(objects) == 1 else "stand"
        return ([clause + (f": {listing(facts, objects)}." if trailing else ".")], "Upright.",
                f"{statement_subject(facts, objects)} {verb} upright{detached}", False)
    if kind == "relative_position":
        reference = params["reference_object"]
        name, ref = phrase(facts, objects[0]), phrase(facts, reference)
        where = {"object_left_of": f"to the left of the {ref} as seen from the robot",
                 "object_right_of": f"to the right of the {ref} as seen from the robot",
                 "object_in_front_of": f"in front of the {ref}, on the robot's side of it",
                 "object_behind": f"behind the {ref}, on its far side from the robot"}[term.func]
        resting = f", resting on the {phrase(facts, contact)}" if isinstance(contact, str) else ""
        return ([f"The {name} must end up {where}, rather than diagonally beside it{resting}."],
                f"{where[0].upper()}{where[1:]}{resting}.",
                f"the {phrase(facts, objects[0], canonical=True)} is {where}"
                + (f", touching the {phrase(facts, contact)}" if isinstance(contact, str) else "") + detached, False)
    if kind == "centered":
        name = phrase(facts, objects[0])
        return ([f"The {name} must end up at the center of the table, touching the table."],
                "At the table's center, touching it.",
                f"the {phrase(facts, objects[0], canonical=True)} is at the center of the table, touching it{detached}",
                False)
    raise ValueError(kind)  # pragma: no cover - FAMILIES is closed


NEGATIVE_CASES = {
    "in_container": ("Objects on the table beside the container: false.", "beside"),
    "ordered_in_container": ("The second object placed first, then the first: false (latched order).", "wrong_order"),
    "groups_in_containers": ("Groups swapped between the containers: false.", "swapped"),
    "on_surface": ("Objects on the table beside the surface: false.", "beside"),
    "stacked": ("Objects side by side on the table: false.", "beside"),
    "picked_up": ("Object lifted clear of the table but not held: false.", "lifted_not_held"),
    "upright": ("Objects turned upside down: false.", "inverted"),
    "relative_position": ("Object placed on the opposite side of the reference: false.", "opposite"),
}


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def _descriptor_clauses(facts: Facts, entry: Any) -> list[str]:
    """The family's relation clauses with the instruction's descriptor in place of the objects' names."""
    kind = family(facts)
    params = facts.terms[0].params
    mode, _count = logical(facts)
    if kind == "groups_in_containers":
        parts = [(entry[group["container"]][0], relation_words(facts, group["container"])[0])
                 for group in params["groups"]]
        (first, where), *rest = parts
        return [f"{_cap(first)} must end up {where}, and " + " and ".join(f"{text} {place}" for text, place in rest)
                + "."]
    text, size = entry
    each = "" if size == "one" else "each "
    if kind == "in_container" and mode == "all":
        contact = _flag(facts.terms[0], "require_contact_with") is True
        touching = (" and touching it" if size == "one" else ", each touching it") if contact else ""
        return [f"{_cap(text)} must end up {relation_words(facts, params['container'])[0]}{touching}."]
    if kind == "on_surface" and mode == "all":
        ref = phrase(facts, params["reference_object"])
        return [f"{_cap(text)} must end up resting on the {ref}, {each}supported by the {ref} itself with its "
                f"center over it."]
    if kind == "stacked" and params.get("order") in (None, "None"):
        return [f"{_cap(text)} must end up stacked on top of each other, each touching the one below it, in any "
                f"order."]
    if kind == "upright":
        return [f"{_cap(text)} must end up standing upright, {each}with its opening facing up."]
    if kind == "picked_up":
        return [f"{_cap(text)} must be held in the gripper and lifted so that all of it is clear above the table "
                f"top, still over the table; any one counts.",
                "The task completes while it is held: once set down or dropped, it does not count."]
    raise ValueError(f"{facts.task}: no descriptor template for {kind} ({mode})")


def _descriptor_release(entry: Any, placed: bool, mode: str) -> str:
    """Release without a count: 'either one' would say how many objects the descriptor covers."""
    if mode == "choose":
        return release_clause(0, placed, "choose")
    where = " there" if placed else ""
    if not isinstance(entry, dict) and entry[1] == "one":
        return f"Let go of it{where}: the task is not complete while any part of the gripper still touches it."
    return f"Let go of them{where}: the task is not complete while the gripper still touches any of them."


def project(facts: Facts) -> Projection:
    kind = family(facts)
    review = TASK_REVIEW.get(facts.task, {})
    objects = targets(facts)
    mode, count = logical(facts)
    term = facts.terms[0]
    released = _released(facts)
    contact = _flag(term, "require_contact_with")
    main, interpretation, official, placed = _main_clauses(facts)
    described = descriptor(facts)
    if described is not None and "clauses" not in review:
        main = _descriptor_clauses(facts, described)
    clauses = list(main)
    release_index = None
    if released:
        release_index = len(clauses)
        if described is not None:
            clauses.append(_descriptor_release(described, placed, mode))
        elif len(objects) == 1:
            clauses.append(_single_release(facts, objects[0], placed))
        else:
            clauses.append(release_clause(len(objects), placed, mode if mode in ("any", "choose") else "all"))
    relation_indices = list(range(len(main)))
    if "clauses" in review:
        clauses = list(review["clauses"])
        intent_indices = set(review.get("intent", []))
        release_index = next((index for index, text in enumerate(clauses) if text.startswith("Let go")), None)
        # A clause may state the scored relation and the unscored intent at once ("out of the bowl, on the
        # table"); the entry's "relation" then names the clauses that carry the relation.
        relation_indices = list(review.get("relation") or [index for index in range(len(clauses))
                                                            if index not in intent_indices and index != release_index])
    contact_text = (f", plus contact with the {'container' if contact is True else contact}"
                    if contact and kind not in ("relative_position",) else "")
    if kind == "relative_position" and isinstance(contact, str):
        contact_text = f", plus contact with the {contact}"
    conditions = [_condition("c1", "; ".join(f"{item.name} = {_call(item)}" for item in facts.terms) + ": "
                             + MECHANISM[kind] + contact_text + ".", interpretation, relation_indices)]
    if release_index is not None:
        conditions.append(_condition("c2", RELEASE_BASIS + ("" if kind != "relative_position" else
                                                            "; with release required the level check is skipped")
                                     + ".", "Release: the gripper touches none of the named objects.",
                                     [release_index]))
    if kind == "stacked":
        conditions.append(_condition(
            "scoring_gap", "The success term is stacked only; it has no gripper_detached term, and check_stacked also "
            "accepts an upper object whose box is above the lower object's bottom face (nesting).",
            "The termination does not require releasing the top object.", [], "withheld_private",
            "A predicate weakness is documented for review, not delivered as an exploit; the instruction's intent (a "
            "stack) is unchanged."))
    if described is not None:
        texts = [text for text, _size in (described.values() if isinstance(described, dict) else [described])]
        conditions.append(_condition(
            "object_identity", f"The instruction {facts.instruction!r} refers by descriptor; the termination names "
            f"{sorted(set(objects))}.", "Which objects the descriptor covers is the task's own inference.", [],
            "withheld_private", "Naming the objects would answer the task (no privileged information); the public "
            f"clauses keep the instruction's descriptor ({'; '.join(texts)})."))
    if review.get("intent_clause"):
        clauses.append(review["intent_clause"])
    intent = sorted(set(review.get("intent", [])) | ({len(clauses) - 1} if review.get("intent_clause") else set()))
    if intent:
        conditions.append(_condition("intent", f"Instruction: {facts.instruction!r}", "Instruction intent the "
                                     "termination does not score.", intent, "task_intent_only"))
    if review.get("gap"):
        conditions.append(_condition("instruction_gap", review["gap"], "Scoring gap between instruction and "
                                     "termination.", [], "withheld_private",
                                     "Recorded for review; the public text keeps the instruction's intent."))
    conditions.append(_condition("private_limits", f"conditionals.{term.func} and its predicate_logic helpers.",
                                 f"{PRIVATE_LIMITS[kind]} withheld.", [], "withheld_private",
                                 "Numeric limits stay off-policy."))
    counter = [{"case": "The reset state: false.", "expected": "pending phase-B witness case reset_state"}]
    if kind in NEGATIVE_CASES:
        text, case = NEGATIVE_CASES[kind]
        counter.append({"case": text, "expected": f"pending phase-B witness case {case}"})
    if mode == "choose":
        counter.append({"case": f"{number(count + 1).capitalize()} of them in place: false (exactly {number(count)}).",
                        "expected": "pending phase-B witness case one_too_many"})
    if mode == "all" and len(set(objects)) > 1 and kind in ("in_container",):
        counter.append({"case": "All but one in place: false.", "expected": "pending phase-B witness case all_but_one"})
    if released:
        counter.append({"case": "Objects in place while one is still gripped: false by gripper_detached "
                        "(source-derived, not executed).", "expected": "source-derived"})
    if kind == "stacked" and len(params_objects(facts)) > 2 and term.params.get("order") not in (None, "None"):
        counter.append({"case": "The same objects stacked in another order: false.",
                        "expected": "pending phase-B witness case wrong_order"})
    return Projection(clauses=clauses, conditions=conditions, official_success=official, counterexamples=counter)


def params_objects(facts: Facts) -> list[str]:
    return list(facts.terms[0].params.get("objects") or [])



__all__ = ["FLAT_REFERENCES", "FRAME_CONTAINERS", "INSTRUCTION_DESCRIPTORS", "MECHANISM", "PRIVATE_LIMITS",
           "Projection", "TASK_REVIEW", "descriptor", "listing", "number", "phrase", "plural", "project",
           "refers_by_descriptor"]
