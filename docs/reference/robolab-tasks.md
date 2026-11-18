# RoboLab tasks

ORA admits all 120 tasks of RoboLab-120 at the pinned revision `ad45d4f9`
(`ora/environment/benchmarks/robolab.py`, decided 2026-11-05). Admission and a
delivered completion contract are separate states:

- **Admitted.** A RunSpec may name the task. The instruction is the task's
  upstream default instruction; the official limit is its `episode_length_s`
  at 15 Hz. Every seed resets to the same authored layout: the environment's
  only reset event is `reset_scene_to_default` and no task adds one, so a seed
  varies the physics, not the scene.
- **Reviewed contract.** The registry's `tasks` hold the public completion
  contracts the resolver delivers: all 120 tasks. Three were reviewed by hand
  on 2026-10-18; 97 generated contracts were promoted on 2026-11-05 after their
  seed-0 witness matched every case, and the other 20 on 2026-11-16, once their
  reviews' witness plans stated a placement or a grasp the generated plan could
  not reach (each review's `witness_plan.plan_fix` says why).
- **Pending review.** A generated contract whose witness has a case that did
  not match stays in the registry's `pending_review` (none at present). Its
  review keeps `status: needs_review` with those witness checks `failed`. The
  resolver withholds such a contract (`task_contract.reason:
  contract_review_pending`) and launch preflight fails the
  `robolab_task_contract` check, so a pending task composes but does not
  launch.

## How a contract is made

`ora dev robolab-tasks generate --source-root $ROBOLAB_ROOT` reads every task
class by AST (no simulator import), checks it against
`robolab/tasks/_metadata/task_metadata.json`, and writes the roster block, one
`<Task>-completion-v1.md` and one `reviews/<Task>-review-v1.json` per pending
task, the registry's `pending_review` and the table below; `--check` reports
drift instead. RoboLab's verdict is the termination manager's `terminated`
flag, the OR of every non-timeout termination term, so the contract restates
those terms and never the partial `subtasks` scores; each review lists where
the two differ (`termination_vs_subtasks`).

The predicate families and what their public clauses say:

| Family | Termination | Public clauses |
|---|---|---|
| `in_container` | `object_in_container` (`all`, `any` or exactly `K`), `object_inside` | the objects inside the container (on a shelf frame or a closed box: on it), touching it where required; released |
| `groups_in_containers` | `object_groups_in_containers` | each group in its own container; released |
| `on_surface` | `object_on_top` | resting on the surface, supported by it with its center over it; released |
| `stacked` | `stacked` | one object on top of the other, touching, in the stated or any order (the term requires no release; that gap stays private) |
| `out_of_container`, `out_onto_surface` | `object_outside_of`, `object_outside_of_and_on_surface` | out of the container (and resting directly on the table); released |
| `picked_up` | `object_picked_up`, or several of them OR-ed | held in the gripper and lifted clear above the table; the task completes while it is held |
| `upright` | `object_upright` | standing upright; released |
| `ordered_in_container` | `objects_placed_in_container_in_order` | both in the container; the first placement order is latched and cannot be undone |
| `relative_position` | `object_left_of` and its three siblings | left of, right of, in front of or behind the reference as seen from the robot; released |
| `centered` | `object_on_center` | standing at the table's center, touching it; released |

Clauses name objects in natural language, never by identifier. An
instruction that leaves the policy to infer which objects it means, by
category, attribute, comparison or property ("the black items", "the larger
object", "the green spoons", "the serving spoons with no holes"), keeps that
descriptor in its contract: naming the objects would answer the task, so the
names stay in the private review (`object_identity`, withheld) and the release
clause gives no count. `task_wording.refers_by_descriptor` selects these 39
tasks from the instruction text and `INSTRUCTION_DESCRIPTORS` holds each
descriptor in the instruction's own words; the generator refuses a task the
two disagree on. An instruction that names its objects concretely ("the
banana", "the yellow block", "the red mug") keeps the names.

A review fills every check it can verify from source: predicate and helpers,
reset and randomization (with the reset relations a contract must mention,
such as a banana that starts in the crate), assets and frames, success timing,
no expert strategy and privacy. `success_witness`, `counterexamples`,
`public_observability` and, for `upright`, `assets_and_frames` wait for the
seed-0 witness. Each review also carries its `compatibility` (ready, ready with
note or blocked, with reasons) and the `witness_plan` the witness runs.

## Witness and promotion

`ora qualify robolab contract-witness --task T ... --gpu N --evidence-root DIR`
runs each task's witness plan in the RoboLab worker, one Kit process at a time:
every case starts from a fresh reset of seed 0, moves objects by direct state
edits (and, for the pick-up family, closes the gripper on the object), then
holds the robot and reads RoboLab's own success termination. A placement turns
the object to fit the opening (a quarter turn, or into a container on its long
axis), takes the free spot nearest the middle and sets it low inside the walls
or on the surface, piling only when nothing is free; a placement beside an
object on a table with no free stretch parks the unnamed clutter of the nearest
stretch off the table and records it as the case's `parked`. The pick-up grasp
waits for the jaw to start closing before it calls the jaw stalled. An
exact-count negative sets K+1 objects down at once, so its verdict counts from
tick 15 (`ignored_ticks`), after they land; `all_but_one` keeps the n-1 objects
nearest the container inside (`count_in`). Where the generated placement cannot
reach the state a case needs, a review's plan may state it instead: `put` sets
an object down at a stated spot (env-frame `center`, or a `reference` object's
centroid moved by `dx`/`dy`), with its lowest point at `table`, `top:<object>`,
`floor:<object>[+dz]` or a height, turned `own`, `yaw:D`, `flat:D` (longest axis
level along yaw D), `long_up` or `tilt:T:A`; `park` moves named clutter off the
table; and `grasp` may name the object's axes for the jaw line, the approach and
across the pads (`order`), how deep along the approach it sits (`depth`), `flip`
it end for end and `shift` it along the jaw line (a pitcher held by its rim).
Such a plan carries a `plan_fix` note that says why. RoboLab latches success at
its first true step, so an exact-count negative whose objects land one at a time
can pass through K and be latched despite the landing window; such a plan sets
all K+1 down in one step (`put` with `settle: false`). It also projects
every named object into the two public cameras at reset. It writes
`DIR/outputs/ora-backends/task-contract-witness/robolab/<Task>-s0-witness.json`
(`ora.robolab-task-contract-witness.v2`) and fills the review's evidence and
witness checks with that file's sha256; a later witness of the task replaces an
earlier run's `witness_worker_failed` or `solved_at_reset` blocker. A RoboLab
worker ends its process with `os._exit` instead of closing Kit
([RoboLab](../environments/robolab.md)). `ora dev robolab-tasks promote` then
moves every review whose nine checks passed, and which is not blocked, into the
registry's `tasks`; run `./scripts/ora lock` and update
`task_contracts.robolab.task_count` and `pending_review_count` in
`tests/data/default-expectations.json` in the same change.

## The roster

`ora spec robolab tasks` prints the same state as JSON lines. Compatibility
codes: `long_horizon` (official limit at least twice the 900 native actions of
the longest task admitted before), `longer_horizon` (above 900),
`many_objects` (five or more objects to move; the default 80 robot decisions
may bind first), `exact_count` (exactly K, one more fails), `wide_object` (no
face fits the open 85 mm gripper), `reorientation`, `order_latched`,
`floor_rack` (the storage rack stands on the floor beside the table),
`remote_scene_geometry` (the scene loads the purple crate from the Omniverse
content server), `scoring_gap` (the termination is weaker than the
instruction; recorded in the private review) and `reset_relation`.

<!-- generated: ora dev robolab-tasks generate -->
| Task | Family | Episode (s) | Official limit | Contract | Compatibility | Notes |
|---|---|---|---|---|---|---|
| `AnimalsInBinTask` | in_container | 90 | 1350 | reviewed | ready with note | longer_horizon, wide_object |
| `AppleAndYogurtInBowlTask` | in_container | 120 | 1800 | reviewed | ready with note | long_horizon |
| `BBQSauceInBinTask` | in_container | 90 | 1350 | reviewed | ready with note | longer_horizon |
| `BagelsOnPlateTask` | on_surface | 60 | 900 | reviewed | ready | - |
| `BananaInBowlTask` | in_container | 50 | 750 | reviewed | ready | - |
| `BananaOnPlateTask` | on_surface | 40 | 600 | reviewed | ready | - |
| `BananaThenRubiksCubeTask` | ordered_in_container | 60 | 900 | reviewed | ready with note | order_latched |
| `BananasInBinOneMoreTask` | in_container | 60 | 900 | reviewed | ready with note | exact_count, scoring_gap, reset_relation |
| `BananasInBinThreeTotalTask` | in_container | 60 | 900 | reviewed | ready with note | exact_count, scoring_gap, reset_relation |
| `BananasInCrateTask` | in_container | 60 | 900 | reviewed | ready with note | remote_scene_geometry, exact_count, reset_relation |
| `BananasOutOfBinTask` | out_of_container | 90 | 1350 | reviewed | ready with note | longer_horizon, reset_relation |
| `BigPumpkinInBinTask` | in_container | 60 | 900 | reviewed | ready with note | wide_object |
| `BlackItemsInBinTask` | in_container | 120 | 1800 | reviewed | ready with note | long_horizon, many_objects, reset_relation |
| `BlockStackingOrderAgnosticTask` | stacked | 90 | 1350 | reviewed | ready with note | longer_horizon, scoring_gap |
| `BlockStackingSpecifiedOrderTask` | stacked | 90 | 1350 | reviewed | ready with note | longer_horizon, scoring_gap |
| `BlocksInBinTask` | in_container | 150 | 2250 | reviewed | ready with note | long_horizon |
| `BowlInBinTask` | in_container | 60 | 900 | reviewed | ready | - |
| `BowlStackingLeftOnRightTask` | in_container | 20 | 300 | reviewed | ready | - |
| `BowlStackingRightOnLeftTask` | in_container | 20 | 300 | reviewed | ready | - |
| `ButterAboveRaisinTask` | on_surface | 40 | 600 | reviewed | ready | - |
| `CannedFoodInBinTask` | in_container | 60 | 900 | reviewed | ready | - |
| `ClampInRightBinTask` | in_container | 60 | 900 | reviewed | ready | - |
| `CleanUpToysTask` | in_container | 300 | 4500 | reviewed | ready with note | long_horizon, many_objects, wide_object, scoring_gap |
| `ClearOrganicObjectsTask` | in_container | 240 | 3600 | reviewed | ready with note | long_horizon, many_objects, wide_object |
| `ClutterPlasticTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon, wide_object |
| `ClutterPumpkinTask` | in_container | 90 | 1350 | reviewed | ready with note | longer_horizon, wide_object |
| `CoffeePotInBinTask` | in_container | 60 | 900 | reviewed | ready | - |
| `CondimentsInBinTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon |
| `CookingClearPlateTask` | out_of_container | 180 | 2700 | reviewed | ready with note | long_horizon, reset_relation |
| `CookingPickPastaToolTask` | in_container | 60 | 900 | reviewed | ready | reset_relation |
| `CubesAndBlocksInBinTask` | in_container | 240 | 3600 | reviewed | ready with note | long_horizon, many_objects |
| `DishesInBinTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon, wide_object |
| `ElectronicsInBinTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon, reset_relation |
| `FoodPacking1BoxesTask` | in_container | 60 | 900 | reviewed | ready | - |
| `FoodPacking1CansTask` | in_container | 60 | 900 | reviewed | ready | - |
| `FoodPacking2BoxesTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon |
| `FoodPacking2CansTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon |
| `FoodPacking3BoxesTask` | in_container | 240 | 3600 | reviewed | ready with note | long_horizon |
| `FoodPacking3CansTask` | in_container | 240 | 3600 | reviewed | ready with note | long_horizon |
| `FoodPackingByColorTask` | groups_in_containers | 120 | 1800 | reviewed | ready with note | long_horizon, wide_object |
| `FruitsGreenLimesOnPlateTask` | on_surface | 90 | 1350 | reviewed | ready with note | longer_horizon |
| `FruitsMovingOrangeOrLimeTask` | in_container | 60 | 900 | reviewed | ready | - |
| `FruitsMovingTask` | in_container | 60 | 900 | reviewed | ready | - |
| `FruitsOnPlate3Task` | on_surface | 200 | 3000 | reviewed | ready with note | long_horizon, exact_count, wide_object, reset_relation |
| `FruitsOnPlateTask` | on_surface | 300 | 4500 | reviewed | ready with note | long_horizon, many_objects, wide_object, reset_relation |
| `FruitsOnionTask` | in_container | 60 | 900 | reviewed | ready | - |
| `FruitsOnionToPlateTask` | on_surface | 60 | 900 | reviewed | ready | - |
| `FruitsOrangesOnPlateTask` | on_surface | 90 | 1350 | reviewed | ready with note | longer_horizon, reset_relation |
| `GrabABagelTask` | picked_up | 30 | 450 | reviewed | ready | - |
| `GrabAFruitTask` | picked_up | 30 | 450 | reviewed | ready | - |
| `GreenSpoonsInPotTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon |
| `HammersInLeftBinTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon |
| `JugsOnShelfTask` | in_container | 120 | 1800 | reviewed | ready with note | long_horizon, floor_rack, wide_object |
| `KeyboardOutOfBinTask` | out_onto_surface | 60 | 900 | reviewed | ready | reset_relation |
| `LargerObjectRaisinBoxInBinTask` | in_container | 30 | 450 | reviewed | ready | - |
| `MarkerInMugTask` | in_container | 40 | 600 | reviewed | ready | - |
| `MouseOnKeyboardTask` | on_surface | 60 | 900 | reviewed | ready | - |
| `MoveBananaToBagelPlateTask` | on_surface | 90 | 1350 | reviewed | ready with note | longer_horizon |
| `MustardAboveRaisinTask` | on_surface | 40 | 600 | reviewed | ready | - |
| `MustardInLeftBinTask` | in_container | 30 | 450 | reviewed | ready | - |
| `MustardInRightBinTask` | in_container | 30 | 450 | reviewed | ready | - |
| `NonHammerToolsInRightBinTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon |
| `OneBottleInSquarePailTask` | in_container | 60 | 900 | reviewed | ready | - |
| `OneBottleOnShelfTask` | in_container | 60 | 900 | reviewed | ready with note | floor_rack |
| `PhoneOrRemoteInBinTask` | in_container | 60 | 900 | reviewed | ready | - |
| `PickDrillTask` | picked_up | 40 | 600 | reviewed | ready | - |
| `PickGlassesTask` | picked_up | 30 | 450 | reviewed | ready | - |
| `PickOrangeObjectTask` | picked_up | 60 | 900 | reviewed | ready | - |
| `PickUpBluePitcherTask` | picked_up | 30 | 450 | reviewed | ready with note | wide_object |
| `PickUpGreenObjectTask` | picked_up | 30 | 450 | reviewed | ready | - |
| `PinkSpoonInPotTask` | in_container | 60 | 900 | reviewed | ready | - |
| `PlasticBottlesInSquarePailTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon |
| `PutBowlOnShelfTopTask` | on_surface | 60 | 900 | reviewed | ready | - |
| `PutMugsOnShelfTask` | on_surface | 180 | 2700 | reviewed | ready with note | long_horizon, wide_object |
| `PutTwoMugsOnShelfTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon, exact_count, wide_object |
| `RecycleCartonTask` | in_container | 90 | 1350 | reviewed | ready with note | longer_horizon |
| `RecycleCartonsOnBoxTask` | in_container | 90 | 1350 | reviewed | ready with note | longer_horizon |
| `RecycleCartonsVerticalCrateTask` | in_container | 90 | 1350 | reviewed | ready with note | longer_horizon |
| `RedDishesInBinTask` | in_container | 60 | 900 | reviewed | ready | - |
| `RedItemsInBinTask` | in_container | 60 | 900 | reviewed | ready | - |
| `ReorientAllMugsTask` | upright | 90 | 1350 | reviewed | ready with note | longer_horizon, reorientation, wide_object, reset_relation |
| `ReorientJugTask` | upright | 60 | 900 | reviewed | ready with note | reorientation, wide_object |
| `ReorientRedMugTask` | upright | 60 | 900 | reviewed | ready with note | reorientation |
| `ReorientWhiteMugsTask` | upright | 60 | 900 | reviewed | ready with note | reorientation, wide_object, reset_relation |
| `RubiksCubeAndBananaTask` | in_container | 60 | 900 | reviewed | ready | - |
| `RubiksCubeBehindBowlTask` | relative_position | 30 | 450 | reviewed | ready | - |
| `RubiksCubeInFrontOfBowlTask` | relative_position | 30 | 450 | reviewed | ready | - |
| `RubiksCubeLeftOfBowlTask` | relative_position | 30 | 450 | reviewed | ready | - |
| `RubiksCubeOrBananaTask` | in_container | 30 | 450 | reviewed | ready | - |
| `RubiksCubeRightOfBowlTask` | relative_position | 30 | 450 | reviewed | ready | - |
| `RubiksCubeTask` | in_container | 40 | 600 | reviewed | ready | - |
| `RubiksCubeThenBananaTask` | ordered_in_container | 60 | 900 | reviewed | ready with note | order_latched |
| `RubiksCubesInBinTask` | in_container | 120 | 1800 | reviewed | ready with note | long_horizon |
| `SauceBottlesCrateTask` | in_container | 40 | 600 | reviewed | ready with note | remote_scene_geometry |
| `SmallPumpkinInBinTask` | in_container | 60 | 900 | reviewed | ready | - |
| `SmallerObjectButterInBinTask` | in_container | 30 | 450 | reviewed | ready | - |
| `SmartphoneInBinTask` | in_container | 60 | 900 | reviewed | ready | - |
| `SpoonInMugTask` | in_container | 60 | 900 | reviewed | ready | reset_relation |
| `SpoonsInPotTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon |
| `Stack3RubiksCubeTask` | stacked | 60 | 900 | reviewed | ready with note | scoring_gap |
| `StackWhiteMugsTask` | stacked | 60 | 900 | reviewed | ready with note | wide_object, scoring_gap |
| `StackYellowOnRedTask` | stacked | 60 | 900 | reviewed | ready with note | scoring_gap |
| `TakeMeasuringSpoonOutTask` | out_of_container | 40 | 600 | reviewed | ready with note | scoring_gap, reset_relation |
| `TakeMugsOffOfShelfTask` | out_onto_surface | 180 | 2700 | reviewed | ready with note | long_horizon, wide_object, reset_relation |
| `TakeSpatulaOffShelfTask` | out_onto_surface | 60 | 900 | reviewed | ready | reset_relation |
| `ThrowAwayAppleTask` | in_container | 60 | 900 | reviewed | ready | - |
| `ThrowAwaySnacksTask` | in_container | 120 | 1800 | reviewed | ready with note | long_horizon |
| `ToolOrganizationBothTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon, scoring_gap |
| `ToolOrganizationTask` | in_container | 180 | 2700 | reviewed | ready with note | long_horizon |
| `ToolsPickingAllHammersTask` | out_onto_surface | 240 | 3600 | reviewed | ready with note | long_horizon, reset_relation |
| `ToolsPickingDrillTask` | out_onto_surface | 60 | 900 | reviewed | ready with note | scoring_gap, reset_relation |
| `ToolsPickingHammerTask` | out_onto_surface | 60 | 900 | reviewed | ready | reset_relation |
| `ToyInBinTask` | in_container | 60 | 900 | reviewed | ready with note | wide_object |
| `UnstackRubiksCubeTask` | out_onto_surface | 90 | 1350 | reviewed | ready with note | longer_horizon, reset_relation |
| `UtensilsInMugTask` | in_container | 90 | 1350 | reviewed | ready with note | longer_horizon |
| `WhiteMugInCenterOfTableTask` | centered | 30 | 450 | reviewed | ready with note | wide_object |
| `WhiteMugsInBinTask` | in_container | 60 | 900 | reviewed | ready with note | wide_object |
| `WoodSpatulaToBowlTask` | in_container | 60 | 900 | reviewed | ready | - |
| `YellowAndWhiteObjectsInBinTask` | in_container | 60 | 900 | reviewed | ready with note | wide_object |
| `YogurtInBowlTask` | in_container | 40 | 600 | reviewed | ready | - |
<!-- end of generated table -->
