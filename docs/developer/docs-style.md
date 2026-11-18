# Documentation style guide

How ORA's documentation pages are organised and written. It applies to every page under
`docs/` and to the pages the docs site generates. [Maintain the documentation](documentation.md)
covers the build tools; this page covers what goes on a page.

## Sections and where a page belongs

The sidebar (`docs-site/navigation.json`) is organised by what the reader is trying to do:

| Section | Holds | Example |
| --- | --- | --- |
| Get started | Installing, connecting a model, the first run | Installation, First episode |
| How ORA works | The one concept page every other page can link to | How ORA works |
| Use ORA | Task-named guides for people running ORA | Free play, Run ORA safely |
| Environments | One page per simulator, in the fixed layout below | RoboTwin |
| Robots and tasks | Generated embodiment and task pages, plus frames and units | ARX X5, RoboDojo tasks |
| Evaluation | What decides success, how runs are counted and reported | Evaluation protocol |
| Reference | Exact contracts: CLI, configuration, schemas, Python API | Policy I/O |
| Contribute | Extending ORA and maintaining it | Tool surfaces, this guide |

Name a guide by the reader's task ("Run ORA safely", "Script the robot"), not by the feature.
Feature names belong in Reference.

## One fact, one place

Every fact has one home page; other pages link to it. Anything the code already knows is
generated, never typed: task lists and instructions, completion contracts, embodiment
manuals, tool lists, counts. `docs-site/scripts/generate-robot-pages.mjs` builds the
Robots and tasks pages from the same files the runtime reads.

## Success

Success is only the environment's official verdict: RoboTwin's `check_success`, LIBERO
termination, RoboDojo's official result. A policy's own `finish_episode` claim never counts.
Any page that mentions success says which verdict decides it. Synthetic has no verdict
(`official_success: null`).

## Page templates

### Environment page

1. **Overview**: the robot, the task set, the hardware it needs and the verdict source, in
   two or three sentences or cards.
2. **Tasks**: a table, or a link to the generated task page.
3. **What the policy sees and does**, as four rows:

   | | |
   | --- | --- |
   | Observation | Cameras, calibration and robot state delivered each step |
   | Action | The supervised action path and its contract version |
   | Success | The official verdict that decides it |
   | Task prompt | Where the instruction comes from |

4. **Install**, then **Run**, then **View results**, then **Reproduction** and
   **Troubleshooting**, in that order. Leave out a section only when it does not apply.

### Task page (generated)

Public instruction, robot, completion contract in plain words, limits (steps or decisions),
seed or layout protocol, and the official verdict that decides success.

### Embodiment page (generated)

The public robot manual: topology, controlled points, frames and units, gripper semantics,
cameras, and the environment boundary paired with it.

### Results

Every rate carries its denominator (`147/773`, not only `19.0%`). Failures are split into
task failure, timeout or budget, and infrastructure fault. Say which cases are excluded and why.

### Troubleshooting entry

Title the entry with the exact error message or symptom, then three short parts:
**Symptoms**, **Cause**, **What to try**.

## Writing

- Plain sentences, one idea each. Prefer a table to a paragraph of parallel facts.
- Every page that runs something states the expected output and what to do when a step fails.
- Say what a command does before showing it. Put one command per code block when readers
  copy them one at a time.
- Use relative links to the Markdown file (`../usage/play.md`), never site URLs.
- Units on every number: metres, radians, seconds, millimetres for depth.
- Never write host names, cluster or node names, internal storage paths, user or session
  names. The public release scan rejects them.

## Checks

`npm run validate --prefix docs-site` rejects broken links, anchors and assets.
`npm run test:content --prefix docs-site` fails when a page is missing from the sidebar or a
sidebar entry has no page, and when page syntax survives that the conversion does not handle.
