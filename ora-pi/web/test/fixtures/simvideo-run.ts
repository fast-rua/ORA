// A run directory around a slice of a real RoboDojo execution capture taken at the play
// backend's simulation-video settings (fixtures/simvideo/: toys, a scripted sweep). The
// operator camera is 1280x720 every 9 physics steps (0.036 s of simulated time) and the robot's
// three cameras 640x480 in the same frames. The slice holds frame 0 (the scene's reset) and
// frames 14-51: the end of `simvideo-above`, then `simvideo-start`, `simvideo-sweep` (130
// settle steps after its motion, the objects it swept still moving) and `simvideo-retreat`
// (50 settle steps). Settling frames are `control` frames with their action's id; each
// action's `action_end` shares its simulated time with the next one's `action_start`. The
// post-action settle records are the run's own; the manifest leaves out the run's identity and
// no PNG is included (the tests read the timeline). The run has no session or Supervisor events.

import { copyFileSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const FIXTURE = resolve(import.meta.dirname, "simvideo");

export function simvideoRun(): string {
  const root = join(mkdtempSync(join(tmpdir(), "ora-simvideo-test-")), "simvideo-toys");
  const capture = join(root, "native", "execution-capture");
  mkdirSync(join(capture, "frames"), { recursive: true });
  copyFileSync(join(FIXTURE, "timeline.jsonl"), join(capture, "timeline.jsonl"));
  copyFileSync(join(FIXTURE, "manifest.json"), join(capture, "manifest.json"));
  copyFileSync(join(FIXTURE, "post-action-settle.jsonl"), join(root, "native", "post-action-settle.jsonl"));
  writeFileSync(join(root, "plan.json"), JSON.stringify({ episode_mode: "play", task: "toys" }));
  return root;
}
