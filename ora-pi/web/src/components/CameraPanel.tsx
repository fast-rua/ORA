// The robot's cameras beside the conversation.
//
//   Live    what the world looks like now: the robot's recorded motion as it happens
//           (native/execution-capture, played at its frame rate, a moment behind), resting on
//           the newest motion's last frame between actions; ORA's world view of the run
//           (ora.orchestrator.world_view) or the latest step's images when there is no recorded motion.
//   Now     (a replay) the world at the replay's clock: the capture frame taken last before it,
//           every camera of it, and the operator view when that was taken later
//   Step N  one step: the motion its action ran (auto-played, with a scrubber), or what the
//           model saw then, with the selected tool call's grounding boxes or marked points.
// When the environment gives an operator view (a camera for the human, never the policy:
// native/operator-view), it is the main picture. Every other camera is shown at the same
// moment in a row under it, each named in its corner (a click makes it the main one, as the
// camera chips do); the images a call gave the model lie over the main picture's lower-left
// corner (live: the newest step's; a step: the selected call's, each to show whole). Over it
// all, the instruction the robot works on. A timeline of the run's steps moves or replays
// through them.
//
// The picture sits in a box of one size and shape, as large as the panel allows: the shape is
// the run's operator camera's (16:9 for a 1280x720 one, else 4:3), fixed for the whole run, and
// every image is fitted into it whole, whatever its source and size (a camera, a wrist camera,
// the operator view, a motion frame, an image a tool returned); the bars around it keep their
// heights, so nothing moves when the picture changes.

import { useCallback, useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";

import { motionCamera, motionSteps, NO_FRAMES, operatorViewAt, useStepFrames, type FramesSummary,
  type StepFrames, type StepMotion } from "../lib/frames.ts";
import { t, useLang } from "../lib/i18n.ts";
import { followLive, type LiveView } from "../lib/live.ts";
import { cameraLabel, stepImage, toolCamera, type Conversation, type StepImage, type ToolView } from "../lib/messages.ts";
import type { Activity } from "../lib/play.ts";
import { Icon } from "./Icons.tsx";
import { ImageStage } from "./ImageStage.tsx";
import { MotionPlayer, type MotionImage } from "./MotionPlayer.tsx";
import { StepTimeline } from "./StepTimeline.tsx";

export type PanelFocus = { mode: "live" } | { mode: "step"; step: number; tool: string | null };

function useLive(enabled: boolean): LiveView | null {
  const [view, setView] = useState<LiveView | null>(null);
  useEffect(() => {
    if (!enabled) return;
    return followLive(setView);
  }, [enabled]);
  return view;
}

const LIVE_SHOWS_FRAMES = new Set(["executing", "settling", "holding", "awaiting_capture", "finished"]);

function motionImages(motion: StepMotion | null, camera: string | null): MotionImage[] {
  if (motion === null || camera === null) return [];
  return motion.frames.flatMap((frame) => (frame.images[camera] ? [{ src: frame.images[camera], simTimeS: frame.simTimeS }] : []));
}

/** A camera's name as its picture's corner says it ("wrist left", "head camera"). */
function cameraName(name: string): string {
  return cameraLabel(name).replace(/_/g, " ");
}

/** Another camera at the same moment, in the row under the main picture; clicking it makes it the main one. */
function CameraThumb({ src, label, onClick }: { src: string; label: string; onClick: () => void }) {
  return (
    <button type="button" className="camera-thumb" onClick={onClick} title={label} aria-label={label}>
      <img src={src} alt={label} draggable={false} />
      <span className="camera-name">{label}</span>
    </button>
  );
}

/** Calls whose images are the cameras' own views (the row under the main picture shows those already). */
const CAMERA_TOOLS = new Set(["observe"]);

/** The newest call of the instruction now running that gave the model images (the live picture's inset shows them). */
function imageTool(conversation: Conversation): ToolView | null {
  for (let index = conversation.items.length - 1; index >= 0; index--) {
    const item = conversation.items[index];
    if (item.kind === "user") break;
    if (item.kind !== "assistant") continue;
    const found = [...item.tools].reverse().find((tool) => tool.images.length > 0 && !CAMERA_TOOLS.has(tool.name));
    if (found) return found;
  }
  return null;
}

export function CameraPanel({ conversation, focus, camera, onCamera, onFocus, replaying, onReplay, liveWorld,
  framesVersion = 0, activity = null, summary = NO_FRAMES, world, stepFrames = null, aspect = 4 / 3, instruction = null,
  onPicture, foot = null }: {
  conversation: Conversation;
  focus: PanelFocus;
  camera: string | null;
  onCamera: (camera: string) => void;
  onFocus: (focus: PanelFocus) => void;
  replaying: boolean;
  /** The timeline's step walk; null leaves it out (a replay has its own clock). */
  onReplay: ((on: boolean) => void) | null;
  /** Whether to follow the world view (off in tests and when the page has no world view). */
  liveWorld: boolean;
  /** Bumped when a new motion or operator view may exist (state.ts). */
  framesVersion?: number;
  /** What the agent does now (the robot moving: its motion streams in). */
  activity?: Activity | null;
  /** The run's recorded motion and operator view (ora_get_frames). */
  summary?: FramesSummary;
  /** A replay's world at its clock (lib/replay.ts worldAt), instead of the live world view. */
  world?: LiveView | null;
  /** A replay's frames of one step (lib/replay.ts stepFrames), instead of asking the run. */
  stepFrames?: ((step: number) => StepFrames | null) | null;
  /** The camera box's shape (width / height), the run's operator camera's, fixed for the run. */
  aspect?: number;
  /** The instruction the robot works on, over the cameras. */
  instruction?: string | null;
  /** A camera picture's shape (width / height) once the main picture has loaded one. */
  onPicture?: (aspect: number) => void;
  /** Under the timeline, at the panel's foot (free play's speed: components/PerfPanel.tsx). */
  foot?: ReactNode;
}) {
  useLang();
  const recorded = world !== undefined;
  const followed = useLive(liveWorld && !recorded);
  const live = recorded ? world : followed;
  const [shown, setShown] = useState<number | null>(null); // an image the selected tool returned, by index
  const [still, setStill] = useState(false); // step mode: what the model saw instead of the motion
  const [frameShown, setFrameShown] = useState<{ index: number; count: number } | null>(null);
  const onFrame = useCallback((index: number, count: number) => setFrameShown({ index, count }), []);
  const latest = conversation.order.length ? conversation.order[conversation.order.length - 1] : null;
  const step = focus.mode === "step" ? focus.step : null;
  const tool: ToolView | null = focus.mode === "step" && focus.tool ? conversation.tools.get(focus.tool) ?? null : null;
  useEffect(() => setShown(null), [focus]);
  useEffect(() => setStill(false), [step]);

  const view = summary.operatorView;
  const viewCamera = view?.camera ?? null;
  const withMotion = motionSteps(summary);
  const newestMotion = withMotion.length ? withMotion[withMotion.length - 1] : null;
  const moving = activity?.kind === "moving" && Number.isInteger(activity.step) ? Number(activity.step) : null;
  // The action that executed last stays the live one until a newer motion is known (no flash of an older one).
  const [recent, setRecent] = useState<number | null>(null);
  useEffect(() => {
    if (moving !== null) setRecent(moving);
  }, [moving]);
  const newest = recent !== null && (newestMotion === null || recent >= newestMotion) ? recent : newestMotion;
  // Live: the action executing now, else the newest one (a replay's world shows the frames of the moment
  // itself); a step: its own (unless a tool's view is asked for).
  const motionStep = focus.mode === "live" ? (recorded ? null : moving ?? newest) : tool === null && !still ? step : null;
  const asked = useStepFrames(stepFrames || motionStep === null ? null : motionStep, framesVersion,
    motionStep !== null && motionStep === moving);
  const frames_: StepFrames | null = stepFrames ? (motionStep === null ? null : stepFrames(motionStep)) : asked;
  const motion = frames_?.motion?.step === motionStep && frames_?.motion?.frames.length ? frames_.motion : null;
  const motionCameras = motion ? [...new Set(motion.frames.flatMap((frame) => Object.keys(frame.images)))] : [];

  const liveFrames = live !== null && live.cameras.length > 0 && LIVE_SHOWS_FRAMES.has(live.phase);
  const shownStep = step ?? (focus.mode === "live" && !liveFrames && motion === null ? latest : null);
  const stepView = shownStep === null ? undefined : conversation.steps.get(shownStep);
  const robotCameras = useMemo(() => {
    const names = motion !== null ? motionCameras
      : focus.mode === "live" && liveFrames ? live!.cameras.map((item) => item.name)
        : (stepView?.observation?.images ?? []).map((image) => image.camera ?? "camera");
    return [...new Set(names)].filter((name) => name !== viewCamera);
  }, [motion, motionCameras.join(), focus.mode, liveFrames, live, stepView, viewCamera]);
  const cameras = [...(view && !tool ? [view.camera] : []), ...robotCameras];
  // While the robot moves, and for a step's motion, the motion is the main picture unless the operator view
  // is recorded moving too (or was chosen): the operator view, taken at each observation, would stand still.
  const motionFirst = (focus.mode === "step" || moving !== null) && motion !== null
    && !motionCameras.includes(viewCamera ?? "");
  const chosen = camera && cameras.includes(camera) ? camera
    : (motionFirst ? robotCameras[0] : undefined) ?? cameras[0] ?? null;
  const robot = chosen !== null && chosen !== viewCamera ? chosen : robotCameras[0] ?? null;
  const operatorStill = view && !tool ? operatorViewAt(summary, focus.mode === "live" ? null : step) : null;

  let main: "motion" | "operator" | "image" | "none" = "none";
  let image: StepImage | null = null;
  let overlay = null;
  let caption = "";
  let motionCam: string | null = null;

  const viewChosen = chosen !== null && chosen === viewCamera;
  // A step's Motion shows motion, through the operator view only when that is recorded moving.
  const stepMotion = focus.mode === "step" && tool === null && !still && motion !== null;
  if (recorded && focus.mode === "live" && liveFrames) {
    // A replay's world at its clock: the chosen camera, the operator view's or a robot camera's other.
    const named = (name: string | null) => (name === null ? null : live!.cameras.find((item) => item.name === name) ?? null);
    const asView = viewCamera !== null && operatorStill ? { name: viewCamera, url: operatorStill.url } : null;
    const frame = named(chosen) ?? (viewChosen ? asView : null) ?? named(robot) ?? live!.cameras[0];
    image = { camera: frame.name, src: frame.url };
    main = "image";
    const sim = live!.frame ? Number(live!.frame.sim_time_s).toFixed(2) : "–";
    caption = [t("camera.recordedCaption", { camera: frame.name === viewCamera ? t("camera.operatorView") : cameraLabel(frame.name),
      sim }), live!.phase === "executing" ? t("camera.recordedMoving") : live!.phase === "settling" ? t("camera.recordedSettling") : null]
      .filter(Boolean).join(" · ");
  } else if (viewChosen && motion !== null && motionCameras.includes(viewCamera!)) {
    main = "motion";
    motionCam = viewCamera;
  } else if (viewChosen && operatorStill && !stepMotion) {
    main = "operator";
    image = { camera: view!.camera, src: operatorStill.url };
    caption = t("camera.operatorCaption", { camera: view!.camera, step: operatorStill.step });
  } else if (motion !== null && (focus.mode === "live" || !tool)) {
    main = "motion";
    motionCam = motionCamera(motion, [chosen, robot]);
  }
  if (main === "none") {
    if (focus.mode === "live" && liveFrames) {
      const frame = live!.cameras.find((item) => item.name === robot) ?? live!.cameras[0];
      image = { camera: frame.name, src: frame.url };
      const info = live!.frame;
      caption = info ? t("camera.liveCaption", { message: live!.message, round: info.round_index + 1,
        sim: Number(info.sim_time_s).toFixed(2), frame: info.capture_index }) : live!.message;
      main = "image";
    } else if (tool && shown !== null && tool.images[shown]) {
      image = tool.images[shown];
      caption = t("camera.returnedImage", { tool: tool.name, n: shown + 1, count: tool.images.length });
      main = "image";
    } else if (stepView) {
      const toolCam = tool ? toolCamera(tool) : null;
      image = stepImage(stepView, tool && toolCam ? toolCam : robot);
      overlay = tool?.overlay && image && (tool.overlay.camera === null || tool.overlay.camera === image.camera) ? tool.overlay : null;
      caption = [t("camera.step", { step: stepView.step }), image?.camera ? cameraLabel(image.camera) : null,
        tool ? `${tool.name}${overlay ? ` · ${overlay.caption}` : ""}` : null,
        focus.mode === "live" && !recorded ? t("camera.latestStep", { message: live?.message || t("camera.noLiveFrames") }) : null]
        .filter(Boolean).join(" · ");
      main = "image";
    } else if (focus.mode === "live") {
      caption = live?.message || (liveWorld ? t("camera.connecting") : t("camera.noFrames"));
    }
  }
  const frames = motionImages(motion, motionCam);
  const following = motionStep !== null && motionStep === moving;
  const motionCaption = (index: number, count: number) => t("camera.motionCaption", {
    step: motionStep, camera: motionCam ? cameraLabel(motionCam) : "", n: index + 1, count,
    sim: frames[index] ? frames[index].simTimeS.toFixed(2) : "",
  });
  const motionIndex = Math.min(frameShown && frameShown.count === frames.length ? frameShown.index : frames.length - 1,
    frames.length - 1);
  if (main === "motion") caption = motionCaption(Math.max(0, motionIndex), frames.length);

  const stepHasMotion = step !== null && Boolean(summary.motion?.steps[String(step)]);
  const shownCamera = main === "motion" ? motionCam : image?.camera ?? chosen;
  const nameOf = (name: string) => (name === viewCamera ? t("camera.operatorView") : cameraName(name));
  // Every other camera at the same moment: the world's frame, the motion's frame, the operator view, or the step's image.
  const pictureOf = (name: string): string | null => {
    const liveFrame = focus.mode === "live" && liveFrames ? live!.cameras.find((item) => item.name === name) : null;
    if (liveFrame) return liveFrame.url;
    if (name === viewCamera) return operatorStill?.url ?? null;
    const moved = motionImages(motion, name);
    if (moved.length) return moved[Math.max(0, Math.min(motionIndex, moved.length - 1))].src;
    const seen = stepView ?? conversation.steps.get(step ?? latest ?? -1);
    return seen?.observation?.images.find((item) => item.camera === name)?.src ?? null;
  };
  const others = cameras.filter((name) => name !== shownCamera)
    .flatMap((name) => { const src = pictureOf(name); return src ? [{ name, src }] : []; });
  // The images a call gave the model, over the picture's corner: the selected call's, or live the newest of the
  // running instruction's (until a newer one replaces it).
  const inset = tool ?? (focus.mode === "live" ? imageTool(conversation) : null);
  return (
    <section className="camera-panel" aria-label={t("camera.cameras")}
      style={{ "--camera-aspect": aspect.toFixed(4) } as CSSProperties}>
      {instruction ? (
        <div className="instruction-head">
          <div className="instruction-label">{t("camera.instruction")}</div>
          <div className="instruction-text" title={instruction}>{instruction}</div>
        </div>
      ) : null}
      <div className={`panel-stage${others.length ? " with-row" : ""}`}
        style={{ "--thumbs": Math.max(2, others.length) } as CSSProperties}>
        <div className="camera-main">
          <div className="camera-box" data-testid="camera-box" data-shows={main}>
            {main === "motion" && frames.length ? (
              <MotionPlayer frames={frames} fps={motion?.fps ?? summary.motion?.fps ?? 10} follow={following}
                autoPlay playKey={`${focus.mode}:${motionStep}:${motionCam}`} caption={motionCaption} onFrame={onFrame} />
            ) : image ? (
              <ImageStage className="main" src={image.src} alt={`${image.camera ?? "camera"}`} overlay={overlay}
                onSize={tool && shown !== null ? undefined : onPicture && ((size) => size.height > 0 && onPicture(size.width / size.height))} />
            ) : <div className="stage-empty"><Icon name="camera" size={28} /><span>{caption || t("camera.noImage")}</span></div>}
            {shownCamera && !(tool && shown !== null) && (main !== "image" || image?.camera === shownCamera) ? (
              <span className="camera-name">{nameOf(shownCamera)}</span>
            ) : null}
            {inset && inset.images.length ? (
              <div className="tool-strip" aria-label={inset.name}>
                <div className="tool-strip-head">
                  <span>{t("camera.toModel", { tool: inset.name })}</span>
                  {tool ? (
                    <button type="button" className={`chip${shown === null ? " on" : ""}`} onClick={() => setShown(null)}>{t("camera.stepView")}</button>
                  ) : null}
                </div>
                <div className="tool-strip-images">
                  {inset.images.map((item, index) => (
                    <button type="button" key={index} className={`strip-thumb${tool && shown === index ? " on" : ""}`}
                      title={t("camera.toolImage", { tool: inset.name, n: index + 1 })}
                      onClick={() => (tool ? setShown(index)
                        : inset.step !== null && onFocus({ mode: "step", step: inset.step, tool: inset.id }))}>
                      <img src={item.src} alt={t("camera.toolImage", { tool: inset.name, n: index + 1 })} />
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </div>
        {others.length ? (
          <div className="camera-row" role="group" aria-label={t("camera.cameras")}>
            {others.map((item) => <CameraThumb key={item.name} src={item.src} label={nameOf(item.name)} onClick={() => onCamera(item.name)} />)}
          </div>
        ) : null}
      </div>
      <div className="panel-caption" data-testid="panel-caption" title={caption}>{caption}</div>
      <div className="panel-bar">
        <div className="segmented" role="tablist">
          <button type="button" role="tab" aria-selected={focus.mode === "live"} className={focus.mode === "live" ? "on" : ""}
            onClick={() => onFocus({ mode: "live" })}>
            <span className={`live-dot${live?.connected || motion !== null ? " on" : ""}`} aria-hidden="true" />
            {" "}{recorded ? t("camera.now") : t("camera.live")}
          </button>
          <button type="button" role="tab" aria-selected={focus.mode === "step"} className={focus.mode === "step" ? "on" : ""}
            disabled={latest === null}
            onClick={() => latest !== null && onFocus({ mode: "step", step: step ?? latest, tool: null })}>
            {step === null ? t("camera.steps") : t("camera.step", { step })}
          </button>
        </div>
        {stepHasMotion && !tool ? (
          <div className="segmented small" role="group" aria-label={t("camera.motion")}>
            <button type="button" className={!still ? "on" : ""} aria-pressed={!still} onClick={() => setStill(false)}>
              <Icon name="film" size={13} /> {t("camera.motion")}
            </button>
            <button type="button" className={still ? "on" : ""} aria-pressed={still} onClick={() => setStill(true)}>
              {t("camera.saw")}
            </button>
          </div>
        ) : null}
        <div className="cameras" role="group" aria-label={t("camera.cameras")}>
          {cameras.map((name) => (
            <button type="button" key={name} className={`chip${name === shownCamera ? " on" : ""}`}
              title={name} onClick={() => onCamera(name)}>
              {name === viewCamera ? <><Icon name="eye" size={13} /> {t("camera.operatorView")}</> : cameraLabel(name)}
            </button>
          ))}
        </div>
      </div>
      <StepTimeline conversation={conversation} current={step} replaying={replaying} onReplay={onReplay}
        motion={summary.motion?.steps ?? {}} onStep={(value) => onFocus({ mode: "step", step: value, tool: null })} />
      {foot}
    </section>
  );
}
