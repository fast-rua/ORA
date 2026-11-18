// ORA's page for one run: the conversation and the robot's cameras on one screen, drawn as ORA's
// terminal beside the robot (two dark panes: the agent's transcript with its prompt and footer on
// the left, the instruction and every camera on the right). A tool
// call or an observation selects its step; the camera panel shows what the model saw then,
// with the call's grounding or marks drawn on it, or the motion its action ran, and the step
// timeline moves through the steps with the conversation in step. The status line above the
// prompt says what the agent does now, with the seconds it has taken.
//
//   App         a live run: the run's events as they come (server-sent events), its composer
//               and controls, its dialogs and notices; on a real robot cell, the cell's controls
//               as the terminal has them (components/CellPanel.tsx)
//   ReplayApp   a recorded run (`ora replay`): the same page at a clock over the run's recorded
//               time (lib/replay.ts), with a replay bar where the composer was; nothing is sent
//
// Both draw the conversation compact or detailed (lib/display.ts), switched above it.

import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from "react";
import { flushSync } from "react-dom";

import { CameraPanel, type PanelFocus } from "./components/CameraPanel.tsx";
import { autoApproveText, CellBanners, CellBar, CellQuestions } from "./components/CellPanel.tsx";
import { SetupView } from "./components/SetupView.tsx";
import { Composer } from "./components/Composer.tsx";
import { ConversationView } from "./components/Conversation.tsx";
import { Dialog, type Answer } from "./components/Dialogs.tsx";
import { Header, type HeaderReplay } from "./components/Header.tsx";
import { Notices } from "./components/Notices.tsx";
import { PerfPanel } from "./components/PerfPanel.tsx";
import { ReplayBar } from "./components/ReplayBar.tsx";
import { TerminalFooter, type FooterInfo } from "./components/TerminalFooter.tsx";
import { call, subscribe } from "./lib/api.ts";
import { answerQuestion, answerRefusal, kindHint, mergeOutcomes, modeStateText, oldestQuestion, PlanStop, runVerb, secondsLeft,
  stopMotion, switchAutoApprove, withdrawStop, switchMode, useCellState, type CellIntent, type CellOutcome, type CellQuestion, type PageCellCommand,
  type PermissionMode } from "./lib/cell.ts";
import type { Outgoing } from "./lib/composer.ts";
import { initialDisplay, rememberDisplay, type Display } from "./lib/display.ts";
import { operatorAspect, useFramesSummary, type FramesSummary, type StepFrames } from "./lib/frames.ts";
import { t, useLang } from "./lib/i18n.ts";
import type { LiveView } from "./lib/live.ts";
import { buildConversation, queuedNext, toolCamera, type Conversation, type LiveTools, type ToolView } from "./lib/messages.ts";
import { timedAnswers, type TimedAnswer } from "./lib/perf.ts";
import { activityText, playPhase, playStatus, type PlayPhase } from "./lib/play.ts";
import { buildTimeline, cameraAspect, countUpTo, knownCameraAspect, framesSummary, momentAt, positionOf, ReplayState, stepFrames, worldAt,
  type MotionSpeed,
  type ReplayOptions, type ReplayScript, type Segment, type Timeline } from "./lib/replay.ts";
import { contextTokens, instructionOf } from "./lib/terminal.ts";
import { initialState, reduce, type PageState } from "./state.ts";

type Json = any;

const REPLAY_EVERY_MS = 1400;
const NO_STEPS: Record<string, number> = {};

function episodeStatus(state: Json, closed: string | null) {
  if (closed) return { label: t("status.runEnded"), detail: t("status.runEndedDetail"), tone: "done" };
  if (state?.isCompacting) return { label: t("status.compacting"), detail: null, tone: "busy" };
  if (state?.isStreaming) return { label: t("status.modelWorking"), detail: null, tone: "busy" };
  return { label: t("status.robotActing"), detail: null, tone: "idle" };
}

/** The time now, four times a second while `ticking` (the live seconds of "thinking… 4 s"). */
function useNow(ticking: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!ticking) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(timer);
  }, [ticking]);
  return ticking ? Math.max(now, Date.now()) : now;
}

/** What the footer below the conversation is told: what the agent does now, where free play is, and the terminal footer's line. */
export interface FooterState {
  doing: string | null;
  activity: string | null;
  play: boolean;
  phase: PlayPhase | null;
  info: FooterInfo;
}

/** The terminal footer's line from what the page has: where the run is, the model's context, its reasoning effort. */
function footerInfo(page: PageState, conversation: Conversation): FooterInfo {
  const play = page.run?.mode === "play" ? page.run.play : null;
  const where = play ? [play.scenes ? t("footer.scene", { scene: play.scene, scenes: play.scenes }) : null,
    play.instructions ? t("footer.instruction", { n: play.instructions }) : null]
    : conversation.order.length ? [t("footer.step", { n: conversation.order[conversation.order.length - 1] })] : [];
  const window = Number(page.state?.model?.contextWindow);
  return { where: where.filter((item): item is string => Boolean(item)), context: contextTokens(page.messages),
    window: Number.isFinite(window) && window > 0 ? window : null, effort: page.state?.thinkingLevel ?? null };
}

/** A recorded run's part of the page. */
interface PageReplay extends HeaderReplay {
  /** The camera box's shape, the run's operator camera's (lib/replay.ts knownCameraAspect), or null when unknown. */
  aspect: number | null;
  world: LiveView | null;
  stepFrames: (step: number) => StepFrames | null;
}

/** In the header: compact or detailed, and every thought open or not. */
function DisplayBar({ display, onDisplay }: { display: Display; onDisplay: (display: Display) => void }) {
  useLang();
  return (
    <div className="display-bar" role="group" aria-label={t("display.label")}>
      <div className="segmented small" role="group">
        <button type="button" className={display.view === "compact" ? "on" : ""} aria-pressed={display.view === "compact"}
          title={t("display.compactTitle")} onClick={() => onDisplay({ ...display, view: "compact" })}>{t("display.compact")}</button>
        <button type="button" className={display.view === "detailed" ? "on" : ""} aria-pressed={display.view === "detailed"}
          title={t("display.detailedTitle")} onClick={() => onDisplay({ ...display, view: "detailed" })}>{t("display.detailed")}</button>
      </div>
      <label className="check">
        <input type="checkbox" checked={display.thinking} onChange={(event) => onDisplay({ ...display, thinking: event.target.checked })} />
        {t("display.expandThinking")}
      </label>
    </div>
  );
}

/** The page itself, live or recorded. */
function Page({ page, summary, liveWorld, clock, replay = null, footer, overlay = null, banners = null, controls = null,
  toolbar = null, answers = null, onError }: {
  page: PageState;
  summary: FramesSummary;
  liveWorld: boolean;
  /** A replay's clock: the moment the page shows (the live page counts on its own clock). */
  clock?: number;
  replay?: PageReplay | null;
  footer: (state: FooterState) => ReactNode;
  overlay?: ReactNode;
  /** Under the header (a real cell's alerts). */
  banners?: ReactNode;
  /** Above the composer (a real cell's questions). */
  controls?: ReactNode;
  /** Below the composer, at the foot of the page, where nothing above moves it (a real cell's buttons). */
  toolbar?: ReactNode;
  /** A real cell's answered motion questions (the speed panel's approval waits), or null: none recorded. */
  answers?: TimedAnswer[] | null;
  onError: (message: string) => void;
}) {
  const lang = useLang();
  const [focus, setFocus] = useState<PanelFocus>({ mode: "live" });
  const [camera, setCamera] = useState<string | null>(null);
  const [walking, setWalking] = useState(false);
  const [scrollTo, setScrollTo] = useState<{ key: string; seq: number } | null>(null);
  const [display, setDisplay] = useState<Display>(() => initialDisplay());
  const seq = useRef(0);
  const exporting = replay?.exporting === true;
  // The camera box's shape: a replay's from its evidence; a live run's from its first operator view, then kept;
  // else the first camera picture's the panel showed (a run whose evidence gives no size), else 4:3.
  const [liveAspect, setLiveAspect] = useState<number | null>(null);
  const [pictureAspect, setPictureAspect] = useState<number | null>(null);
  const seenAspect = operatorAspect(summary);
  useEffect(() => {
    if (!replay && liveAspect === null && seenAspect !== null) setLiveAspect(seenAspect);
  }, [replay, liveAspect, seenAspect]);
  const onPicture = useCallback((value: number) => setPictureAspect((current) => current ?? value), []);
  const aspect = replay?.aspect ?? liveAspect ?? pictureAspect ?? 4 / 3;

  const run = page.run;
  const play = run?.mode === "play";
  const phase = playPhase(run?.play);
  useEffect(() => {
    document.title = run?.runId && !replay?.publish ? `ORA · ${run.runId}` : "ORA";
  }, [run?.runId, replay?.publish]);

  const live: LiveTools = page.tools as LiveTools;
  const playContext = useMemo(() => (play ? { messages: run?.playMessages ?? [], endings: run?.playEndings ?? [],
    moves: page.moves, pending: page.pending, running: phase === "running" } : null),
  [play, run?.playMessages, run?.playEndings, page.moves, page.pending, phase]);
  const conversation = useMemo(() => buildConversation(page.messages, page.streaming, live, page.sent, playContext),
    // The language is a dependency: overlay captions are built in it.
    [page.messages, page.streaming, live, page.sent, playContext, lang]);

  const showStep = useCallback((step: number, tool: string | null, scroll: boolean) => {
    setFocus({ mode: "step", step, tool });
    const key = conversation.steps.get(step)?.key;
    if (scroll && key) setScrollTo({ key, seq: ++seq.current });
  }, [conversation]);

  const onTool = useCallback((tool: ToolView) => {
    setWalking(false);
    const step = tool.step ?? conversation.order[conversation.order.length - 1];
    if (step === undefined) return;
    const toolCam = toolCamera(tool);
    if (toolCam) setCamera(toolCam);
    showStep(step, tool.id, false);
  }, [conversation, showStep]);

  const onStep = useCallback((step: number | null) => {
    if (step === null) return;
    setWalking(false);
    showStep(step, null, false);
  }, [showStep]);

  const onMotion = useCallback((step: number) => {
    setWalking(false);
    showStep(step, null, false);
  }, [showStep]);

  // The timeline's walk: one step after another, the conversation following (a live page's).
  useEffect(() => {
    if (!walking) return;
    const order = conversation.order;
    const timer = setInterval(() => {
      const at = focus.mode === "step" ? order.indexOf(focus.step) : -1;
      const next = order[at + 1];
      if (next === undefined) {
        setWalking(false);
        return;
      }
      showStep(next, null, true);
    }, REPLAY_EVERY_MS);
    return () => clearInterval(timer);
  }, [walking, focus, conversation, showStep]);

  const onWalk = (on: boolean) => {
    if (on && (focus.mode !== "step" || focus.step === conversation.order[conversation.order.length - 1])) {
      const first = conversation.order[0];
      if (first !== undefined) showStep(first, null, true);
    }
    setWalking(on);
  };

  const changeDisplay = useCallback((next: Display) => {
    setDisplay(next);
    if (!exporting) rememberDisplay(next);
  }, [exporting]);

  const activity = page.closed ? null : page.activity;
  const busy = activity !== null && activity.kind !== "idle" && (!play || phase === "running");
  const tick = useNow(busy && clock === undefined);
  const now = clock ?? tick;
  const doing = busy ? activityText(activity, now) : null;
  const status = play ? (() => {
    const value = playStatus(run?.play, page.lastInstruction);
    const tone = page.closed ? "done" : value.phase === "running" ? "busy" : value.phase === "ended" ? "done" : "wait";
    return { label: page.closed && value.phase !== "ended" ? t("status.sessionEnded") : value.label,
      detail: value.detail, tone };
  })() : episodeStatus(page.state, page.closed);

  const setup = play && run?.play?.state === "setup" ? run.play.setup : null;
  const empty = setup ? <SetupView setup={setup} /> : !page.loaded ? t("page.connecting")
    : play ? (phase === "awaiting_instruction" || phase === null ? (replay ? t("page.replayEmpty") : t("page.emptyPlay"))
      : t("page.instructionStarting"))
      : t("page.emptyEpisode");
  const motion = summary.motion?.steps ?? NO_STEPS;
  const info = footerInfo(page, conversation);
  const instruction = useMemo(() => instructionOf(conversation), [conversation]);
  const conversationFocus = useMemo(() => (focus.mode === "step" ? { step: focus.step, tool: focus.tool }
    : { step: null, tool: null }), [focus]);

  return (
    <div className={`app${replay ? " replay" : ""}${exporting ? " exporting" : ""}`}>
      {page.closed && !replay ? <div className="banner">{t("page.closed", { reason: page.closed })}</div> : null}
      {banners}
      <main className={`split${aspect >= 1.5 ? " wide" : ""}`} data-aspect={aspect.toFixed(4)}>
        <section className="left" aria-label="Conversation">
          <Header run={run} state={page.state} status={status} closed={page.closed} statuses={page.statuses} onError={onError}
            replay={replay}>
            {exporting ? null : <DisplayBar display={display} onDisplay={changeDisplay} />}
          </Header>
          <ConversationView conversation={conversation} focus={conversationFocus} play={play} scrollTo={scrollTo}
            onTool={onTool} onStep={onStep} empty={empty} onMotion={onMotion} motion={motion} display={display} />
          {page.queue.steering.length + page.queue.followUp.length > 0 && !play ? (
            <div className="queue">{t("chat.queueLine", { steering: page.queue.steering.length, followUp: page.queue.followUp.length })}</div>
          ) : null}
          {controls}
          {footer({ doing, activity: busy ? activity!.kind : null, play, phase, info })}
          {toolbar}
        </section>
        <CameraPanel conversation={conversation} focus={focus} camera={camera} liveWorld={liveWorld}
          framesVersion={page.framesVersion} activity={activity} summary={summary}
          world={replay ? replay.world : undefined} stepFrames={replay ? replay.stepFrames : null} aspect={aspect}
          onPicture={onPicture}
          onCamera={(name) => {
            setCamera(name);
            if (focus.mode === "step" && focus.tool) setFocus({ mode: "step", step: focus.step, tool: null });
          }}
          onFocus={(next) => {
            setWalking(false);
            if (next.mode === "step") showStep(next.step, next.tool, true);
            else setFocus(next);
          }}
          replaying={walking} onReplay={replay ? null : onWalk} instruction={instruction}
          foot={play && !exporting ? <PerfPanel perf={page.perf} answers={answers} replay={replay !== null} /> : null} />
      </main>
      <div className="page-foot">
        {replay ? (
          <>
            <span className="replay-dot" aria-hidden="true" />
            <span>{t("replay.caption")} · </span>
            <span className="replay-label" data-testid="replay-label">{replay.label ?? t("replay.label", { speed: replay.speed })}</span>
          </>
        ) : (
          <>
            <span className={`connection${page.connected && !page.closed ? " on" : ""}`} aria-hidden="true" />
            <span>{page.closed ? t("header.runEnded", { reason: page.closed }) : page.connected ? t("header.connected")
              : t("header.reconnecting")}</span>
          </>
        )}
      </div>
      {overlay}
    </div>
  );
}

/**
 * The question as the confirmation shows it: its title, id and time left, its words, what approving and
 * declining mean for its kind, the reason or the value sent.
 */
export function questionText(question: CellQuestion, reason: string | null, now: number = Date.now() / 1000,
                             value: string | null = null): string {
  const lines = [t("cell.questionHead", { title: question.title, id: question.id, s: secondsLeft(question, now) }),
    ...String(question.text ?? "").split("\n").filter((line) => line.trim())];
  const hint = kindHint(question);
  if (hint) lines.push(hint);
  if (reason) lines.push(t("cell.questionReason", { reason }));
  if (value) lines.push(t("cell.questionValue", { value }));
  return lines.join("\n");
}

interface PageConfirm {
  title: string;
  message: string;
  yes: () => void;
  /** A real cell's command: only the Yes button confirms it (Enter says no). */
  danger?: boolean;
}

/** A live run, over its web server's events. */
export function App({ liveWorld = true }: { liveWorld?: boolean }) {
  const [page, dispatch] = useReducer(reduce, initialState);
  const [confirm, setConfirm] = useState<PageConfirm | null>(null);

  /** Free play's state as it arrives, each one (a stop sent while the agent planned is settled on it). */
  const onPlay = useRef<((record: Json) => void) | null>(null);
  useEffect(() => subscribe((event) => {
    if (event?.type === "play") onPlay.current?.(event.play);
    dispatch({ type: "event", event });
  }, (connected) => dispatch({ type: "connected", connected })), []);

  const play = page.run?.mode === "play";
  const summary = useFramesSummary(page.framesVersion, page.loaded);
  const notify = useCallback((text: string, tone: "info" | "warning" | "error" = "info") => dispatch({ type: "notice", text, tone }), []);
  const dismiss = useCallback((id: string) => dispatch({ type: "dismiss", id }), []);

  /**
   * The stop on a real cell while an instruction runs (set below, once the cell's state is known), else
   * nothing: the motion in progress ends where the arms are, and one the agent still plans is refused.
   */
  const stopRef = useRef<((text: string | null) => void) | null>(null);

  const send = async (message: Extract<Outgoing, { kind: "message" }>): Promise<boolean> => {
    // Shown at once (queued), before the run admits it: while the robot moves that can take a while.
    const next = message.as === "queue";
    if (play) dispatch({ type: "pending", text: message.text, next });
    try {
      await call("/api/send", { text: message.text, as: message.as });
      if (message.as === "follow_up") notify(t("notice.followUp"));
      return true;
    } catch (error) {
      dispatch({ type: "unpend", text: message.text, next });
      notify(t("notice.notSent", { reason: (error as Error).message }), "error");
      return false;
    }
  };

  const control = useCallback((name: "stop_instruction" | "reset" | "quit") => {
    // On a real cell the motion in progress ends where the arms are first: the Supervisor reads the stop only
    // once the motion it executes has ended; one the agent still plans is refused (as the terminal's Esc).
    if (name === "stop_instruction") stopRef.current?.(null);
    const go = () => call("/api/play", { control: name }).then(() => notify(name === "stop_instruction"
      ? t("notice.stopping") : name === "reset" ? t("notice.resetting") : t("notice.quitting")))
      .catch((error: Error) => notify(t("notice.refused", { control: t(`control.${name}`), reason: error.message }), "error"));
    if (name === "quit") {
      setConfirm({ title: t("dialog.endTitle"), message: t("dialog.endMessage"), yes: () => void go() });
    } else void go();
  }, [notify]);

  const stopRun = () => {
    // ORA's /stop asks for confirmation; the dialog comes to this page (it asked).
    call("/api/send", { text: "/stop", as: "prompt" }).catch((error: Error) => notify(t("notice.stop", { reason: error.message }), "error"));
  };

  const answer = (request: Json, response: Answer) => {
    call("/api/dialog", { id: request.id, ...response }).catch((error: Error) => notify(error.message, "warning"));
  };

  // A real robot cell (free play on rig_mode read or on): its state from the gateway, its commands.
  const [cellState, refreshCell] = useCellState(play);
  const cell = cellState?.cell ? cellState : null;
  const [cellLocal, setCellLocal] = useState<CellOutcome[]>([]);
  const [running, setRunning] = useState<ReadonlySet<string>>(() => new Set());
  const outcomes = useMemo(() => mergeOutcomes(page.run?.cellCommands, cellLocal), [page.run?.cellCommands, cellLocal]);

  const cellRun = useCallback((name: string, verb: string, action: () => Promise<CellOutcome>): Promise<void> => {
    notify(t("cell.running", { name, verb }));
    setRunning((current) => new Set([...current, name]));
    return action().then((result) => {
      setCellLocal((list) => [...list, { ...result, at: result.at ?? Date.now() }].slice(-50));
      if (!result.ok) notify(`${t("cell.failed", { name, verb, s: Number(result.seconds ?? 0).toFixed(1) })}: ${result.text}`, "error");
    }).catch((error: Error) => notify(`${t("cell.failed", { name, verb, s: "0.0" })}: ${error.message}`, "error"))
      .finally(() => {
        setRunning((current) => new Set([...current].filter((item) => item !== name)));
        refreshCell();
      });
  }, [notify, refreshCell]);

  const onVerb = useCallback((command: PageCellCommand) => {
    const go = () => cellRun(command.name, command.verb, () => runVerb(command, command.confirm));
    if (!command.confirm) return go();
    setConfirm({ title: t(`cell.confirm.${command.verb}.title` as never), message: t(`cell.confirm.${command.verb}.message` as never),
      yes: go, danger: true });
  }, [cellRun]);

  const onAnswer = useCallback((question: CellQuestion, verb: "approve" | "decline", reason: string | null,
                                value: string | null = null) => {
    // Refused before anything is sent, as the terminal refuses it (a value missing, or one no question takes).
    const refused = answerRefusal(question, verb, value);
    if (refused !== null) return notify(refused, "warning");
    setConfirm({ title: t(`cell.confirm.${verb}.title`), message: questionText(question, reason, Date.now() / 1000, value),
      danger: true, yes: () => cellRun(verb, verb, () => answerQuestion(question.id, verb, reason, value)) });
  }, [cellRun, notify]);

  const onSwitch = useCallback((on: boolean) => {
    const go = () => cellRun("yolo", "auto-approve", () => switchAutoApprove(on));
    if (!on) return go();
    setConfirm({ title: t("cell.confirm.yolo.title"), message: t("cell.confirm.yolo.message"), yes: go, danger: true });
  }, [cellRun]);

  /** The permission mode: `auto` only after its confirmation (the terminal's y/N), the others at once. */
  const onMode = useCallback((mode: PermissionMode) => {
    const go = () => cellRun("mode", "mode", () => switchMode(mode));
    if (mode !== "auto") return go();
    setConfirm({ title: t("cell.confirm.mode.title"), message: t("cell.confirm.mode.message"), yes: go, danger: true });
  }, [cellRun]);

  /** /halt: the motion in progress ends where the arms are, held (never asked: it holds, it never drops). */
  const onHalt = useCallback(() => void cellRun("halt", "stop-motion", stopMotion), [cellRun]);
  // Esc, Stop instruction and Stop & send on a real cell while an instruction runs: the motion in progress
  // ends where the arms are; while the agent plans, the same stop refuses the motion its reply proposes
  // (at its approval, before anything moves), and is withdrawn when no motion came (lib/cell.ts PlanStop).
  const planStop = useRef(new PlanStop());
  const planStopRun = useRef<Promise<void> | null>(null);
  const moving = page.activity?.kind === "moving";
  const instructionRunning = playPhase(page.run?.play) === "running";
  stopRef.current = cell === null || !instructionRunning ? null : (text) => {
    if (moving) return onHalt();
    planStop.current.sent(text);
    planStopRun.current = cellRun("halt", "stop-motion", stopMotion);
  };
  onPlay.current = (record) => {
    if (!planStop.current.settle(record)) return;
    // After the stop's own command: one still starting would write the stop again.
    const after = planStopRun.current ?? Promise.resolve();
    planStopRun.current = null;
    void after.then(() => withdrawStop()).catch(() => undefined);
  };

  /** A cell command typed in the composer, as the terminal takes it. */
  const onCell = useCallback((intent: CellIntent) => {
    const name = intent.kind === "verb" ? intent.command.name : intent.kind === "answer" ? intent.verb
      : intent.kind === "mode" || intent.kind === "usage" && intent.name === "mode" ? "mode" : "yolo";
    if (intent.kind === "usage") return notify(t(intent.name === "mode" ? "cell.modeUsage" : "cell.yoloUsage"));
    if (intent.kind === "halt") return onHalt();
    if (intent.kind === "verb") return onVerb(intent.command);
    if (intent.kind === "mode" && intent.mode !== "status") return onMode(intent.mode);
    if (cell === null) return notify(t("cell.loading", { name }), "warning");
    if (intent.kind === "mode") {
      return cell.mode ? notify(modeStateText(cell.mode)) : notify(t("cell.loading", { name }), "warning");
    }
    if (intent.kind === "switch") {
      if (intent.state === "status") return notify(autoApproveText(cell.auto_approve));
      return onSwitch(intent.state === "on");
    }
    const question = oldestQuestion(cell, Date.now() / 1000);
    if (question === null) return notify(t("cell.none", { name }));
    onAnswer(question, intent.verb, intent.reason, intent.value);
  }, [cell, notify, onVerb, onSwitch, onAnswer, onMode, onHalt]);

  const answers = useMemo(() => (cell && Array.isArray(cell.answers) ? timedAnswers(cell.answers) : null), [cell]);
  const banners = cell ? <CellBanners state={cell} /> : null;
  const controls = cell
    ? <CellQuestions state={cell} busy={running.has("approve") || running.has("decline")} onAnswer={onAnswer} /> : null;
  const toolbar = cell
    ? <CellBar state={cell} outcomes={outcomes} running={running} onVerb={onVerb} onSwitch={onSwitch} onHalt={onHalt} /> : null;

  const dialog = page.dialogs[0] ?? null;
  const overlay = (
    <>
      <Notices notices={page.notices} widgets={page.widgets} onDismiss={dismiss} />
      {dialog ? <Dialog key={dialog.id} request={dialog} onAnswer={(response) => answer(dialog, response)} />
        : confirm ? (
          <Dialog key="page-confirm" request={{ id: "page-confirm", method: "confirm", title: confirm.title, message: confirm.message,
            danger: confirm.danger === true }}
            onAnswer={(response) => {
              setConfirm(null);
              if ("confirmed" in response && response.confirmed) confirm.yes();
            }} />
        ) : null}
    </>
  );

  return (
    <Page page={page} summary={summary} liveWorld={liveWorld} onError={(message) => notify(message, "error")} overlay={overlay}
      banners={banners} controls={controls} toolbar={toolbar} answers={answers}
      footer={({ doing, activity, phase, info }) => (
        <Composer play={play} phase={phase} streaming={Boolean(page.state?.isStreaming)} closed={Boolean(page.closed)}
          doing={doing} activity={activity} cell={play && (cellState === null || cellState.cell !== null)} onCell={onCell}
          realCell={cell !== null} onStop={(text) => stopRef.current?.(text)} onNotice={(text) => notify(text)}
          queued={play ? queuedNext(page.run?.playMessages, page.pendingNext) : []}
          draft={page.draft} onSend={send} onControl={control} onStopRun={stopRun} onRefused={(reason) => notify(reason, "warning")}
          footer={<TerminalFooter info={info} mode={cell?.mode ? { state: cell.mode, rigMode: cell.cell!.rig_mode,
            busy: running.has("mode"), onSwitch: onMode } : null} />} />
      )} />
  );
}

/** Two animation frames: React's effects and the browser's layout have run. */
function frames(count = 2): Promise<void> {
  return new Promise((resolve) => {
    const next = (left: number) => (left <= 0 ? resolve() : requestAnimationFrame(() => next(left - 1)));
    next(count);
  });
}

/** A short digest of what the page shows (a video skips drawing a frame that equals the one before). */
function pageDigest(): string {
  const root = document.querySelector(".app");
  const list = document.querySelector(".conversation");
  const text = `${root?.innerHTML ?? ""}|${list?.scrollTop ?? 0}`;
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return `${text.length}:${(hash >>> 0).toString(16)}`;
}

declare global {
  interface Window {
    /** A recorded run's clock, driven from outside to make a video (ora/app/replay_export.py). */
    __oraReplay?: {
      /** The replay: its recorded start and end, its playback length and stretches (lib/replay.ts Timeline). */
      info: () => { start: number; end: number; notes: string[]; runId: string | null; mode: string; duration: number;
        speed: number; motion: MotionSpeed; segments: Segment[]; aspect: number };
      /** The page at a recorded moment; resolves once drawn, its images decoded, with a digest of the page. */
      seek: (time: number) => Promise<string>;
      /** The page at a playback moment (ms from the replay's start), as `seek`. */
      seekPlayback: (ms: number) => Promise<string>;
      /** The playback moment of a recorded time. */
      playbackOf: (time: number) => number;
    };
  }
}

/** The corner's note: recorded, and at what rate the stretch on screen plays. */
function replayLabel(timeline: Timeline, segment: Segment | null, activity: string | null): string {
  if (timeline.motion === "same") return t("replay.label", { speed: timeline.speed });
  if (segment?.kind === "motion") {
    return timeline.motion === 1 ? t("replay.labelMotionReal") : t("replay.labelMotion", { speed: timeline.motion });
  }
  if (activity === "moving") return t("replay.labelMoving", { speed: timeline.speed });
  if (activity !== null && activity !== "idle" && activity !== "observing") return t("replay.labelThinking", { speed: timeline.speed });
  return t("replay.labelWaiting", { speed: timeline.speed });
}

/** A recorded run at its clock. */
function ReplayPage({ script, options }: { script: ReplayScript; options: ReplayOptions }) {
  useLang();
  const replayState = useMemo(() => new ReplayState(script), [script]);
  const [speed, setSpeed] = useState(options.speed);
  const [motion, setMotion] = useState<MotionSpeed>(options.motion);
  const timeline = useMemo(() => buildTimeline(script, speed, motion), [script, speed, motion]);
  // The playback position (ms of playback time); the recorded time and the motion's frame follow from it.
  const [position, setPosition] = useState(() => positionOf(timeline, script.start + options.from * 1000));
  const [playing, setPlaying] = useState(options.autoplay && !options.exporting);
  const positionRef = useRef(position);
  positionRef.current = position;
  const timelineRef = useRef(timeline);
  const moment = momentAt(script, timeline, position);
  const clock = moment.wall;

  // Another speed or motion speed keeps the recorded moment on screen.
  useEffect(() => {
    if (timelineRef.current === timeline) return;
    const wall = momentAt(script, timelineRef.current, positionRef.current).wall;
    timelineRef.current = timeline;
    setPosition(positionOf(timeline, wall));
  }, [script, timeline]);

  // Playback time advances with real time; each stretch maps it to its own rate.
  useEffect(() => {
    if (!playing) return;
    let handle = 0;
    let last = performance.now();
    const advance = (time: number) => {
      const next = Math.min(timeline.duration, positionRef.current + (time - last));
      last = time;
      positionRef.current = next;
      setPosition(next);
      if (next >= timeline.duration) setPlaying(false);
      else handle = requestAnimationFrame(advance);
    };
    handle = requestAnimationFrame(advance);
    return () => cancelAnimationFrame(handle);
  }, [playing, timeline]);

  useEffect(() => {
    const settle = async () => {
      await frames(2);
      await Promise.all([...document.images].map((image) => (image.complete || typeof image.decode !== "function" ? null
        : image.decode().catch(() => null))));
      await frames(1);
      return pageDigest();
    };
    window.__oraReplay = {
      info: () => ({ start: script.start, end: script.end, notes: script.notes, runId: script.runId, mode: script.mode,
        duration: timeline.duration, speed: timeline.speed, motion: timeline.motion, segments: timeline.segments,
        aspect: cameraAspect(script) }),
      seek: async (time: number) => {
        flushSync(() => setPosition(positionOf(timeline, Math.max(script.start, Math.min(script.end, time)))));
        return settle();
      },
      seekPlayback: async (ms: number) => {
        flushSync(() => setPosition(Math.max(0, Math.min(timeline.duration, ms))));
        return settle();
      },
      playbackOf: (time: number) => positionOf(timeline, time),
    };
    return () => {
      delete window.__oraReplay;
    };
  }, [script, timeline]);

  const page = useMemo(() => replayState.at(clock), [replayState, clock]);
  const captured = moment.captured ?? (script.capture ? countUpTo(script.capture.frames, clock) : 0);
  const viewed = script.operatorView ? countUpTo(script.operatorView.frames, clock) : 0;
  const summary = useMemo(() => framesSummary(script, captured, viewed), [script, captured, viewed]);
  const moving = page.activity?.kind === "moving" || moment.segment?.kind === "motion";
  const settling = moment.settling;
  const world = useMemo(() => worldAt(script, captured, viewed, moving, settling), [script, captured, viewed, moving, settling]);
  const steps = useCallback((step: number) => stepFrames(script, summary, step, clock), [script, summary, clock]);
  const label = replayLabel(timeline, moment.segment, page.activity?.kind ?? null);
  const aspect = useMemo(() => knownCameraAspect(script), [script]);
  const replay: PageReplay = { speed, publish: options.publish, exporting: options.exporting, label, aspect, world,
    stepFrames: steps };
  const marks = useMemo(() => script.instructions.map((item) => ({ index: item.index, start: item.start })), [script]);

  return (
    <Page page={page} summary={summary} liveWorld={false} clock={clock} replay={replay} onError={() => undefined}
      footer={({ doing, activity, info }) => (
        <ReplayBar start={script.start} end={script.end} clock={clock} playing={playing} speed={speed} motion={motion}
          instructions={marks} doing={doing} activity={activity} exporting={options.exporting} info={info} onPlay={setPlaying}
          onSpeed={setSpeed} onMotion={setMotion}
          onSeek={(time) => setPosition(positionOf(timeline, Math.max(script.start, Math.min(script.end, time))))} />
      )} />
  );
}

/** A recorded run (`ora replay`): its evidence, read once, then the page at the replay's clock. */
export function ReplayApp({ options }: { options: ReplayOptions }) {
  useLang();
  const [script, setScript] = useState<ReplayScript | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    call("/api/replay").then((data) => setScript(data as ReplayScript)).catch((reason: Error) => setError(reason.message));
  }, []);
  if (script === null) {
    return <div className="replay-loading" role="status">{error ? t("replay.failed", { reason: error }) : t("replay.loading")}</div>;
  }
  return <ReplayPage script={script} options={options} />;
}

