// The conversation: observations, instructions, the agent's replies and its tool calls, each
// with the image of its step. A tool call selects its step for the camera panel; the selected
// step's calls are marked. In free play a message sent while an instruction runs is shown joined
// to that instruction: queued until the agent reads it, then with the step it read it at; the
// agent's next words are marked as its answer, and where each instruction ended is marked.
//
// Drawn as ORA's terminal transcript: the model's words and each call after a dot, a call as
// `name(key=value, …)` with what its result says under it after "⎿" (lib/terminal.ts), thoughts
// and observations as dim lines, the operator's instruction on its own row.
//
// Two ways to draw it (lib/display.ts), the same on the live page and in a replay:
//   compact   one line per thought ("Thought for 12 s") and per tool call (the call and its result's
//             lines), opened by a click to the full thinking, the arguments, the result and its
//             images; an observation is one line, its images folded away
//   detailed  each tool call as a card with its step's image, each observation with its images
// "Expand all thinking" opens every thought in either.

import { memo, useEffect, useRef, useState, type ReactNode } from "react";

import { DEFAULT_DISPLAY, type Display } from "../lib/display.ts";
import { seconds, t, useLang } from "../lib/i18n.ts";
import { renderMarkdown } from "../lib/markdown.ts";
import { cameraLabel, stepImage, toolCamera, type Conversation as ConversationData, type Item,
  type ObservationView, type StepView, type ToolView } from "../lib/messages.ts";
import { endingKind, endingText } from "../lib/play.ts";
import { callText, commandLines, isScriptResult, outputTail, resultLines, scriptLines } from "../lib/terminal.ts";
import { ImageStage } from "./ImageStage.tsx";
import { Icon, Working } from "./Icons.tsx";

export interface Focus {
  step: number | null;
  tool: string | null;
}

/** Within this distance of its end the list follows new items. */
const FOLLOW_MARGIN_PX = 80;
/** A scroll event this soon after the operator's wheel, touch, key or scrollbar is theirs. */
const OPERATOR_SCROLL_MS = 1000;
const SCROLL_KEYS = new Set(["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "]);
const NO_MOTION: Record<string, number> = {};

const Markdown = memo(function Markdown({ text }: { text: string }) {
  return <div className="markdown" dangerouslySetInnerHTML={{ __html: renderMarkdown(text) }} />;
});

function stepLabel(step: number | null): string {
  return step === null ? "" : t("chat.step", { step });
}

/** One thought: a line ("Thought for 12 s") that opens to the whole thinking. */
function Thinking({ text, durationMs, streaming, expanded }: {
  text: string;
  durationMs: number | null;
  streaming: boolean;
  expanded: boolean;
}) {
  const [open, setOpen] = useState(expanded);
  useEffect(() => setOpen(expanded), [expanded]);
  const label = streaming ? t("chat.thinkingNow")
    : durationMs !== null ? t("chat.thoughtFor", { s: seconds(Math.max(1000, durationMs)) }) : t("chat.thought");
  return (
    <div className={`thinking${open ? " open" : ""}`} data-thinking={open ? "open" : "closed"}>
      <button type="button" className="fold-line thinking-line" aria-expanded={open}
        title={open ? t("chat.hideThinking") : t("chat.showThinking")} onClick={() => setOpen(!open)}>
        <span>{label}</span>
      </button>
      {open ? <pre className="thinking-text">{text}</pre> : null}
    </div>
  );
}

function pretty(value: unknown, fallback = ""): string {
  const text = value === undefined || value === null ? fallback
    : typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return text.length > 20_000 ? `${text.slice(0, 20_000)}\n…` : text;
}

/** What a tool call's image is: what the tool returned, else the step's view through the tool's camera. */
function toolPicture(tool: ToolView, step: StepView | undefined) {
  const own = tool.images[0] ?? null;
  return { view: own ?? stepImage(step, toolCamera(tool)), overlay: own ? null : tool.overlay };
}

/** A code-mode command (ORA's bash with a `command`): its program is drawn under the call (ScriptBody). */
function isScript(tool: ToolView): boolean {
  return tool.name === "bash" && commandLines(tool.args) !== null;
}

/**
 * A code-mode command as it runs and after: its lines (all of them when open), how it ended, each robot call its
 * program made in order (a dry run marked as a plan), the error that ended it and the end of what it printed.
 */
function ScriptBody({ tool, open }: { tool: ToolView; open: boolean }) {
  const command = commandLines(tool.args, open) ?? [];
  const result = isScriptResult(tool.result) ? tool.result : null;
  const lines = result ? scriptLines(result, open) : [];
  const output = result ? outputTail(result, open) : { lines: [], above: 0 };
  return (
    <div className="script" data-script-status={result?.status ?? tool.status}>
      <pre className="script-command">{command.join("\n")}</pre>
      {lines.length ? (
        <div className="tool-summary script-summary">
          {lines.map((line, index) => (
            <div key={index} className={`tool-summary-line script-${line.kind}${line.failed ? " failed" : ""}`
              + `${line.dryRun ? " dry-run" : ""}`} data-call={line.kind === "call" ? index : undefined}>{line.text}</div>
          ))}
        </div>
      ) : null}
      {output.lines.length ? (
        <pre className="script-output">
          {[...(output.above ? [t("script.linesAbove", { n: output.above })] : []), ...output.lines].join("\n")}
        </pre>
      ) : result && open ? <div className="muted script-output-empty">{t("script.noOutput")}</div> : null}
      {result && tool.images.length ? (
        <div className="muted">{tool.images.length === 1 ? t("chat.oneImageToModel") : t("chat.imagesToModel", { n: tool.images.length })}</div>
      ) : null}
    </div>
  );
}

/** The call as the terminal writes it: `name(key=value, …)`, a compute's lines below; a command's name alone. */
function CallHead({ tool }: { tool: ToolView }) {
  if (isScript(tool)) {
    return <span className="tool-call"><span className="tool-name">{tool.name}</span></span>;
  }
  const { args, more } = callText(tool.name, tool.args);
  return (
    <span className="tool-call">
      <span className="tool-name">{tool.name}</span><span className="tool-args">{args}</span>
      {more.map((line, index) => <span key={index} className="tool-more">{line}</span>)}
    </span>
  );
}

/** What the result says, under the call after "⎿" (lib/terminal.ts resultLines). */
function ResultLines({ tool }: { tool: ToolView }) {
  const lines = resultLines(tool);
  if (!lines.length) return null;
  return (
    <div className="tool-summary">
      {lines.map((line, index) => <div key={index} className="tool-summary-line">{line}</div>)}
    </div>
  );
}

/** Compact: one line per call (its name and a short line of its arguments) that opens to the whole call. */
function ToolLine({ tool, step, focused, onSelect, onMotion, hasMotion }: {
  tool: ToolView;
  step: StepView | undefined;
  focused: boolean;
  onSelect: (tool: ToolView) => void;
  onMotion: ((step: number) => void) | null;
  hasMotion: boolean;
}) {
  const [open, setOpen] = useState(false);
  const { view, overlay } = toolPicture(tool, step);
  return (
    <div className={`tool-line status-${tool.status}${focused ? " focused" : ""}${open ? " open" : ""}`} data-tool={tool.name}
      data-step={tool.step ?? undefined}>
      <button type="button" className="fold-line tool-head" aria-expanded={open}
        title={tool.step === null ? t("chat.showCall") : t("chat.showStep", { step: tool.step })}
        onClick={() => {
          setOpen(!open);
          onSelect(tool);
        }}>
        <span className={`dot ${tool.status}`} aria-hidden="true" />
        <CallHead tool={tool} />
        {tool.step !== null ? <span className="chip step-chip">{stepLabel(tool.step)}</span> : null}
      </button>
      {isScript(tool) ? <ScriptBody tool={tool} open={open} /> : <ResultLines tool={tool} />}
      {open ? (
        <div className="tool-open">
          {view ? (
            <div className="tool-body">
              <ImageStage className="thumb" src={view.src} alt={`${tool.name} · ${view.camera ?? "image"}`} overlay={overlay}
                onClick={() => onSelect(tool)} />
              <div className="tool-meta">
                {tool.overlay ? <div className="tool-overlay-caption">{tool.overlay.caption}</div> : null}
                {tool.images.length > 1 ? <div className="muted">{t("chat.images", { n: tool.images.length })}</div> : null}
                {hasMotion && onMotion && tool.step !== null ? (
                  <button type="button" className="link motion-link" onClick={() => onMotion(tool.step!)}>
                    <Icon name="film" size={13} /> {t("chat.motion")}
                  </button>
                ) : null}
              </div>
            </div>
          ) : null}
          {isScript(tool) ? null : <div className="fold-title">{t("chat.arguments")}</div>}
          {isScript(tool) ? null : <pre className="tool-result">{pretty(tool.args, "{}")}</pre>}
          <div className="fold-title">{t("chat.resultTitle")}</div>
          <pre className="tool-result">{tool.text ? pretty(tool.result ?? tool.text) : t("chat.noResult")}</pre>
        </div>
      ) : null}
    </div>
  );
}

/** Detailed: a card with the step's image, its overlay caption and the result to open. */
function ToolCard({ tool, step, focused, onSelect, onMotion, hasMotion }: {
  tool: ToolView;
  step: StepView | undefined;
  focused: boolean;
  onSelect: (tool: ToolView) => void;
  onMotion: ((step: number) => void) | null;
  hasMotion: boolean;
}) {
  const [open, setOpen] = useState(false);
  const { view, overlay } = toolPicture(tool, step);
  return (
    <div className={`tool-card status-${tool.status}${focused ? " focused" : ""}`} data-tool={tool.name}
      data-step={tool.step ?? undefined}>
      <button type="button" className="tool-head" onClick={() => onSelect(tool)}
        title={tool.step === null ? t("chat.showCall") : t("chat.showStep", { step: tool.step })}>
        <span className={`dot ${tool.status}`} aria-hidden="true" />
        <CallHead tool={tool} />
        {tool.step !== null ? <span className="chip step-chip">{stepLabel(tool.step)}</span> : null}
      </button>
      {isScript(tool) ? <ScriptBody tool={tool} open={open} /> : <ResultLines tool={tool} />}
      <div className="tool-body">
        {view ? (
          <ImageStage className="thumb" src={view.src} alt={`${tool.name} · ${view.camera ?? "image"}`} overlay={overlay}
            onClick={() => onSelect(tool)} />
        ) : null}
        <div className="tool-meta">
          {tool.overlay ? <div className="tool-overlay-caption">{tool.overlay.caption}</div> : null}
          {tool.images.length > 1 ? <div className="muted">{t("chat.images", { n: tool.images.length })}</div> : null}
          {tool.status === "running" ? <Working label={t("chat.running")} /> : null}
          {tool.status === "pending" ? <span className="muted">{t("chat.waitingResult")}</span> : null}
          {hasMotion && onMotion && tool.step !== null ? (
            <button type="button" className="link motion-link" onClick={() => onMotion(tool.step!)}>
              <Icon name="film" size={13} /> {t("chat.motion")}
            </button>
          ) : null}
          {tool.text ? (
            <button type="button" className="link" onClick={() => setOpen(!open)}>
              <Icon name="chevron" size={12} /> {open ? t("chat.hideResult") : t("chat.result")}
            </button>
          ) : null}
        </div>
      </div>
      {open ? <pre className="tool-result">{pretty(tool.result ?? tool.text)}</pre> : null}
    </div>
  );
}

function ObservationImages({ observation, onSelect }: { observation: ObservationView; onSelect: (step: number | null) => void }) {
  return (
    <div className="observation-images">
      {observation.images.map((image, index) => (
        <figure key={`${image.camera}-${index}`} onClick={() => onSelect(observation.step)}>
          <ImageStage className="thumb" src={image.src} alt={image.camera ?? "camera"} overlay={null} />
          <figcaption title={image.camera ?? undefined}>{image.camera ? cameraLabel(image.camera) : t("chat.camera")}</figcaption>
        </figure>
      ))}
    </div>
  );
}

function ObservationCard({ observation, focused, onSelect, compact }: {
  observation: ObservationView;
  focused: boolean;
  onSelect: (step: number | null) => void;
  compact: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [images, setImages] = useState(false);
  if (compact) {
    return (
      <div className={`observation-line${focused ? " focused" : ""}${images ? " open" : ""}`} data-step={observation.step ?? undefined}>
        <button type="button" className="fold-line observation-head" aria-expanded={images} onClick={() => {
          setImages(!images);
          onSelect(observation.step);
        }}>
          <Icon name="camera" size={13} />
          <span>{t("chat.observation")}</span>
          {observation.images.length ? (
            <span className="muted">{observation.images.length === 1 ? t("chat.observationImage")
              : t("chat.observationImages", { n: observation.images.length })}</span>
          ) : null}
          {observation.step !== null ? <span className="chip step-chip">{stepLabel(observation.step)}</span> : null}
        </button>
        {images ? (
          <div className="tool-open">
            {observation.images.length ? <ObservationImages observation={observation} onSelect={onSelect} /> : null}
            {observation.text ? <pre className="tool-result">{observation.text}</pre> : null}
          </div>
        ) : null}
      </div>
    );
  }
  return (
    <div className={`observation${focused ? " focused" : ""}`} data-step={observation.step ?? undefined}>
      <button type="button" className="observation-head" onClick={() => onSelect(observation.step)}>
        <Icon name="camera" size={14} />
        <span>{t("chat.observation")}</span>
        {observation.step !== null ? <span className="chip step-chip">{stepLabel(observation.step)}</span> : null}
      </button>
      {observation.images.length ? <ObservationImages observation={observation} onSelect={onSelect} /> : null}
      {observation.text ? (
        <button type="button" className="link" onClick={() => setOpen(!open)}>
          <Icon name="chevron" size={12} /> {open ? t("chat.hideWhatModelRead") : t("chat.whatModelRead")}
        </button>
      ) : null}
      {open ? <pre className="tool-result">{observation.text}</pre> : null}
    </div>
  );
}

function JoinedBubble({ item }: { item: Extract<Item, { kind: "joined" }> }) {
  const who = item.state === "next" ? t("chat.you")
    : item.instruction !== null ? t("chat.joins", { n: item.instruction }) : t("chat.joinsRunning");
  const state = item.state === "delivered" ? (item.step !== null ? t("chat.readAt", { step: item.step }) : null)
    : item.state === "queued" ? t("chat.queued") : t("chat.startsNext");
  return (
    <div className={`bubble user joined ${item.state}`} data-joined={item.state}>
      <div className="who">
        <span>{who}</span>
        {state ? <span className={`message-state ${item.state}`}>{item.state === "queued" ? <i className="pulse" /> : null}{state}</span> : null}
      </div>
      <div className="text">{item.text}</div>
    </div>
  );
}

function EndedMarker({ item }: { item: Extract<Item, { kind: "ended" }> }) {
  const kind = endingKind(item.ended);
  const extra = item.seconds !== null && item.decisions !== null
    ? t("chat.endedTime", { seconds: Math.max(1, Math.round(item.seconds)), decisions: item.decisions }) : null;
  return (
    <div className={`ended ended-${kind}`} role="status" data-ended={item.ended}>
      <span className="ended-mark" aria-hidden="true">{kind === "reported" ? "✓" : "■"}</span>
      <span>{t("chat.ended", { n: item.n, ending: endingText(item.ended), step: item.step ?? "?" })}</span>
      {extra ? <span className="muted">{extra}</span> : null}
    </div>
  );
}

interface ItemProps {
  item: Item;
  conversation: ConversationData;
  /** The selected call or step, as far as this item is concerned. */
  focusedStep: boolean;
  focusedTool: string | null;
  play: boolean;
  display: Display;
  onTool: (tool: ToolView) => void;
  onStep: (step: number | null) => void;
  onMotion: ((step: number) => void) | null;
  motion: Record<string, number>;
  /** The page's language: an item draws again when it changes. */
  lang: string;
}

const ItemView = memo(function ItemView({ item, conversation, focusedStep, focusedTool, play, display, onTool, onStep, onMotion,
  motion }: ItemProps) {
  const compact = display.view === "compact";
  switch (item.kind) {
    case "observation":
      return <ObservationCard observation={item.observation} focused={focusedStep} onSelect={onStep} compact={compact} />;
    case "user":
      return (
        <div className="bubble user instruction">
          <div className="who">{play ? (item.instruction !== null ? t("chat.instructionN", { n: item.instruction })
            : t("chat.instruction")) : t("chat.you")}</div>
          <div className="text">{item.text}</div>
        </div>
      );
    case "joined":
      return <JoinedBubble item={item} />;
    case "ended":
      return <EndedMarker item={item} />;
    case "sent":
      return (
        <div className={`bubble user sent${item.ok ? "" : " refused"}`}>
          <div className="who">{play ? t("chat.sent") : t("chat.operatorMessage")}</div>
          <div className="text">{item.text}</div>
          {item.error ? <div className="error-line">{item.error}</div> : null}
        </div>
      );
    case "note":
      return (
        <div className="note" data-note={item.title}>
          <span className="chip">{item.title === "ora.continuation" ? t("chat.continuation") : item.title}</span> {item.text}
        </div>
      );
    case "skipped":
      return <div className="note skipped" data-skipped="superseded">{t("chat.superseded")}</div>;
    case "assistant": {
      const Tool = compact ? ToolLine : ToolCard;
      return (
        <div className={`bubble assistant${item.streaming ? " streaming" : ""}${item.answers ? " answer" : ""}`}>
          {item.answers ? <div className="who answer-label">{t("chat.answer")}</div> : null}
          {item.thinking.length ? (
            <Thinking text={item.thinking.join("\n\n")} durationMs={item.durationMs} expanded={display.thinking}
              streaming={item.streaming && !item.text && !item.tools.length} />
          ) : null}
          {item.text ? (item.streaming ? <div className="text streaming-text">{item.text}</div> : <Markdown text={item.text} />) : null}
          {item.tools.map((tool) => (
            <Tool key={tool.id} tool={tool} step={tool.step === null ? undefined : conversation.steps.get(tool.step)}
              focused={focusedTool === tool.id} onSelect={onTool} onMotion={onMotion}
              hasMotion={tool.step !== null && Boolean(motion[String(tool.step)])
                && conversation.steps.get(tool.step)?.tools.at(-1)?.id === tool.id} />
          ))}
          {item.truncated ? (
            <div className="truncated-line" data-truncated="output_limit">
              {item.truncated.outputTokens !== null ? t("chat.truncatedTokens", { n: item.truncated.outputTokens })
                : t("chat.truncated")}
            </div>
          ) : null}
          {item.error ? <div className="error-line">{item.error}</div> : null}
          {item.streaming && !item.text && !item.tools.length && !item.thinking.length ? <Working label={t("chat.working")} /> : null}
        </div>
      );
    }
  }
});

/** Whether an item holds the focused call or step (so only those items draw again when the focus moves). */
function focusOf(item: Item, focus: Focus): { step: boolean; tool: string | null } {
  if (item.kind === "observation") return { step: focus.step !== null && item.observation.step === focus.step, tool: null };
  if (item.kind === "assistant" && focus.tool !== null && item.tools.some((tool) => tool.id === focus.tool)) {
    return { step: false, tool: focus.tool };
  }
  return { step: false, tool: null };
}

export const ConversationView = memo(function ConversationView({ conversation, focus, play, scrollTo, onTool, onStep, empty,
  onMotion = null, motion = NO_MOTION, display = DEFAULT_DISPLAY }: {
  conversation: ConversationData;
  focus: Focus;
  play: boolean;
  /** A conversation key to bring into view (a step chosen elsewhere), with a counter to repeat it. */
  scrollTo: { key: string; seq: number } | null;
  onTool: (tool: ToolView) => void;
  onStep: (step: number | null) => void;
  empty: ReactNode;
  /** Play the motion a step's action ran (the camera panel). */
  onMotion?: ((step: number) => void) | null;
  /** Recorded motion frames by step (ora_get_frames). */
  motion?: Record<string, number>;
  /** Compact or detailed, and whether every thought starts open (lib/display.ts). */
  display?: Display;
}) {
  const lang = useLang();
  const list = useRef<HTMLDivElement>(null);
  const inner = useRef<HTMLDivElement>(null);
  // Whether the list follows its end. Only the operator's own scrolling leaves the end: a scroll event
  // also comes from the page's own jump to the end, and content that grew by more than the margin
  // before the browser delivered it would otherwise read as the operator scrolling up.
  const stick = useRef(true);
  const operatorScrolled = useRef(0);
  const dragging = useRef(false);
  const byOperator = () => {
    operatorScrolled.current = Date.now();
  };

  useEffect(() => {
    const element = list.current;
    if (element && stick.current) element.scrollTop = element.scrollHeight;
  }, [conversation, display]);

  // Whatever grows the conversation later (a reply streaming in, an image loading, the box below
  // changing size) keeps the newest item in view while the operator follows the end.
  useEffect(() => {
    const element = list.current;
    const content = inner.current;
    if (!element || !content || typeof ResizeObserver === "undefined") return;
    const follow = () => {
      if (stick.current) element.scrollTop = element.scrollHeight;
    };
    const observer = new ResizeObserver(follow);
    observer.observe(content);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!scrollTo || !list.current) return;
    const target = [...list.current.querySelectorAll<HTMLElement>("[data-key]")].find((item) => item.dataset.key === scrollTo.key);
    if (target) {
      stick.current = false;
      target.scrollIntoView?.({ block: "center", behavior: "smooth" });
    }
  }, [scrollTo]);

  return (
    <div ref={list} className={`conversation view-${display.view}`} aria-live="polite" lang={lang === "zh" ? "zh-CN" : undefined}
      onScroll={(event) => {
        const element = event.currentTarget;
        if (element.scrollHeight - element.scrollTop - element.clientHeight < FOLLOW_MARGIN_PX) {
          stick.current = true;
        } else if (dragging.current || Date.now() - operatorScrolled.current < OPERATOR_SCROLL_MS) {
          stick.current = false;
        } else if (stick.current) {
          element.scrollTop = element.scrollHeight;  // the content grew past the end: keep following it
        }
      }} onWheel={byOperator} onTouchMove={byOperator} onKeyDown={(event) => {
        if (SCROLL_KEYS.has(event.key)) byOperator();
      }} onPointerDown={(event) => {
        // The scrollbar is the list itself; a press on an item inside it is not a scroll.
        if (event.target !== event.currentTarget) return;
        dragging.current = true;
        const release = () => {
          dragging.current = false;
          byOperator();
          window.removeEventListener("pointerup", release);
          window.removeEventListener("pointercancel", release);
        };
        window.addEventListener("pointerup", release);
        window.addEventListener("pointercancel", release);
      }}>
      {conversation.items.length === 0 ? <div className="empty">{empty}</div> : null}
      <div ref={inner} className="conversation-items">
        {conversation.items.map((item) => {
          const focused = focusOf(item, focus);
          return (
            <div key={item.key} data-key={item.key} className="item">
              <ItemView item={item} conversation={conversation} focusedStep={focused.step} focusedTool={focused.tool} play={play}
                display={display} onTool={onTool} onStep={onStep} onMotion={onMotion} motion={motion} lang={lang} />
            </div>
          );
        })}
      </div>
    </div>
  );
});
