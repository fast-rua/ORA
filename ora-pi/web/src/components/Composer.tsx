// What the operator types, and the run's controls.
//
// Enter sends: Pi's prompt while the agent is idle, a steer while it works
// (Alt+Enter: a follow-up). In an episode a message is an operator message the
// Supervisor delivers before the next model call; in free play it is the next
// instruction. While a free-play instruction runs, the send modes beside the
// status line say what Enter does with a message (Alt+Enter always queues):
//   Steer        it joins the running instruction: the agent reads it before its
//                next model call (a motion in progress finishes first);
//   Queue        the Supervisor holds it and it opens the next instruction once
//                this one ends (a `queued_instruction`; shown above the box until
//                then);
//   Stop & send  on a real cell, the motion in progress ends at once where the
//                arms are (/halt), and one the agent still plans is refused before
//                it moves (the same stop: App.tsx), and the message is a steer the
//                agent reads with the fresh observation; elsewhere nothing can cut
//                a motion short, so the message is queued and the instruction is
//                stopped (Esc): it opens the next one.
// With no instruction running they collapse to Send. Free play's Stop instruction
// (Esc), Next scene and End session are its controls; an episode's Stop run asks
// for confirmation (in this page).
//
// Typing "/" at the start of the box lists the commands this page takes
// (lib/slash.ts), narrowed as the name is typed: ↑/↓ move, Tab completes, Enter
// completes (or runs a command already typed in full), Esc closes the list only.
//
// Drawn as the terminal's prompt: above the box the status line (what the agent
// does now and for how long, and the send modes), the box between two rules
// after "❯", and the footer under it.

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { CellIntent } from "../lib/cell.ts";
import { outgoing, type Outgoing } from "../lib/composer.ts";
import { t, useLang } from "../lib/i18n.ts";
import type { PlayPhase } from "../lib/play.ts";
import { isComplete, slashSuggestions, type SlashItem } from "../lib/slash.ts";
import { Icon } from "./Icons.tsx";

export type SendMode = "steer" | "queue" | "stop";
const SEND_MODES: readonly SendMode[] = ["steer", "queue", "stop"];

type Message = Extract<Outgoing, { kind: "message" }>;

export function Composer({ play, phase, streaming, closed, draft, onSend, onControl, onStopRun, onRefused, doing = null,
  activity = null, cell = false, realCell = false, onCell, onStop, onNotice, queued = [], footer = null }: {
  play: boolean;
  phase: PlayPhase | null;
  streaming: boolean;
  closed: boolean;
  /** What the agent does now, with its seconds, or null. */
  doing?: string | null;
  activity?: string | null;
  draft: { text: string; at: number } | null;
  onSend: (message: Message) => Promise<boolean>;
  onControl: (control: "stop_instruction" | "reset" | "quit") => void;
  onStopRun: () => void;
  onRefused: (reason: string) => void;
  /** The run may drive a real cell (its state may still be loading): its commands, typed, act on it (lib/cell.ts). */
  cell?: boolean;
  /** The run drives a real cell (its state names one): its commands are listed, and Stop & send halts the motion. */
  realCell?: boolean;
  onCell?: (intent: CellIntent) => void;
  /**
   * Stop & send on a real cell: the motion in progress ends where the arms are, and one the agent still plans
   * is refused before it moves; `text` is the message sent with it.
   */
  onStop?: (text: string) => void;
  /** Free play: the messages held for the next instruction (lib/messages.ts queuedNext). */
  queued?: ReadonlyArray<{ key: string; text: string }>;
  onNotice?: (text: string) => void;
  /** The terminal's footer under the prompt box (TerminalFooter). */
  footer?: ReactNode;
}) {
  useLang();
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<SendMode>("steer");
  const [selected, setSelected] = useState(0);
  /** The text the list was closed at (Esc); it opens again once the text changes. */
  const [dismissed, setDismissed] = useState<string | null>(null);
  const composing = useRef(false);
  const area = useRef<HTMLTextAreaElement>(null);
  const list = useRef<HTMLUListElement>(null);

  const running = play && phase === "running";

  useEffect(() => {
    if (draft) setText(draft.text);
  }, [draft]);

  // Each new instruction starts with Steer: a mode chosen for the last one never surprises the next.
  const wasRunning = useRef(running);
  useEffect(() => {
    if (running && !wasRunning.current) setMode("steer");
    wasRunning.current = running;
  }, [running]);

  const suggestions = useMemo(() => (dismissed === text ? [] : slashSuggestions(text, { play, cell: realCell })),
    [text, dismissed, play, realCell]);
  const open = suggestions.length > 0 && !closed;
  const current = open ? suggestions[Math.min(selected, suggestions.length - 1)] : null;

  useEffect(() => setSelected(0), [text]);
  useEffect(() => {
    const row = list.current?.querySelector<HTMLElement>('[aria-selected="true"]');
    row?.scrollIntoView?.({ block: "nearest" });
  }, [selected, open]);

  useEffect(() => {
    if (!play) return;
    const onKey = (event: KeyboardEvent) => {
      // The list's own Esc (it closes the list) never reaches here.
      if (event.key === "Escape" && !event.defaultPrevented && phase === "running" && !document.querySelector(".dialog")) {
        event.preventDefault();
        onControl("stop_instruction");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [play, phase, onControl]);

  const complete = (item: SlashItem) => {
    setText(item.insert);
    setDismissed(null);
    area.current?.focus();
  };

  const submit = async (queue: boolean) => {
    const next = outgoing(text, { play, streaming, followUp: !play && queue, cell });
    if (next === null || busy) return;
    if (next.kind === "refused") return onRefused(next.reason);
    if (next.kind === "cell") {
      setText("");
      return onCell?.(next.intent);
    }
    if (next.kind === "control") {
      setText("");
      return onControl(next.control);
    }
    if (play) {
      const how: SendMode = !running || next.text.startsWith("/") ? "steer" : queue ? "queue" : mode;
      setText("");
      area.current?.focus();
      if (how === "queue" || (how === "stop" && !realCell)) {
        // Held by the Supervisor until the running instruction ends, then the next one. Stop & send off a
        // real cell: nothing cuts a motion short, so the instruction stops (Esc) and the message opens the next.
        // Both at once: the run admits a message only between the robot's moves, and the stop must not wait.
        const sending = onSend({ ...next, as: "queue" });
        if (how === "stop") onControl("stop_instruction");
        else onNotice?.(t("notice.queued"));
        if (!(await sending)) setText((value) => value || next.text);
        return;
      }
      if (how === "stop") {
        onStop?.(next.text);
        onNotice?.(t("notice.halting"));
      }
      // Free play: the box empties at once and the message shows queued in the conversation (the run
      // admits it only between the robot's moves); another may follow. A refused one comes back.
      if (!(await onSend(next))) setText((value) => value || next.text);
      return;
    }
    setBusy(true);
    try {
      if (await onSend(next)) setText("");
    } finally {
      setBusy(false);
      area.current?.focus();
    }
  };

  const placeholder = closed ? t("composer.closed")
    : play ? (phase === "running" ? (mode === "queue" ? t("composer.placeholder.queue")
      : mode === "stop" ? t(realCell ? "composer.placeholder.stopCell" : "composer.placeholder.stop") : t("composer.running"))
      : phase === "ended" ? t("composer.ending") : t("composer.idle"))
      : t("composer.episode");

  const hint = play ? t("composer.hintPlay") : t("composer.hintEpisode") + (streaming ? t("composer.hintFollowUp") : "");
  const modeTitle = (value: SendMode) => t(value === "stop" && realCell ? "composer.modeTitle.stopCell" : `composer.modeTitle.${value}`);
  const sendLabel = running ? t(`composer.mode.${mode}`) : play ? t("composer.send") : streaming ? t("composer.steer") : t("composer.send");
  return (
    <div className={`composer${running ? " running" : ""}`}>
      <div className="status-line">
        {doing ? (
          <div className={`activity activity-${activity}`} data-testid="activity" role="status">
            <span className="spinner" aria-hidden="true" />
            <span className="doing" data-testid="doing">{doing}</span>
            {running ? <span className="activity-hint">({t("composer.activityHint")})</span> : null}
          </div>
        ) : null}
        {running && !closed ? (
          <div className="send-modes" data-testid="send-modes">
            <span className="send-modes-label" aria-hidden="true">{t("composer.enter")}</span>
            <div className="segmented small" role="radiogroup" aria-label={t("composer.modes")}>
              {SEND_MODES.map((value) => (
                <button type="button" key={value} role="radio" aria-checked={mode === value} data-mode={value}
                  className={`${mode === value ? "on" : ""}${value === "stop" ? " mode-stop" : ""}`} title={modeTitle(value)}
                  onClick={() => { setMode(value); area.current?.focus(); }}>
                  {t(`composer.mode.${value}`)}
                </button>
              ))}
            </div>
          </div>
        ) : null}
      </div>
      {queued.length ? (
        <ul className="composer-queue" aria-label={t("composer.queued")}>
          {queued.map((item) => (
            <li key={item.key} data-testid="queued">
              <span className="queued-mark" aria-hidden="true">⧗</span>
              <span className="queued-label">{t("composer.queued")}</span>
              <span className="queued-text">{item.text}</span>
            </li>
          ))}
        </ul>
      ) : null}
      <div className="prompt-wrap">
        {open ? (
          <div className="slash" data-testid="slash">
            <ul ref={list} id="slash-list" role="listbox" aria-label={t("slash.list")}>
              {suggestions.map((item, index) => (
                <li key={item.insert} id={`slash-${index}`} role="option" aria-selected={item === current}
                  className={item === current ? "on" : ""}
                  onMouseEnter={() => setSelected(index)}
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => complete(item)}>
                  <span className="slash-name">{item.label}{item.args ? <span className="slash-args"> {item.args}</span> : null}</span>
                  <span className="slash-description">{item.description}</span>
                </li>
              ))}
            </ul>
            <div className="slash-keys">{t("slash.keys")}</div>
          </div>
        ) : null}
        <div className="prompt" title={hint}>
          <span className="prompt-mark" aria-hidden="true">❯</span>
          <textarea ref={area} value={text} placeholder={placeholder} disabled={(closed && !cell) || (play && phase === "ended" && !cell)}
            rows={1} aria-label={play ? t("composer.instruction") : t("composer.message")}
            aria-autocomplete="list" aria-controls={open ? "slash-list" : undefined}
            aria-activedescendant={open && current ? `slash-${suggestions.indexOf(current)}` : undefined}
            onChange={(event) => setText(event.target.value)}
            onCompositionStart={() => { composing.current = true; }}
            onCompositionEnd={() => { composing.current = false; }}
            onKeyDown={(event) => {
              if (composing.current || event.nativeEvent.isComposing) return;
              if (open && current) {
                if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                  event.preventDefault();
                  const step = event.key === "ArrowDown" ? 1 : -1;
                  setSelected((index) => (index + step + suggestions.length) % suggestions.length);
                  return;
                }
                if (event.key === "Tab" && !event.shiftKey) {
                  event.preventDefault();
                  complete(current);
                  return;
                }
                if (event.key === "Escape") {
                  // Closes the list only: the running instruction goes on.
                  event.preventDefault();
                  event.stopPropagation();
                  setDismissed(text);
                  return;
                }
                if (event.key === "Enter" && !event.shiftKey && !isComplete(text, current)) {
                  event.preventDefault();
                  complete(current);
                  return;
                }
              }
              if (event.key !== "Enter" || event.shiftKey) return;
              event.preventDefault();
              void submit(event.altKey);
            }} />
          <div className="actions">
            {play ? (
              <>
                <button type="button" className={`ghost${phase === "running" ? " stop" : ""}`} disabled={closed || phase !== "running"}
                  onClick={() => onControl("stop_instruction")} title={t("composer.stopInstructionTitle")}>
                  <Icon name="stop" size={13} /> {t("composer.stopInstruction")}
                </button>
                <button type="button" className="ghost" disabled={closed || phase === "ended"} onClick={() => onControl("reset")}
                  title={t("composer.nextSceneTitle")}>
                  <Icon name="skip" size={13} /> {t("composer.nextScene")}
                </button>
                <button type="button" className="ghost danger" disabled={closed || phase === "ended"} onClick={() => onControl("quit")}
                  title={t("composer.endSessionTitle")}>
                  <Icon name="exit" size={13} /> {t("composer.endSession")}
                </button>
              </>
            ) : (
              <button type="button" className="ghost danger" disabled={closed} onClick={onStopRun}
                title={t("composer.stopRunTitle")}>
                <Icon name="stop" size={13} /> {t("composer.stopRun")}
              </button>
            )}
            <button type="button" className="primary" disabled={closed || busy || !text.trim()} onClick={() => void submit(false)}
              title={running ? modeTitle(mode) : undefined}>
              <Icon name="send" size={13} /> {sendLabel}
            </button>
          </div>
        </div>
      </div>
      {footer}
    </div>
  );
}
