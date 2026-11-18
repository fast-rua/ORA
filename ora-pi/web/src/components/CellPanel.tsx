// A real robot cell's controls, as free play's terminal has them (lib/cell.ts): only on a run that
// drives a real cell (rig_mode read or on).
//
//   CellBanners    under the header: the held arms' alerts (critical ones as errors) and, once the
//                  play worker that held the arms has ended, that nothing controls them now
//   CellQuestions  above the composer: each question the robot's worker waits on (enabling the arms,
//                  a motion, releasing them in place) or a guided calibration asks (a task done by
//                  hand, a value, saving the result), its words whole, the seconds left, Approve and
//                  Decline (with a reason; a value question's Approve only once its value is typed;
//                  a task's and a save's buttons say what they mean), each bound to that question's id
//   CellBar        at the page's foot, below the composer: the latest outcomes, the terminal's and the
//                  page's, and the latest motion that ran without asking (safe mode or auto-approve,
//                  with the numbers safe mode checked); Home (at once), Hold, Float, Release, E-stop,
//                  Diagnose; the auto-approve switch (the permission mode is in the footer:
//                  TerminalFooter). The buttons keep their place: what changes (an outcome, a question's card,
//                  the composer's status line) grows above them
//
// The page's App asks first where the terminal asks y/N, then runs the command (App.tsx).

import { useEffect, useState } from "react";

import { CELL_VERBS, modeStateOf, modeStateText, secondsLeft, switchState, VALUE_MAX_CHARS, valuePlaceholder, type CellOutcome,
  type CellQuestion, type CellState, type PageCellCommand, type SelfAnswer } from "../lib/cell.ts";
import { t, useLang, type Key } from "../lib/i18n.ts";
import { Icon } from "./Icons.tsx";

type Json = any;

/** The time now in seconds, every second while `ticking`. */
function useSeconds(ticking: boolean): number {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!ticking) return;
    const timer = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(timer);
  }, [ticking]);
  return now;
}

export function CellBanners({ state }: { state: CellState }) {
  useLang();
  const alerts = state.alerts ?? [];
  const ended = state.worker_ended ?? null;
  if (!alerts.length && !ended) return null;
  return (
    <div className="cell-banners" data-testid="cell-banners">
      {ended ? (
        <div className="cell-banner tone-error" role="alert">{t("cell.workerGone", { pid: ended.pid })}</div>
      ) : null}
      {alerts.map((alert, index) => (
        <div key={`${alert.code}|${alert.arm ?? ""}|${alert.motor ?? ""}|${index}`}
          className={`cell-banner tone-${alert.level === "critical" ? "error" : "warning"}`}
          role={alert.level === "critical" ? "alert" : "status"}>{alert.text}</div>
      ))}
    </div>
  );
}

function kindLabel(kind: string): string {
  const key = `cell.kind.${kind}`;
  return ["enable", "motion", "release", "task", "value", "write"].includes(kind) ? t(key as Key) : kind;
}

/** A button's words: a task's and a save's say what they mean; the others approve or decline. */
function answerLabel(kind: string, verb: "approve" | "decline"): string {
  return kind === "task" || kind === "write" ? t(`cell.${verb}.${kind}` as Key) : t(`cell.${verb}`);
}

export type AnswerHandler = (question: CellQuestion, verb: "approve" | "decline", reason: string | null,
  value: string | null) => void;

function Question({ question, now, busy, onAnswer }: {
  question: CellQuestion;
  now: number;
  busy: boolean;
  onAnswer: AnswerHandler;
}) {
  const [reason, setReason] = useState("");
  const [value, setValue] = useState("");
  const asksValue = question.kind === "value";
  const left = secondsLeft(question, now);
  return (
    <section className="cell-question" aria-label={question.title} data-testid="cell-question" data-id={question.id}>
      <div className="cell-question-head">
        <span className="cell-kind">{kindLabel(question.kind)}</span>
        <strong>{question.title}</strong>
        <span className="cell-left">{t("cell.questionLeft", { id: question.id, s: left })}</span>
      </div>
      {question.text ? <div className="cell-question-text">{question.text}</div> : null}
      {question.details && Object.keys(question.details).length ? (
        <details className="cell-details">
          <summary>{t("cell.questionDetails")}</summary>
          <pre>{JSON.stringify(question.details, null, 2)}</pre>
        </details>
      ) : null}
      <div className="cell-question-actions">
        {asksValue ? (
          <input value={value} maxLength={VALUE_MAX_CHARS} placeholder={valuePlaceholder(question)} aria-label={t("cell.value")}
            onChange={(event) => setValue(event.target.value)} />
        ) : null}
        <input value={reason} maxLength={500} placeholder={t("cell.reasonPlaceholder")} aria-label={t("cell.reasonPlaceholder")}
          onChange={(event) => setReason(event.target.value)} />
        <button type="button" className="ghost danger" disabled={busy}
          onClick={() => onAnswer(question, "decline", reason.trim() || null, null)}>{answerLabel(question.kind, "decline")}</button>
        <button type="button" className="primary" disabled={busy || (asksValue && !value.trim())}
          onClick={() => onAnswer(question, "approve", null, asksValue ? value.trim() : null)}>
          {answerLabel(question.kind, "approve")}
        </button>
      </div>
    </section>
  );
}

export function CellQuestions({ state, busy, onAnswer }: {
  state: CellState;
  busy: boolean;
  onAnswer: AnswerHandler;
}) {
  useLang();
  const questions = [...(state.pending ?? [])].sort((a, b) => a.created_unix_s - b.created_unix_s);
  const now = useSeconds(questions.length > 0);
  const open = questions.filter((question) => secondsLeft(question, now) > 0);
  if (!open.length) return null;
  return (
    <div className="cell-questions" role="region" aria-label={t("cell.questions")}>
      {open.map((question) => <Question key={question.id} question={question} now={now} busy={busy} onAnswer={onAnswer} />)}
    </div>
  );
}

/** The switch in plain words. */
export function autoApproveText(state: Json): string {
  if (state?.on) return t("cell.autoOn", { pid: state.pid ?? "?", by: state.by ?? "?" });
  return t("cell.autoOff") + (state?.stale ? ` (${state.stale})` : "");
}

/** An outcome's text as the conversation entry says it (a switch's state and a mode in plain words). */
export function outcomeText(item: CellOutcome): string {
  if (item.ok && item.name === "yolo") {
    const state = switchState(item.text);
    if (state) return autoApproveText(state);
  }
  if (item.ok && item.name === "mode") {
    const state = modeStateOf(item.text);
    if (state) return modeStateText(state);
  }
  return item.text;
}

function SelfAnswered({ item }: { item: SelfAnswer }) {
  return (
    <li className="ok self" data-testid="cell-self-answered">
      <span className="mark" aria-hidden="true">▸</span>
      <span className="what">{t("cell.selfAnswered", { title: item.title, by: item.by })}</span>
    </li>
  );
}

function Outcome({ item }: { item: CellOutcome }) {
  const head = t(item.ok ? "cell.done" : "cell.failed", { name: item.name, verb: item.verb, s: item.seconds.toFixed(1) });
  const via = item.via === "web page" ? t("cell.viaPage") : t("cell.viaTerminal");
  const text = outcomeText(item);
  const long = text.includes("\n") || text.length > 160;
  return (
    <li className={item.ok ? "ok" : "failed"} data-testid="cell-outcome">
      <span className="mark" aria-hidden="true">{item.ok ? "✓" : "✗"}</span>
      <span className="what">{head} · {via}</span>
      {text && !long ? <span className="text">{text}</span> : null}
      {text && long ? <details><summary>{t("cell.output")}</summary><pre>{text}</pre></details> : null}
    </li>
  );
}

export function CellBar({ state, outcomes, running, onVerb, onSwitch, onHalt }: {
  state: CellState;
  outcomes: CellOutcome[];
  /** Names of the commands running now (another may start meanwhile: /estop while /home runs). */
  running: ReadonlySet<string>;
  onVerb: (command: PageCellCommand) => void;
  onSwitch: (on: boolean) => void;
  /** /halt: the motion in progress ends where the arms are. */
  onHalt?: () => void;
}) {
  useLang();
  const cell = state.cell!;
  const auto = state.auto_approve ?? { on: false };
  const shown = outcomes.slice(-3);
  const self = (state.self_answered ?? []).at(-1) ?? null;
  return (
    <div className="cell-bar" role="region" aria-label={t("cell.region")} data-testid="cell-bar">
      {/* Above the buttons: a new outcome grows the bar upwards and never moves a button under the pointer. */}
      {shown.length || self ? (
        <ul className="cell-outcomes">
          {shown.map((item) => <Outcome key={item.id} item={item} />)}
          {self ? <SelfAnswered key={`self-${self.id}`} item={self} /> : null}
        </ul>
      ) : null}
      <div className="cell-row">
        <span className="cell-label" title={cell.cell_config ?? ""}>
          {t("cell.label", { kind: cell.kind, mode: cell.rig_mode })}
        </span>
        <div className="cell-verbs" role="group" aria-label={t("cell.verbs")}>
          {onHalt ? (
            <button type="button" className="ghost" data-verb="stop-motion" title={t("cell.haltTitle")} onClick={onHalt}>
              {t("cell.halt")}
              {running.has("halt") ? <span className="pulse" aria-label={t("cell.runningShort")} /> : null}
            </button>
          ) : null}
          {CELL_VERBS.map((command) => (
            <button type="button" key={command.name} data-verb={command.verb}
              className={command.verb === "home" ? "primary" : command.verb === "estop" ? "ghost stop" : "ghost"}
              title={t(`cell.verbTitle.${command.verb}` as Key)} onClick={() => onVerb(command)}>
              {command.verb === "estop" ? <Icon name="stop" size={14} /> : null}
              {t(`cell.verb.${command.verb}` as Key)}{command.confirm ? "…" : ""}
              {running.has(command.name) ? <span className="pulse" aria-label={t("cell.runningShort")} /> : null}
            </button>
          ))}
        </div>
      </div>
      {cell.rig_mode === "on" ? (
        <div className="cell-row cell-auto">
          <span className={auto.on ? "cell-auto-on" : ""} data-testid="cell-auto">{autoApproveText(auto)}</span>
          <button type="button" className={auto.on ? "ghost" : "ghost danger"} onClick={() => onSwitch(!auto.on)}>
            {auto.on ? t("cell.autoTurnOff") : t("cell.autoTurnOn")}
          </button>
        </div>
      ) : null}
    </div>
  );
}
