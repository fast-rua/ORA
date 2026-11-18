// The page's header, at the top of the conversation as the terminal's welcome lines: ORA's mark,
// the run's mode, its model and reasoning effort, the run; where it is; the page's language; free
// play's session tools (model, reasoning, compaction, name, new session, fork)
// when the run takes them. An episode binds all of them in its RunPlan, so the
// page shows none (the agent process would refuse them: ora_get_run's `refused`).
// A replay has none either, and when published leaves the run's ID out (the page's foot says it is
// recorded and at what speed); a video has no language switch. What the agent does now is the
// status line above the composer.

import { useState, type ReactNode } from "react";

import { call } from "../lib/api.ts";
import { allows } from "../lib/composer.ts";
import { setLang, t, useLang, type Lang } from "../lib/i18n.ts";
import { Icon, Mark } from "./Icons.tsx";

type Json = any;

function Language() {
  const lang = useLang();
  const choose = (next: Lang) => () => setLang(next);
  return (
    <div className="segmented small lang" role="group" aria-label={t("header.language")}>
      <button type="button" className={lang === "en" ? "on" : ""} aria-pressed={lang === "en"} onClick={choose("en")}>EN</button>
      <button type="button" className={lang === "zh" ? "on" : ""} aria-pressed={lang === "zh"} onClick={choose("zh")} lang="zh-CN">中文</button>
    </div>
  );
}

export interface HeaderReplay {
  speed: number;
  /** The corner's note: recorded, and the rate of what is on screen ("robot motion · real time", "thinking · 16×"). */
  label?: string;
  /** No run ID on the page. */
  publish: boolean;
  /** A video is being made: no controls. */
  exporting: boolean;
}

export function Header({ run, state, status, closed, statuses, onError, replay = null, children = null }: {
  run: Json;
  state: Json;
  status: { label: string; detail: string | null; tone: string };
  closed: string | null;
  statuses: Record<string, string>;
  onError: (message: string) => void;
  /** A recorded run (ora replay), or null for a live one. */
  replay?: HeaderReplay | null;
  /** More of the page's controls, beside the language switch (how the conversation is drawn). */
  children?: ReactNode;
}) {
  useLang();
  const play = run?.mode === "play";
  const model = state?.model ? String(state.model.id ?? state.model.name ?? "") : null;
  const effort = state?.thinkingLevel ? t("header.effort", { level: state.thinkingLevel }) : null;
  return (
    <header className="header">
      <Mark />
      <div className="brand">
        <div className="brand-line">
          <span className="brand-name">ORA</span>
          {run ? <span className={`badge mode-${run.mode}`}>{play ? t("header.freePlay") : t("header.episode")}</span> : null}
          <span className={`phase tone-${status.tone}`} data-testid="phase">
            <span className="phase-label">{status.label}</span>
            {status.detail ? <span className="phase-detail">{status.detail}</span> : null}
          </span>
        </div>
        <div className="brand-line dim" title={state?.model ? `${state.model.provider ?? ""}/${model}` : undefined}>
          {[model, effort].filter(Boolean).join(" · ") || "\u00a0"}
        </div>
        <div className="brand-line dim">
          ~/ora{replay?.publish ? null : (
            <> · <span className="run-id" title={run?.runId ?? ""}>{run?.runId ?? (replay ? "" : t("header.connecting"))}</span></>
          )}
        </div>
      </div>
      <div className="header-right">
        {Object.entries(statuses).map(([key, value]) => <span key={key} className="chip status-chip" title={key}>{value}</span>)}
        {children}
        {replay?.exporting ? null : <Language />}
        {play && run && !closed && !replay ? <SessionTools run={run} state={state} onError={onError} /> : null}
      </div>
    </header>
  );
}

function SessionTools({ run, state, onError }: { run: Json; state: Json; onError: (message: string) => void }) {
  const [open, setOpen] = useState(false);
  const [models, setModels] = useState<Json[] | null>(null);
  const [levels, setLevels] = useState<string[] | null>(null);
  const [forks, setForks] = useState<Json[] | null>(null);
  const run_ = async (work: () => Promise<unknown>) => {
    try {
      await work();
    } catch (error) {
      onError((error as Error).message);
    }
  };
  const toggle = () => {
    const next = !open;
    setOpen(next);
    if (next && allows(run, "set_model")) void run_(async () => setModels((await call("/api/models")).models ?? []));
    if (next && allows(run, "set_thinking_level")) {
      void run_(async () => setLevels((await call("/api/thinking-levels")).levels ?? []));
    }
  };
  const any = ["set_model", "set_thinking_level", "compact", "set_session_name", "new_session", "fork"].some((command) => allows(run, command));
  if (!any) return null;
  return (
    <div className="session-tools">
      <button type="button" className="icon-button" onClick={toggle} title={t("session.menu")} aria-expanded={open}>
        <Icon name="settings" />
      </button>
      {open ? (
        <div className="menu" role="menu">
          {allows(run, "set_model") ? (
            <label>{t("session.model")}
              <select value={state?.model ? `${state.model.provider}/${state.model.id}` : ""} onChange={(event) => {
                const [provider, ...rest] = event.target.value.split("/");
                void run_(() => call("/api/model", { provider, modelId: rest.join("/") }));
              }}>
                {(models ?? (state?.model ? [state.model] : [])).map((item: Json) => (
                  <option key={`${item.provider}/${item.id}`} value={`${item.provider}/${item.id}`}>{item.provider}/{item.id}</option>
                ))}
              </select>
            </label>
          ) : null}
          {allows(run, "set_thinking_level") ? (
            <label>{t("session.reasoning")}
              <select value={state?.thinkingLevel ?? ""} onChange={(event) => void run_(() => call("/api/thinking", { level: event.target.value }))}>
                {(levels ?? [state?.thinkingLevel].filter(Boolean)).map((level: string) => <option key={level} value={level}>{level}</option>)}
              </select>
            </label>
          ) : null}
          {allows(run, "compact") ? (
            <button type="button" className="ghost" disabled={state?.isStreaming || state?.isCompacting}
              onClick={() => void run_(() => call("/api/compact", {}))}>{t("session.compact")}</button>
          ) : null}
          {allows(run, "set_session_name") ? (
            <button type="button" className="ghost" onClick={() => {
              const name = window.prompt(t("session.renamePrompt"), state?.sessionName ?? "");
              if (name && name.trim()) void run_(() => call("/api/name", { name: name.trim() }));
            }}>{t("session.rename")}</button>
          ) : null}
          {allows(run, "new_session") ? (
            <button type="button" className="ghost" disabled={state?.isStreaming}
              onClick={() => window.confirm(t("session.newConfirm"))
                && void run_(() => call("/api/new-session", {}))}>{t("session.new")}</button>
          ) : null}
          {allows(run, "fork") ? (
            <button type="button" className="ghost" disabled={state?.isStreaming}
              onClick={() => void run_(async () => setForks((await call("/api/fork-messages")).messages ?? []))}>
              {t("session.fork")}</button>
          ) : null}
          {forks ? (
            <div className="fork-list">
              {forks.length === 0 ? <span className="muted">{t("session.noFork")}</span> : null}
              {forks.map((item: Json) => (
                <button type="button" key={item.entryId} className="ghost" onClick={() => {
                  setForks(null);
                  void run_(() => call("/api/fork", { entryId: item.entryId }));
                }}>{String(item.text ?? "").slice(0, 80)}</button>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
