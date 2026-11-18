// Free play's speed at the foot of the camera panel (lib/perf.ts): one line always (tok/s, the time to
// the first token, the loop between robot actions against the target), and, opened, a drawer over the
// cameras (it never moves them or the conversation) with the model calls, each loop's breakdown as a
// stacked bar (the latest and the median of the last ones, on one scale with the target marked) and
// what the loop would need to meet the target. The target is the operator's, remembered in this browser.

import { useMemo, useState } from "react";

import { duration, t, useLang } from "../lib/i18n.ts";
import { adviceLines, ms, rememberTarget, SEGMENTS, segmentName, storedTarget, summarize, tokensPerSecond, WINDOW,
  type Loop, type PerfLog, type SegmentKind, type TimedAnswer } from "../lib/perf.ts";

type Segments = Record<SegmentKind, number | null>;

/** One stacked bar: each known segment's width on the shared scale, the target as a mark. */
function LoopBar({ label, segments, total, scaleMs, targetMs, testId }: {
  label: string;
  segments: Segments;
  total: string;
  scaleMs: number;
  targetMs: number;
  testId: string;
}) {
  const parts = SEGMENTS.flatMap((kind) => (segments[kind] === null || segments[kind]! <= 0 ? [] : [[kind, segments[kind]!] as const]));
  return (
    <div className="perf-bar-row" data-testid={testId}>
      <div className="perf-bar-label"><span>{label}</span><span className="perf-bar-total">{total}</span></div>
      <div className="perf-bar" role="img" aria-label={`${label}: ${parts.map(([kind, value]) => `${segmentName(kind)} ${ms(value)}`).join(", ")}`}>
        {parts.map(([kind, value]) => (
          <span key={kind} className={`perf-seg seg-${kind}`} style={{ width: `${(value / scaleMs) * 100}%` }}
            title={`${segmentName(kind)} · ${ms(value)}`} data-segment={kind} />
        ))}
        <i className="perf-target-mark" style={{ left: `${Math.min(100, (targetMs / scaleMs) * 100)}%` }}
          title={t("perf.loop.targetMark", { target: duration(targetMs / 1000) })} />
      </div>
    </div>
  );
}

function approvalNote(loop: Loop | null, answers: TimedAnswer[] | null): string | null {
  if (answers === null) return t("perf.approval.none");
  if (loop === null) return null;
  if (!loop.approvalKnown) return loop.finishedMs === null ? null : t("perf.approval.unknown");
  if (loop.approvalBy === null) return null;
  return t(loop.approved === false ? "perf.approval.declined" : "perf.approval.by", { by: loop.approvalBy });
}

export function PerfPanel({ perf, answers, replay = false }: {
  perf: PerfLog;
  /** A real cell's answered motion questions (the gateway's), or null: no approval record. */
  answers: TimedAnswer[] | null;
  replay?: boolean;
}) {
  useLang();
  const [open, setOpen] = useState(false);
  const [target, setTarget] = useState(() => storedTarget());
  const [draft, setDraft] = useState(String(target));
  const summary = useMemo(() => summarize(perf, answers, target), [perf, answers, target]);
  const targetText = duration(target);
  const whole = !summary.timedTokens;
  const line = summary.last === null ? t("perf.noCalls") : t("perf.strip", {
    tps: tokensPerSecond(summary.window.tps), ttft: ms(summary.window.ttftMs), loop: ms(summary.medians.totalMs), target: targetText });

  const latest = summary.latest;
  const targetMs = target * 1000;
  const medianSum = SEGMENTS.reduce((sum, kind) => sum + (summary.medians.segments[kind] ?? 0), 0);
  const latestSum = latest === null ? 0 : SEGMENTS.reduce((sum, kind) => sum + (latest.segments[kind] ?? 0), 0);
  const scaleMs = Math.max(targetMs, medianSum, latestSum, latest?.totalMs ?? 0, summary.medians.totalMs ?? 0, 1) * 1.05;
  // A replay says once what its evidence lacks (perf.replayNote).
  const note = replay ? null : approvalNote(latest, answers);

  const commit = (text: string) => {
    const value = Number(text.replace(",", "."));
    if (Number.isFinite(value) && value > 0 && value <= 3600) {
      setTarget(value);
      rememberTarget(value);
    } else setDraft(String(target));
  };

  const rows: Array<[string, string, string]> = [
    [whole ? t("perf.row.tpsWhole") : t("perf.row.tps"), tokensPerSecond(summary.lastStats.tps), tokensPerSecond(summary.window.tps)],
    [t("perf.row.ttft"), ms(summary.lastStats.ttftMs), ms(summary.window.ttftMs)],
    [t("perf.row.duration"), ms(summary.lastStats.durationMs), ms(summary.window.durationMs)],
    [t("perf.row.tokens"), String(summary.lastStats.outputTokens ?? "—"), summary.window.outputTokens === null ? "—"
      : String(Math.round(summary.window.outputTokens))],
    [t("perf.row.callsPerAct"), latest === null ? "—" : String(latest.calls), summary.callsPerAct === null ? "—" : String(summary.callsPerAct)],
  ];

  return (
    <div className={`perf${open ? " open" : ""}`} data-testid="perf">
      {open ? (
        <div className="perf-drawer" role="region" aria-label={t("perf.title")} data-testid="perf-drawer">
          <div className="perf-head">
            <span className="perf-title">{t("perf.title")}</span>
            <label className="perf-target" title={t("perf.targetTitle")}>
              {t("perf.target")}
              <input type="number" min="0.1" step="0.5" value={draft} data-testid="perf-target"
                onChange={(event) => setDraft(event.target.value)} onBlur={(event) => commit(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") commit((event.target as HTMLInputElement).value);
                }} />
              <span className="dim">s</span>
            </label>
          </div>
          <table className="perf-table">
            <thead>
              <tr><th>{t("perf.model")}</th><th>{t("perf.last")}</th><th>{t("perf.window", { n: WINDOW })}</th></tr>
            </thead>
            <tbody>
              {rows.map(([name, last, median]) => <tr key={name}><td>{name}</td><td>{last}</td><td>{median}</td></tr>)}
            </tbody>
          </table>
          <div className="perf-section">{t("perf.loop")}</div>
          {latest === null ? <div className="dim">{t("perf.loop.none")}</div> : (
            <>
              <LoopBar testId="perf-latest" label={t("perf.loop.latest", { step: latest.step })} segments={latest.segments}
                total={latest.totalMs === null ? t("perf.loop.inProgress") : ms(latest.totalMs)} scaleMs={scaleMs} targetMs={targetMs} />
              {summary.medians.count ? (
                <LoopBar testId="perf-median" label={t("perf.loop.median", { n: summary.medians.count })}
                  segments={summary.medians.segments} total={ms(summary.medians.totalMs)} scaleMs={scaleMs} targetMs={targetMs} />
              ) : null}
              <ul className="perf-legend">
                {SEGMENTS.map((kind) => (
                  <li key={kind}>
                    <span className={`perf-swatch seg-${kind}`} aria-hidden="true" />
                    <span>{segmentName(kind)}</span>
                    <span className="perf-legend-value">{ms(latest.segments[kind])}
                      {summary.medians.count ? <span className="dim"> · {ms(summary.medians.segments[kind])}</span> : null}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
          {note ? <div className="perf-note dim">{note}</div> : null}
          <div className="perf-advice" data-testid="perf-advice">
            {adviceLines(summary.advice).map((text, index, all) => (
              <p key={index} className={index === all.length - 1 && all.length > 1 ? "dim" : ""}>{text}</p>
            ))}
          </div>
          {replay ? <div className="perf-note dim">{t("perf.replayNote")}</div> : null}
        </div>
      ) : null}
      <button type="button" className="perf-strip" data-testid="perf-strip" aria-expanded={open}
        title={open ? t("perf.hide") : t("perf.show")} onClick={() => setOpen((value) => !value)}>
        <span className="perf-strip-mark" aria-hidden="true">{open ? "▾" : "▸"}</span>
        <span className="perf-strip-text">{line}</span>
        {summary.advice?.kind === "over" ? <span className="perf-strip-over">+{ms(summary.advice.overMs)}</span> : null}
      </button>
    </div>
  );
}
