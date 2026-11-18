// A real cell's setup before its worker starts (`ora play --setup`, ora/orchestrator/episode/cell_setup.py),
// drawn where the conversation will be: the checklist with each step's state, the latest progress lines, and
// how it ended. The Supervisor's announcement carries the whole view, so the latest one draws it all. Its
// questions are the cell's question cards (CellQuestions), answered as a worker's are.

import { t, useLang } from "../lib/i18n.ts";

type Json = any;

const MARKS: Record<string, string> = { done: "✓", running: "✻", failed: "✗", stale: "!", missing: "○" };

/** Which title a setup view takes: where it is now (the terminal's setupTitle). */
export function setupTitle(setup: Json): "status" | "running" | "done" | "skipped" | "failed" {
  if (setup?.phase === "done") return setup.skipped ? "skipped" : "done";
  return setup?.phase === "running" || setup?.phase === "failed" ? setup.phase : "status";
}

export function SetupView({ setup }: { setup: Json }) {
  useLang();
  const steps: Json[] = Array.isArray(setup?.steps) ? setup.steps : [];
  const lines: string[] = Array.isArray(setup?.lines) ? setup.lines.map(String) : [];
  const title = setupTitle(setup);
  const end = title === "failed" ? setup?.error ?? setup?.result?.why : null;
  return (
    <section className={`setup ${title}`} data-testid="cell-setup">
      <div className="setup-head">{t(`setup.title.${title}`)}</div>
      <ul className="setup-steps">
        {steps.map((step) => (
          <li key={String(step?.id)} className={`setup-step ${String(step?.state ?? "")}`}>
            <span className="mark">{MARKS[step?.state] ?? "·"}</span>
            <span className="name">{String(step?.title ?? step?.id)}</span>
            {step?.why || step?.detail ? <span className="note">{String(step.why || step.detail)}</span> : null}
          </li>
        ))}
      </ul>
      {lines.length ? <pre className="setup-lines">{lines.join("\n")}</pre> : null}
      {end ? <div className="setup-end">{String(end)}</div> : null}
      {title === "running" || (title === "status" && setup?.next) ? <div className="setup-hint">{t("setup.answer")}</div> : null}
    </section>
  );
}
