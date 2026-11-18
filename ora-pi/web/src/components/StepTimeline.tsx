// The run's observation steps as a strip: one tick per step (marked when its tool calls
// drew perception or failed, and when its action's motion was recorded), the selected one
// highlighted; previous, next and a replay that walks the steps with the conversation.
// Each step says how long it took: the agent's thinking (from its observation to its last
// tool result) and the robot's move (from there to the next observation, or the
// Supervisor's measured execution time).

import { duration, t, useLang } from "../lib/i18n.ts";
import type { Conversation, StepView } from "../lib/messages.ts";
import { Icon } from "./Icons.tsx";

function latency(view: StepView | undefined): string | null {
  if (!view || (view.think === null && view.move === null)) return null;
  return t("timeline.latency", { think: duration(view.think), move: duration(view.move) });
}

export function StepTimeline({ conversation, current, replaying, onReplay, onStep, motion = {} }: {
  conversation: Conversation;
  current: number | null;
  replaying: boolean;
  /** The step walk; null leaves its button out (a replay's own clock moves the page). */
  onReplay: ((on: boolean) => void) | null;
  onStep: (step: number) => void;
  /** Recorded motion frames by step (ora_get_frames). */
  motion?: Record<string, number>;
}) {
  useLang();
  const order = conversation.order;
  if (!order.length) return <div className="timeline empty-timeline">{t("timeline.empty")}</div>;
  const index = current === null ? -1 : order.indexOf(current);
  const go = (offset: number) => {
    const next = order[Math.min(order.length - 1, Math.max(0, (index < 0 ? order.length : index) + offset))];
    if (next !== undefined) onStep(next);
  };
  const shown = conversation.steps.get(current ?? order[order.length - 1]);
  const timing = latency(shown);
  return (
    <div className="timeline">
      <button type="button" className="icon-button" onClick={() => go(-1)} disabled={index === 0} title={t("timeline.previous")}>
        <Icon name="prev" />
      </button>
      {onReplay ? (
        <button type="button" className="icon-button" onClick={() => onReplay(!replaying)}
          title={replaying ? t("timeline.pause") : t("timeline.replay")} aria-pressed={replaying}>
          <Icon name={replaying ? "pause" : "play"} />
        </button>
      ) : null}
      <button type="button" className="icon-button" onClick={() => go(1)} disabled={index === order.length - 1} title={t("timeline.next")}>
        <Icon name="next" />
      </button>
      <div className="ticks" role="listbox" aria-label={t("timeline.label")}>
        {order.map((step) => {
          const view = conversation.steps.get(step)!;
          const failed = view.tools.some((tool) => tool.status === "error");
          const drawn = view.tools.some((tool) => tool.overlay !== null);
          const moved = Boolean(motion[String(step)]);
          const tools = view.tools.map((tool) => tool.name).join(", ") || t("timeline.observation");
          const time = latency(view);
          return (
            <button type="button" key={step} role="option" aria-selected={step === current}
              className={`tick${step === current ? " on" : ""}${failed ? " failed" : ""}${drawn ? " drawn" : ""}${moved ? " moved" : ""}`}
              title={[t("timeline.tick", { step, tools }), time].filter(Boolean).join(" · ")}
              onClick={() => onStep(step)}>
              <span />
            </button>
          );
        })}
      </div>
      <span className="timeline-label" data-testid="timeline-label">
        {index < 0 ? (order.length === 1 ? t("timeline.oneStep") : t("timeline.steps", { n: order.length }))
          : t("timeline.at", { step: current, index: index + 1, count: order.length })}
        {timing ? <span className="latency"> · {timing}</span> : null}
      </span>
    </div>
  );
}
