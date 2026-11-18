// A replay's own bar, where the live page has its composer: what the agent does at the
// replay's clock (with its seconds, counted on the recorded time), play and pause, the
// position over the run's recorded time with a mark where each instruction began, the speed
// of the waiting and thinking, and the speed of the robot's motion against its simulated time
// (real time by default, or the same speed as the rest). Nothing here reaches the run: a recorded
// run takes no messages.
// Drawn as the terminal's prompt: the status line, the controls between two rules, the footer.
// While a video is made the controls give way to an empty prompt.

import { clockText, MOTION_SPEEDS, SPEEDS, type MotionSpeed } from "../lib/replay.ts";
import { t, useLang } from "../lib/i18n.ts";
import { Icon } from "./Icons.tsx";
import { TerminalFooter, type FooterInfo } from "./TerminalFooter.tsx";

export function ReplayBar({ start, end, clock, playing, speed, motion, instructions, doing, activity, exporting, onPlay, onSeek,
  onSpeed, onMotion, info }: {
  start: number;
  end: number;
  clock: number;
  playing: boolean;
  speed: number;
  motion: MotionSpeed;
  /** Where each instruction began (ms, the run's clock). */
  instructions: Array<{ index: number; start: number }>;
  doing: string | null;
  activity: string | null;
  exporting: boolean;
  onPlay: (playing: boolean) => void;
  onSeek: (time: number) => void;
  onSpeed: (speed: number) => void;
  onMotion: (motion: MotionSpeed) => void;
  /** What the terminal's footer says at the replay's clock. */
  info: FooterInfo;
}) {
  useLang();
  const span = Math.max(1, end - start);
  const ended = clock >= end;
  const status = (
    <div className="status-line">
      <div className={`replay-activity activity-${activity ?? "idle"}${doing ? "" : " idle"}`} data-testid="activity" role="status">
        {doing ? <span className="doing" data-testid="doing">{doing}</span> : <span className="muted">{t("replay.idle")}</span>}
      </div>
    </div>
  );
  // A video: the terminal as it looked, an empty prompt with its cursor.
  if (exporting) {
    return (
      <div className="replay-bar exporting">
        {status}
        <div className="prompt"><span className="prompt-mark" aria-hidden="true">❯</span><span className="cursor" aria-hidden="true" /></div>
        <TerminalFooter info={info} />
      </div>
    );
  }
  const speeds = SPEEDS.includes(speed) ? SPEEDS : [...SPEEDS, speed].sort((a, b) => a - b);
  const motions = MOTION_SPEEDS.includes(motion) ? MOTION_SPEEDS : [motion, ...MOTION_SPEEDS];
  return (
    <div className="replay-bar" data-testid="replay-bar">
      {status}
      <div className="prompt replay-controls">
        <button type="button" className="icon-button" aria-pressed={playing} data-testid="replay-toggle"
          title={ended ? t("replay.restart") : playing ? t("replay.pause") : t("replay.play")}
          onClick={() => {
            if (ended) onSeek(start);
            onPlay(ended ? true : !playing);
          }}>
          <Icon name={playing ? "pause" : ended ? "replay" : "play"} />
        </button>
        <div className="replay-track">
          <input type="range" className="replay-scrub" aria-label={t("replay.position")} min={0} max={span} step={250}
            value={Math.min(span, Math.max(0, clock - start))} onChange={(event) => onSeek(start + Number(event.target.value))} />
          <div className="replay-marks" aria-hidden="true">
            {instructions.map((item) => (
              <span key={item.index} className="replay-mark" title={t("replay.instruction", { n: item.index + 1 })}
                style={{ left: `${(100 * Math.min(span, Math.max(0, item.start - start))) / span}%` }} />
            ))}
          </div>
        </div>
        <span className="replay-time" data-testid="replay-time">{clockText(clock - start)} / {clockText(span)}</span>
        <select className="replay-speed" aria-label={t("replay.speed")} value={speed}
          onChange={(event) => onSpeed(Number(event.target.value))}>
          {speeds.map((value) => <option key={value} value={value}>{value}×</option>)}
        </select>
        <select className="replay-speed" aria-label={t("replay.motionSpeed")} value={String(motion)}
          title={t("replay.motionSpeedTitle")}
          onChange={(event) => onMotion(event.target.value === "same" ? "same" : Number(event.target.value))}>
          {motions.map((value) => (
            <option key={String(value)} value={String(value)}>
              {value === "same" ? t("replay.motionSame") : value === 1 ? t("replay.motionReal") : t("replay.motionAt", { speed: value })}
            </option>
          ))}
        </select>
      </div>
      <TerminalFooter info={info} />
    </div>
  );
}
